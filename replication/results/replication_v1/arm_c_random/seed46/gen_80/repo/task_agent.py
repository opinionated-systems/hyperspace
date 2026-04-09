"""
Task agent: solves a given task with a single LLM call.

Reimplemented from facebookresearch/HyperAgents task_agent.py.
Same interface, same JSON output format, same extraction logic.

This is the INITIAL task agent. The meta agent modifies this file
during self-improvement. The evaluation harness loads whatever
task_agent.py exists at the agent's repo path.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from agent.llm_client import get_response_from_llm, EVAL_MODEL

logger = logging.getLogger(__name__)

# Cache configuration
_CACHE_DIR = os.environ.get("TASK_AGENT_CACHE_DIR", "")
_cache_hits = 0
_cache_misses = 0

# Maximum cache entries to prevent unbounded growth
_MAX_CACHE_ENTRIES = 10000


def _get_cache_path(inputs: dict, model: str) -> Path | None:
    """Generate cache file path for given inputs."""
    if not _CACHE_DIR:
        return None
    cache_key = hashlib.sha256(
        json.dumps(inputs, sort_keys=True, default=str).encode()
    ).hexdigest()
    cache_dir = Path(_CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{model.replace('/', '_')}_{cache_key}.json"


def _load_from_cache(cache_path: Path) -> tuple[str, list[dict]] | None:
    """Load cached response if available."""
    global _cache_hits
    if cache_path.exists():
        try:
            with open(cache_path) as f:
                data = json.load(f)
            _cache_hits += 1
            logger.info(f"Cache hit ({_cache_hits} total hits, {_cache_misses} misses)")
            return data["prediction"], data["msg_history"]
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
    return None


def _save_to_cache(cache_path: Path, prediction: str, msg_history: list[dict]) -> None:
    """Save response to cache with size management."""
    global _cache_misses
    _cache_misses += 1
    try:
        cache_dir = cache_path.parent
        # Check cache size and clean if needed
        if cache_dir.exists():
            cache_files = list(cache_dir.glob("*.json"))
            if len(cache_files) >= _MAX_CACHE_ENTRIES:
                # Remove oldest files (by modification time)
                cache_files.sort(key=lambda p: p.stat().st_mtime)
                for old_file in cache_files[:100]:  # Remove 100 oldest
                    old_file.unlink(missing_ok=True)
                logger.info("Cache cleanup: removed 100 oldest entries")
        
        with open(cache_path, "w") as f:
            json.dump({
                "prediction": prediction,
                "msg_history": msg_history,
            }, f, default=str)
        logger.info(f"Cache miss - saved to cache ({_cache_hits} hits, {_cache_misses} misses)")
    except Exception as e:
        logger.warning(f"Failed to save cache: {e}")


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks and markdown code blocks.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    Also handles markdown code blocks with json tag and plain JSON objects.
    """
    results = []
    
    # First, try to find <json> blocks
    search_from = 0
    while True:
        start = text.find("<json>", search_from)
        if start == -1:
            break
        content_start = start + 6
        end = text.find("</json>", content_start)
        if end == -1:
            break
        inner = text[content_start:end].strip()
        search_from = end + 7
        try:
            parsed = json.loads(inner)
            if isinstance(parsed, dict):
                results.append(parsed)
        except json.JSONDecodeError:
            # Try to find JSON object within the content
            try:
                obj_start = inner.find('{')
                obj_end = inner.rfind('}')
                if obj_start != -1 and obj_end != -1 and obj_end > obj_start:
                    parsed = json.loads(inner[obj_start:obj_end+1])
                    if isinstance(parsed, dict):
                        results.append(parsed)
            except json.JSONDecodeError:
                continue
    
    # Then, try markdown code blocks with json
    search_from = 0
    while True:
        start = text.find("```json", search_from)
        if start == -1:
            break
        content_start = start + 7
        end = text.find("```", content_start)
        if end == -1:
            break
        inner = text[content_start:end].strip()
        search_from = end + 3
        try:
            parsed = json.loads(inner)
            if isinstance(parsed, dict):
                results.append(parsed)
        except json.JSONDecodeError:
            continue
    
    # Finally, try to find any JSON objects with "response" key directly in text
    # This handles cases where the model outputs raw JSON without tags
    if not results:
        try:
            # Find all potential JSON objects by tracking brace depth
            i = 0
            while i < len(text):
                if text[i] == '{':
                    start = i
                    depth = 1
                    i += 1
                    while i < len(text) and depth > 0:
                        if text[i] == '{':
                            depth += 1
                        elif text[i] == '}':
                            depth -= 1
                        i += 1
                    if depth == 0:
                        try:
                            obj = json.loads(text[start:i])
                            if isinstance(obj, dict) and "response" in obj:
                                results.append(obj)
                        except json.JSONDecodeError:
                            pass
                else:
                    i += 1
        except Exception:
            pass
    
    return results or None


def _extract_json_fallback(text: str) -> list[dict] | None:
    """Fallback JSON extraction using regex for unwrapped JSON objects.

    This handles cases where the model outputs valid JSON without <json> tags.
    Searches for JSON objects with a "response" key.
    Improved to handle nested braces more robustly.
    """
    results = []
    
    # First, try to find JSON objects by parsing brace depth
    # This handles nested structures better than regex
    def find_json_objects(s: str) -> list[str]:
        """Find potential JSON objects by tracking brace depth."""
        objects = []
        i = 0
        while i < len(s):
            if s[i] == '{':
                start = i
                depth = 1
                i += 1
                while i < len(s) and depth > 0:
                    if s[i] == '{':
                        depth += 1
                    elif s[i] == '}':
                        depth -= 1
                    i += 1
                if depth == 0:
                    objects.append(s[start:i])
            else:
                i += 1
        return objects
    
    # Try to parse each potential JSON object
    for obj_str in find_json_objects(text):
        try:
            obj = json.loads(obj_str)
            if isinstance(obj, dict) and "response" in obj:
                results.append(obj)
        except json.JSONDecodeError:
            continue
    
    # If no results, try regex pattern for simpler cases
    if not results:
        # Pattern to match JSON objects with response key (simpler cases)
        pattern = r'\{[^{}]*"response"[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.finditer(pattern, text, re.DOTALL)
        for match in matches:
            try:
                obj = json.loads(match.group())
                if "response" in obj:
                    results.append(obj)
            except json.JSONDecodeError:
                continue

    # If still no results, try to parse the entire text as JSON
    if not results:
        try:
            obj = json.loads(text.strip())
            if isinstance(obj, dict) and "response" in obj:
                results.append(obj)
        except json.JSONDecodeError:
            pass
    
    # Final fallback: try to extract any dict-like structure with response key
    if not results:
        # Look for patterns like "response": "..." or 'response': '...'
        response_pattern = r'["\']response["\']\s*:\s*["\']([^"\']+)["\']'
        match = re.search(response_pattern, text)
        if match:
            results.append({"response": match.group(1)})

    return results or None


def _format_imo_prompt(inputs: dict) -> str:
    """Format a structured prompt for IMO grading problems.
    
    This improves performance by providing clear structure and context.
    """
    domain = inputs.get("domain", "")
    problem = inputs.get("problem", "")
    solution = inputs.get("solution", "")
    grading_guidelines = inputs.get("grading_guidelines", "")
    student_answer = inputs.get("student_answer", "")
    
    prompt = f"""You are an expert mathematics grader for the International Mathematical Olympiad (IMO).

Your task is to evaluate a student's answer to a mathematical problem and provide a grade.

## Problem Domain
{domain}

## Problem Statement
{problem}

## Official Solution
{solution}

## Grading Guidelines
{grading_guidelines}

## Student's Answer
{student_answer}

## Instructions
Carefully analyze the student's answer against the official solution and grading guidelines.
Provide your evaluation in the following JSON format:

<json>
{{
    "response": "Your grade/evaluation here. Be specific about what the student got right and wrong."
}}
</json>

Important: 
- The response field should contain your complete evaluation/grade
- Be thorough and justify your assessment
- If the problem asks for a specific format (e.g., a number, a proof), ensure your evaluation addresses whether the student met those requirements"""
    
    return prompt


class TaskAgent:
    """Task agent that solves IMO grading problems with robust JSON extraction and caching."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Check cache first
        cache_path = _get_cache_path(inputs, self.model)
        if cache_path:
            cached = _load_from_cache(cache_path)
            if cached:
                return cached

        # Use structured prompt for IMO grading
        instruction = _format_imo_prompt(inputs)

        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
        except Exception as e:
            self.log_fn(f"Error calling LLM: {e}")
            return "Error: LLM call failed", [{"role": "system", "text": f"Error: {e}"}]

        # Check if we got a valid response
        if not msg_history or len(msg_history) < 2:
            self.log_fn("Warning: Empty or incomplete message history from LLM")
            return "Error: No response from LLM", msg_history if msg_history else [{"role": "system", "text": "No response"}]

        # Extract prediction from JSON using primary method
        prediction = "None"
        extraction_method = "primary"
        raw_response = msg_history[-1].get("text", "")
        
        if not raw_response or not raw_response.strip():
            self.log_fn("Warning: Empty response text from LLM")
            return "Error: Empty response from LLM", msg_history
        
        try:
            extracted = _extract_jsons(raw_response)
            if extracted and len(extracted) > 0 and "response" in extracted[-1]:
                prediction = extracted[-1]["response"]
                self.log_fn(f"Primary extraction succeeded, response type: {type(prediction).__name__}")
            else:
                # Try fallback extraction
                self.log_fn("Primary extraction failed, trying fallback...")
                extracted = _extract_json_fallback(raw_response)
                if extracted and len(extracted) > 0 and "response" in extracted[-1]:
                    prediction = extracted[-1]["response"]
                    extraction_method = "fallback"
                    self.log_fn(f"Fallback extraction succeeded, response type: {type(prediction).__name__}")
                else:
                    self.log_fn("Warning: No valid JSON with 'response' key found in LLM output")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            # Try fallback on exception
            try:
                extracted = _extract_json_fallback(raw_response)
                if extracted and len(extracted) > 0 and "response" in extracted[-1]:
                    prediction = extracted[-1]["response"]
                    extraction_method = "fallback"
                    self.log_fn(f"Fallback extraction succeeded after exception, response type: {type(prediction).__name__}")
            except Exception as fallback_e:
                self.log_fn(f"Fallback extraction also failed: {fallback_e}")

        self.log_fn(f"Extraction method used: {extraction_method}")
        
        # Convert prediction to string, handling various types
        if prediction is None:
            prediction = "None"
        elif isinstance(prediction, (list, dict)):
            prediction = json.dumps(prediction)
        else:
            prediction = str(prediction)
        
        # Save to cache if enabled
        if cache_path:
            _save_to_cache(cache_path, prediction, msg_history)
            
        return prediction, msg_history
