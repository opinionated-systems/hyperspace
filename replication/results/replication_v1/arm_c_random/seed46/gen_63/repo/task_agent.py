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
import re
from functools import lru_cache

from agent.llm_client import get_response_from_llm, EVAL_MODEL

logger = logging.getLogger(__name__)

# Simple in-memory cache for LLM responses to avoid redundant calls
_response_cache: dict[str, tuple[str, list[dict]]] = {}


def _make_cache_key(inputs: dict, model: str) -> str:
    """Create a deterministic cache key from inputs."""
    key_data = json.dumps(inputs, sort_keys=True) + model
    return hashlib.sha256(key_data.encode()).hexdigest()[:32]


def clear_response_cache() -> int:
    """Clear the response cache. Returns number of entries cleared."""
    global _response_cache
    count = len(_response_cache)
    _response_cache.clear()
    return count


def get_cache_stats() -> dict:
    """Get cache statistics."""
    return {
        "entries": len(_response_cache),
        "keys": [k[:8] + "..." for k in list(_response_cache.keys())[:5]],
    }


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    """
    results = []
    search_from = 0
    while True:
        start = text.find("<json>", search_from)
        if start == -1:
            break
        end = text.find("</json>", start)
        if end == -1:
            break
        inner = text[start + 6:end].strip()
        search_from = end + 7
        try:
            results.append(json.loads(inner))
        except json.JSONDecodeError:
            continue
    return results or None


def _extract_json_fallback(text: str) -> list[dict] | None:
    """Fallback JSON extraction using regex for unwrapped JSON objects.

    This handles cases where the model outputs valid JSON without <json> tags.
    Searches for JSON objects with a "response" key.
    Uses a robust brace-matching approach to handle nested structures.
    """
    results = []

    # Find all potential JSON object starting positions
    i = 0
    while i < len(text):
        # Look for opening brace followed by "response"
        idx = text.find('"response"', i)
        if idx == -1:
            break

        # Find the opening brace before "response"
        brace_start = text.rfind('{', 0, idx)
        if brace_start == -1:
            i = idx + 1
            continue

        # Find the matching closing brace using brace counting
        brace_count = 0
        brace_end = -1
        in_string = False
        escape_next = False

        for j in range(brace_start, len(text)):
            char = text[j]

            if escape_next:
                escape_next = False
                continue

            if char == '\\' and in_string:
                escape_next = True
                continue

            if char == '"' and not in_string:
                in_string = True
            elif char == '"' and in_string:
                in_string = False

            if not in_string:
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        brace_end = j + 1
                        break

        if brace_end > brace_start:
            json_str = text[brace_start:brace_end]
            try:
                obj = json.loads(json_str)
                if isinstance(obj, dict) and "response" in obj:
                    results.append(obj)
            except json.JSONDecodeError:
                pass
            i = brace_end
        else:
            i = idx + 1

    # If brace matching fails, try to parse the entire text as JSON
    if not results:
        try:
            obj = json.loads(text.strip())
            if isinstance(obj, dict) and "response" in obj:
                results.append(obj)
        except json.JSONDecodeError:
            pass

    return results or None


def _extract_json_heuristic(text: str) -> list[dict] | None:
    """Third-level extraction using heuristics for malformed JSON.
    
    Handles common LLM output issues like:
    - Truncated JSON (missing closing braces)
    - Extra text before/after JSON
    - Single quotes instead of double quotes
    - Missing quotes around keys
    """
    results = []
    
    # Try to find anything that looks like a response value
    # Pattern: look for "response" followed by some value
    response_patterns = [
        r'"response"\s*:\s*"([^"]*)"',  # Standard double-quoted string
        r'"response"\s*:\s*\{([^}]*)\}',  # Nested object (simplified)
        r'"response"\s*:\s*(\d+(?:\.\d+)?)',  # Number
        r'"response"\s*:\s*(true|false|null)',  # Boolean/null
    ]
    
    for pattern in response_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            if isinstance(match, tuple):
                match = match[0]
            # Try to reconstruct a valid JSON object
            try:
                # If it's a simple string value
                if pattern.endswith('"([^"]*)"'):
                    results.append({"response": match})
                # If it's a number
                elif pattern.endswith('(\d+(?:\.\d+)?)'):
                    # Try int first, then float
                    try:
                        val = int(match)
                    except ValueError:
                        val = float(match)
                    results.append({"response": val})
                # If it's boolean/null
                elif pattern.endswith('(true|false|null)'):
                    val = {"true": True, "false": False, "null": None}.get(match.lower())
                    results.append({"response": val})
            except Exception:
                pass
    
    # Last resort: try to extract any quoted string after "response"
    if not results:
        loose_match = re.search(r'["\']?response["\']?\s*[:=]\s*["\']?([^"\'\n,}]+)', text, re.IGNORECASE)
        if loose_match:
            results.append({"response": loose_match.group(1).strip()})
    
    return results or None


class TaskAgent:
    """Task agent that solves IMO grading problems with robust JSON extraction and caching."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "", use_cache: bool = True) -> None:
        self.model = model
        self.log_fn = logger.info
        self.use_cache = use_cache

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Check cache first if enabled
        if self.use_cache:
            cache_key = _make_cache_key(inputs, self.model)
            if cache_key in _response_cache:
                self.log_fn(f"Cache hit for key {cache_key[:8]}...")
                cached_prediction, cached_history = _response_cache[cache_key]
                return cached_prediction, list(cached_history)  # Return copy

        # Build a more structured prompt for better LLM understanding
        domain = inputs.get("domain", "Unknown")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = f"""You are an expert grader for {domain} problems. Your task is to evaluate a student's answer against the correct solution and grading guidelines.

## Problem
{problem}

## Correct Solution
{solution}

## Grading Guidelines
{grading_guidelines}

## Student's Answer
{student_answer}

## Your Task
Evaluate the student's answer and provide your assessment. Your response should be a single value (e.g., a score, grade, or evaluation) wrapped in the JSON format below.

Respond ONLY in the following JSON format:
<json>
{{
    "response": "your_evaluation_here"
}}
</json>

Important:
- The "response" field should contain your final evaluation (score, grade, or assessment)
- Do not include any explanation outside the JSON block
- Ensure the JSON is valid and properly formatted"""

        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
        except Exception as e:
            import traceback
            error_details = f"Error calling LLM: {e}\n{traceback.format_exc()}"
            self.log_fn(error_details)
            return "Error: LLM call failed", [{"role": "system", "text": error_details}]

        # Extract prediction from JSON using primary method
        prediction = "None"
        extraction_method = "primary"
        raw_response = msg_history[-1]["text"]
        
        # Log the raw response for debugging (truncated if very long)
        log_response = raw_response[:500] + "..." if len(raw_response) > 500 else raw_response
        self.log_fn(f"Raw LLM response: {log_response}")
        
        try:
            extracted = _extract_jsons(raw_response)
            if extracted and len(extracted) > 0:
                last_json = extracted[-1]
                if isinstance(last_json, dict) and "response" in last_json:
                    prediction = last_json["response"]
                    self.log_fn(f"Primary extraction successful. Found {len(extracted)} JSON block(s)")
                else:
                    self.log_fn(f"Primary extraction: last JSON missing 'response' key. Keys: {list(last_json.keys()) if isinstance(last_json, dict) else 'N/A'}")
                    # Try fallback extraction
                    extracted = _extract_json_fallback(raw_response)
                    if extracted and len(extracted) > 0 and "response" in extracted[-1]:
                        prediction = extracted[-1]["response"]
                        extraction_method = "fallback"
            else:
                self.log_fn("Primary extraction: no JSON blocks found with <json> tags")
                # Try fallback extraction
                extracted = _extract_json_fallback(raw_response)
                if extracted and len(extracted) > 0 and "response" in extracted[-1]:
                    prediction = extracted[-1]["response"]
                    extraction_method = "fallback"
                    self.log_fn("Fallback extraction successful")
                else:
                    self.log_fn("Fallback extraction: no valid JSON with 'response' key found")
                    # Try heuristic extraction as last resort
                    extracted = _extract_json_heuristic(raw_response)
                    if extracted and len(extracted) > 0 and "response" in extracted[-1]:
                        prediction = extracted[-1]["response"]
                        extraction_method = "heuristic"
                        self.log_fn("Heuristic extraction successful")
        except Exception as e:
            self.log_fn(f"Error in primary extraction: {type(e).__name__}: {e}")
            # Try fallback on exception
            try:
                extracted = _extract_json_fallback(raw_response)
                if extracted and len(extracted) > 0 and "response" in extracted[-1]:
                    prediction = extracted[-1]["response"]
                    extraction_method = "fallback"
                    self.log_fn("Fallback extraction successful after primary failure")
                else:
                    # Try heuristic extraction
                    extracted = _extract_json_heuristic(raw_response)
                    if extracted and len(extracted) > 0 and "response" in extracted[-1]:
                        prediction = extracted[-1]["response"]
                        extraction_method = "heuristic"
                        self.log_fn("Heuristic extraction successful after primary failure")
                    else:
                        self.log_fn("Fallback extraction: no valid JSON with 'response' key found")
            except Exception as fallback_e:
                self.log_fn(f"Fallback extraction also failed: {type(fallback_e).__name__}: {fallback_e}")

        self.log_fn(f"Extraction method used: {extraction_method}, prediction type: {type(prediction).__name__}")
        
        # Store in cache if enabled
        if self.use_cache:
            cache_key = _make_cache_key(inputs, self.model)
            _response_cache[cache_key] = (str(prediction), list(msg_history))
            self.log_fn(f"Cached result with key {cache_key[:8]}...")
        
        return str(prediction), msg_history
