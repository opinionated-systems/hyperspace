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
import time
from functools import lru_cache

from agent.llm_client import get_response_from_llm, EVAL_MODEL

logger = logging.getLogger(__name__)

# Simple in-memory cache for LLM responses to avoid redundant calls
_response_cache: dict[str, tuple[str, list[dict]]] = {}
_cache_access_times: dict[str, float] = {}  # Track last access for LRU eviction
_cache_hits: int = 0
_cache_misses: int = 0
_MAX_CACHE_SIZE: int = 500  # Reduced from 1000 for better memory management


def _get_cache_key(inputs: dict, model: str) -> str:
    """Generate a cache key from inputs and model."""
    key_data = json.dumps(inputs, sort_keys=True) + model
    return hashlib.sha256(key_data.encode()).hexdigest()[:32]


def clear_response_cache() -> None:
    """Clear the response cache. Useful for testing or memory management."""
    global _response_cache, _cache_access_times, _cache_hits, _cache_misses
    _response_cache.clear()
    _cache_access_times.clear()
    _cache_hits = 0
    _cache_misses = 0
    logger.info("Response cache cleared")


def get_cache_stats() -> dict:
    """Get cache statistics for monitoring and debugging."""
    return _get_cache_stats()


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    Also handles markdown code blocks and raw JSON objects.
    Includes enhanced error recovery for common JSON formatting issues.
    """
    results = []
    search_from = 0
    
    # First, try to find <json>...</json> blocks
    while True:
        start = text.find("<json>", search_from)
        if start == -1:
            break
        end = text.find("</json>", start)
        if end == -1:
            break
        inner = text[start + 6:end].strip()
        search_from = end + 7
        
        # Try multiple parsing strategies
        parsed = _try_parse_json(inner)
        if parsed:
            results.append(parsed)
    
    # If no <json> blocks found, try markdown code blocks
    if not results:
        md_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
        for match in re.finditer(md_pattern, text, re.DOTALL):
            parsed = _try_parse_json(match.group(1).strip())
            if parsed:
                results.append(parsed)
    
    # Last resort: try to find any JSON-like structure with "response" field
    if not results:
        # Look for objects with "response" key - more permissive pattern
        json_pattern = r'\{[^{}]*"response"[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        for match in re.finditer(json_pattern, text):
            parsed = _try_parse_json(match.group())
            if parsed:
                results.append(parsed)
    
    return results or None


def _try_parse_json(text: str) -> dict | None:
    """Attempt to parse JSON with multiple recovery strategies.
    
    Returns the parsed dict if successful, None otherwise.
    """
    # Strategy 1: Direct parsing
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Strategy 2: Remove trailing commas before closing braces/brackets
    try:
        cleaned = re.sub(r',(\s*[}\]])', r'\1', text)
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    
    # Strategy 3: Fix single quotes to double quotes (common LLM error)
    try:
        # Replace single quotes around keys and string values
        fixed = re.sub(r"'([^']*?)'\s*:", r'"\1":', text)
        fixed = re.sub(r":\s*'([^']*?)'", r': "\1"', fixed)
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    
    # Strategy 4: Extract just the response field if present
    try:
        response_match = re.search(r'"response"\s*:\s*"([^"]*?)"', text)
        if response_match:
            return {"response": response_match.group(1)}
    except Exception:
        pass
    
    # Strategy 5: Handle unescaped newlines in strings (another common LLM error)
    try:
        # Replace newlines within string values with escaped newlines
        # This is a best-effort approach
        fixed = re.sub(r'(?<=")([^"]*?)\n([^"]*?)(?=")', r'\1\\n\2', text)
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    
    # Strategy 6: Extract reasoning and response fields separately
    try:
        reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]*?)"', text, re.DOTALL)
        response_match = re.search(r'"response"\s*:\s*"([^"]*?)"', text, re.DOTALL)
        result = {}
        if reasoning_match:
            result["reasoning"] = reasoning_match.group(1).replace('\\n', '\n')
        if response_match:
            result["response"] = response_match.group(1)
        if result:
            return result
    except Exception:
        pass
    
    return None


class TaskAgent:
    """Task agent that solves IMO grading problems with enhanced reasoning and caching."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "", use_cache: bool = True) -> None:
        self.model = model
        self.log_fn = logger.info
        self.use_cache = use_cache

    def get_cache_stats(self) -> dict:
        """Get current cache statistics."""
        return get_cache_stats()

    def _build_prompt(self, inputs: dict) -> str:
        """Build a structured prompt for IMO grading with chain-of-thought reasoning."""
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        return f"""You are an expert IMO (International Mathematical Olympiad) grader.

Your task is to evaluate a student's solution to a mathematical problem.

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

1. First, analyze the student's answer step by step. Compare it against the official solution.
2. Identify any errors, missing steps, or creative alternative approaches.
3. Consider the grading guidelines carefully.
4. Provide your reasoning before giving the final grade.

Respond in JSON format with the following schema:
<json>
{{
    "reasoning": "Your detailed analysis of the student's answer...",
    "response": "Your final grade/evaluation (e.g., '7', '6', 'Partial credit: 3', 'Incorrect', etc.)"
}}
</json>

The "response" field should contain only the final grade/evaluation, while "reasoning" contains your full analysis."""

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with enhanced reasoning and caching.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        global _response_cache, _cache_hits, _cache_misses, _cache_access_times
        
        # Check cache first if enabled
        if self.use_cache:
            cache_key = _get_cache_key(inputs, self.model)
            if cache_key in _response_cache:
                _cache_hits += 1
                _cache_access_times[cache_key] = time.time()
                self.log_fn(f"Cache hit for problem: {inputs.get('problem', '')[:50]}... (hits: {_cache_hits}, misses: {_cache_misses})")
                return _response_cache[cache_key]
            _cache_misses += 1
        
        instruction = self._build_prompt(inputs)

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON with better error handling
        prediction = "None"
        reasoning = ""
        try:
            # Try to extract from the last assistant message
            last_msg = msg_history[-1]["text"] if msg_history else ""
            extracted = _extract_jsons(last_msg)
            
            if extracted:
                last_json = extracted[-1]
                if "response" in last_json:
                    prediction = last_json["response"]
                if "reasoning" in last_json:
                    reasoning = last_json["reasoning"]
                
                # Log the reasoning for debugging
                if reasoning:
                    self.log_fn(f"Reasoning: {reasoning[:200]}...")
            else:
                # Fallback: try to find any JSON-like structure
                json_match = re.search(r'\{[^}]*"response"[^}]*\}', last_msg)
                if json_match:
                    try:
                        fallback = json.loads(json_match.group())
                        prediction = fallback.get("response", "None")
                    except json.JSONDecodeError:
                        pass
                        
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        result = (str(prediction), msg_history)
        
        # Store in cache if enabled
        if self.use_cache:
            cache_key = _get_cache_key(inputs, self.model)
            _response_cache[cache_key] = result
            _cache_access_times[cache_key] = time.time()
            # LRU eviction: remove least recently used entries when over limit
            if len(_response_cache) > _MAX_CACHE_SIZE:
                # Find and remove the least recently used entry
                lru_key = min(_cache_access_times, key=_cache_access_times.get)
                del _response_cache[lru_key]
                del _cache_access_times[lru_key]
                logger.debug(f"Cache eviction: removed LRU entry, size now {len(_response_cache)}")
        
        return result


# Module-level cache management functions
def _get_cache_stats() -> dict:
    """Internal function to get cache statistics."""
    total = _cache_hits + _cache_misses
    hit_rate = _cache_hits / total if total > 0 else 0.0
    return {
        "size": len(_response_cache),
        "max_size": _MAX_CACHE_SIZE,
        "hits": _cache_hits,
        "misses": _cache_misses,
        "hit_rate": round(hit_rate, 4),
        "total_requests": total,
    }
