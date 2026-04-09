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
from typing import Any
from functools import lru_cache

from agent.llm_client import get_response_from_llm, EVAL_MODEL
from agent.utils.validation import validate_inputs, sanitize_string

logger = logging.getLogger(__name__)

# Simple in-memory cache for LLM responses to improve performance
# Uses OrderedDict for LRU (Least Recently Used) eviction policy
from collections import OrderedDict

_response_cache: OrderedDict[str, tuple[str, list[dict]]] = OrderedDict()
_cache_hits = 0
_cache_misses = 0
_MAX_CACHE_SIZE = 1000


def _get_cache_key(inputs: dict) -> str:
    """Generate a cache key from inputs dict."""
    # Normalize inputs for consistent caching
    normalized = json.dumps(inputs, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(normalized.encode()).hexdigest()[:32]


def get_cache_stats() -> dict[str, int]:
    """Get cache statistics."""
    global _cache_hits, _cache_misses
    total = _cache_hits + _cache_misses
    hit_rate = (_cache_hits / total * 100) if total > 0 else 0
    return {
        "hits": _cache_hits,
        "misses": _cache_misses,
        "total": total,
        "hit_rate_percent": round(hit_rate, 2),
        "cache_size": len(_response_cache),
    }


def clear_cache() -> None:
    """Clear the response cache."""
    global _response_cache, _cache_hits, _cache_misses
    _response_cache.clear()
    _cache_hits = 0
    _cache_misses = 0
    logger.info("Response cache cleared")


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
    Uses a robust brace-matching algorithm to handle nested structures.
    """
    results = []
    
    # Find all potential JSON object start positions
    for start in re.finditer(r'\{', text):
        start_idx = start.start()
        
        # Try to find the matching closing brace
        brace_count = 1
        end_idx = start_idx + 1
        
        while brace_count > 0 and end_idx < len(text):
            if text[end_idx] == '{':
                brace_count += 1
            elif text[end_idx] == '}':
                brace_count -= 1
            end_idx += 1
        
        if brace_count == 0:
            # Found a complete JSON object
            json_str = text[start_idx:end_idx]
            try:
                obj = json.loads(json_str)
                if isinstance(obj, dict) and "response" in obj:
                    results.append(obj)
            except json.JSONDecodeError:
                continue
    
    # If no results found, try to find JSON arrays containing objects with "response"
    if not results:
        for start in re.finditer(r'\[', text):
            start_idx = start.start()
            
            # Try to find the matching closing bracket
            bracket_count = 1
            brace_count = 0
            in_string = False
            escape_next = False
            end_idx = start_idx + 1
            
            while bracket_count > 0 and end_idx < len(text):
                char = text[end_idx]
                
                if escape_next:
                    escape_next = False
                elif char == '\\':
                    escape_next = True
                elif char == '"' and not in_string:
                    in_string = True
                elif char == '"' and in_string:
                    in_string = False
                elif not in_string:
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                    elif char == '[':
                        bracket_count += 1
                    elif char == ']':
                        bracket_count -= 1
                
                end_idx += 1
            
            if bracket_count == 0:
                json_str = text[start_idx:end_idx]
                try:
                    arr = json.loads(json_str)
                    if isinstance(arr, list):
                        for obj in arr:
                            if isinstance(obj, dict) and "response" in obj:
                                results.append(obj)
                except json.JSONDecodeError:
                    continue
    
    # Final fallback: try to parse the entire text as JSON
    if not results:
        try:
            obj = json.loads(text.strip())
            if isinstance(obj, dict) and "response" in obj:
                results.append(obj)
            elif isinstance(obj, list):
                for item in obj:
                    if isinstance(item, dict) and "response" in item:
                        results.append(item)
        except json.JSONDecodeError:
            pass

    return results or None


class TaskAgent:
    """Task agent that solves IMO grading problems with robust JSON extraction and response caching."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "", enable_cache: bool = True) -> None:
        self.model = model
        self.log_fn = logger.info
        self.call_count = 0
        self.enable_cache = enable_cache

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        global _cache_hits, _cache_misses, _response_cache
        
        self.call_count += 1
        self.log_fn(f"TaskAgent call #{self.call_count} starting")
        
        # Validate inputs using the new validation module
        is_valid, error_msg = validate_inputs(inputs)
        if not is_valid:
            self.log_fn(f"Error: {error_msg}")
            return f"Error: {error_msg}", [{"role": "system", "text": error_msg}]
        
        # Log input keys for debugging
        input_keys = list(inputs.keys())
        self.log_fn(f"Input keys: {input_keys}")

        # Check cache if enabled
        if self.enable_cache:
            cache_key = _get_cache_key(inputs)
            if cache_key in _response_cache:
                _cache_hits += 1
                # Move to end (most recently used) for LRU tracking
                cached_prediction, cached_history = _response_cache.pop(cache_key)
                _response_cache[cache_key] = (cached_prediction, cached_history)
                self.log_fn(f"Cache hit! Using cached response (hit rate: {get_cache_stats()['hit_rate_percent']:.1f}%)")
                return cached_prediction, cached_history
            _cache_misses += 1

        instruction = f"""You are an agent.

Task input:
```
{inputs}
```

Respond in JSON format with the following schema:
<json>
{{
    "response": ...
}}
</json>"""

        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
            self.log_fn(f"LLM call successful, response length: {len(response)}")
        except Exception as e:
            self.log_fn(f"Error calling LLM: {e}")
            return "Error: LLM call failed", [{"role": "system", "text": f"Error: {e}"}]

        # Extract prediction from JSON using primary method
        prediction = "None"
        extraction_method = "primary"
        try:
            extracted = _extract_jsons(msg_history[-1]["text"])
            if extracted and "response" in extracted[-1]:
                prediction = extracted[-1]["response"]
                self.log_fn(f"Primary extraction successful")
            else:
                # Try fallback extraction
                self.log_fn(f"Primary extraction returned no valid response, trying fallback")
                extracted = _extract_json_fallback(msg_history[-1]["text"])
                if extracted and "response" in extracted[-1]:
                    prediction = extracted[-1]["response"]
                    extraction_method = "fallback"
                    self.log_fn(f"Fallback extraction successful")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            # Try fallback on exception
            try:
                extracted = _extract_json_fallback(msg_history[-1]["text"])
                if extracted and "response" in extracted[-1]:
                    prediction = extracted[-1]["response"]
                    extraction_method = "fallback"
                    self.log_fn(f"Fallback extraction successful after primary error")
            except Exception as fallback_e:
                self.log_fn(f"Fallback extraction also failed: {fallback_e}")

        # Store in cache if enabled
        if self.enable_cache:
            # Add new entry (will be at the end as most recently used)
            _response_cache[cache_key] = (str(prediction), msg_history)
            # Limit cache size using LRU eviction - remove oldest (first) entries
            while len(_response_cache) > _MAX_CACHE_SIZE:
                oldest_key = next(iter(_response_cache))
                del _response_cache[oldest_key]

        self.log_fn(f"Extraction method used: {extraction_method}, prediction type: {type(prediction).__name__}")
        return str(prediction), msg_history
