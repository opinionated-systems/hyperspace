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
from typing import Any
from collections import OrderedDict
from functools import lru_cache

from agent.llm_client import get_response_from_llm, EVAL_MODEL
from agent.utils.validation import validate_inputs, sanitize_string

logger = logging.getLogger(__name__)

# LRU cache with access tracking for better performance
_response_cache: OrderedDict[str, tuple[str, list[dict], float]] = OrderedDict()
_cache_hits = 0
_cache_misses = 0
_cache_max_size = 1000
_cache_ttl_seconds = 3600  # 1 hour TTL for cache entries


def _get_cache_key(inputs: dict) -> str:
    """Generate a cache key from inputs dict."""
    # Normalize inputs for consistent caching
    normalized = json.dumps(inputs, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(normalized.encode()).hexdigest()[:32]


def get_cache_stats() -> dict[str, Any]:
    """Get cache statistics."""
    global _cache_hits, _cache_misses
    total = _cache_hits + _cache_misses
    hit_rate = (_cache_hits / total * 100) if total > 0 else 0
    
    # Calculate average age of cache entries
    current_time = time.time()
    ages = [current_time - entry[2] for entry in _response_cache.values()]
    avg_age = sum(ages) / len(ages) if ages else 0
    
    return {
        "hits": _cache_hits,
        "misses": _cache_misses,
        "total": total,
        "hit_rate_percent": round(hit_rate, 2),
        "cache_size": len(_response_cache),
        "max_size": _cache_max_size,
        "avg_entry_age_seconds": round(avg_age, 2),
        "ttl_seconds": _cache_ttl_seconds,
    }


def clear_cache() -> None:
    """Clear the response cache."""
    global _response_cache, _cache_hits, _cache_misses
    _response_cache.clear()
    _cache_hits = 0
    _cache_misses = 0
    logger.info("Response cache cleared")


def _cleanup_expired_cache_entries() -> int:
    """Remove expired cache entries based on TTL. Returns number removed."""
    global _response_cache
    current_time = time.time()
    expired_keys = [
        key for key, (_, _, timestamp) in _response_cache.items()
        if current_time - timestamp > _cache_ttl_seconds
    ]
    for key in expired_keys:
        del _response_cache[key]
    return len(expired_keys)


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    Also handles nested JSON objects within the tags.
    """
    results = []
    search_from = 0
    while True:
        start = text.find("<json>", search_from)
        if start == -1:
            break
        
        # Find the matching </json> tag, accounting for nested braces
        tag_start = start + 6
        end = text.find("</json>", tag_start)
        if end == -1:
            break
        
        # Extract content and try to parse
        inner = text[tag_start:end].strip()
        
        # Try to find valid JSON by checking brace balance
        # This handles cases where model outputs multiple JSON objects
        brace_count = 0
        json_start = -1
        for i, char in enumerate(inner):
            if char == '{':
                if brace_count == 0:
                    json_start = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and json_start != -1:
                    # Found a complete JSON object
                    try:
                        obj = json.loads(inner[json_start:i+1])
                        results.append(obj)
                    except json.JSONDecodeError:
                        pass
                    json_start = -1
        
        # If no balanced JSON found, try the whole content
        if not results or json_start != -1:
            try:
                obj = json.loads(inner)
                results.append(obj)
            except json.JSONDecodeError:
                pass
        
        search_from = end + 7
    
    return results or None


def _extract_json_fallback(text: str) -> list[dict] | None:
    """Fallback JSON extraction using brace balance for unwrapped JSON objects.

    This handles cases where the model outputs valid JSON without <json> tags.
    Searches for JSON objects with a "response" key using brace counting.
    """
    results = []
    
    # Find all potential JSON objects by tracking brace balance
    i = 0
    while i < len(text):
        # Look for opening brace
        if text[i] == '{':
            brace_count = 1
            start = i
            i += 1
            
            # Track braces until balanced
            while i < len(text) and brace_count > 0:
                if text[i] == '{':
                    brace_count += 1
                elif text[i] == '}':
                    brace_count -= 1
                i += 1
            
            # If balanced, try to parse
            if brace_count == 0:
                try:
                    obj = json.loads(text[start:i])
                    if isinstance(obj, dict) and "response" in obj:
                        results.append(obj)
                except json.JSONDecodeError:
                    pass
        else:
            i += 1

    # If no results, try to find any JSON-like structure with response key
    if not results:
        try:
            # Try to parse the entire text as JSON
            obj = json.loads(text.strip())
            if isinstance(obj, dict) and "response" in obj:
                results.append(obj)
        except json.JSONDecodeError:
            pass

    # Last resort: try to extract anything that looks like a response value
    if not results:
        # Look for "response": "..." or "response": ... patterns
        response_match = re.search(r'"response"\s*:\s*("(?:[^"\\]|\\.)*"|[^,}\]]+)', text, re.DOTALL)
        if response_match:
            value = response_match.group(1).strip()
            # Try to parse as JSON value
            try:
                if value.startswith('"') and value.endswith('"'):
                    parsed_value = json.loads(value)
                else:
                    parsed_value = value
                results.append({"response": parsed_value})
            except json.JSONDecodeError:
                # Just use the raw value
                results.append({"response": value.strip('"')})

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
            # Periodic cleanup of expired entries (every 100 calls)
            if self.call_count % 100 == 0:
                removed = _cleanup_expired_cache_entries()
                if removed > 0:
                    self.log_fn(f"Cleaned up {removed} expired cache entries")
            
            cache_key = _get_cache_key(inputs)
            if cache_key in _response_cache:
                cached_prediction, cached_history, timestamp = _response_cache[cache_key]
                # Check if entry is still fresh
                if time.time() - timestamp <= _cache_ttl_seconds:
                    _cache_hits += 1
                    # Move to end (most recently used)
                    _response_cache.move_to_end(cache_key)
                    self.log_fn(f"Cache hit! Using cached response (hit rate: {get_cache_stats()['hit_rate_percent']:.1f}%)")
                    return cached_prediction, cached_history
                else:
                    # Entry expired, remove it
                    del _response_cache[cache_key]
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
            # Store with timestamp for TTL tracking
            _response_cache[cache_key] = (str(prediction), msg_history, time.time())
            # Move to end (most recently used)
            _response_cache.move_to_end(cache_key)
            # Limit cache size using LRU eviction
            while len(_response_cache) > _cache_max_size:
                # Remove oldest (first) entry
                oldest_key = next(iter(_response_cache))
                del _response_cache[oldest_key]
                self.log_fn(f"Cache full, evicted oldest entry")

        self.log_fn(f"Extraction method used: {extraction_method}, prediction type: {type(prediction).__name__}")
        return str(prediction), msg_history
