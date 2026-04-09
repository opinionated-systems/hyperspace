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

# Simple in-memory cache for task results
_task_cache: dict[str, tuple[str, list[dict]]] = {}
MAX_CACHE_SIZE = 100


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
    """
    results = []
    # Pattern to match JSON objects (handles nested braces)
    pattern = r'\{[^{}]*"response"[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.finditer(pattern, text, re.DOTALL)

    for match in matches:
        try:
            obj = json.loads(match.group())
            if "response" in obj:
                results.append(obj)
        except json.JSONDecodeError:
            continue

    # If regex fails, try to find any JSON-like structure
    if not results:
        try:
            # Try to parse the entire text as JSON
            obj = json.loads(text.strip())
            if isinstance(obj, dict) and "response" in obj:
                results.append(obj)
        except json.JSONDecodeError:
            pass

    return results or None


def _compute_cache_key(inputs: dict) -> str:
    """Compute a cache key from task inputs.
    
    Uses a hash of the normalized inputs to create a unique key.
    Only considers the problem, solution, and student_answer for caching,
    as these are the core content that determines the grading result.
    """
    # Normalize the key fields
    key_fields = {
        "problem": inputs.get("problem", ""),
        "solution": inputs.get("solution", ""),
        "student_answer": inputs.get("student_answer", ""),
        "grading_guidelines": inputs.get("grading_guidelines", ""),
    }
    # Create a deterministic JSON representation
    key_str = json.dumps(key_fields, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(key_str.encode()).hexdigest()[:32]


def _get_cached_result(cache_key: str) -> tuple[str, list[dict]] | None:
    """Get a cached result if available."""
    return _task_cache.get(cache_key)


def _set_cached_result(cache_key: str, result: tuple[str, list[dict]]) -> None:
    """Cache a result with LRU eviction."""
    global _task_cache
    
    # Simple LRU: if cache is full, clear half of it
    if len(_task_cache) >= MAX_CACHE_SIZE:
        # Remove oldest half of entries
        keys_to_remove = list(_task_cache.keys())[:MAX_CACHE_SIZE // 2]
        for key in keys_to_remove:
            del _task_cache[key]
        logger.info(f"Cache evicted {len(keys_to_remove)} entries")
    
    _task_cache[cache_key] = result


def clear_task_cache() -> None:
    """Clear the task cache. Useful for testing or memory management."""
    global _task_cache
    _task_cache.clear()
    logger.info("Task cache cleared")


class TaskAgent:
    """Task agent that solves IMO grading problems with robust JSON extraction and caching."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "", enable_cache: bool = True) -> None:
        self.model = model
        self.log_fn = logger.info
        self.call_count = 0
        self.cache_hits = 0
        self.enable_cache = enable_cache

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
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

        # Check cache for existing result
        if self.enable_cache:
            cache_key = _compute_cache_key(inputs)
            cached_result = _get_cached_result(cache_key)
            if cached_result is not None:
                self.cache_hits += 1
                self.log_fn(f"Cache hit! (hit #{self.cache_hits}, cache size: {len(_task_cache)})")
                return cached_result
            self.log_fn(f"Cache miss (cache size: {len(_task_cache)})")

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

        self.log_fn(f"Extraction method used: {extraction_method}, prediction type: {type(prediction).__name__}")
        
        result = (str(prediction), msg_history)
        
        # Cache the result
        if self.enable_cache:
            cache_key = _compute_cache_key(inputs)
            _set_cached_result(cache_key, result)
            self.log_fn(f"Result cached (cache size: {len(_task_cache)})")
        
        return result
    
    def get_stats(self) -> dict:
        """Get agent statistics including cache performance."""
        return {
            "call_count": self.call_count,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": self.cache_hits / max(1, self.call_count - self.cache_hits),
            "cache_size": len(_task_cache),
            "cache_enabled": self.enable_cache,
        }
