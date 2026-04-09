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
from functools import lru_cache
from dataclasses import dataclass, field

from agent.llm_client import get_response_from_llm, EVAL_MODEL
from agent.utils.validation import validate_inputs, sanitize_string

logger = logging.getLogger(__name__)

# Simple in-memory cache for LLM responses to improve performance
_response_cache: dict[str, tuple[str, list[dict]]] = {}
_cache_hits = 0
_cache_misses = 0


@dataclass
class PerformanceStats:
    """Track performance statistics for task agent calls."""
    total_calls: int = 0
    total_llm_time: float = 0.0
    total_extraction_time: float = 0.0
    errors: list[str] = field(default_factory=list)
    
    @property
    def avg_llm_time(self) -> float:
        return self.total_llm_time / self.total_calls if self.total_calls > 0 else 0.0
    
    @property
    def avg_extraction_time(self) -> float:
        return self.total_extraction_time / self.total_calls if self.total_calls > 0 else 0.0
    
    def to_dict(self) -> dict:
        return {
            "total_calls": self.total_calls,
            "avg_llm_time_ms": round(self.avg_llm_time * 1000, 2),
            "avg_extraction_time_ms": round(self.avg_extraction_time * 1000, 2),
            "error_count": len(self.errors),
        }


# Global performance stats
_perf_stats = PerformanceStats()


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


def get_performance_stats() -> dict:
    """Get performance statistics."""
    global _perf_stats
    return _perf_stats.to_dict()


def reset_performance_stats() -> None:
    """Reset performance statistics."""
    global _perf_stats
    _perf_stats = PerformanceStats()
    logger.info("Performance stats reset")


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
    Also handles nested JSON objects within the response field.
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
        except json.JSONDecodeError as e:
            # Try to extract partial JSON if possible
            try:
                # Look for the response field specifically
                response_match = re.search(r'"response"\s*:\s*(.+?)(?:,\s*"|$)', inner, re.DOTALL)
                if response_match:
                    response_value = response_match.group(1).strip()
                    # Try to parse as JSON, or keep as string
                    try:
                        parsed_value = json.loads(response_value)
                    except json.JSONDecodeError:
                        parsed_value = response_value
                    results.append({"response": parsed_value})
                    logger.debug(f"Recovered partial JSON from malformed block: {e}")
            except Exception:
                pass
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


class TaskAgent:
    """Task agent that solves IMO grading problems with robust JSON extraction, response caching, and performance monitoring."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "", enable_cache: bool = True, track_performance: bool = True) -> None:
        self.model = model
        self.log_fn = logger.info
        self.call_count = 0
        self.enable_cache = enable_cache
        self.track_performance = track_performance

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        global _cache_hits, _cache_misses, _response_cache, _perf_stats
        
        self.call_count += 1
        self.log_fn(f"TaskAgent call #{self.call_count} starting")
        
        # Validate inputs using the new validation module
        is_valid, error_msg = validate_inputs(inputs)
        if not is_valid:
            self.log_fn(f"Error: {error_msg}")
            if self.track_performance:
                _perf_stats.errors.append(f"Call {self.call_count}: {error_msg}")
            return f"Error: {error_msg}", [{"role": "system", "text": error_msg}]
        
        # Log input keys for debugging
        input_keys = list(inputs.keys())
        self.log_fn(f"Input keys: {input_keys}")

        # Check cache if enabled
        if self.enable_cache:
            cache_key = _get_cache_key(inputs)
            if cache_key in _response_cache:
                _cache_hits += 1
                cached_prediction, cached_history = _response_cache[cache_key]
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

        # Time the LLM call
        llm_start = time.time()
        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
            llm_time = time.time() - llm_start
            if self.track_performance:
                _perf_stats.total_llm_time += llm_time
            self.log_fn(f"LLM call successful, response length: {len(response)}, time: {llm_time:.3f}s")
        except Exception as e:
            llm_time = time.time() - llm_start
            if self.track_performance:
                _perf_stats.total_llm_time += llm_time
                _perf_stats.errors.append(f"Call {self.call_count}: LLM error - {e}")
            self.log_fn(f"Error calling LLM: {e}")
            return "Error: LLM call failed", [{"role": "system", "text": f"Error: {e}"}]

        # Extract prediction from JSON using primary method
        extraction_start = time.time()
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
                if self.track_performance:
                    _perf_stats.errors.append(f"Call {self.call_count}: Extraction error - {fallback_e}")
        
        extraction_time = time.time() - extraction_start
        if self.track_performance:
            _perf_stats.total_extraction_time += extraction_time
            _perf_stats.total_calls += 1

        # Store in cache if enabled
        if self.enable_cache:
            _response_cache[cache_key] = (str(prediction), msg_history)
            # Limit cache size to prevent memory issues
            if len(_response_cache) > 1000:
                # Remove oldest entries (simple FIFO)
                oldest_key = next(iter(_response_cache))
                del _response_cache[oldest_key]

        self.log_fn(f"Extraction method used: {extraction_method}, prediction type: {type(prediction).__name__}, extraction time: {extraction_time:.3f}s")
        return str(prediction), msg_history
