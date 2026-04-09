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
from dataclasses import dataclass, field
from functools import lru_cache

from agent.llm_client import get_response_from_llm, EVAL_MODEL
from agent.utils.validation import validate_inputs, sanitize_string

logger = logging.getLogger(__name__)


@dataclass
class _CacheEntry:
    """Cache entry with TTL support for automatic expiration."""
    prediction: str
    msg_history: list[dict]
    timestamp: float = field(default_factory=time.time)
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)


# Enhanced in-memory cache with TTL and LRU eviction
_response_cache: dict[str, _CacheEntry] = {}
_cache_hits = 0
_cache_misses = 0
_cache_config = {
    "max_size": 1000,
    "ttl_seconds": 3600,  # 1 hour default TTL
    "enable_lru": True,   # Enable LRU eviction when cache is full
}


def _get_cache_key(inputs: dict) -> str:
    """Generate a cache key from inputs dict with enhanced normalization."""
    # Normalize inputs for consistent caching
    # Remove whitespace variations that don't affect semantic meaning
    normalized_inputs = {}
    for key, value in inputs.items():
        if isinstance(value, str):
            # Normalize whitespace but preserve structure
            normalized_inputs[key] = " ".join(value.split())
        else:
            normalized_inputs[key] = value
    
    normalized = json.dumps(normalized_inputs, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(normalized.encode()).hexdigest()[:32]


def get_cache_stats() -> dict[str, Any]:
    """Get comprehensive cache statistics."""
    global _cache_hits, _cache_misses
    total = _cache_hits + _cache_misses
    hit_rate = (_cache_hits / total * 100) if total > 0 else 0
    
    # Calculate additional metrics
    now = time.time()
    expired_count = sum(
        1 for entry in _response_cache.values()
        if now - entry.timestamp > _cache_config["ttl_seconds"]
    )
    total_accesses = sum(entry.access_count for entry in _response_cache.values())
    
    return {
        "hits": _cache_hits,
        "misses": _cache_misses,
        "total": total,
        "hit_rate_percent": round(hit_rate, 2),
        "cache_size": len(_response_cache),
        "max_size": _cache_config["max_size"],
        "expired_entries": expired_count,
        "total_accesses": total_accesses,
        "avg_access_count": round(total_accesses / len(_response_cache), 2) if _response_cache else 0,
    }


def clear_cache() -> None:
    """Clear the response cache and reset statistics."""
    global _response_cache, _cache_hits, _cache_misses
    _response_cache.clear()
    _cache_hits = 0
    _cache_misses = 0
    logger.info("Response cache cleared")


def configure_cache(max_size: int | None = None, ttl_seconds: float | None = None, enable_lru: bool | None = None) -> dict:
    """Configure cache parameters.
    
    Args:
        max_size: Maximum number of entries in cache
        ttl_seconds: Time-to-live for cache entries in seconds
        enable_lru: Whether to use LRU eviction policy
        
    Returns:
        Current cache configuration
    """
    global _cache_config
    if max_size is not None:
        _cache_config["max_size"] = max(max_size, 10)  # Minimum 10 entries
    if ttl_seconds is not None:
        _cache_config["ttl_seconds"] = max(ttl_seconds, 60)  # Minimum 1 minute
    if enable_lru is not None:
        _cache_config["enable_lru"] = enable_lru
    return dict(_cache_config)


def _cleanup_expired_entries() -> int:
    """Remove expired entries from cache. Returns number of entries removed."""
    global _response_cache
    now = time.time()
    ttl = _cache_config["ttl_seconds"]
    expired_keys = [
        key for key, entry in _response_cache.items()
        if now - entry.timestamp > ttl
    ]
    for key in expired_keys:
        del _response_cache[key]
    return len(expired_keys)


def _evict_lru_entry() -> None:
    """Evict least recently used entry when cache is full."""
    if not _response_cache:
        return
    # Find entry with oldest last_accessed time
    lru_key = min(_response_cache.keys(), key=lambda k: _response_cache[k].last_accessed)
    del _response_cache[lru_key]
    logger.debug(f"LRU eviction: removed entry {lru_key[:8]}...")


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
            
            # Periodic cleanup of expired entries (every 100 requests)
            if self.call_count % 100 == 0:
                cleaned = _cleanup_expired_entries()
                if cleaned > 0:
                    self.log_fn(f"Cache cleanup: removed {cleaned} expired entries")
            
            # Check for cache hit
            if cache_key in _response_cache:
                entry = _response_cache[cache_key]
                now = time.time()
                
                # Check if entry is expired
                if now - entry.timestamp > _cache_config["ttl_seconds"]:
                    _cache_misses += 1
                    self.log_fn(f"Cache entry expired, treating as miss")
                    del _response_cache[cache_key]
                else:
                    # Valid cache hit - update access stats
                    _cache_hits += 1
                    entry.access_count += 1
                    entry.last_accessed = now
                    self.log_fn(f"Cache hit! Using cached response (hit rate: {get_cache_stats()['hit_rate_percent']:.1f}%, access count: {entry.access_count})")
                    return entry.prediction, entry.msg_history
            else:
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
            # Check if we need to evict entries
            if len(_response_cache) >= _cache_config["max_size"]:
                if _cache_config["enable_lru"]:
                    _evict_lru_entry()
                else:
                    # Simple FIFO eviction
                    oldest_key = next(iter(_response_cache))
                    del _response_cache[oldest_key]
            
            # Create new cache entry with metadata
            _response_cache[cache_key] = _CacheEntry(
                prediction=str(prediction),
                msg_history=msg_history,
            )
            self.log_fn(f"Cached response (cache size: {len(_response_cache)}/{_cache_config['max_size']})")

        self.log_fn(f"Extraction method used: {extraction_method}, prediction type: {type(prediction).__name__}")
        return str(prediction), msg_history

    def forward_batch(
        self,
        inputs_list: list[dict],
        continue_on_error: bool = True,
    ) -> list[tuple[str, list[dict]]]:
        """Process multiple inputs in batch.

        Args:
            inputs_list: List of input dicts, each with domain, problem, solution, etc.
            continue_on_error: If True, continue processing remaining items after an error.
                               If False, stop on first error.

        Returns:
            List of (prediction, msg_history) tuples, one per input.
        """
        results = []
        total = len(inputs_list)
        
        self.log_fn(f"Batch processing started: {total} items")
        
        for i, inputs in enumerate(inputs_list, 1):
            self.log_fn(f"Processing item {i}/{total}")
            try:
                prediction, msg_history = self.forward(inputs)
                results.append((prediction, msg_history))
            except Exception as e:
                error_msg = f"Error processing item {i}: {e}"
                self.log_fn(error_msg)
                results.append((f"Error: {e}", [{"role": "system", "text": error_msg}]))
                if not continue_on_error:
                    self.log_fn("Stopping batch processing due to error (continue_on_error=False)")
                    break
        
        self.log_fn(f"Batch processing completed: {len(results)}/{total} items processed")
        return results
