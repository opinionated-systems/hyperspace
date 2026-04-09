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
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from agent.llm_client import get_response_from_llm, EVAL_MODEL

logger = logging.getLogger(__name__)


@dataclass
class _CacheEntry:
    """Cache entry with TTL support for automatic expiration."""
    prediction: str
    history: list[dict]
    timestamp: float
    access_count: int = 0
    
    def is_expired(self, ttl_seconds: float) -> bool:
        """Check if the cache entry has expired."""
        return time.time() - self.timestamp > ttl_seconds


# Enhanced LRU cache with TTL support
# This improves performance when similar inputs are processed
# and prevents stale cache entries from persisting indefinitely
_RESPONSE_CACHE: dict[str, _CacheEntry] = {}
_MAX_CACHE_SIZE = 128
_CACHE_TTL_SECONDS = 3600  # 1 hour default TTL
_CACHE_HITS = 0
_CACHE_MISSES = 0


def _get_cache_key(inputs: dict) -> str:
    """Generate a cache key from input dict with enhanced normalization."""
    # Normalize by sorting keys and using a stable JSON representation
    # Also handle nested dicts/lists consistently
    def normalize(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: normalize(v) for k, v in sorted(obj.items())}
        elif isinstance(obj, list):
            return [normalize(item) for item in obj]
        return obj
    
    normalized = json.dumps(normalize(inputs), separators=(',', ':'))
    return hashlib.sha256(normalized.encode()).hexdigest()[:32]


def get_cache_stats() -> dict[str, Any]:
    """Get current cache statistics."""
    global _CACHE_HITS, _CACHE_MISSES
    total = _CACHE_HITS + _CACHE_MISSES
    hit_rate = _CACHE_HITS / total if total > 0 else 0.0
    return {
        "hits": _CACHE_HITS,
        "misses": _CACHE_MISSES,
        "total": total,
        "hit_rate": f"{hit_rate:.2%}",
        "size": len(_RESPONSE_CACHE),
        "max_size": _MAX_CACHE_SIZE,
    }


def clear_cache() -> None:
    """Clear the response cache."""
    global _RESPONSE_CACHE, _CACHE_HITS, _CACHE_MISSES
    _RESPONSE_CACHE.clear()
    _CACHE_HITS = 0
    _CACHE_MISSES = 0
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
    """Fallback JSON extraction using multiple strategies for robust parsing.

    This handles cases where the model outputs valid JSON without <json> tags,
    or when the JSON is malformed. Uses a multi-layered approach:
    1. Brace depth parsing for nested structures
    2. Regex patterns for simpler cases
    3. Full text parsing
    4. Fuzzy extraction for malformed JSON
    """
    results = []
    
    # Strategy 1: Find JSON objects by parsing brace depth (handles nesting)
    def find_json_objects(s: str) -> list[str]:
        """Find potential JSON objects by tracking brace depth."""
        objects = []
        i = 0
        in_string = False
        escape_next = False
        
        while i < len(s):
            char = s[i]
            
            # Handle string escaping
            if escape_next:
                escape_next = False
                i += 1
                continue
            if char == '\\':
                escape_next = True
                i += 1
                continue
            if char == '"' and not in_string:
                in_string = True
                i += 1
                continue
            if char == '"' and in_string:
                in_string = False
                i += 1
                continue
            
            # Only track braces outside of strings
            if not in_string:
                if char == '{':
                    start = i
                    depth = 1
                    i += 1
                    while i < len(s) and depth > 0:
                        c = s[i]
                        if c == '\\':
                            i += 2
                            continue
                        if c == '"':
                            # Skip to end of string
                            i += 1
                            while i < len(s) and s[i] != '"':
                                if s[i] == '\\':
                                    i += 1
                                i += 1
                            if i < len(s):
                                i += 1
                            continue
                        if c == '{':
                            depth += 1
                        elif c == '}':
                            depth -= 1
                        i += 1
                    if depth == 0:
                        objects.append(s[start:i])
                    continue
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
    
    # Strategy 2: Regex pattern for simpler cases (no nesting)
    if not results:
        pattern = r'\{[^{}]*"response"[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.finditer(pattern, text, re.DOTALL)
        for match in matches:
            try:
                obj = json.loads(match.group())
                if "response" in obj:
                    results.append(obj)
            except json.JSONDecodeError:
                continue

    # Strategy 3: Try to parse the entire text as JSON
    if not results:
        try:
            obj = json.loads(text.strip())
            if isinstance(obj, dict) and "response" in obj:
                results.append(obj)
        except json.JSONDecodeError:
            pass
    
    # Strategy 4: Fuzzy extraction for malformed JSON
    if not results:
        # Look for patterns like "response": "..." or 'response': '...'
        # Handle both single and double quotes, with escaped quotes inside
        response_patterns = [
            r'"response"\s*:\s*"((?:[^"\\]|\\.)*)"',  # Double quotes
            r"'response'\s*:\s*'((?:[^'\\]|\\.)*)'",  # Single quotes
            r'"response"\s*:\s*(\d+)',  # Numeric values
            r'"response"\s*:\s*(true|false|null)',  # Boolean/null
        ]
        for pattern in response_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = match.group(1)
                # Unescape if needed
                value = value.replace('\\"', '"').replace("\\'", "'")
                # Only use non-empty values
                if value and value.strip():
                    results.append({"response": value})
                    break

    return results or None


class TaskAgent:
    """Task agent that solves IMO grading problems with robust JSON extraction.
    
    Features:
    - Robust JSON extraction with fallback strategies
    - Response caching with TTL support to avoid redundant LLM calls
    - Comprehensive error handling
    - Cache statistics tracking
    """

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
        global _CACHE_HITS, _CACHE_MISSES
        
        # Check cache first to avoid redundant LLM calls
        cache_key = _get_cache_key(inputs)
        if cache_key in _RESPONSE_CACHE:
            entry = _RESPONSE_CACHE[cache_key]
            if not entry.is_expired(_CACHE_TTL_SECONDS):
                _CACHE_HITS += 1
                entry.access_count += 1
                stats = get_cache_stats()
                self.log_fn(f"Cache hit! {stats['hit_rate']} hit rate ({stats['hits']}/{stats['total']})")
                # Return a copy to avoid modifying cached data
                return entry.prediction, list(entry.history)
            else:
                # Entry expired, remove it
                del _RESPONSE_CACHE[cache_key]
                self.log_fn("Cache entry expired, fetching fresh response")
        
        _CACHE_MISSES += 1
        
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
        
        # Cache the result for future identical inputs (LRU eviction with access count)
        if len(_RESPONSE_CACHE) >= _MAX_CACHE_SIZE:
            # Remove least recently used entry (lowest access count, then oldest)
            lru_key = min(
                _RESPONSE_CACHE.keys(),
                key=lambda k: (_RESPONSE_CACHE[k].access_count, _RESPONSE_CACHE[k].timestamp)
            )
            del _RESPONSE_CACHE[lru_key]
            self.log_fn(f"Evicted LRU cache entry: {lru_key[:8]}...")
        
        _RESPONSE_CACHE[cache_key] = _CacheEntry(
            prediction=prediction,
            history=list(msg_history),
            timestamp=time.time(),
            access_count=0
        )
        stats = get_cache_stats()
        self.log_fn(f"Cached result (cache size: {stats['size']}/{stats['max_size']}, hit rate: {stats['hit_rate']})")
            
        return prediction, msg_history
