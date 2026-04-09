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
from collections import OrderedDict

from agent.llm_client import get_response_from_llm, EVAL_MODEL
from agent.utils.validation import validate_inputs, sanitize_string

logger = logging.getLogger(__name__)

# LRU cache for LLM responses with automatic eviction
_MAX_CACHE_SIZE = 1000
_response_cache: OrderedDict[str, tuple[str, list[dict]]] = OrderedDict()
_cache_hits = 0
_cache_misses = 0


def _normalize_inputs(inputs: dict) -> dict:
    """Normalize inputs for consistent caching and processing.
    
    - Strips whitespace from string values
    - Converts domain to lowercase
    - Removes redundant whitespace
    """
    normalized = {}
    for key, value in inputs.items():
        if isinstance(value, str):
            # Strip leading/trailing whitespace and normalize internal whitespace
            normalized[key] = re.sub(r'\s+', ' ', value.strip())
        else:
            normalized[key] = value
    
    # Normalize domain to lowercase
    if 'domain' in normalized and isinstance(normalized['domain'], str):
        normalized['domain'] = normalized['domain'].lower().strip()
    
    return normalized


def _get_cache_key(inputs: dict) -> str:
    """Generate a cache key from inputs dict."""
    # Normalize inputs for consistent caching
    normalized = _normalize_inputs(inputs)
    # Use full hash for better collision resistance
    normalized_str = json.dumps(normalized, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(normalized_str.encode()).hexdigest()


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
        "max_cache_size": _MAX_CACHE_SIZE,
    }


def clear_cache() -> None:
    """Clear the response cache."""
    global _response_cache, _cache_hits, _cache_misses
    _response_cache.clear()
    _cache_hits = 0
    _cache_misses = 0
    logger.info("Response cache cleared")


def _add_to_cache(key: str, value: tuple[str, list[dict]]) -> None:
    """Add entry to LRU cache, evicting oldest if at capacity."""
    global _response_cache
    
    # If key exists, move it to end (most recently used)
    if key in _response_cache:
        _response_cache.move_to_end(key)
    
    # Add new entry
    _response_cache[key] = value
    
    # Evict oldest entries if over capacity
    while len(_response_cache) > _MAX_CACHE_SIZE:
        _response_cache.popitem(last=False)


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    Handles nested JSON structures by tracking brace depth.
    """
    results = []
    search_from = 0
    while True:
        start = text.find("<json>", search_from)
        if start == -1:
            break
        end_tag_start = text.find("</json>", start)
        if end_tag_start == -1:
            break
        
        # Extract content between tags
        inner_start = start + 6
        inner = text[inner_start:end_tag_start].strip()
        
        # Try to parse the inner content
        try:
            results.append(json.loads(inner))
        except json.JSONDecodeError:
            # Try to find valid JSON by tracking brace depth
            # This handles cases where there are nested structures
            brace_depth = 0
            json_start = -1
            for i, char in enumerate(inner):
                if char == '{':
                    if brace_depth == 0:
                        json_start = i
                    brace_depth += 1
                elif char == '}':
                    brace_depth -= 1
                    if brace_depth == 0 and json_start != -1:
                        try:
                            candidate = inner[json_start:i+1]
                            results.append(json.loads(candidate))
                            break  # Found valid JSON
                        except json.JSONDecodeError:
                            continue
        
        search_from = end_tag_start + 7
    
    return results or None


def _extract_json_fallback(text: str) -> list[dict] | None:
    """Fallback JSON extraction using multiple strategies.

    This handles cases where the model outputs valid JSON without <json> tags.
    Uses multiple strategies: brace depth tracking, regex, and full text parsing.
    """
    results = []
    
    # Strategy 1: Find JSON objects by tracking brace depth
    brace_depth = 0
    json_start = -1
    for i, char in enumerate(text):
        if char == '{':
            if brace_depth == 0:
                json_start = i
            brace_depth += 1
        elif char == '}':
            brace_depth -= 1
            if brace_depth == 0 and json_start != -1:
                try:
                    candidate = text[json_start:i+1]
                    obj = json.loads(candidate)
                    if "response" in obj:
                        results.append(obj)
                except json.JSONDecodeError:
                    pass
                json_start = -1
    
    # Strategy 2: Pattern matching for simple cases (if depth tracking failed)
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
    
    # Strategy 4: Try to extract JSON from markdown code blocks
    if not results:
        code_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
        for match in re.finditer(code_block_pattern, text, re.DOTALL):
            try:
                obj = json.loads(match.group(1).strip())
                if isinstance(obj, dict) and "response" in obj:
                    results.append(obj)
            except json.JSONDecodeError:
                continue
    
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

        # Store in cache if enabled using LRU eviction
        if self.enable_cache:
            _add_to_cache(cache_key, (str(prediction), msg_history))

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
