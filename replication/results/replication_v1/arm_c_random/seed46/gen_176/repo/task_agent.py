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
from agent.utils.validation import validate_inputs, validate_inputs_typed, sanitize_string

logger = logging.getLogger(__name__)

# Simple in-memory cache for LLM responses to improve performance
_response_cache: dict[str, tuple[str, list[dict]]] = {}
_cache_hits = 0
_cache_misses = 0


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
    Enhanced to handle nested JSON objects and malformed tags.
    """
    results = []
    search_from = 0
    max_iterations = 100  # Prevent infinite loops on malformed input
    iterations = 0
    
    while iterations < max_iterations:
        iterations += 1
        start = text.find("<json>", search_from)
        if start == -1:
            break
        end = text.find("</json>", start)
        if end == -1:
            # Malformed: opening tag without closing tag
            logger.warning(f"Malformed JSON tag: found <json> at position {start} but no closing </json>")
            break
        
        inner = text[start + 6:end].strip()
        search_from = end + 7
        
        # Skip empty blocks
        if not inner:
            continue
            
        try:
            parsed = json.loads(inner)
            if isinstance(parsed, dict):
                results.append(parsed)
            elif isinstance(parsed, list):
                # Handle case where JSON contains a list of objects
                for item in parsed:
                    if isinstance(item, dict):
                        results.append(item)
            else:
                # Wrap non-object JSON in a dict
                results.append({"value": parsed})
        except json.JSONDecodeError as e:
            logger.debug(f"JSON decode error at position {start}: {e}")
            continue
    
    if iterations >= max_iterations:
        logger.warning(f"Reached max iterations ({max_iterations}) while extracting JSON - possible malformed input")
    
    return results or None


def _extract_json_fallback(text: str) -> list[dict] | None:
    """Fallback JSON extraction using regex for unwrapped JSON objects.

    This handles cases where the model outputs valid JSON without <json> tags.
    Searches for JSON objects with a "response" key.
    Enhanced with better nested brace handling and multiple extraction strategies.
    """
    results = []
    
    # Strategy 1: Pattern to match JSON objects with response key (handles nested braces)
    # This uses a more sophisticated pattern that can handle one level of nesting
    pattern = r'\{[^{}]*"response"[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = list(re.finditer(pattern, text, re.DOTALL))
    
    for match in matches:
        try:
            obj = json.loads(match.group())
            if "response" in obj:
                results.append(obj)
        except json.JSONDecodeError:
            continue
    
    # Strategy 2: If no results, try a more aggressive brace-matching approach
    if not results:
        # Find all potential JSON object starts
        brace_starts = [m.start() for m in re.finditer(r'\{', text)]
        
        for start in brace_starts:
            # Try to find matching closing brace
            brace_count = 0
            end = start
            for i, char in enumerate(text[start:], start):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end = i + 1
                        break
            
            if end > start:
                try:
                    candidate = text[start:end]
                    obj = json.loads(candidate)
                    if isinstance(obj, dict) and "response" in obj:
                        results.append(obj)
                except json.JSONDecodeError:
                    continue
    
    # Strategy 3: Try to parse the entire text as JSON
    if not results:
        try:
            obj = json.loads(text.strip())
            if isinstance(obj, dict) and "response" in obj:
                results.append(obj)
            elif isinstance(obj, list):
                # Handle list of objects
                for item in obj:
                    if isinstance(item, dict) and "response" in item:
                        results.append(item)
        except json.JSONDecodeError:
            pass
    
    # Strategy 4: Look for JSON in code blocks
    if not results:
        code_block_pattern = r'```(?:json)?\s*([\s\S]*?)```'
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

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "", enable_cache: bool = True, strict_validation: bool = False) -> None:
        self.model = model
        self.log_fn = logger.info
        self.call_count = 0
        self.enable_cache = enable_cache
        self.strict_validation = strict_validation
        self._validation_stats: list[dict] = []

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
        
        # Validate inputs using the enhanced validation with metadata collection
        is_valid, error_msg, metadata = validate_inputs_typed(inputs, strict=self.strict_validation)
        if not is_valid:
            self.log_fn(f"Error: {error_msg}")
            return f"Error: {error_msg}", [{"role": "system", "text": error_msg}]
        
        # Store validation stats for potential debugging/analysis
        self._validation_stats.append(metadata)
        
        # Log input keys and field types for debugging
        input_keys = list(inputs.keys())
        self.log_fn(f"Input keys: {input_keys}")
        self.log_fn(f"Field types: {metadata['field_types']}")

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

        # Store in cache if enabled
        if self.enable_cache:
            _response_cache[cache_key] = (str(prediction), msg_history)
            # Limit cache size to prevent memory issues
            if len(_response_cache) > 1000:
                # Remove oldest entries (simple FIFO)
                oldest_key = next(iter(_response_cache))
                del _response_cache[oldest_key]

        self.log_fn(f"Extraction method used: {extraction_method}, prediction type: {type(prediction).__name__}")
        return str(prediction), msg_history
    
    def get_validation_stats(self) -> list[dict]:
        """Get collected validation statistics from all forward() calls.
        
        Returns:
            List of validation metadata dictionaries, one per successful call
        """
        return list(self._validation_stats)
    
    def clear_validation_stats(self) -> None:
        """Clear the collected validation statistics."""
        self._validation_stats.clear()
