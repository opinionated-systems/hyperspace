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

from agent.llm_client import get_response_from_llm, EVAL_MODEL

logger = logging.getLogger(__name__)

# Simple in-memory cache for grading results to avoid redundant LLM calls
_grading_cache: dict[str, tuple[str, float]] = {}
_CACHE_MAX_SIZE = 100


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


def _extract_json_with_retry(text: str, max_retries: int = 2) -> list[dict] | None:
    """Extract JSON with multiple fallback strategies.
    
    First tries standard extraction, then attempts to fix common
    JSON formatting issues like trailing commas.
    """
    # First attempt: standard extraction
    result = _extract_jsons(text)
    if result:
        return result
    
    # Retry with common fixes
    for attempt in range(max_retries):
        try:
            # Try to find and fix JSON with trailing commas
            fixed_text = re.sub(r',(\s*[}\]])', r'\1', text)
            result = _extract_jsons(fixed_text)
            if result:
                logger.debug(f"JSON extraction succeeded after fix attempt {attempt + 1}")
                return result
        except Exception:
            pass
    
    return None


def _extract_json_flexible(text: str) -> dict | None:
    """Extract JSON using multiple fallback strategies.
    
    Tries multiple approaches:
    1. Standard <json>...</json> extraction
    2. Direct JSON object parsing from the entire text
    3. Regex-based extraction of JSON-like structures
    """
    # Strategy 1: Standard extraction
    result = _extract_json_with_retry(text)
    if result:
        return result[-1] if result else None
    
    # Strategy 2: Try to parse the entire text as JSON
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    
    # Strategy 3: Look for JSON objects with regex
    # Match content between outermost braces
    json_pattern = re.compile(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', re.DOTALL)
    matches = json_pattern.findall(text)
    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue
    
    return None


def _compute_cache_key(inputs: dict) -> str:
    """Compute a cache key from grading inputs."""
    # Hash the key components for efficient lookup
    key_data = json.dumps(inputs, sort_keys=True)
    return hashlib.sha256(key_data.encode()).hexdigest()[:32]


def _get_cached_result(cache_key: str) -> tuple[str, float] | None:
    """Get cached result if it exists and hasn't expired."""
    if cache_key in _grading_cache:
        result, timestamp = _grading_cache[cache_key]
        # Cache entries expire after 1 hour (3600 seconds)
        if time.time() - timestamp < 3600:
            logger.debug(f"Cache hit for key {cache_key[:8]}...")
            return (result, timestamp)
        else:
            # Remove expired entry
            del _grading_cache[cache_key]
    return None


def _cache_result(cache_key: str, result: str) -> None:
    """Cache a grading result with LRU eviction."""
    global _grading_cache
    
    # Evict oldest entries if cache is full
    if len(_grading_cache) >= _CACHE_MAX_SIZE:
        # Remove oldest 20% of entries
        sorted_items = sorted(_grading_cache.items(), key=lambda x: x[1][1])
        for key, _ in sorted_items[:_CACHE_MAX_SIZE // 5]:
            del _grading_cache[key]
        logger.debug(f"Cache evicted {len(sorted_items[:_CACHE_MAX_SIZE // 5])} old entries")
    
    _grading_cache[cache_key] = (result, time.time())
    logger.debug(f"Cached result for key {cache_key[:8]}...")


def clear_grading_cache() -> int:
    """Clear all cached grading results. Returns number of entries cleared."""
    global _grading_cache
    count = len(_grading_cache)
    _grading_cache.clear()
    logger.info(f"Cleared {count} entries from grading cache")
    return count


def get_cache_stats() -> dict:
    """Get statistics about the grading cache."""
    global _grading_cache
    current_time = time.time()
    expired_count = 0
    valid_count = 0
    
    for result, timestamp in _grading_cache.values():
        if current_time - timestamp >= 3600:
            expired_count += 1
        else:
            valid_count += 1
    
    return {
        "total_entries": len(_grading_cache),
        "valid_entries": valid_count,
        "expired_entries": expired_count,
        "max_size": _CACHE_MAX_SIZE,
        "utilization_pct": round(len(_grading_cache) / _CACHE_MAX_SIZE * 100, 1),
    }


class TaskAgent:
    """Task agent that solves IMO grading problems with caching support."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "", use_cache: bool = True) -> None:
        self.model = model
        self.log_fn = logger.info
        self.use_cache = use_cache

    def _format_grading_prompt(self, inputs: dict) -> str:
        """Format the grading prompt with structured instructions."""
        domain = inputs.get("domain", "mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        return f"""You are an expert grader for {domain} problems. Your task is to evaluate a student's answer against the official solution and grading guidelines.

## Problem Statement
{problem}

## Official Solution
{solution}

## Grading Guidelines
{grading_guidelines}

## Student's Answer
{student_answer}

## Your Task
Evaluate the student's answer carefully. Consider:
1. Mathematical correctness and reasoning
2. Completeness of the solution
3. Adherence to the grading guidelines
4. Partial credit where applicable

Provide your evaluation in the following JSON format:
<json>
{{
    "response": "Your detailed grading feedback here. Include: (1) whether the answer is correct/partially correct/incorrect, (2) specific points awarded and why, (3) any errors or omissions noted."
}}
</json>"""

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with caching support.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Check cache first if enabled
        if self.use_cache:
            cache_key = _compute_cache_key(inputs)
            cached = _get_cached_result(cache_key)
            if cached:
                self.log_fn(f"Cache hit - returning cached grading result")
                return cached[0], []

        instruction = self._format_grading_prompt(inputs)

        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
        except Exception as e:
            self.log_fn(f"LLM call failed: {e}")
            return f"Error: LLM call failed - {e}", []

        # Extract prediction from JSON with multiple fallback strategies
        prediction = "None"
        try:
            if msg_history and len(msg_history) > 0:
                last_message = msg_history[-1].get("text", "")
                extracted = _extract_json_flexible(last_message)
                if extracted and "response" in extracted:
                    prediction = extracted["response"]
                    self.log_fn(f"Successfully extracted grading response")
                else:
                    # Fallback: use the raw response if JSON extraction fails
                    prediction = last_message[:1000]  # Truncate for safety
                    self.log_fn(f"JSON extraction failed, using raw response (truncated)")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            # Last resort fallback
            try:
                prediction = str(response)[:1000]
            except:
                prediction = "Error: Could not extract or generate response"

        # Cache the result if caching is enabled
        if self.use_cache and prediction and not prediction.startswith("Error:"):
            _cache_result(cache_key, str(prediction))

        return str(prediction), msg_history
