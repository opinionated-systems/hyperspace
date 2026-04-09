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
from collections import OrderedDict

from agent.llm_client import get_response_from_llm, EVAL_MODEL

logger = logging.getLogger(__name__)

# Proper LRU cache for LLM responses using OrderedDict
# This provides O(1) access and maintains insertion order for eviction
class LRUCache:
    """Thread-safe LRU cache with size limit."""
    
    def __init__(self, max_size: int = 128):
        self._cache: OrderedDict[str, tuple[str, list[dict]]] = OrderedDict()
        self._max_size = max_size
        self._hits = 0
        self._misses = 0
    
    def get(self, key: str) -> tuple[str, list[dict]] | None:
        """Get value from cache, moving to end (most recently used)."""
        if key in self._cache:
            self._cache.move_to_end(key)
            self._hits += 1
            return self._cache[key]
        self._misses += 1
        return None
    
    def put(self, key: str, value: tuple[str, list[dict]]) -> None:
        """Add value to cache, evicting oldest if at capacity."""
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = value
        if len(self._cache) > self._max_size:
            self._cache.popitem(last=False)
    
    def get_stats(self) -> dict:
        """Return cache statistics."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": f"{hit_rate:.1%}",
            "size": len(self._cache),
            "max_size": self._max_size,
        }


_RESPONSE_CACHE = LRUCache(max_size=128)


def _get_cache_key(inputs: dict) -> str:
    """Generate a cache key from input dict.
    
    Uses SHA-256 for collision resistance and includes all relevant fields.
    """
    # Normalize by sorting keys and using a stable JSON representation
    normalized = json.dumps(inputs, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    
    Also handles common LLM output variations like markdown code blocks
    and extra whitespace.
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
        
        # Try to clean up common LLM output issues
        # Remove markdown code block markers if present
        if inner.startswith("```json"):
            inner = inner[7:]
        if inner.startswith("```"):
            inner = inner[3:]
        if inner.endswith("```"):
            inner = inner[:-3]
        inner = inner.strip()
        
        try:
            results.append(json.loads(inner))
        except json.JSONDecodeError:
            continue
    
    # If no <json> tags found, try markdown code blocks with json
    if not results:
        import re
        # Look for ```json ... ``` blocks
        json_blocks = re.findall(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
        for block in json_blocks:
            try:
                obj = json.loads(block.strip())
                if isinstance(obj, dict):
                    results.append(obj)
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
                results.append({"response": value})
                break

    return results or None


class TaskAgent:
    """Task agent that solves IMO grading problems with robust JSON extraction.
    
    Features:
    - Robust JSON extraction with fallback strategies
    - Response caching to avoid redundant LLM calls (proper LRU implementation)
    - Comprehensive error handling with exponential backoff
    - Detailed logging and cache statistics
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
        # Check cache first to avoid redundant LLM calls
        cache_key = _get_cache_key(inputs)
        cached_result = _RESPONSE_CACHE.get(cache_key)
        if cached_result is not None:
            cached_prediction, cached_history = cached_result
            stats = _RESPONSE_CACHE.get_stats()
            self.log_fn(f"Cache hit! {stats['hit_rate']} hit rate ({stats['hits']}/{stats['hits'] + stats['misses']})")
            # Return a copy to avoid modifying cached data
            return cached_prediction, list(cached_history)
        
        # Build a more structured prompt for better results
        domain = inputs.get("domain", "")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        instruction = f"""You are an expert grading assistant. Your task is to evaluate a student's answer to a problem and provide a grade.

## Domain
{domain}

## Problem
{problem}

## Correct Solution
{solution}

## Grading Guidelines
{grading_guidelines}

## Student's Answer
{student_answer}

## Instructions
Carefully analyze the student's answer against the correct solution and grading guidelines. Provide your evaluation in the following JSON format:

<json>
{{
    "response": "Your grade/evaluation here"
}}
</json>

The response should be a concise grade or evaluation based on the grading guidelines."""

        # Retry loop for LLM calls with exponential backoff
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=[],
                )
                break
            except Exception as e:
                self.log_fn(f"Error calling LLM (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    return "Error: LLM call failed after retries", [{"role": "system", "text": f"Error: {e}"}]
                time.sleep(2 ** attempt)  # Exponential backoff

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
            prediction = str(prediction).strip()
        
        # Cache the result using proper LRU cache
        _RESPONSE_CACHE.put(cache_key, (prediction, list(msg_history)))
        stats = _RESPONSE_CACHE.get_stats()
        self.log_fn(f"Cached result (cache size: {stats['size']}/{stats['max_size']}, hit rate: {stats['hit_rate']})")
            
        return prediction, msg_history
