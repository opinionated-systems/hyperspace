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
from agent.config import get_config

logger = logging.getLogger(__name__)


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    Also handles markdown code blocks with json language specifier.
    Includes enhanced error recovery for malformed JSON.
    """
    results = []
    search_from = 0
    
    # First, try to find explicit <json> tags
    while True:
        start = text.find("<json>", search_from)
        if start == -1:
            break
        end = text.find("</json>", start)
        if end == -1:
            break
        inner = text[start + 6:end].strip()
        search_from = end + 7
        
        # Try multiple parsing strategies
        parsed = _try_parse_json(inner)
        if parsed is not None:
            results.append(parsed)
    
    # If no <json> tags found, look for markdown code blocks
    if not results:
        md_pattern = r'```(?:json)?\s*([\s\S]*?)```'
        matches = re.findall(md_pattern, text)
        for match in matches:
            parsed = _try_parse_json(match.strip())
            if parsed is not None:
                results.append(parsed)
    
    return results or None


def _try_parse_json(text: str) -> dict | None:
    """Try to parse JSON with multiple recovery strategies.
    
    Returns the parsed dict or None if all strategies fail.
    """
    # Strategy 1: Direct parsing
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Strategy 2: Clean common issues and retry
    cleaned = _clean_json_string(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    
    # Strategy 3: Try to extract just the first complete JSON object
    # This handles cases where there's trailing garbage
    try:
        # Find the last complete object by matching braces
        brace_count = 0
        last_valid_end = -1
        in_string = False
        escape_next = False
        
        for i, char in enumerate(text):
            if escape_next:
                escape_next = False
                continue
            if char == '\\' and in_string:
                escape_next = True
                continue
            if char == '"' and not in_string:
                in_string = True
                continue
            if char == '"' and in_string:
                in_string = False
                continue
            
            if not in_string:
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        last_valid_end = i
        
        if last_valid_end > 0:
            truncated = text[:last_valid_end + 1]
            try:
                return json.loads(truncated)
            except json.JSONDecodeError:
                cleaned_truncated = _clean_json_string(truncated)
                try:
                    return json.loads(cleaned_truncated)
                except json.JSONDecodeError:
                    pass
    except Exception:
        pass
    
    return None


def _clean_json_string(text: str) -> str:
    """Clean up common JSON formatting issues."""
    # Remove trailing commas before closing braces/brackets
    text = re.sub(r',(\s*[}\]])', r'\1', text)
    # Fix single quotes to double quotes (basic handling)
    text = re.sub(r"'([^']*?)'", r'"\1"', text)
    # Remove comments
    text = re.sub(r'//.*?\n', '\n', text)
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    return text.strip()


def _extract_any_json(text: str) -> list[dict] | None:
    """Fallback JSON extraction that looks for any JSON objects in text.
    
    Uses a robust brace-matching algorithm that handles nested structures
    and string literals correctly. Leverages _try_parse_json for parsing.
    """
    results = []
    brace_count = 0
    start_idx = -1
    in_string = False
    escape_next = False
    
    for i, char in enumerate(text):
        if escape_next:
            escape_next = False
            continue
        if char == '\\' and in_string:
            escape_next = True
            continue
        if char == '"' and not in_string:
            in_string = True
            continue
        if char == '"' and in_string:
            in_string = False
            continue
        
        if not in_string:
            if char == '{':
                if brace_count == 0:
                    start_idx = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start_idx != -1:
                    # Use the robust parsing helper
                    candidate = text[start_idx:i+1]
                    parsed = _try_parse_json(candidate)
                    if parsed is not None:
                        results.append(parsed)
                    start_idx = -1
    
    return results or None


def _extract_json_with_regex(text: str) -> list[dict] | None:
    """Additional fallback using regex to find JSON-like structures.
    
    Leverages _try_parse_json for robust parsing of extracted candidates.
    """
    results = []
    # Look for JSON blocks that might be wrapped in markdown code blocks
    json_pattern = r'```(?:json)?\s*([\s\S]*?)```'
    matches = re.findall(json_pattern, text)
    for match in matches:
        # Try parsing the full match first
        parsed = _try_parse_json(match.strip())
        if parsed is not None and isinstance(parsed, dict):
            results.append(parsed)
            continue
        
        # Try to find nested JSON objects within the match
        brace_count = 0
        start_idx = -1
        for i, char in enumerate(match):
            if char == '{':
                if brace_count == 0:
                    start_idx = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start_idx != -1:
                    candidate = match[start_idx:i+1]
                    parsed = _try_parse_json(candidate)
                    if parsed is not None:
                        results.append(parsed)
                    start_idx = -1
    return results or None


class _CacheEntry:
    """Cache entry with timestamp for TTL support."""
    
    def __init__(self, result: tuple[str, list[dict]], timestamp: float) -> None:
        self.result = result
        self.timestamp = timestamp


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning.
    
    Features:
    - TTL-based caching with configurable expiration
    - Cache size limits to prevent memory bloat
    - Configurable retry settings from AgentConfig
    - Cache statistics for monitoring
    """

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        config = get_config()
        self.model = model
        self.log_fn = logger.info
        self.max_retries = config.max_retries
        self.base_delay = config.base_delay
        self._cache: dict[str, _CacheEntry] = {}
        self._cache_enabled = True
        self._cache_ttl_seconds: float = 3600.0  # 1 hour default TTL
        self._max_cache_size: int = 1000  # Maximum number of cached entries
        self._cache_hits = 0
        self._cache_misses = 0
        self._cache_evictions = 0

    def _get_cache_key(self, inputs: dict) -> str:
        """Generate a cache key from inputs using SHA-256 for better collision resistance."""
        # Normalize inputs by sorting keys and handling nested dicts
        key_data = json.dumps(inputs, sort_keys=True, ensure_ascii=True, default=str)
        return hashlib.sha256(key_data.encode()).hexdigest()

    def _is_cache_entry_valid(self, entry: _CacheEntry) -> bool:
        """Check if a cache entry is still valid (not expired)."""
        if self._cache_ttl_seconds <= 0:
            return True  # No TTL, always valid
        return (time.time() - entry.timestamp) < self._cache_ttl_seconds

    def _evict_expired_entries(self) -> None:
        """Remove expired entries from cache."""
        if self._cache_ttl_seconds <= 0:
            return
        expired_keys = [
            key for key, entry in self._cache.items()
            if not self._is_cache_entry_valid(entry)
        ]
        for key in expired_keys:
            del self._cache[key]
            self._cache_evictions += 1

    def _enforce_cache_size_limit(self) -> None:
        """Enforce cache size limit by removing oldest entries."""
        if len(self._cache) <= self._max_cache_size:
            return
        # Sort by timestamp and remove oldest entries
        sorted_items = sorted(self._cache.items(), key=lambda x: x[1].timestamp)
        num_to_remove = len(self._cache) - self._max_cache_size
        for key, _ in sorted_items[:num_to_remove]:
            del self._cache[key]
            self._cache_evictions += 1

    def clear_cache(self) -> None:
        """Clear the response cache."""
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        self._cache_evictions = 0

    def set_cache_enabled(self, enabled: bool) -> None:
        """Enable or disable caching."""
        self._cache_enabled = enabled

    def set_cache_ttl(self, ttl_seconds: float) -> None:
        """Set cache TTL in seconds. 0 or negative disables TTL."""
        self._cache_ttl_seconds = ttl_seconds

    def set_max_cache_size(self, max_size: int) -> None:
        """Set maximum cache size."""
        self._max_cache_size = max(max_size, 1)

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0.0
        return {
            "size": len(self._cache),
            "max_size": self._max_cache_size,
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "evictions": self._cache_evictions,
            "hit_rate": hit_rate,
            "ttl_seconds": self._cache_ttl_seconds,
        }

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Check cache first
        if self._cache_enabled:
            # Clean up expired entries periodically
            self._evict_expired_entries()
            
            cache_key = self._get_cache_key(inputs)
            if cache_key in self._cache:
                entry = self._cache[cache_key]
                if self._is_cache_entry_valid(entry):
                    self._cache_hits += 1
                    self.log_fn(f"Cache hit: returning cached result (hit rate: {self.get_cache_stats()['hit_rate']:.2%})")
                    return entry.result
                else:
                    # Entry expired, remove it
                    del self._cache[cache_key]
                    self._cache_evictions += 1
            self._cache_misses += 1

        # Extract task components for better prompting
        domain = inputs.get("domain", "Unknown")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = f"""You are an expert mathematical grader for International Mathematical Olympiad (IMO) problems.

Your task is to evaluate a student's answer to a mathematics problem and provide a grade.

DOMAIN: {domain}

PROBLEM:
{problem}

OFFICIAL SOLUTION:
{solution}

GRADING GUIDELINES:
{grading_guidelines}

STUDENT'S ANSWER:
{student_answer}

Think step by step:
1. Analyze what the problem is asking and identify key concepts
2. Review the official solution approach and identify critical steps
3. Compare the student's answer to the official solution - check for:
   - Correctness of the final answer
   - Validity of the reasoning process
   - Completeness of the solution
   - Mathematical rigor and clarity
4. Check if the student followed the grading guidelines precisely
5. Determine the appropriate grade based on the guidelines

IMPORTANT: Your response must be valid JSON wrapped in <json> tags. Do not include any text outside the JSON tags.

Respond in this exact format:
<json>
{{
    "reasoning": "Your detailed chain-of-thought analysis here. Explain your evaluation process clearly.",
    "response": "The final grade/prediction (e.g., 'Correct', 'Incorrect', 'Partial', or a numeric score)"
}}
</json>"""

        # Retry loop with exponential backoff for improved reliability
        last_exception = None
        for attempt in range(self.max_retries):
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=[],
                )
                break
            except Exception as e:
                last_exception = e
                self.log_fn(f"LLM call failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    delay = self.base_delay * (2 ** attempt)
                    self.log_fn(f"Retrying in {delay:.1f}s...")
                    time.sleep(delay)
                else:
                    self.log_fn("Max retries reached, returning error prediction")
                    return "Error: LLM call failed", []

        # Extract prediction from JSON with multiple fallback mechanisms
        prediction = "None"
        try:
            last_message = msg_history[-1]["text"]
            
            # Try primary extraction method (explicit <json> tags)
            extracted = _extract_jsons(last_message)
            
            # Fallback 1: generic JSON extraction from braces
            if extracted is None:
                extracted = _extract_any_json(last_message)
            
            # Fallback 2: regex-based extraction for markdown code blocks
            if extracted is None:
                extracted = _extract_json_with_regex(last_message)
            
            if extracted:
                # Prefer response field, but accept other common field names
                last_json = extracted[-1]
                if "response" in last_json:
                    prediction = last_json["response"]
                elif "grade" in last_json:
                    prediction = last_json["grade"]
                elif "answer" in last_json:
                    prediction = last_json["answer"]
                elif "result" in last_json:
                    prediction = last_json["result"]
                elif "evaluation" in last_json:
                    prediction = last_json["evaluation"]
                elif "prediction" in last_json:
                    prediction = last_json["prediction"]
                elif "score" in last_json:
                    prediction = last_json["score"]
                else:
                    # If no known field, use the first string value found
                    for key, value in last_json.items():
                        if isinstance(value, str):
                            prediction = value
                            break
                        elif isinstance(value, (int, float)):
                            prediction = str(value)
                            break
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        result = (str(prediction), msg_history)
        
        # Cache the result with timestamp
        if self._cache_enabled:
            cache_key = self._get_cache_key(inputs)
            self._enforce_cache_size_limit()
            self._cache[cache_key] = _CacheEntry(result, time.time())
        
        return result
