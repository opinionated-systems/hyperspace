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
import threading
import time
from functools import lru_cache

from agent.llm_client import get_response_from_llm, EVAL_MODEL

logger = logging.getLogger(__name__)

# Simple in-memory cache for LLM responses to avoid redundant calls
_response_cache: dict[str, tuple[str, list[dict]]] = {}
_MAX_CACHE_SIZE = 100
_cache_access_times: dict[str, float] = {}
_cache_lock = threading.Lock()


def _get_cache_key(inputs: dict, model: str) -> str:
    """Generate a cache key from inputs and model.
    
    Creates a deterministic, collision-resistant key that handles
    various input types safely.
    """
    try:
        # Create a deterministic key from the inputs
        key_data = json.dumps(inputs, sort_keys=True, ensure_ascii=True) + model
        return hashlib.sha256(key_data.encode('utf-8')).hexdigest()[:32]
    except (TypeError, ValueError) as e:
        # Fallback for non-serializable inputs
        logger.warning(f"Failed to serialize inputs for cache key: {e}")
        fallback_key = str(hash(str(sorted(inputs.keys())))) + model
        return hashlib.sha256(fallback_key.encode('utf-8')).hexdigest()[:32]


def _get_cached_response(cache_key: str) -> tuple[str, list[dict]] | None:
    """Get cached response if available.
    
    Thread-safe and updates access time for LRU tracking.
    """
    with _cache_lock:
        if cache_key in _response_cache:
            _cache_access_times[cache_key] = time.time()
            return _response_cache[cache_key]
        return None


def _set_cached_response(cache_key: str, result: tuple[str, list[dict]]) -> None:
    """Cache a response, maintaining size limit with LRU eviction.
    
    Thread-safe implementation that evicts least recently used entries
    when cache reaches capacity.
    """
    with _cache_lock:
        # Evict oldest entries if at capacity
        while len(_response_cache) >= _MAX_CACHE_SIZE:
            if _cache_access_times:
                # Find and remove least recently used entry
                lru_key = min(_cache_access_times, key=_cache_access_times.get)
                _response_cache.pop(lru_key, None)
                _cache_access_times.pop(lru_key, None)
            else:
                # Fallback: remove arbitrary entry
                if _response_cache:
                    first_key = next(iter(_response_cache))
                    _response_cache.pop(first_key, None)
                break
        
        _response_cache[cache_key] = result
        _cache_access_times[cache_key] = time.time()


def get_cache_stats() -> dict:
    """Get cache statistics for monitoring.
    
    Returns:
        Dictionary with cache size, capacity, and other metrics.
    """
    with _cache_lock:
        return {
            "size": len(_response_cache),
            "capacity": _MAX_CACHE_SIZE,
            "utilization": len(_response_cache) / _MAX_CACHE_SIZE,
            "has_access_times": len(_cache_access_times) > 0,
        }


def clear_cache() -> None:
    """Clear all cached responses."""
    with _cache_lock:
        _response_cache.clear()
        _cache_access_times.clear()
        logger.info("Response cache cleared")


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    Handles nested JSON and malformed content gracefully.
    """
    if not text or not isinstance(text, str):
        return None
        
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
        except json.JSONDecodeError:
            # Try to extract partial JSON if possible
            try:
                # Handle cases where JSON might have trailing content
                # Find the last valid JSON object by trying progressively shorter strings
                for end_idx in range(len(inner), 0, -1):
                    try:
                        partial = json.loads(inner[:end_idx])
                        if isinstance(partial, dict):
                            results.append(partial)
                            break
                    except json.JSONDecodeError:
                        continue
            except Exception:
                pass
            continue
            
    return results if results else None


def _extract_json_from_markdown(text: str) -> list[dict] | None:
    """Extract JSON objects from markdown code blocks.

    Fallback for when <json> tags are not used but markdown code blocks are.
    Handles various code block formats and nested content.
    """
    if not text or not isinstance(text, str):
        return None
        
    results = []
    # Match ```json ... ``` or just ``` ... ``` blocks
    # Use non-greedy matching but with a limit to prevent catastrophic backtracking
    pattern = r'```(?:json)?\s*([\s\S]{0,100000}?)```'
    try:
        matches = re.findall(pattern, text, re.DOTALL)
    except re.error:
        # Fallback if regex fails
        matches = []
        
    for match in matches:
        content = match.strip()
        if not content:
            continue
            
        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                results.append(parsed)
        except json.JSONDecodeError:
            # Try to find JSON object within the content
            # Look for content between first { and last }
            try:
                start = content.find('{')
                end = content.rfind('}')
                if start != -1 and end != -1 and end > start:
                    inner_json = content[start:end+1]
                    parsed = json.loads(inner_json)
                    if isinstance(parsed, dict):
                        results.append(parsed)
            except (json.JSONDecodeError, ValueError):
                pass
            continue
            
    return results if results else None


def _normalize_grade(grade: str) -> str:
    """Normalize grade to a standard format.

    Handles various grade formats and normalizes them.
    Enhanced to handle numeric grades, boolean values, and more variations.
    """
    # Handle None or non-string inputs
    if grade is None:
        return 'None'
    
    if not isinstance(grade, str):
        # Handle boolean values
        if isinstance(grade, bool):
            return 'Correct' if grade else 'Incorrect'
        # Handle numeric types
        if isinstance(grade, (int, float)):
            if grade == 0:
                return 'Incorrect'
            elif grade >= 1:
                return 'Correct'
            elif 0 < grade < 1:
                return 'Partial'
        grade = str(grade)
    
    grade = grade.strip().lower()
    
    # Handle empty string
    if not grade:
        return 'None'
    
    # First, check for numeric grades (0, 1, 2, etc.)
    # These are common in IMO-style grading
    try:
        numeric_grade = float(grade)
        if numeric_grade == 0:
            return 'Incorrect'
        elif numeric_grade >= 1:
            return 'Correct'
        elif 0 < numeric_grade < 1:
            return 'Partial'
    except ValueError:
        pass
    
    # Map common variations to standard formats
    correct_variations = [
        'correct', 'right', 'true', 'yes', 'full', 'full credit', 
        'full marks', 'complete', 'valid', 'accepted', 'pass', 'success',
        'solved', 'accurate', 'perfect', 'excellent'
    ]
    incorrect_variations = [
        'incorrect', 'wrong', 'false', 'no', 'none', 'zero',
        'invalid', 'rejected', 'fail', 'error', 'mistake',
        'unsolved', 'unsuccessful', 'failed'
    ]
    partial_variations = [
        'partial', 'partial credit', 'half', 'incomplete',
        'partially correct', 'partially right', 'some credit',
        'partially solved', 'almost', 'minor errors'
    ]
    
    # Check for exact matches first, then substring matches
    if grade in correct_variations:
        return 'Correct'
    if grade in incorrect_variations:
        return 'Incorrect'
    if grade in partial_variations:
        return 'Partial'
    
    # Substring matching for more flexibility
    if any(v in grade for v in correct_variations):
        return 'Correct'
    elif any(v in grade for v in incorrect_variations):
        return 'Incorrect'
    elif any(v in grade for v in partial_variations):
        return 'Partial'
    
    # Return original if no normalization applied
    return grade.strip()


class TaskAgent:
    """Task agent that solves IMO grading problems with enhanced reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "", use_cache: bool = True) -> None:
        self.model = model
        self.log_fn = logger.info
        self.use_cache = use_cache

    def _validate_inputs(self, inputs: dict) -> tuple[bool, str]:
        """Validate input dictionary has required fields.
        
        Returns:
            (is_valid, error_message)
        """
        # First check if inputs is a dict
        if not isinstance(inputs, dict):
            return False, f"inputs must be a dictionary, got {type(inputs).__name__}"
        
        required_fields = ["domain", "problem", "solution", "grading_guidelines", "student_answer"]
        
        for field in required_fields:
            if field not in inputs:
                return False, f"Missing required field: {field}"
            if not isinstance(inputs[field], str):
                return False, f"Field {field} must be a string, got {type(inputs[field]).__name__}"
        
        # Check for empty strings that might cause issues
        empty_fields = [f for f in required_fields if not inputs[f].strip()]
        if empty_fields:
            self.log_fn(f"Warning: Empty fields detected: {empty_fields}")
        
        # Validate field lengths to prevent issues with very long inputs
        max_field_length = 100000  # 100KB limit per field
        for field in required_fields:
            if len(inputs[field]) > max_field_length:
                return False, f"Field {field} exceeds maximum length of {max_field_length} characters"
        
        return True, ""

    def _build_grading_prompt(self, inputs: dict) -> str:
        """Build a structured grading prompt with clear evaluation criteria."""
        domain = inputs.get("domain", "")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        return f"""You are an expert mathematical grader specializing in {domain} problems.

Your task is to evaluate a student's answer to a mathematics problem and assign a grade.

## Problem Statement:
{problem}

## Official Solution:
{solution}

## Grading Guidelines:
{grading_guidelines}

## Student's Answer:
{student_answer}

## Evaluation Framework:

Follow this structured evaluation process:

### Step 1: Problem Understanding
- What is the problem asking for?
- What are the key constraints and conditions?
- What is the expected answer format?

### Step 2: Solution Analysis
- What is the correct approach according to the official solution?
- What are the critical steps that must be present?
- What constitutes a complete vs. incomplete solution?

### Step 3: Student Answer Evaluation
- Did the student understand the problem correctly?
- What approach did the student take?
- Are the student's steps logically valid?
- Did the student show sufficient work and reasoning?
- Is the final answer mathematically correct?

### Step 4: Grade Assignment
Based on the grading guidelines, assign the appropriate grade considering:
- Correctness of the final answer
- Validity of the reasoning process
- Completeness of the solution
- Adherence to the expected solution method

## Response Format:

You MUST respond in JSON format wrapped in <json> tags:

<json>
{{
    "reasoning": "Your detailed step-by-step analysis following the Evaluation Framework above",
    "response": "The final grade you assign (e.g., '0', '1', '2', 'Correct', 'Incorrect', 'Partial')"
}}
</json>

Important: The "response" field must contain ONLY the grade value, nothing else."""

    def _extract_prediction(self, msg_history: list[dict]) -> str:
        """Extract prediction from message history with multiple fallback strategies."""
        if not msg_history:
            return "None"
        
        last_message = msg_history[-1].get("text", "")
        
        # Strategy 1: Extract from <json> tags
        extracted = _extract_jsons(last_message)
        if extracted:
            return self._get_grade_from_json(extracted[-1])
        
        # Strategy 2: Extract from markdown code blocks
        extracted = _extract_json_from_markdown(last_message)
        if extracted:
            return self._get_grade_from_json(extracted[-1])
        
        # Strategy 3: Look for grade patterns in plain text
        return self._extract_grade_from_text(last_message)

    def _get_grade_from_json(self, json_obj: dict) -> str:
        """Extract grade from JSON object with field priority."""
        # Priority order for grade fields
        priority_fields = ["response", "grade", "answer", "result", "score", "evaluation"]
        
        for field in priority_fields:
            if field in json_obj:
                value = json_obj[field]
                if isinstance(value, str):
                    return _normalize_grade(value)
                elif isinstance(value, (int, float)):
                    return str(value)
        
        # If no recognized field, use the first string value found
        for key, value in json_obj.items():
            if isinstance(value, str):
                return _normalize_grade(value)
            elif isinstance(value, (int, float)):
                return str(value)
        
        return "None"

    def _extract_grade_from_text(self, text: str) -> str:
        """Extract grade from plain text using pattern matching."""
        # Look for explicit grade statements
        patterns = [
            r'grade[\s]*[:=][\s]*["\']?([^"\'\n]+)["\']?',
            r'response[\s]*[:=][\s]*["\']?([^"\'\n]+)["\']?',
            r'final grade[\s]*[:=][\s]*["\']?([^"\'\n]+)["\']?',
            r'assign[\s]+["\']?([^"\'\n]+)["\']?',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return _normalize_grade(match.group(1).strip())
        
        return "None"

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with enhanced error handling.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        start_time = time.time()
        
        # Validate inputs first
        is_valid, error_msg = self._validate_inputs(inputs)
        if not is_valid:
            self.log_fn(f"Input validation failed: {error_msg}")
            return f"Error: {error_msg}", []

        # Check cache if enabled
        cache_key = _get_cache_key(inputs, self.model)
        if self.use_cache:
            cached = _get_cached_response(cache_key)
            if cached is not None:
                self.log_fn(f"Using cached response (cache hit for key: {cache_key[:8]}...)")
                return cached

        instruction = self._build_grading_prompt(inputs)
        
        # Log prompt size for debugging
        prompt_size = len(instruction)
        if prompt_size > 50000:
            self.log_fn(f"Warning: Large prompt size: {prompt_size} characters")

        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
        except Exception as e:
            self.log_fn(f"LLM call failed: {e}")
            logger.exception("LLM call failed in TaskAgent.forward")
            return "None", []

        # Extract prediction using multiple strategies
        prediction = self._extract_prediction(msg_history)
        
        if prediction == "None":
            self.log_fn(f"Failed to extract prediction from response: {response[:200] if response else 'empty'}")

        result = (str(prediction), msg_history)
        
        # Cache the result if caching is enabled
        if self.use_cache:
            _set_cached_response(cache_key, result)
        
        duration = time.time() - start_time
        self.log_fn(f"TaskAgent completed in {duration:.2f}s with prediction: {prediction}")
        
        return result
