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
from functools import lru_cache

from agent.llm_client import get_response_from_llm, EVAL_MODEL

logger = logging.getLogger(__name__)

# Simple in-memory cache for grading results to improve performance
# Key: hash of (problem + student_answer), Value: (prediction, reasoning)
_grading_cache: dict[str, tuple[str, str]] = {}
_MAX_CACHE_SIZE = 1000


def _compute_cache_key(problem: str, student_answer: str) -> str:
    """Compute a cache key from problem and student answer."""
    content = f"{problem}::{student_answer}"
    return hashlib.sha256(content.encode()).hexdigest()[:32]


def _get_cached_result(problem: str, student_answer: str) -> tuple[str, str] | None:
    """Get cached grading result if available."""
    key = _compute_cache_key(problem, student_answer)
    if key in _grading_cache:
        logger.info(f"Cache hit for problem hash {key[:8]}...")
        return _grading_cache[key]
    return None


def _set_cached_result(problem: str, student_answer: str, prediction: str, reasoning: str) -> None:
    """Cache a grading result."""
    key = _compute_cache_key(problem, student_answer)
    # Simple LRU eviction: if cache is full, clear half of it
    if len(_grading_cache) >= _MAX_CACHE_SIZE:
        # Remove oldest half of entries (simple approach)
        keys_to_remove = list(_grading_cache.keys())[:_MAX_CACHE_SIZE // 2]
        for k in keys_to_remove:
            del _grading_cache[k]
        logger.info(f"Cache evicted {len(keys_to_remove)} entries")
    _grading_cache[key] = (prediction, reasoning)
    logger.info(f"Cached result for problem hash {key[:8]}...")


def clear_cache() -> None:
    """Clear the grading cache. Useful for testing."""
    global _grading_cache
    size = len(_grading_cache)
    _grading_cache.clear()
    logger.info(f"Cleared grading cache ({size} entries)")


def get_cache_stats() -> dict:
    """Get cache statistics."""
    return {
        "size": len(_grading_cache),
        "max_size": _MAX_CACHE_SIZE,
        "utilization": len(_grading_cache) / _MAX_CACHE_SIZE,
    }


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    Also handles nested JSON objects within the content.
    
    Enhanced version with better handling of malformed JSON and edge cases.
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
            # Try to find partial JSON if closing tag is missing
            inner = text[start + 6:].strip()
            # Look for a complete JSON object even without closing tag
            try:
                json_obj = _extract_json_with_brace_matching(inner)
                if json_obj and isinstance(json_obj, dict):
                    results.append(json_obj)
            except Exception:
                pass
            break
        
        inner = text[start + 6:end].strip()
        search_from = end + 7
        
        # Try to parse the inner content as JSON
        try:
            parsed = json.loads(inner)
            if isinstance(parsed, dict):
                results.append(parsed)
        except json.JSONDecodeError:
            # If direct parsing fails, try to extract JSON using brace matching
            try:
                json_obj = _extract_json_with_brace_matching(inner)
                if json_obj and isinstance(json_obj, dict):
                    results.append(json_obj)
            except Exception:
                # Try one more fallback: look for JSON-like structures
                try:
                    # Handle cases where quotes might be escaped differently
                    cleaned = inner.replace('\\"', '"').replace("\\'", "'")
                    parsed = json.loads(cleaned)
                    if isinstance(parsed, dict):
                        results.append(parsed)
                except Exception:
                    continue
    
    # Also try to find JSON objects not wrapped in <json> tags
    if not results:
        try:
            json_obj = _extract_json_with_brace_matching(text)
            if json_obj and isinstance(json_obj, dict) and "response" in json_obj:
                results.append(json_obj)
        except Exception:
            pass
    
    return results or None


def _extract_json_with_brace_matching(text: str) -> dict | None:
    """Extract a JSON object from text using brace counting.
    
    This handles nested JSON objects properly by tracking brace depth.
    Enhanced version with better handling of strings and edge cases.
    """
    start_idx = text.find('{')
    if start_idx == -1:
        return None
    
    brace_count = 1
    i = start_idx + 1
    in_string = False
    escape_next = False
    
    while i < len(text) and brace_count > 0:
        char = text[i]
        
        if escape_next:
            escape_next = False
        elif char == '\\':
            escape_next = True
        elif char == '"' and not in_string:
            in_string = True
        elif char == '"' and in_string:
            in_string = False
        elif not in_string:
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
        
        i += 1
    
    if brace_count == 0:
        candidate = text[start_idx:i]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            # Try cleaning up common issues
            try:
                # Remove trailing commas before closing braces
                cleaned = re.sub(r',(\s*[}\]])', r'\1', candidate)
                return json.loads(cleaned)
            except json.JSONDecodeError:
                pass
    return None


def _extract_json_with_regex(text: str) -> list[dict] | None:
    """Fallback JSON extraction using regex for malformed outputs.
    
    Handles nested JSON objects by using a stack-based brace matching approach
    instead of simple regex that fails on nested structures.
    """
    results = []
    # Try to find JSON objects in code blocks
    code_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    matches = re.findall(code_block_pattern, text, re.DOTALL)
    for match in matches:
        try:
            results.append(json.loads(match.strip()))
        except json.JSONDecodeError:
            continue
    
    # Try to find JSON objects with curly braces using stack-based matching
    # This handles nested objects properly
    if not results:
        i = 0
        while i < len(text):
            # Find the start of a potential JSON object
            if text[i] == '{':
                start = i
                brace_count = 1
                i += 1
                # Track braces to find the matching closing brace
                while i < len(text) and brace_count > 0:
                    if text[i] == '{':
                        brace_count += 1
                    elif text[i] == '}':
                        brace_count -= 1
                    i += 1
                # If we found a complete object with "response" key, try to parse it
                if brace_count == 0:
                    candidate = text[start:i]
                    if '"response"' in candidate:
                        try:
                            results.append(json.loads(candidate))
                        except json.JSONDecodeError:
                            continue
            else:
                i += 1
    
    return results or None


def _extract_response_value(text: str) -> str | None:
    """Last-resort extraction: try to find a response value directly.
    
    This handles cases where the JSON is malformed but we can still
    extract the value after "response": using regex patterns.
    
    Enhanced version with better logging and support for additional formats.
    """
    # Pattern to match "response": followed by a value (number, string, or boolean)
    # Handles: "response": 7, "response": "7", "response": true, etc.
    patterns = [
        # Number (integer or float) - with optional quotes around the key
        (r'["\']?response["\']?\s*:\s*(-?\d+(?:\.\d+)?)', 'number'),
        # String in double quotes
        (r'["\']?response["\']?\s*:\s*"([^"]*)"', 'string_double'),
        # String in single quotes
        (r"['\"]?response['\"]?\s*:\s*'([^']*)'", 'string_single'),
        # Boolean or null
        (r'["\']?response["\']?\s*:\s*(true|false|null)', 'boolean'),
    ]
    
    logger.debug(f"Attempting direct response extraction from text of length {len(text)}")
    
    for pattern, pattern_type in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = match.group(1)
            logger.debug(f"Matched pattern '{pattern_type}': {value}")
            
            # Try to convert to appropriate type
            try:
                # Try integer first
                result = str(int(value))
                logger.debug(f"Converted to integer: {result}")
                return result
            except ValueError:
                try:
                    # Try float
                    result = str(float(value))
                    logger.debug(f"Converted to float: {result}")
                    return result
                except ValueError:
                    # Return as string
                    logger.debug(f"Returning as string: {value}")
                    return value
    
    logger.debug("No response value found with direct extraction")
    return None


def _normalize_prediction(prediction: str | int | float | bool | None) -> str:
    """Normalize prediction to a standardized string format.
    
    This ensures consistent output format for grading results,
    handling various numeric and string representations.
    
    Args:
        prediction: The raw prediction value (could be number, string, bool, or None)
        
    Returns:
        A normalized string representation of the prediction
    """
    if prediction is None:
        return "None"
    
    # Handle boolean values
    if isinstance(prediction, bool):
        return "true" if prediction else "false"
    
    # Handle numeric values - ensure consistent formatting
    if isinstance(prediction, (int, float)):
        # If it's a whole number, return as int string
        if isinstance(prediction, float) and prediction.is_integer():
            return str(int(prediction))
        return str(prediction)
    
    # Handle string values - strip whitespace and normalize
    if isinstance(prediction, str):
        cleaned = prediction.strip()
        # Try to normalize numeric strings
        try:
            num_val = float(cleaned)
            if num_val.is_integer():
                return str(int(num_val))
            return str(num_val)
        except ValueError:
            # Not a number, return cleaned string
            return cleaned.lower() if cleaned.lower() in ("true", "false", "null", "none") else cleaned
    
    # Default: convert to string
    return str(prediction)


class TaskAgent:
    """Task agent that solves IMO grading problems."""

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
        # Extract key fields for better prompt construction
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        # Check cache first to avoid redundant LLM calls
        cached = _get_cached_result(problem, student_answer)
        if cached is not None:
            prediction, reasoning = cached
            self.log_fn(f"Using cached result: {prediction}")
            # Return a minimal msg_history indicating cache hit
            return prediction, [{"role": "system", "text": f"Cache hit: {reasoning}"}]

        instruction = f"""You are an expert mathematical grader evaluating student solutions to competition mathematics problems.

Your task is to grade a student's answer to an IMO-level mathematics problem. You must carefully analyze:
1. The problem statement
2. The official solution
3. The grading guidelines (rubric)
4. The student's submitted answer

Then provide your evaluation in the specified JSON format.

PROBLEM DOMAIN: {domain}

PROBLEM STATEMENT:
```
{problem}
```

OFFICIAL SOLUTION:
```
{solution}
```

GRADING GUIDELINES (RUBRIC):
```
{grading_guidelines}
```

STUDENT'S ANSWER:
```
{student_answer}
```

INSTRUCTIONS:
1. Read the problem carefully and understand what is being asked.
2. Study the official solution to understand the correct approach.
3. Review the grading guidelines to understand how points are awarded.
4. Analyze the student's answer step by step:
   - Did they understand the problem correctly?
   - Did they use the right approach?
   - Are their calculations correct?
   - Did they provide a complete proof/solution?
   - Where did they make errors, if any?
5. Assign a score based on the grading guidelines.

CHAIN-OF-THOUGHT REASONING (think step by step before answering):
Before providing your final JSON response, work through your reasoning:
- Identify the key concepts and techniques required for the solution
- Compare the student's approach to the official solution
- Note any partial credit the student may deserve
- Consider edge cases or alternative valid approaches
- Determine the final score based on the rubric

IMPORTANT: Your response MUST be a valid JSON object wrapped in <json> tags. Do not include any text outside the JSON tags.

Respond in the following exact format:
<json>
{{
    "response": <your numerical score or evaluation>,
    "reasoning": "<brief explanation of your grading decision>"
}}
</json>

The "response" field should contain your final grading decision (typically a number or specific evaluation string as specified in the grading guidelines).
The "reasoning" field should contain a brief explanation of how you arrived at your decision.

Examples of valid responses:
- For a numeric score: <json>{{"response": 7, "reasoning": "Complete solution with correct proof"}}</json>
- For a string evaluation: <json>{{"response": "Correct", "reasoning": "All steps verified"}}</json>
- For a boolean: <json>{{"response": true, "reasoning": "Valid counterexample provided"}}</json>

Ensure your JSON is properly formatted with no trailing commas or syntax errors."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON with multiple fallback strategies
        prediction = "None"
        raw_text = msg_history[-1]["text"]
        
        # Log the raw response for debugging
        self.log_fn(f"Raw LLM response (first 1000 chars): {raw_text[:1000]}")
        
        extraction_attempts = [
            ("_extract_jsons", lambda: _extract_jsons(raw_text)),
            ("_extract_json_with_regex", lambda: _extract_json_with_regex(raw_text)),
        ]
        
        for attempt_name, attempt_fn in extraction_attempts:
            try:
                extracted = attempt_fn()
                if extracted and len(extracted) > 0:
                    # Check the last extracted JSON object for "response" key
                    last_obj = extracted[-1]
                    if isinstance(last_obj, dict) and "response" in last_obj:
                        prediction = last_obj["response"]
                        self.log_fn(f"Successfully extracted prediction using {attempt_name}: {prediction}")
                        break
            except Exception as e:
                self.log_fn(f"Extraction attempt {attempt_name} failed: {e}")
                continue
        
        # Last resort: try to extract response value directly from malformed JSON
        if prediction == "None":
            direct_value = _extract_response_value(raw_text)
            if direct_value is not None:
                prediction = direct_value
                self.log_fn(f"Extracted prediction using direct value extraction: {prediction}")
        
        if prediction == "None":
            self.log_fn(f"Failed to extract prediction from response. Raw text: {raw_text[:500]}")

        # Normalize the prediction to ensure consistent output format
        normalized_prediction = _normalize_prediction(prediction)
        self.log_fn(f"Final normalized prediction: {normalized_prediction}")
        
        # Cache the result for future use
        # Extract reasoning from the last extraction attempt if available
        reasoning = "No reasoning extracted"
        for attempt_name, attempt_fn in extraction_attempts:
            try:
                extracted = attempt_fn()
                if extracted and len(extracted) > 0:
                    last_obj = extracted[-1]
                    if isinstance(last_obj, dict) and "reasoning" in last_obj:
                        reasoning = last_obj["reasoning"]
                        break
            except Exception:
                continue
        _set_cached_result(problem, student_answer, normalized_prediction, reasoning)
        
        return normalized_prediction, msg_history
