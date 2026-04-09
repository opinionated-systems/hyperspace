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

from agent.llm_client import get_response_from_llm, EVAL_MODEL

logger = logging.getLogger(__name__)

# Required input fields for grading tasks
REQUIRED_INPUT_FIELDS = {"domain", "problem", "solution", "grading_guidelines", "student_answer"}

# Simple in-memory cache for LLM responses
_response_cache: dict[str, tuple[str, list[dict]]] = {}
MAX_CACHE_SIZE = 100


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
        except json.JSONDecodeError as e:
            logger.debug(f"JSON decode error: {e}, content: {inner[:100]}...")
            continue
    return results or None


def _extract_json_fallback(text: str) -> dict | None:
    """Fallback JSON extraction using regex patterns for common formats.
    
    Attempts to extract JSON from code blocks or raw JSON objects.
    """
    # Try to extract from markdown code blocks
    code_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    matches = re.findall(code_block_pattern, text, re.DOTALL)
    for match in matches:
        try:
            return json.loads(match.strip())
        except json.JSONDecodeError:
            continue
    
    # Try to find raw JSON objects
    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.findall(json_pattern, text, re.DOTALL)
    for match in matches:
        try:
            parsed = json.loads(match)
            if "response" in parsed:
                return parsed
        except json.JSONDecodeError:
            continue
    
    return None


def _validate_inputs(inputs: dict) -> tuple[bool, str]:
    """Validate that all required input fields are present.
    
    Returns:
        (is_valid, error_message)
    """
    if not isinstance(inputs, dict):
        return False, f"Expected dict, got {type(inputs).__name__}"
    
    missing = REQUIRED_INPUT_FIELDS - set(inputs.keys())
    if missing:
        return False, f"Missing required fields: {sorted(missing)}"
    
    # Check for empty values
    empty_fields = [k for k in REQUIRED_INPUT_FIELDS if not str(inputs.get(k, "")).strip()]
    if empty_fields:
        return False, f"Empty required fields: {sorted(empty_fields)}"
    
    return True, ""


def _get_cache_key(inputs: dict, model: str) -> str:
    """Generate a cache key from inputs and model."""
    # Normalize inputs for consistent hashing
    normalized = json.dumps(inputs, sort_keys=True, default=str)
    key_data = f"{model}:{normalized}"
    return hashlib.sha256(key_data.encode()).hexdigest()


def _get_cached_response(cache_key: str) -> tuple[str, list[dict]] | None:
    """Get cached response if available."""
    return _response_cache.get(cache_key)


def _cache_response(cache_key: str, prediction: str, msg_history: list[dict]) -> None:
    """Cache response with LRU eviction."""
    global _response_cache
    
    # Simple LRU: if cache is full, clear half of it
    if len(_response_cache) >= MAX_CACHE_SIZE:
        # Remove oldest half (first items in dict)
        items_to_remove = len(_response_cache) // 2
        for _ in range(items_to_remove):
            if _response_cache:
                _response_cache.pop(next(iter(_response_cache)))
        logger.info(f"Cache evicted {items_to_remove} entries")
    
    _response_cache[cache_key] = (prediction, msg_history)


class TaskAgent:
    """Task agent that solves IMO grading problems with enhanced extraction and caching."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "", use_cache: bool = True) -> None:
        self.model = model
        self.log_fn = logger.info
        self.use_cache = use_cache
        self.stats = {
            "total_calls": 0, 
            "cache_hits": 0,
            "json_extracted": 0, 
            "fallback_used": 0, 
            "failed": 0,
            "validation_errors": 0
        }

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        self.stats["total_calls"] += 1
        
        # Validate inputs
        is_valid, error_msg = _validate_inputs(inputs)
        if not is_valid:
            self.log_fn(f"Input validation failed: {error_msg}")
            self.stats["validation_errors"] += 1
            return f"Error: {error_msg}", []
        
        # Sanitize inputs for JSON serialization
        try:
            sanitized_inputs = json.loads(json.dumps(inputs, default=str))
        except (TypeError, ValueError) as e:
            self.log_fn(f"Input sanitization failed: {e}")
            self.stats["validation_errors"] += 1
            return f"Error: Invalid input data - {e}", []
        
        # Check cache if enabled
        cache_key = None
        if self.use_cache:
            cache_key = _get_cache_key(sanitized_inputs, self.model)
            cached = _get_cached_response(cache_key)
            if cached:
                self.stats["cache_hits"] += 1
                self.log_fn(f"Cache hit for request (total hits: {self.stats['cache_hits']})")
                return cached
        
        # Store cache_key for later use
        self._current_cache_key = cache_key
        
        instruction = f"""You are an expert grading agent for mathematical problem solving. Your task is to evaluate a student's answer against the correct solution and grading guidelines.

## Task Input:
```json
{json.dumps(sanitized_inputs, indent=2)}
```

## Evaluation Guidelines:

1. **Understand the Problem**: Read the problem statement carefully to understand what is being asked.

2. **Review the Solution**: Study the provided correct solution to understand the expected approach and answer.

3. **Analyze the Student's Answer**: 
   - Check if the student understood the problem correctly
   - Evaluate the mathematical reasoning and steps
   - Verify calculations and final answers
   - Look for partial credit opportunities

4. **Apply Grading Guidelines**: Use the provided grading guidelines to determine the appropriate score and feedback.

5. **Provide Constructive Feedback**: 
   - Highlight what the student did well
   - Point out specific errors or misconceptions
   - Suggest improvements for future problems

## Response Format:

Respond ONLY in JSON format with the following schema:

<json>
{{
    "response": "Your detailed evaluation here. Include: 1) Summary of the student's approach, 2) Correct aspects identified, 3) Errors or gaps found, 4) Suggested score/grade based on guidelines, 5) Constructive feedback for improvement"
}}
</json>

## Example of a Good Evaluation:

<json>
{{
    "response": "The student correctly identified the problem as requiring [specific technique]. They successfully completed steps 1-3, demonstrating good understanding of [concept]. However, they made an error in step 4 where they [describe error]. The final answer is incorrect due to this calculation mistake. Based on the grading guidelines, this would receive partial credit of X/Y points. Recommendation: Review [specific concept] and practice [specific skill] to avoid similar errors in the future."
}}
</json>"""

        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
        except Exception as e:
            self.log_fn(f"LLM call failed: {e}")
            self.stats["failed"] += 1
            return "Error: LLM call failed", []

        # Extract prediction from JSON
        prediction = "None"
        extraction_method = "none"
        
        try:
            if msg_history and len(msg_history) > 0:
                last_message = msg_history[-1].get("text", "")
                
                if not last_message or not last_message.strip():
                    self.log_fn("Empty response from LLM")
                    self.stats["failed"] += 1
                    return "Error: Empty response from LLM", msg_history
                
                # Primary extraction method
                extracted = _extract_jsons(last_message)
                if extracted:
                    # Try to find response in any of the extracted JSONs
                    for item in reversed(extracted):
                        if isinstance(item, dict) and "response" in item:
                            prediction = item["response"]
                            extraction_method = "primary"
                            self.stats["json_extracted"] += 1
                            break
                
                if extraction_method == "none":
                    # Fallback extraction
                    fallback = _extract_json_fallback(last_message)
                    if fallback and isinstance(fallback, dict) and "response" in fallback:
                        prediction = fallback["response"]
                        extraction_method = "fallback"
                        self.stats["fallback_used"] += 1
                        self.log_fn(f"Used fallback extraction for response")
                
                if extraction_method == "none":
                    # Last resort: use raw text (cleaned)
                    cleaned_text = last_message.strip()
                    # Remove common markdown artifacts
                    cleaned_text = re.sub(r'^```[\w]*\n?', '', cleaned_text)
                    cleaned_text = re.sub(r'\n?```$', '', cleaned_text)
                    prediction = cleaned_text[:1000]  # Limit length but allow more context
                    extraction_method = "raw"
                    self.log_fn(f"Using raw text extraction (limited to 1000 chars)")
                        
                self.log_fn(f"Extraction method: {extraction_method}, prediction length: {len(str(prediction))}")
                
                # Validate prediction is not empty
                if not str(prediction).strip():
                    self.log_fn("Extracted prediction is empty")
                    self.stats["failed"] += 1
                    return "Error: Empty prediction extracted", msg_history
                    
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            self.stats["failed"] += 1
            return f"Error: Extraction failed - {e}", msg_history

        # Cache the successful response
        if self.use_cache and hasattr(self, '_current_cache_key') and self._current_cache_key:
            _cache_response(self._current_cache_key, str(prediction), msg_history)
        
        return str(prediction), msg_history
    
    def get_stats(self) -> dict[str, Any]:
        """Return extraction statistics."""
        return self.stats.copy()
    
    def clear_cache(self) -> None:
        """Clear the response cache."""
        global _response_cache
        _response_cache.clear()
        self.log_fn("Response cache cleared")
