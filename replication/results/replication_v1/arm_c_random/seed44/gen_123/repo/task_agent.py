"""
Task agent: solves a given task with a single LLM call.

Reimplemented from facebookresearch/HyperAgents task_agent.py.
Same interface, same JSON output format, same extraction logic.

This is the INITIAL task agent. The meta agent modifies this file
during self-improvement. The evaluation harness loads whatever
task_agent.py exists at the agent's repo path.
"""

from __future__ import annotations

import json
import logging
import re
import hashlib
from typing import Any
from functools import lru_cache

from agent.llm_client import get_response_from_llm, EVAL_MODEL

logger = logging.getLogger(__name__)

# Simple in-memory cache for LLM responses to avoid redundant calls
_response_cache: dict[str, tuple[str, list[dict]]] = {}


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    Also attempts to parse raw JSON objects if no <json> tags are found.
    
    Enhanced to handle nested braces, multiple JSON objects, and common
    formatting issues like trailing commas and comments.
    """
    if not text or not isinstance(text, str):
        return None
        
    results = []
    search_from = 0
    
    # First pass: extract from <json> tags
    while True:
        start = text.find("<json>", search_from)
        if start == -1:
            break
        end = text.find("</json>", start)
        if end == -1:
            break
        inner = text[start + 6:end].strip()
        search_from = end + 7
        
        # Try to clean up common JSON issues before parsing
        cleaned = _clean_json_string(inner)
        
        try:
            results.append(json.loads(cleaned))
        except json.JSONDecodeError as e:
            # Try original if cleaning failed
            try:
                results.append(json.loads(inner))
            except json.JSONDecodeError:
                logger.debug(f"Failed to parse JSON from <json> block: {e}")
                continue
    
    # Second pass: try to find JSON objects directly if no <json> tags found
    if not results:
        # Try to find JSON objects by looking for balanced braces
        brace_start = 0
        while brace_start < len(text):
            brace_start = text.find("{", brace_start)
            if brace_start == -1:
                break
            
            # Find matching closing brace
            brace_count = 0
            brace_end = brace_start
            for i, char in enumerate(text[brace_start:], start=brace_start):
                if char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        brace_end = i
                        break
            
            if brace_end > brace_start:
                potential_json = text[brace_start:brace_end + 1]
                # Clean up common JSON issues
                cleaned = _clean_json_string(potential_json)
                try:
                    parsed = json.loads(cleaned)
                    if isinstance(parsed, dict):
                        results.append(parsed)
                except json.JSONDecodeError:
                    # Try original if cleaning failed
                    try:
                        parsed = json.loads(potential_json)
                        if isinstance(parsed, dict):
                            results.append(parsed)
                    except json.JSONDecodeError:
                        pass
                brace_start = brace_end + 1
            else:
                break
    
    return results if results else None


def _clean_json_string(json_str: str) -> str:
    """Clean up common JSON formatting issues.
    
    Handles:
    - Trailing commas before closing braces/brackets
    - Single quotes instead of double quotes
    - Python-style True/False/None instead of true/false/null
    - Control characters and invalid escape sequences
    - Unicode escape sequences
    """
    import re
    
    if not json_str or not isinstance(json_str, str):
        return json_str
    
    # Remove control characters except tab, newline, carriage return
    cleaned = ''.join(char for char in json_str if ord(char) >= 32 or char in '\t\n\r')
    
    # Remove trailing commas before } or ]
    cleaned = re.sub(r',(\s*[}\]])', r'\1', cleaned)
    
    # Replace single quotes with double quotes (simple cases)
    # This is a basic fix - complex nested quotes may still fail
    cleaned = re.sub(r"'([^']*?)':", r'"\1":', cleaned)
    cleaned = re.sub(r":\s*'([^']*?)'([,}\]])", r': "\1"\2', cleaned)
    
    # Replace Python booleans/None with JSON equivalents
    cleaned = re.sub(r'\bTrue\b', 'true', cleaned)
    cleaned = re.sub(r'\bFalse\b', 'false', cleaned)
    cleaned = re.sub(r'\bNone\b', 'null', cleaned)
    
    # Fix common escape sequence issues
    cleaned = cleaned.replace('\\"', '"')  # Fix double-escaped quotes
    
    return cleaned


def _format_inputs(inputs: dict) -> str:
    """Format task inputs into a structured prompt.
    
    Provides better structure and context for the LLM.
    Orders fields logically for grading tasks.
    """
    # Define preferred order for grading task fields
    preferred_order = [
        "domain",
        "problem", 
        "solution",
        "grading_guidelines",
        "student_answer"
    ]
    
    parts = []
    
    # First add fields in preferred order
    for key in preferred_order:
        if key in inputs:
            value = inputs[key]
            parts.append(f"=== {key.upper().replace('_', ' ')} ===\n{value}\n")
    
    # Then add any remaining fields
    for key, value in inputs.items():
        if key not in preferred_order:
            parts.append(f"=== {key.upper().replace('_', ' ')} ===\n{value}\n")
    
    return "\n".join(parts)


def _validate_grading_response(response: dict) -> tuple[bool, str]:
    """Validate that the grading response has the required structure.
    
    Returns:
        (is_valid, error_message)
    """
    if not isinstance(response, dict):
        return False, "Response is not a dictionary"
    
    if "response" not in response:
        return False, "Missing 'response' key in JSON"
    
    response_value = response["response"]
    if not isinstance(response_value, (str, int, float, bool)):
        return False, f"'response' value has unsupported type: {type(response_value)}"
    
    return True, ""


class TaskAgent:
    """Task agent that solves IMO grading problems with improved error handling,
    response caching, and retry logic."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "", use_cache: bool = True) -> None:
        self.model = model
        self.log_fn = logger.info
        self.call_count = 0
        self.success_count = 0
        self.error_count = 0
        self.cache_hits = 0
        self.use_cache = use_cache

    def _get_cache_key(self, inputs: dict) -> str:
        """Generate a cache key from inputs."""
        # Create a deterministic hash of the inputs
        input_str = json.dumps(inputs, sort_keys=True, default=str)
        return hashlib.md5(input_str.encode()).hexdigest()

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        self.call_count += 1
        self.log_fn(f"TaskAgent call #{self.call_count} starting")
        
        # Check cache first if enabled
        if self.use_cache:
            cache_key = self._get_cache_key(inputs)
            if cache_key in _response_cache:
                self.cache_hits += 1
                self.log_fn(f"Cache hit! Returning cached response (hit #{self.cache_hits})")
                cached_prediction, cached_history = _response_cache[cache_key]
                self.success_count += 1
                return cached_prediction, cached_history
        
        # Format inputs more clearly
        formatted_inputs = _format_inputs(inputs)
        
        instruction = f"""You are an expert grading agent. Your task is to evaluate student answers based on the provided problem, solution, and grading guidelines.

{formatted_inputs}

Respond in JSON format with the following schema:
<json>
{{
    "response": "Your evaluation result here"
}}
</json>

Ensure your response is valid JSON and follows the schema exactly.

Think step by step:
1. First, understand the problem and what is being asked
2. Review the provided solution to understand the correct approach
3. Examine the grading guidelines carefully
4. Evaluate the student's answer against the solution and guidelines
5. Formulate your final evaluation in the required JSON format"""

        # Retry loop for LLM calls
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=[],
                )
                self.log_fn(f"LLM call completed, response length: {len(response)}")
                break
            except Exception as e:
                self.log_fn(f"Error in LLM call (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    self.error_count += 1
                    return "Error: LLM call failed after retries", []
                import time
                time.sleep(2 ** attempt)  # Exponential backoff

        # Extract prediction from JSON with better error handling
        prediction = "None"
        try:
            if msg_history and len(msg_history) > 0:
                last_message = msg_history[-1]
                text_content = last_message.get("text", "")
                extracted = _extract_jsons(text_content)
                if extracted:
                    last_extracted = extracted[-1]
                    is_valid, error_msg = _validate_grading_response(last_extracted)
                    if is_valid and isinstance(last_extracted, dict) and "response" in last_extracted:
                        prediction = last_extracted["response"]
                        self.success_count += 1
                        self.log_fn(f"Successfully extracted prediction: {str(prediction)[:100]}")
                        
                        # Cache the successful response
                        if self.use_cache:
                            cache_key = self._get_cache_key(inputs)
                            _response_cache[cache_key] = (str(prediction), msg_history)
                    else:
                        self.error_count += 1
                        self.log_fn(f"Invalid grading response: {error_msg}")
                else:
                    self.error_count += 1
                    self.log_fn("No JSON found in response")
            else:
                self.error_count += 1
                self.log_fn("Empty message history")
        except Exception as e:
            self.error_count += 1
            self.log_fn(f"Error extracting prediction: {e}")

        self.log_fn(f"TaskAgent stats: {self.success_count} successes, {self.error_count} errors, {self.cache_hits} cache hits out of {self.call_count} calls")
        return str(prediction), msg_history

    def get_stats(self) -> dict:
        """Return agent performance statistics."""
        total = max(1, self.call_count)
        return {
            "total_calls": self.call_count,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "cache_hits": self.cache_hits,
            "success_rate": self.success_count / total,
            "cache_hit_rate": self.cache_hits / total,
        }
    
    def clear_cache(self) -> None:
        """Clear the response cache."""
        global _response_cache
        _response_cache.clear()
        self.log_fn("Response cache cleared")
