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
from typing import Any

from agent.llm_client import get_response_from_llm, EVAL_MODEL

logger = logging.getLogger(__name__)


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    Also attempts to parse raw JSON objects if no <json> tags are found.
    
    Enhanced to handle nested braces, multiple JSON objects, and 
    common LLM output formats like markdown code blocks.
    """
    if not text or not isinstance(text, str):
        logger.debug("Invalid input to _extract_jsons: empty or non-string")
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
        try:
            results.append(json.loads(inner))
        except json.JSONDecodeError as e:
            logger.debug(f"Failed to parse JSON from <json> block: {e}")
            continue
    
    # Second pass: try markdown code blocks with json
    if not results:
        search_from = 0
        while True:
            # Look for ```json or ``` blocks
            start = text.find("```json", search_from)
            if start == -1:
                start = text.find("```", search_from)
                if start == -1:
                    break
                code_start = start + 3
            else:
                code_start = start + 7
            
            end = text.find("```", code_start)
            if end == -1:
                break
            
            inner = text[code_start:end].strip()
            search_from = end + 3
            try:
                parsed = json.loads(inner)
                if isinstance(parsed, dict):
                    results.append(parsed)
            except json.JSONDecodeError:
                pass
    
    # Third pass: try to find JSON objects directly if no tags found
    if not results:
        # Try to find JSON objects by looking for balanced braces
        brace_start = 0
        while brace_start < len(text):
            brace_start = text.find("{", brace_start)
            if brace_start == -1:
                break
            
            # Find matching closing brace using stack-based approach
            brace_count = 0
            brace_end = brace_start
            in_string = False
            escape_next = False
            
            for i, char in enumerate(text[brace_start:], start=brace_start):
                if escape_next:
                    escape_next = False
                    continue
                if char == "\\":
                    escape_next = True
                    continue
                if char == '"' and not escape_next:
                    in_string = not in_string
                    continue
                if not in_string:
                    if char == "{":
                        brace_count += 1
                    elif char == "}":
                        brace_count -= 1
                        if brace_count == 0:
                            brace_end = i
                            break
            
            if brace_end > brace_start:
                potential_json = text[brace_start:brace_end + 1]
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


def _format_inputs(inputs: dict) -> str:
    """Format task inputs into a structured prompt.
    
    Provides better structure and context for the LLM with clear section headers.
    """
    parts = []
    
    # Define preferred order for common fields
    preferred_order = ["domain", "problem", "solution", "grading_guidelines", "student_answer"]
    
    # Add fields in preferred order if they exist
    for key in preferred_order:
        if key in inputs:
            value = inputs[key]
            parts.append(f"=== {key.replace('_', ' ').title()} ===\n{value}\n")
    
    # Add any remaining fields not in preferred order
    for key, value in inputs.items():
        if key not in preferred_order:
            parts.append(f"=== {key.replace('_', ' ').title()} ===\n{value}\n")
    
    return "\n".join(parts)


def _validate_grading_response(response: dict) -> tuple[bool, str]:
    """Validate that the grading response has the required structure.
    
    Returns:
        (is_valid, error_message)
    """
    if not isinstance(response, dict):
        return False, f"Response is not a dictionary, got {type(response).__name__}"
    
    if "response" not in response:
        available_keys = list(response.keys())
        return False, f"Missing 'response' key in JSON. Available keys: {available_keys}"
    
    response_value = response["response"]
    if not isinstance(response_value, (str, int, float, bool)):
        type_name = type(response_value).__name__
        return False, f"'response' value has unsupported type: {type_name}. Expected str, int, float, or bool."
    
    # Additional validation: check for empty response
    if isinstance(response_value, str) and not response_value.strip():
        return False, "'response' value is empty or whitespace only"
    
    return True, ""


class TaskAgent:
    """Task agent that solves IMO grading problems with improved error handling and caching."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "", cache_size: int = 100) -> None:
        self.model = model
        self.log_fn = logger.info
        self.call_count = 0
        self.success_count = 0
        self.error_count = 0
        self._cache: dict[str, tuple[str, list[dict]]] = {}
        self._cache_size = cache_size
        self._cache_hits = 0

    def _get_cache_key(self, inputs: dict) -> str:
        """Generate a cache key from inputs."""
        # Create a deterministic key from the inputs
        key_parts = []
        for key in sorted(inputs.keys()):
            value = inputs[key]
            if isinstance(value, str):
                key_parts.append(f"{key}:{value[:200]}")
            else:
                key_parts.append(f"{key}:{str(value)[:200]}")
        return "|".join(key_parts)

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        self.call_count += 1
        
        # Check cache first
        cache_key = self._get_cache_key(inputs)
        if cache_key in self._cache:
            self._cache_hits += 1
            self.log_fn(f"TaskAgent call #{self.call_count} CACHE HIT (hit rate: {self._cache_hits}/{self.call_count})")
            return self._cache[cache_key]
        
        self.log_fn(f"TaskAgent call #{self.call_count} starting")
        
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

        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
            self.log_fn(f"LLM call completed, response length: {len(response)}")
        except Exception as e:
            self.log_fn(f"Error in LLM call: {e}")
            self.error_count += 1
            return "Error: LLM call failed", []

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

        self.log_fn(f"TaskAgent stats: {self.success_count} successes, {self.error_count} errors out of {self.call_count} calls")
        
        # Store in cache with LRU eviction
        if len(self._cache) >= self._cache_size:
            # Remove oldest entry (first one)
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        self._cache[cache_key] = (str(prediction), msg_history)
        
        return str(prediction), msg_history

    def get_stats(self) -> dict:
        """Return agent performance statistics including cache metrics."""
        return {
            "total_calls": self.call_count,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "success_rate": self.success_count / max(1, self.call_count),
            "cache_hits": self._cache_hits,
            "cache_size": len(self._cache),
            "cache_hit_rate": self._cache_hits / max(1, self.call_count),
        }
