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
    Includes improved handling for nested braces and malformed JSON.
    
    Args:
        text: The text to extract JSON from.
        
    Returns:
        A list of parsed JSON objects, or None if no valid JSON found.
    """
    results = []
    search_from = 0
    extraction_errors = []
    
    while True:
        start = text.find("<json>", search_from)
        if start == -1:
            break
        end = text.find("</json>", start)
        if end == -1:
            extraction_errors.append(f"Unclosed <json> tag at position {start}")
            break
        inner = text[start + 6:end].strip()
        search_from = end + 7
        try:
            results.append(json.loads(inner))
        except json.JSONDecodeError as e:
            extraction_errors.append(f"JSON parse error at position {start}: {e}")
            continue
    
    # Fallback 1: Try to extract JSON from markdown code blocks
    if not results:
        json_code_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
        matches = re.findall(json_code_block_pattern, text, re.DOTALL)
        for match in matches:
            try:
                results.append(json.loads(match.strip()))
            except json.JSONDecodeError as e:
                extraction_errors.append(f"Code block JSON parse error: {e}")
    
    # Fallback 2: Try to find JSON objects directly if no <json> tags or code blocks
    if not results:
        try:
            # Look for JSON-like structures with braces
            brace_start = text.find("{")
            brace_end = text.rfind("}")
            if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
                potential_json = text[brace_start:brace_end + 1]
                results.append(json.loads(potential_json))
        except json.JSONDecodeError as e:
            extraction_errors.append(f"Fallback JSON parse error: {e}")
    
    # Fallback 3: Try to find multiple JSON objects using brace counting
    if not results:
        results = _extract_multiple_jsons(text, extraction_errors)
    
    # Log extraction details for debugging
    if extraction_errors and logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"JSON extraction errors: {extraction_errors}")
    
    return results or None


def _extract_multiple_jsons(text: str, extraction_errors: list) -> list[dict]:
    """Extract multiple JSON objects from text using brace counting.
    
    This handles cases where the LLM outputs multiple JSON objects
    concatenated together without proper separation.
    
    Args:
        text: The text to extract JSON from.
        extraction_errors: List to append extraction errors to.
        
    Returns:
        A list of parsed JSON objects.
    """
    results = []
    i = 0
    n = len(text)
    
    while i < n:
        # Find the start of a potential JSON object
        if text[i] == '{':
            # Use brace counting to find the matching closing brace
            brace_count = 1
            j = i + 1
            in_string = False
            escape_next = False
            
            while j < n and brace_count > 0:
                char = text[j]
                
                if escape_next:
                    escape_next = False
                elif char == '\\':
                    escape_next = True
                elif char == '"' and not escape_next:
                    in_string = not in_string
                elif not in_string:
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                
                j += 1
            
            if brace_count == 0:
                # Found a complete JSON object
                potential_json = text[i:j]
                try:
                    results.append(json.loads(potential_json))
                except json.JSONDecodeError as e:
                    extraction_errors.append(f"Multiple JSON extraction error at position {i}: {e}")
                i = j
            else:
                i += 1
        else:
            i += 1
    
    return results


def _format_inputs(inputs: dict) -> str:
    """Format task inputs into a structured prompt.
    
    Provides better structure and context for the LLM.
    Handles edge cases like empty inputs and non-string values.
    """
    if not inputs:
        return "No inputs provided."
    
    parts = []
    for key, value in inputs.items():
        # Convert non-string values to string representation
        if not isinstance(value, str):
            value = str(value)
        # Skip empty values but include the key
        if value.strip():
            parts.append(f"{key}:\n{value}\n")
        else:
            parts.append(f"{key}: (empty)\n")
    return "\n".join(parts) if parts else "No valid inputs provided."


def _validate_response_schema(response: dict) -> tuple[bool, str]:
    """Validate that the response follows the expected schema.
    
    Returns:
        (is_valid, error_message)
    """
    if not isinstance(response, dict):
        return False, f"Response is not a dict, got {type(response).__name__}"
    
    if "response" not in response:
        return False, "Response missing required 'response' key"
    
    if not isinstance(response["response"], str):
        return False, f"'response' value is not a string, got {type(response['response']).__name__}"
    
    if len(response["response"].strip()) == 0:
        return False, "'response' value is empty"
    
    return True, ""


def _select_best_json(extracted: list[dict]) -> dict | None:
    """Select the best JSON object from a list of extracted objects.
    
    Prioritizes objects that pass schema validation and have non-empty
    response values. Returns the first valid object found.
    
    Args:
        extracted: List of extracted JSON objects.
        
    Returns:
        The best JSON object, or None if none are valid.
    """
    if not extracted:
        return None
    
    # First pass: look for objects that pass full validation
    for obj in extracted:
        is_valid, _ = _validate_response_schema(obj)
        if is_valid:
            return obj
    
    # Second pass: look for objects with a 'response' key (even if empty)
    for obj in extracted:
        if isinstance(obj, dict) and "response" in obj:
            return obj
    
    # Third pass: return the first dict object
    for obj in extracted:
        if isinstance(obj, dict):
            return obj
    
    # Last resort: return the first object
    return extracted[0]


class TaskAgent:
    """Task agent that solves IMO grading problems with improved error handling."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "", max_retries: int = 2) -> None:
        self.model = model
        self.log_fn = logger.info
        self.call_count = 0
        self.max_retries = max_retries

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        self.call_count += 1
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

Ensure your response is valid JSON and follows the schema exactly."""

        # Retry loop with validation
        for attempt in range(self.max_retries + 1):
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=[],
                )
                self.log_fn(f"LLM call completed (attempt {attempt + 1}), response length: {len(response)}")
            except Exception as e:
                self.log_fn(f"Error in LLM call (attempt {attempt + 1}): {e}")
                if attempt == self.max_retries:
                    return "Error: LLM call failed", []
                continue

            # Extract prediction from JSON with better error handling
            prediction = "None"
            validation_passed = False
            
            try:
                if msg_history and len(msg_history) > 0:
                    last_message = msg_history[-1]
                    text_content = last_message.get("text", "")
                    extracted = _extract_jsons(text_content)
                    if extracted:
                        # Use the new selection function to pick the best JSON object
                        best_json = _select_best_json(extracted)
                        if best_json:
                            is_valid, error_msg = _validate_response_schema(best_json)
                            if is_valid:
                                prediction = best_json["response"]
                                validation_passed = True
                                self.log_fn(f"Successfully extracted and validated prediction: {str(prediction)[:100]}")
                            else:
                                self.log_fn(f"Response validation failed: {error_msg}")
                                # Try to extract any useful content even if validation fails
                                if isinstance(best_json, dict) and "response" in best_json:
                                    prediction = str(best_json["response"])
                                if attempt < self.max_retries:
                                    # Add validation feedback to the instruction for retry
                                    instruction += f"\n\nPrevious response was invalid: {error_msg}. Please fix and respond with valid JSON."
                        else:
                            self.log_fn("No valid JSON objects found after selection")
                            if attempt < self.max_retries:
                                instruction += "\n\nPrevious response did not contain valid JSON. Please wrap your response in <json>...</json> tags."
                    else:
                        self.log_fn("No JSON found in response")
                        if attempt < self.max_retries:
                            instruction += "\n\nPrevious response did not contain valid JSON. Please wrap your response in <json>...</json> tags."
                else:
                    self.log_fn("Empty message history")
            except Exception as e:
                self.log_fn(f"Error extracting prediction (attempt {attempt + 1}): {e}")
            
            if validation_passed:
                return str(prediction), msg_history
            
            if attempt == self.max_retries:
                self.log_fn(f"Max retries ({self.max_retries}) reached, returning best effort result")
                return str(prediction), msg_history
        
        return "Error: Unexpected end of forward method", []
