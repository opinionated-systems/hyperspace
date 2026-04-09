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
    Includes improved handling for nested braces, malformed JSON, and 
    common LLM output patterns like trailing commas and comments.
    
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
        
        # Try to parse the JSON, with cleanup for common LLM issues
        parsed = _parse_json_with_cleanup(inner)
        if parsed is not None:
            results.append(parsed)
        else:
            extraction_errors.append(f"JSON parse error at position {start}")
            continue
    
    # Fallback 1: Try to find JSON objects directly if no <json> tags
    if not results:
        try:
            # Look for JSON-like structures with braces
            brace_start = text.find("{")
            brace_end = text.rfind("}")
            if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
                potential_json = text[brace_start:brace_end + 1]
                parsed = _parse_json_with_cleanup(potential_json)
                if parsed is not None:
                    results.append(parsed)
        except Exception as e:
            extraction_errors.append(f"Fallback JSON parse error: {e}")
    
    # Fallback 2: Try to extract JSON from markdown code blocks
    if not results:
        json_code_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
        matches = re.findall(json_code_block_pattern, text, re.DOTALL)
        for match in matches:
            parsed = _parse_json_with_cleanup(match.strip())
            if parsed is not None:
                results.append(parsed)
            else:
                extraction_errors.append(f"Code block JSON parse error")
    
    # Log extraction details for debugging
    if extraction_errors and logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"JSON extraction errors: {extraction_errors}")
    
    return results or None


def _parse_json_with_cleanup(text: str) -> dict | None:
    """Parse JSON text with cleanup for common LLM output issues.
    
    Handles:
    - Trailing commas in objects and arrays
    - Single-quoted strings (converted to double-quoted)
    - Comments (// and /* */ style, removed)
    - Control characters (escaped)
    
    Args:
        text: The JSON text to parse.
        
    Returns:
        Parsed JSON dict, or None if parsing fails.
    """
    if not text or not text.strip():
        return None
    
    # First, try raw parsing
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Apply cleanups
    cleaned = text
    
    # Remove // comments (but not inside strings)
    lines = []
    for line in cleaned.split('\n'):
        in_string = False
        escape_next = False
        result = []
        for i, char in enumerate(line):
            if escape_next:
                result.append(char)
                escape_next = False
                continue
            if char == '\\':
                result.append(char)
                escape_next = True
                continue
            if char == '"' and not in_string:
                in_string = True
                result.append(char)
                continue
            if char == '"' and in_string:
                in_string = False
                result.append(char)
                continue
            if not in_string and char == '/' and i + 1 < len(line) and line[i + 1] == '/':
                break
            result.append(char)
        lines.append(''.join(result))
    cleaned = '\n'.join(lines)
    
    # Remove /* */ comments (simple approach)
    while '/*' in cleaned and '*/' in cleaned:
        start = cleaned.find('/*')
        end = cleaned.find('*/', start) + 2
        if start != -1 and end > start:
            cleaned = cleaned[:start] + cleaned[end:]
        else:
            break
    
    # Remove trailing commas before } or ]
    cleaned = re.sub(r',\s*}', '}', cleaned)
    cleaned = re.sub(r',\s*\]', ']', cleaned)
    
    # Try parsing with cleanups
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    
    # Last resort: try with single-quote to double-quote conversion
    # This is risky but sometimes LLMs output single-quoted JSON
    try:
        # Simple conversion: replace ' with " but be careful with apostrophes
        # This is a best-effort approach
        double_quoted = cleaned.replace("'", '"')
        return json.loads(double_quoted)
    except json.JSONDecodeError:
        pass
    
    return None


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
                        last_extracted = extracted[-1]
                        is_valid, error_msg = _validate_response_schema(last_extracted)
                        if is_valid:
                            prediction = last_extracted["response"]
                            validation_passed = True
                            self.log_fn(f"Successfully extracted and validated prediction: {str(prediction)[:100]}")
                        else:
                            self.log_fn(f"Response validation failed: {error_msg}")
                            if attempt < self.max_retries:
                                # Add validation feedback to the instruction for retry
                                instruction += f"\n\nPrevious response was invalid: {error_msg}. Please fix and respond with valid JSON."
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
