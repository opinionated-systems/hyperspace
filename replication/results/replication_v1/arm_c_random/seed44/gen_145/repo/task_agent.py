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
    """Extract JSON objects from <json>...</json> blocks with enhanced robustness.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    Also attempts to parse raw JSON objects if no <json> tags are found.
    Includes additional cleanup for common LLM output issues.
    """
    if not text or not isinstance(text, str):
        return None
    
    # Remove Unicode BOM and normalize whitespace
    text = text.lstrip('\ufeff\u200b\u200c\u200d\ufeff')
        
    results = []
    search_from = 0
    
    # First pass: extract from <json> tags (case insensitive)
    text_lower = text.lower()
    while True:
        start = text_lower.find("<json>", search_from)
        if start == -1:
            break
        end = text_lower.find("</json>", start)
        if end == -1:
            break
        # Use original case text for extraction
        inner = text[start + 6:end].strip()
        search_from = end + 7
        
        if not inner:
            continue
        
        # Try to parse, with cleanup for common issues
        parsed = _try_parse_json(inner)
        if parsed is not None:
            results.append(parsed)
    
    # Second pass: try to find JSON objects directly if no <json> tags found
    if not results:
        # Look for JSON-like structures with braces
        brace_start = text.find("{")
        while brace_start != -1:
            # Try to find a valid JSON object starting at this brace
            parsed = _try_parse_json_at_position(text, brace_start)
            if parsed is not None:
                results.append(parsed)
                # Continue searching after this object
                # Find where this object ends
                obj_end = _find_json_end(text, brace_start)
                if obj_end > brace_start:
                    brace_start = text.find("{", obj_end)
                else:
                    brace_start = text.find("{", brace_start + 1)
            else:
                brace_start = text.find("{", brace_start + 1)
    
    return results or None


def _try_parse_json(text: str) -> dict | None:
    """Try to parse a string as JSON with various cleanup attempts.
    
    Returns the parsed dict or None if parsing fails.
    """
    if not text or not isinstance(text, str):
        return None
    
    # Direct parse attempt
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
        return None
    except json.JSONDecodeError:
        pass
    
    # Try with cleanup
    cleaned = _clean_json_string(text)
    try:
        result = json.loads(cleaned)
        if isinstance(result, dict):
            return result
        return None
    except json.JSONDecodeError:
        pass
    
    # Try to extract first valid JSON object
    return _try_parse_json_at_position(cleaned, 0)


def _find_json_end(text: str, start: int) -> int:
    """Find the end position of a JSON object starting at start.
    
    Returns the index of the closing brace or -1 if not found.
    """
    if start < 0 or start >= len(text) or text[start] != '{':
        return -1
    
    count = 0
    in_string = False
    escape_next = False
    
    for i in range(start, len(text)):
        char = text[i]
        
        if escape_next:
            escape_next = False
            continue
        
        if char == '\\':
            escape_next = True
            continue
        
        if char == '"':
            in_string = not in_string
            continue
        
        if not in_string:
            if char == '{':
                count += 1
            elif char == '}':
                count -= 1
                if count == 0:
                    return i
    
    return -1


def _try_parse_json_at_position(text: str, start: int) -> dict | None:
    """Try to parse a JSON object starting at the given position.
    
    Returns the parsed dict or None if parsing fails.
    """
    if start < 0 or start >= len(text):
        return None
    
    # Find the end of the JSON object
    end = _find_json_end(text, start)
    if end == -1:
        return None
    
    # Try to parse the extracted object
    potential_json = text[start:end + 1]
    
    try:
        result = json.loads(potential_json)
        if isinstance(result, dict):
            return result
        return None
    except json.JSONDecodeError:
        # Try with cleanup
        cleaned = _clean_json_string(potential_json)
        try:
            result = json.loads(cleaned)
            if isinstance(result, dict):
                return result
            return None
        except json.JSONDecodeError:
            return None


def _clean_json_string(text: str) -> str:
    """Clean up common JSON formatting issues from LLM outputs.
    
    Handles:
    - Trailing commas before closing braces/brackets
    - Single quotes instead of double quotes
    - Comments (// and /* */)
    - Unescaped control characters (newlines, tabs) in strings
    - Unescaped quotes inside strings
    - Extra whitespace
    - Unicode BOM and other invisible characters
    - Multiple consecutive whitespace
    """
    import re
    
    if not text or not isinstance(text, str):
        return text if text else ""
    
    # Remove Unicode BOM and other invisible characters at start
    text = text.lstrip('\ufeff\u200b\u200c\u200d\ufeff')
    
    # Remove single-line comments (but not inside strings)
    # Use a more careful approach that respects string boundaries
    result_lines = []
    for line in text.split('\n'):
        # Find // that's not inside a string
        in_str = False
        escape = False
        comment_start = -1
        for i, char in enumerate(line):
            if escape:
                escape = False
                continue
            if char == '\\':
                escape = True
                continue
            if char == '"':
                in_str = not in_str
            elif not in_str and char == '/' and i + 1 < len(line) and line[i + 1] == '/':
                comment_start = i
                break
        if comment_start >= 0:
            line = line[:comment_start]
        result_lines.append(line)
    text = '\n'.join(result_lines)
    
    # Remove multi-line comments (but be careful with nested braces)
    # Use a non-greedy approach with proper nesting awareness
    while '/*' in text and '*/' in text:
        start = text.find('/*')
        end = text.find('*/', start + 2)
        if end == -1:
            break
        text = text[:start] + ' ' + text[end + 2:]
    
    # Remove trailing commas before } or ] (but not inside strings)
    # This regex needs to be careful about string boundaries
    # A simpler approach: process character by character
    result = []
    in_string = False
    escape_next = False
    last_non_ws = None
    
    for char in text:
        if escape_next:
            result.append(char)
            escape_next = False
            if not in_string:
                last_non_ws = char if not char.isspace() else last_non_ws
            continue
        
        if char == '\\':
            result.append(char)
            escape_next = True
            if not in_string:
                last_non_ws = char
            continue
        
        if char == '"':
            in_string = not in_string
            result.append(char)
            if not in_string:
                last_non_ws = char
            continue
        
        if not in_string:
            if char in '}]':
                # Remove trailing comma if present
                if last_non_ws == ',':
                    # Find and remove the last comma
                    for i in range(len(result) - 1, -1, -1):
                        if result[i] == ',':
                            # Check this isn't part of a string
                            result.pop(i)
                            break
                        elif not result[i].isspace():
                            break
            if not char.isspace():
                last_non_ws = char
        
        result.append(char)
    
    text = ''.join(result)
    
    # Replace single quotes with double quotes (only outside strings)
    # This is tricky - we need to identify string boundaries first
    result = []
    in_string = False
    escape_next = False
    i = 0
    while i < len(text):
        char = text[i]
        if escape_next:
            result.append(char)
            escape_next = False
            i += 1
            continue
        if char == '\\':
            result.append(char)
            escape_next = True
            i += 1
            continue
        if char == '"':
            in_string = not in_string
            result.append(char)
            i += 1
            continue
        if not in_string and char == "'":
            # Check if this looks like a string delimiter
            # Look for matching closing quote
            j = i + 1
            while j < len(text) and text[j] != "'":
                if text[j] == '\\' and j + 1 < len(text):
                    j += 2
                else:
                    j += 1
            if j < len(text):
                # Found matching quote, convert content
                content = text[i+1:j]
                # Escape any double quotes in the content
                content = content.replace('"', '\\"')
                result.append('"')
                result.append(content)
                result.append('"')
                i = j + 1
                continue
            else:
                # No matching quote, treat as regular char
                result.append(char)
                i += 1
                continue
        result.append(char)
        i += 1
    
    text = ''.join(result)
    
    # Handle unquoted keys (like {response: "test"})
    # Add quotes around unquoted keys that look like identifiers
    result = []
    i = 0
    in_string = False
    escape_next = False
    while i < len(text):
        char = text[i]
        if escape_next:
            result.append(char)
            escape_next = False
            i += 1
            continue
        if char == '\\':
            result.append(char)
            escape_next = True
            i += 1
            continue
        if char == '"':
            in_string = not in_string
            result.append(char)
            i += 1
            continue
        
        # Check for unquoted key pattern: identifier followed by colon (not in string)
        if not in_string and (char.isalpha() or char == '_'):
            # Check if this looks like an unquoted key
            j = i
            while j < len(text) and (text[j].isalnum() or text[j] == '_'):
                j += 1
            # Skip whitespace
            k = j
            while k < len(text) and text[k].isspace():
                k += 1
            if k < len(text) and text[k] == ':':
                # This is an unquoted key, add quotes
                key = text[i:j]
                result.append('"')
                result.append(key)
                result.append('"')
                i = j
                continue
        
        result.append(char)
        i += 1
    
    text = ''.join(result)
    
    # Fix unescaped control characters and quotes in string values
    result = []
    i = 0
    in_string = False
    escape_next = False
    
    while i < len(text):
        char = text[i]
        
        if escape_next:
            result.append(char)
            escape_next = False
            i += 1
            continue
        
        if char == '\\':
            result.append(char)
            escape_next = True
            i += 1
            continue
        
        if char == '"':
            if not in_string:
                in_string = True
                result.append(char)
            else:
                # We're inside a string, check if this quote should be escaped
                next_chars = text[i+1:i+5].strip()
                if next_chars and next_chars[0] in [':', ',', '}', ']', ' ', '\n', '\t', '']:
                    in_string = False
                    result.append(char)
                else:
                    result.append('\\"')
            i += 1
            continue
        
        # Handle control characters inside strings
        if in_string and char in ['\n', '\t', '\r']:
            if char == '\n':
                result.append('\\n')
            elif char == '\t':
                result.append('\\t')
            elif char == '\r':
                result.append('\\r')
            i += 1
            continue
        
        result.append(char)
        i += 1
    
    return ''.join(result).strip()


def _format_inputs(inputs: dict) -> str:
    """Format task inputs into a structured prompt with improved formatting.
    
    Provides better structure and context for the LLM.
    Handles various input types and ensures clean formatting.
    Includes robust error handling for edge cases.
    """
    if not inputs or not isinstance(inputs, dict):
        return ""
    
    parts = []
    
    # Define priority order for common fields to ensure logical flow
    priority_fields = ["domain", "problem", "solution", "grading_guidelines", "student_answer"]
    
    def clean_value(value: Any, key: str = "") -> str:
        """Clean and format a value for the prompt."""
        try:
            if value is None:
                return ""
            if isinstance(value, str):
                # Handle special cases for student_answer
                if key == "student_answer":
                    # Preserve the original format but clean it up
                    cleaned = value.strip()
                    # Remove any trailing "None" that might be added by error
                    if cleaned.endswith("None"):
                        cleaned = cleaned[:-4].strip()
                    return cleaned
                return value.strip()
            if isinstance(value, (list, dict)):
                # Pretty print with indentation for readability
                return json.dumps(value, ensure_ascii=False, indent=2)
            return str(value).strip()
        except Exception as e:
            # Fallback for any conversion errors
            logger.warning(f"Error cleaning value for key '{key}': {e}")
            return str(value) if value is not None else ""
    
    # Add priority fields first in order
    for key in priority_fields:
        if key in inputs:
            try:
                value = clean_value(inputs[key], key)
                if value:  # Only add non-empty values
                    title = key.replace('_', ' ').title()
                    parts.append(f"## {title}\n{value}\n")
            except Exception as e:
                logger.warning(f"Error formatting priority field '{key}': {e}")
                # Try simple fallback
                try:
                    parts.append(f"## {key}\n{str(inputs[key])}\n")
                except Exception:
                    pass
    
    # Add any remaining fields not in priority list
    for key, value in inputs.items():
        if key not in priority_fields:
            try:
                value = clean_value(value, key)
                if value:
                    title = key.replace('_', ' ').title()
                    parts.append(f"## {title}\n{value}\n")
            except Exception as e:
                logger.warning(f"Error formatting field '{key}': {e}")
                # Try simple fallback
                try:
                    parts.append(f"## {key}\n{str(value)}\n")
                except Exception:
                    pass
    
    return "\n".join(parts)


def _validate_grading_response(response: Any) -> tuple[bool, str]:
    """Validate that the grading response has the required structure.
    
    Returns:
        (is_valid, error_message)
    """
    if response is None:
        return False, "Response is None"
    
    if not isinstance(response, dict):
        return False, f"Response is not a dictionary, got {type(response).__name__}"
    
    if len(response) == 0:
        return False, "Response is an empty dictionary"
    
    # Check for the required 'response' key
    if "response" not in response:
        # Check for common typos or variations (case-insensitive)
        response_keys = [k for k in response.keys() if k.lower() in ['response', 'result', 'answer', 'output', 'evaluation']]
        if response_keys:
            # Use the first matching key as a fallback
            matched_key = response_keys[0]
            logger.info(f"Using alternative key '{matched_key}' instead of 'response'")
            response["response"] = response[matched_key]
            return True, ""
        return False, f"Missing 'response' key in JSON. Available keys: {list(response.keys())}"
    
    response_value = response["response"]
    
    # Accept various types but convert to string for consistency
    if response_value is None:
        return False, "'response' value is None"
    
    if isinstance(response_value, (dict, list)):
        # Nested structures are allowed but will be converted to string
        return True, ""
    
    if not isinstance(response_value, (str, int, float, bool)):
        return False, f"'response' value has unsupported type: {type(response_value).__name__}"
    
    # Additional validation: check for empty string responses
    if isinstance(response_value, str) and not response_value.strip():
        logger.warning("Response value is an empty string")
        # Still accept it, but log the warning
    
    return True, ""


class TaskAgent:
    """Task agent that solves IMO grading problems with improved error handling and retry logic."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "", max_retries: int = 2) -> None:
        self.model = model
        self.log_fn = logger.info
        self.call_count = 0
        self.success_count = 0
        self.error_count = 0
        self.retry_count = 0
        self.max_retries = max_retries

    def _build_instruction(self, formatted_inputs: str, is_retry: bool = False) -> str:
        """Build the instruction prompt for the LLM.
        
        For retries, adds stronger emphasis on JSON format compliance.
        """
        retry_note = ""
        if is_retry:
            retry_note = """

⚠️ IMPORTANT: Your previous response could not be parsed correctly. 
You MUST respond with valid JSON only, wrapped in <json>...</json> tags.
- Use double quotes for all strings (not single quotes)
- Ensure all braces and brackets are properly closed
- Do not include any text outside the JSON tags
- The JSON must contain a "response" key with your evaluation"""
        
        return f"""You are an expert grading agent. Your task is to evaluate student answers based on the provided problem, solution, and grading guidelines.

{formatted_inputs}

You MUST respond in the following JSON format:
<json>
{{
    "response": "Your evaluation result here"
}}
</json>

Requirements:
1. Your response MUST be valid JSON wrapped in <json>...</json> tags
2. The JSON object MUST contain a "response" key
3. Use double quotes for all strings (not single quotes)
4. Ensure all braces {{}} and brackets [] are properly balanced
5. Do not include any explanatory text outside the JSON tags{retry_note}

Evaluation Process:
1. First, understand the problem and what is being asked
2. Review the provided solution to understand the correct approach
3. Examine the grading guidelines carefully
4. Evaluate the student's answer against the solution and guidelines
5. Formulate your final evaluation in the required JSON format

Your "response" value should be a clear, concise evaluation of the student's answer."""

    def _extract_prediction(self, msg_history: list[dict]) -> tuple[str, bool]:
        """Extract prediction from message history.
        
        Returns:
            (prediction, success)
        """
        if not msg_history:
            self.log_fn("Empty message history")
            return "None", False
        
        if not isinstance(msg_history, list):
            self.log_fn(f"Invalid message history type: {type(msg_history)}")
            return "None", False
        
        # Try multiple messages in history, starting from the most recent
        for msg_idx in range(len(msg_history) - 1, -1, -1):
            message = msg_history[msg_idx]
            if not isinstance(message, dict):
                continue
            
            # Try both 'text' and 'content' keys
            text_content = message.get("text", "") or message.get("content", "")
            if not text_content or not isinstance(text_content, str):
                continue
            
            # Skip error messages
            if text_content.startswith("Error:"):
                continue
            
            extracted = _extract_jsons(text_content)
            if not extracted:
                continue
            
            # Try each extracted JSON object, starting from the last one
            for last_extracted in reversed(extracted):
                is_valid, error_msg = _validate_grading_response(last_extracted)
                
                if is_valid:
                    prediction = last_extracted.get("response")
                    if prediction is not None:
                        # Convert to string, handling nested structures
                        if isinstance(prediction, (dict, list)):
                            try:
                                prediction = json.dumps(prediction, ensure_ascii=False)
                            except (TypeError, ValueError) as e:
                                self.log_fn(f"Failed to serialize nested response: {e}")
                                prediction = str(prediction)
                        else:
                            prediction = str(prediction)
                        
                        self.log_fn(f"Successfully extracted prediction from message {msg_idx}: {prediction[:100]}")
                        return prediction, True
            
            # If we found JSON but none were valid, log the error
            if extracted:
                self.log_fn(f"Found {len(extracted)} JSON objects but none were valid: {error_msg}")
        
        # If we get here, no valid prediction was found in any message
        self.log_fn("No valid grading response found in any message")
        return "None", False

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with retry logic.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        self.call_count += 1
        self.log_fn(f"TaskAgent call #{self.call_count} starting")
        
        # Format inputs more clearly
        formatted_inputs = _format_inputs(inputs)
        
        # Try initial call
        prediction, msg_history, success = self._try_call(formatted_inputs, is_retry=False)
        
        # Retry on failure if we have retries left
        attempt = 0
        while not success and attempt < self.max_retries:
            self.retry_count += 1
            attempt += 1
            self.log_fn(f"Retry attempt {attempt}/{self.max_retries}")
            prediction, msg_history, success = self._try_call(formatted_inputs, is_retry=True)
        
        # Update stats
        if success:
            self.success_count += 1
        else:
            self.error_count += 1
        
        self.log_fn(f"TaskAgent stats: {self.success_count} successes, {self.error_count} errors, {self.retry_count} retries out of {self.call_count} calls")
        return prediction, msg_history

    def _try_call(self, formatted_inputs: str, is_retry: bool = False) -> tuple[str, list[dict], bool]:
        """Make a single LLM call and extract prediction.
        
        Returns:
            (prediction, msg_history, success)
        """
        if not formatted_inputs:
            self.log_fn("Empty formatted inputs")
            return "Error: Empty inputs", [], False
        
        instruction = self._build_instruction(formatted_inputs, is_retry)
        
        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
            
            # Validate response
            if not isinstance(response, str):
                self.log_fn(f"Invalid response type from LLM: {type(response)}")
                return "Error: Invalid response type", msg_history if isinstance(msg_history, list) else [], False
            
            self.log_fn(f"LLM call completed, response length: {len(response)}")
            
            # Check for error responses
            if response.startswith("Error:"):
                self.log_fn(f"LLM returned error: {response}")
                return response, msg_history, False
                
        except Exception as e:
            self.log_fn(f"Error in LLM call: {type(e).__name__}: {e}")
            return f"Error: LLM call failed - {type(e).__name__}", [], False

        # Extract prediction from JSON
        prediction, success = self._extract_prediction(msg_history)
        return prediction, msg_history, success

    def get_stats(self) -> dict:
        """Return agent performance statistics."""
        return {
            "total_calls": self.call_count,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "retry_count": self.retry_count,
            "success_rate": self.success_count / max(1, self.call_count),
        }
