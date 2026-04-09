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
    
    text = text.strip()
    if not text:
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
        
        if not inner:
            continue
        
        # Try to parse, with cleanup for common issues
        try:
            results.append(json.loads(inner))
        except json.JSONDecodeError:
            # Try cleaning up common issues: trailing commas, comments, etc.
            cleaned = _clean_json_string(inner)
            try:
                results.append(json.loads(cleaned))
            except json.JSONDecodeError:
                # Last resort: try to extract just the first valid JSON object
                try:
                    # Find the first { and matching }
                    brace_start = cleaned.find("{")
                    if brace_start != -1:
                        # Try to find matching brace by counting
                        count = 0
                        for i, char in enumerate(cleaned[brace_start:]):
                            if char == '{':
                                count += 1
                            elif char == '}':
                                count -= 1
                                if count == 0:
                                    potential_json = cleaned[brace_start:brace_start + i + 1]
                                    results.append(json.loads(potential_json))
                                    break
                except (json.JSONDecodeError, ValueError):
                    continue
    
    # Second pass: try to find JSON objects directly if no <json> tags found
    if not results:
        # Look for JSON-like structures with braces
        brace_start = text.find("{")
        brace_end = text.rfind("}")
        if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
            potential_json = text[brace_start:brace_end + 1]
            try:
                results.append(json.loads(potential_json))
            except json.JSONDecodeError:
                # Try with cleanup
                cleaned = _clean_json_string(potential_json)
                try:
                    results.append(json.loads(cleaned))
                except json.JSONDecodeError:
                    # Try to find first valid JSON object
                    brace_start = cleaned.find("{")
                    if brace_start != -1:
                        count = 0
                        for i, char in enumerate(cleaned[brace_start:]):
                            if char == '{':
                                count += 1
                            elif char == '}':
                                count -= 1
                                if count == 0:
                                    potential_json = cleaned[brace_start:brace_start + i + 1]
                                    try:
                                        results.append(json.loads(potential_json))
                                    except (json.JSONDecodeError, ValueError):
                                        pass
                                    break
    
    # Third pass: try to parse the entire text as JSON if it looks like JSON
    if not results:
        text_stripped = text.strip()
        if text_stripped.startswith("{") and text_stripped.endswith("}"):
            try:
                results.append(json.loads(text_stripped))
            except json.JSONDecodeError:
                cleaned = _clean_json_string(text_stripped)
                try:
                    results.append(json.loads(cleaned))
                except (json.JSONDecodeError, ValueError):
                    pass
    
    # Fourth pass: look for JSON in code blocks (```json ... ```)
    if not results:
        import re
        # Match JSON code blocks - handle both ```json and ``` formats
        # Use a more robust pattern that handles newlines properly
        json_blocks = re.findall(r'```(?:json)?\s*\n?(.*?)```', text, re.DOTALL)
        for block in json_blocks:
            block = block.strip()
            if block.startswith('{') and block.endswith('}'):
                try:
                    results.append(json.loads(block))
                except json.JSONDecodeError:
                    cleaned = _clean_json_string(block)
                    try:
                        results.append(json.loads(cleaned))
                    except (json.JSONDecodeError, ValueError):
                        pass
            elif '{' in block and '}' in block:
                # Try to extract JSON from within the block
                brace_start = block.find('{')
                brace_end = block.rfind('}')
                if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
                    potential_json = block[brace_start:brace_end + 1]
                    try:
                        results.append(json.loads(potential_json))
                    except json.JSONDecodeError:
                        cleaned = _clean_json_string(potential_json)
                        try:
                            results.append(json.loads(cleaned))
                        except (json.JSONDecodeError, ValueError):
                            pass
    
    return results if results else None


def _clean_json_string(text: str) -> str:
    """Clean up common JSON formatting issues from LLM outputs.
    
    Handles:
    - Trailing commas before closing braces/brackets
    - Single quotes instead of double quotes
    - Comments (// and /* */)
    - Unescaped control characters (newlines, tabs) in strings
    - Unescaped quotes inside strings
    - Extra whitespace
    - Missing quotes around keys
    - Missing quotes around string values
    - Unicode escape sequences
    - BOM markers
    """
    import re
    
    if not text or not isinstance(text, str):
        return ""
    
    # Remove BOM if present
    if text.startswith('\ufeff'):
        text = text[1:]
    
    # Remove single-line and multi-line comments (but be careful not to remove content inside strings)
    # Use a state machine approach to properly handle comments
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
        
        # Check for comments only when not in a string
        if not in_string:
            # Check for single-line comment
            if char == '/' and i + 1 < len(text) and text[i + 1] == '/':
                # Skip until end of line (but don't consume the newline or closing brace)
                i += 2  # Skip the //
                while i < len(text) and text[i] not in ['\n', '}', ']']:
                    i += 1
                continue
            # Check for multi-line comment
            if char == '/' and i + 1 < len(text) and text[i + 1] == '*':
                # Skip until */
                i += 2
                while i < len(text) - 1:
                    if text[i] == '*' and text[i + 1] == '/':
                        i += 2
                        break
                    i += 1
                continue
        
        result.append(char)
        i += 1
    
    text = ''.join(result)
    
    # Remove trailing commas before } or ]
    text = re.sub(r',\s*}', '}', text)
    text = re.sub(r',\s*\]', ']', text)
    
    # Remove trailing whitespace before closing braces/brackets (from comment removal)
    text = re.sub(r'\s+}', '}', text)
    text = re.sub(r'\s+\]', ']', text)
    
    # Replace single quotes with double quotes (carefully)
    # Only replace quotes that appear to be delimiting strings
    text = re.sub(r"(?<=[{\s,:\[])'([^']*)'(?=[}\s,:\]])", r'"\1"', text)
    
    # Fix unescaped control characters and quotes in string values
    # Use a state machine approach to properly handle string contents
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
                # Look ahead to see if this looks like the end of a string
                # (followed by : or , or } or ] or whitespace)
                next_chars = text[i+1:i+5].strip()
                if next_chars and next_chars[0] in [':', ',', '}', ']', ' ', '\n', '\t']:
                    # This is likely a legitimate end of string
                    in_string = False
                    result.append(char)
                else:
                    # This is likely an unescaped quote inside the string
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
    
    def clean_value(value: Any) -> str:
        """Clean and format a value for the prompt."""
        try:
            if value is None:
                return ""
            if isinstance(value, str):
                # Handle empty or whitespace-only strings
                stripped = value.strip()
                if not stripped:
                    return ""
                # Escape any markdown code block markers to prevent formatting issues
                return stripped.replace('```', '``\u200b`')
            if isinstance(value, (list, dict)):
                # Handle empty collections
                if not value:
                    return ""
                return json.dumps(value, ensure_ascii=False, indent=2)
            if isinstance(value, bool):
                return "true" if value else "false"
            if isinstance(value, (int, float)):
                return str(value)
            return str(value).strip()
        except (TypeError, ValueError) as e:
            # Fallback for any conversion errors
            try:
                return str(value) if value is not None else ""
            except Exception:
                return ""
    
    def format_title(key: str) -> str:
        """Format a key into a readable title."""
        try:
            return key.replace('_', ' ').title()
        except Exception:
            return str(key)
    
    # Add priority fields first in order
    for key in priority_fields:
        if key in inputs:
            value = clean_value(inputs[key])
            if value:  # Only add non-empty values
                title = format_title(key)
                parts.append(f"## {title}\n{value}\n")
    
    # Add any remaining fields not in priority list
    for key, value in inputs.items():
        if key not in priority_fields:
            value = clean_value(value)
            if value:
                title = format_title(key)
                parts.append(f"## {title}\n{value}\n")
    
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
    
    # Check for response key with case-insensitive fallback
    response_key = None
    if "response" in response:
        response_key = "response"
    else:
        # Check for common typos or variations (case-insensitive)
        lower_keys = {k.lower(): k for k in response.keys()}
        for candidate in ['response', 'result', 'answer', 'output', 'evaluation', 'grade', 'grading']:
            if candidate in lower_keys:
                response_key = lower_keys[candidate]
                break
    
    if response_key is None:
        return False, f"Missing 'response' key in JSON. Available keys: {list(response.keys())}"
    
    response_value = response[response_key]
    
    # Accept various types but convert to string for consistency
    if response_value is None:
        return False, "'response' value is None"
    
    if isinstance(response_value, (dict, list)):
        # Nested structures are allowed but will be converted to string
        return True, ""
    
    if isinstance(response_value, bool):
        return True, ""
    
    if isinstance(response_value, (int, float)):
        return True, ""
    
    if isinstance(response_value, str):
        # Check for empty or whitespace-only strings
        if not response_value.strip():
            return False, "'response' value is empty or whitespace-only"
        return True, ""
    
    return False, f"'response' value has unsupported type: {type(response_value).__name__}"


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
IMPORTANT: Your previous response could not be parsed. You MUST respond with valid JSON only.
Do not include any text outside the JSON tags. Ensure proper JSON syntax with double quotes."""
        
        return f"""You are an expert grading agent. Your task is to evaluate student answers based on the provided problem, solution, and grading guidelines.

{formatted_inputs}

Respond in JSON format with the following schema:
<json>
{{
    "response": "Your evaluation result here"
}}
</json>

Ensure your response is valid JSON and follows the schema exactly.{retry_note}

Think step by step:
1. First, understand the problem and what is being asked
2. Review the provided solution to understand the correct approach
3. Examine the grading guidelines carefully
4. Evaluate the student's answer against the solution and guidelines
5. Formulate your final evaluation in the required JSON format"""

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
        
        # Try multiple messages in history (in case of tool calls or multiple turns)
        messages_to_try = list(reversed(msg_history))
        
        for msg in messages_to_try:
            if not isinstance(msg, dict):
                continue
            
            # Handle different message formats
            text_content = None
            if "text" in msg:
                text_content = msg["text"]
            elif "content" in msg:
                text_content = msg["content"]
            else:
                continue
            
            if not text_content or not isinstance(text_content, str):
                continue
            
            # Skip error messages
            if text_content.startswith("Error:"):
                continue
            
            extracted = _extract_jsons(text_content)
            if not extracted:
                continue
            
            # Try each extracted JSON object
            for extracted_obj in reversed(extracted):
                is_valid, error_msg = _validate_grading_response(extracted_obj)
                
                if is_valid:
                    # Find the response key (supports case-insensitive matching)
                    prediction = None
                    if "response" in extracted_obj:
                        prediction = extracted_obj.get("response")
                    else:
                        # Try case-insensitive key matching
                        lower_keys = {k.lower(): k for k in extracted_obj.keys()}
                        for candidate in ['response', 'result', 'answer', 'output', 'evaluation', 'grade', 'grading']:
                            if candidate in lower_keys:
                                prediction = extracted_obj.get(lower_keys[candidate])
                                break
                    
                    if prediction is not None:
                        # Convert to string, handling nested structures
                        if isinstance(prediction, (dict, list)):
                            try:
                                prediction = json.dumps(prediction, ensure_ascii=False)
                            except (TypeError, ValueError) as e:
                                self.log_fn(f"Failed to serialize nested response: {e}")
                                prediction = str(prediction)
                        elif isinstance(prediction, bool):
                            prediction = "true" if prediction else "false"
                        else:
                            prediction = str(prediction)
                        
                        self.log_fn(f"Successfully extracted prediction: {prediction[:100]}")
                        return prediction, True
        
        # If we get here, no valid prediction was found
        self.log_fn("No valid grading response found in message history")
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
