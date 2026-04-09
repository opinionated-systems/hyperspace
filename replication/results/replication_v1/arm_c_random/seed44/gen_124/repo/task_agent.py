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
    
    # Clean the text first - remove common problematic characters
    text = text.replace('\x00', '')  # Remove null bytes
    text = text.replace('\r\n', '\n').replace('\r', '\n')  # Normalize line endings
        
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
        
        # Try to parse, with multiple fallback strategies
        parsed = _try_parse_json(inner)
        if parsed is not None:
            results.append(parsed)
    
    # Second pass: look for JSON code blocks (```json ... ```)
    if not results:
        code_block_pattern = r'```(?:json)?\s*([\s\S]*?)```'
        import re
        for match in re.finditer(code_block_pattern, text):
            inner = match.group(1).strip()
            if inner:
                parsed = _try_parse_json(inner)
                if parsed is not None:
                    results.append(parsed)
    
    # Third pass: try to find JSON objects directly with brace matching
    if not results:
        parsed = _extract_json_by_braces(text)
        if parsed is not None:
            results.append(parsed)
    
    # Fourth pass: look for JSON-like structures with relaxed parsing
    if not results:
        # Try to find anything that looks like a JSON object
        relaxed_pattern = r'\{[^{}]*"[^"]+"[^{}]*\}'
        for match in re.finditer(relaxed_pattern, text):
            potential = match.group(0)
            parsed = _try_parse_json(potential)
            if parsed is not None:
                results.append(parsed)
                break  # Only take the first valid one from this pass
    
    return results or None


def _try_parse_json(text: str) -> dict | list | None:
    """Try to parse JSON with multiple cleanup strategies.
    
    Returns the parsed object or None if all strategies fail.
    """
    if not text:
        return None
    
    # Pre-clean the text
    text = text.strip()
    text = text.replace('\x00', '')  # Remove null bytes
    text = text.replace('\r\n', '\n').replace('\r', '\n')  # Normalize line endings
    
    # Strategy 1: Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Strategy 2: Clean and parse
    cleaned = _clean_json_string(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    
    # Strategy 3: Extract just the first valid JSON object with proper brace matching
    try:
        # Find the first { and try to match braces with string awareness
        start = text.find('{')
        if start != -1:
            brace_count = 0
            in_string = False
            escape_next = False
            for i, char in enumerate(text[start:], start):
                if escape_next:
                    escape_next = False
                    continue
                if char == '\\':
                    escape_next = True
                    continue
                if char == '"' and not in_string:
                    in_string = True
                elif char == '"' and in_string:
                    in_string = False
                elif not in_string:
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            potential = text[start:i+1]
                            try:
                                return json.loads(potential)
                            except json.JSONDecodeError:
                                cleaned = _clean_json_string(potential)
                                try:
                                    return json.loads(cleaned)
                                except json.JSONDecodeError:
                                    pass
                            break
    except Exception:
        pass
    
    # Strategy 4: Try to fix common LLM JSON errors
    try:
        # Fix unquoted keys (e.g., {key: "value"} -> {"key": "value"})
        import re
        fixed = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', text)
        # Fix single quotes to double quotes (carefully)
        fixed = fixed.replace("'", '"')
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass
    except Exception:
        pass
    
    return None


def _extract_json_by_braces(text: str) -> dict | list | None:
    """Extract JSON by matching braces with proper nesting.
    
    This handles cases where JSON is embedded in other text.
    Tries multiple starting points and validates each potential match.
    """
    if not text:
        return None
    
    # Clean the text first
    text = text.replace('\x00', '')
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    # Find all potential JSON starting points
    start_indices = [m.start() for m in re.finditer(r'\{', text)]
    
    # Try each starting point, prioritizing those that look like objects with keys
    scored_starts = []
    for start in start_indices:
        score = 0
        # Look ahead for indicators of a good JSON object
        lookahead = text[start:start+100]
        if '"' in lookahead or "'" in lookahead:
            score += 1  # Has quoted strings
        if ':' in lookahead:
            score += 2  # Has key-value separator
        if '"response"' in lookahead or "'response'" in lookahead:
            score += 5  # Has the expected key
        scored_starts.append((score, start))
    
    # Sort by score descending (higher score = more likely to be valid JSON)
    scored_starts.sort(reverse=True)
    
    for score, start in scored_starts:
        # Try to find matching closing brace
        brace_count = 0
        in_string = False
        escape_next = False
        
        for i, char in enumerate(text[start:], start):
            if escape_next:
                escape_next = False
                continue
            if char == '\\':
                escape_next = True
                continue
            if char == '"' and not in_string:
                in_string = True
            elif char == '"' and in_string:
                in_string = False
            elif not in_string:
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        potential = text[start:i+1]
                        parsed = _try_parse_json(potential)
                        if parsed is not None:
                            return parsed
                        break  # Move to next starting point if this one failed
                        
    return None


def _clean_json_string(text: str) -> str:
    """Clean up common JSON formatting issues from LLM outputs.
    
    Handles:
    - Trailing commas before closing braces/brackets
    - Single quotes instead of double quotes
    - Comments (// and /* */)
    - Extra whitespace and newlines
    - Unescaped newlines in strings
    - Missing quotes around keys
    - Unicode BOM and special characters
    - Control characters
    - Line ending normalization
    """
    import re
    
    if not text:
        return ""
    
    # Remove BOM if present
    text = text.lstrip('\ufeff')
    
    # Normalize line endings first
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    # Remove null bytes
    text = text.replace('\x00', '')
    
    # Remove control characters except tab, newline
    text = ''.join(char for char in text if ord(char) >= 32 or char in '\t\n')
    
    # Remove single-line comments (but not inside strings)
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
        if char == '"' and not in_string:
            in_string = True
            result.append(char)
        elif char == '"' and in_string:
            in_string = False
            result.append(char)
        elif not in_string and char == '/' and i + 1 < len(text):
            if text[i + 1] == '/':
                # Skip to end of line
                while i < len(text) and text[i] != '\n':
                    i += 1
                continue
            elif text[i + 1] == '*':
                # Skip multi-line comment
                i += 2
                while i < len(text) - 1:
                    if text[i] == '*' and text[i + 1] == '/':
                        i += 2
                        break
                    i += 1
                continue
            else:
                result.append(char)
        else:
            result.append(char)
        i += 1
    
    text = ''.join(result)
    
    # Remove trailing commas before } or ] (not inside strings)
    # Use a more robust approach with state tracking
    result = []
    in_string = False
    escape_next = False
    last_non_ws = None
    
    for char in text:
        if escape_next:
            result.append(char)
            escape_next = False
            if not in_string:
                last_non_ws = char
            continue
        if char == '\\':
            result.append(char)
            escape_next = True
            if not in_string:
                last_non_ws = char
            continue
        if char == '"' and not in_string:
            in_string = True
            result.append(char)
        elif char == '"' and in_string:
            in_string = False
            result.append(char)
            last_non_ws = char
        elif not in_string:
            if char in ' \t\n':
                result.append(char)
            elif char in '}]':
                # Remove trailing comma if present
                while result and result[-1] in ' \t\n':
                    result.pop()
                if result and result[-1] == ',':
                    result.pop()
                result.append(char)
                last_non_ws = char
            else:
                result.append(char)
                last_non_ws = char
        else:
            # Inside string - escape unescaped newlines
            if char == '\n':
                result.append('\\n')
            else:
                result.append(char)
    
    text = ''.join(result)
    
    # Replace single quotes with double quotes for JSON keys and string values
    # This is tricky - we need to be careful not to replace quotes inside strings
    def replace_quotes(match):
        # Replace outer single quotes with double quotes
        inner = match.group(1)
        # Escape any double quotes inside
        inner = inner.replace('"', '\\"')
        return f'"{inner}"'
    
    # Pattern: single-quoted strings that look like JSON keys or values
    text = re.sub(r"(?<=[{\s,:\[])'((?:[^'\\]|\\.)*)'(?=[}\s,:\]])", replace_quotes, text)
    
    # Fix missing quotes around keys (e.g., {key: "value"} -> {"key": "value"})
    text = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', text)
    
    # Normalize whitespace
    text = text.strip()
    
    return text


def _format_inputs(inputs: dict) -> str:
    """Format task inputs into a structured prompt with improved formatting.
    
    Provides better structure and context for the LLM.
    Handles various input types and ensures clean formatting.
    Includes validation and sanitization of inputs.
    """
    if not isinstance(inputs, dict):
        logger.warning(f"Expected dict for inputs, got {type(inputs)}")
        return str(inputs) if inputs else ""
    
    parts = []
    
    # Define priority order for common fields to ensure logical flow
    priority_fields = ["domain", "problem", "solution", "grading_guidelines", "student_answer"]
    
    def _sanitize_value(value: Any) -> str:
        """Sanitize and convert value to string."""
        if value is None:
            return ""
        if isinstance(value, str):
            # Remove null bytes and control characters
            value = value.replace('\x00', '')
            # Normalize line endings
            value = value.replace('\r\n', '\n').replace('\r', '\n')
            return value.strip()
        if isinstance(value, (list, dict)):
            try:
                return json.dumps(value, indent=2)
            except (TypeError, ValueError):
                return str(value)
        return str(value).strip()
    
    # Add priority fields first in order
    for key in priority_fields:
        if key in inputs:
            value = _sanitize_value(inputs[key])
            if value:  # Only add non-empty values
                # Format key nicely
                formatted_key = key.replace('_', ' ').title()
                parts.append(f"## {formatted_key}\n{value}\n")
    
    # Add any remaining fields not in priority list
    for key, value in inputs.items():
        if key not in priority_fields:
            value = _sanitize_value(value)
            if value:
                formatted_key = key.replace('_', ' ').title()
                parts.append(f"## {formatted_key}\n{value}\n")
    
    result = "\n".join(parts)
    
    # Final validation
    if not result.strip():
        logger.warning("Formatted inputs resulted in empty string")
        # Return a minimal representation of available keys
        available_keys = list(inputs.keys())
        return f"## Available Fields\n{', '.join(available_keys)}"
    
    return result


def _validate_grading_response(response: dict | list | Any) -> tuple[bool, str]:
    """Validate that the grading response has the required structure.
    
    Returns:
        (is_valid, error_message)
    """
    if response is None:
        return False, "Response is None"
    
    if not isinstance(response, dict):
        return False, f"Response is not a dictionary, got {type(response).__name__}"
    
    if len(response) == 0:
        return False, "Response dictionary is empty"
    
    if "response" not in response:
        # Check for common misspellings or variations
        possible_keys = [k for k in response.keys() if k.lower() in ['response', 'result', 'answer', 'output']]
        if possible_keys:
            return False, f"Missing 'response' key. Did you mean: {', '.join(possible_keys)}?"
        available_keys = list(response.keys())[:5]  # Limit to first 5 keys
        return False, f"Missing 'response' key. Available keys: {available_keys}"
    
    response_value = response["response"]
    
    # Accept various types but convert to string for consistency
    if response_value is None:
        return False, "'response' value is None"
    
    if isinstance(response_value, (dict, list)):
        # Nested structures are valid but should be converted to string
        return True, ""
    
    if not isinstance(response_value, (str, int, float, bool)):
        return False, f"'response' value has unsupported type: {type(response_value).__name__}"
    
    # Check for empty response
    if isinstance(response_value, str) and not response_value.strip():
        return True, ""  # Empty string is technically valid but might indicate an issue
    
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
        """Extract prediction from message history with enhanced error handling.
        
        Returns:
            (prediction, success)
        """
        if not msg_history:
            self.log_fn("Empty message history")
            return "None", False
        
        if not isinstance(msg_history, list):
            self.log_fn(f"Invalid message history type: {type(msg_history)}")
            return "None", False
        
        last_message = msg_history[-1]
        if not isinstance(last_message, dict):
            self.log_fn(f"Invalid last message type: {type(last_message)}")
            return "None", False
        
        # Try multiple keys for text content
        text_content = None
        for key in ["text", "content", "message", "output"]:
            if key in last_message:
                text_content = last_message.get(key)
                if text_content:
                    break
        
        if not text_content:
            self.log_fn(f"No text content found in message. Keys: {list(last_message.keys())}")
            return "None", False
        
        if not isinstance(text_content, str):
            text_content = str(text_content)
        
        # Extract JSON
        extracted = _extract_jsons(text_content)
        if not extracted:
            self.log_fn(f"No JSON found in response. Content preview: {text_content[:200]}...")
            return "None", False
        
        # Try each extracted JSON object (in case the last one is invalid)
        for i, candidate in enumerate(reversed(extracted)):
            is_valid, error_msg = _validate_grading_response(candidate)
            if is_valid:
                prediction = candidate.get("response")
                if prediction is not None:
                    # Convert to string, handling nested structures
                    if isinstance(prediction, (dict, list)):
                        try:
                            prediction = json.dumps(prediction)
                        except (TypeError, ValueError):
                            prediction = str(prediction)
                    prediction = str(prediction)
                    self.log_fn(f"Successfully extracted prediction (attempt {i+1}): {prediction[:100]}")
                    return prediction, True
        
        # All candidates failed validation
        self.log_fn(f"All {len(extracted)} JSON candidates failed validation. Last error: {error_msg}")
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
        
        # Validate inputs
        if not isinstance(inputs, dict):
            error_msg = f"Invalid inputs type: expected dict, got {type(inputs).__name__}"
            self.log_fn(error_msg)
            self.error_count += 1
            return f"Error: {error_msg}", []
        
        # Check for required fields
        required_fields = ["problem", "student_answer"]
        missing_fields = [f for f in required_fields if f not in inputs or not inputs[f]]
        if missing_fields:
            error_msg = f"Missing required fields: {', '.join(missing_fields)}"
            self.log_fn(error_msg)
            self.error_count += 1
            return f"Error: {error_msg}", []
        
        # Format inputs more clearly
        try:
            formatted_inputs = _format_inputs(inputs)
        except Exception as e:
            error_msg = f"Error formatting inputs: {e}"
            self.log_fn(error_msg)
            self.error_count += 1
            return f"Error: {error_msg}", []
        
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
            # Ensure we return a valid prediction even on failure
            if not prediction or prediction == "None":
                prediction = "Error: Failed to extract valid prediction"
        
        self.log_fn(f"TaskAgent stats: {self.success_count} successes, {self.error_count} errors, {self.retry_count} retries out of {self.call_count} calls")
        return prediction, msg_history

    def _try_call(self, formatted_inputs: str, is_retry: bool = False) -> tuple[str, list[dict], bool]:
        """Make a single LLM call and extract prediction with comprehensive error handling.
        
        Returns:
            (prediction, msg_history, success)
        """
        # Validate inputs
        if not formatted_inputs:
            self.log_fn("Empty formatted inputs")
            return "Error: Empty inputs", [], False
        
        instruction = self._build_instruction(formatted_inputs, is_retry)
        
        # Validate instruction was built
        if not instruction:
            self.log_fn("Failed to build instruction")
            return "Error: Failed to build instruction", [], False
        
        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
            
            # Validate response
            if not response:
                self.log_fn("Empty response from LLM")
                return "Error: Empty LLM response", msg_history, False
            
            self.log_fn(f"LLM call completed, response length: {len(response)}")
            
        except Exception as e:
            error_msg = str(e)
            self.log_fn(f"Error in LLM call: {error_msg}")
            # Categorize errors for better debugging
            if "timeout" in error_msg.lower():
                return f"Error: LLM timeout - {error_msg}", [], False
            elif "rate limit" in error_msg.lower() or "429" in error_msg:
                return f"Error: Rate limited - {error_msg}", [], False
            elif "authentication" in error_msg.lower() or "401" in error_msg or "403" in error_msg:
                return f"Error: Authentication failed - {error_msg}", [], False
            else:
                return f"Error: LLM call failed - {error_msg}", [], False

        # Extract prediction from JSON
        prediction, success = self._extract_prediction(msg_history)
        
        # Additional validation of prediction
        if success and prediction:
            # Check for common error indicators in prediction
            if prediction.startswith("Error:") or prediction == "None":
                self.log_fn(f"Prediction indicates error: {prediction}")
                return prediction, msg_history, False
        
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
