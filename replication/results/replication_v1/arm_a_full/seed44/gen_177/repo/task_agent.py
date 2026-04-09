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

from agent.llm_client import get_response_from_llm, EVAL_MODEL

logger = logging.getLogger(__name__)


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    Also handles nested JSON objects within the content.
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
        
        # Try to parse the inner content as JSON
        try:
            results.append(json.loads(inner))
        except json.JSONDecodeError:
            # If direct parsing fails, try to extract JSON using brace matching
            # This handles cases where there might be nested structures
            try:
                json_obj = _extract_json_with_brace_matching(inner)
                if json_obj:
                    results.append(json_obj)
            except Exception:
                continue
    
    # Also try to find JSON objects without <json> tags (fallback)
    if not results:
        try:
            json_obj = _extract_json_with_brace_matching(text)
            if json_obj and "response" in json_obj:
                results.append(json_obj)
        except Exception:
            pass
    
    return results or None


def _extract_json_with_brace_matching(text: str) -> dict | None:
    """Extract a JSON object from text using brace counting.
    
    This handles nested JSON objects properly by tracking brace depth.
    """
    start_idx = text.find('{')
    if start_idx == -1:
        return None
    
    brace_count = 1
    i = start_idx + 1
    while i < len(text) and brace_count > 0:
        if text[i] == '{':
            brace_count += 1
        elif text[i] == '}':
            brace_count -= 1
        i += 1
    
    if brace_count == 0:
        candidate = text[start_idx:i]
        try:
            return json.loads(candidate)
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
    Improved to properly distinguish between numeric and string values.
    """
    # Pattern to match "response": followed by a value (number, string, or boolean)
    # Handles: "response": 7, "response": "7", "response": true, etc.
    # Order matters: try string patterns first to avoid misinterpreting string numbers as numeric
    patterns = [
        # String in double quotes (check first to avoid misinterpreting "7" as number 7)
        (r'["\']?response["\']?\s*:\s*"([^"]*)"', 'string_double'),
        # String in single quotes
        (r"['\"]?response['\"]?\s*:\s*'([^']*)'", 'string_single'),
        # Boolean or null (check before numbers to avoid partial matches)
        (r'["\']?response["\']?\s*:\s*(true|false|null)\b', 'boolean'),
        # Number (integer or float) - with optional quotes around the key
        # Use negative lookahead to ensure we're not matching inside a string
        (r'["\']?response["\']?\s*:\s*(-?\d+(?:\.\d+)?)\b', 'number'),
    ]
    
    logger.debug(f"Attempting direct response extraction from text of length {len(text)}")
    
    for pattern, pattern_type in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = match.group(1)
            logger.debug(f"Matched pattern '{pattern_type}': {value}")
            
            # Handle string values first (preserve them as-is)
            if pattern_type in ('string_double', 'string_single'):
                # For string values, check if they look like booleans
                if value.lower() in ('true', 'false', 'null'):
                    logger.debug(f"String value looks like boolean/null: {value}")
                    return value.lower()
                # Check if it's a numeric string that should stay as string
                if value.strip() == value and value:  # Non-empty, no extra whitespace
                    logger.debug(f"Returning string value as-is: {value}")
                    return value
                # Empty string is valid
                logger.debug(f"Returning empty or whitespace string: '{value}'")
                return value
            
            # Handle boolean/null values
            if pattern_type == 'boolean':
                logger.debug(f"Returning boolean/null as string: {value}")
                return value.lower()
            
            # Handle numeric values (only if explicitly numeric pattern matched)
            if pattern_type == 'number':
                try:
                    # Try integer first
                    int_val = int(value)
                    # Check if it was actually a float that looks like int
                    if '.' in value:
                        float_val = float(value)
                        if float_val != int_val:
                            result = str(float_val)
                            logger.debug(f"Converted to float: {result}")
                            return result
                    result = str(int_val)
                    logger.debug(f"Converted to integer: {result}")
                    return result
                except ValueError:
                    try:
                        result = str(float(value))
                        logger.debug(f"Converted to float: {result}")
                        return result
                    except ValueError:
                        # Shouldn't happen with valid regex match, but handle gracefully
                        logger.debug(f"Regex matched but conversion failed, returning as string: {value}")
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
        
        # Handle special string values first
        lower_cleaned = cleaned.lower()
        if lower_cleaned in ("true", "false", "null", "none"):
            return lower_cleaned
        
        # Try to normalize numeric strings, but be careful about precision
        # Only convert if it looks like a pure number (not something like "7 points")
        if cleaned and not any(c.isalpha() for c in cleaned):
            try:
                # Check if it's a valid numeric format
                num_val = float(cleaned)
                # Check if it was originally an integer representation
                if '.' not in cleaned and 'e' not in cleaned.lower():
                    return str(int(num_val))
                # For floats, preserve precision but remove trailing zeros
                if num_val.is_integer():
                    return str(int(num_val))
                return str(num_val).rstrip('0').rstrip('.') if '.' in str(num_val) else str(num_val)
            except ValueError:
                pass
        
        # Return cleaned string as-is (preserving case for non-special values)
        return cleaned
    
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

IMPORTANT: Your response MUST be a valid JSON object wrapped in <json> tags. Do not include any text outside the JSON tags.

Respond in the following exact format:
<json>
{{
    "response": <your numerical score or evaluation>
}}
</json>

The "response" field should contain your final grading decision (typically a number or specific evaluation string as specified in the grading guidelines).

Examples of valid responses:
- For a numeric score: <json>{{"response": 7}}</json>
- For a string evaluation: <json>{{"response": "Correct"}}</json>
- For a boolean: <json>{{"response": true}}</json>

Ensure your JSON is properly formatted with no trailing commas or syntax errors.

CRITICAL: Only output the JSON object wrapped in <json> tags. Do not include any explanations, reasoning, or additional text before or after the JSON."""

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
        
        # First, try to find JSON within <json> tags (most common format)
        extraction_attempts = [
            ("_extract_jsons", lambda: _extract_jsons(raw_text)),
            ("_extract_json_with_regex", lambda: _extract_json_with_regex(raw_text)),
        ]
        
        for attempt_name, attempt_fn in extraction_attempts:
            try:
                extracted = attempt_fn()
                if extracted and len(extracted) > 0:
                    # Check all extracted JSON objects for "response" key, preferring the last one
                    for obj in reversed(extracted):
                        if isinstance(obj, dict) and "response" in obj:
                            prediction = obj["response"]
                            self.log_fn(f"Successfully extracted prediction using {attempt_name}: {prediction}")
                            break
                    if prediction != "None":
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
        
        return normalized_prediction, msg_history
