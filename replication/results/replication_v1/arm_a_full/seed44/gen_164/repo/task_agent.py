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
    Enhanced to handle markdown code blocks and various formatting.
    
    Args:
        text: The text to extract JSON objects from.
        
    Returns:
        A list of parsed JSON objects, or None if no valid JSON found.
    """
    results = []
    search_from = 0
    
    # First try to find <json>...</json> blocks
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
    
    # If no <json> blocks found, try markdown code blocks with json tag
    if not results:
        json_code_pattern = r'```json\s*\n(.*?)\n```'
        matches = re.findall(json_code_pattern, text, re.DOTALL)
        for match in matches:
            try:
                results.append(json.loads(match.strip()))
            except json.JSONDecodeError:
                # Try brace matching as fallback
                try:
                    json_obj = _extract_json_with_brace_matching(match)
                    if json_obj:
                        results.append(json_obj)
                except Exception:
                    continue
    
    return results or None


def _extract_json_with_brace_matching(text: str) -> dict | None:
    """Extract a JSON object from text using brace counting.
    
    This handles nested JSON objects properly by tracking brace depth.
    Uses a stack-based approach to find matching braces, which correctly
    handles nested structures that would break simple regex patterns.
    
    Args:
        text: The text to search for JSON objects.
        
    Returns:
        A parsed JSON dict if found and valid, otherwise None.
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
    
    This function is used as a last resort when standard extraction methods fail.
    It searches for JSON objects in markdown code blocks and then falls back
    to finding JSON objects with curly braces that contain a "response" key.
    
    Args:
        text: The text to search for JSON objects.
        
    Returns:
        A list of parsed JSON objects, or None if no valid JSON found.
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
    Fixed to properly handle decimal numbers without truncating.
    
    Args:
        text: The text to search for a response value.
        
    Returns:
        The extracted response value as a string, or None if not found.
    """
    # Pattern to match "response": followed by a value (number, string, or boolean)
    # Handles: "response": 7, "response": 7.5, "response": "7", "response": true, etc.
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
                # Check if it's a float first (has decimal point)
                if '.' in value:
                    result = str(float(value))
                    logger.debug(f"Converted to float: {result}")
                    return result
                # Otherwise try integer
                result = str(int(value))
                logger.debug(f"Converted to integer: {result}")
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
    handling various numeric and string representations. The function
    handles edge cases like None values, boolean representations,
    numeric strings, and special string values.
    
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
        if not cleaned:
            return "None"
        
        # Try to normalize numeric strings
        try:
            num_val = float(cleaned)
            if num_val.is_integer():
                return str(int(num_val))
            return str(num_val)
        except ValueError:
            # Not a number, return cleaned string
            lower_cleaned = cleaned.lower()
            if lower_cleaned in ("true", "false", "null", "none"):
                return lower_cleaned
            return cleaned
    
    # Default: convert to string
    return str(prediction)


class TaskAgent:
    """Task agent that solves IMO grading problems.
    
    This agent evaluates student solutions to competition mathematics problems
    by comparing them against official solutions and grading guidelines.
    It uses a language model to analyze the student's work and produce
    a numerical or categorical grade.
    
    Attributes:
        model: The model identifier to use for LLM calls.
        log_fn: The logging function for recording agent activity.
    """

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
        
        # Validate required inputs
        if not problem:
            self.log_fn("Warning: Empty problem statement provided")
        if not student_answer:
            self.log_fn("Warning: Empty student answer provided")

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
    "response": <your numerical score or evaluation>,
    "confidence": <number between 0.0 and 1.0>,
    "reasoning": "<brief explanation of your grading decision>"
}}
</json>

The fields should contain:
- "response": Your final grading decision (typically a number or specific evaluation string as specified in the grading guidelines)
- "confidence": A number between 0.0 and 1.0 indicating your confidence in this grade (1.0 = very confident, 0.0 = not confident at all)
- "reasoning": A brief explanation (1-2 sentences) of why you assigned this grade

Examples of valid responses:
- For a numeric score: <json>{{"response": 7, "confidence": 0.95, "reasoning": "Complete solution with correct proof"}}</json>
- For a string evaluation: <json>{{"response": "Correct", "confidence": 0.9, "reasoning": "All steps are valid"}}</json>
- For a boolean: <json>{{"response": true, "confidence": 0.85, "reasoning": "Correct approach demonstrated"}}</json>

Ensure your JSON is properly formatted with no trailing commas or syntax errors."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON with multiple fallback strategies
        prediction = "None"
        
        # Validate msg_history has at least one entry
        if not msg_history or len(msg_history) == 0:
            self.log_fn("Error: Empty message history returned from LLM")
            return "None", msg_history
        
        raw_text = msg_history[-1].get("text", "")
        
        # Log the raw response for debugging
        self.log_fn(f"Raw LLM response (first 1000 chars): {raw_text[:1000]}")
        
        if not raw_text:
            self.log_fn("Error: Empty response text from LLM")
            return "None", msg_history
        
        extraction_attempts = [
            ("_extract_jsons", lambda: _extract_jsons(raw_text)),
            ("_extract_json_with_regex", lambda: _extract_json_with_regex(raw_text)),
        ]
        
        confidence = None
        reasoning = None
        
        for attempt_name, attempt_fn in extraction_attempts:
            try:
                extracted = attempt_fn()
                if extracted and len(extracted) > 0:
                    # Check the last extracted JSON object for "response" key
                    last_obj = extracted[-1]
                    if isinstance(last_obj, dict) and "response" in last_obj:
                        prediction = last_obj["response"]
                        confidence = last_obj.get("confidence")
                        reasoning = last_obj.get("reasoning")
                        self.log_fn(f"Successfully extracted prediction using {attempt_name}: {prediction}")
                        if confidence is not None:
                            self.log_fn(f"Confidence: {confidence}")
                        if reasoning is not None:
                            self.log_fn(f"Reasoning: {reasoning}")
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
        
        # Store additional metadata in msg_history for potential downstream use
        if msg_history and len(msg_history) > 0:
            msg_history[-1]["_grading_metadata"] = {
                "confidence": confidence,
                "reasoning": reasoning,
            }
        
        return normalized_prediction, msg_history
