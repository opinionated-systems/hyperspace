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
    
    Enhanced to handle multiple JSON formats and edge cases.
    """
    results = []
    search_from = 0
    
    # Normalize common variations of json tags
    text = text.replace("<JSON>", "<json>").replace("</JSON>", "</json>")
    # Only replace ```json blocks, not all ``` (which could be closing other code blocks)
    import re
    text = re.sub(r'```json\s*\n', '<json>', text)
    text = re.sub(r'\n```\s*$', '</json>', text)
    text = re.sub(r'</json>\s*</json>', '</json>', text)
    
    while True:
        start = text.find("<json>", search_from)
        if start == -1:
            break
        end = text.find("</json>", start)
        if end == -1:
            break
        inner = text[start + 6:end].strip()
        search_from = end + 7
        
        # Skip empty content
        if not inner:
            continue
        
        # Try to parse the inner content as JSON
        try:
            results.append(json.loads(inner))
        except json.JSONDecodeError:
            # If direct parsing fails, try to extract JSON using brace matching
            try:
                json_obj = _extract_json_with_brace_matching(inner)
                if json_obj:
                    results.append(json_obj)
            except Exception:
                # Try to clean up common JSON issues
                try:
                    cleaned = _clean_json_string(inner)
                    if cleaned:
                        results.append(json.loads(cleaned))
                except Exception:
                    continue
    return results or None


def _clean_json_string(text: str) -> str | None:
    """Clean up common JSON formatting issues.
    
    Handles trailing commas, single quotes, and other common issues.
    """
    if not text or not text.strip():
        return None
    
    # Remove leading/trailing whitespace and newlines
    text = text.strip()
    
    # Find the JSON object boundaries
    start = text.find('{')
    end = text.rfind('}')
    
    if start == -1 or end == -1 or start > end:
        return None
    
    text = text[start:end+1]
    
    # Remove trailing commas before closing braces/brackets
    import re
    # Remove trailing commas before } or ]
    text = re.sub(r',(\s*[}\]])', r'\1', text)
    
    # Replace single quotes with double quotes for keys and string values
    # This is a simplified approach - handles simple cases
    text = re.sub(r"'([^']*?)'\s*:", r'"\1":', text)  # keys
    text = re.sub(r":\s*'([^']*?)'", r': "\1"', text)  # values
    
    return text


def _extract_json_with_brace_matching(text: str) -> dict | None:
    """Extract a JSON object from text using brace counting.
    
    This handles nested JSON objects properly by tracking brace depth.
    Also handles string literals to avoid counting braces inside strings.
    """
    start_idx = text.find('{')
    if start_idx == -1:
        return None
    
    brace_count = 1
    i = start_idx + 1
    in_string = False
    escape_next = False
    
    while i < len(text) and brace_count > 0:
        char = text[i]
        
        if escape_next:
            escape_next = False
            i += 1
            continue
        
        if char == '\\' and in_string:
            escape_next = True
            i += 1
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
        
        i += 1
    
    if brace_count == 0:
        candidate = text[start_idx:i]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            # Try cleaning the candidate before giving up
            try:
                cleaned = _clean_json_string(candidate)
                if cleaned:
                    return json.loads(cleaned)
            except Exception:
                pass
    return None


def _extract_json_with_regex(text: str) -> list[dict] | None:
    """Fallback JSON extraction using regex for malformed outputs.
    
    Handles nested JSON objects by using a stack-based brace matching approach
    instead of simple regex that fails on nested structures.
    Also handles string literals to avoid counting braces inside strings.
    """
    results = []
    # Try to find JSON objects in code blocks
    code_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    matches = re.findall(code_block_pattern, text, re.DOTALL)
    for match in matches:
        try:
            results.append(json.loads(match.strip()))
        except json.JSONDecodeError:
            # Try cleaning before giving up
            try:
                cleaned = _clean_json_string(match.strip())
                if cleaned:
                    results.append(json.loads(cleaned))
            except Exception:
                continue
    
    # Try to find JSON objects with curly braces using stack-based matching
    # This handles nested objects properly, respecting string literals
    if not results:
        i = 0
        while i < len(text):
            # Find the start of a potential JSON object
            if text[i] == '{':
                start = i
                brace_count = 1
                i += 1
                in_string = False
                escape_next = False
                
                # Track braces to find the matching closing brace
                while i < len(text) and brace_count > 0:
                    char = text[i]
                    
                    if escape_next:
                        escape_next = False
                        i += 1
                        continue
                    
                    if char == '\\' and in_string:
                        escape_next = True
                        i += 1
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
                    
                    i += 1
                
                # If we found a complete object with "response" key, try to parse it
                if brace_count == 0:
                    candidate = text[start:i]
                    if '"response"' in candidate or "'response'" in candidate:
                        try:
                            results.append(json.loads(candidate))
                        except json.JSONDecodeError:
                            # Try cleaning before giving up
                            try:
                                cleaned = _clean_json_string(candidate)
                                if cleaned:
                                    results.append(json.loads(cleaned))
                            except Exception:
                                continue
            else:
                i += 1
    
    return results or None


def _extract_response_value(text: str) -> str | None:
    """Last-resort extraction: try to find a response value directly.
    
    This handles cases where the JSON is malformed but we can still
    extract the value after "response": using regex patterns.
    Enhanced to handle more edge cases and provide better logging.
    """
    if not text or not text.strip():
        return None
    
    # Pattern to match "response": followed by a value (number, string, or boolean)
    # Handles: "response": 7, "response": "7", "response": true, etc.
    # Enhanced patterns to handle more edge cases
    patterns = [
        # Number (integer or float) - with optional whitespace
        (r'"response"\s*:\s*(-?\d+(?:\.\d+)?)', 'number'),
        # String in double quotes - handle escaped quotes
        (r'"response"\s*:\s*"((?:[^"\\]|\\.)*)"', 'string_double'),
        # String in single quotes
        (r"'response'\s*:\s*'([^']*)'", 'string_single'),
        # Boolean or null
        (r'"response"\s*:\s*(true|false|null)', 'boolean'),
        # Alternative: response as key without quotes
        (r'response\s*:\s*(-?\d+(?:\.\d+)?)', 'number_noquotes'),
        (r'response\s*:\s*"([^"]*)"', 'string_noquotes'),
    ]
    
    for pattern, pattern_type in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = match.group(1)
            
            # Handle escaped characters in strings
            if pattern_type in ('string_double', 'string_single'):
                value = value.replace('\\"', '"').replace("\\'", "'")
            
            # Try to convert to appropriate type
            try:
                # Try integer first
                return str(int(value))
            except ValueError:
                try:
                    # Try float
                    return str(float(value))
                except ValueError:
                    # Return as string (lowercase booleans)
                    if value.lower() in ('true', 'false', 'null'):
                        return value.lower()
                    # Return as string
                    return value
    
    return None


def _normalize_prediction(prediction: str) -> str:
    """Normalize prediction value for consistent output.
    
    Handles various formats and ensures consistent string representation.
    Enhanced to handle more edge cases and provide better type conversion.
    """
    if prediction is None:
        return "None"
    
    # Convert to string if not already
    pred_str = str(prediction).strip()
    
    # Handle empty strings
    if not pred_str:
        return "None"
    
    # Handle boolean strings (case-insensitive)
    pred_lower = pred_str.lower()
    if pred_lower in ('true', 'yes', 'correct', '1', 'right'):
        return "Correct"
    if pred_lower in ('false', 'no', 'incorrect', '0', 'wrong'):
        return "Incorrect"
    
    # Try to extract just the numeric part if there's extra text
    numeric_match = re.match(r'^(-?\d+(?:\.\d+)?)', pred_str)
    if numeric_match:
        num_str = numeric_match.group(1)
        # Remove trailing .0 for integers
        if '.' in num_str:
            try:
                float_val = float(num_str)
                if float_val == int(float_val):
                    return str(int(float_val))
            except ValueError:
                pass
        return num_str
    
    # Handle common grade formats (e.g., "7/7", "5 points")
    grade_match = re.match(r'^(\d+(?:\.\d+)?)\s*(?:/\s*\d+|\s*points?)?$', pred_str, re.IGNORECASE)
    if grade_match:
        return grade_match.group(1)
    
    return pred_str


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

GRADING PRINCIPLES:
- Be precise and follow the grading guidelines exactly
- Award partial credit when the student shows meaningful progress
- Deduct points for logical gaps, missing cases, or incorrect conclusions
- Consider alternative valid approaches that differ from the official solution
- Check for complete proofs - incomplete reasoning should receive partial credit

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

FINAL CHECK: Before responding, verify that:
1. Your JSON is valid and parseable
2. The "response" field contains only the final answer (number, string, or boolean)
3. There are no extra fields or comments in the JSON
4. The JSON is properly wrapped in <json>...</json> tags"""

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
        
        extraction_attempts = [
            ("_extract_jsons", lambda: _extract_jsons(raw_text)),
            ("_extract_json_with_regex", lambda: _extract_json_with_regex(raw_text)),
        ]
        
        for attempt_name, attempt_fn in extraction_attempts:
            try:
                extracted = attempt_fn()
                if extracted and len(extracted) > 0:
                    # Check the last extracted JSON object for "response" key
                    last_obj = extracted[-1]
                    if isinstance(last_obj, dict) and "response" in last_obj:
                        prediction = last_obj["response"]
                        self.log_fn(f"Successfully extracted prediction using {attempt_name}: {prediction}")
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

        # Normalize the prediction for consistent output
        normalized_prediction = _normalize_prediction(prediction)
        if normalized_prediction != str(prediction):
            self.log_fn(f"Normalized prediction from '{prediction}' to '{normalized_prediction}'")
        
        return normalized_prediction, msg_history
