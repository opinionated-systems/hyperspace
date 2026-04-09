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
    
    Improved to handle:
    - Multiple JSON objects in a single <json> block
    - Nested JSON structures
    - Malformed JSON with recovery attempts
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
                    continue
            except Exception:
                pass
            
            # Try to find and parse multiple JSON objects in the content
            # This handles cases where the LLM put multiple objects in one block
            try:
                # Look for JSON objects using brace matching throughout the content
                i = 0
                while i < len(inner):
                    if inner[i] == '{':
                        brace_start = i
                        brace_count = 1
                        i += 1
                        while i < len(inner) and brace_count > 0:
                            if inner[i] == '{':
                                brace_count += 1
                            elif inner[i] == '}':
                                brace_count -= 1
                            i += 1
                        if brace_count == 0:
                            candidate = inner[brace_start:i]
                            try:
                                obj = json.loads(candidate)
                                results.append(obj)
                            except json.JSONDecodeError:
                                pass
                    else:
                        i += 1
            except Exception:
                continue
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
    """
    # Pattern to match "response": followed by a value (number, string, or boolean)
    # Handles: "response": 7, "response": "7", "response": true, etc.
    patterns = [
        # Number (integer or float)
        r'"response"\s*:\s*(-?\d+(?:\.\d+)?)',
        # String in double quotes
        r'"response"\s*:\s*"([^"]*)"',
        # String in single quotes
        r"'response'\s*:\s*'([^']*)'",
        # Boolean or null
        r'"response"\s*:\s*(true|false|null)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = match.group(1)
            # Try to convert to appropriate type
            try:
                # Try integer first
                return str(int(value))
            except ValueError:
                try:
                    # Try float
                    return str(float(value))
                except ValueError:
                    # Return as string
                    return value
    
    return None


def _normalize_prediction(prediction: str) -> str:
    """Normalize prediction value for consistent output.
    
    Handles various formats and ensures consistent string representation.
    Improved to handle edge cases and provide better type preservation.
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
    
    # Handle common grading terms
    if pred_lower in ('partial', 'partially correct', 'incomplete'):
        return "Partial"
    
    # Try to extract just the numeric part if there's extra text
    numeric_match = re.match(r'^(-?\d+(?:\.\d+)?)', pred_str)
    if numeric_match:
        return numeric_match.group(1)
    
    # Handle quoted strings by removing quotes
    if (pred_str.startswith('"') and pred_str.endswith('"')) or \
       (pred_str.startswith("'") and pred_str.endswith("'")):
        return pred_str[1:-1]
    
    return pred_str


def _is_valid_prediction(prediction: str) -> bool:
    """Check if a prediction value is valid.
    
    A valid prediction is either:
    - A numeric value (integer or float)
    - A recognized boolean/evaluation string
    - A non-empty string that isn't "None"
    """
    if prediction is None:
        return False
    
    pred_str = str(prediction).strip()
    
    # Check for empty or None
    if not pred_str or pred_str == "None":
        return False
    
    # Check if it's a valid number
    try:
        float(pred_str)
        return True
    except ValueError:
        pass
    
    # Check if it's a recognized evaluation term
    valid_terms = {'correct', 'incorrect', 'partial', 'true', 'false', 
                 'yes', 'no', 'right', 'wrong', 'partially correct', 
                 'incomplete', '0', '1'}
    if pred_str.lower() in valid_terms:
        return True
    
    # Any non-empty string is considered valid
    return len(pred_str) > 0


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
        
        # Track extraction attempts for better debugging
        extraction_results = []
        
        extraction_attempts = [
            ("_extract_jsons", lambda: _extract_jsons(raw_text)),
            ("_extract_json_with_regex", lambda: _extract_json_with_regex(raw_text)),
        ]
        
        for attempt_name, attempt_fn in extraction_attempts:
            try:
                extracted = attempt_fn()
                extraction_results.append((attempt_name, extracted))
                if extracted and len(extracted) > 0:
                    # Check the last extracted JSON object for "response" key
                    last_obj = extracted[-1]
                    if isinstance(last_obj, dict) and "response" in last_obj:
                        prediction = last_obj["response"]
                        self.log_fn(f"Successfully extracted prediction using {attempt_name}: {prediction}")
                        break
            except Exception as e:
                extraction_results.append((attempt_name, f"Error: {e}"))
                self.log_fn(f"Extraction attempt {attempt_name} failed: {e}")
                continue
        
        # Last resort: try to extract response value directly from malformed JSON
        if prediction == "None":
            direct_value = _extract_response_value(raw_text)
            if direct_value is not None:
                prediction = direct_value
                self.log_fn(f"Extracted prediction using direct value extraction: {prediction}")
            else:
                extraction_results.append(("_extract_response_value", "No match found"))
        
        if prediction == "None":
            self.log_fn(f"Failed to extract prediction from response. Raw text: {raw_text[:500]}")
            self.log_fn(f"Extraction attempts summary: {extraction_results}")

        # Normalize the prediction for consistent output
        normalized_prediction = _normalize_prediction(prediction)
        if normalized_prediction != str(prediction):
            self.log_fn(f"Normalized prediction from '{prediction}' to '{normalized_prediction}'")
        
        # Validate the final prediction
        if not _is_valid_prediction(normalized_prediction):
            self.log_fn(f"Warning: Invalid prediction '{normalized_prediction}' detected, defaulting to 'None'")
            normalized_prediction = "None"
        
        return normalized_prediction, msg_history
