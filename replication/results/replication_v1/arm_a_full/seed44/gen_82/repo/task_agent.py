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
    
    Improved to properly handle boolean and null values, and to extract
    values from nested JSON structures.
    """
    # Pattern to match "response": followed by a value (number, string, or boolean)
    # Handles: "response": 7, "response": "7", "response": true, etc.
    patterns = [
        # String in double quotes (check first to avoid partial matches)
        (r'"response"\s*:\s*"([^"]*)"', 'string'),
        # String in single quotes
        (r"'response'\s*:\s*'([^']*)'", 'string'),
        # Number (integer or float) - must be a complete number, not part of another
        (r'"response"\s*:\s*(-?\d+(?:\.\d+)?)(?:\s*[,}\]])', 'number'),
        # Boolean or null
        (r'"response"\s*:\s*(true|false|null)(?:\s*[,}\]])', 'literal'),
    ]
    
    for pattern, value_type in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = match.group(1)
            
            if value_type == 'string':
                # Return string values directly
                return value
            elif value_type == 'literal':
                # Return boolean/null values capitalized
                return value.lower()
            elif value_type == 'number':
                # Try to convert to appropriate numeric type
                try:
                    # Try integer first
                    if '.' not in value:
                        return str(int(value))
                    else:
                        return str(float(value))
                except ValueError:
                    return value
    
    # Additional fallback: try to find any JSON-like structure with a response field
    # This handles cases where the JSON might be embedded in other text
    fallback_pattern = r'[\s\S]*?"response"\s*:\s*([^,}\]]+)'
    match = re.search(fallback_pattern, text, re.IGNORECASE)
    if match:
        value = match.group(1).strip()
        # Remove quotes if present
        if (value.startswith('"') and value.endswith('"')) or \
           (value.startswith("'") and value.endswith("'")):
            value = value[1:-1]
        return value if value else None
    
    return None


def _normalize_prediction(prediction: str) -> str:
    """Normalize prediction value for consistent output.
    
    Handles various formats and ensures consistent string representation.
    Specifically optimized for IMO grading scenarios with numeric scores
    and boolean evaluations.
    """
    if prediction is None:
        return "None"
    
    # Convert to string if not already
    pred_str = str(prediction).strip()
    
    # Handle empty strings
    if not pred_str:
        return "None"
    
    # Handle boolean strings (case-insensitive)
    bool_true = ('true', 'yes', 'correct', 'right', 'valid', 'pass', 'success')
    bool_false = ('false', 'no', 'incorrect', 'wrong', 'invalid', 'fail', 'failure')
    
    pred_lower = pred_str.lower()
    if pred_lower in bool_true:
        return "Correct"
    if pred_lower in bool_false:
        return "Incorrect"
    
    # Try to extract just the numeric part if there's extra text
    # This handles cases like "7 points" or "Score: 5"
    numeric_match = re.search(r'(-?\d+(?:\.\d+)?)', pred_str)
    if numeric_match:
        num_str = numeric_match.group(1)
        # Normalize numeric format (remove trailing .0 for integers)
        try:
            if '.' in num_str:
                float_val = float(num_str)
                if float_val.is_integer():
                    return str(int(float_val))
            return num_str
        except ValueError:
            pass
    
    # Handle common IMO grading strings
    imo_grades = {
        '0': '0', '1': '1', '2': '2', '3': '3', '4': '4',
        '5': '5', '6': '6', '7': '7',
        'full': '7', 'full marks': '7', 'full score': '7',
        'zero': '0', 'none': '0', 'no points': '0',
        'partial': 'Partial', 'incomplete': 'Partial',
    }
    
    if pred_lower in imo_grades:
        return imo_grades[pred_lower]
    
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

IMPORTANT: Your response MUST be a valid JSON object wrapped in <json> tags. Do not include any text outside the JSON tags.

Respond in the following exact format:
<json>
{{
    "response": <your numerical score or evaluation>
}}
</json>

The "response" field should contain your final grading decision (typically a number or specific evaluation string as specified in the grading guidelines).

Examples of valid responses:
- For a numeric score (IMO 0-7 scale): <json>{{"response": 7}}</json>
- For a numeric score with partial credit: <json>{{"response": 3}}</json>
- For a string evaluation: <json>{{"response": "Correct"}}</json>
- For a boolean: <json>{{"response": true}}</json>
- For zero score: <json>{{"response": 0}}</json>

CRITICAL RULES:
1. The response value must be a single value (number, string, or boolean)
2. Do NOT include explanations, reasoning, or comments inside the JSON
3. Do NOT use quotes around numbers - use 7 not "7"
4. Ensure your JSON is properly formatted with no trailing commas
5. The JSON must be parseable by standard JSON parsers

FINAL CHECK: Before responding, verify that:
1. Your JSON is valid and parseable
2. The "response" field contains only the final answer (number, string, or boolean)
3. There are no extra fields or comments in the JSON
4. The JSON is properly wrapped in <json>...</json> tags
5. No text appears before <json> or after </json>"""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON with multiple fallback strategies
        prediction = "None"
        raw_text = msg_history[-1]["text"]
        
        # Log the raw response for debugging (truncated if very long)
        log_length = min(1500, len(raw_text))
        self.log_fn(f"Raw LLM response (first {log_length} chars): {raw_text[:log_length]}")
        
        # Check if response contains <json> tags at all
        has_json_tags = "<json>" in raw_text and "</json>" in raw_text
        if not has_json_tags:
            self.log_fn("Warning: Response does not contain <json>...</json> tags")
        
        extraction_attempts = [
            ("_extract_jsons", lambda: _extract_jsons(raw_text)),
            ("_extract_json_with_regex", lambda: _extract_json_with_regex(raw_text)),
        ]
        
        extraction_success = False
        for attempt_name, attempt_fn in extraction_attempts:
            try:
                extracted = attempt_fn()
                if extracted and len(extracted) > 0:
                    # Check all extracted JSON objects for "response" key
                    # Prefer the last one as it's likely the final answer
                    for obj in reversed(extracted):
                        if isinstance(obj, dict) and "response" in obj:
                            prediction = obj["response"]
                            # Handle None response value
                            if prediction is None:
                                prediction = "null"
                            self.log_fn(f"Successfully extracted prediction using {attempt_name}: {prediction}")
                            extraction_success = True
                            break
                    if extraction_success:
                        break
            except Exception as e:
                self.log_fn(f"Extraction attempt {attempt_name} failed: {e}")
                continue
        
        # Last resort: try to extract response value directly from malformed JSON
        if not extraction_success or prediction in ("None", "null"):
            direct_value = _extract_response_value(raw_text)
            if direct_value is not None:
                prediction = direct_value
                self.log_fn(f"Extracted prediction using direct value extraction: {prediction}")
                extraction_success = True
        
        if not extraction_success:
            self.log_fn(f"Failed to extract prediction from response. Raw text preview: {raw_text[:500]}")
            # Try to find any numeric value that might be a score
            import re as _re
            score_match = _re.search(r'\b([0-7])\b', raw_text)
            if score_match:
                potential_score = score_match.group(1)
                self.log_fn(f"Found potential score in response: {potential_score}")
                prediction = potential_score

        # Normalize the prediction for consistent output
        normalized_prediction = _normalize_prediction(prediction)
        if normalized_prediction != str(prediction):
            self.log_fn(f"Normalized prediction from '{prediction}' to '{normalized_prediction}'")
        
        # Final validation for IMO grading context
        if normalized_prediction not in ("0", "1", "2", "3", "4", "5", "6", "7", 
                                          "Correct", "Incorrect", "Partial", "None"):
            self.log_fn(f"Warning: Unusual prediction value '{normalized_prediction}' for IMO grading")
        
        return normalized_prediction, msg_history
