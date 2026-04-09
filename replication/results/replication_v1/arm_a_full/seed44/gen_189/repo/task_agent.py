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
    """Extract JSON objects from <json>...</json> blocks and markdown code blocks.
    
    Uses a unified approach to find and parse JSON objects from various formats.
    Handles nested braces properly using brace counting for malformed JSON.
    
    Args:
        text: The text to extract JSON objects from.
        
    Returns:
        A list of parsed JSON objects, or None if no valid JSON found.
    """
    results = []
    
    # Try <json>...</json> blocks first (preferred format)
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
        
        try:
            results.append(json.loads(inner))
        except json.JSONDecodeError:
            # Fallback: try brace matching for malformed JSON
            json_obj = _extract_json_with_brace_matching(inner)
            if json_obj:
                results.append(json_obj)
    
    # Try markdown code blocks if no <json> blocks found
    if not results:
        # Match ```json ... ``` or just ``` ... ```
        code_pattern = r'```(?:json)?\s*\n(.*?)\n?```'
        for match in re.finditer(code_pattern, text, re.DOTALL):
            content = match.group(1).strip()
            try:
                results.append(json.loads(content))
            except json.JSONDecodeError:
                json_obj = _extract_json_with_brace_matching(content)
                if json_obj:
                    results.append(json_obj)
    
    return results if results else None


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
    """Fallback JSON extraction for malformed outputs.
    
    Uses brace counting to find complete JSON objects, then filters for those
    containing a "response" key. This handles nested structures properly.
    
    Args:
        text: The text to search for JSON objects.
        
    Returns:
        A list of parsed JSON objects, or None if no valid JSON found.
    """
    results = []
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
                if '"response"' in candidate or "'response'" in candidate:
                    try:
                        results.append(json.loads(candidate))
                    except json.JSONDecodeError:
                        pass
        else:
            i += 1
    
    return results if results else None


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

Your task is to grade a student's answer to an IMO-level mathematics problem. Grade strictly according to the provided grading guidelines.

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

GRADING INSTRUCTIONS:
1. First, identify what the problem is asking for and what constitutes a complete solution.
2. Study the official solution to understand the key insights and required steps.
3. Carefully review the grading guidelines - these define exactly how points are awarded.
4. Analyze the student's answer against the official solution:
   - Check if they stated the correct answer
   - Verify their approach matches the official solution or is mathematically valid
   - Check each step for correctness
   - Identify any missing steps or logical gaps
   - Note any calculation errors
5. Assign a score STRICTLY according to the grading guidelines.

IMPORTANT RULES:
- Award partial credit only when explicitly allowed by the grading guidelines
- If the guidelines specify point deductions for certain errors, apply them consistently
- A complete, correct solution should receive full marks
- An empty or completely incorrect answer should receive 0

You MUST respond with a valid JSON object wrapped in <json> tags. No text outside the tags.

Format:
<json>
{{
    "response": <numerical score>,
    "confidence": <0.0-1.0>,
    "reasoning": "<1-2 sentence explanation>"
}}
</json>

Examples:
<json>{{"response": 7, "confidence": 0.95, "reasoning": "Complete solution with correct proof matching official solution"}}</json>
<json>{{"response": 3, "confidence": 0.8, "reasoning": "Correct approach but missing final step and one calculation error"}}</json>
<json>{{"response": 0, "confidence": 0.9, "reasoning": "Incorrect approach, failed to use key insight from official solution"}}</json>"""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON with multiple fallback strategies
        prediction = "None"
        confidence = None
        reasoning = None
        
        # Validate msg_history has at least one entry
        if not msg_history or len(msg_history) == 0:
            self.log_fn("Error: Empty message history returned from LLM")
            return "None", msg_history
        
        raw_text = msg_history[-1].get("text", "")
        
        # Log the raw response for debugging
        self.log_fn(f"Raw LLM response (first 500 chars): {raw_text[:500]}")
        
        if not raw_text:
            self.log_fn("Error: Empty response text from LLM")
            return "None", msg_history
        
        # Primary extraction: try _extract_jsons first (handles <json> tags and code blocks)
        try:
            extracted = _extract_jsons(raw_text)
            if extracted:
                for obj in extracted:
                    if isinstance(obj, dict) and "response" in obj:
                        prediction = obj["response"]
                        confidence = obj.get("confidence")
                        reasoning = obj.get("reasoning")
                        self.log_fn(f"Extracted prediction via _extract_jsons: {prediction}")
                        break
        except Exception as e:
            self.log_fn(f"Primary extraction failed: {e}")
        
        # Fallback: try regex-based extraction
        if prediction == "None":
            try:
                extracted = _extract_json_with_regex(raw_text)
                if extracted:
                    for obj in extracted:
                        if isinstance(obj, dict) and "response" in obj:
                            prediction = obj["response"]
                            confidence = obj.get("confidence")
                            reasoning = obj.get("reasoning")
                            self.log_fn(f"Extracted prediction via _extract_json_with_regex: {prediction}")
                            break
            except Exception as e:
                self.log_fn(f"Regex extraction failed: {e}")
        
        # Last resort: direct value extraction
        if prediction == "None":
            direct_value = _extract_response_value(raw_text)
            if direct_value is not None:
                prediction = direct_value
                self.log_fn(f"Extracted prediction via direct value extraction: {prediction}")
        
        if prediction == "None":
            self.log_fn(f"Failed to extract prediction. Raw text preview: {raw_text[:300]}")

        # Normalize the prediction
        normalized_prediction = _normalize_prediction(prediction)
        self.log_fn(f"Final prediction: {normalized_prediction}")
        
        # Store metadata in msg_history
        if msg_history:
            msg_history[-1]["_grading_metadata"] = {
                "confidence": confidence,
                "reasoning": reasoning,
            }
        
        return normalized_prediction, msg_history
