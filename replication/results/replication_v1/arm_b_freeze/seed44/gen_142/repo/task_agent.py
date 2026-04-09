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
    Also handles cases with comments or trailing content after the JSON.
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
        
        # Try to parse the inner content directly
        try:
            results.append(json.loads(inner))
            continue
        except json.JSONDecodeError:
            pass
        
        # Try to extract just the JSON object, ignoring trailing comments/text
        # Look for the last complete JSON object by finding balanced braces
        brace_count = 0
        json_end = -1
        in_string = False
        escape_next = False
        
        for i, char in enumerate(inner):
            if escape_next:
                escape_next = False
                continue
            if char == '\\' and in_string:
                escape_next = True
                continue
            if char == '"' and not in_string:
                in_string = True
                continue
            if char == '"' and in_string:
                in_string = False
                continue
            if not in_string:
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_end = i + 1
        
        # If we found a complete JSON object, try to parse just that part
        if json_end > 0:
            try:
                results.append(json.loads(inner[:json_end]))
            except json.JSONDecodeError:
                # Fall back to the flexible fixer
                fixed = _fix_json(inner[:json_end])
                if fixed:
                    results.append(fixed)
        else:
            # Try the flexible fixer on the whole inner content
            fixed = _fix_json(inner)
            if fixed:
                results.append(fixed)
    
    return results or None


def _extract_json_flexible(text: str) -> dict | None:
    """Extract JSON with multiple fallback strategies.
    
    Tries multiple approaches in order:
    1. Standard <json>...</json> blocks
    2. Markdown code blocks with json
    3. Raw JSON objects in text (with nested brace handling)
    4. Look for JSON with "reasoning" and "response" keys
    5. LLM may output JSON with single quotes - try to fix
    6. Look for numeric response values in text as last resort
    7. Handle boolean true/false responses
    8. Handle yes/no/correct/incorrect text responses
    """
    # Strategy 1: Standard <json> tags
    results = _extract_jsons(text)
    if results:
        return results[-1]
    
    # Strategy 2: Markdown code blocks
    json_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    for match in re.finditer(json_block_pattern, text, re.DOTALL):
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            # Try fixing common JSON issues
            fixed = _fix_json(match.group(1).strip())
            if fixed:
                return fixed
            continue
    
    # Strategy 3: Look for JSON-like structures with balanced braces
    # This handles nested JSON objects properly
    def find_json_objects(s: str) -> list[str]:
        """Find all JSON-like objects with balanced braces."""
        objects = []
        i = 0
        while i < len(s):
            if s[i] == '{':
                start = i
                brace_count = 1
                i += 1
                while i < len(s) and brace_count > 0:
                    if s[i] == '{':
                        brace_count += 1
                    elif s[i] == '}':
                        brace_count -= 1
                    i += 1
                if brace_count == 0:
                    objects.append(s[start:i])
            else:
                i += 1
        return objects
    
    for obj_str in find_json_objects(text):
        try:
            parsed = json.loads(obj_str)
            if isinstance(parsed, dict) and "response" in parsed:
                return parsed
        except json.JSONDecodeError:
            # Try fixing common JSON issues
            fixed = _fix_json(obj_str)
            if fixed and "response" in fixed:
                return fixed
            continue
    
    # Strategy 4: Look for JSON with "response" key (simple pattern)
    json_pattern = r'\{\s*"response"\s*:[^\}]+\}'
    for match in re.finditer(json_pattern, text, re.DOTALL):
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
    
    # Strategy 5: Look for explicit response declarations in text
    # Pattern: "response": 0 or "response": 1 or response is 0/1
    response_patterns = [
        r'["\']?response["\']?\s*[:=]\s*(\d)',
        r'response\s*is\s*(\d)',
        r'answer\s*is\s*(\d)',
        r'mark\s*as\s*(\d)',
        r'grade\s*as\s*(\d)',
    ]
    for pattern in response_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            val = match.group(1)
            if val in ('0', '1'):
                return {
                    "reasoning": f"Extracted from text pattern: {match.group(0)}",
                    "response": int(val)
                }
    
    # Strategy 6: Look for standalone 0 or 1 at end of text
    # This catches cases where LLM just outputs the number
    standalone_pattern = r'(?:^|\s)([01])(?:\s*$|\s*\.)'
    match = re.search(standalone_pattern, text.strip())
    if match:
        val = match.group(1)
        return {
            "reasoning": f"Extracted standalone value: {val}",
            "response": int(val)
        }
    
    # Strategy 7: Look for boolean true/false in JSON-like context
    # Handle cases like {"response": true} or {"response": false}
    bool_pattern = r'["\']?response["\']?\s*[:=]\s*(true|false)'
    match = re.search(bool_pattern, text, re.IGNORECASE)
    if match:
        val = match.group(1).lower()
        return {
            "reasoning": f"Extracted boolean value: {val}",
            "response": 1 if val == "true" else 0
        }
    
    # Strategy 8: Look for yes/no/correct/incorrect in text
    # This handles cases where LLM responds with natural language
    text_lower = text.lower().strip()
    
    # Check for clear correct/incorrect indicators at the end of the response
    correct_indicators = [
        r'the answer is correct',
        r'this is correct',
        r'mark as correct',
        r'grade as correct',
        r'correct\s*[:\.]?\s*$',
        r'yes[.!]?\s*$',
        r'right[.!]?\s*$',
        r'valid[.!]?\s*$',
        r'pass[.!]?\s*$',
        r'approved[.!]?\s*$',
    ]
    
    incorrect_indicators = [
        r'the answer is incorrect',
        r'this is incorrect',
        r'mark as incorrect',
        r'grade as incorrect',
        r'incorrect\s*[:\.]?\s*$',
        r'no[.!]?\s*$',
        r'wrong[.!]?\s*$',
        r'invalid[.!]?\s*$',
        r'fail[.!]?\s*$',
        r'rejected[.!]?\s*$',
    ]
    
    for pattern in correct_indicators:
        if re.search(pattern, text_lower):
            return {
                "reasoning": f"Extracted from text: '{pattern}' indicates correct answer",
                "response": 1
            }
    
    for pattern in incorrect_indicators:
        if re.search(pattern, text_lower):
            return {
                "reasoning": f"Extracted from text: '{pattern}' indicates incorrect answer",
                "response": 0
            }
    
    return None


def _fix_json(text: str) -> dict | None:
    """Attempt to fix common JSON formatting issues.
    
    Handles:
    - Single quotes instead of double quotes
    - Trailing commas
    - Missing quotes around keys
    - Boolean values (true/false) in various formats
    - Unquoted numeric values
    - Newlines and special characters in strings
    """
    import ast
    
    # Try Python literal eval (handles single quotes)
    try:
        # Replace single quotes with double quotes for JSON compatibility
        # First, try ast.literal_eval which handles Python dict syntax
        result = ast.literal_eval(text)
        if isinstance(result, dict):
            return result
    except (ValueError, SyntaxError):
        pass
    
    # Manual fix: replace single quotes with double quotes
    # Be careful with apostrophes in text
    try:
        # Simple approach: replace ' with " for key/value delimiters
        # This is a heuristic and may not work for all cases
        fixed = text.replace("'", '"')
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    
    # Try fixing trailing commas
    try:
        # Remove trailing commas before closing braces/brackets
        fixed = re.sub(r',(\s*[}\]])', r'\1', text)
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    
    # Try fixing Python-style booleans (True/False) to JSON-style (true/false)
    try:
        fixed = text.replace("True", "true").replace("False", "false")
        fixed = fixed.replace("None", "null")
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    
    # Try fixing unquoted keys by adding quotes
    try:
        # Pattern to match unquoted keys: word characters followed by colon
        fixed = re.sub(r'(\w+)(\s*:)', r'"\1"\2', text)
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    
    return None


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.max_retries = 3

    def _build_grading_prompt(
        self,
        domain: str,
        problem: str,
        solution: str,
        grading_guidelines: str,
        student_answer: str,
    ) -> str:
        """Build a structured grading prompt with clear instructions."""
        # Check for empty/invalid answers early to provide focused prompt
        student_clean = str(student_answer).strip() if student_answer else ""
        is_empty = not student_clean or student_clean.lower() in (
            "", "i don't know", "i dont know", "idk", "n/a", "none", "null", 
            "empty", "blank", "no answer", "skip", "?"
        )
        
        if is_empty:
            return f"""You are an expert {domain} grader evaluating student solutions.

The student has provided an EMPTY or INVALID answer. This must be marked as INCORRECT (0).

=== PROBLEM ===
{problem}

=== STUDENT'S ANSWER ===
{student_answer if student_answer else "[EMPTY - NO ANSWER PROVIDED]"}

=== GRADING INSTRUCTIONS ===
The student answer is empty, blank, or indicates they don't know the answer.
According to the grading criteria, empty answers must be marked as incorrect (0).

=== RESPONSE FORMAT ===
You MUST respond with a valid JSON object wrapped in <json> tags:

<json>
{{
    "reasoning": "The student provided an empty or invalid answer with no work shown. Marking as incorrect.",
    "response": 0
}}
</json>

The "response" field MUST be 0 for empty/invalid answers."""
        
        return f"""You are an expert {domain} grader evaluating student solutions.

Your task is to grade a student's answer by comparing it to the correct solution and following the grading guidelines.

=== PROBLEM ===
{problem}

=== CORRECT SOLUTION ===
{solution}

=== GRADING GUIDELINES ===
{grading_guidelines}

=== STUDENT'S ANSWER ===
{student_answer}

=== GRADING INSTRUCTIONS ===
Think step by step:
1. Analyze what the problem is asking for
2. Review the correct solution approach
3. Compare the student's answer to the correct solution
4. Check if the student followed the grading guidelines
5. Determine if the student's answer is correct (1) or incorrect (0)

Important grading criteria:
- The answer must be mathematically correct
- The reasoning must be sound and complete
- Partial credit is NOT given - the answer is either fully correct (1) or incorrect (0)
- If the student answer is empty, blank, or says "I don't know", mark as incorrect (0)

=== RESPONSE FORMAT - CRITICAL ===
You MUST respond with ONLY a valid JSON object wrapped in <json> tags. Do not include any text before or after the JSON.

The JSON must have exactly these two fields:
- "reasoning": A string explaining your analysis
- "response": An INTEGER - either 0 or 1 (NOT a string, NOT true/false)

Example for a CORRECT answer:
<json>
{{
    "reasoning": "The student's answer matches the correct solution. They correctly applied the formula and arrived at the right answer.",
    "response": 1
}}
</json>

Example for an INCORRECT answer:
<json>
{{
    "reasoning": "The student's answer is wrong. They made an error in the calculation at step 3, leading to an incorrect final result.",
    "response": 0
}}
</json>

CRITICAL RULES:
1. The "response" field MUST be the INTEGER 0 or 1, not a string like "0" or "1"
2. Do NOT use true/false - use 1 for correct, 0 for incorrect
3. Do NOT add any text outside the <json> tags
4. Ensure the JSON is valid - check for proper quotes, commas, and braces"""

    def _validate_prediction(self, prediction: any) -> str | None:
        """Validate and normalize prediction value.
        
        Returns normalized prediction string or None if invalid.
        Handles various formats including booleans, numbers, strings,
        and edge cases like numpy types.
        """
        # Handle None/null
        if prediction is None:
            return None
            
        # Handle various formats
        if isinstance(prediction, bool):
            return "1" if prediction else "0"
            
        # Handle numpy types (common in ML pipelines)
        if hasattr(prediction, 'item'):  # numpy scalar
            try:
                prediction = prediction.item()
            except (AttributeError, ValueError):
                pass
                
        if isinstance(prediction, (int, float)):
            # Handle edge cases like 0.9999999 or 1.0000001
            if abs(prediction - 1) < 0.001:
                return "1"
            elif abs(prediction - 0) < 0.001:
                return "0"
            # Also check exact values
            if prediction == 1 or prediction == 1.0:
                return "1"
            elif prediction == 0 or prediction == 0.0:
                return "0"
                
        if isinstance(prediction, str):
            pred_clean = prediction.strip().lower()
            # Direct numeric match
            if pred_clean == "1":
                return "1"
            elif pred_clean == "0":
                return "0"
            # Boolean-like strings
            elif pred_clean in ("true", "correct", "yes", "right", "valid", "pass", "approved", "accept"):
                return "1"
            elif pred_clean in ("false", "incorrect", "no", "wrong", "invalid", "fail", "rejected", "deny"):
                return "0"
            # Try to extract number from string like "The answer is 1"
            num_match = re.search(r'\b([01])\b', pred_clean)
            if num_match:
                return num_match.group(1)
            # Check if string starts with 0 or 1 (e.g., "1 - correct")
            if pred_clean.startswith("1"):
                return "1"
            elif pred_clean.startswith("0"):
                return "0"
                
        # Handle lists/arrays - take first element if it's 0 or 1
        if isinstance(prediction, (list, tuple)) and len(prediction) > 0:
            first = prediction[0]
            if isinstance(first, (int, float)):
                if first == 1 or first == 1.0:
                    return "1"
                elif first == 0 or first == 0.0:
                    return "0"
                    
        return None

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Extract key fields for better prompting
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        # Early check for empty/invalid student answers
        student_clean = str(student_answer).strip().lower() if student_answer else ""
        is_empty = not student_clean or student_clean in (
            "", "i don't know", "i dont know", "idk", "n/a", "none", "null",
            "empty", "blank", "no answer", "skip", "?", "???", "....", "----",
            "not sure", "unsure", "don't know", "dont know", "no idea",
            "can't answer", "cant answer", "unable to answer", "n/a", "na",
            "undefined", "nan", "inf", "infinity", "-", "--", "---"
        )
        
        # Also check for answers that are just whitespace or punctuation
        if not is_empty:
            # Remove all whitespace and punctuation
            stripped = re.sub(r'[\s\W_]+', '', student_clean)
            if not stripped or len(stripped) < 2:  # Single character or empty
                is_empty = True
        
        if is_empty:
            self.log_fn(f"Empty or invalid student answer detected early: '{student_answer}', returning 0")
            return "0", []

        # Build structured prompt
        instruction = self._build_grading_prompt(
            domain=domain,
            problem=problem,
            solution=solution,
            grading_guidelines=grading_guidelines,
            student_answer=student_answer,
        )

        msg_history = []
        last_error = None
        
        # Try with retries for robustness
        for attempt in range(self.max_retries):
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=msg_history if msg_history else [],
                )
                
                # Extract prediction from JSON using flexible extraction
                response_text = msg_history[-1]["text"] if msg_history else ""
                extracted = _extract_json_flexible(response_text)
                
                if extracted and "response" in extracted:
                    prediction = extracted["response"]
                    validated = self._validate_prediction(prediction)
                    
                    if validated is not None:
                        self.log_fn(f"Valid prediction: {validated} (raw: {prediction}, attempt: {attempt + 1})")
                        return validated, msg_history
                    else:
                        self.log_fn(f"Invalid prediction value: {prediction}, retrying... (attempt {attempt + 1}/{self.max_retries})")
                        # Add feedback to history for next attempt with more context
                        feedback = (
                            f"Your response was invalid. The 'response' field must be 0 or 1, but got: {prediction} "
                            f"(type: {type(prediction).__name__}). "
                            f"Please provide a valid JSON response with 'response' set to 0 or 1. "
                            f"Example: <json>{{'reasoning': 'The answer is correct', 'response': 1}}</json>"
                        )
                        msg_history.append({"role": "user", "text": feedback})
                else:
                    self.log_fn(f"No valid JSON found in response, retrying... (attempt {attempt + 1}/{self.max_retries})")
                    # Add feedback to history for next attempt with more context
                    feedback = (
                        "Your response did not contain valid JSON. "
                        "Please respond with a JSON object wrapped in <json> tags containing 'reasoning' and 'response' fields. "
                        "The 'response' field must be an integer: 0 for incorrect or 1 for correct. "
                        "Example: <json>{'reasoning': 'The student made an error', 'response': 0}</json>"
                    )
                    if response_text:
                        # Show what we received to help debug
                        preview = response_text[:200] + "..." if len(response_text) > 200 else response_text
                        feedback += f"\n\nI received: {preview}"
                    msg_history.append({"role": "user", "text": feedback})
                    
            except Exception as e:
                last_error = e
                self.log_fn(f"Error on attempt {attempt + 1}: {e}")
                if attempt < self.max_retries - 1:
                    # Clear history on error to start fresh
                    msg_history = []
                else:
                    break
        
        # Final fallback: return "0" if all retries failed
        self.log_fn(f"All retries failed (last error: {last_error}), returning default prediction 0")
        return "0", msg_history
