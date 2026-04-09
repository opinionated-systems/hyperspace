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
        try:
            results.append(json.loads(inner))
        except json.JSONDecodeError:
            continue
    return results or None


def _extract_json_flexible(text: str) -> dict | None:
    """Extract JSON with multiple fallback strategies.
    
    Tries multiple approaches in order:
    1. Standard <json>...</json> blocks
    2. Markdown code blocks with json
    3. Raw JSON objects in text (with nested brace handling)
    4. Look for JSON with "reasoning" and "response" keys
    5. LLM may output JSON with single quotes - try to fix
    6. Look for numeric response values outside of JSON
    7. Look for yes/no patterns indicating correctness
    8. Look for JSON with nested structures and arrays
    9. Handle truncated or malformed JSON
    """
    if not text or not isinstance(text, str):
        return None
        
    text = text.strip()
    if not text:
        return None
    
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
    
    # Strategy 5: Look for standalone response values in text
    # Sometimes LLM outputs just "response": 1 or similar
    standalone_pattern = r'"response"\s*:\s*(\d+|true|false)'
    match = re.search(standalone_pattern, text, re.IGNORECASE)
    if match:
        value = match.group(1).lower()
        if value in ('1', 'true'):
            return {"response": 1, "reasoning": "Extracted from standalone response value"}
        elif value in ('0', 'false'):
            return {"response": 0, "reasoning": "Extracted from standalone response value"}
    
    # Strategy 6: Look for explicit correctness statements
    correctness_patterns = [
        (r'(?:the\s+)?(?:student\s+)?(?:answer\s+)?is\s+(?:correct|right|valid|true)', 1),
        (r'(?:the\s+)?(?:student\s+)?(?:answer\s+)?is\s+(?:incorrect|wrong|invalid|false)', 0),
        (r'(?:mark(?:ed)?\s+as?\s+)(?:correct|right|valid)', 1),
        (r'(?:mark(?:ed)?\s+as?\s+)(?:incorrect|wrong|invalid)', 0),
        (r'(?:grade|score)\s*[:=]\s*(1|correct)', 1),
        (r'(?:grade|score)\s*[:=]\s*(0|incorrect)', 0),
    ]
    
    text_lower = text.lower()
    for pattern, expected_response in correctness_patterns:
        if re.search(pattern, text_lower):
            return {"response": expected_response, "reasoning": f"Extracted from text pattern matching correctness statement"}
    
    # Strategy 7: Look for yes/no patterns at the start of sentences
    yes_no_patterns = [
        (r'^(?:yes|correct|right|valid|true)\b', 1),
        (r'^(?:no|incorrect|wrong|invalid|false)\b', 0),
        (r'\n(?:yes|correct|right|valid|true)\b', 1),
        (r'\n(?:no|incorrect|wrong|invalid|false)\b', 0),
    ]
    
    for pattern, expected_response in yes_no_patterns:
        if re.search(pattern, text_lower):
            return {"response": expected_response, "reasoning": f"Extracted from yes/no pattern in text"}
    
    # Strategy 8: Look for JSON with reasoning field and response field
    # Handle cases where JSON might have additional fields
    extended_json_pattern = r'\{[^}]*"reasoning"[^}]*"response"[^}]*\}'
    for match in re.finditer(extended_json_pattern, text, re.DOTALL | re.IGNORECASE):
        try:
            parsed = json.loads(match.group(0))
            if "response" in parsed:
                return parsed
        except json.JSONDecodeError:
            fixed = _fix_json(match.group(0))
            if fixed and "response" in fixed:
                return fixed
    
    # Strategy 9: Try to extract just the response value from malformed JSON
    # Look for patterns like: response: 1, "response":1, etc.
    loose_response_pattern = r'["\']?response["\']?\s*[:=]\s*["\']?(\d+|true|false)["\']?'
    match = re.search(loose_response_pattern, text, re.IGNORECASE)
    if match:
        value = match.group(1).lower()
        if value in ('1', 'true'):
            return {"response": 1, "reasoning": "Extracted from loose response pattern"}
        elif value in ('0', 'false'):
            return {"response": 0, "reasoning": "Extracted from loose response pattern"}
    
    return None


def _fix_json(text: str) -> dict | None:
    """Attempt to fix common JSON formatting issues.
    
    Handles:
    - Single quotes instead of double quotes
    - Trailing commas
    - Missing quotes around keys
    - Unquoted numeric values
    - Comments in JSON
    - Newlines in strings
    - Escaped characters
    - Truncated JSON (missing closing braces)
    - Unicode escape sequences
    - Mixed quote styles
    """
    import ast
    
    if not text or not isinstance(text, str):
        return None
    
    text = text.strip()
    if not text:
        return None
    
    # Try Python literal eval (handles single quotes, trailing commas)
    try:
        result = ast.literal_eval(text)
        if isinstance(result, dict):
            return result
    except (ValueError, SyntaxError):
        pass
    
    # Try fixing single quotes more carefully
    try:
        # Replace single quotes with double quotes, but be careful with apostrophes
        # This is a heuristic approach
        fixed = text.replace("'", '"')
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    
    # Try removing trailing commas
    try:
        # Remove trailing commas before closing braces/brackets
        fixed = re.sub(r',(\s*[}\]])', r'\1', text)
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    
    # Try fixing unquoted keys (simple heuristic)
    try:
        # Add quotes around unquoted keys (keys that look like identifiers)
        fixed = re.sub(r'(\{|,\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', text)
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    
    # Try removing comments
    try:
        # Remove // comments and /* */ comments
        fixed = re.sub(r'//[^\n]*', '', text)
        fixed = re.sub(r'/\*.*?\*/', '', fixed, flags=re.DOTALL)
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    
    # Try fixing newlines in strings by escaping them
    try:
        # Replace literal newlines in the middle of strings with escaped newlines
        fixed = re.sub(r'("[^"]*?)\n([^"]*?")', r'\1\\n\2', text)
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    
    # Try fixing truncated JSON by adding missing closing braces
    try:
        # Count opening and closing braces
        open_count = text.count('{')
        close_count = text.count('}')
        if open_count > close_count:
            # Add missing closing braces
            fixed = text + ('}' * (open_count - close_count))
            return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    
    # Try fixing mixed quote styles (some keys with quotes, some without)
    try:
        # More aggressive unquoted key fixing
        fixed = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', text)
        # Also try to fix values that should be quoted
        fixed = re.sub(r':\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*([,}])', r': "\1"\2', fixed)
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    
    # Try extracting just the response field if it's a simple object
    try:
        response_match = re.search(r'["\']?response["\']?\s*[:=]\s*(\d+|"[^"]*"|true|false)', text, re.IGNORECASE)
        if response_match:
            value = response_match.group(1)
            # Try to parse the value
            try:
                parsed_value = json.loads(value)
            except json.JSONDecodeError:
                parsed_value = value.strip('"')
            return {"response": parsed_value, "reasoning": "Extracted from partial JSON fix"}
    except Exception:
        pass
    
    # Try to extract reasoning and response separately
    try:
        reasoning_match = re.search(r'["\']?reasoning["\']?\s*[:=]\s*["\']([^"\']+)["\']', text, re.IGNORECASE)
        response_match = re.search(r'["\']?response["\']?\s*[:=]\s*(\d+|true|false)', text, re.IGNORECASE)
        if response_match:
            value = response_match.group(1).lower()
            if value in ('1', 'true'):
                response_val = 1
            elif value in ('0', 'false'):
                response_val = 0
            else:
                response_val = int(value)
            reasoning = reasoning_match.group(1) if reasoning_match else "Extracted from partial JSON"
            return {"response": response_val, "reasoning": reasoning}
    except Exception:
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
            "empty", "blank", "no answer", "skip", "?", "...", "-"
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

=== RESPONSE FORMAT - MANDATORY ===
You MUST respond with ONLY a valid JSON object wrapped in <json> tags. No other text.

<json>
{{
    "reasoning": "The student provided an empty or invalid answer with no work shown. Marking as incorrect.",
    "response": 0
}}
</json>

CRITICAL RULES:
1. The "response" field MUST be the NUMBER 0 (not a string "0").
2. The entire response must be wrapped in <json>...</json> tags.
3. The JSON must be valid - use double quotes for strings and keys.
4. Do not include any text outside the <json> tags.
5. The response field must be a NUMBER (1 or 0), not a string."""
        
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

=== RESPONSE FORMAT - MANDATORY ===
You MUST respond with ONLY a valid JSON object wrapped in <json> tags. No other text before or after.

<json>
{{
    "reasoning": "Your detailed step-by-step analysis here. Explain your thought process clearly.",
    "response": 1
}}
</json>

OR

<json>
{{
    "reasoning": "Your detailed step-by-step analysis here. Explain why the answer is wrong.",
    "response": 0
}}
</json>

CRITICAL RULES - FOLLOW EXACTLY:
1. The "response" field MUST be a NUMBER: either 1 (correct) or 0 (incorrect). 
   - CORRECT: "response": 1
   - INCORRECT: "response": "1" (string) or "response": true (boolean)
2. The entire response must be wrapped in <json>...</json> tags.
3. The JSON must be valid - use double quotes for strings and keys.
4. Do not include any text outside the <json> tags.
5. The response field must be a NUMBER (1 or 0), not a string ("1" or "0").
6. Do not add any markdown formatting, explanations, or notes outside the JSON block."""

    def _validate_prediction(self, prediction: any) -> str | None:
        """Validate and normalize prediction value.
        
        Returns normalized prediction string or None if invalid.
        Handles various formats including booleans, numbers, strings,
        and edge cases like numeric strings with whitespace.
        """
        # Handle None
        if prediction is None:
            return None
            
        # Handle booleans
        if isinstance(prediction, bool):
            return "1" if prediction else "0"
            
        # Handle numbers (int, float)
        if isinstance(prediction, (int, float)):
            # Use a small epsilon for float comparison
            if abs(prediction - 1) < 0.001 or prediction == 1:
                return "1"
            elif abs(prediction - 0) < 0.001 or prediction == 0:
                return "0"
            # Reject other numeric values
            return None
            
        # Handle strings
        if isinstance(prediction, str):
            pred_clean = prediction.strip().lower()
            
            # Direct numeric matches
            if pred_clean in ("1", "1.0", "1.00"):
                return "1"
            elif pred_clean in ("0", "0.0", "0.00"):
                return "0"
                
            # Boolean-like strings
            elif pred_clean in ("true", "correct", "yes", "right", "valid", "accurate"):
                return "1"
            elif pred_clean in ("false", "incorrect", "no", "wrong", "invalid", "inaccurate"):
                return "0"
                
            # Check for numeric values with whitespace
            if re.match(r'^\s*1\s*$', prediction):
                return "1"
            elif re.match(r'^\s*0\s*$', prediction):
                return "0"
                
            # Check for numeric values embedded in text (e.g., "The answer is 1")
            # Look for standalone 1 or 0 at word boundaries
            if re.search(r'\b1\b', pred_clean) and not re.search(r'\b0\b', pred_clean):
                return "1"
            elif re.search(r'\b0\b', pred_clean) and not re.search(r'\b1\b', pred_clean):
                return "0"
                
        # Handle lists/arrays - take first element if it's a valid prediction
        if isinstance(prediction, (list, tuple)) and len(prediction) > 0:
            return self._validate_prediction(prediction[0])
            
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
        student_clean = str(student_answer).strip() if student_answer else ""
        is_empty = not student_clean or student_clean.lower() in (
            "", "i don't know", "i dont know", "idk", "n/a", "none", "null", 
            "empty", "blank", "no answer", "skip", "?", "...", "-", 
            "na", "not applicable", "unsure", "unknown", "can't answer",
            "cannot answer", "no idea", "not sure", "unclear"
        )
        
        if is_empty:
            self.log_fn("Empty or invalid student answer detected early, returning 0")
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
        all_errors = []
        
        # Try with retries for robustness
        for attempt in range(self.max_retries):
            try:
                # Add retry context to help LLM correct itself
                current_instruction = instruction
                if attempt > 0 and all_errors:
                    # Add context about previous failures
                    error_context = f"\n\n=== PREVIOUS ATTEMPT FAILED ===\nError: {all_errors[-1]}\n\nPlease ensure your response follows the exact JSON format specified above."
                    current_instruction = instruction + error_context
                
                response, msg_history, info = get_response_from_llm(
                    msg=current_instruction,
                    model=self.model,
                    msg_history=msg_history if msg_history else [],
                )
                
                # Extract prediction from JSON using flexible extraction
                response_text = msg_history[-1]["text"] if msg_history else ""
                
                # Log the raw response for debugging
                if attempt == 0:
                    self.log_fn(f"Raw response (first 500 chars): {response_text[:500]}")
                else:
                    self.log_fn(f"Retry {attempt} response (first 300 chars): {response_text[:300]}")
                
                extracted = _extract_json_flexible(response_text)
                
                if extracted and "response" in extracted:
                    prediction = extracted["response"]
                    validated = self._validate_prediction(prediction)
                    
                    if validated is not None:
                        self.log_fn(f"Valid prediction: {validated} (raw: {prediction}, attempt: {attempt + 1})")
                        return validated, msg_history
                    else:
                        error_msg = f"Invalid prediction value: {prediction} (type: {type(prediction).__name__})"
                        self.log_fn(f"{error_msg}, retrying... (attempt {attempt + 1})")
                        all_errors.append(error_msg)
                        last_error = error_msg
                else:
                    if extracted:
                        error_msg = f"Extracted data missing 'response' key. Keys found: {list(extracted.keys())}"
                        self.log_fn(f"{error_msg}, retrying... (attempt {attempt + 1})")
                    else:
                        error_msg = "No valid JSON found in response"
                        self.log_fn(f"{error_msg}, retrying... (attempt {attempt + 1})")
                    all_errors.append(error_msg)
                    last_error = error_msg
                    
            except Exception as e:
                error_msg = f"Exception on attempt {attempt + 1}: {type(e).__name__}: {e}"
                self.log_fn(error_msg)
                all_errors.append(error_msg)
                last_error = str(e)
                if attempt == self.max_retries - 1:
                    break
        
        # Final fallback: return "0" if all retries failed
        error_summary = " | ".join(all_errors) if all_errors else last_error
        self.log_fn(f"All retries failed. Errors: {error_summary}. Returning default prediction 0")
        return "0", msg_history
