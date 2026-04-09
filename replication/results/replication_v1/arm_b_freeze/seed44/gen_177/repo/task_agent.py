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
    7. Look for response values in various formats
    8. Pattern matching for correctness statements
    """
    if not text or not isinstance(text, str):
        return None
    
    # Clean up the text
    text = text.strip()
    
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
            fixed = _fix_json(match.group(0))
            if fixed:
                return fixed
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
    
    # Strategy 6: Look for response in single quotes
    single_quote_pattern = r"'response'\s*:\s*(\d+|'[^']*'|true|false)"
    match = re.search(single_quote_pattern, text, re.IGNORECASE)
    if match:
        value = match.group(1).lower().strip("'\"")
        if value in ('1', 'true', 'correct', 'yes'):
            return {"response": 1, "reasoning": "Extracted from single-quoted response value"}
        elif value in ('0', 'false', 'incorrect', 'no'):
            return {"response": 0, "reasoning": "Extracted from single-quoted response value"}
    
    # Strategy 7: Look for explicit correctness statements
    correctness_patterns = [
        (r'(?:the\s+)?(?:student\s+)?(?:answer\s+)?is\s+(?:correct|right|valid|true)', 1),
        (r'(?:the\s+)?(?:student\s+)?(?:answer\s+)?is\s+(?:incorrect|wrong|invalid|false)', 0),
        (r'(?:mark(?:ed)?\s+as?\s+)(?:correct|right|valid)', 1),
        (r'(?:mark(?:ed)?\s+as?\s+)(?:incorrect|wrong|invalid)', 0),
        (r'(?:grade|score)\s*[:=]\s*(1|correct)', 1),
        (r'(?:grade|score)\s*[:=]\s*(0|incorrect)', 0),
        (r'(?:answer\s+is\s+)(?:correct|right|valid)', 1),
        (r'(?:answer\s+is\s+)(?:incorrect|wrong|invalid)', 0),
        (r'(?:should\s+be\s+)(?:marked\s+)?(?:as\s+)?(?:correct|right)', 1),
        (r'(?:should\s+be\s+)(?:marked\s+)?(?:as\s+)?(?:incorrect|wrong)', 0),
    ]
    
    text_lower = text.lower()
    for pattern, expected_response in correctness_patterns:
        if re.search(pattern, text_lower):
            return {"response": expected_response, "reasoning": f"Extracted from text pattern matching correctness statement"}
    
    # Strategy 8: Look for final verdict/conclusion patterns
    verdict_patterns = [
        (r'(?:final\s+)?(?:verdict|conclusion|decision|result)\s*[:=]\s*(?:correct|right|valid|true|1)', 1),
        (r'(?:final\s+)?(?:verdict|conclusion|decision|result)\s*[:=]\s*(?:incorrect|wrong|invalid|false|0)', 0),
        (r'(?:therefore|thus|hence|so)\s*,?\s*(?:the\s+)?(?:answer\s+)?is\s+(?:correct|right|valid)', 1),
        (r'(?:therefore|thus|hence|so)\s*,?\s*(?:the\s+)?(?:answer\s+)?is\s+(?:incorrect|wrong|invalid)', 0),
    ]
    
    for pattern, expected_response in verdict_patterns:
        if re.search(pattern, text_lower):
            return {"response": expected_response, "reasoning": f"Extracted from verdict/conclusion pattern"}
    
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
    - Unicode issues
    - Missing closing braces/brackets
    """
    import ast
    
    # Clean up the text first
    text = text.strip()
    
    # Try Python literal eval (handles single quotes, trailing commas)
    try:
        result = ast.literal_eval(text)
        if isinstance(result, dict):
            return result
    except (ValueError, SyntaxError):
        pass
    
    # Try fixing single quotes more carefully - only replace quotes that are
    # likely to be JSON string delimiters, not apostrophes in words
    try:
        # A more careful approach: replace quotes that appear to be delimiters
        # This regex looks for single quotes that are followed by a colon or comma
        # or preceded by a colon, comma, bracket, or brace
        fixed = re.sub(r"(?<=[{\[,:\s])'|'(?=[:}\],])", '"', text)
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
    
    # Try fixing newlines in strings (replace with escaped newlines)
    try:
        # Find string content and escape newlines within them
        def escape_newlines_in_strings(match):
            content = match.group(1)
            # Escape unescaped newlines
            content = re.sub(r'(?<!\\)\n', r'\\n', content)
            content = re.sub(r'(?<!\\)\r', r'\\r', content)
            content = re.sub(r'(?<!\\)\t', r'\\t', content)
            return '"' + content + '"'
        
        # Match content between double quotes, handling escaped quotes
        fixed = re.sub(r'"((?:[^"\\]|\\.)*)"', escape_newlines_in_strings, text)
        return json.loads(fixed)
    except (json.JSONDecodeError, re.error):
        pass
    
    # Try to fix missing closing braces by adding them
    try:
        open_braces = text.count('{') - text.count('}')
        open_brackets = text.count('[') - text.count(']')
        fixed = text + ('}' * open_braces) + (']' * open_brackets)
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    
    # Try extracting just the first valid JSON object
    try:
        # Find the first { and try to parse from there
        start = text.find('{')
        if start != -1:
            # Try progressively shorter substrings
            for end in range(len(text), start, -1):
                try:
                    substring = text[start:end]
                    # Balance braces
                    open_count = substring.count('{') - substring.count('}')
                    if open_count > 0:
                        substring += '}' * open_count
                    result = json.loads(substring)
                    if isinstance(result, dict):
                        return result
                except json.JSONDecodeError:
                    continue
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
        student_clean = str(student_answer).strip() if student_answer is not None else ""
        
        # Expanded list of empty/invalid answer indicators
        empty_indicators = {
            "", "i don't know", "i dont know", "idk", "n/a", "none", "null", 
            "empty", "blank", "no answer", "skip", "?", "...", "-", "--", "---",
            "na", "n/a", "not applicable", "not available", "no solution",
            "can't solve", "cannot solve", "unsure", "unknown", "?",
            "i'm not sure", "im not sure", "no idea", "dont know",
            "[empty]", "[blank]", "[no answer]", "[none]",
        }
        
        is_empty = not student_clean or student_clean.lower() in empty_indicators
        
        # Also check for answers that are just whitespace or punctuation
        if not is_empty:
            stripped = re.sub(r'[\s\?\!\.\,\-\_\*\+\=\(\)\[\]\{\}\|\&\^\%\$\#\@\~`]+', '', student_clean)
            if not stripped:
                is_empty = True
        
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
- Check for common errors: calculation mistakes, wrong formulas, missing steps, incorrect units

=== RESPONSE FORMAT - CRITICAL ===
You MUST respond with a valid JSON object wrapped in <json> tags. This is REQUIRED.

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

CRITICAL RULES:
1. The "response" field MUST be either 1 (correct) or 0 (incorrect). No other values are accepted.
2. The entire response must be wrapped in <json>...</json> tags.
3. The JSON must be valid - use double quotes for strings and keys.
4. Do not include any text outside the <json> tags.
5. The response field must be a NUMBER (1 or 0), not a string ("1" or "0").
6. The reasoning field must be a string explaining your decision.
7. Do not use single quotes in the JSON - only double quotes are valid.
8. Make sure all opening braces {{ have matching closing braces }}.
9. Make sure all opening brackets [ have matching closing brackets ]."""

    def _validate_prediction(self, prediction: any) -> str | None:
        """Validate and normalize prediction value.
        
        Returns normalized prediction string or None if invalid.
        """
        # Handle None
        if prediction is None:
            return None
            
        # Handle various formats
        if isinstance(prediction, bool):
            return "1" if prediction else "0"
        if isinstance(prediction, (int, float)):
            # Handle NaN and infinity
            if isinstance(prediction, float):
                if prediction != prediction:  # NaN check
                    return None
                if prediction == float('inf') or prediction == float('-inf'):
                    return None
            if prediction == 1 or prediction == 1.0:
                return "1"
            elif prediction == 0 or prediction == 0.0:
                return "0"
        if isinstance(prediction, str):
            pred_clean = prediction.strip().lower()
            # Direct numeric matches
            if pred_clean in ("1", "1.0", "+1"):
                return "1"
            elif pred_clean in ("0", "0.0", "+0"):
                return "0"
            # Boolean-like strings
            elif pred_clean in ("true", "correct", "yes", "right", "valid", "accurate", "proper", "acceptable"):
                return "1"
            elif pred_clean in ("false", "incorrect", "no", "wrong", "invalid", "inaccurate", "improper", "unacceptable"):
                return "0"
            # Check for numeric values embedded in text (with optional whitespace)
            if re.match(r'^\s*1\s*$', prediction):
                return "1"
            elif re.match(r'^\s*0\s*$', prediction):
                return "0"
            # Check for numeric values at the start of the string
            match = re.match(r'^\s*(\d+)', prediction)
            if match:
                num = int(match.group(1))
                if num == 1:
                    return "1"
                elif num == 0:
                    return "0"
        # Handle lists/arrays - take first element if it exists
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
        student_clean = str(student_answer).strip() if student_answer is not None else ""
        
        # Expanded list of empty/invalid answer indicators
        empty_indicators = {
            "", "i don't know", "i dont know", "idk", "n/a", "none", "null", 
            "empty", "blank", "no answer", "skip", "?", "...", "-", "--", "---",
            "na", "n/a", "not applicable", "not available", "no solution",
            "can't solve", "cannot solve", "unsure", "unknown", "?",
            "i'm not sure", "im not sure", "no idea", "dont know",
            "[empty]", "[blank]", "[no answer]", "[none]",
        }
        
        is_empty = not student_clean or student_clean.lower() in empty_indicators
        
        # Also check for answers that are just whitespace or punctuation
        if not is_empty:
            # Check if answer contains only whitespace and common punctuation
            stripped = re.sub(r'[\s\?\!\.\,\-\_\*\+\=\(\)\[\]\{\}\|\&\^\%\$\#\@\~`]+', '', student_clean)
            if not stripped:
                is_empty = True
        
        if is_empty:
            display_answer = student_clean[:50] + "..." if len(student_clean) > 50 else student_clean
            self.log_fn(f"Empty or invalid student answer detected: '{display_answer}'")
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
        all_responses = []  # Track all responses for potential pattern analysis
        
        # Try with retries for robustness
        for attempt in range(self.max_retries):
            try:
                # Add retry context to instruction on subsequent attempts
                current_instruction = instruction
                if attempt > 0 and last_error:
                    current_instruction = f"{instruction}\n\n[Previous attempt failed: {last_error}. Please ensure your response is valid JSON with 'response' and 'reasoning' fields wrapped in <json> tags.]"
                
                response, msg_history, info = get_response_from_llm(
                    msg=current_instruction,
                    model=self.model,
                    msg_history=msg_history if msg_history else [],
                )
                
                # Extract prediction from JSON using flexible extraction
                response_text = msg_history[-1]["text"] if msg_history else ""
                all_responses.append(response_text)
                
                # Log the raw response for debugging
                if attempt == 0:
                    self.log_fn(f"Raw response (first 500 chars): {response_text[:500]}")
                
                extracted = _extract_json_flexible(response_text)
                
                if extracted and "response" in extracted:
                    prediction = extracted["response"]
                    validated = self._validate_prediction(prediction)
                    
                    if validated is not None:
                        self.log_fn(f"Valid prediction: {validated} (raw: {prediction}, attempt: {attempt + 1})")
                        return validated, msg_history
                    else:
                        self.log_fn(f"Invalid prediction value: {prediction}, retrying... (attempt {attempt + 1})")
                        last_error = f"Invalid prediction value: {prediction}"
                else:
                    self.log_fn(f"No valid JSON found in response, retrying... (attempt {attempt + 1})")
                    if extracted:
                        self.log_fn(f"Extracted data missing 'response' key: {list(extracted.keys())}")
                    last_error = "No valid JSON with 'response' key found"
                    
            except Exception as e:
                self.log_fn(f"Error on attempt {attempt + 1}: {e}")
                last_error = str(e)
                if attempt == self.max_retries - 1:
                    break
        
        # Before giving up, try to extract any signal from all responses
        # Look for consensus in the responses
        if all_responses:
            correct_count = 0
            incorrect_count = 0
            for resp in all_responses:
                resp_lower = resp.lower()
                # Count explicit correctness mentions
                if any(word in resp_lower for word in ['correct', 'right', 'valid', 'true', 'accurate']):
                    correct_count += 1
                if any(word in resp_lower for word in ['incorrect', 'wrong', 'invalid', 'false', 'inaccurate']):
                    incorrect_count += 1
            
            # If there's a clear consensus, use it
            if correct_count > incorrect_count and correct_count > 0:
                self.log_fn(f"Consensus extraction: correct (correct={correct_count}, incorrect={incorrect_count})")
                return "1", msg_history
            elif incorrect_count > correct_count and incorrect_count > 0:
                self.log_fn(f"Consensus extraction: incorrect (correct={correct_count}, incorrect={incorrect_count})")
                return "0", msg_history
        
        # Final fallback: return "0" if all retries failed
        self.log_fn(f"All retries failed (last error: {last_error}), returning default prediction 0")
        return "0", msg_history
