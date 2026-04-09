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
    Also handles markdown code blocks with json and inline JSON objects.
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
        try:
            results.append(json.loads(inner))
        except json.JSONDecodeError:
            # Try to fix common JSON issues
            try:
                # Remove trailing commas before closing braces/brackets
                fixed = re.sub(r',(\s*[}\]])', r'\1', inner)
                results.append(json.loads(fixed))
            except json.JSONDecodeError:
                # Try to fix unescaped newlines in strings
                try:
                    fixed = re.sub(r'(?<!\\)\n', '\\n', inner)
                    results.append(json.loads(fixed))
                except json.JSONDecodeError:
                    continue
    
    # Also try markdown code blocks
    if not results:
        code_block_pattern = r'```(?:json)?\s*(.*?)\s*```'
        for match in re.finditer(code_block_pattern, text, re.DOTALL):
            try:
                inner = match.group(1).strip()
                results.append(json.loads(inner))
            except json.JSONDecodeError:
                # Try to fix common JSON issues
                try:
                    fixed = re.sub(r',(\s*[}\]])', r'\1', inner)
                    results.append(json.loads(fixed))
                except json.JSONDecodeError:
                    # Try to fix unescaped newlines in strings
                    try:
                        fixed = re.sub(r'(?<!\\)\n', '\\n', inner)
                        results.append(json.loads(fixed))
                    except json.JSONDecodeError:
                        continue
    
    # Try to find JSON objects with "reasoning" and "response" fields
    if not results:
        # Look for patterns like {"reasoning": "...", "response": "..."}
        json_pattern = r'\{\s*"reasoning"\s*:\s*"[^"]*"\s*,\s*"response"\s*:\s*"[^"]*"\s*\}'
        for match in re.finditer(json_pattern, text, re.DOTALL):
            try:
                results.append(json.loads(match.group(0)))
            except json.JSONDecodeError:
                continue
    
    return results or None


def _extract_any_json(text: str) -> list[dict] | None:
    """Fallback JSON extraction that looks for any JSON objects in text.
    
    Uses a stack-based approach to properly handle nested braces and
    attempts to fix common JSON formatting issues.
    """
    results = []
    # Try to find JSON objects between curly braces
    brace_count = 0
    start_idx = -1
    in_string = False
    escape_next = False
    
    for i, char in enumerate(text):
        if escape_next:
            escape_next = False
            continue
        
        if char == '\\' and in_string:
            escape_next = True
            continue
        
        if char == '"' and not in_string:
            in_string = True
        elif char == '"' and in_string:
            in_string = False
        
        if not in_string:
            if char == '{':
                if brace_count == 0:
                    start_idx = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start_idx != -1:
                    json_str = text[start_idx:i+1]
                    try:
                        obj = json.loads(json_str)
                        results.append(obj)
                    except json.JSONDecodeError:
                        # Try to fix common issues with multiple strategies
                        fixed_strategies = [
                            # Strategy 1: Remove trailing commas
                            lambda s: re.sub(r',(\s*[}\]])', r'\1', s),
                            # Strategy 2: Fix unescaped newlines in strings
                            lambda s: re.sub(r'(?<!\\)\n', '\\n', s),
                            # Strategy 3: Fix unescaped tabs in strings
                            lambda s: re.sub(r'(?<!\\)\t', '\\t', s),
                            # Strategy 4: Fix unescaped carriage returns
                            lambda s: re.sub(r'(?<!\\)\r', '\\r', s),
                            # Strategy 5: Fix single quotes used as JSON delimiters
                            lambda s: re.sub(r"(?<!\\)'", '"', s),
                            # Strategy 6: Remove BOM and control characters
                            lambda s: ''.join(c for c in s if ord(c) >= 32 or c in '\n\r\t'),
                        ]
                        
                        for strategy in fixed_strategies:
                            try:
                                fixed = strategy(json_str)
                                obj = json.loads(fixed)
                                results.append(obj)
                                break
                            except json.JSONDecodeError:
                                continue
                    start_idx = -1
    
    return results or None


def _validate_grade(prediction: str, grading_guidelines: str) -> tuple[str, bool]:
    """Validate and normalize the grade prediction.
    
    Args:
        prediction: The raw prediction string from the LLM
        grading_guidelines: The grading guidelines to check against
        
    Returns:
        tuple of (normalized_prediction, is_valid)
    """
    if not prediction or prediction == "None":
        return "None", False
    
    prediction = prediction.strip()
    
    # Common grade patterns - expanded for better coverage
    valid_grades = {
        # Binary grades
        "correct": "Correct",
        "incorrect": "Incorrect",
        "right": "Correct",
        "wrong": "Incorrect",
        "true": "Correct",
        "false": "Incorrect",
        "yes": "Correct",
        "no": "Incorrect",
        "valid": "Correct",
        "invalid": "Incorrect",
        "accepted": "Correct",
        "rejected": "Incorrect",
        "pass": "Correct",
        "fail": "Incorrect",
        # Partial grades
        "partial": "Partial",
        "partially correct": "Partial",
        "incomplete": "Partial",
        "partial credit": "Partial",
        "half correct": "Partial",
        "mostly correct": "Partial",
        "mostly incorrect": "Partial",
        # Numeric grades (0-10 scale)
        "0": "0",
        "1": "1",
        "2": "2",
        "3": "3",
        "4": "4",
        "5": "5",
        "6": "6",
        "7": "7",
        "8": "8",
        "9": "9",
        "10": "10",
        # Additional numeric formats
        "0/10": "0",
        "1/10": "1",
        "2/10": "2",
        "3/10": "3",
        "4/10": "4",
        "5/10": "5",
        "6/10": "6",
        "7/10": "7",
        "8/10": "8",
        "9/10": "9",
        "10/10": "10",
        # Letter grades (A-F scale) - map to 0-10 scale
        "a+": "10",
        "a": "9",
        "a-": "8",
        "b+": "7",
        "b": "6",
        "b-": "5",
        "c+": "4",
        "c": "3",
        "c-": "2",
        "d+": "1",
        "d": "1",
        "d-": "1",
        "f": "0",
        "e": "0",
    }
    
    # Check for exact match (case-insensitive)
    pred_lower = prediction.lower()
    if pred_lower in valid_grades:
        return valid_grades[pred_lower], True
    
    # Check for percentage patterns FIRST (e.g., "50%", "100%") - before individual digits
    import re as regex
    percent_match = regex.search(r'(\d+)%', prediction)
    if percent_match:
        percent = int(percent_match.group(1))
        # Convert percentage to 0-10 scale (e.g., 50% -> 5, 100% -> 10)
        # Use proper rounding: 50/10 = 5.0 -> 5, 55/10 = 5.5 -> 6
        score = min(10, max(0, int(round(percent / 10.0))))
        return str(score), True
    
    # Check if prediction contains a valid grade (longer matches first)
    sorted_keys = sorted(valid_grades.keys(), key=len, reverse=True)
    for key in sorted_keys:
        if key in pred_lower:
            return valid_grades[key], True
    
    # Check for numeric patterns - extract first valid number 0-10
    if any(char.isdigit() for char in prediction):
        numbers = regex.findall(r'\d+', prediction)
        if numbers:
            num = int(numbers[0])
            if 0 <= num <= 10:
                return str(num), True
            # If number is outside 0-10 range, cap it
            if num > 10:
                return "10", True
            if num < 0:
                return "0", True
    
    # Check for decimal scores (e.g., "7.5", "8.3") and round to nearest integer
    decimal_match = regex.search(r'(\d+\.\d+)', prediction)
    if decimal_match:
        decimal_val = float(decimal_match.group(1))
        if 0 <= decimal_val <= 10:
            rounded = int(round(decimal_val))
            return str(rounded), True
        if decimal_val > 10:
            return "10", True
        if decimal_val < 0:
            return "0", True
    
    # Check for fraction patterns (e.g., "3/5", "7/10")
    fraction_match = regex.search(r'(\d+)/(\d+)', prediction)
    if fraction_match:
        numerator = int(fraction_match.group(1))
        denominator = int(fraction_match.group(2))
        if denominator > 0:
            # Convert to 0-10 scale
            score = min(10, max(0, int(round(numerator / denominator * 10))))
            return str(score), True
    
    # Check for letter grades with +/- suffixes (e.g., "Grade: A+", "Score: B")
    letter_grade_match = regex.search(r'\b([a-f][+-]?)\b', pred_lower)
    if letter_grade_match:
        letter_grade = letter_grade_match.group(1)
        if letter_grade in valid_grades:
            return valid_grades[letter_grade], True
    
    # Return original if no normalization possible
    return prediction, True


def _extract_grade_from_text(text: str) -> str:
    """Extract a grade from raw text when JSON parsing fails.
    
    This is a final fallback that looks for common grade patterns
    in the raw text output.
    
    Args:
        text: The raw text to search for grades
        
    Returns:
        The extracted grade or "None" if no grade found
    """
    import re as regex
    
    text_lower = text.lower()
    
    # Look for explicit grade statements
    grade_patterns = [
        # "Grade: X" or "The grade is X"
        r'(?:grade|score|mark|evaluation)(?:\s*[:=]\s*|\s+is\s+|\s+of\s+)([\w\s/%.+-]+?)(?:\s*[.\n]|$)',
        # "I give a grade of X"
        r'(?:give|assign|award)(?:\s+a?\s*)(?:grade|score|mark)(?:\s+of\s+)([\w\s/%.+-]+?)(?:\s*[.\n]|$)',
        # "The answer is X" (for binary grades)
        r'(?:answer|result|conclusion)(?:\s+is\s+)(correct|incorrect|partial|right|wrong)(?:\s*[.\n]|$)',
        # "Final grade: X"
        r'final\s+(?:grade|score|mark)(?:\s*[:=]\s*)([\w\s/%.+-]+?)(?:\s*[.\n]|$)',
    ]
    
    for pattern in grade_patterns:
        match = regex.search(pattern, text_lower, regex.IGNORECASE)
        if match:
            grade_text = match.group(1).strip()
            # Validate the extracted grade
            normalized, is_valid = _validate_grade(grade_text, "")
            if is_valid and normalized != "None":
                return normalized
    
    # Look for standalone grades at the end of the text
    # Common patterns like "Correct", "7/10", "A+", etc.
    standalone_patterns = [
        r'\b(correct|incorrect|partial|right|wrong|pass|fail)\b',
        r'\b([0-9]|10)(?:\s*/\s*10)?\b',
        r'\b([a-f][+-]?)\b',
        r'(\d+)%',
    ]
    
    for pattern in standalone_patterns:
        matches = regex.findall(pattern, text_lower)
        if matches:
            for match in matches:
                grade_text = match.strip() if isinstance(match, str) else match[0].strip()
                normalized, is_valid = _validate_grade(grade_text, "")
                if is_valid and normalized != "None":
                    return normalized
    
    return "None"


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

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
        # Extract task components for better prompting
        domain = inputs.get("domain", "Unknown")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = f"""You are an expert mathematical grader for International Mathematical Olympiad (IMO) problems.

Your task is to evaluate a student's answer to a mathematics problem and provide a grade.

DOMAIN: {domain}

PROBLEM:
{problem}

OFFICIAL SOLUTION:
{solution}

GRADING GUIDELINES:
{grading_guidelines}

STUDENT'S ANSWER:
{student_answer}

Think step by step:
1. Analyze what the problem is asking - identify key concepts and required steps
2. Review the official solution approach - understand the correct method
3. Compare the student's answer to the official solution - check for:
   - Correctness of the final answer
   - Validity of the reasoning process
   - Completeness of the solution
   - Mathematical rigor and clarity
4. Check if the student followed the grading guidelines - look for specific criteria mentioned
5. Determine the appropriate grade based on the evidence

IMPORTANT GRADING INSTRUCTIONS:
- Be objective and consistent in your evaluation
- Consider partial credit for correct reasoning even if the final answer is wrong
- Consider the student's approach, not just the final result
- If the guidelines specify a numeric scale (0-10), use that scale
- If the guidelines specify binary grading, use "Correct" or "Incorrect"
- If partial credit is allowed, use "Partial" for incomplete but partially correct answers
- The grade in your response field MUST match the format specified in the guidelines

Your response MUST be in the following JSON format (wrapped in <json> tags):
<json>
{{
    "reasoning": "Your detailed chain-of-thought analysis here. Explain your evaluation process clearly.",
    "response": "The final grade/prediction (e.g., 'Correct', 'Incorrect', 'Partial', or a numeric score like '7')"
}}
</json>

CRITICAL: Ensure the JSON is valid and properly formatted. The "response" field must contain ONLY the grade value without any additional text, explanation, or formatting."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON with fallback mechanisms
        prediction = "None"
        extraction_method = "none"
        try:
            last_message = msg_history[-1]["text"]
            
            # Try primary extraction method
            extracted = _extract_jsons(last_message)
            extraction_method = "json_tags"
            
            # Fallback to generic JSON extraction if primary fails
            if extracted is None:
                extracted = _extract_any_json(last_message)
                extraction_method = "any_json"
            
            if extracted:
                # Prefer response field, but accept other common field names
                last_json = extracted[-1]
                if "response" in last_json:
                    prediction = last_json["response"]
                elif "grade" in last_json:
                    prediction = last_json["grade"]
                elif "answer" in last_json:
                    prediction = last_json["answer"]
                elif "result" in last_json:
                    prediction = last_json["result"]
                elif "evaluation" in last_json:
                    prediction = last_json["evaluation"]
                else:
                    # If no known field, use the first string value found
                    for key, value in last_json.items():
                        if isinstance(value, str):
                            prediction = value
                            break
                
                # Validate and normalize the grade
                normalized, is_valid = _validate_grade(prediction, grading_guidelines)
                if is_valid:
                    prediction = normalized
                    self.log_fn(f"Grade extracted via {extraction_method}: {prediction}")
                else:
                    self.log_fn(f"Warning: Invalid grade '{prediction}', using normalized: {normalized}")
                    prediction = normalized
            else:
                # Final fallback: try to extract grade directly from text
                self.log_fn(f"Warning: No JSON found in response, attempting text extraction")
                prediction = _extract_grade_from_text(last_message)
                if prediction != "None":
                    extraction_method = "text_fallback"
                    self.log_fn(f"Grade extracted via text fallback: {prediction}")
                else:
                    self.log_fn(f"Warning: Could not extract grade from response, using 'None'")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
