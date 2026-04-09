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


def _extract_json_from_markdown(text: str) -> list[dict] | None:
    """Extract JSON objects from markdown code blocks.

    Fallback for when <json> tags are not used but markdown code blocks are.
    """
    results = []
    # Match ```json ... ``` or just ``` ... ``` blocks
    pattern = r'```(?:json)?\s*(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    for match in matches:
        try:
            results.append(json.loads(match.strip()))
        except json.JSONDecodeError:
            continue
    return results or None


def _normalize_grade(grade: str) -> str:
    """Normalize grade to a standard format.

    Handles various grade formats. Preserves numeric grades (0-7) as they
    are commonly used in IMO-style grading. Maps text variations to
    standard categories. Also handles fractional grades like 0.5, 1/2, etc.
    """
    if grade is None:
        return "None"
    
    if not isinstance(grade, str):
        grade = str(grade)
    
    grade = grade.strip()
    grade_lower = grade.lower()
    
    # Handle empty string
    if not grade:
        return "None"
    
    # Check for numeric grades (0-7) - preserve these for IMO-style grading
    if grade.isdigit():
        num = int(grade)
        if 0 <= num <= 7:
            return grade  # Preserve numeric grades as-is
    
    # Check for decimal/float grades (e.g., 0.5, 1.5, 2.5)
    try:
        float_grade = float(grade)
        if 0 <= float_grade <= 7:
            # Return as string, preserving decimal if needed
            if float_grade == int(float_grade):
                return str(int(float_grade))
            return str(float_grade)
    except ValueError:
        pass
    
    # Check for fraction grades (e.g., "1/2", "3/4")
    if '/' in grade:
        try:
            parts = grade.split('/')
            if len(parts) == 2:
                numerator = float(parts[0].strip())
                denominator = float(parts[1].strip())
                if denominator != 0:
                    result = numerator / denominator
                    if 0 <= result <= 7:
                        return str(result)
        except (ValueError, ZeroDivisionError):
            pass
    
    # Map common text variations to standard formats
    correct_variations = [
        'correct', 'right', 'true', 'yes', 'full', 'full credit', 
        'full marks', 'complete', 'valid', 'accepted', 'pass',
        'solved', 'solution correct', 'answer correct', 'perfect',
        'excellent', 'good', 'success', 'accurate'
    ]
    incorrect_variations = [
        'incorrect', 'wrong', 'false', 'no', 'none', 'zero',
        'invalid', 'rejected', 'fail', 'error', 'mistake',
        'unsolved', 'not correct', 'not valid', 'bad', 'poor',
        'unacceptable', 'flawed', 'erroneous'
    ]
    partial_variations = [
        'partial', 'partial credit', 'half', 'incomplete',
        'partially correct', 'partially right', 'some credit',
        'partially solved', 'in progress', 'almost', 'nearly',
        'mostly correct', 'minor error', 'small mistake'
    ]
    
    # Check for exact matches first
    if grade_lower in correct_variations:
        return 'Correct'
    if grade_lower in incorrect_variations:
        return 'Incorrect'
    if grade_lower in partial_variations:
        return 'Partial'
    
    # Substring matching for more flexibility
    if any(v in grade_lower for v in correct_variations):
        return 'Correct'
    elif any(v in grade_lower for v in incorrect_variations):
        return 'Incorrect'
    elif any(v in grade_lower for v in partial_variations):
        return 'Partial'
    
    # Return original if no normalization applied
    return grade


class TaskAgent:
    """Task agent that solves IMO grading problems with enhanced reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def _build_grading_prompt(self, inputs: dict) -> str:
        """Build a comprehensive grading prompt focused on accurate evaluation."""
        domain = inputs.get("domain", "")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        return f"""You are an expert mathematical grader for {domain} problems. Your task is to evaluate student answers accurately and assign appropriate grades.

PROBLEM:
{problem}

OFFICIAL SOLUTION:
{solution}

GRADING GUIDELINES:
{grading_guidelines}

STUDENT'S ANSWER:
{student_answer}

INSTRUCTIONS:
1. Carefully read the problem and official solution
2. Analyze the student's answer step by step:
   - Check if the final answer is correct
   - Evaluate the reasoning process
   - Identify any errors or gaps in logic
   - Note any creative or alternative valid approaches
3. Compare against the grading guidelines to determine the appropriate grade
4. Consider partial credit for incomplete but valid reasoning

GRADING CRITERIA:
- Correct: Full marks for complete and correct solution
- Partial: Partial credit for incomplete or partially correct work
- Incorrect: No credit for wrong answers or invalid reasoning
- Numeric grades (0-7): Use as specified in guidelines

Respond in this exact JSON format:
<json>
{{
    "reasoning": "Detailed analysis of the student's answer including: (1) correctness of final answer, (2) validity of reasoning, (3) completeness of solution, (4) any errors found",
    "response": "GRADE"
}}
</json>

Replace GRADE with the appropriate value from the grading guidelines. Be precise and consistent with your grading."""

    def _extract_prediction(self, msg_history: list[dict]) -> str:
        """Extract prediction from message history with multiple fallback strategies."""
        if not msg_history:
            return "None"
        
        last_message = msg_history[-1].get("text", "")
        if not last_message:
            return "None"
        
        # Strategy 1: Extract from <json> tags (preferred format)
        extracted = _extract_jsons(last_message)
        if extracted:
            return self._get_grade_from_json(extracted[-1])
        
        # Strategy 2: Extract from markdown code blocks
        extracted = _extract_json_from_markdown(last_message)
        if extracted:
            return self._get_grade_from_json(extracted[-1])
        
        # Strategy 3: Look for grade patterns in plain text
        text_grade = self._extract_grade_from_text(last_message)
        if text_grade != "None":
            return text_grade
        
        # Strategy 4: Last resort - look for numeric grades
        numeric_grade = self._extract_numeric_grade(last_message)
        if numeric_grade != "None":
            return numeric_grade
        
        return "None"

    def _get_grade_from_json(self, json_obj: dict) -> str:
        """Extract grade from JSON object with field priority."""
        priority_fields = ["response", "grade", "answer", "result", "score"]
        
        for field in priority_fields:
            if field in json_obj:
                value = json_obj[field]
                if isinstance(value, str):
                    return _normalize_grade(value)
                elif isinstance(value, (int, float)):
                    return str(int(value)) if value == int(value) else str(value)
        
        # Fallback: use first string or numeric value
        for key, value in json_obj.items():
            if isinstance(value, str):
                return _normalize_grade(value)
            elif isinstance(value, (int, float)):
                return str(int(value)) if value == int(value) else str(value)
        
        return "None"

    def _extract_grade_from_text(self, text: str) -> str:
        """Extract grade from plain text using pattern matching."""
        # Look for explicit grade statements with various formats
        patterns = [
            r'response[\s]*[:=][\s]*["\']?([^"\'\n]+)["\']?',
            r'grade[\s]*[:=][\s]*["\']?([^"\'\n]+)["\']?',
            r'(?:the\s+)?grade\s+(?:is|of)\s+["\']?(\d+|[^"\'\n]+)["\']?',
            r'(?:score|points?|mark)[\s]*[:=][\s]*([\d.]+)',
            r'(?:assigned|given|awarded)[\s]+(?:grade|score|mark)[\s]*[:=]?\s*["\']?([^"\'\n]+)["\']?',
            r'(?:final|total|overall)[\s]+(?:grade|score)[\s]*[:=]?\s*["\']?([^"\'\n]+)["\']?',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return _normalize_grade(match.group(1).strip())
        
        # Look for standalone numeric/decimal grades at end of text
        end_match = re.search(r'(?:^|\n)\s*([\d.]+)\s*$', text, re.IGNORECASE)
        if end_match:
            return _normalize_grade(end_match.group(1).strip())
        
        # Look for fraction grades (e.g., "1/2", "3/4")
        fraction_match = re.search(r'(?:^|\n|\s)(\d+/\d+)(?:\s|$|\n)', text, re.IGNORECASE)
        if fraction_match:
            return _normalize_grade(fraction_match.group(1).strip())
        
        # Look for explicit correctness indicators
        correctness_patterns = [
            (r'\b(correct|right|true|valid|accepted|full\s+credit|perfect|excellent)\b', 'Correct'),
            (r'\b(incorrect|wrong|false|invalid|rejected|no\s+credit|flawed|erroneous)\b', 'Incorrect'),
            (r'\b(partial|partially\s+correct|partial\s+credit|half\s+credit|almost|nearly|mostly\s+correct)\b', 'Partial'),
        ]
        for pattern, grade in correctness_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return grade
        
        return "None"

    def _extract_numeric_grade(self, text: str) -> str:
        """Extract numeric grade as a last resort fallback."""
        # Look for standalone numbers 0-7 (common IMO-style grades)
        patterns = [
            r'(?:^|\n|\s)([0-7])(?:\s*$|\s*\n)',
            r'grade\s*(?:is|:)?\s*([0-7])(?:\s|$|\.)',
            r'score\s*(?:is|:)?\s*([0-7])(?:\s|$|\.)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return "None"

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        instruction = self._build_grading_prompt(inputs)

        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
        except Exception as e:
            self.log_fn(f"LLM call failed: {e}")
            return "None", []

        prediction = self._extract_prediction(msg_history)
        return str(prediction), msg_history
