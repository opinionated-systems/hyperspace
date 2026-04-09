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

    Handles various grade formats and normalizes them.
    Enhanced to handle numeric grades, IMO-style scoring, and more variations.
    """
    if grade is None:
        return 'None'
    
    if not isinstance(grade, str):
        grade = str(grade)
    
    grade = grade.strip().lower()
    
    # Handle empty strings
    if not grade:
        return 'None'
    
    # First, check for numeric grades (0, 1, 2, etc.)
    # These are common in IMO-style grading
    try:
        numeric_grade = float(grade)
        if numeric_grade == 0:
            return 'Incorrect'
        elif numeric_grade >= 1:
            return 'Correct'
        elif 0 < numeric_grade < 1:
            return 'Partial'
    except ValueError:
        pass
    
    # Map common variations to standard formats
    correct_variations = [
        'correct', 'right', 'true', 'yes', 'full', 'full credit', 
        'full marks', 'complete', 'valid', 'accepted', 'pass', 'solved',
        'success', 'accurate', 'perfect', 'excellent', 'good'
    ]
    incorrect_variations = [
        'incorrect', 'wrong', 'false', 'no', 'none', 'zero',
        'invalid', 'rejected', 'fail', 'error', 'mistake', 'unsolved',
        'failure', 'bad', 'poor', 'unacceptable'
    ]
    partial_variations = [
        'partial', 'partial credit', 'half', 'incomplete',
        'partially correct', 'partially right', 'some credit',
        'partially solved', 'almost', 'nearly', 'minor errors',
        'mostly correct', 'significant progress'
    ]
    
    # Check for exact matches first, then substring matches
    if grade in correct_variations:
        return 'Correct'
    if grade in incorrect_variations:
        return 'Incorrect'
    if grade in partial_variations:
        return 'Partial'
    
    # Substring matching for more flexibility
    if any(v in grade for v in correct_variations):
        return 'Correct'
    elif any(v in grade for v in incorrect_variations):
        return 'Incorrect'
    elif any(v in grade for v in partial_variations):
        return 'Partial'
    
    # Handle IMO-style letter grades (A, B, C, D, F, etc.)
    if grade in ['a', 'b']:
        return 'Correct'
    if grade in ['d', 'f']:
        return 'Incorrect'
    if grade == 'c':
        return 'Partial'
    
    # Return original if no normalization applied
    return grade.strip()


class TaskAgent:
    """Task agent that solves IMO grading problems with enhanced reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def _build_grading_prompt(self, inputs: dict) -> str:
        """Build a structured grading prompt with clear evaluation criteria."""
        domain = inputs.get("domain", "")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        return f"""You are an expert mathematical grader specializing in {domain} problems.

Your task is to evaluate a student's answer to a mathematics problem and assign a grade.

Problem:
{problem}

Official Solution:
{solution}

Grading Guidelines:
{grading_guidelines}

Student Answer:
{student_answer}

Evaluate the student's answer carefully. Consider:
1. Is the answer mathematically correct?
2. Does it follow the grading guidelines?
3. Is the reasoning sound?

Respond with a JSON object in this format:
<json>{{"grade": "Correct" or "Incorrect" or "Partial"}}</json>

Use "Correct" if the answer is fully correct, "Incorrect" if completely wrong, and "Partial" if partially correct."""

    def _extract_prediction(self, msg_history: list[dict]) -> str:
        """Extract the grade prediction from the LLM response.
        
        Uses multiple extraction strategies and fallback mechanisms
        to handle various response formats from different LLMs.
        """
        if not msg_history:
            return "None"
        
        # Get the last assistant message
        text = ""
        for msg in reversed(msg_history):
            if msg.get("role") == "assistant":
                text = msg.get("text", "")
                break
        
        if not text:
            return "None"
        
        # Strategy 1: Try to extract JSON from <json> tags
        json_results = _extract_jsons(text)
        if json_results:
            for result in json_results:
                if isinstance(result, dict) and "grade" in result:
                    return _normalize_grade(result["grade"])
        
        # Strategy 2: Try markdown code blocks
        markdown_results = _extract_json_from_markdown(text)
        if markdown_results:
            for result in markdown_results:
                if isinstance(result, dict) and "grade" in result:
                    return _normalize_grade(result["grade"])
        
        # Strategy 3: Look for explicit grade statements with various formats
        patterns = [
            # Standard grade assignment patterns
            r'grade[\s]*[:=][\s]*["\']?([^"\'\n]+)["\']?',
            r'response[\s]*[:=][\s]*["\']?([^"\'\n]+)["\']?',
            r'final grade[\s]*[:=][\s]*["\']?([^"\'\n]+)["\']?',
            r'assign[\s]+["\']?([^"\'\n]+)["\']?',
            # IMO-style numeric grades (0, 1, 2, 3, etc.)
            r'(?:score|points?|mark)[\s]*[:=][\s]*(\d+)',
            r'(?:the\s+)?grade\s+(?:is|of)\s+["\']?(\d+|[^"\'\n]+)["\']?',
            # Evaluation result patterns
            r'evaluation[\s]*[:=][\s]*["\']?([^"\'\n]+)["\']?',
            r'result[\s]*[:=][\s]*["\']?([^"\'\n]+)["\']?',
            # Answer/verdict patterns
            r'(?:the\s+)?(?:answer|verdict|decision)\s+(?:is\s+)?["\']?([^"\'\n]+)["\']?',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return _normalize_grade(match.group(1).strip())
        
        # Strategy 4: Look for standalone numeric grades at end of text
        end_patterns = [
            r'(?:^|\n)\s*(\d+)\s*$',  # Standalone number at end
            r'grade\s*[:\-]?\s*(\d+)(?:\s|$)',  # Grade followed by number
            r'(?:score|mark)\s*[:\-]?\s*(\d+)(?:\s|$)',
        ]
        for pattern in end_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return _normalize_grade(match.group(1).strip())
        
        # Strategy 5: Look for explicit correctness indicators
        correctness_patterns = [
            (r'\b(correct|right|true|valid|accepted|full\s+credit)\b', 'Correct'),
            (r'\b(incorrect|wrong|false|invalid|rejected|no\s+credit)\b', 'Incorrect'),
            (r'\b(partial|partially\s+correct|partial\s+credit|half\s+credit)\b', 'Partial'),
        ]
        for pattern, grade in correctness_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return grade
        
        return "None"

    def _build_simple_prompt(self, inputs: dict) -> str:
        """Build a simplified grading prompt as fallback for retry."""
        domain = inputs.get("domain", "")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        return f"""Grade this {domain} problem.

Problem: {problem}

Official Solution: {solution}

Grading Guidelines: {grading_guidelines}

Student Answer: {student_answer}

Respond with ONLY a JSON object in this exact format:
<json>{{"grade": "0" or "1" or "2" or "Correct" or "Incorrect" or "Partial"}}</json>"""

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with enhanced error handling.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # First attempt with detailed prompt
        instruction = self._build_grading_prompt(inputs)
        all_msg_history = []
        last_response = ""

        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
            last_response = response
            all_msg_history.extend(msg_history)
        except Exception as e:
            self.log_fn(f"LLM call failed: {e}")
            return "None", []

        # Extract prediction using multiple strategies
        prediction = self._extract_prediction(msg_history)
        
        # Retry with simplified prompt if extraction failed
        if prediction == "None":
            self.log_fn(f"First attempt failed to extract prediction, retrying with simplified prompt")
            try:
                simple_instruction = self._build_simple_prompt(inputs)
                response, retry_history, info = get_response_from_llm(
                    msg=simple_instruction,
                    model=self.model,
                    msg_history=[],
                )
                last_response = response
                all_msg_history.extend(retry_history)
                prediction = self._extract_prediction(retry_history)
            except Exception as e:
                self.log_fn(f"Retry LLM call failed: {e}")
        
        if prediction == "None":
            self.log_fn(f"Failed to extract prediction from response: {last_response[:200] if last_response else 'empty'}")
        else:
            self.log_fn(f"Successfully extracted prediction: {prediction}")

        return str(prediction), all_msg_history
