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
    Supports numeric grades (0, 1, 2, etc.) and text grades.
    """
    if not isinstance(grade, str):
        grade = str(grade)
    
    grade = grade.strip().lower()
    
    # Direct numeric grade mapping for IMO-style grading (0-7)
    if grade.isdigit():
        num = int(grade)
        if num == 0:
            return '0'
        elif num >= 1:
            return str(num)
    
    # Check for decimal grades
    try:
        num = float(grade)
        if num == 0:
            return '0'
        elif num == int(num):
            return str(int(num))
        else:
            return str(num)
    except ValueError:
        pass
    
    # Map common variations to standard formats
    correct_variations = ['correct', 'right', 'true', 'yes', 'full', 'full credit', 'full marks', 'perfect', 'complete']
    incorrect_variations = ['incorrect', 'wrong', 'false', 'no', 'none', 'zero', 'fail', 'failed', 'incomplete', 'missing']
    partial_variations = ['partial', 'partial credit', 'half', 'some', 'in progress']
    
    if any(v in grade for v in correct_variations):
        return 'Correct'
    elif any(v in grade for v in incorrect_variations):
        return 'Incorrect'
    elif any(v in grade for v in partial_variations):
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

## Problem Statement:
{problem}

## Official Solution:
{solution}

## Grading Guidelines:
{grading_guidelines}

## Student's Answer:
{student_answer}

## Evaluation Framework:

Follow this structured evaluation process:

### Step 1: Problem Understanding
- What is the problem asking for?
- What are the key constraints and conditions?
- What is the expected answer format?

### Step 2: Solution Analysis
- What is the correct approach according to the official solution?
- What are the critical steps that must be present?
- What constitutes a complete vs. incomplete solution?

### Step 3: Student Answer Evaluation
- Did the student understand the problem correctly?
- What approach did the student take?
- Are the student's steps logically valid?
- Did the student show sufficient work and reasoning?
- Is the final answer mathematically correct?

### Step 4: Grade Assignment
Based on the grading guidelines, assign the appropriate grade considering:
- Correctness of the final answer
- Validity of the reasoning process
- Completeness of the solution
- Adherence to the expected solution method

## IMO Grading Scale (0-7 points):
- 7: Complete, correct solution with full reasoning
- 6: Minor flaw in an otherwise correct solution
- 5: Significant progress with substantial solution
- 3-4: Partial progress with some correct elements
- 1-2: Minor progress, some relevant ideas
- 0: No progress, incorrect, or irrelevant

## Response Format:

You MUST respond in JSON format wrapped in <json> tags:

<json>
{{
    "reasoning": "Your detailed step-by-step analysis following the Evaluation Framework above",
    "response": "The final grade you assign - MUST be a single number 0-7 or 'Correct'/'Incorrect'/'Partial'"
}}
</json>

Important: The "response" field must contain ONLY the grade value (e.g., "7", "0", "Correct", "Incorrect", "Partial"), nothing else."""

    def _extract_prediction(self, msg_history: list[dict]) -> str:
        """Extract prediction from message history with multiple fallback strategies."""
        if not msg_history:
            return "None"
        
        last_message = msg_history[-1].get("text", "")
        
        # Strategy 1: Extract from <json> tags
        extracted = _extract_jsons(last_message)
        if extracted:
            result = self._get_grade_from_json(extracted[-1])
            if result != "None":
                return result
        
        # Strategy 2: Extract from markdown code blocks
        extracted = _extract_json_from_markdown(last_message)
        if extracted:
            result = self._get_grade_from_json(extracted[-1])
            if result != "None":
                return result
        
        # Strategy 3: Look for grade patterns in plain text
        result = self._extract_grade_from_text(last_message)
        if result != "None":
            return result
        
        # Strategy 4: Look for standalone numbers 0-7 in the last few lines
        lines = last_message.strip().split('\n')
        for line in reversed(lines[-5:]):  # Check last 5 lines
            line = line.strip()
            # Look for standalone digits 0-7
            if line.isdigit() and 0 <= int(line) <= 7:
                return line
            # Look for patterns like "Grade: 7" or "Score: 0"
            match = re.search(r'(?:grade|score|points|mark)[\s]*[:\-]?\s*([0-7])\b', line, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return "None"

    def _get_grade_from_json(self, json_obj: dict) -> str:
        """Extract grade from JSON object with field priority."""
        if not isinstance(json_obj, dict):
            return "None"
        
        # Priority order for grade fields
        priority_fields = ["response", "grade", "answer", "result", "score", "evaluation", "prediction"]
        
        for field in priority_fields:
            if field in json_obj:
                value = json_obj[field]
                if isinstance(value, str):
                    normalized = _normalize_grade(value)
                    if normalized != "None":
                        return normalized
                elif isinstance(value, (int, float)):
                    return str(int(value)) if value == int(value) else str(value)
        
        # If no recognized field, use the first string/numeric value found
        for key, value in json_obj.items():
            if isinstance(value, str):
                normalized = _normalize_grade(value)
                if normalized != "None":
                    return normalized
            elif isinstance(value, (int, float)):
                return str(int(value)) if value == int(value) else str(value)
        
        return "None"

    def _extract_grade_from_text(self, text: str) -> str:
        """Extract grade from plain text using pattern matching."""
        # Look for explicit grade statements with improved patterns
        patterns = [
            # Direct grade assignment patterns
            r'grade[\s]*[:=][\s]*["\']?([^"\'\n,]+)["\']?',
            r'response[\s]*[:=][\s]*["\']?([^"\'\n,]+)["\']?',
            r'final grade[\s]*[:=][\s]*["\']?([^"\'\n,]+)["\']?',
            r'assign[\s]+["\']?([^"\'\n,]+)["\']?',
            # IMO-specific patterns
            r'score[\s]*[:=][\s]*["\']?([^"\'\n,]+)["\']?',
            r'points[\s]*[:=][\s]*["\']?([^"\'\n,]+)["\']?',
            r'award[\s]+["\']?([^"\'\n,]+)["\']?',
            # Grade at end of sentence
            r'grade (?:is|of)[\s]+["\']?([^"\'\n,.]+)["\']?',
            r'(?:is|should be)[\s]+["\']?([^"\'\n,.]{1,20})["\']?[\s]*\.?$',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                return _normalize_grade(match.group(1).strip())
        
        # Fallback: look for standalone numeric grades (0-7 for IMO)
        # Search for patterns like "Grade: 7" or "Score: 0"
        numeric_patterns = [
            r'(?:grade|score|points)[\s]*[:\-]?\s*([0-7])\b',
            r'\b([0-7])\s*(?:points?|marks?)\b',
        ]
        for pattern in numeric_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return "None"

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with enhanced error handling.

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

        # Extract prediction using multiple strategies
        prediction = self._extract_prediction(msg_history)
        
        if prediction == "None":
            self.log_fn(f"Failed to extract prediction from response: {response[:200] if response else 'empty'}")

        return str(prediction), msg_history
