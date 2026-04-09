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


def _extract_json_fuzzy(text: str) -> list[dict] | None:
    """Fallback JSON extraction for malformed responses.
    
    Attempts to find JSON objects even without proper <json> tags
    by looking for curly brace patterns.
    """
    results = []
    # Try to find JSON objects between curly braces
    brace_count = 0
    start_idx = -1
    for i, char in enumerate(text):
        if char == '{':
            if brace_count == 0:
                start_idx = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start_idx != -1:
                try:
                    json_str = text[start_idx:i+1]
                    results.append(json.loads(json_str))
                except json.JSONDecodeError:
                    pass
                start_idx = -1
    return results or None


def _normalize_grade(grade: str) -> str:
    """Normalize grade format for consistency.
    
    Handles various formats like '7/7', '7', 'Full marks', etc.
    """
    if not grade or not isinstance(grade, str):
        return str(grade) if grade else "None"
    
    grade = grade.strip()
    
    # Check for fraction format (e.g., "7/7", "3/7")
    if '/' in grade:
        return grade
    
    # Check for numeric only (assume out of 7 for IMO)
    try:
        num = float(grade)
        if 0 <= num <= 7:
            return f"{int(num)}/7"
    except ValueError:
        pass
    
    # Check for text-based grades
    lower = grade.lower()
    if any(word in lower for word in ['full', 'complete', 'correct', '7']):
        return '7/7'
    if any(word in lower for word in ['partial', 'partial credit']):
        return grade  # Keep as is for partial credit
    if any(word in lower for word in ['zero', 'none', 'incorrect', 'wrong', '0']):
        return '0/7'
    
    return grade


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
        # Extract key fields for structured prompting
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = f"""You are an expert IMO (International Mathematical Olympiad) grader. Your task is to evaluate a student's solution to a mathematical problem.

## Domain
{domain}

## Problem Statement
{problem}

## Official Solution
{solution}

## Grading Guidelines
{grading_guidelines}

## Student's Answer
{student_answer}

## Your Task
1. First, analyze the student's answer step by step
2. Compare it against the official solution and grading guidelines
3. Identify any errors, omissions, or creative valid approaches
4. Provide a detailed reasoning of your evaluation
5. Finally, give your grade/assessment in the response field

IMPORTANT: IMO problems are graded on a scale of 0-7 points. Use the format "X/7" for your grade.

Respond in JSON format with the following schema:
<json>
{{
    "reasoning": "Your detailed step-by-step analysis and evaluation reasoning",
    "response": "Your final grade/assessment (e.g., '7/7', 'Partial credit: 3/7', 'Incorrect: 0/7')"
}}
</json>"""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON with fallback strategies
        prediction = "None"
        try:
            # Try primary extraction method first
            extracted = _extract_jsons(msg_history[-1]["text"])
            
            # Fallback to fuzzy extraction if primary fails
            if not extracted:
                extracted = _extract_json_fuzzy(msg_history[-1]["text"])
                if extracted:
                    self.log_fn("Used fuzzy JSON extraction fallback")
            
            if extracted:
                last_json = extracted[-1]
                # Try multiple possible field names for the grade
                for field in ["response", "grade", "score", "assessment", "result"]:
                    if field in last_json:
                        prediction = last_json[field]
                        break
                
                # Normalize the grade format
                prediction = _normalize_grade(prediction)
                
                self.log_fn(f"Extracted grade: {prediction}")
            else:
                self.log_fn("No JSON found in response, attempting text extraction")
                # Last resort: try to extract grade from text directly
                text = msg_history[-1]["text"]
                # Look for patterns like "7/7", "3/7", etc.
                grade_match = re.search(r'(\d+)/7', text)
                if grade_match:
                    prediction = grade_match.group(0)
                    self.log_fn(f"Extracted grade via regex: {prediction}")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
