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
    Also handles markdown code blocks (```json...```) for flexibility.
    """
    results = []
    search_from = 0
    
    # First try <json>...</json> tags
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
            # Try to extract JSON from within the content if it's wrapped
            try:
                # Look for JSON object boundaries
                brace_start = inner.find('{')
                brace_end = inner.rfind('}')
                if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
                    results.append(json.loads(inner[brace_start:brace_end+1]))
            except json.JSONDecodeError:
                continue
    
    # Also try markdown code blocks if no results yet
    if not results:
        search_from = 0
        while True:
            start = text.find("```json", search_from)
            if start == -1:
                # Try without 'json' specifier
                start = text.find("```", search_from)
                if start == -1:
                    break
                end_marker = text.find("```", start + 3)
            else:
                end_marker = text.find("```", start + 7)
            
            if end_marker == -1:
                break
            
            if text.find("```json", search_from) == start:
                inner = text[start + 7:end_marker].strip()
            else:
                inner = text[start + 3:end_marker].strip()
            
            search_from = end_marker + 3
            try:
                results.append(json.loads(inner))
            except json.JSONDecodeError:
                # Try to find JSON object within the code block
                try:
                    brace_start = inner.find('{')
                    brace_end = inner.rfind('}')
                    if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
                        results.append(json.loads(inner[brace_start:brace_end+1]))
                except json.JSONDecodeError:
                    continue
    
    return results or None


def _extract_any_json(text: str) -> list[dict] | None:
    """Fallback JSON extraction that looks for any JSON objects in text."""
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
                    obj = json.loads(text[start_idx:i+1])
                    results.append(obj)
                except json.JSONDecodeError:
                    pass
                start_idx = -1
    return results or None


def _normalize_grade(grade: str) -> str:
    """Normalize grade string to standard format.
    
    Converts various grade formats to a consistent set of values:
    - 'Correct', 'True', 'Yes', '1', '7/7', 'Full' -> 'Correct'
    - 'Incorrect', 'False', 'No', '0', '0/7', 'None' -> 'Incorrect'
    - 'Partial', 'Partially Correct', 'Partial Credit' -> 'Partial'
    """
    if not isinstance(grade, str):
        grade = str(grade)
    
    grade_lower = grade.lower().strip()
    
    # Correct variations
    if any(x in grade_lower for x in ['correct', 'true', 'yes', 'full', 'right', 'valid']):
        if 'partial' not in grade_lower and 'incomplete' not in grade_lower:
            return 'Correct'
    
    # Incorrect variations
    if any(x in grade_lower for x in ['incorrect', 'false', 'no', 'wrong', 'invalid', 'none', 'error']):
        return 'Incorrect'
    
    # Partial variations
    if any(x in grade_lower for x in ['partial', 'incomplete', 'some']):
        return 'Partial'
    
    # Numeric grades (IMO-style 0-7 scale)
    try:
        num = float(grade.split('/')[0]) if '/' in grade else float(grade)
        if num >= 6:
            return 'Correct'
        elif num <= 1:
            return 'Incorrect'
        else:
            return 'Partial'
    except (ValueError, TypeError):
        pass
    
    return grade


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

Your task is to evaluate a student's answer to a mathematics problem and provide a grade with detailed reasoning.

DOMAIN: {domain}

PROBLEM:
{problem}

OFFICIAL SOLUTION:
{solution}

GRADING GUIDELINES:
{grading_guidelines}

STUDENT'S ANSWER:
{student_answer}

Think step by step and provide a thorough analysis:
1. PROBLEM ANALYSIS: Identify the key mathematical concepts, theorems, and techniques required
2. SOLUTION REVIEW: Analyze the official solution's approach, key steps, and expected answer format
3. STUDENT WORK ANALYSIS: 
   - Identify what approach the student took
   - Note any correct steps or valid insights
   - Identify errors, gaps, or misconceptions
4. GRADING CRITERIA CHECK:
   - Verify if the student met each criterion in the grading guidelines
   - Note partial credit for incomplete but valid reasoning
5. FINAL DETERMINATION: Assign grade based on completeness, correctness, and adherence to guidelines

Respond in JSON format with the following schema:
<json>
{{
    "reasoning": "Your detailed chain-of-thought analysis covering all 5 steps above",
    "response": "The final grade/prediction (e.g., 'Correct', 'Incorrect', 'Partial', or a numeric score)"
}}
</json>

Important: Be objective and consistent. Award partial credit when the student shows valid reasoning even if the final answer is incorrect."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON with fallback mechanisms
        prediction = "None"
        try:
            last_message = msg_history[-1]["text"]
            
            # Try primary extraction method
            extracted = _extract_jsons(last_message)
            
            # Fallback to generic JSON extraction if primary fails
            if extracted is None:
                extracted = _extract_any_json(last_message)
            
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
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        # Normalize the grade to standard format
        prediction = _normalize_grade(prediction)

        return str(prediction), msg_history
