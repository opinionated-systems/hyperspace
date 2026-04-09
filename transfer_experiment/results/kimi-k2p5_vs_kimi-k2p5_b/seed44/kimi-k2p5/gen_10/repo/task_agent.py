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

VALID_GRADES = ["correct", "incorrect", "partial", "almost"]


def _extract_grade(text: str) -> str:
    """Extract grade from LLM response text.
    
    Tries multiple strategies in order of reliability:
    1. Parse JSON from <json> or ```json blocks
    2. Look for "grade": "value" pattern
    3. Look for quoted grade values
    4. Look for standalone grade keywords
    """
    text_lower = text.lower()
    
    # Strategy 1: Extract JSON from <json> blocks
    json_blocks = []
    start = 0
    while True:
        json_start = text.find("<json>", start)
        if json_start == -1:
            break
        json_end = text.find("</json>", json_start)
        if json_end == -1:
            break
        json_blocks.append(text[json_start + 6:json_end].strip())
        start = json_end + 7
    
    # Also try ```json blocks
    start = 0
    while True:
        json_start = text.find("```json", start)
        if json_start == -1:
            break
        json_end = text.find("```", json_start + 7)
        if json_end == -1:
            break
        json_blocks.append(text[json_start + 7:json_end].strip())
        start = json_end + 3
    
    # Try to parse each JSON block
    for block in json_blocks:
        try:
            data = json.loads(block)
            if isinstance(data, dict):
                # Try "grade" field first
                if "grade" in data:
                    val = str(data["grade"]).lower().strip().strip('"').strip("'")
                    if val in VALID_GRADES:
                        return val
                # Try other common field names
                for field in ["result", "prediction", "response", "answer"]:
                    if field in data:
                        val = str(data[field]).lower().strip().strip('"').strip("'")
                        if val in VALID_GRADES:
                            return val
                # Try any string value in the dict
                for val in data.values():
                    val_str = str(val).lower().strip().strip('"').strip("'")
                    if val_str in VALID_GRADES:
                        return val_str
        except json.JSONDecodeError:
            continue
    
    # Strategy 2: Look for "grade": "value" pattern (even outside JSON blocks)
    grade_pattern = re.search(r'["\']?grade["\']?\s*[:=]\s*["\']?([a-z]+)["\']?', text_lower)
    if grade_pattern:
        val = grade_pattern.group(1)
        if val in VALID_GRADES:
            return val
    
    # Strategy 3: Look for quoted grade values
    for grade in VALID_GRADES:
        if f'"{grade}"' in text_lower or f"'{grade}'" in text_lower:
            return grade
    
    # Strategy 4: Look for standalone grade keywords in last 200 chars (conclusion area)
    conclusion = text_lower[-200:] if len(text_lower) > 200 else text_lower
    # Check in order of specificity
    for grade in ["almost", "partial", "incorrect", "correct"]:
        if grade in conclusion:
            return grade
    
    # Strategy 5: Look anywhere in text as last resort
    for grade in ["almost", "partial", "incorrect", "correct"]:
        if grade in text_lower:
            return grade
    
    return "none"


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
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = f"""You are an expert mathematical olympiad grader. Grade the student's solution based on the official solution and grading guidelines.

## Problem:
{problem}

## Official Solution:
{solution}

## Student's Answer:
{student_answer}

## Grading Guidelines (FOLLOW THESE EXACTLY):
{grading_guidelines}

## Grade Definitions:
- **correct**: Complete, correct solution with valid proof
- **incorrect**: Wrong, fundamental errors, no meaningful progress
- **partial**: Significant progress but incomplete (found invariant/lemma but didn't finish)
- **almost**: Nearly complete, only minor errors/gaps

## Critical Rules:
1. The grading guidelines explicitly state what constitutes each grade - FOLLOW THEM
2. If guidelines list criteria under (Partial), solutions meeting those criteria MUST be graded PARTIAL
3. Only grade INCORRECT if there is NO meaningful progress
4. When uncertain, choose the LOWER grade

## Response Format (JSON ONLY):
<json>
{{
    "grade": "correct" | "incorrect" | "partial" | "almost"
}}
</json>

Grade:"""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from the last message
        prediction = "none"
        try:
            last_message = msg_history[-1]["text"]
            prediction = _extract_grade(last_message)
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return prediction, msg_history
