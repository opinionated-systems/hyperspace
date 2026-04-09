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


def _extract_numeric_score(text: str) -> str:
    """Extract a numeric score from text.
    
    Looks for patterns like:
    - "7" (just a number)
    - "Score: 7"
    - "Points: 7"
    - "The answer deserves 7 points"
    Returns the first number found, or the original text if no number found.
    """
    # First try to find a number at the start of the string
    text = text.strip()
    
    # Look for explicit score/points patterns
    patterns = [
        r'(?:score|points?|grade|mark)s?\s*[:=]\s*(\d+)',
        r'(?:deserves?|earns?|gets?|receives?)\s+(\d+)\s*(?:score|points?|grade|mark)s?',
        r'^(\d+)\s*(?:score|points?|grade|mark)s?',
        r'^(\d+)$',
        r'^(\d+)[\s\.]',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
    
    # If no pattern matched, look for any standalone number
    numbers = re.findall(r'\b(\d+)\b', text)
    if numbers:
        # Return the first number that looks like a score (typically 0-10)
        for num in numbers:
            if 0 <= int(num) <= 10:
                return num
        # If no score-like number, return the first number found
        return numbers[0]
    
    return text


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
        # Extract fields from inputs for better prompt formatting
        domain = inputs.get("domain", "")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        instruction = f"""You are an expert mathematics grader. Your task is to evaluate a student's answer to a mathematics problem and assign a numeric score.

## Problem:
{problem}

## Correct Solution:
{solution}

## Grading Guidelines:
{grading_guidelines}

## Student's Answer:
{student_answer}

## Your Task:
Carefully evaluate the student's answer against the grading guidelines. The grading guidelines specify what criteria must be met for each score level.

IMPORTANT: You must output ONLY a numeric score (like 7, 0, 3, etc.) representing the points the student earned. Do not output text descriptions or explanations.

Respond in JSON format with the following schema:
<json>
{{
    "response": "NUMERIC_SCORE_HERE"
}}
</json>

Example responses:
- For a fully correct answer: <json>{{"response": "7"}}</json>
- For an incorrect answer: <json>{{"response": "0"}}</json>
- For a partially correct answer: <json>{{"response": "3"}}</json>"""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON
        prediction = "None"
        try:
            extracted = _extract_jsons(msg_history[-1]["text"])
            if extracted and "response" in extracted[-1]:
                raw_response = extracted[-1]["response"]
                # Try to extract numeric score from the response
                prediction = _extract_numeric_score(str(raw_response))
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
