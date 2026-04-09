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


def _extract_response_flexible(text: str) -> str | None:
    """Extract the response value using multiple fallback methods.
    
    Tries various patterns to find the evaluation result.
    """
    text = text.strip()
    
    # Method 1: Try to extract from <json> blocks
    jsons = _extract_jsons(text)
    if jsons:
        for j in jsons:
            if isinstance(j, dict) and "response" in j:
                return j["response"]
    
    # Method 2: Look for JSON-like patterns with "response" field
    import re
    
    # Pattern: "response": "Correct" or 'response': 'Correct'
    response_pattern = r'["\']?response["\']?\s*:\s*["\'](Correct|Incorrect|Partial)["\']'
    match = re.search(response_pattern, text, re.IGNORECASE)
    if match:
        return match.group(1)
    
    # Method 3: Look for standalone JSON objects
    json_pattern = r'\{[^}]*["\']?response["\']?\s*:\s*["\'](Correct|Incorrect|Partial)["\'][^}]*\}'
    match = re.search(json_pattern, text, re.IGNORECASE | re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if "response" in parsed:
                return parsed["response"]
        except:
            pass
    
    # Method 4: Direct keyword search in the text
    text_lower = text.lower()
    # Look for explicit mentions of the verdict
    if '"correct"' in text_lower or "'correct'" in text_lower or "correct" in text_lower[-200:]:
        # Check if it's not negated
        if "incorrect" not in text_lower[-200:]:
            return "Correct"
    if '"incorrect"' in text_lower or "'incorrect'" in text_lower or "incorrect" in text_lower[-200:]:
        return "Incorrect"
    if '"partial"' in text_lower or "'partial'" in text_lower or "partial" in text_lower[-200:]:
        return "Partial"
    
    return None


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
        # Extract fields from inputs for better formatting
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = f"""You are an expert mathematics grader for the International Mathematical Olympiad (IMO). Your task is to evaluate a student's solution to a mathematical problem.

## Problem Statement:
{problem}

## Reference Solution:
{solution}

## Grading Guidelines:
{grading_guidelines}

## Student's Answer:
{student_answer}

## Your Task:
Evaluate the student's answer based on the grading guidelines. Determine if the answer is:
- "Correct" - The student has provided a complete and correct solution
- "Incorrect" - The student's answer is wrong or incomplete
- "Partial" - The student has made partial progress but the solution is incomplete

Provide your reasoning first, then respond with your final evaluation in the following JSON format (you MUST use exactly this format):

<json>
{{
    "response": "Correct"
}}
</json>

OR

<json>
{{
    "response": "Incorrect"
}}
</json>

OR

<json>
{{
    "response": "Partial"
}}
</json>

IMPORTANT: Your final answer MUST be in the JSON format shown above, with exactly one of "Correct", "Incorrect", or "Partial" as the value."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction using flexible extraction
        prediction = "None"
        try:
            last_message = msg_history[-1]["text"]
            extracted = _extract_response_flexible(last_message)
            if extracted:
                prediction = extracted
            else:
                self.log_fn(f"Could not extract prediction from response")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
