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


class TaskAgent:
    """Task agent that solves IMO grading problems."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def _validate_inputs(self, inputs: dict) -> tuple[bool, str]:
        """Validate that all required fields are present in inputs.
        
        Returns:
            (is_valid, error_message)
        """
        required_fields = ["domain", "problem", "solution", "grading_guidelines", "student_answer"]
        missing = [f for f in required_fields if f not in inputs]
        if missing:
            return False, f"Missing required fields: {missing}"
        return True, ""

    def _build_grading_prompt(self, inputs: dict) -> str:
        """Build a structured prompt for the IMO grading task."""
        return f"""You are an expert mathematics grader for the International Mathematical Olympiad (IMO).

Your task is to evaluate a student's solution to a mathematical problem and provide a grade.

PROBLEM DOMAIN: {inputs['domain']}

PROBLEM STATEMENT:
{inputs['problem']}

OFFICIAL SOLUTION:
{inputs['solution']}

GRADING GUIDELINES:
{inputs['grading_guidelines']}

STUDENT'S ANSWER TO EVALUATE:
{inputs['student_answer']}

Instructions:
1. Carefully read the problem, official solution, and grading guidelines
2. Analyze the student's answer for correctness and completeness
3. Compare the student's approach with the official solution
4. Assign an appropriate grade based on the grading guidelines
5. Provide your grade in the JSON format below

Respond in JSON format with the following schema:
<json>
{{
    "response": "Your grade here (e.g., '7', '6', '0', 'Partial credit', etc.)"
}}
</json>

Important: The response field must contain only the grade value, without additional explanation."""

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Validate inputs
        is_valid, error_msg = self._validate_inputs(inputs)
        if not is_valid:
            self.log_fn(f"Input validation failed: {error_msg}")
            return f"Error: {error_msg}", []

        # Build structured prompt
        instruction = self._build_grading_prompt(inputs)

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON with improved error handling
        prediction = "None"
        try:
            if msg_history and len(msg_history) > 0:
                extracted = _extract_jsons(msg_history[-1]["text"])
                if extracted:
                    last_extract = extracted[-1]
                    if isinstance(last_extract, dict) and "response" in last_extract:
                        prediction = last_extract["response"]
                        # Validate prediction is a string or number
                        if not isinstance(prediction, (str, int, float)):
                            prediction = str(prediction)
                    else:
                        self.log_fn(f"No 'response' key found in extracted JSON: {last_extract}")
                else:
                    self.log_fn("No JSON blocks found in response")
            else:
                self.log_fn("Empty message history")
        except json.JSONDecodeError as e:
            self.log_fn(f"JSON decode error: {e}")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
