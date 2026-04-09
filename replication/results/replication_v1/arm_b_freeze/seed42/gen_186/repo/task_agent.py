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
from typing import Any

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


def _validate_inputs(inputs: dict) -> tuple[bool, str]:
    """Validate that required fields are present in the input.
    
    Returns:
        (is_valid, error_message)
    """
    required_fields = ["domain", "problem", "solution", "grading_guidelines", "student_answer"]
    missing = [f for f in required_fields if f not in inputs]
    if missing:
        return False, f"Missing required fields: {', '.join(missing)}"
    return True, ""


class TaskAgent:
    """Task agent that solves IMO grading problems."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.call_count = 0

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        self.call_count += 1
        
        # Validate inputs
        is_valid, error_msg = _validate_inputs(inputs)
        if not is_valid:
            self.log_fn(f"[Call {self.call_count}] Input validation failed: {error_msg}")
            return f"Error: {error_msg}", []

        # Build a more structured prompt for better grading accuracy
        instruction = f"""You are an expert grader for mathematical olympiad problems.

Your task is to evaluate a student's answer based on the provided problem, solution, and grading guidelines.

Problem Domain: {inputs.get('domain', 'N/A')}

Problem Statement:
{inputs.get('problem', 'N/A')}

Official Solution:
{inputs.get('solution', 'N/A')}

Grading Guidelines:
{inputs.get('grading_guidelines', 'N/A')}

Student's Answer:
{inputs.get('student_answer', 'N/A')}

Please evaluate the student's answer and provide your assessment.

Respond in JSON format with the following schema:
<json>
{{
    "response": "Your evaluation/grade here"
}}
</json>"""

        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
        except Exception as e:
            self.log_fn(f"[Call {self.call_count}] LLM call failed: {e}")
            return f"Error: LLM call failed - {e}", []

        # Extract prediction from JSON
        prediction = "None"
        try:
            extracted = _extract_jsons(msg_history[-1]["text"])
            if extracted and "response" in extracted[-1]:
                prediction = extracted[-1]["response"]
                self.log_fn(f"[Call {self.call_count}] Successfully extracted prediction")
            else:
                self.log_fn(f"[Call {self.call_count}] No valid response field found in JSON")
        except Exception as e:
            self.log_fn(f"[Call {self.call_count}] Error extracting prediction: {e}")

        return str(prediction), msg_history
