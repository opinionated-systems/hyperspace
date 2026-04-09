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

__version__ = "1.3.0"


def _format_grading_input(inputs: dict) -> str:
    """Format grading inputs into a readable string for logging."""
    domain = inputs.get("domain", "N/A")
    problem_preview = inputs.get("problem", "")[:80]
    return f"[Domain: {domain}] Problem: {problem_preview}..."


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

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Build a structured prompt with clearly labeled fields
        domain = inputs.get("domain", "")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = f"""You are an expert grading agent. Your task is to evaluate a student's answer and classify it as "Correct", "Incorrect", or "Partial".

Domain: {domain}

Problem:
{problem}

Reference Solution:
{solution}

Grading Guidelines:
{grading_guidelines}

Student's Answer:
{student_answer}

Carefully analyze the student's answer against the reference solution and grading guidelines. Check each criterion in the grading guidelines to determine if the student's answer meets the requirements.

Respond in JSON format with the following schema:
<json>
{{
    "response": "your detailed evaluation here",
    "prediction": "Correct" or "Incorrect" or "Partial"
}}
</json>

Rules for classification:
- "Correct": The student's answer fully satisfies all criteria in the grading guidelines and reaches the correct conclusion.
- "Partial": The student's answer shows some correct reasoning or meets some criteria but is incomplete or has significant gaps.
- "Incorrect": The student's answer is fundamentally wrong, reaches an incorrect conclusion, or fails to meet the key criteria."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON
        prediction = "None"
        try:
            extracted = _extract_jsons(msg_history[-1]["text"])
            if extracted:
                last_json = extracted[-1]
                # Prefer explicit "prediction" field if present
                if "prediction" in last_json:
                    prediction = last_json["prediction"]
                elif "response" in last_json:
                    prediction = last_json["response"]
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
