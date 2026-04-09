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

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Extract fields from inputs for better prompting
        domain = inputs.get("domain", "")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        instruction = f"""You are an expert mathematical grader for IMO (International Mathematical Olympiad) problems.

Your task is to evaluate a student's answer based on the problem, official solution, and grading guidelines.

DOMAIN: {domain}

PROBLEM:
{problem}

OFFICIAL SOLUTION:
{solution}

GRADING GUIDELINES:
{grading_guidelines}

STUDENT'S ANSWER:
{student_answer}

Evaluate the student's answer carefully. Consider:
1. Does the student understand the problem correctly?
2. Are the key mathematical insights present?
3. Is the reasoning logically sound?
4. Does the answer match the official solution's conclusions?

Respond with ONE of these exact labels in JSON format:
- "correct" - The answer is fully correct with proper reasoning
- "incorrect" - The answer is wrong or contains fundamental errors
- "partial" - The answer has some correct elements but is incomplete or has minor issues

Respond in JSON format:
<json>
{{
    "response": "correct" | "incorrect" | "partial"
}}
</json>"""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON - try both response and msg_history
        prediction = "None"
        try:
            # First try to extract from the response text directly
            extracted = _extract_jsons(response)
            if extracted and "response" in extracted[-1]:
                prediction = extracted[-1]["response"]
            else:
                # Fallback: try msg_history
                for msg in reversed(msg_history):
                    if "text" in msg:
                        extracted = _extract_jsons(msg["text"])
                        if extracted and "response" in extracted[-1]:
                            prediction = extracted[-1]["response"]
                            break
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        # Validate prediction is one of the expected values
        valid_predictions = {"correct", "incorrect", "partial"}
        if prediction not in valid_predictions:
            self.log_fn(f"Invalid prediction '{prediction}', defaulting to incorrect")
            prediction = "incorrect"

        return str(prediction), msg_history
