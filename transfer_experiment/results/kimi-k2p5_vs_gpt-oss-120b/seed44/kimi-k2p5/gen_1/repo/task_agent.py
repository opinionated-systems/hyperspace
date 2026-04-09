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
        # Extract key fields for better prompting
        domain = inputs.get("domain", "")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        instruction = f"""You are an expert mathematical grader. Your task is to evaluate a student's answer to a mathematics problem.

DOMAIN: {domain}

PROBLEM:
{problem}

OFFICIAL SOLUTION:
{solution}

GRADING GUIDELINES:
{grading_guidelines}

STUDENT'S ANSWER TO EVALUATE:
{student_answer}

Based on the grading guidelines, determine if the student's answer is:
- "correct" - if the answer is fully correct
- "incorrect" - if the answer is wrong or significantly flawed  
- "partial" - if the answer has some correct elements but is incomplete or has minor errors

You must respond with ONLY a JSON object in the following format (no other text):
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

        # Extract prediction from JSON with multiple fallback strategies
        prediction = "None"
        try:
            # Try to extract from JSON tags
            extracted = _extract_jsons(msg_history[-1]["text"])
            if extracted and "response" in extracted[-1]:
                prediction = extracted[-1]["response"]
            else:
                # Fallback: look for direct answer in text
                text = msg_history[-1]["text"].lower()
                if "correct" in text and "incorrect" not in text and "partial" not in text:
                    prediction = "correct"
                elif "partial" in text:
                    prediction = "partial"
                elif "incorrect" in text:
                    prediction = "incorrect"
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            # Final fallback: simple keyword matching
            text = msg_history[-1]["text"].lower()
            if "partial" in text:
                prediction = "partial"
            elif "incorrect" in text:
                prediction = "incorrect"
            elif "correct" in text:
                prediction = "correct"

        return str(prediction), msg_history
