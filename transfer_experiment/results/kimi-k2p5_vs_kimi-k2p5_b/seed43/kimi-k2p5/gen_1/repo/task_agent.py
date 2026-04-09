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
        # Extract fields from inputs with defaults
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = f"""You are an expert mathematical grader evaluating student solutions to competition mathematics problems.

Your task is to evaluate the student's answer and determine if it is:
- "correct": The student has provided a complete and correct solution.
- "incorrect": The student has provided an incorrect solution or the answer is wrong.
- "partial": The student has made partial progress but the solution is incomplete or has significant gaps.

## Problem Statement:
{problem}

## Official Solution:
{solution}

## Grading Guidelines:
{grading_guidelines}

## Student's Answer:
{student_answer}

## Your Task:
Carefully analyze the student's answer against the official solution and grading guidelines. Determine whether the student's answer should be classified as "correct", "incorrect", or "partial".

Respond in JSON format with the following schema:
<json>
{{
    "response": "correct" | "incorrect" | "partial"
}}
</json>

Provide only the JSON response with your classification."""

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
                prediction = extracted[-1]["response"]
                # Normalize the prediction to expected values
                pred_str = str(prediction).lower().strip()
                if pred_str in ["correct", "incorrect", "partial"]:
                    prediction = pred_str
                elif "correct" in pred_str and "partial" not in pred_str and "incorrect" not in pred_str:
                    prediction = "correct"
                elif "partial" in pred_str:
                    prediction = "partial"
                elif "incorrect" in pred_str or "wrong" in pred_str:
                    prediction = "incorrect"
                else:
                    prediction = "incorrect"  # Default fallback
            else:
                self.log_fn("No valid JSON response found, defaulting to 'incorrect'")
                prediction = "incorrect"
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            prediction = "incorrect"

        return str(prediction), msg_history
