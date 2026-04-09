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
        # Extract fields from inputs for better prompt construction
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        instruction = f"""You are an expert mathematical grader for International Mathematical Olympiad (IMO) problems.

Your task is to evaluate a student's answer to a mathematical problem and assign a grade based on the provided grading guidelines.

## Problem Statement:
{problem}

## Official Solution:
{solution}

## Grading Guidelines:
{grading_guidelines}

## Student's Answer:
{student_answer}

## Your Task:
Carefully analyze the student's answer against the official solution and grading guidelines. Determine if the student's answer is:
- "Correct" - The answer is fully correct and complete
- "Incorrect" - The answer is wrong or fundamentally flawed
- "Partial" - The answer has some correct elements but is incomplete or has errors

Provide your assessment in the following JSON format:
<json>
{{
    "response": "Correct" | "Incorrect" | "Partial"
}}
</json>

Respond with ONLY the JSON block containing your grading decision."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON
        prediction = "None"
        try:
            if msg_history and len(msg_history) > 0:
                extracted = _extract_jsons(msg_history[-1]["text"])
                if extracted:
                    last_extract = extracted[-1]
                    if "response" in last_extract:
                        prediction = last_extract["response"]
                    elif "grade" in last_extract:
                        prediction = last_extract["grade"]
                    elif "evaluation" in last_extract:
                        prediction = last_extract["evaluation"]
                    # Normalize the prediction to expected values
                    prediction_str = str(prediction).strip().lower()
                    if "correct" in prediction_str and "incorrect" not in prediction_str and "partial" not in prediction_str:
                        prediction = "Correct"
                    elif "partial" in prediction_str:
                        prediction = "Partial"
                    elif "incorrect" in prediction_str or "wrong" in prediction_str:
                        prediction = "Incorrect"
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
