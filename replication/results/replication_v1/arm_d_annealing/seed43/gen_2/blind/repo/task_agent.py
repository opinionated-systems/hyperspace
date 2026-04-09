"""
Task agent: solves a given task with chain-of-thought reasoning.

Reimplemented from facebookresearch/HyperAgents task_agent.py.
Enhanced with step-by-step reasoning for better IMO grading accuracy.

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
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with step-by-step reasoning.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Extract fields for better prompt engineering
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = f"""You are an expert {domain} grader evaluating student solutions.

Your task is to grade a student's answer to a problem by comparing it against the official solution and grading guidelines.

## Problem Statement:
{problem}

## Official Solution:
{solution}

## Grading Guidelines:
{grading_guidelines}

## Student's Answer:
{student_answer}

## Instructions:
Think through this step-by-step:
1. Analyze what the problem is asking for
2. Review the official solution and identify key steps/results
3. Compare the student's answer against the solution
4. Check if the student met all criteria in the grading guidelines
5. Determine if the answer is correct (1), partially correct (0.5), or incorrect (0)

Respond in JSON format with the following schema:
<json>
{{
    "reasoning": "Your detailed step-by-step analysis here",
    "response": 1, 0.5, or 0
}}
</json>

The "response" field must be a number: 1 (correct), 0.5 (partially correct), or 0 (incorrect)."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON with validation
        prediction = "None"
        try:
            extracted = _extract_jsons(msg_history[-1]["text"])
            if extracted:
                last_json = extracted[-1]
                if "response" in last_json:
                    raw_response = last_json["response"]
                    # Validate the response is a valid grade
                    if isinstance(raw_response, (int, float)):
                        if raw_response in [0, 0.5, 1]:
                            prediction = raw_response
                        else:
                            # Round to nearest valid grade
                            prediction = round(raw_response * 2) / 2
                            prediction = max(0, min(1, prediction))
                            self.log_fn(f"Rounded invalid grade {raw_response} to {prediction}")
                    elif isinstance(raw_response, str):
                        # Try to parse string response
                        try:
                            val = float(raw_response)
                            prediction = round(val * 2) / 2
                            prediction = max(0, min(1, prediction))
                        except ValueError:
                            # Check for keywords
                            lower = raw_response.lower()
                            if "correct" in lower and "partial" not in lower:
                                prediction = 1
                            elif "partial" in lower or "half" in lower:
                                prediction = 0.5
                            elif "incorrect" in lower or "wrong" in lower:
                                prediction = 0
                            else:
                                self.log_fn(f"Could not parse response: {raw_response}")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
