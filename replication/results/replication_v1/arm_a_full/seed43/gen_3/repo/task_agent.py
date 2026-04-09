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
        # Extract fields for structured prompting
        domain = inputs.get("domain", "")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = f"""You are an expert mathematical grader specializing in IMO (International Mathematical Olympiad) problems.

Your task is to evaluate a student's answer to a mathematical problem and assign a numerical score.

## Problem Domain
{domain}

## Problem Statement
{problem}

## Official Solution
{solution}

## Grading Guidelines
{grading_guidelines}

## Student's Answer
{student_answer}

## Your Task

Carefully evaluate the student's answer against the official solution and grading guidelines.

Follow this evaluation process:
1. Identify the key steps and insights required in the official solution
2. Check which steps the student has correctly completed
3. Identify any errors, omissions, or incorrect reasoning in the student's answer
4. Apply the grading guidelines to determine the appropriate score
5. Consider partial credit for correct intermediate steps even if the final answer is wrong

Important grading principles:
- Award points for correct mathematical reasoning, even if presented differently from the official solution
- Deduct points for logical errors, missing cases, or incorrect conclusions
- Consider the completeness of the solution (did they prove all required claims?)
- Check for proper justification of key steps

Respond in JSON format with the following schema:
<json>
{{
    "reasoning": "Your detailed reasoning for the score assigned, explaining what the student did correctly and what was missing or incorrect",
    "response": <numerical_score>
}}
</json>

The response field must contain only a numerical value (integer or decimal) representing the score."""

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
                # Try to get response field, fall back to other common fields
                last_json = extracted[-1]
                if "response" in last_json:
                    prediction = last_json["response"]
                elif "score" in last_json:
                    prediction = last_json["score"]
                elif "grade" in last_json:
                    prediction = last_json["grade"]
                else:
                    # If no recognized field, use the first numeric value found
                    for key, value in last_json.items():
                        if isinstance(value, (int, float)):
                            prediction = value
                            break
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
