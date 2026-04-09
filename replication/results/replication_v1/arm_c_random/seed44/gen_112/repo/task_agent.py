"""
Task agent: solves a given task with chain-of-thought reasoning.

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
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with chain-of-thought reasoning.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Extract fields for better prompting
        domain = inputs.get("domain", "")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = f"""You are an expert grader for {domain} problems. Your task is to evaluate a student's answer against the correct solution and grading guidelines.

## Problem Statement
{problem}

## Correct Solution
{solution}

## Grading Guidelines
{grading_guidelines}

## Student's Answer
{student_answer}

## Instructions
1. First, analyze the student's answer step by step. Compare it to the correct solution.
2. Identify what the student did correctly and what errors they made.
3. Consider the grading guidelines carefully.
4. Provide your reasoning for the grade you will assign.
5. Finally, provide your grade assessment in the JSON format below.

IMPORTANT: The "response" field must contain ONLY one of these exact grade labels:
- "correct" - The student's answer is fully correct and complete
- "almost" - The student's answer is nearly correct with only minor gaps
- "partial" - The student's answer has significant gaps but some correct elements
- "incorrect" - The student's answer is wrong or does not address the problem

Respond in JSON format with the following schema:
<json>
{{
    "reasoning": "Your detailed step-by-step analysis and reasoning...",
    "response": "correct" | "almost" | "partial" | "incorrect"
}}
</json>"""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON
        prediction = "None"
        valid_grades = {"correct", "almost", "partial", "incorrect"}
        try:
            extracted = _extract_jsons(msg_history[-1]["text"])
            if extracted:
                # Prefer "response" field, but fallback to other common fields
                last_json = extracted[-1]
                raw_response = None
                if "response" in last_json:
                    raw_response = last_json["response"]
                elif "grade" in last_json:
                    raw_response = last_json["grade"]
                elif "answer" in last_json:
                    raw_response = last_json["answer"]
                elif "result" in last_json:
                    raw_response = last_json["result"]
                
                if raw_response:
                    # Normalize and validate the response
                    normalized = raw_response.strip().lower()
                    # Check if it contains one of the valid grade labels
                    for grade in valid_grades:
                        if grade in normalized:
                            prediction = grade
                            break
                    else:
                        # If no valid grade found, use the raw response
                        prediction = raw_response
                else:
                    # If no recognized field, use the whole JSON as string
                    prediction = json.dumps(last_json)
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
