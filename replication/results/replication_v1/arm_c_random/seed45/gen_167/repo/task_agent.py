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


def _extract_json_fallback(text: str) -> list[dict] | None:
    """Fallback JSON extraction for cases where tags are missing/malformed.

    Attempts to find JSON objects by looking for curly brace pairs.
    """
    results = []
    i = 0
    while i < len(text):
        # Find opening brace
        start = text.find("{", i)
        if start == -1:
            break

        # Track brace depth to find matching close
        depth = 1
        end = start + 1
        in_string = False
        escape_next = False

        while end < len(text) and depth > 0:
            char = text[end]
            if escape_next:
                escape_next = False
            elif char == "\\":
                escape_next = True
            elif char == '"' and not in_string:
                in_string = True
            elif char == '"' and in_string:
                in_string = False
            elif not in_string:
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
            end += 1

        if depth == 0:
            json_str = text[start:end]
            try:
                results.append(json.loads(json_str))
            except json.JSONDecodeError:
                pass
            i = end
        else:
            i = start + 1

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
        # Extract key fields for structured prompting
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = f"""You are an expert IMO (International Mathematical Olympiad) grader. Your task is to evaluate a student's solution to a mathematical problem.

## Domain
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
1. First, analyze the student's answer step by step
2. Compare it against the official solution and grading guidelines
3. Identify any errors, omissions, or creative valid approaches
4. Provide a detailed reasoning of your evaluation
5. Finally, give your grade/assessment in the response field

Respond in JSON format with the following schema:
<json>
{{
    "reasoning": "Your detailed step-by-step analysis and evaluation reasoning",
    "response": "Your final grade/assessment (e.g., '7/7', 'Partial credit: 3/7', 'Incorrect: 0/7')"
}}
</json>"""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON
        prediction = "None"
        try:
            response_text = msg_history[-1]["text"]
            extracted = _extract_jsons(response_text)
            # Fallback to brace-based extraction if no <json> tags found
            if not extracted:
                extracted = _extract_json_fallback(response_text)
            if extracted:
                last_json = extracted[-1]
                if "response" in last_json:
                    prediction = last_json["response"]
                elif "grade" in last_json:
                    prediction = last_json["grade"]
                elif "reasoning" in last_json and len(last_json) == 1:
                    # Handle case where only reasoning is provided
                    prediction = "See reasoning"
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
