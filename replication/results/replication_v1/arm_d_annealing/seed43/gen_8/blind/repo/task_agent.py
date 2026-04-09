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
        # Build a structured prompt for IMO grading with chain-of-thought
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = f"""You are an expert {domain} grader for the International Mathematical Olympiad (IMO).

Your task is to evaluate a student's answer based on the official solution and grading guidelines.

## Problem:
{problem}

## Official Solution:
{solution}

## Grading Guidelines:
{grading_guidelines}

## Student's Answer:
{student_answer}

## Instructions:
1. First, analyze the student's answer step by step. Compare it against the official solution.
2. Identify what the student got correct and what they got wrong or missed.
3. Consider the grading guidelines for partial credit.
4. Provide your reasoning for the grade you assign.
5. Finally, provide your grade assessment in the JSON format below.

Respond in JSON format with the following schema:
<json>
{{
    "reasoning": "Your detailed step-by-step analysis and reasoning for the grade",
    "response": "The final grade (e.g., '7', '6', '5', '0', 'Partial credit: 3/7', etc.)"
}}
</json>

The response field should contain only the grade assessment. Be precise and follow the grading guidelines closely."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON with improved error handling
        prediction = "None"
        try:
            extracted = _extract_jsons(msg_history[-1]["text"])
            if extracted:
                # Try to get response from the last valid JSON
                last_json = extracted[-1]
                if "response" in last_json:
                    prediction = last_json["response"]
                elif "grade" in last_json:
                    prediction = last_json["grade"]
                else:
                    # If no recognized field, use the whole JSON as string
                    prediction = str(last_json)
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            # Fallback: try to extract any numeric grade from the response
            try:
                import re
                # Look for patterns like "Grade: 7" or "Score: 5/7" or just "7"
                grade_patterns = [
                    r'grade[d]?\s*[:=]?\s*([0-9]+(?:\s*/\s*\d+)?)',
                    r'score[d]?\s*[:=]?\s*([0-9]+(?:\s*/\s*\d+)?)',
                    r'response\s*[:=]?\s*["\']?([0-9]+(?:\s*/\s*\d+)?)',
                ]
                text = msg_history[-1]["text"]
                for pattern in grade_patterns:
                    match = re.search(pattern, text, re.IGNORECASE)
                    if match:
                        prediction = match.group(1)
                        break
            except Exception:
                pass

        return str(prediction), msg_history
