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
        """Run the task agent on a single problem with structured reasoning.

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

        instruction = f"""You are an expert mathematical grader for the International Mathematical Olympiad (IMO).

Your task is to evaluate a student's answer to a mathematical problem and provide a grade.

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

## Instructions

1. **Analyze the student's answer step by step**:
   - Check if the student understood the problem correctly
   - Verify each step of their reasoning
   - Compare their approach with the official solution
   - Identify any errors, gaps, or creative insights

2. **Apply the grading guidelines**:
   - Consider partial credit for correct intermediate steps
   - Note any missing justifications or logical gaps
   - Award points for novel correct approaches

3. **Provide your final grade** in the JSON format below.

Respond in JSON format with the following schema:
<json>
{{
    "thinking": "Your detailed chain-of-thought analysis here...",
    "evaluation": {{
        "correctness": "full|partial|none",
        "key_strengths": ["strength 1", "strength 2"],
        "key_weaknesses": ["weakness 1", "weakness 2"],
        "partial_credit_notes": "explanation of partial credit if applicable"
    }},
    "response": "Your final grade (e.g., '7', '3', '0', 'Full points', 'Partial credit', etc.)"
}}
</json>

The "response" field should contain ONLY the final grade value, matching the format expected by the evaluation system."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON with better error handling
        prediction = "None"
        try:
            last_message = msg_history[-1]["text"]
            extracted = _extract_jsons(last_message)
            if extracted:
                last_json = extracted[-1]
                if "response" in last_json:
                    prediction = last_json["response"]
                elif "grade" in last_json:
                    prediction = last_json["grade"]
                else:
                    # Try to find any string value that might be the grade
                    for key, value in last_json.items():
                        if isinstance(value, str) and value.strip():
                            prediction = value
                            break
                    for key, value in last_json.items():
                        if isinstance(value, (int, float)):
                            prediction = str(value)
                            break
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            # Fallback: try to extract any numeric-looking grade from the response
            try:
                last_message = msg_history[-1]["text"]
                # Look for patterns like "Grade: 7" or "Final grade: 3"
                grade_match = re.search(r'(?:grade|score|points?)[:\s]*(\d+(?:\.\d+)?)', last_message, re.IGNORECASE)
                if grade_match:
                    prediction = grade_match.group(1)
            except Exception:
                pass

        return str(prediction), msg_history
