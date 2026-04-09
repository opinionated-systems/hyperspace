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
        """Run the task agent on a single problem with step-by-step reasoning.

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

Your task is to evaluate a student's answer to a mathematical problem. Follow these steps:

1. **Understand the Problem**: Read the problem statement carefully and identify what is being asked.

2. **Review the Official Solution**: Study the provided solution to understand the correct approach and key insights.

3. **Analyze the Grading Guidelines**: Pay close attention to the grading rubric and point allocation.

4. **Evaluate the Student's Answer**: Compare the student's answer against the official solution and grading guidelines.
   - Identify correct steps and valid reasoning
   - Identify errors, gaps, or incorrect claims
   - Check if partial credit should be awarded based on the guidelines

5. **Provide Your Assessment**: Give a clear, reasoned evaluation.

**Problem Domain**: {domain}

**Problem Statement**:
{problem}

**Official Solution**:
{solution}

**Grading Guidelines**:
{grading_guidelines}

**Student's Answer**:
{student_answer}

Respond in JSON format with the following schema:
<json>
{{
    "reasoning": "Your step-by-step analysis and reasoning process",
    "assessment": "Summary of the evaluation",
    "response": "Your final grading decision (e.g., '7', '6', 'Partial credit: 3/7', etc.)"
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
            extracted = _extract_jsons(msg_history[-1]["text"])
            if extracted:
                # Try to get response field, fall back to assessment if not present
                last_extract = extracted[-1]
                if "response" in last_extract:
                    prediction = last_extract["response"]
                elif "assessment" in last_extract:
                    prediction = last_extract["assessment"]
                else:
                    # Use first available string value
                    for key, value in last_extract.items():
                        if isinstance(value, str):
                            prediction = value
                            break
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
