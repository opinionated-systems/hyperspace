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
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

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
        # Extract fields for better structured prompting
        domain = inputs.get("domain", "")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = f"""You are an expert mathematical grader for the International Mathematical Olympiad (IMO).

Your task is to evaluate a student's answer to a mathematical problem. Follow these steps:

1. **Understand the Problem**: Read the problem statement carefully.
2. **Review the Official Solution**: Understand the expected approach and answer.
3. **Analyze the Grading Guidelines**: Note the specific criteria for awarding points.
4. **Evaluate the Student's Answer**: Compare the student's work against the solution and guidelines.
5. **Provide Reasoning**: Explain your evaluation step-by-step.
6. **Assign a Score**: Give the final score based on the grading guidelines.

## Problem Information
- **Domain**: {domain}

## Problem Statement
{problem}

## Official Solution
{solution}

## Grading Guidelines
{grading_guidelines}

## Student's Answer
{student_answer}

## Your Task
Evaluate the student's answer and respond in JSON format with the following schema:

<json>
{{
    "reasoning": "Your detailed step-by-step evaluation explaining how you arrived at the score",
    "score_breakdown": {{
        "points_awarded": "Number of points awarded",
        "justification": "Why these points were awarded or deducted"
    }},
    "response": "The final score (a number, typically 0-7 for IMO problems)"
}}
</json>

Important:
- Be thorough in your reasoning
- Justify each point awarded or deducted
- The "response" field must contain only the numeric score"""

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
                # Try to get response field first
                if "response" in extracted[-1]:
                    prediction = extracted[-1]["response"]
                # Fallback: try to extract numeric score from reasoning if response is missing
                elif "reasoning" in extracted[-1] and "score_breakdown" in extracted[-1]:
                    # Try to find a numeric score in the reasoning
                    reasoning = extracted[-1]["reasoning"]
                    score_info = extracted[-1]["score_breakdown"]
                    if isinstance(score_info, dict) and "points_awarded" in score_info:
                        prediction = score_info["points_awarded"]
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
