"""
Task agent: solves a given task with structured reasoning and verification.

Reimplemented from facebookresearch/HyperAgents task_agent.py.
Enhanced with chain-of-thought reasoning and self-verification.

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


def _format_grading_prompt(inputs: dict) -> str:
    """Format the grading task with structured reasoning steps."""
    domain = inputs.get("domain", "")
    problem = inputs.get("problem", "")
    solution = inputs.get("solution", "")
    grading_guidelines = inputs.get("grading_guidelines", "")
    student_answer = inputs.get("student_answer", "")
    
    return f"""You are an expert grader for {domain} problems.

Your task is to evaluate a student's answer by following a structured reasoning process.

## Problem Statement
{problem}

## Correct Solution
{solution}

## Grading Guidelines
{grading_guidelines}

## Student's Answer
{student_answer}

## Instructions

Follow these steps in your reasoning:

1. **Understand the Problem**: Briefly summarize what the problem is asking.

2. **Analyze the Correct Solution**: Identify the key steps and final answer.

3. **Evaluate Student's Answer**: 
   - Check if the student's approach is correct
   - Identify any errors or misconceptions
   - Compare with the grading guidelines

4. **Determine Score**: Based on the guidelines, assign an appropriate score.

5. **Provide Feedback**: Explain your reasoning for the score.

Respond in JSON format with the following schema:
<json>
{{
    "reasoning": "Your step-by-step analysis",
    "score": "The assigned score (numeric or as specified in guidelines)",
    "feedback": "Explanation of the score",
    "response": "Final answer - the score"
}}
</json>"""


class TaskAgent:
    """Task agent that solves IMO grading problems with structured reasoning."""

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
        instruction = _format_grading_prompt(inputs)

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
                result = extracted[-1]
                # Try to get response field first, then score, then fallback
                if "response" in result:
                    prediction = result["response"]
                elif "score" in result:
                    prediction = result["score"]
                else:
                    prediction = str(result)
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
