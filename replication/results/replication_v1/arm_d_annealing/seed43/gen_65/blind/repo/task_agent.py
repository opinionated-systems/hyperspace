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

    def _build_grading_prompt(self, inputs: dict) -> str:
        """Build a structured prompt for IMO grading with chain-of-thought reasoning."""
        domain = inputs.get("domain", "")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        return f"""You are an expert mathematical grader for the International Mathematical Olympiad (IMO). Your task is to evaluate a student's solution based on the official solution and grading guidelines.

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

Please evaluate the student's answer following these steps:

1. **Understand the Problem**: Briefly summarize what the problem is asking and the key mathematical concepts involved.

2. **Analyze the Official Solution**: Identify the critical steps, key insights, and the logical flow of the official solution.

3. **Review Grading Guidelines**: Note the specific criteria, partial credit rules, and common errors mentioned.

4. **Evaluate Student's Answer**: 
   - Compare the student's approach to the official solution
   - Identify any correct steps, partial progress, or valid alternative approaches
   - Note any errors, gaps, or misconceptions
   - Check if the student provided a complete proof or just progress

5. **Determine Score**: Based on the IMO 7-point scale and the grading guidelines, assign an appropriate score (0-7).

Respond in JSON format with the following schema:
<json>
{{
    "reasoning": "Your detailed chain-of-thought analysis covering steps 1-4 above",
    "score": "The numerical score (0-7) as a number or string",
    "response": "The final score (0-7) - this is the prediction"
}}
</json>

Important: The "response" field must contain only the final score (0-7)."""

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with chain-of-thought reasoning.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        instruction = self._build_grading_prompt(inputs)

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON with fallback strategies
        prediction = "None"
        try:
            extracted = _extract_jsons(msg_history[-1]["text"])
            if extracted:
                # Try to get response field first
                if "response" in extracted[-1]:
                    prediction = extracted[-1]["response"]
                # Fallback to score field if response not present
                elif "score" in extracted[-1]:
                    prediction = extracted[-1]["score"]
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        # Validate prediction is a valid IMO score (0-7)
        try:
            score_val = float(str(prediction))
            if not (0 <= score_val <= 7):
                self.log_fn(f"Invalid score {score_val}, defaulting to 0")
                prediction = "0"
        except (ValueError, TypeError):
            # If not a valid number, try to extract a number from the text
            numbers = re.findall(r'\b([0-7])\b', str(prediction))
            if numbers:
                prediction = numbers[-1]  # Take the last number found
            else:
                self.log_fn(f"Could not extract valid score from '{prediction}', defaulting to 0")
                prediction = "0"

        return str(prediction), msg_history
