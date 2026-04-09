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
        # Extract fields with defaults for safety
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        # Build a structured prompt with chain-of-thought reasoning
        instruction = f"""You are an expert mathematics grader for the International Mathematical Olympiad (IMO).

Your task is to evaluate a student's answer to a mathematical problem and assign a score based on the official grading guidelines.

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

Please evaluate the student's answer following these steps:

1. **Understand the Problem**: Briefly summarize what the problem is asking and the key mathematical concepts involved.

2. **Analyze the Official Solution**: Identify the key steps, techniques, and results that constitute a complete correct solution.

3. **Review Grading Guidelines**: Note the specific criteria for partial credit and what constitutes a complete solution.

4. **Evaluate Student's Answer**: 
   - Check if the final answer is correct
   - Verify if the reasoning is sound and complete
   - Identify any gaps, errors, or missing steps
   - Compare against the grading guidelines for partial credit

5. **Assign Score**: Based on your analysis, assign the appropriate score according to the grading guidelines.

Respond in JSON format with the following schema:
<json>
{{
    "analysis": "Your detailed analysis of the student's answer, including what they did correctly and any errors or omissions",
    "reasoning": "Step-by-step reasoning for your grading decision",
    "response": "The final score (a number or string as specified in the grading guidelines)"
}}
</json>

Be thorough in your analysis but concise in your reasoning. The "response" field should contain only the final score."""

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
                # Try to get response field, fallback to other common fields
                last_json = extracted[-1]
                if "response" in last_json:
                    prediction = last_json["response"]
                elif "score" in last_json:
                    prediction = last_json["score"]
                elif "grade" in last_json:
                    prediction = last_json["grade"]
                elif "answer" in last_json:
                    prediction = last_json["answer"]
                else:
                    # If no recognized field, use the first value
                    prediction = list(last_json.values())[0] if last_json else "None"
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
