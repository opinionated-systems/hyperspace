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
        # Extract fields from inputs for better prompt construction
        domain = inputs.get("domain", "")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        instruction = f"""You are an expert mathematical grader evaluating student solutions to competition mathematics problems (IMO, USAMO, etc.).

Your task is to analyze the student's answer and determine the correct grade based on the provided solution and grading guidelines.

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

1. **Read the problem carefully** - Understand what is being asked and what constitutes a complete solution.

2. **Study the official solution** - This shows the intended approach and key insights required.

3. **Review the grading guidelines** - These specify how points are allocated and what constitutes each grade level:
   - "Correct" (7/7): Complete, rigorous solution with no significant errors
   - "Almost" (6/7 or partial): Solution is nearly complete but has minor, non-negligible mistakes
   - "Partial" (1-5/7): Contains some correct observations or partial progress but incomplete
   - "Incorrect" (0/7): No meaningful progress toward the solution

4. **Analyze the student's answer**:
   - Identify what the student got right
   - Identify errors, gaps, or unjustified claims
   - Compare against the official solution and grading rubric
   - Determine if key theorems or steps are properly justified

5. **Provide your reasoning** - Explain your analysis step by step before giving the final grade.

6. **Output the final grade** in the JSON format below.

Respond in JSON format with the following schema:
<json>
{{
    "reasoning": "Your detailed analysis of the student's answer, comparing to the solution and explaining the grade",
    "response": "One of: Correct, Almost, Partial, or Incorrect"
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
                if "response" in extracted[-1]:
                    prediction = extracted[-1]["response"]
                elif "grade" in extracted[-1]:
                    prediction = extracted[-1]["grade"]
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
