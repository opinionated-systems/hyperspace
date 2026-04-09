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
        # Extract fields for better structured prompting
        domain = inputs.get("domain", "")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = f"""You are an expert mathematical grader for the International Mathematical Olympiad (IMO).

Your task is to evaluate a student's answer to a mathematical problem and assign an appropriate score based on the official solution and grading guidelines.

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

Follow this structured evaluation process:

### Step 1: Problem Understanding
- Identify the key mathematical concepts and techniques required
- Note the critical steps in the official solution
- Understand the scoring rubric from the grading guidelines

### Step 2: Student Answer Analysis
- Check if the student stated the final answer correctly
- Identify which solution steps the student completed
- Note any missing or incorrect steps
- Evaluate the logical flow and mathematical rigor

### Step 3: Partial Credit Assessment
- Award points for each correct step completed
- Deduct points for logical gaps or errors
- Consider alternative valid approaches
- Be generous with partial credit when reasoning is sound

### Step 4: Final Score Determination
- Sum the points earned across all steps
- Verify against the grading guidelines
- Ensure consistency with the official scoring rubric

Respond in JSON format with the following schema:
<json>
{{
    "reasoning": "Your detailed analysis following the steps above. Include: (1) key concepts identified, (2) steps completed by student, (3) partial credit breakdown, (4) justification for final score",
    "response": "The final score (e.g., '0', '1', '2', '7', etc.)"
}}
</json>

Be thorough in your reasoning, generous with partial credit for correct reasoning, and precise in your final scoring."""

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
                # Try to get response field first, fallback to other common fields
                last_extract = extracted[-1]
                if "response" in last_extract:
                    prediction = last_extract["response"]
                elif "score" in last_extract:
                    prediction = last_extract["score"]
                elif "answer" in last_extract:
                    prediction = last_extract["answer"]
                else:
                    # If no recognized field, use the first value found
                    prediction = list(last_extract.values())[0] if last_extract else "None"
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
