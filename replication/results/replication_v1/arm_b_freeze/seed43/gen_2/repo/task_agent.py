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

        # Build a structured prompt with enhanced chain-of-thought reasoning
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

Please evaluate the student's answer following these systematic steps:

### Step 1: Problem Deconstruction
- Identify the key mathematical concepts, theorems, and techniques required
- Determine what constitutes a complete and correct solution
- Note any common pitfalls or alternative valid approaches

### Step 2: Solution Mapping
- Break down the official solution into key milestones/steps
- Identify which steps are essential for full credit vs. partial credit
- Note the scoring rubric breakpoints from the grading guidelines

### Step 3: Student Work Analysis
- Map the student's answer against the official solution milestones
- Identify what the student did correctly (correct claims, valid reasoning, proper technique)
- Identify errors, gaps, or invalid reasoning (logical flaws, incorrect calculations, missing justifications)
- Check if the final answer matches the expected result

### Step 4: Partial Credit Assessment
- Award credit for each correctly completed milestone
- Deduct credit for significant errors or omissions
- Consider if alternative valid approaches were used correctly
- Apply the grading guidelines rubric precisely

### Step 5: Final Score Determination
- Sum the partial credits to determine the final score
- Ensure the score aligns with the grading guidelines
- Double-check that the score reflects both correct work and identified errors

Respond in JSON format with the following schema:
<json>
{{
    "analysis": "Detailed evaluation of the student's answer: what was correct, what was incorrect, and what was missing. Be specific about mathematical claims and reasoning.",
    "reasoning": "Step-by-step justification for the score: which milestones were met, which were not, and how partial credit was calculated per the grading guidelines.",
    "response": "The final score as a number or string (exactly as specified in the grading guidelines)"
}}
</json>

Important: The "response" field must contain ONLY the final score value, with no additional text or explanation."""

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
