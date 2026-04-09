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
        """Run the task agent on a single problem.

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

        instruction = f"""You are an expert grader for the International Mathematical Olympiad (IMO). Your task is to evaluate a student's answer to a mathematical problem with precision, consistency, and strict adherence to the grading guidelines.

## Domain
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

Please evaluate the student's answer following this structured approach:

### Step 1: Problem Deconstruction
- Identify the core mathematical concepts and techniques required
- Note any key theorems, formulas, or insights needed
- Understand what constitutes a complete and correct solution
- Identify the critical components that must be present for full credit

### Step 2: Solution Mapping
- Break down the official solution into key milestones/steps
- Identify partial credit opportunities from the grading guidelines
- Note common error patterns that affect scoring
- Map each milestone to specific point values if specified

### Step 3: Student Answer Analysis
- Verify if the student's approach is mathematically valid
- Check for: correct problem interpretation, valid proof structure, accurate calculations
- Identify gaps: missing steps, logical errors, computational mistakes
- Assess partial credit: which milestones were achieved?
- Look for: correct final answers with wrong reasoning (partial credit), correct methods with calculation errors, incomplete proofs

### Step 4: Grade Justification
- Map student's achievements to the grading rubric precisely
- Provide specific evidence for the assigned grade
- Consider: correctness, completeness, clarity, and rigor
- If the guidelines specify numeric points, assign the exact point value earned

Respond in JSON format with the following schema:
<json>
{{
    "reasoning": "Your detailed chain-of-thought analysis following the 4 steps above...",
    "response": "The final grade/score as specified in the grading guidelines"
}}
</json>

IMPORTANT: The "response" field must contain ONLY the grade/score (e.g., "7", "2", "0", "Correct", "Incorrect", etc.) exactly as specified in the grading guidelines. Do not add explanations, quotes, or extra text in this field."""

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
                # Try to get response field, fall back to other common fields
                last_json = extracted[-1]
                if "response" in last_json:
                    prediction = last_json["response"]
                elif "grade" in last_json:
                    prediction = last_json["grade"]
                elif "score" in last_json:
                    prediction = last_json["score"]
                elif "answer" in last_json:
                    prediction = last_json["answer"]
                else:
                    # If no recognized field, use the whole JSON as string
                    prediction = json.dumps(last_json)
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
