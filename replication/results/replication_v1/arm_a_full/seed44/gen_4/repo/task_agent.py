"""
Task agent: solves a given task with chain-of-thought reasoning.

Reimplemented from facebookresearch/HyperAgents task_agent.py.
Same interface, same JSON output format, same extraction logic.

This version includes chain-of-thought reasoning and structured analysis
for improved IMO grading accuracy.
"""

from __future__ import annotations

import json
import logging

from agent.llm_client import get_response_from_llm, EVAL_MODEL

logger = logging.getLogger(__name__)


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks."""
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
        """Run the task agent on a single problem with structured reasoning."""
        domain = inputs.get("domain", "")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = f"""You are an expert mathematical grader for IMO-level problems.

Your task is to evaluate a student's answer by following a structured reasoning process.

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

Follow this step-by-step process:

1. **Understand the Problem**: Identify what the problem is asking and the key mathematical concepts involved.

2. **Analyze the Official Solution**: Break down the correct solution into key steps and insights.

3. **Review Grading Guidelines**: Understand the criteria for partial and full credit.

4. **Evaluate Student's Answer**: 
   - Check if the answer is correct (matches the official solution)
   - Identify any errors, omissions, or misconceptions
   - Note any correct alternative approaches
   - Assess partial credit based on grading guidelines

5. **Determine Final Grade**: Assign a score based on your analysis.

Respond in JSON format with the following schema:
<json>
{{
    "reasoning": "Your detailed chain-of-thought analysis here...",
    "is_correct": true/false,
    "score": "Full marks/Partial marks/No marks/Other",
    "response": "Your final grading decision here"
}}
</json>

The "response" field should contain your final answer that will be used for evaluation."""

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
                # Prefer the structured response field, fallback to score
                if "response" in result:
                    prediction = result["response"]
                elif "score" in result:
                    prediction = result["score"]
                elif "is_correct" in result:
                    prediction = "Correct" if result["is_correct"] else "Incorrect"
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
