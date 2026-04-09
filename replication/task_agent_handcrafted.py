"""
Task agent for IMO grading.

Written by Claude Opus 4.6 by reading the task description. No evolutionary loop.
Tests whether DGM-H's 200-generation search finds anything a human couldn't.
"""

from __future__ import annotations

import json
import logging
import re

from agent.llm_client import get_response_from_llm, EVAL_MODEL

logger = logging.getLogger(__name__)


class TaskAgent:
    """Hand-crafted IMO grading agent."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = f"""You are an expert IMO grader. Grade the student's answer.

## Problem
{problem}

## Official Solution
{solution}

## Grading Guidelines
{guidelines}

## Student's Answer
{student_answer}

## Your Task
Compare the student's answer against the official solution and grading guidelines.
Respond with exactly one of these four grades:
- Correct
- Almost
- Partial
- Incorrect

Respond in JSON:
<json>
{{"response": "<grade>"}}
</json>"""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        prediction = "None"
        try:
            text = msg_history[-1]["text"]
            # Try <json> tags first
            start = text.find("<json>")
            end = text.find("</json>")
            if start != -1 and end != -1:
                inner = text[start + 6:end].strip()
                obj = json.loads(inner)
                if "response" in obj:
                    prediction = obj["response"]
            # Fallback: look for grade labels directly
            if prediction == "None":
                text_lower = text.lower()
                for label in ["correct", "almost", "partial", "incorrect"]:
                    if label in text_lower:
                        prediction = label.capitalize()
                        if label == "incorrect":
                            prediction = "Incorrect"
                        break
        except Exception as e:
            self.log_fn(f"Error: {e}")

        return str(prediction).strip().lower(), msg_history
