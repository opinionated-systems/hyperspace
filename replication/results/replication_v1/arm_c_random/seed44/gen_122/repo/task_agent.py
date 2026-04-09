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
        """Run the task agent on a single problem with chain-of-thought reasoning.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Extract fields for better prompting
        domain = inputs.get("domain", "")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = f"""You are an expert grader for {domain} problems. Your task is to evaluate a student's answer against the correct solution and grading guidelines.

## Problem Statement
{problem}

## Correct Solution
{solution}

## Grading Guidelines
{grading_guidelines}

## Student's Answer
{student_answer}

## Instructions
1. First, analyze the student's answer step by step. Compare it to the correct solution.
2. Identify what the student did correctly and what errors they made.
3. Consider the grading guidelines carefully.
4. Provide your reasoning for the grade you will assign.
5. Finally, provide your grade assessment in the JSON format below.

## Grade Labels
You must assign exactly one of these four grade labels:
- "Correct" - The solution is complete and correct (full credit)
- "Almost" - The solution is nearly complete with minor gaps or errors (high partial credit)
- "Partial" - The solution has significant gaps but some correct elements (low partial credit)
- "Incorrect" - The solution is wrong or has no valid content (no credit)

Respond in JSON format with the following schema:
<json>
{{
    "reasoning": "Your detailed step-by-step analysis and reasoning...",
    "response": "Correct" or "Almost" or "Partial" or "Incorrect"
}}
</json>

IMPORTANT: The "response" field must contain ONLY one of the four grade labels above, with no additional text."""

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
                # Prefer "response" field, but fallback to other common fields
                last_json = extracted[-1]
                if "response" in last_json:
                    prediction = last_json["response"]
                elif "grade" in last_json:
                    prediction = last_json["grade"]
                elif "answer" in last_json:
                    prediction = last_json["answer"]
                elif "result" in last_json:
                    prediction = last_json["result"]
                else:
                    # If no recognized field, use the whole JSON as string
                    prediction = json.dumps(last_json)
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        # Normalize prediction to one of the four valid grade labels
        prediction = _normalize_grade(str(prediction))

        return prediction, msg_history


def _normalize_grade(text: str) -> str:
    """Normalize a grade prediction to one of the four valid labels.
    
    Handles cases where the model outputs verbose text instead of just the label.
    """
    text_lower = text.lower().strip()
    
    # Direct match
    valid_labels = ["correct", "almost", "partial", "incorrect"]
    for label in valid_labels:
        if text_lower == label:
            return label.capitalize()
    
    # Check for label embedded in text (e.g., "Score: 10/10, Grade: Correct")
    # Priority order: check for "incorrect" first (most specific), then others
    if "incorrect" in text_lower or "wrong" in text_lower or "no credit" in text_lower:
        return "Incorrect"
    if "almost" in text_lower or "nearly complete" in text_lower or "high partial" in text_lower:
        return "Almost"
    if "partial" in text_lower or "some correct" in text_lower or "low partial" in text_lower:
        return "Partial"
    if "correct" in text_lower or "full credit" in text_lower or "complete" in text_lower:
        # Make sure it's not "incorrect" which we already checked
        return "Correct"
    
    # Check for numeric scores
    if "10/10" in text or "full" in text_lower:
        return "Correct"
    if "0/" in text or "0 out" in text_lower:
        return "Incorrect"
    
    # Default fallback
    return text
