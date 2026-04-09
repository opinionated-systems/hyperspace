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
        # Extract fields from inputs
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        instruction = f"""You are an expert mathematics grader for the International Mathematical Olympiad (IMO). Your task is to evaluate a student's solution to a mathematical problem.

## Problem Statement:
{problem}

## Official Solution:
{solution}

## Grading Guidelines:
{grading_guidelines}

## Student's Answer:
{student_answer}

## Your Task:
Evaluate the student's answer and classify it into EXACTLY ONE of the following four categories:
- "correct": The solution is complete and correct.
- "partial": The solution has significant progress but is incomplete or has minor gaps.
- "almost": The solution is nearly complete but has minor mistakes that are not negligible.
- "incorrect": The solution is fundamentally wrong or makes no significant progress.

## CRITICAL INSTRUCTION:
You MUST respond ONLY with a JSON object in the following format. Do NOT include any other text, reasoning, or explanation before or after the JSON.

<json>
{{
    "response": "correct" | "partial" | "almost" | "incorrect"
}}
</json>

The value of "response" must be exactly one of: "correct", "partial", "almost", or "incorrect".
Do not include any other text in your response."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON
        prediction = "incorrect"  # Default fallback
        try:
            # Try to extract from assistant's response
            assistant_text = msg_history[-1]["text"] if msg_history else ""
            
            # First try to extract from <json> tags
            extracted = _extract_jsons(assistant_text)
            if extracted and len(extracted) > 0:
                if "response" in extracted[-1]:
                    prediction = extracted[-1]["response"]
                elif len(extracted[-1]) == 1:
                    # If only one key exists, use its value
                    prediction = list(extracted[-1].values())[0]
            
            # Normalize the prediction
            prediction = str(prediction).lower().strip()
            valid_labels = ["correct", "partial", "almost", "incorrect"]
            
            if prediction not in valid_labels:
                # Try to find a valid label in the response
                found_label = None
                for label in valid_labels:
                    if label in prediction:
                        found_label = label
                        break
                
                if found_label:
                    prediction = found_label
                else:
                    # Last resort: check if any label appears in the full text
                    for label in valid_labels:
                        if f'"{label}"' in assistant_text or f"'{label}'" in assistant_text:
                            prediction = label
                            break
                    else:
                        prediction = "incorrect"  # Default fallback
                        
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            prediction = "incorrect"  # Default fallback on error

        return str(prediction), msg_history
