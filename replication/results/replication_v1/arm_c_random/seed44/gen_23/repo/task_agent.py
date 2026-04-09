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
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with structured reasoning.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Extract key fields for better structured prompting
        domain = inputs.get("domain", "")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = f"""You are an expert grader for the International Mathematical Olympiad (IMO). Your task is to evaluate a student's answer based on the problem, official solution, and grading guidelines.

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

1. **Analyze the student's answer step by step**:
   - Compare the student's approach to the official solution
   - Identify any errors, omissions, or alternative valid approaches
   - Check if the student proved all necessary claims

2. **Determine the score** based on the grading guidelines:
   - IMO problems are typically scored 0-7 points
   - Partial credit is given for partial progress
   - Be precise about what points are earned or lost

3. **Provide your final evaluation** in the JSON format below.

Respond in JSON format with the following schema:
<json>
{{
    "reasoning": "Your detailed step-by-step analysis of the student's answer",
    "score": "The numerical score (0-7 or as specified in guidelines)",
    "response": "The final score as a number or string"
}}
</json>

Important: The "response" field should contain only the final score that will be used for evaluation."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON with improved error handling
        prediction = "None"
        try:
            extracted = _extract_jsons(msg_history[-1]["text"])
            if extracted:
                last_json = extracted[-1]
                # Prefer "response" field, fallback to "score" if available
                if "response" in last_json:
                    prediction = last_json["response"]
                elif "score" in last_json:
                    prediction = last_json["score"]
                else:
                    # If no recognized field, use the first value
                    prediction = list(last_json.values())[0] if last_json else "None"
                self.log_fn(f"Extracted prediction: {prediction}")
            else:
                self.log_fn("No JSON found in response, attempting fallback extraction")
                # Fallback: try to extract any number that looks like a score
                text = msg_history[-1]["text"]
                # Look for patterns like "score": 5 or "response": "7"
                score_match = re.search(r'["\']?(?:score|response)["\']?\s*[:=]\s*["\']?(\d+)["\']?', text, re.IGNORECASE)
                if score_match:
                    prediction = score_match.group(1)
                    self.log_fn(f"Fallback extraction found: {prediction}")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
