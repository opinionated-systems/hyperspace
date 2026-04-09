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
        # Extract fields from inputs for better prompt formatting
        domain = inputs.get("domain", "")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        # Build a more structured and detailed prompt for better grading
        instruction = f"""You are an expert mathematics grader with deep expertise in mathematical problem solving and pedagogy. Your task is to carefully evaluate a student's answer to a mathematics problem.

## Problem Statement:
{problem}

## Correct Solution:
{solution}

## Grading Guidelines:
{grading_guidelines}

## Student's Answer:
{student_answer}

## Your Evaluation Task:
Please evaluate the student's answer systematically:

1. **Understanding**: Does the student correctly understand what the problem is asking?
   - Did they identify the key concepts and requirements?
   - Did they set up the problem correctly?

2. **Approach and Reasoning**: Is the student's approach valid?
   - Is their reasoning logically sound?
   - Did they use appropriate mathematical techniques?
   - Are there any logical gaps or errors in their reasoning?

3. **Execution**: Did the student execute their approach correctly?
   - Are the calculations correct?
   - Is the work shown clearly and organized?

4. **Conclusion**: Did the student arrive at the correct final answer?
   - Is the answer properly justified?
   - Did they answer the specific question asked?

5. **Partial Credit Assessment**: Based on the grading guidelines, what score or assessment does this answer deserve?
   - Consider what the student did correctly
   - Consider where they made errors
   - Apply the grading guidelines precisely

## Response Format:
Provide your evaluation as a concise numeric score or descriptive assessment that directly addresses the grading guidelines. Be specific and justify your evaluation based on the criteria above.

Respond in JSON format with the following schema:
<json>
{{
    "response": "Your evaluation score or assessment here (e.g., '7/7', 'Partial credit: 3/7', 'Correct', 'Incorrect', etc.)"
}}
</json>"""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON with improved error handling
        prediction = "None"
        try:
            extracted = _extract_jsons(msg_history[-1]["text"])
            if extracted and len(extracted) > 0:
                # Get the last JSON object (most recent response)
                last_json = extracted[-1]
                if "response" in last_json:
                    prediction = last_json["response"]
                elif "score" in last_json:
                    prediction = last_json["score"]
                elif "evaluation" in last_json:
                    prediction = last_json["evaluation"]
                else:
                    # If no recognized key, use the first value found
                    prediction = str(list(last_json.values())[0]) if last_json else "None"
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            # Try to extract any meaningful text from the response as fallback
            try:
                response_text = msg_history[-1].get("text", "")
                # Look for common patterns in grading responses
                if "correct" in response_text.lower():
                    prediction = "Correct"
                elif "incorrect" in response_text.lower():
                    prediction = "Incorrect"
                elif "partial" in response_text.lower():
                    prediction = "Partial credit"
            except Exception:
                pass

        return str(prediction), msg_history
