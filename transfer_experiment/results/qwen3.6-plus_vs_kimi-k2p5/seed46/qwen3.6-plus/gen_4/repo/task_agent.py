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

__version__ = "1.1.0"


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


def _extract_json_from_markdown(text: str) -> list[dict] | None:
    """Extract JSON objects from markdown code blocks."""
    results = []
    # Match ```json ... ``` or just ``` ... ``` blocks
    pattern = r'```(?:json)?\s*(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    for match in matches:
        try:
            results.append(json.loads(match.strip()))
        except json.JSONDecodeError:
            continue
    return results or None


def _extract_any_json(text: str) -> list[dict] | None:
    """Extract JSON objects using multiple strategies."""
    # Try <json> tags first
    results = _extract_jsons(text)
    if results:
        return results

    # Try markdown code blocks
    results = _extract_json_from_markdown(text)
    if results:
        return results

    # Try to find JSON objects directly
    results = []
    # Look for patterns that look like JSON objects
    pattern = r'\{[^{}]*"[^"]+"[^{}]*\}'
    matches = re.findall(pattern, text)
    for match in matches:
        try:
            results.append(json.loads(match))
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
        # Build a more detailed prompt that encourages structured output
        problem = inputs.get('problem', '')
        solution = inputs.get('solution', '')
        grading_guidelines = inputs.get('grading_guidelines', '')
        student_answer = inputs.get('student_answer', '')

        instruction = f"""You are an expert mathematical grader evaluating student answers to competition-level math problems.

PROBLEM:
{problem}

OFFICIAL SOLUTION:
{solution}

GRADING GUIDELINES:
{grading_guidelines}

STUDENT'S ANSWER:
{student_answer}

Carefully analyze the student's answer step by step:
1. Compare the student's reasoning against the official solution
2. Check if the student's approach is mathematically valid
3. Identify any errors, gaps, or incomplete reasoning
4. Apply the grading guidelines strictly

Grade labels and their meanings:
- "correct": The student's answer is fully correct with valid reasoning matching the solution
- "incorrect": The student's answer is fundamentally wrong or has major errors
- "partial": The student has some correct elements but the solution is incomplete or has significant gaps
- "almost": The student's answer is nearly correct with only minor errors or omissions

Respond ONLY in JSON format with the following schema:
<json>
{{
    "response": "correct"
}}
</json>

The response field must contain exactly one of: "correct", "incorrect", "partial", "almost".
"""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON using multiple strategies
        prediction = "None"
        try:
            extracted = _extract_any_json(response)
            if extracted:
                # Try to find a response field
                for item in extracted:
                    if isinstance(item, dict):
                        if "response" in item:
                            prediction = item["response"]
                            break
                        # Also check for common alternative field names
                        for key in ["grade", "label", "result", "evaluation", "answer"]:
                            if key in item:
                                prediction = item[key]
                                break
                        if prediction != "None":
                            break

            # If no JSON found, try to extract a simple label from the text
            if prediction == "None":
                # Look for common grade labels in the response
                text_lower = response.lower()
                if "correct" in text_lower and "incorrect" not in text_lower:
                    prediction = "correct"
                elif "incorrect" in text_lower or "wrong" in text_lower:
                    prediction = "incorrect"
                elif "partial" in text_lower:
                    prediction = "partial"
                elif "almost" in text_lower:
                    prediction = "almost"
                else:
                    # Use the first line of the response as prediction
                    first_line = response.strip().split('\n')[0][:100]
                    if first_line:
                        prediction = first_line
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
