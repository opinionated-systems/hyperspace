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
    Also falls back to scanning for bare JSON objects if no tags are found.
    Handles nested braces within JSON values by tracking brace depth.
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

    # Fallback: if no tagged JSON found, try to extract bare JSON objects
    if not results:
        try:
            # Try parsing the entire text as JSON first
            parsed = json.loads(text.strip())
            if isinstance(parsed, dict):
                results.append(parsed)
            elif isinstance(parsed, list):
                results.extend(item for item in parsed if isinstance(item, dict))
        except json.JSONDecodeError:
            # Try to find JSON-like objects using regex as last resort
            # Use a more robust pattern that handles nested braces
            pattern = r'\{(?:[^{}]|(?R))*\}'
            # Since Python re doesn't support (?R), use a manual approach
            results = _extract_json_objects_manual(text)

    return results or None


def _extract_json_objects_manual(text: str) -> list[dict]:
    """Manually extract JSON objects by tracking brace depth."""
    results = []
    i = 0
    while i < len(text):
        if text[i] == '{':
            # Find the matching closing brace
            depth = 0
            start = i
            in_string = False
            escape_next = False
            j = i
            while j < len(text):
                ch = text[j]
                if escape_next:
                    escape_next = False
                    j += 1
                    continue
                if ch == '\\' and in_string:
                    escape_next = True
                    j += 1
                    continue
                if ch == '"' and not escape_next:
                    in_string = not in_string
                elif not in_string:
                    if ch == '{':
                        depth += 1
                    elif ch == '}':
                        depth -= 1
                        if depth == 0:
                            candidate = text[start:j+1]
                            try:
                                obj = json.loads(candidate)
                                if isinstance(obj, dict):
                                    results.append(obj)
                            except json.JSONDecodeError:
                                pass
                            i = j
                            break
                j += 1
            else:
                # No matching brace found
                pass
        i += 1
    return results


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
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", inputs.get("response", ""))

        instruction = f"""You are an expert math grader for IMO-level problems. Your task is to carefully evaluate the student's answer and assign a grade.

Problem:
{problem}

Reference Solution:
{solution}

Grading Guidelines:
{grading_guidelines}

Student's Answer:
{student_answer}

Instructions for grading:
1. Carefully read and understand the problem statement.
2. Study the reference solution to identify key steps, insights, and the final answer.
3. Review the grading guidelines for specific criteria.
4. Analyze the student's answer step-by-step:
   - Does the student understand the problem correctly?
   - Are the key steps and reasoning sound?
   - Is the final answer correct?
   - Are there any gaps or errors in the reasoning?
5. Compare the student's work against the reference solution and grading guidelines.

Grading criteria:
- "correct": The student's answer is fully correct with sound reasoning and the right final answer.
- "incorrect": The student's answer is fundamentally wrong, has major errors, or arrives at an incorrect conclusion.
- "partial": The student's answer shows some correct reasoning or partial progress but is incomplete or has minor errors.

Respond in JSON format with the following schema:
<json>
{{
    "response": "correct" or "incorrect" or "partial",
    "reasoning": "detailed explanation of your grading decision, referencing specific parts of the student's answer"
}}
</json>"""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON
        prediction = "None"
        try:
            extracted = _extract_jsons(msg_history[-1]["text"])
            if extracted and "response" in extracted[-1]:
                prediction = extracted[-1]["response"]
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
