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

__version__ = "1.4.0"


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    Also falls back to scanning for bare JSON objects if no tags are found.
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
            # Use a pattern that handles one level of nested braces
            pattern = r'\{(?:[^{}]|\{[^{}]*\})*\}'
            matches = re.findall(pattern, text)
            for match in matches:
                try:
                    obj = json.loads(match)
                    if isinstance(obj, dict):
                        results.append(obj)
                except json.JSONDecodeError:
                    continue
            # If still no results, try the simpler pattern
            if not results:
                pattern = r'\{[^{}]*\}'
                matches = re.findall(pattern, text)
                for match in matches:
                    try:
                        obj = json.loads(match)
                        if isinstance(obj, dict):
                            results.append(obj)
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
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        domain = inputs.get("domain", "")

        instruction = f"""You are an expert IMO (International Mathematical Olympiad) grader. Your task is to evaluate a student's answer to a math problem.

**Problem:**
{problem}

**Official Solution:**
{solution}

**Grading Guidelines:**
{grading_guidelines}

**Student's Answer:**
{student_answer}

Carefully analyze the student's answer step by step:
1. Compare the student's reasoning against the official solution
2. Check if the student's approach is mathematically valid
3. Identify any errors, gaps, or missing steps
4. Determine if the final answer is correct
5. Assign a grade based on the grading guidelines

Grading categories:
- "correct": The student's answer is fully correct with valid reasoning and correct conclusion
- "incorrect": The student's answer has fundamental errors, wrong approach, or incorrect conclusion
- "partial": The student's answer has some correct elements but is incomplete or has minor errors
- "almost": The student's answer is nearly correct with only minor mistakes or omissions

Respond in JSON format with the following schema:
<json>
{{
    "reasoning": "Your step-by-step analysis of the student's answer",
    "response": "correct" or "incorrect" or "partial" or "almost"
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
