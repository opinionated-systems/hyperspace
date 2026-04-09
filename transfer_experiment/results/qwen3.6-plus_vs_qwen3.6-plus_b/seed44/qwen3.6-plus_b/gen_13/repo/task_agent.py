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

__version__ = "1.7.0"


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

        instruction = f"""You are an expert IMO (International Mathematical Olympiad) grader specializing in {domain if domain else "mathematics"}. Your task is to evaluate a student's answer to a math problem.

**Problem:**
{problem}

**Official Solution:**
{solution}

**Grading Guidelines:**
{grading_guidelines}

**Student's Answer:**
{student_answer}

Perform a structured analysis:

**Step 1 — Identify key ideas in the official solution:**
List the essential mathematical steps, lemmas, or insights required for a complete solution.

**Step 2 — Map the student's work:**
For each key idea from Step 1, determine whether the student:
- Correctly established it
- Attempted it but with errors
- Did not address it at all

**Step 3 — Evaluate mathematical validity:**
Check each step the student did attempt for logical correctness. Note any gaps, unjustified claims, or computational errors.

**Step 4 — Assign a grade using this rubric:**
- "correct": All key steps present, reasoning valid, conclusion matches. No significant errors.
- "almost": Right approach, most steps correct, but 1-2 minor errors or omissions. The student clearly understands the solution path.
- "partial": Genuine mathematical progress made — e.g., correct lemmas, useful invariants, correct setup, valid reasoning for part of the problem. The student shows understanding but the solution is incomplete or has significant errors in later steps.
- "incorrect": Fundamentally flawed approach, wrong conclusion with no valid intermediate work, or no substantive mathematical progress.

**Step 5 — Cross-check with grading guidelines:**
The grading guidelines above may specify exact criteria for partial credit. If the student's work matches any described partial-progress scenario, use that guidance.

Respond in JSON format with the following schema:
<json>
{{
    "reasoning": "Your structured analysis: list key ideas from the official solution, map what the student achieved for each, note errors, then justify your grade",
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
