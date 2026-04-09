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
    """Task agent that grades student answers to competition math problems."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single grading problem.

        Args:
            inputs: dict with keys:
                - domain: subject area of the problem
                - problem: the competition math problem statement
                - solution: the official/reference solution
                - grading_guidelines: rubric describing what earns credit
                - student_answer: the student's submitted solution attempt

        Returns:
            (prediction, msg_history) where prediction is one of:
                "correct"   – the student's answer is fully correct
                "partial"   – the student's answer is partially correct
                "incorrect" – the student's answer is wrong or missing
        """
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        domain = inputs.get("domain", "")

        instruction = f"""You are an expert mathematical grader evaluating a student's solution to a competition mathematics problem.

## Your task
Read the problem, the official solution, the grading guidelines, and the student's answer.
Then decide whether the student's answer deserves a grade of **"correct"**, **"partial"**, or **"incorrect"**.

### Grading definitions
- **correct**   – The student's answer is fully correct and complete. All key steps are present and logically sound. Minor presentation issues are acceptable.
- **partial**   – The student's answer contains meaningful correct progress (e.g. a correct key lemma, a correct invariant, a correct special case) but is missing essential steps, contains a significant gap, or reaches a wrong final conclusion.
- **incorrect** – The student's answer is wrong, trivially incomplete, or does not engage meaningfully with the problem.

---

## Problem ({domain})
{problem}

---

## Official Solution
{solution}

---

## Grading Guidelines
{guidelines}

---

## Student's Answer
{student_answer}

---

## Instructions
1. Carefully compare the student's answer against the official solution and grading guidelines.
2. Identify which key ideas or steps the student has correctly established.
3. Identify any gaps, errors, or missing steps.
4. Assign one of the three grades: "correct", "partial", or "incorrect".

Respond **only** with a JSON block in the following format (no other text outside the block):
<json>
{{
    "reasoning": "<brief explanation of your grading decision>",
    "response": "<correct | partial | incorrect>"
}}
</json>"""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON with improved normalization
        prediction = "incorrect"
        try:
            # Try to extract from the last message in history first
            if msg_history and "text" in msg_history[-1]:
                extracted = _extract_jsons(msg_history[-1]["text"])
                if extracted and "response" in extracted[-1]:
                    raw = str(extracted[-1]["response"]).strip().lower()
                    # Normalize to one of the three valid labels
                    if "correct" in raw and "partial" not in raw and "in" not in raw:
                        prediction = "correct"
                    elif "partial" in raw:
                        prediction = "partial"
                    else:
                        prediction = "incorrect"
            
            # Fallback: try the direct response if history extraction failed
            if prediction == "incorrect":
                extracted = _extract_jsons(response)
                if extracted and "response" in extracted[-1]:
                    raw = str(extracted[-1]["response"]).strip().lower()
                    if "correct" in raw and "partial" not in raw and "in" not in raw:
                        prediction = "correct"
                    elif "partial" in raw:
                        prediction = "partial"
                    else:
                        prediction = "incorrect"
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return prediction, msg_history
