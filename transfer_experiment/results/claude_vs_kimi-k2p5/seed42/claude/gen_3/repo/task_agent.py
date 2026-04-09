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

# Valid grade labels
_VALID_GRADES = {"correct", "partial", "incorrect"}


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


def _extract_grade_fallback(text: str) -> str | None:
    """Fallback: scan the text for a bare grade keyword.

    Looks for lines like 'Grade: correct' or a standalone grade word
    near the end of the response, in case the model forgot the JSON tags.
    """
    # Try labelled patterns first: "grade: correct", "verdict: partial", etc.
    labelled = re.search(
        r'(?:grade|verdict|assessment|result)\s*[:\-]\s*(correct|partial|incorrect)',
        text,
        re.IGNORECASE,
    )
    if labelled:
        return labelled.group(1).lower()

    # Try a bare JSON-like fragment without tags: {"response": "correct"}
    bare_json = re.search(
        r'\{[^{}]*"response"\s*:\s*"(correct|partial|incorrect)"[^{}]*\}',
        text,
        re.IGNORECASE,
    )
    if bare_json:
        return bare_json.group(1).lower()

    # Last resort: find the last occurrence of a valid grade word as a
    # standalone token (surrounded by non-word characters or string boundaries).
    matches = list(re.finditer(r'\b(correct|partial|incorrect)\b', text, re.IGNORECASE))
    if matches:
        return matches[-1].group(1).lower()

    return None


class TaskAgent:
    """Task agent that solves IMO grading problems."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_file = log_file
        self.log_fn = logger.info

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines,
                    student_answer

        Returns:
            (prediction, msg_history)
        """
        problem            = inputs.get("problem", "")
        official_solution  = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer     = inputs.get("student_answer", "")

        instruction = f"""You are an expert mathematics competition grader.
Your task is to evaluate a student's answer to a competition problem and assign
one of three grades: "correct", "partial", or "incorrect".

Grade Definitions
-----------------
- "correct"   : The student's answer is fully correct and complete.
  All key ideas, claims, and conclusions match the official solution
  (minor presentation differences are acceptable). An answer labelled
  "Almost" in the grading guidelines (minor mistakes only) should also
  be graded "partial", NOT "correct".
- "partial"   : The student's answer contains meaningful progress toward
  the correct answer but is incomplete or contains non-trivial errors.
  This includes:
    * A correct high-level approach with a wrong or missing final step.
    * A correct answer without adequate justification.
    * A solution that handles only some cases correctly.
    * An answer labelled "Almost" in the grading guidelines (solution is
      nearly complete but has minor non-negligible mistakes).
    * Correctly identifying key lemmas or invariants without completing
      the full proof.
- "incorrect" : The student's answer is wrong, contains fundamental errors,
  or makes no meaningful progress toward the solution. A response that
  merely restates the problem, uses circular reasoning, or arrives at a
  clearly wrong conclusion should be graded "incorrect".

Critical Distinction: "partial" vs "incorrect"
-----------------------------------------------
Grade "partial" only when the student demonstrates genuine mathematical
insight or makes real progress. A long but fundamentally flawed answer
is still "incorrect". Ask: "Does this answer contain at least one
non-trivial correct idea that moves toward the solution?"

Problem
-------
{problem}

Official Solution
-----------------
{official_solution}

Grading Guidelines
------------------
{grading_guidelines}

Student's Answer
----------------
{student_answer}

Grading Instructions
--------------------
Step 1. Carefully read the problem, the official solution, and the grading
        guidelines.
Step 2. Evaluate the student's answer: identify what is correct, what is
        wrong or missing, and how it compares to the official solution.
Step 3. Check the grading guidelines for specific criteria. If the guidelines
        mention "(Almost)" criteria, those correspond to "partial" grade.
        If the guidelines mention "(Partial)" criteria, those also correspond
        to "partial" grade.
Step 4. Decide on a grade: "correct", "partial", or "incorrect".
Step 5. Write your final answer as a JSON object enclosed in <json> tags.
        The "response" field MUST be exactly one of the three strings:
        "correct", "partial", or "incorrect".

Example output format (fill in your own reasoning and grade):
<json>
{{
    "reasoning": "The student correctly identified the key invariant and provided a complete proof for both directions.",
    "response": "correct"
}}
</json>

Now provide your grading decision below."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # ------------------------------------------------------------------ #
        # Extract prediction from the assistant's response.                   #
        # Strategy:                                                            #
        #   1. Look for <json>...</json> blocks in the last assistant message. #
        #   2. If that fails, try a regex fallback on the full response text.  #
        # ------------------------------------------------------------------ #
        prediction = "None"
        try:
            assistant_text = msg_history[-1]["content"]
        except (IndexError, KeyError):
            try:
                assistant_text = msg_history[-1].get("text", response or "")
            except Exception:
                assistant_text = response or ""

        try:
            extracted = _extract_jsons(assistant_text)
            if extracted:
                # Prefer the last JSON block that contains a valid "response" key.
                for obj in reversed(extracted):
                    grade = str(obj.get("response", "")).strip().lower()
                    if grade in _VALID_GRADES:
                        prediction = grade
                        break
                else:
                    # No valid grade found in any block; take the last block's value.
                    last_response = str(extracted[-1].get("response", "")).strip().lower()
                    if last_response:
                        prediction = last_response
        except Exception as e:
            self.log_fn(f"Error extracting prediction from JSON blocks: {e}")

        # Fallback: scan the raw text for a grade keyword.
        if prediction == "None":
            try:
                fallback = _extract_grade_fallback(assistant_text)
                if fallback and fallback in _VALID_GRADES:
                    prediction = fallback
                    self.log_fn("Used fallback grade extraction.")
            except Exception as e:
                self.log_fn(f"Error in fallback grade extraction: {e}")

        return str(prediction), msg_history
