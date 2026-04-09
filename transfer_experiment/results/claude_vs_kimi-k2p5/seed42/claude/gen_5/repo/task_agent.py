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

# Valid grade labels (output space)
_VALID_GRADES = {"correct", "partial", "incorrect"}


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks.

    Uses str.find to locate outermost tag pairs, avoiding the lazy .*?
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


def _get_assistant_text(msg_history: list[dict], response: str) -> str:
    """Robustly extract the assistant's text from the message history.

    Different LLM client implementations store the assistant reply under
    different keys ("content", "text").  Try both before falling back to
    the raw ``response`` string.
    """
    if not msg_history:
        return response or ""
    last = msg_history[-1]
    # Standard OpenAI-style key
    if isinstance(last.get("content"), str):
        return last["content"]
    # Some clients use "text"
    if isinstance(last.get("text"), str):
        return last["text"]
    # content may be a list of content blocks (e.g. Anthropic)
    if isinstance(last.get("content"), list):
        parts = []
        for block in last["content"]:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
            elif isinstance(block, str):
                parts.append(block)
        if parts:
            return "\n".join(parts)
    return response or ""


class TaskAgent:
    """Task agent that grades IMO-style competition problems."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_file = log_file
        self.log_fn = logger.info

    # ---------------------------------------------------------------------- #
    # Prompt construction                                                      #
    # ---------------------------------------------------------------------- #

    def _build_instruction(
        self,
        problem: str,
        official_solution: str,
        grading_guidelines: str,
        student_answer: str,
    ) -> str:
        """Return the grading prompt."""
        return f"""You are an expert mathematics competition grader.
Your task is to evaluate a student's answer to a competition problem and assign
one of three grades: "correct", "partial", or "incorrect".

════════════════════════════════════════════════════════════════
GRADE DEFINITIONS
════════════════════════════════════════════════════════════════

"correct"
  The student's answer is fully correct and complete.
  • Every essential logical step is present and sound.
  • All key claims match the official solution.
  • Minor presentational differences are acceptable.
  ⚠ HARD RULE: If the grading guidelines label the answer "(Almost)",
    it MUST be graded "partial", never "correct".  "(Almost)" means
    there are minor but non-negligible mistakes.

"partial"
  The student makes genuine, non-trivial mathematical progress but
  the answer is incomplete or contains errors.  This includes:
  • A correct high-level approach with a wrong or missing final step.
  • Correctly identifying the key invariant, lemma, or construction
    without completing the full proof.
  • A solution that handles only some cases correctly.
  • An answer satisfying any "(Partial)" criterion in the guidelines.
  • An answer labelled "(Almost)" in the guidelines.
  ⚠ HARD RULE: "partial" requires GENUINE mathematical insight that
    is SPECIFIC to this problem.  A long but fundamentally flawed or
    circular answer is still "incorrect".
  ⚠ HARD RULE: The student must EXPLICITLY demonstrate the partial
    criterion — merely attempting the right approach without actually
    establishing the criterion does NOT qualify.

"incorrect"
  The student's answer is wrong, fundamentally flawed, or makes no
  meaningful progress.  This includes:
  • Arriving at a clearly wrong conclusion.
  • Circular reasoning or restating the problem.
  • A plausible-sounding narrative with no real mathematical substance.
  • A valid approach with a critical error so early that no useful
    progress is made.
  • An answer that does NOT satisfy any "(Partial)" or "(Almost)"
    criterion listed in the grading guidelines.

════════════════════════════════════════════════════════════════
DECISION CHECKLIST  (work through every item in order)
════════════════════════════════════════════════════════════════

[A] ALMOST CHECK — do this FIRST
    Does the grading guideline explicitly label this answer "(Almost)"?
    → YES: grade MUST be "partial".  Stop here.
    → NO:  continue.

[B] CORRECT CHECK
    1. Are ALL essential steps present and logically sound?
    2. Does the student reach the correct final conclusion?
    → Both YES: grade is "correct".  Stop here.
    → Otherwise: continue.

[C] PARTIAL CHECK — apply the Three-Question Test
    Q1. Does the student identify at least one non-trivial correct idea
        SPECIFIC to this problem (not just general mathematical knowledge)?
    Q2. Does that idea genuinely advance toward the solution?
    Q3. Is the idea actually correct (not merely plausible-sounding)?
    Q4. Does the student EXPLICITLY establish at least one of the
        "(Partial)" criteria listed in the grading guidelines?
    → ALL FOUR YES: grade is "partial".
    → ANY NO: grade is "incorrect".

════════════════════════════════════════════════════════════════
PROBLEM
════════════════════════════════════════════════════════════════
{problem}

════════════════════════════════════════════════════════════════
OFFICIAL SOLUTION
════════════════════════════════════════════════════════════════
{official_solution}

════════════════════════════════════════════════════════════════
GRADING GUIDELINES
════════════════════════════════════════════════════════════════
{grading_guidelines}

════════════════════════════════════════════════════════════════
STUDENT'S ANSWER
════════════════════════════════════════════════════════════════
{student_answer}

════════════════════════════════════════════════════════════════
GRADING INSTRUCTIONS
════════════════════════════════════════════════════════════════
Step 1. Read the problem and official solution to understand what a
        complete proof requires.
Step 2. Read the grading guidelines carefully.  Note every "(Partial)"
        and "(Almost)" criterion verbatim.
Step 3. Work through checklist items [A] → [B] → [C] in order.
        For each item, quote the specific evidence from the student's
        answer that supports your decision.
Step 4. State your final grade and write it in the JSON block below.

Output your final answer as a JSON object enclosed in <json> tags.
The "response" field MUST be exactly one of: "correct", "partial",
or "incorrect" (lowercase, no quotes around the value in JSON).

Example:
<json>
{{
    "checklist": {{
        "A_almost": "No '(Almost)' label found in guidelines.",
        "B_correct": "Student reaches correct conclusion but step 3 is missing.",
        "C_partial_q1": "Yes — student identifies the mod-4 invariant.",
        "C_partial_q2": "Yes — this directly constrains which pairs work.",
        "C_partial_q3": "Yes — the invariant argument is valid.",
        "C_partial_q4": "Yes — matches Partial criterion 1 verbatim."
    }},
    "reasoning": "Not almost. Not fully correct (missing step 3). Passes all four partial questions. Grade: partial.",
    "response": "partial"
}}
</json>

Now provide your grading decision below."""

    # ---------------------------------------------------------------------- #
    # Prediction extraction                                                    #
    # ---------------------------------------------------------------------- #

    def _extract_prediction(self, assistant_text: str) -> str:
        """Extract and validate the grade from the assistant's response.

        Strategy:
          1. Look for <json>...</json> blocks; prefer the last block with a
             valid "response" key.
          2. If that fails, use regex fallback on the full text.
          3. Default to "incorrect" if nothing is found.
        """
        prediction = "None"

        try:
            extracted = _extract_jsons(assistant_text)
            if extracted:
                for obj in reversed(extracted):
                    grade = str(obj.get("response", "")).strip().lower()
                    if grade in _VALID_GRADES:
                        prediction = grade
                        break
                else:
                    # No valid grade in any block; take the last block's value.
                    last_val = str(extracted[-1].get("response", "")).strip().lower()
                    if last_val:
                        prediction = last_val
        except Exception as exc:
            self.log_fn(f"Error extracting prediction from JSON blocks: {exc}")

        if prediction == "None":
            try:
                fallback = _extract_grade_fallback(assistant_text)
                if fallback and fallback in _VALID_GRADES:
                    prediction = fallback
                    self.log_fn("Used fallback grade extraction.")
            except Exception as exc:
                self.log_fn(f"Error in fallback grade extraction: {exc}")

        if prediction not in _VALID_GRADES:
            self.log_fn(f"Could not extract valid grade; defaulting to 'incorrect'. Raw: {assistant_text[:200]}")
            prediction = "incorrect"

        return prediction

    # ---------------------------------------------------------------------- #
    # Main entry point                                                         #
    # ---------------------------------------------------------------------- #

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

        instruction = self._build_instruction(
            problem=problem,
            official_solution=official_solution,
            grading_guidelines=grading_guidelines,
            student_answer=student_answer,
        )

        response, msg_history, _info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        assistant_text = _get_assistant_text(msg_history, response)
        prediction = self._extract_prediction(assistant_text)

        return prediction, msg_history
