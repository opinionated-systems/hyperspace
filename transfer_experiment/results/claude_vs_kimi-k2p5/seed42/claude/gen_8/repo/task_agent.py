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
SCORING CONTEXT  (read this first — it shapes every decision)
════════════════════════════════════════════════════════════════

The three grades map to point values:
  "correct"   → 7 points
  "partial"   → 1 point
  "incorrect" → 0 points

The grading guidelines may list "(Almost)" and "(Partial)" criteria.
Their intended meanings for your grading are:

  "(Almost)" criterion  — the student's answer is essentially complete
    with only minor, non-negligible gaps or mistakes.  An answer that
    meets an "(Almost)" criterion is very close to fully correct and
    should be graded "correct".  (The gap between "almost" and "correct"
    is far smaller than the gap between "almost" and "partial".)

  "(Partial)" criterion — the student made genuine, non-trivial progress
    on a specific sub-task but did not come close to a full solution.
    An answer that meets a "(Partial)" criterion (but no "(Almost)"
    criterion) should be graded "partial".

════════════════════════════════════════════════════════════════
GRADE DEFINITIONS
════════════════════════════════════════════════════════════════

"correct"
  The student's answer is fully correct and complete, OR it is nearly
  complete with only minor gaps (i.e., it satisfies an "(Almost)"
  criterion in the grading guidelines).
  • Every essential logical step is present and logically sound, OR
    only minor non-negligible mistakes remain.
  • The student reaches the correct final conclusion (or comes very
    close to it with a small, identifiable gap).
  • Minor presentational or notational differences are acceptable.
  ✔ RULE: If the student's answer satisfies ANY "(Almost)" criterion
    in the grading guidelines, the grade MUST be "correct".

"partial"
  The student makes genuine, non-trivial mathematical progress but the
  answer is substantially incomplete or contains significant errors.
  Award "partial" when the student's answer EXPLICITLY and VERIFIABLY
  satisfies AT LEAST ONE "(Partial)" criterion listed in the grading
  guidelines.
  ✔ GENEROUS STANDARD FOR PARTIAL: If the student has clearly achieved
    the mathematical content described by a "(Partial)" criterion —
    even if their write-up is informal, uses different notation, or
    arrives at it as a by-product of a larger (failed) argument — that
    counts.  The key question is: did the student actually establish
    the mathematical fact or take the mathematical step described?
  ✔ A long, detailed attempt that contains the key sub-result of a
    "(Partial)" criterion DOES earn "partial", even if the overall
    solution is wrong.
  ⚠ Do NOT award "partial" based on vague similarity to a criterion,
    general mathematical knowledge, or a long but ultimately flawed
    argument that never reaches the criterion's specific conclusion.
  ⚠ An answer that satisfies an "(Almost)" criterion is "correct",
    not "partial" (see above).

"incorrect"
  The student's answer is wrong, fundamentally flawed, or makes no
  meaningful progress toward the solution.  This includes:
  • Arriving at a clearly wrong conclusion.
  • Circular reasoning or merely restating the problem.
  • A plausible-sounding narrative with no real mathematical substance.
  • A valid approach with a critical error so early that no useful
    progress is made.
  • An answer that does NOT explicitly satisfy any "(Partial)" or
    "(Almost)" criterion in the grading guidelines.
  ⚠ A long, detailed, or confident-sounding answer is NOT evidence of
    partial credit.  Only explicit criterion matches count.

════════════════════════════════════════════════════════════════
DECISION PROCEDURE  (follow steps in order; stop at first match)
════════════════════════════════════════════════════════════════

STEP 1 — CORRECT CHECK
  a. Are ALL essential logical steps present and sound?
  b. Does the student reach the correct final conclusion?
  → BOTH YES → grade is "correct".  STOP.
  → OTHERWISE → continue to Step 2.

STEP 2 — ALMOST CHECK
  Read every "(Almost)" criterion in the grading guidelines verbatim.
  For each criterion, ask: does the student's answer EXPLICITLY satisfy
  it?  Quote the specific passage as evidence.
  → ANY criterion satisfied → grade is "correct".  STOP.
  → NONE satisfied          → continue to Step 3.

STEP 3 — PARTIAL CHECK
  Read every "(Partial)" criterion in the grading guidelines verbatim.
  For each criterion, ask: has the student actually established the
  specific mathematical fact or completed the specific mathematical
  step described by the criterion?  Quote the specific passage as
  evidence.  Apply a generous standard: credit the student if the
  mathematical substance is there, even if the presentation is
  informal or the criterion is reached as part of a larger attempt.
  → ANY criterion substantively satisfied → grade is "partial".  STOP.
  → NONE satisfied                        → grade is "incorrect".

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
Step 1. Read the problem and official solution carefully to understand
        what a complete proof requires.
Step 2. List every "(Almost)" criterion and every "(Partial)" criterion
        from the grading guidelines verbatim.
Step 3. Follow the DECISION PROCEDURE above (Steps 1 → 2 → 3).
        For each criterion you check, quote the specific passage from
        the student's answer that supports or refutes a match.
        For "(Partial)" criteria, be generous: if the student has
        established the mathematical substance of the criterion anywhere
        in their answer (even as a lemma or intermediate step in a
        larger failed argument), that counts.
Step 4. State your final grade and write it in the JSON block below.

Output your final answer as a JSON object enclosed in <json> tags.
The "response" field MUST be exactly one of: "correct", "partial",
or "incorrect" (lowercase, no quotes around the value in JSON).

Example A — answer meets an "(Almost)" criterion → "correct":
<json>
{{
    "almost_criteria": ["Applied infinite descent but did not complete the final step."],
    "partial_criteria": ["Proved c >= 3.", "Identified the mod-4 invariant."],
    "step1_correct": "Answer is not fully complete — missing the final step.",
    "step2_almost": "Student applied infinite descent (matches Almost criterion 1). Grade: correct.",
    "reasoning": "Student's answer satisfies Almost criterion 1; grade is correct.",
    "response": "correct"
}}
</json>

Example B — answer meets a "(Partial)" criterion only → "partial":
<json>
{{
    "almost_criteria": ["Applied infinite descent but did not complete the final step."],
    "partial_criteria": ["Proved c >= 3.", "Identified the mod-4 invariant."],
    "step1_correct": "Answer is incomplete — does not reach the final conclusion.",
    "step2_almost": "No Almost criterion is satisfied.",
    "step3_partial": "Student proved c >= 3 in their Step 2 (matches Partial criterion 1). Grade: partial.",
    "reasoning": "Student satisfies Partial criterion 1 but no Almost criterion; grade is partial.",
    "response": "partial"
}}
</json>

Example C — answer meets no criterion → "incorrect":
<json>
{{
    "almost_criteria": ["Applied infinite descent but did not complete the final step."],
    "partial_criteria": ["Proved c >= 3.", "Identified the mod-4 invariant."],
    "step1_correct": "Answer reaches a wrong conclusion.",
    "step2_almost": "No Almost criterion is satisfied.",
    "step3_partial": "Student never establishes c >= 3 or the mod-4 invariant; no Partial criterion met.",
    "reasoning": "No criterion is satisfied; grade is incorrect.",
    "response": "incorrect"
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
