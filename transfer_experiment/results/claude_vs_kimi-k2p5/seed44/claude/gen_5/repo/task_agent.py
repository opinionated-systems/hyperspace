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

# Valid prediction labels (the harness also uses "almost" as a label,
# which we treat as "partial" since it represents near-correct work).
_VALID_LABELS = {"correct", "partial", "incorrect", "almost"}


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


def _normalise_label(raw: str) -> str:
    """Map a raw model response string to one of the canonical labels.

    The evaluation harness accepts: "correct", "partial", "incorrect".
    "almost" (used in some rubrics) is treated as "partial".

    The key fix here is to use exact-match checks rather than substring
    checks, which previously caused "incorrect" to match "correct".
    """
    raw = raw.strip().lower()

    # Exact match first (most reliable)
    if raw in _VALID_LABELS:
        if raw == "almost":
            return "partial"
        return raw

    # Prefix / contained-word matching as fallback, ordered carefully so
    # that "incorrect" is checked before "correct" to avoid false positives.
    if raw.startswith("incorrect") or raw == "wrong":
        return "incorrect"
    if raw.startswith("almost"):
        return "partial"
    if raw.startswith("partial"):
        return "partial"
    if raw.startswith("correct"):
        return "correct"

    # Last-resort keyword scan — check longer/more-specific words first.
    if "incorrect" in raw or "wrong" in raw:
        return "incorrect"
    if "almost" in raw:
        return "partial"
    if "partial" in raw:
        return "partial"
    if "correct" in raw:
        return "correct"

    return "incorrect"


def _extract_prediction(text: str) -> str | None:
    """Try multiple strategies to pull a label out of model output."""
    # Strategy 1: <json>...</json> block
    extracted = _extract_jsons(text)
    if extracted:
        for obj in reversed(extracted):
            if "response" in obj:
                return _normalise_label(str(obj["response"]))

    # Strategy 2: bare JSON object anywhere in the text
    for match in re.finditer(r'\{[^{}]*"response"\s*:\s*"([^"]+)"[^{}]*\}', text):
        return _normalise_label(match.group(1))

    # Strategy 3: look for the label on its own line or after a colon
    for pattern in [
        r'"response"\s*:\s*"([^"]+)"',
        r'(?i)\bgrade\s*:\s*(correct|partial|incorrect|almost)\b',
        r'(?i)\b(incorrect|almost|partial|correct)\b',
    ]:
        m = re.search(pattern, text)
        if m:
            return _normalise_label(m.group(1))

    return None


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
        problem        = inputs.get("problem", "")
        solution       = inputs.get("solution", "")
        guidelines     = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        domain         = inputs.get("domain", "")

        instruction = f"""You are an expert mathematical grader evaluating a student's solution to a competition mathematics problem.

## Your task
Read the problem, the official solution, the grading guidelines, and the student's answer.
Decide whether the student's answer deserves a grade of **"correct"**, **"partial"**, or **"incorrect"**.

---

### Grading definitions

- **correct** – The student's answer is fully correct and complete. All key steps required by the grading guidelines are present and logically sound. Minor arithmetic slips or notational issues that do not affect the logical flow are acceptable. A solution that is essentially complete but has a small, easily-fixable gap may still be **"correct"** if the core argument is solid.

- **partial** – The student's answer contains **at least one** of the specific milestones listed under "(Partial)" or "(Almost)" in the grading guidelines. Even a single correct key idea, lemma, invariant, or structural observation that appears in the guidelines earns "partial". An answer that is nearly complete but has one significant gap or error is **"partial"**, not "incorrect". An answer described as "almost" correct in the guidelines is **"partial"**, not "correct".

- **incorrect** – The student's answer does not achieve **any** of the specific milestones listed in the grading guidelines. An answer is "incorrect" only when it fails to demonstrate even one of the listed partial-credit ideas.

---

### Critical calibration rules — apply these carefully

**Rule 1 — Generous partial credit:** If the student's work contains *any* of the ideas listed under "(Partial)" in the grading guidelines, the grade is at least **"partial"**. Do not grade as "incorrect" if even one partial milestone is met. When in doubt between "incorrect" and "partial", choose **"partial"**.

**Rule 2 — "Almost" always means "partial":** If the grading guidelines have an "(Almost)" section and the student's solution matches those criteria (nearly complete, minor mistakes), the grade is **"partial"**, never "correct". The word "almost" in the guidelines explicitly signals that the solution is not fully correct.

**Rule 3 — Correct vs partial:** Grade as "correct" if the student's solution covers all the key ideas and the argument is complete and sound. Small presentation gaps or minor errors that do not undermine the proof are acceptable for "correct". Only downgrade to "partial" if there is a genuine logical gap, missing case, or unjustified claim that materially affects the solution's validity.

**Rule 4 — Do not over-penalise:** A lengthy, sophisticated solution that gets the right answer via a valid (even if non-standard) method should be graded on its mathematical content, not on whether it matches the official solution's approach. If the student's method is correct, grade it as "correct" even if it differs from the official solution.

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

## Grading procedure — work through every step before giving your verdict

**Step 1 — List partial-credit milestones:** Copy out every milestone listed under "(Partial)" and "(Almost)" in the grading guidelines.

**Step 2 — Check each milestone:** For each milestone from Step 1, state explicitly **yes** or **no** (with a one-sentence justification) whether the student has achieved it.

**Step 3 — Partial-credit gate:** If the student achieved **at least one** milestone in Step 2, the grade is at minimum **"partial"**. If the student achieved **none**, the grade is **"incorrect"** — stop here.

**Step 4 — Correctness check (only if at least one milestone was met):** Determine whether the student's solution is fully complete and gap-free.
  - List any genuine logical gaps, missing cases, or unjustified claims.
  - If there are no non-trivial gaps → grade is **"correct"**.
  - If there are non-trivial gaps or errors → grade is **"partial"**.
  - Special case: if the student's solution matches the "(Almost)" criteria in the guidelines (nearly complete but with minor mistakes), the grade is **"partial"**, not "correct".

Respond with a JSON block in the following format:
<json>
{{
    "milestones_achieved": "<for each guideline milestone: milestone text — yes/no + brief reason>",
    "gaps_and_errors": "<genuine logical gaps or errors; write 'none' if fully correct>",
    "reasoning": "<one-sentence justification of your verdict>",
    "response": "<correct | partial | incorrect>"
}}
</json>"""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from the model's response, trying multiple sources.
        prediction = "incorrect"
        try:
            # Try the last assistant message first, then the raw response string.
            sources = []
            if msg_history:
                sources.append(msg_history[-1].get("text", ""))
            if response:
                sources.append(response)

            for src in sources:
                label = _extract_prediction(src)
                if label is not None:
                    prediction = label
                    break
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return prediction, msg_history
