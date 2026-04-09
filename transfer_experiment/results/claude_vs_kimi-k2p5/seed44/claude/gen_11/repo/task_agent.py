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

    Uses strict priority ordering: incorrect > partial > correct to avoid
    false positives. The word "incorrect" contains "correct" as a substring,
    so we must check for "incorrect" first in all matching strategies.
    """
    if not raw or not isinstance(raw, str):
        return "incorrect"

    raw = raw.strip().lower().strip(".!?,:;\"'")

    if not raw:
        return "incorrect"

    # Exact match first (most reliable)
    if raw in _VALID_LABELS:
        if raw == "almost":
            return "partial"
        return raw

    # Prefix matching — STRICT ORDER: incorrect first, then partial, then correct
    if raw.startswith("incorrect") or raw.startswith("wrong"):
        return "incorrect"
    if raw.startswith("almost") or raw.startswith("partial") or raw.startswith("nearly"):
        return "partial"
    if raw.startswith("correct") or raw.startswith("right"):
        return "correct"

    # Last-resort keyword scan — check longer/more-specific words first.
    # "incorrect" must be checked before "correct" (substring containment).
    if "incorrect" in raw or "wrong" in raw:
        return "incorrect"
    if "almost" in raw or "partial" in raw or "nearly" in raw:
        return "partial"
    if "correct" in raw:
        return "correct"

    return "incorrect"


def _extract_prediction(text: str) -> str | None:
    """Try multiple strategies to pull a label out of model output."""
    if not text or not isinstance(text, str):
        return None

    text = text.strip()

    # Strategy 1: <json>...</json> block (most reliable)
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
                "correct"   -- the student's answer is fully correct
                "partial"   -- the student's answer is partially correct
                "incorrect" -- the student's answer is wrong or missing
        """
        problem        = inputs.get("problem", "")
        solution       = inputs.get("solution", "")
        guidelines     = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        domain         = inputs.get("domain", "")

        instruction = f"""You are an expert grader for competition mathematics. Your task is to evaluate a student's solution and assign one of three grades: **"correct"**, **"partial"**, or **"incorrect"**.

---

## GRADING RUBRIC STRUCTURE

The grading guidelines use two types of labelled criteria:

**"(Partial)" milestones** — Specific mathematical achievements that earn partial credit.
- If the student demonstrates ANY one of these milestones (even imperfectly), the minimum grade is **"partial"**.
- A milestone is "demonstrated" if the student has engaged with its core mathematical content, even if the execution has gaps or errors.

**"(Almost)" criteria** — Descriptions of near-complete work that has a specific flaw or gap.
- These describe a *type of student work*, not a ceiling on the grade.
- If the student's work **matches** an (Almost) criterion (the described flaw IS present), grade is **"partial"**.
- If the student's work **exceeds** an (Almost) criterion (the described flaw is ABSENT, work is complete), grade is **"correct"**.
- NEVER cap at "partial" merely because an "(Almost)" section exists in the rubric.

---

## GRADE DEFINITIONS

### "correct"
The student's solution is complete and logically sound.
- All key steps required by the grading guidelines are present and valid.
- A valid non-standard approach (coordinates, generating functions, etc.) that genuinely solves the problem is **"correct"** even if it differs from the official solution.
- Minor arithmetic slips, notational issues, or trivially-fixable gaps that do not affect the logical flow are acceptable.
- Do NOT downgrade to "partial" for cosmetic issues, different-but-valid approaches, or merely because an "(Almost)" section exists in the rubric.

### "partial"
The student's answer contains meaningful mathematical progress toward at least one specific milestone from the grading guidelines, but is not fully correct.
- Achieving ANY single (Partial) milestone earns at minimum "partial".
- Work matching an (Almost) criterion (nearly complete, specific flaw present) earns "partial".
- The milestone need not be perfectly proved — substantive engagement with its core mathematical content is sufficient.
- **When uncertain between "partial" and "incorrect", prefer "partial"** if any listed milestone is plausibly addressed with real mathematical content.

### "incorrect"
The student's answer achieves NONE of the specific milestones listed in the grading guidelines.
- The answer is fundamentally wrong, trivially incomplete, or does not engage with the core difficulty.
- Merely restating the problem, setting up notation, or making trivial observations without advancing toward a listed milestone earns "incorrect".
- Reserve "incorrect" for answers that show no substantive progress toward any listed milestone.

---

## CALIBRATION RULES (apply all before deciding)

**Rule 1 — Milestone specificity:** Only milestones *explicitly listed* in the "(Partial)" section count toward partial credit. General mathematical activity unrelated to any listed milestone does NOT earn "partial".

**Rule 2 — Evaluate actual work vs. rubric labels:** The "(Almost)" section describes what near-complete work looks like. Ask: does the student's work match that description (flaw present → "partial") or exceed it (flaw absent → "correct")? Never cap at "partial" just because an "(Almost)" section exists.

**Rule 3 — Non-standard correct solutions:** A complete, logically sound solution using a different method is **"correct"** if it genuinely solves the problem. Do not penalise for valid alternative approaches.

**Rule 4 — Generous correctness:** Only downgrade from "correct" to "partial" if there is a genuine, non-trivial logical gap, missing case, or unjustified claim that materially undermines the argument. Small presentation gaps or minor errors that do not affect the proof's validity are acceptable for "correct".

**Rule 5 — Partial vs. incorrect boundary:** Award "partial" whenever the student has made real mathematical progress toward a listed milestone, even if incomplete. Reserve "incorrect" only for answers with no substantive engagement with any listed milestone.

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

## GRADING PROCEDURE — complete every step in order

**Step 1 — List milestones:** Copy every milestone from the grading guidelines, labelling each as (Partial) or (Almost).

**Step 2 — Milestone check:** For each (Partial) milestone, state ACHIEVED / PARTIALLY ACHIEVED / NOT ACHIEVED with a one-sentence justification. Quote the specific part of the student's answer that supports your decision.
- ACHIEVED: student clearly demonstrates the milestone.
- PARTIALLY ACHIEVED: student engages substantively with the milestone's core mathematical content, even if execution is incomplete or has errors.
- NOT ACHIEVED: student shows no engagement with the milestone's core mathematical content.

**Step 3 — Partial-credit gate:**
- If at least one (Partial) milestone is ACHIEVED or PARTIALLY ACHIEVED → grade is at minimum "partial". Proceed to Step 4.
- If no (Partial) milestone is ACHIEVED or PARTIALLY ACHIEVED → tentatively "incorrect". Proceed to Step 4 to check (Almost) criteria.

**Step 4 — Almost assessment:** For each (Almost) criterion, ask: does the student's work match this description?
- YES (student's work has the described flaw/gap) → grade is **"partial"**. Stop.
- NO (student's work is complete, the described flaw is absent) → this criterion does not apply; continue to Step 5.
- If no (Almost) criteria exist or none match → continue to Step 5.

**Step 5 — Correctness check:**
- List every genuine logical gap, missing case, or unjustified claim in the student's work.
- Does the student's solution, taken as a whole, constitute a complete and valid proof/solution? Consider non-standard approaches on their own merits.
- NO non-trivial gaps → grade is **"correct"**.
- Non-trivial gaps exist AND at least one milestone was achieved or partially achieved → **"partial"**.
- Non-trivial gaps exist AND no milestone was achieved or partially achieved → **"incorrect"**.

Respond with a JSON block in the following format:
<json>
{{
    "milestones_achieved": "<for each guideline milestone: milestone text — ACHIEVED/PARTIALLY ACHIEVED/NOT ACHIEVED + one-sentence reason quoting the student's work>",
    "almost_assessment": "<for each (Almost) criterion: YES the student's work matches it (flaw present) or NO it does not (work exceeds it) + reason>",
    "gaps_and_errors": "<list genuine logical gaps or errors; write 'none' if fully correct>",
    "reasoning": "<one-sentence justification of your final verdict>",
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
