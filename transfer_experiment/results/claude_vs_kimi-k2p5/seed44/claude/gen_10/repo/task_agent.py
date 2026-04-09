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

        instruction = f"""You are a precise and calibrated expert grader evaluating a student's solution to a competition mathematics problem.

## Your task
Read the problem, the official solution, the grading guidelines, and the student's answer.
Decide whether the student's answer deserves a grade of **"correct"**, **"partial"**, or **"incorrect"**.

---

### Understanding the grading guidelines structure

The grading guidelines contain labelled sections:

- **(Partial)** milestones: specific achievements that earn partial credit. If the student
  demonstrates ANY one of these (even imperfectly but substantively), the grade is at
  minimum "partial".
- **(Almost)** criteria: descriptions of near-complete work that still has a non-negligible
  flaw or gap. These describe a *type of student work*, not a ceiling on the grade.
  - If the student's work **matches** an (Almost) criterion (nearly complete, minor flaw),
    grade is **"partial"**.
  - If the student's work **exceeds** an (Almost) criterion (the flaw is absent, the work
    is complete), grade is **"correct"** — do NOT cap at "partial" just because an
    (Almost) criterion exists in the rubric.

**Critical distinction:** The presence of an "(Almost)" section in the guidelines does NOT
mean the student's answer is "almost correct". It means the rubric describes what
near-complete work looks like. Evaluate the student's actual work independently.

---

### Grade definitions

**"correct"** -- The student's solution is complete and logically sound.
- All key steps required by the grading guidelines are present and valid.
- A valid non-standard approach that reaches the correct conclusion is **"correct"** even
  if it differs entirely from the official solution.
- Minor arithmetic slips, notational issues, or trivially-fixable gaps that do not affect
  the logical flow are acceptable for "correct".
- Do NOT downgrade to "partial" for cosmetic or presentational issues.
- Do NOT downgrade to "partial" merely because the approach differs from the official one.
- Do NOT downgrade to "partial" merely because an "(Almost)" section exists in the rubric;
  only downgrade if the student's work actually matches the (Almost) description.

**"partial"** -- The student's answer contains meaningful progress toward at least one
specific milestone from the grading guidelines but is not fully correct.
- A single correct key idea, lemma, invariant, or structural observation that is
  *explicitly listed* in the "(Partial)" section of the guidelines earns "partial".
- Work that matches an "(Almost)" criterion (nearly complete, one non-trivial gap) is
  **"partial"**.
- **Err on the side of "partial" over "incorrect"**: if the student has made genuine
  mathematical progress toward a listed milestone — even if the execution is incomplete
  or has errors — award "partial". The milestone need not be perfectly proved; it is
  enough that the student has substantively engaged with and advanced toward it.
- When in doubt between "partial" and "incorrect", choose **"partial"** if any listed
  milestone is plausibly addressed with real mathematical content.

**"incorrect"** -- The student's answer achieves **none** of the specific milestones
listed in the grading guidelines.
- The answer is fundamentally wrong, trivially incomplete, or does not engage with the
  core difficulty of the problem.
- Merely restating the problem, setting up notation, or computing trivial cases without
  advancing toward a listed milestone earns "incorrect".
- A correct final answer with no supporting argument is "incorrect" unless the guidelines
  explicitly award credit for the answer alone.
- Reserve "incorrect" for answers that show no substantive progress toward any milestone.

---

### The five calibration rules -- apply all five before deciding

**Rule 1 -- Generous partial gate:** To earn "partial", the student must have made
substantive progress toward at least one milestone *specifically listed* in the "(Partial)"
section of the grading guidelines. "Substantive progress" means the student has engaged
with the core mathematical content of the milestone, even if the proof is incomplete or
has gaps. General mathematical activity that is entirely unrelated to any listed milestone
does NOT earn "partial".

**Rule 2 -- Evaluate the student's actual work, not the rubric labels:** The "(Almost)"
section describes what near-complete work looks like. Ask: does the student's work match
that description (flaw present → "partial") or exceed it (flaw absent, work complete →
"correct")? Never cap at "partial" just because an "(Almost)" section exists.

**Rule 3 -- Non-standard correct solutions:** A complete, logically sound solution that
uses a different method from the official solution is **"correct"** if it genuinely
solves the problem. Do not penalise for using coordinates, generating functions, or other
valid alternative approaches.

**Rule 4 -- Generous correctness:** Only downgrade from "correct" to "partial" if there
is a genuine, non-trivial logical gap, missing case, or unjustified claim that materially
undermines the argument. Small presentation gaps or minor errors that do not affect the
proof's validity are acceptable for "correct".

**Rule 5 -- Partial vs incorrect boundary:** Award "partial" whenever the student has
made real mathematical progress toward a listed milestone, even if the work is incomplete.
Reserve "incorrect" only for answers that show no substantive engagement with any listed
milestone. When uncertain, prefer "partial" over "incorrect".

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

## Grading procedure -- complete every step in order

**Step 1 -- Extract milestones:** List every milestone from the grading guidelines,
labelling each as (Partial) or (Almost). If there are no explicit milestones, note that.

**Step 2 -- Milestone check:** For each milestone, state ACHIEVED, PARTIALLY ACHIEVED,
or NOT ACHIEVED with a one-sentence justification. Quote the specific part of the
student's answer that supports your decision. Be generous: if the student has engaged
substantively with the mathematical content of a milestone, even imperfectly, mark it
PARTIALLY ACHIEVED. Only mark NOT ACHIEVED if the student shows no engagement with that
milestone's core mathematical content.

**Step 3 -- Partial-credit gate:**
- If at least one (Partial) milestone is ACHIEVED or PARTIALLY ACHIEVED -> grade is at
  minimum "partial". Continue to Step 4.
- If no (Partial) milestone is ACHIEVED or PARTIALLY ACHIEVED -> tentatively "incorrect".
  Continue to Step 4 to check (Almost) criteria before finalising.

**Step 4 -- Almost assessment:**
- For each (Almost) criterion, ask: does the student's work match this description?
  - If YES (student's work has the described flaw/gap) -> grade is **"partial"**. Stop.
  - If NO (student's work is complete, the described flaw is absent) -> this (Almost)
    criterion does not apply; continue to Step 5.
- If no (Almost) criteria exist or none match -> continue to Step 5.

**Step 5 -- Correctness check:**
- List every genuine logical gap, missing case, or unjustified claim in the student's work.
- Ask: does the student's solution, taken as a whole, constitute a complete and valid
  proof/solution? Consider non-standard approaches on their own merits.
- If there are NO non-trivial gaps -> grade is **"correct"**.
- If there ARE non-trivial gaps AND at least one milestone was achieved or partially
  achieved -> **"partial"**.
- If there ARE non-trivial gaps AND no milestone was achieved or partially achieved ->
  **"incorrect"**.

Respond with a JSON block in the following format:
<json>
{{
    "milestones_achieved": "<for each guideline milestone: milestone text -- ACHIEVED/PARTIALLY ACHIEVED/NOT ACHIEVED + one-sentence reason>",
    "almost_assessment": "<for each (Almost) criterion: does the student's work match it? yes/no + reason>",
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
