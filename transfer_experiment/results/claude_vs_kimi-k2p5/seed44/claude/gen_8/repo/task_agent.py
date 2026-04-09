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

### Grade definitions -- memorise these before grading

**"correct"** -- The student's solution is complete and logically sound.
- All key steps required by the grading guidelines are present and valid.
- A valid non-standard approach that reaches the correct conclusion is **"correct"** even
  if it differs entirely from the official solution.
- Minor arithmetic slips, notational issues, or trivially-fixable gaps that do not affect
  the logical flow are acceptable for "correct".
- Do NOT downgrade to "partial" for cosmetic or presentational issues.
- Do NOT downgrade to "partial" merely because the approach differs from the official one.

**"partial"** -- The student's answer contains at least one specific milestone from the
grading guidelines but is not fully correct.
- A single correct key idea, lemma, invariant, or structural observation that is
  *explicitly listed* in the "(Partial)" or "(Almost)" section of the guidelines earns
  "partial".
- "Almost correct" solutions (nearly complete, one non-trivial gap) are **"partial"**.
- The "(Almost)" label in the rubric means the student is very close but has a
  non-negligible flaw -- this is **"partial"**, never "correct".
- When in doubt between "partial" and "incorrect", choose **"partial"** if any listed
  milestone is plausibly met.

**"incorrect"** -- The student's answer achieves **none** of the specific milestones
listed in the grading guidelines.
- The answer is fundamentally wrong, trivially incomplete, or does not engage with the
  core difficulty of the problem.
- Merely restating the problem, setting up notation, or computing trivial cases without
  achieving a listed milestone earns "incorrect".
- A correct final answer with no supporting argument is "incorrect" unless the guidelines
  explicitly award credit for the answer alone.
- **Do NOT award "partial" unless a specific listed milestone is clearly demonstrated.**

---

### The five calibration rules -- apply all five before deciding

**Rule 1 -- Milestone gate for partial:** To earn "partial", the student must have
demonstrably achieved at least one milestone *specifically listed* in the "(Partial)" or
"(Almost)" section of the grading guidelines. General mathematical activity that does not
correspond to a listed milestone does NOT earn "partial".

**Rule 2 -- "(Almost)" is a hard cap at "partial":** If the student's solution matches
any "(Almost)" criterion in the guidelines (nearly complete, minor non-negligible
mistakes), the grade is **"partial"** -- never "correct". This cap is absolute.

**Rule 3 -- Non-standard correct solutions:** A complete, logically sound solution that
uses a different method from the official solution is **"correct"** if it genuinely
solves the problem. Do not penalise for using coordinates, generating functions, or other
valid alternative approaches.

**Rule 4 -- Generous correctness:** Only downgrade from "correct" to "partial" if there
is a genuine, non-trivial logical gap, missing case, or unjustified claim that materially
undermines the argument. Small presentation gaps or minor errors that do not affect the
proof's validity are acceptable for "correct".

**Rule 5 -- Strict partial/incorrect boundary:** Do not award "partial" for vague
progress, restating the problem, or writing relevant-looking mathematics that does not
achieve a listed milestone. The milestone must be specifically and explicitly achieved.

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

**Step 2 -- Milestone check:** For each milestone, state ACHIEVED or NOT ACHIEVED with a
one-sentence justification. Quote the specific part of the student's answer that supports
your decision. Be strict: the milestone must be clearly and explicitly demonstrated, not
merely attempted or hinted at.

**Step 3 -- Partial-credit gate:**
- If at least one milestone is ACHIEVED -> grade is at minimum "partial". Continue to Step 4.
- If no milestone is ACHIEVED -> grade is **"incorrect"**. Stop here.

**Step 4 -- Almost gate (apply before correctness check):**
- Does the student's solution match any (Almost) criterion (nearly complete, minor
  non-negligible flaw)? If YES -> grade is **"partial"**. Stop here. Do NOT upgrade to
  "correct" even if the solution looks nearly perfect.

**Step 5 -- Correctness check (only if Steps 3 and 4 both passed):**
- List every genuine logical gap, missing case, or unjustified claim.
- Ask: does the student's solution, taken as a whole, constitute a complete and valid
  proof/solution? Consider non-standard approaches on their own merits.
- If there are NO non-trivial gaps -> grade is **"correct"**.
- If there ARE non-trivial gaps -> grade is **"partial"**.

Respond with a JSON block in the following format:
<json>
{{
    "milestones_achieved": "<for each guideline milestone: milestone text -- ACHIEVED/NOT ACHIEVED + one-sentence reason>",
    "almost_match": "<yes/no -- does the student meet any (Almost) criterion? give reason>",
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
