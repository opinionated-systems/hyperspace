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

# Valid prediction labels.  The harness uses all four as distinct categories:
#   "correct"   – fully correct solution
#   "almost"    – near-complete; only minor non-negligible mistakes remain
#   "partial"   – meaningful progress but substantially incomplete
#   "incorrect" – no substantive progress toward any milestone
_VALID_LABELS = {"correct", "almost", "partial", "incorrect"}

# Ordinal rank used for NMAE: incorrect=0, partial=1, almost=2, correct=3
_LABEL_RANK = {"incorrect": 0, "partial": 1, "almost": 2, "correct": 3}


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


def _normalise_label(raw: str) -> str:
    """Map a raw model response string to one of the four canonical labels.

    The evaluation harness accepts exactly: "correct", "almost", "partial",
    "incorrect".  Priority order for substring matching (most specific first):
      incorrect > almost > partial > correct
    This avoids the "incorrect" contains "correct" substring trap.
    """
    if not raw or not isinstance(raw, str):
        return "incorrect"

    raw = raw.strip().lower().strip(".!?,:;\"'` ")

    if not raw:
        return "incorrect"

    # Exact match (most reliable)
    if raw in _VALID_LABELS:
        return raw

    # Common synonyms / multi-word phrases — check negative first
    _INCORRECT_SYNONYMS = {
        "wrong", "false", "error", "invalid", "no", "none",
        "not correct", "not right", "not valid", "bad", "fail",
        "failed", "fails", "unsatisfactory", "unacceptable",
    }
    _ALMOST_SYNONYMS = {
        "nearly correct", "almost correct", "nearly complete",
        "almost complete", "minor error only", "minor mistake only",
        "minor errors only", "minor mistakes only",
    }
    _PARTIAL_SYNONYMS = {
        "mostly correct", "partially correct", "partly correct",
        "partially right", "mostly right", "some progress",
        "partial credit", "half correct", "incomplete",
        "significant progress", "substantial progress", "good progress",
        "on the right track", "approaching correct", "mostly there",
        "largely correct", "close but incomplete",
    }
    _CORRECT_SYNONYMS = {
        "right", "true", "valid", "yes", "full", "complete",
        "fully correct", "perfect", "excellent", "satisfactory",
        "acceptable", "pass", "passed", "complete solution",
        "fully solved", "correct solution", "valid solution",
    }

    if raw in _INCORRECT_SYNONYMS:
        return "incorrect"
    if raw in _ALMOST_SYNONYMS:
        return "almost"
    if raw in _PARTIAL_SYNONYMS:
        return "partial"
    if raw in _CORRECT_SYNONYMS:
        return "correct"

    # Prefix matching — STRICT ORDER: incorrect, almost, partial, correct
    if raw.startswith("incorrect") or raw.startswith("wrong") or raw.startswith("error"):
        return "incorrect"
    if raw.startswith("almost") or raw.startswith("nearly complete"):
        return "almost"
    if raw.startswith("partial") or raw.startswith("nearly") or raw.startswith("mostly"):
        return "partial"
    if raw.startswith("correct") or raw.startswith("right"):
        return "correct"

    # Last-resort keyword scan — most-specific / longest matches first.
    # "incorrect" must be checked before "correct" (substring containment).
    if "incorrect" in raw or "wrong" in raw or "mistake" in raw:
        return "incorrect"
    if "almost" in raw:
        return "almost"
    if "partial" in raw or "nearly" in raw or "mostly" in raw or "incomplete" in raw:
        return "partial"
    if "correct" in raw or "right" in raw or "valid" in raw:
        return "correct"

    return "incorrect"


def _extract_prediction(text: str) -> str | None:
    """Try multiple strategies to pull a label out of model output.

    Returns the first valid label found, prioritising structured JSON output.
    All four harness labels are recognised: correct / almost / partial / incorrect.
    """
    if not text or not isinstance(text, str):
        return None

    text = text.strip()

    # Strategy 1: <json>...</json> block (most reliable)
    extracted = _extract_jsons(text)
    if extracted:
        for obj in reversed(extracted):
            if "response" in obj:
                label = _normalise_label(str(obj["response"]))
                if label in _VALID_LABELS:
                    return label

    # Strategy 2: bare JSON object with "response" field
    json_patterns = [
        r'\{[^{}]*"response"\s*:\s*"([^"]+)"[^{}]*\}',
        r"\{[^{}]*'response'\s*:\s*'([^']+)'[^{}]*\}",
        r'\{\s*"response"\s*:\s*"([^"]+)"\s*\}',
    ]
    for pattern in json_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            label = _normalise_label(match.group(1))
            if label in _VALID_LABELS:
                return label

    # Strategy 3: explicit field patterns (grade:, verdict:, etc.)
    _LABEL_RE = r"(correct|almost|partial|incorrect)"
    field_patterns = [
        rf'(?i)\bgrade\s*[:=]\s*"?{_LABEL_RE}"?\b',
        rf'(?i)\bverdict\s*[:=]\s*"?{_LABEL_RE}"?\b',
        rf'(?i)\bresponse\s*[:=]\s*"?{_LABEL_RE}"?\b',
        rf'(?i)\bresult\s*[:=]\s*"?{_LABEL_RE}"?\b',
        rf'(?i)\bstatus\s*[:=]\s*"?{_LABEL_RE}"?\b',
        rf'(?i)\blabel\s*[:=]\s*"?{_LABEL_RE}"?\b',
    ]
    for pattern in field_patterns:
        m = re.search(pattern, text)
        if m:
            return _normalise_label(m.group(1))

    # Strategy 4: contextual phrases
    context_patterns = [
        rf'(?i)the\s+(?:answer|grade|verdict)\s+is\s+{_LABEL_RE}',
        rf'(?i)(?:therefore|thus|hence|conclusion|final\s+grade)\s*[:,-]?\s*{_LABEL_RE}',
        rf'(?i)(?:assessment|evaluation)\s*[:,-]?\s*{_LABEL_RE}',
    ]
    for pattern in context_patterns:
        m = re.search(pattern, text)
        if m:
            return _normalise_label(m.group(1))

    # Strategy 5: labels in quotes — longest labels first to avoid prefix matches
    _LABELS_JOINED = "|".join(sorted(_VALID_LABELS, key=len, reverse=True))
    for pattern in [
        rf'"({_LABELS_JOINED})"',
        rf"'({_LABELS_JOINED})'",
        rf"`({_LABELS_JOINED})`",
    ]:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return _normalise_label(m.group(1))

    # Strategy 6: last resort — whole-word scan in priority order.
    # Check "incorrect" before "correct" to avoid substring false-positives.
    for label in ["incorrect", "almost", "partial", "correct"]:
        if re.search(rf'(?i)\b{label}\b', text):
            return label

    return None


def _has_positive_achievement(milestones: str) -> bool:
    """Return True if the milestones text contains ACHIEVED or PARTIALLY ACHIEVED
    (but not only NOT ACHIEVED entries).

    'NOT ACHIEVED' contains 'ACHIEVED' as a substring, so we strip all
    'not achieved' occurrences before checking for the remaining 'achieved'.
    """
    if not milestones:
        return False
    cleaned = re.sub(r"\bnot\s+achieved\b", "", milestones, flags=re.IGNORECASE)
    return bool(re.search(r"\bachieved\b", cleaned, re.IGNORECASE))


def _almost_flaw_is_present(almost_txt: str) -> bool:
    """Return True if the almost_assessment field confirms the (Almost) flaw IS PRESENT.

    Checks for "IS PRESENT" while excluding "IS ABSENT" to avoid false positives
    when both phrases appear in the same field.
    """
    if not almost_txt:
        return False
    txt = almost_txt.lower()
    # "is present" must appear and "is absent" must NOT dominate
    has_present = "is present" in txt
    has_absent = "is absent" in txt
    if not has_present:
        return False
    if has_absent:
        # Both appear — count occurrences to determine which dominates
        present_count = txt.count("is present")
        absent_count = txt.count("is absent")
        return present_count > absent_count
    return True


def _almost_flaw_is_absent(almost_txt: str) -> bool:
    """Return True if the almost_assessment field confirms the (Almost) flaw IS ABSENT."""
    if not almost_txt:
        return False
    txt = almost_txt.lower()
    has_absent = "is absent" in txt
    has_present = "is present" in txt
    if not has_absent:
        return False
    if has_present:
        absent_count = txt.count("is absent")
        present_count = txt.count("is present")
        return absent_count > present_count
    return True


def _post_process_prediction(prediction: str, sources: list[str]) -> str:
    """Apply consistency rules to the extracted prediction using JSON fields.

    Corrects common model errors:
    - gaps listed but response="correct"       → downgrade to "almost" or "partial"
    - (Almost) flaw IS PRESENT + "correct"     → downgrade to "almost"
    - (Almost) flaw IS ABSENT + "almost"       → upgrade to "correct" (if no gaps)
    - milestone achieved + "incorrect"         → upgrade to "partial"
    - no milestones achieved + "partial"       → downgrade to "incorrect"
    - reasoning mentions near-correctness but response="incorrect" → upgrade to "partial"
    """
    for src in sources:
        extracted = _extract_jsons(src)
        if not extracted:
            continue

        for obj in reversed(extracted):
            resp_field = str(obj.get("response", "")).strip().lower()
            gaps_text  = str(obj.get("gaps_and_errors", "")).strip().lower()
            almost_txt = str(obj.get("almost_assessment", "")).strip().lower()
            milestones = str(obj.get("milestones_achieved", "")).strip().lower()
            reasoning  = str(obj.get("reasoning", "")).strip().lower()

            # Normalise the response field itself
            if resp_field in _VALID_LABELS:
                prediction = resp_field

            # ── Rule 1: gaps exist but verdict is "correct" ──────────────────
            # If the model listed real gaps, it cannot be "correct".
            has_real_gaps = (
                gaps_text
                and gaps_text not in {"none", "n/a", "no gaps", "no errors", ""}
                and not gaps_text.startswith("none")
            )
            if has_real_gaps and prediction == "correct":
                # Check whether the (Almost) flaw is present → "almost"
                # otherwise fall back to "partial"
                if _almost_flaw_is_present(almost_txt):
                    prediction = "almost"
                else:
                    prediction = "partial"

            # ── Rule 2: (Almost) flaw IS PRESENT → must be "almost" or lower ─
            # If the model confirmed the flaw is present but said "correct",
            # downgrade to "almost".
            if _almost_flaw_is_present(almost_txt) and prediction == "correct":
                prediction = "almost"

            # ── Rule 3: (Almost) flaw IS ABSENT → cannot be "almost" ─────────
            # If the model confirmed the flaw is absent but said "almost",
            # upgrade to "correct" (no other gaps were listed).
            if (
                _almost_flaw_is_absent(almost_txt)
                and prediction == "almost"
                and not has_real_gaps
            ):
                prediction = "correct"

            # ── Rule 4: milestone achieved but verdict is "incorrect" ─────────
            # Any ACHIEVED or PARTIALLY ACHIEVED milestone → at least "partial".
            if _has_positive_achievement(milestones) and prediction == "incorrect":
                prediction = "partial"

            # ── Rule 5: no milestones achieved but verdict is "partial" ───────
            # If every milestone is NOT ACHIEVED and there is no (Almost) match,
            # downgrade to "incorrect".
            almost_met = _almost_flaw_is_present(almost_txt)
            if (
                milestones
                and not _has_positive_achievement(milestones)
                and not almost_met
                and prediction == "partial"
            ):
                prediction = "incorrect"

            # ── Rule 6: reasoning signals near-correctness but verdict is "incorrect" ──
            # If the model's own reasoning explicitly mentions achieving a milestone
            # or partial credit, but the verdict is "incorrect", upgrade to "partial".
            # Only fires on unambiguously positive signals to avoid false positives
            # from negated contexts like "no progress" or "no milestone achieved".
            if prediction == "incorrect" and reasoning:
                # Look for explicit positive achievement phrases
                _positive_achievement_phrases = [
                    "achieved milestone",
                    "milestone achieved",
                    "partially achieved",
                    "milestone 1 achieved",
                    "milestone 2 achieved",
                    "rule c",
                    "at least one milestone",
                    "one milestone",
                ]
                _has_positive_reasoning = any(
                    phrase in reasoning for phrase in _positive_achievement_phrases
                )
                if _has_positive_reasoning:
                    prediction = "partial"

            return prediction  # use the first JSON block found

    return prediction


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
                "almost"    -- near-complete; only minor non-negligible mistakes
                "partial"   -- meaningful progress but substantially incomplete
                "incorrect" -- no substantive progress toward any milestone
        """
        problem        = inputs.get("problem", "")
        solution       = inputs.get("solution", "")
        guidelines     = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        domain         = inputs.get("domain", "")

        instruction = f"""You are an expert grader evaluating a student's solution to a competition mathematics problem.

## GRADING SCALE — four labels, read all definitions carefully

**"correct"**: The student's solution is complete and fully valid.
- Every key step, lemma, and logical connection required by the grading guidelines is present and sound.
- A trivially-fixable arithmetic slip or minor notational issue is acceptable.
- Any missing proof step, unjustified claim, or non-trivial logical gap disqualifies "correct".

**"almost"**: The student's solution is nearly complete but has exactly one specific minor flaw described in the rubric.
- The overall approach and structure are correct; the solution would be complete if only that one flaw were fixed.
- Use "almost" ONLY when ALL THREE conditions hold:
  (a) The grading guidelines contain an "(Almost)" section.
  (b) The specific flaw described in the (Almost) section IS PRESENT in the student's work.
  (c) The student's approach is otherwise sound — no large gaps beyond the described flaw.
- ⚠️ CRITICAL: If the (Almost) flaw IS PRESENT and the approach is otherwise sound, you MUST output "almost", NOT "correct".
- Do NOT assign "almost" if the (Almost) flaw is ABSENT from the student's work (that may be "correct").
- Do NOT assign "almost" if the student has large gaps beyond the described minor flaw (that is "partial").

**"partial"**: The student has made meaningful, substantive mathematical progress.
- The student achieved AT LEAST ONE specific milestone listed in the "(Partial)" section.
- A milestone is ACHIEVED if the student clearly demonstrates it.
- A milestone is PARTIALLY ACHIEVED if the student engages substantively with its core content, even if execution is incomplete.
- Be GENEROUS: award "partial" whenever the student has made real mathematical progress toward any listed milestone.
- "partial" is the correct label for solutions that are mostly complete but have non-trivial gaps (not "correct" or "almost").

**"incorrect"**: The student's answer is wrong or makes no meaningful progress.
- The student achieved NONE of the specific milestones listed in the "(Partial)" section.
- Merely restating the problem, making trivial observations, or producing a correct final answer with no supporting argument earns "incorrect".

---

## DECISION PROCEDURE — follow Rules A through D in strict order; stop at the first match

**Rule A — Full correctness check:**
- List every genuine logical gap, missing case, or unjustified claim in the student's work.
- If ZERO non-trivial gaps exist → grade is **"correct"**. Stop.

**Rule B — (Almost) check (apply only if Rule A did not fire):**
1. Does the grading guidelines contain an "(Almost)" section?
2. Is the specific flaw described there PRESENT in the student's work? (Quote the evidence.)
3. Is the student's overall approach otherwise sound — would the solution be correct if only that flaw were fixed?
→ If YES to all three → grade is **"almost"**. Stop. ⚠️ You MUST output "almost" here, not "correct".
→ If the flaw is ABSENT → do NOT assign "almost"; re-examine Rule A or continue to Rule C.
→ If the student has additional large gaps beyond the described flaw → do NOT assign "almost"; go to Rule C.

**Rule C — (Partial) milestone check (apply only if Rules A and B did not fire):**
- For EACH milestone in the "(Partial)" section, independently assess:
  - ACHIEVED: student clearly demonstrates the milestone.
  - PARTIALLY ACHIEVED: student engages substantively with the milestone's core content, even if execution is incomplete.
  - NOT ACHIEVED: student shows no meaningful engagement with the milestone's content.
- If AT LEAST ONE milestone is ACHIEVED or PARTIALLY ACHIEVED → grade is **"partial"**. Stop.
- IMPORTANT: Be generous here. If the student has made any real mathematical progress toward a milestone — even partial — award "partial".

**Rule D — Default:**
- No milestone achieved or partially achieved, no (Almost) match, not fully correct → grade is **"incorrect"**.

---

## CALIBRATION NOTES

- **"almost" vs "correct"**: The (Almost) flaw must be PRESENT in the student's work to assign "almost". If the flaw IS PRESENT and the approach is otherwise sound, you MUST output "almost" — outputting "correct" when the flaw is present is a grading error. If the flaw is absent, the student may deserve "correct" (no other gaps) or "partial" (other gaps exist).
- **"almost" vs "partial"**: "almost" means overwhelmingly complete with one specific blemish matching the rubric. "partial" means substantially incomplete or has significant errors. Do not confuse them.
- **"partial" vs "incorrect"**: This is the most common grading error. Be GENEROUS with "partial". If the student has engaged substantively with even one milestone — even imperfectly — award "partial". Only assign "incorrect" if there is truly zero meaningful progress toward any milestone.
- **Non-standard solutions**: A complete, logically sound solution using a different method is "correct". Do not penalise valid alternative approaches.
- **Strict correct**: Only assign "correct" if there are genuinely zero non-trivial gaps AND the (Almost) flaw (if any) is absent.

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

**Step 1 — List all milestones:** Copy every milestone from the grading guidelines verbatim, labelling each as (Partial) or (Almost).

**Step 2 — Gap analysis (Rule A):** List every genuine logical gap, missing case, or unjustified claim in the student's work. Write "none" if fully correct. If "none" → verdict is "correct"; skip Steps 3–4.

**Step 3 — (Almost) check (Rule B):** For each (Almost) criterion:
- State whether the described flaw IS PRESENT or IS ABSENT in the student's work.
- Quote the specific part of the student's answer that supports your decision.
- ⚠️ If the flaw IS PRESENT and the overall approach is otherwise sound (no large additional gaps) → verdict is "almost"; skip Step 4. You MUST output "almost" in the response field, not "correct".
- If the flaw IS ABSENT → do not assign "almost"; continue to Step 4.

**Step 4 — (Partial) milestone check (Rule C):** For each (Partial) milestone:
- State ACHIEVED / PARTIALLY ACHIEVED / NOT ACHIEVED with a one-sentence justification.
- Quote the specific part of the student's answer that supports your decision.
- Remember: be GENEROUS. Substantive engagement with a milestone's core content counts as PARTIALLY ACHIEVED.
- If at least one is ACHIEVED or PARTIALLY ACHIEVED → verdict is "partial".
- If ALL are NOT ACHIEVED → verdict is "incorrect".

**Step 5 — Self-check before writing JSON:**
- If you wrote "IS PRESENT" for any (Almost) criterion AND the approach is otherwise sound → response MUST be "almost".
- If you wrote "ACHIEVED" or "PARTIALLY ACHIEVED" for any (Partial) milestone → response MUST be "partial" or better.
- If response is "correct" but you listed any gaps → change response to "almost" (if (Almost) flaw present) or "partial".

Respond with a JSON block in the following format:
<json>
{{
    "gaps_and_errors": "<list genuine logical gaps or errors; write 'none' if fully correct>",
    "almost_assessment": "<for each (Almost) criterion: IS PRESENT / IS ABSENT in student's work + quoted evidence; or 'n/a' if no (Almost) section>",
    "milestones_achieved": "<for each (Partial) milestone: text — ACHIEVED/PARTIALLY ACHIEVED/NOT ACHIEVED + one-sentence reason quoting the student's work>",
    "reasoning": "<one-sentence justification of your final verdict citing the decisive rule>",
    "response": "<correct | almost | partial | incorrect>"
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

            # Apply consistency post-processing using structured JSON fields.
            prediction = _post_process_prediction(prediction, sources)

        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return prediction, msg_history
