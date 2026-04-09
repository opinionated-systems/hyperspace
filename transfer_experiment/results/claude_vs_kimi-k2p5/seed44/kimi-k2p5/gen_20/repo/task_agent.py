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

# Valid prediction labels. The harness uses all four as distinct categories:
#   "correct"   – fully correct solution
#   "almost"    – near-complete; only minor non-negligible mistakes remain
#   "partial"   – meaningful progress but substantially incomplete
#   "incorrect" – no substantive progress toward any milestone
_VALID_LABELS = {"correct", "almost", "partial", "incorrect"}

# Ordinal rank used for consistency checks: incorrect=0, partial=1, almost=2, correct=3
_LABEL_RANK = {"incorrect": 0, "partial": 1, "almost": 2, "correct": 3}


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
    """Map a raw model response string to one of the four canonical labels.

    The evaluation harness accepts exactly: "correct", "almost", "partial",
    "incorrect". Priority order for substring matching (most specific first):
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
        "minor mistake", "minor error", "small mistake", "small error",
        "not completed", "incomplete verification", "verification incomplete",
        "omitted case", "missing case", "minor gap", "small gap",
        "nearly there", "almost there", "one small issue", "one minor issue",
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
    
    milestones_lower = milestones.lower().strip()
    
    # Handle n/a or empty cases
    if milestones_lower in ["n/a", "na", "none", "", "not applicable"]:
        return False
    
    # First check for explicit positive indicators
    positive_indicators = ["achieved", "partially achieved", "completed", "satisfied"]
    negative_indicators = ["not achieved", "not completed", "not satisfied", "missing"]
    
    # Remove all negative indicators first
    cleaned = milestones_lower
    for neg in negative_indicators:
        cleaned = re.sub(rf"\b{neg}\b", "", cleaned)
    
    # Check if any positive indicators remain
    for pos in positive_indicators:
        if re.search(rf"\b{pos}\b", cleaned):
            return True
    
    # Additional check: if milestones section exists and has content but no explicit "not achieved"
    # and mentions student work or progress, consider it as having achievement
    if len(milestones_lower) > 10 and "not achieved" not in milestones_lower:
        # Check for evidence of student work being described
        work_indicators = ["student", "answer", "solution", "proof", "shows", "demonstrates", 
                          "found", "identified", "proved", "established"]
        if any(w in milestones_lower for w in work_indicators):
            return True
    
    return False


def _post_process_prediction(prediction: str, sources: list[str]) -> str:
    """Apply consistency rules to the extracted prediction using JSON fields.

    Corrects common model errors:
    - gaps listed but response="correct"       → downgrade to "almost" or "partial"
    - (Almost) flaw IS PRESENT + "correct"     → downgrade to "almost"
    - (Almost) flaw IS ABSENT + "almost"       → upgrade to "correct" (if no gaps)
    - milestone achieved + "incorrect"         → upgrade to "partial"
    - no milestones achieved + "partial"       → downgrade to "incorrect"
    """
    for src in sources:
        extracted = _extract_jsons(src)
        if not extracted:
            continue

        for obj in reversed(extracted):
            resp_field = str(obj.get("response", "")).strip().lower()
            gaps_text = str(obj.get("gaps_and_errors", "")).strip().lower()
            almost_txt = str(obj.get("almost_assessment", "")).strip().lower()
            milestones_txt = str(obj.get("milestones_achieved", "")).strip().lower()
            reasoning_txt = str(obj.get("reasoning", "")).strip().lower()

            # Determine if gaps exist
            has_gaps = gaps_text not in ["", "none", "n/a", "na", "no gaps", "no errors", "fully correct"]

            # Determine if (Almost) flaw is present - more robust detection
            # Check for explicit "IS PRESENT" or similar affirmative language
            almost_present = False
            almost_absent = False
            
            # Look for explicit presence indicators
            presence_indicators = [
                "is present", "flaw is present", "criterion is present",
                "mistake is present", "error is present", "issue is present",
                "the flaw is present", "the criterion is present",
                "present in", "found in", "exists in", "observed in",
                "student's work contains", "student has", "student made",
                "contains minor", "has minor", "minor mistakes", "minor errors",
                "verification incomplete", "not completed", "omitted case",
                "missing case", "small gap", "minor gap", "is present: yes",
                "present: yes", "flaw present", "criterion present"
            ]
            absence_indicators = [
                "is absent", "flaw is absent", "criterion is absent",
                "mistake is absent", "error is absent", "issue is absent",
                "the flaw is absent", "the criterion is absent",
                "absent in", "not present", "not found", "does not exist",
                "not observed", "student did not", "no evidence of",
                "no minor mistakes", "no errors", "complete verification",
                "fully justified", "all cases covered", "is present: no",
                "present: no", "flaw absent", "criterion absent"
            ]
            
            almost_lower = almost_txt.lower()
            for indicator in presence_indicators:
                if indicator in almost_lower:
                    almost_present = True
                    break
            
            for indicator in absence_indicators:
                if indicator in almost_lower:
                    almost_absent = True
                    break
            
            # Also check reasoning field for almost-related keywords
            reasoning_lower = reasoning_txt.lower()
            if "rule b" in reasoning_lower or "almost" in reasoning_lower:
                # If reasoning mentions Rule B or almost, check if it's the deciding factor
                if "flaw" in reasoning_lower or "criterion" in reasoning_lower or "almost" in reasoning_lower:
                    # Check if the model's reasoning suggests almost
                    if any(x in reasoning_lower for x in ["is present", "present", "flaw", "minor mistake", "not completed", "minor error", "verification"]):
                        almost_present = True

            # Determine if milestones were achieved
            has_achievement = _has_positive_achievement(milestones_txt)

            # Rule 1: gaps exist but response is "correct" → downgrade
            if has_gaps and prediction == "correct":
                # If (Almost) flaw is present, downgrade to "almost"
                if almost_present:
                    return "almost"
                # Otherwise downgrade to "partial"
                return "partial"

            # Rule 2: (Almost) flaw IS PRESENT but response is "correct" → downgrade to "almost"
            if almost_present and prediction == "correct":
                return "almost"

            # Rule 3: (Almost) flaw IS ABSENT but response is "almost" → upgrade if no gaps
            if almost_absent and prediction == "almost" and not has_gaps:
                return "correct"

            # Rule 4: milestone achieved but response is "incorrect" → upgrade to "partial"
            if has_achievement and prediction == "incorrect":
                return "partial"

            # Rule 5: no milestones achieved but response is "partial" → downgrade to "incorrect"
            if not has_achievement and prediction == "partial":
                # Check if milestones section exists and is not N/A
                if milestones_txt and milestones_txt != "n/a":
                    return "incorrect"
            
            # Rule 6: reasoning mentions "almost" or "Rule B" as decisive but response is not "almost"
            # This catches cases where model correctly identified the flaw but didn't output "almost"
            if prediction != "almost" and almost_present:
                # Check if reasoning explicitly mentions Rule B or almost criterion as decisive
                if "rule b" in reasoning_lower or "almost criterion" in reasoning_lower or "almost" in reasoning_lower:
                    if any(x in reasoning_lower for x in ["decisive", "determines", "verdict", "grade is", "almost", "therefore", "thus"]):
                        return "almost"
            
            # Rule 7: if almost_assessment mentions specific almost keywords and response is "correct"
            # but there are gaps, force "almost"
            if prediction == "correct" and has_gaps:
                almost_keywords = ["not completed", "minor mistake", "verification", "omitted", "incomplete proof", "minor error", "small error", "minor gap"]
                if any(kw in almost_lower for kw in almost_keywords):
                    return "almost"
            
            # Rule 8: if almost_assessment is not empty and not n/a, and mentions almost-related content
            # but response is "partial" or "incorrect", check if it should be "almost"
            if prediction in ["partial", "incorrect"] and almost_txt and almost_txt != "n/a":
                # Check if the almost assessment contains content suggesting the flaw is present
                almost_content_indicators = ["minor", "small", "verification", "omitted", "incomplete", "not completed"]
                if any(kw in almost_lower for kw in almost_content_indicators):
                    # Check if the solution is otherwise sound (not too many gaps)
                    gap_count = gaps_text.count("missing") + gaps_text.count("gap") + gaps_text.count("error")
                    if not has_gaps or gap_count < 3:
                        return "almost"
            
            # Rule 9: if response is "almost" but no almost indicators found, check if it should be "correct" or "partial"
            if prediction == "almost" and not almost_present:
                if not has_gaps:
                    return "correct"
                # Check if there are substantial gaps that would make it "partial" instead
                gap_count = gaps_text.count("missing") + gaps_text.count("gap") + gaps_text.count("error")
                if gap_count > 2:
                    return "partial"
            
            # Rule 10: if reasoning explicitly mentions "almost" as the verdict but response is different
            if "verdict is almost" in reasoning_lower or "grade is almost" in reasoning_lower or "almost" in resp_field:
                if prediction != "almost" and almost_present:
                    return "almost"
            
            # Rule 11: if gaps_and_errors contains "none" or similar but response is "partial" or "incorrect"
            # check if it should be upgraded
            if not has_gaps and prediction in ["partial", "incorrect"]:
                # If no gaps and no almost flaw, should be "correct"
                if not almost_present:
                    return "correct"
                # If no gaps but almost flaw present, should be "almost"
                if almost_present:
                    return "almost"

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
                "correct"   – the student's answer is fully correct
                "partial"   – the student's answer is partially correct
                "incorrect" – the student's answer is wrong or missing
        """
        problem        = inputs.get("problem", "")
        solution       = inputs.get("solution", "")
        guidelines     = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        domain         = inputs.get("domain", "")

        instruction = f"""You are a strict but fair expert grader evaluating a student's solution to a competition mathematics problem.

## Your task
Read the problem, the official solution, the grading guidelines, and the student's answer.
Then decide whether the student's answer deserves a grade of **"correct"**, **"almost"**, **"partial"**, or **"incorrect"**.

### Grading definitions — read carefully before grading

**CORRECT**: The student's answer is fully correct and complete.
- Every key step, lemma, and logical connection required by the grading guidelines is present and sound.
- A minor notational slip or trivially-fixable arithmetic error is acceptable.
- **CRITICAL**: Any missing proof step, unjustified claim, or logical gap disqualifies "correct".
- When in doubt between "correct" and "almost", choose **"almost"**.

**ALMOST**: The student's answer is nearly complete with only minor non-negligible mistakes.
- The solution is overwhelmingly correct but has one specific flaw described in the "(Almost)" section of the grading guidelines.
- **CRITICAL**: The "(Almost)" flaw MUST be PRESENT in the student's work to assign "almost". Look for phrases like "not completed", "minor mistakes", "verification incomplete", "omitted case", etc.
- "Almost" is closer to correct than "partial" — it represents work that would be "correct" if not for a specific blemish.
- **Common "almost" scenarios**: incomplete verification, minor calculation errors in an otherwise correct approach, missing a small case that doesn't affect the main argument, proof structure correct but one step not fully justified.
- If the "(Almost)" flaw is ABSENT, do not use this grade; use "correct" (if no gaps) or "partial" (if other gaps exist).
- **EXAMPLE**: If guidelines say "(Almost) Verification contains minor mistakes only" and the student's verification has small errors → grade "almost".
- **EXAMPLE**: If guidelines say "(Almost) Minor mistakes" and student has one small arithmetic error in an otherwise perfect proof → grade "almost".
- **EXAMPLE**: If guidelines say "(Almost) Not completed" and student has a complete solution but missing final verification step → grade "almost".

**PARTIAL**: The student's answer contains meaningful correct progress but is substantially incomplete.
- The student achieved at least one specific milestone from the "(Partial)" section of the grading guidelines.
- The progress must be substantive — it must address a genuine difficulty of the problem.
- "Mostly correct" or "incomplete" answers with significant gaps belong here.
- Be GENEROUS: if the student made ANY substantive progress toward a milestone, award "partial".
- Only assign "incorrect" if there is truly zero meaningful progress toward any milestone.
- **EXAMPLES of partial credit**: 
  - Student found a key invariant or pattern but didn't complete the proof
  - Student proved one direction of a biconditional but not the other
  - Student set up the right approach but made significant errors in execution
  - Student identified the right cases but only solved some of them
  - Student made significant progress on the problem but the solution has major logical gaps
  - Student understood the core concept but application was incomplete or flawed
- **KEY DISTINCTION**: "partial" = substantial progress with significant gaps remaining; "almost" = nearly complete with only one minor flaw

**INCORRECT**: The student's answer is wrong or makes no meaningful progress.
- The student made no substantive progress toward solving the problem.
- No key insights, lemmas, or structural observations from the grading guidelines are present.
- Merely restating the problem or making trivial observations earns "incorrect".
- A correct final numerical answer with no supporting argument is **incorrect** unless guidelines explicitly award credit for the answer alone.

### Critical distinctions

**almost vs correct**: "almost" means overwhelmingly complete with one specific blemish matching the rubric. Only assign "correct" if the solution is truly complete and gap-free. If the "(Almost)" flaw is present, you MUST choose "almost" over "correct".
- Example: Guidelines say "(Almost) Verification incomplete" and student has a complete proof but verification is missing → "almost"
- Example: Guidelines say "(Almost) Minor calculation error" and student has one small arithmetic mistake → "almost"

**almost vs partial**: "almost" is much closer to correct than "partial". "almost" = one minor flaw away from correct; "partial" = substantial gaps or errors remain. A solution that is mostly complete with one small issue is "almost", not "partial".
- "almost": 90-99% complete, one specific minor issue
- "partial": 10-80% complete, significant work remains

**partial vs incorrect**: Be GENEROUS with "partial". If the student engaged substantively with even one milestone — even imperfectly — award "partial". Only assign "incorrect" for truly zero meaningful progress.
- "partial": Student found a key insight, proved a lemma, or made substantial progress on a milestone
- "incorrect": Only restated the problem, made trivial observations, or went completely wrong direction

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
- **CRITICAL**: If the (Almost) criterion describes something like "not completed", "minor mistakes", "verification incomplete", "omitted case" — check carefully if this matches the student's work.
- **IMPORTANT**: The "(Almost)" section in guidelines describes a SPECIFIC flaw. If that flaw IS PRESENT in the student's work → verdict is "almost".
- If the flaw IS PRESENT and the overall approach is otherwise sound (no large additional gaps) → verdict is "almost"; skip Step 4.
- If the flaw IS ABSENT → do not assign "almost"; continue to Step 4.
- **DECISION RULE**: Only assign "almost" if: (1) the specific (Almost) flaw is PRESENT, AND (2) the solution is otherwise complete with no other significant gaps.

**Step 4 — (Partial) milestone check (Rule C):** For each (Partial) milestone:
- State ACHIEVED / PARTIALLY ACHIEVED / NOT ACHIEVED with a one-sentence justification.
- Quote the specific part of the student's answer that supports your decision.
- Remember: be GENEROUS. Substantive engagement with a milestone's core content counts as PARTIALLY ACHIEVED.
- If at least one is ACHIEVED or PARTIALLY ACHIEVED → verdict is "partial".
- If ALL are NOT ACHIEVED → verdict is "incorrect".
- **DECISION RULE**: When in doubt between "partial" and "incorrect", choose "partial" if the student showed any meaningful mathematical insight.

Respond with a JSON block in the following format:
<json>
{{
    "gaps_and_errors": "<list genuine logical gaps or errors; write 'none' if fully correct>",
    "almost_assessment": "<for each (Almost) criterion: IS PRESENT / IS ABSENT in student's work + quoted evidence; or 'n/a' if no (Almost) section>",
    "milestones_achieved": "<for each (Partial) milestone: text - ACHIEVED/PARTIALLY ACHIEVED/NOT ACHIEVED + one-sentence reason quoting the student's work>",
    "reasoning": "<one-sentence justification of your final verdict citing the decisive rule>",
    "response": "<correct | almost | partial | incorrect>"
}}
</json>

**FINAL VERIFICATION**: Before outputting the JSON, double-check:
1. If gaps_and_errors is "none" and almost_assessment shows IS ABSENT, response should be "correct"
2. If almost_assessment shows IS PRESENT and gaps are minor, response should be "almost"
3. If milestones_achieved shows at least one ACHIEVED or PARTIALLY ACHIEVED, response should be "partial" (unless rules above apply)
4. If all milestones are NOT ACHIEVED and no almost flaw, response should be "incorrect"
5. The response field MUST be exactly one of: correct, almost, partial, incorrect"""

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
