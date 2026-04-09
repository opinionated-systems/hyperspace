"""
Task agent: solves a given task with a single LLM call (plus optional
verification passes for borderline grades).

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

# Valid grade labels (output space).
# "almost" is a distinct 4th label used by the eval harness (worth 6/7 points).
# It applies when the student's answer satisfies an "(Almost)" criterion in the
# grading guidelines — essentially complete but with minor, non-negligible gaps.
_VALID_GRADES = {"correct", "almost", "partial", "incorrect"}

# Grades where a verification pass is worthwhile (borderline decisions).
# "correct"   -> verify whether any "(Almost)" criterion actually applies
#                (most common error: awarding "correct" when "almost" is right).
# "incorrect" -> verify whether any "(Partial)" criterion was missed
#                (second most common error: missing partial credit).
# "almost"    -> verify whether "almost" is truly warranted vs "partial"/"incorrect"
#                (over-predicting "almost" is very costly: 5/7 pts penalty per error).
# "partial"   -> verify whether "partial" is truly warranted vs "incorrect"
#                (false partial for true-incorrect is a common error: 1 pt awarded
#                 for a worthless answer).  Also verify whether "correct" applies
#                 (under-grading a complete solution as partial is costly: 6 pts lost).
# NOTE: "partial" is intentionally NOT verified for upgrade to "almost".
#       Empirical data shows partial->almost upgrades cause far more false positives
#       (5/7 pt penalty each) than they correct.  "almost" is only ~5% of answers.
_VERIFY_GRADES = {"incorrect", "correct", "almost", "partial"}

# Empirical observation: the second partial scan (Pass 3) causes more false
# positives (incorrect->partial) than it corrects.  Empirical data shows that
# after two independent graders both award "incorrect", a third pass rarely
# finds genuine partial credit and instead introduces noise.  Pass 3 is
# therefore disabled.
_ENABLE_PASS3_SECOND_PARTIAL_SCAN = False

# Pass 2f: a third independent partial check using a "devil's advocate" framing,
# run when BOTH Pass 1 AND Pass 2c return "incorrect".  Unlike the disabled Pass 3
# (which was a simple repeat of Pass 2c), Pass 2f uses a completely different
# prompt framing that focuses on SPECIFIC sub-results rather than overall quality.
# Empirical motivation: partial recall has declined from 0.60 (gen_30) to 0.40
# (gen_38) -- a 20% drop.  The two-pass approach (Pass 1 + Pass 2c) is not
# sufficient; a third independent check with a different framing is needed.
# Pass 2f is STRICT about what counts as partial (must quote a specific passage)
# but THOROUGH in searching for it (checks every criterion independently).
_ENABLE_PASS2F_THIRD_PARTIAL_SCAN = True

# Expected base rates for calibration (from empirical data):
# correct ~35%, incorrect ~35%, partial ~25%, almost ~5%
_BASE_RATES = {"correct": 0.35, "almost": 0.05, "partial": 0.25, "incorrect": 0.35}

# Empirical precision of "almost" predictions across many grading runs.
# NOTE: The previous value of 0.04 was causing the agent to be so conservative
# about "almost" that it achieved 0% recall on true "almost" cases.  The true
# base rate of "almost" is ~5%, and when grading guidelines explicitly list
# "(Almost)" criteria, the agent should be able to detect them reliably.
# Raising this to reflect that explicit criterion matches ARE meaningful.
_ALMOST_EMPIRICAL_PRECISION = 0.40

# Empirical observation (updated gen_36): predicting "incorrect" for a "correct"
# answer is now the MOST COSTLY systematic error (7 pts per occurrence, 5 cases
# in gen_36 = −35 pts total).  The first-pass grader is too conservative about
# "correct" -- it invents gaps in solutions that are actually complete.
# The fix is in the Pass 1 prompt: add "benefit of the doubt" and "do not invent
# gaps" guidance to make the grader more generous about awarding "correct".
_CORRECT_UNDERGRADE_IS_MOST_COSTLY = True

# ── Empirical calibration notes (gen_38 eval, 100 problems) ──────────────────
# Error analysis on the most recent evaluation run (gen_38 val, 58% accuracy):
#
#   incorrect→partial: 11 cases, +11 pts ← Pass 2c over-awards partial.
#   partial→incorrect: 10 cases, −10 pts ← MOST FREQUENT ERROR: grader misses
#                                            partial credit even after Pass 2c.
#                                            Fix: add Pass 2f (third independent
#                                            partial check with different framing).
#   partial→correct  :  4 cases, +24 pts ← grader over-awards "correct" to partial
#                                            answers.  Fix: stricter correct gate.
#   correct→partial  :  3 cases, −18 pts ← grader under-awards correct answers.
#                                            Fix: benefit-of-the-doubt for "correct".
#   incorrect→correct:  2 cases, +14 pts ← grader over-awards "correct" to truly
#                                            incorrect answers.
#   correct→incorrect:  2 cases, −14 pts ← grader invents gaps.
#   incorrect→almost :  2 cases, +12 pts ← Pass 1 over-awards almost.
#   almost→correct   :  3 cases,  +3 pts ← almost-check pass (2a) misses "almost".
#   almost→incorrect :  1 case,   −6 pts ← Pass 2d over-fires.
#   almost→partial   :  1 case,   −5 pts ← Pass 2d over-fires.
#   partial→almost   :  1 case,   +5 pts ← Pass 1 over-awards almost.
#   correct→almost   :  2 cases,  −2 pts ← Pass 2a over-fires.
#
# KEY TREND: partial recall has declined from 0.60 (gen_30) to 0.40 (gen_38).
# This is the most significant regression.  The two-pass approach (Pass 1 + Pass 2c)
# is not sufficient for partial credit detection.  Adding Pass 2f (third independent
# partial check with a different framing) is the highest-priority fix.
#
# Net priority (by absolute pts impact):
#   1. partial→incorrect (−10 pts): add Pass 2f for third independent partial check
#   2. partial→correct   (+24 pts): tighten "correct" gate
#   3. correct→partial   (−18 pts): more generous "correct" for complete solutions
#   4. incorrect→correct (+14 pts): tighten "correct" gate
#   5. correct→incorrect (−14 pts): do-not-invent-gaps guidance
#   6. incorrect→partial (+11 pts): Pass 2c is already generous; don't over-correct
# ─────────────────────────────────────────────────────────────────────────────

# Pass 2b (correct→partial downgrade) is DISABLED: empirical data shows it
# produces more false downgrades (correct→incorrect, +14 pts lost) than it
# corrects (correct→partial, +6 pts saved).  Net effect is negative.
_ENABLE_PASS2B_CORRECT_DOWNGRADE = False

# Pass 2e (partial verify) is DISABLED: empirical data shows it produces more
# false downgrades (partial→incorrect, +7 pts lost) than it corrects.
# Net effect is negative.
_ENABLE_PASS2E_PARTIAL_VERIFY = False


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

    Looks for lines like 'Grade: almost' or a standalone grade word
    near the end of the response, in case the model forgot the JSON tags.
    """
    # Try labelled patterns first: "grade: almost", "verdict: partial", etc.
    labelled = re.search(
        r'(?:grade|verdict|assessment|result)\s*[:\-]\s*(correct|almost|partial|incorrect)',
        text,
        re.IGNORECASE,
    )
    if labelled:
        return labelled.group(1).lower()

    # Try a bare JSON-like fragment without tags: {"response": "almost"}
    bare_json = re.search(
        r'\{[^{}]*"response"\s*:\s*"(correct|almost|partial|incorrect)"[^{}]*\}',
        text,
        re.IGNORECASE,
    )
    if bare_json:
        return bare_json.group(1).lower()

    # Last resort: find the last occurrence of a valid grade word as a
    # standalone token (surrounded by non-word characters or string boundaries).
    # Check "almost" before "correct" so it is not shadowed.
    matches = list(re.finditer(r'\b(correct|almost|partial|incorrect)\b', text, re.IGNORECASE))
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
one of FOUR grades: "correct", "almost", "partial", or "incorrect".

================================================================
SCORING CONTEXT  (read this first -- it shapes every decision)
================================================================

The four grades map to point values:
  "correct"   -> 7 points   (fully correct; every essential step present and sound)
  "almost"    -> 6 points   (essentially complete; satisfies an "(Almost)" criterion)
  "partial"   -> 1 point    (genuine non-trivial progress; satisfies a "(Partial)" criterion)
  "incorrect" -> 0 points   (wrong, circular, or no meaningful progress)

EXPECTED BASE RATES (use as a calibration prior):
  Roughly 35% of answers are "correct", 35% are "incorrect",
  25% are "partial", and only 5% are "almost".
  "almost" is VERY RARE -- only ~1 in 20 answers.
  CALIBRATION CHECK: If you are about to assign "incorrect", pause and
  re-read every "(Partial)" criterion one more time.  Partial credit is
  awarded to ~25% of answers; missing it is the most common grading error.

================================================================
GRADE DEFINITIONS
================================================================

"correct"  (7 pts)
  Every essential logical step is present and sound AND the student
  reaches the correct final conclusion with no significant mathematical
  gap.  Minor presentational or notational differences are acceptable.

  BENEFIT OF THE DOUBT: If a step is standard or follows immediately
    from what the student wrote, do NOT penalise for not spelling it out.
    A terse but logically sound proof earns "correct".

  *** DO NOT INVENT GAPS: Only downgrade from "correct" if you can
    identify a SPECIFIC, CONCRETE missing step.  Vague concerns,
    terse presentation, or a different approach from the official
    solution are NOT reasons to downgrade. ***

  STRICT COMPLETENESS REQUIREMENT: Do NOT award "correct" if:
    - The student only proves part of what is required (e.g., proves
      necessary conditions but not sufficiency, or proves one direction
      of an iff but not the other, or handles some cases but not all).
    - The student only satisfies a "(Partial)" criterion and does NOT
      complete the full proof.  A well-written partial solution is still
      only worth 1 point.
    - The student satisfies an "(Almost)" criterion -- that earns "almost".
    BEFORE awarding "correct", explicitly verify:
      (a) Does the student's answer go BEYOND all "(Partial)" criteria?
      (b) Does the student actually COMPLETE the full proof?
      (c) Does the student reach the CORRECT FINAL ANSWER/CONCLUSION?
      (d) Does ANY "(Almost)" criterion apply?  If yes -> "almost".

"almost"  (6 pts)
  The student's answer is essentially complete but has minor,
  non-negligible gaps or mistakes -- it satisfies at least one
  "(Almost)" criterion in the grading guidelines.

  RULE: If the student's answer satisfies ANY "(Almost)" criterion,
    the grade MUST be "almost".  STOP -- do not check further.

  REQUIREMENTS for "almost" -- ALL must be true:
    (A) The student's overall approach is CORRECT (not just plausible).
    (B) The solution is GENUINELY NEAR-COMPLETE (90%+ done).
    (C) The mistakes are TRULY MINOR (not fundamental errors).
    (D) The answer EXPLICITLY satisfies a specific "(Almost)" criterion.

  SPECIAL RULE for "omitted the case" or "almost complete" criteria:
    These apply ONLY when the student uses the SAME overall approach as
    the official solution AND has completed essentially ALL of the proof
    except the named omission.

  *** STRONG DEFAULT: When uncertain between "almost" and "correct",
    choose "correct".  When uncertain between "almost" and "partial",
    choose "partial".  Only award "almost" when you are CERTAIN the
    answer is genuinely near-complete with the correct approach AND
    you can quote a specific "(Almost)" criterion that is satisfied. ***

"partial"  (1 pt)
  The student makes genuine, non-trivial mathematical progress but the
  answer is substantially incomplete or contains significant errors.
  Award "partial" when the student's answer EXPLICITLY and VERIFIABLY
  satisfies AT LEAST ONE "(Partial)" criterion listed in the grading
  guidelines.

  *** GENEROUS STANDARD FOR PARTIAL CREDIT ***
  Credit the student if the mathematical substance of a "(Partial)"
  criterion is present ANYWHERE in their answer -- even if:
    - The write-up is informal or uses different notation.
    - The criterion is reached as a by-product of a larger failed argument.
    - The student introduces the exact object/construction named in the
      criterion (e.g. "considered a prime p | xy+1") even without completing
      the argument from there.
    - The student correctly identifies the answer/value named in the criterion.
    - The student proves the required fact as a lemma or intermediate step.

  *** CRITICAL: Do NOT require the student to have a complete or correct
    overall solution to earn partial credit.  Partial credit is for
    sub-results only.  Even a completely wrong overall solution can earn
    "partial" if it contains the specific sub-result named in a criterion. ***

  Do NOT award "partial" based on vague similarity to a criterion,
    general mathematical knowledge, or a long but ultimately flawed
    argument that never reaches the criterion's specific conclusion.

"incorrect"  (0 pts)
  The student's answer is wrong, fundamentally flawed, or makes no
  meaningful progress toward the solution.

  BEFORE CHOOSING "incorrect": Re-read every single "(Partial)"
    criterion one more time.  For each one, ask: "Is there ANY passage
    in the student's answer that establishes this specific fact, even
    informally or as part of a failed larger argument?"  Only if the
    answer is definitively NO for every criterion should you choose
    "incorrect".

================================================================
DECISION PROCEDURE  (follow steps in order; stop at first match)
================================================================

STEP 1 -- CORRECT CHECK
  a. Are ALL essential logical steps present and sound?
  b. Does the student reach the correct final conclusion?
  c. Does the student prove BOTH directions if required?
  d. Does the student's answer go BEYOND all "(Partial)" criteria?
     (i.e., is it a complete proof, not just a sub-result?)
  -> ALL YES -> BEFORE marking "correct", run the CORRECT GATE below.
  -> OTHERWISE -> continue to Step 2.

  CORRECT GATE (mandatory before awarding "correct"):
    - Does ANY "(Almost)" criterion apply?  If yes -> "almost".
    - Does the student only satisfy a "(Partial)" criterion without
      completing the full proof?  If yes -> "partial".
    - Is there a SPECIFIC, IDENTIFIABLE gap?  If yes -> downgrade.
      Do NOT invent gaps.  Only downgrade for concrete missing steps.
    Only if ALL checks pass -> grade is "correct".

STEP 2 -- ALMOST CHECK
  Read every "(Almost)" criterion in the grading guidelines verbatim.
  For each criterion, ask: does the student's answer EXPLICITLY satisfy
  it?  Quote the specific passage as evidence.
  CRITICAL TESTS for "almost" -- ALL must pass:
    (i)  Is the student's overall approach CORRECT (not just plausible)?
    (ii) Is the solution GENUINELY NEAR-COMPLETE (90%+ done)?
    (iii) Are the mistakes truly MINOR (not fundamental errors)?
    (iv) Does the student reach the CORRECT final conclusion (or come
         within one minor step of it)?
  -> ALL TESTS PASS for any criterion -> grade is "almost".  STOP.
  -> NONE satisfied          -> continue to Step 3.

STEP 3 -- PARTIAL CHECK  <- THIS STEP IS CRITICAL; DO NOT SKIP IT
  *** THIS IS THE MOST COMMONLY MISSED STEP -- BE THOROUGH ***
  Read every "(Partial)" criterion in the grading guidelines verbatim.
  For EACH criterion, perform this two-part test:
    (a) What specific mathematical object, fact, or step does this
        criterion require?  State it precisely.
    (b) Search the student's answer carefully.  Does the student
        establish that object/fact/step ANYWHERE -- even informally,
        even as a lemma, even as part of a larger failed argument?
        Quote the specific passage.
  Apply a GENEROUS standard: if the mathematical substance is present,
  credit it.  Do not penalise informal presentation or notation.
  IMPORTANT: Even if the student's overall solution is completely wrong,
  they may still satisfy a partial criterion for a specific sub-result.
  Check EVERY criterion independently of the overall solution quality.
  -> ANY criterion substantively satisfied -> grade is "partial".  STOP.
  -> NONE satisfied after checking ALL criteria -> grade is "incorrect".

================================================================
COMMON GRADING MISTAKES TO AVOID
================================================================

MISTAKE 1 -- Missing "partial" credit (MOST FREQUENT MISTAKE):
  Carefully check EVERY "(Partial)" criterion.  A student who
  establishes even one specific sub-result deserves "partial".
  Do not skip this step even if the overall solution is wrong.
  Do not require the student to have a complete or correct solution
  to earn partial credit -- partial credit is for sub-results only.
  *** Even a completely wrong overall solution can earn "partial" if
  it contains the specific sub-result named in a criterion. ***

MISTAKE 2 -- Awarding "correct" when the solution is incomplete:
  If the student proves necessary conditions but not sufficiency (or
  vice versa), or proves one case but not all cases, the grade is NOT
  "correct".  Check whether the student addresses ALL parts of the
  problem.  A well-written partial solution is still only worth 1 point.

MISTAKE 3 -- Downgrading "correct" to "incorrect" for invented gaps:
  Do not penalise a student for omitting steps that are trivially
  implied by what they wrote, or for using a valid approach that
  differs from the official solution.  If the student's argument is
  logically sound and reaches the correct conclusion, award "correct"
  even if the presentation is terse or uses different notation.
  *** ONLY downgrade from "correct" if you can name a SPECIFIC,
  CONCRETE missing step.  Vague concerns are NOT sufficient. ***

MISTAKE 4 -- Awarding "almost" for a fundamentally wrong answer:
  A long, detailed answer that uses the wrong approach or reaches the
  wrong conclusion is NOT "almost".  "Almost" requires the student to
  be on the right track.  If the student's final answer or key claim
  is wrong, default to "partial" or "incorrect".

MISTAKE 5 -- Awarding "partial" for vague similarity:
  The student must establish the SPECIFIC mathematical fact named in
  the criterion, not just something vaguely related.

MISTAKE 6 -- Awarding "correct" when an "(Almost)" criterion applies:
  Before finalising "correct", explicitly verify that NONE of the
  "(Almost)" criteria in the grading guidelines are satisfied.  If
  even one applies, the grade must be "almost", not "correct".

MISTAKE 7 -- Over-awarding "almost":
  "almost" is only ~5% of answers.  Only award "almost" when you are
  CERTAIN the answer is genuinely near-complete with the correct approach
  AND you can quote a specific "(Almost)" criterion that is satisfied.

MISTAKE 8 -- Skipping partial check because the solution looks wrong:
  Even if the student's overall approach is completely wrong, they may
  still satisfy a "(Partial)" criterion for a specific sub-result.
  ALWAYS check every partial criterion, regardless of overall quality.

MISTAKE 9 -- Awarding "correct" when the student only satisfies a "(Partial)" criterion:
  A student's answer can be long, detailed, confident, and mathematically
  sophisticated while only establishing a sub-result named in a "(Partial)"
  criterion.  Such an answer earns "partial" (1 pt), NOT "correct" (7 pts).

================================================================
PROBLEM
================================================================
{problem}

================================================================
OFFICIAL SOLUTION
================================================================
{official_solution}

================================================================
GRADING GUIDELINES
================================================================
{grading_guidelines}

================================================================
STUDENT'S ANSWER
================================================================
{student_answer}

================================================================
GRADING INSTRUCTIONS
================================================================
Step 1. Read the problem and official solution carefully to understand
        what a complete proof requires.  List every essential component
        a complete solution must contain.
Step 2. List every "(Almost)" criterion and every "(Partial)" criterion
        from the grading guidelines verbatim.
Step 3. *** MANDATORY PARTIAL SCAN -- DO THIS FIRST, BEFORE ANYTHING ELSE ***
        Go through EACH "(Partial)" criterion one by one:
        For each criterion:
          (a) State what specific fact/object/step it requires.
          (b) Search the student's answer for ANY passage that
              establishes this fact, even informally, even as part of
              a failed larger argument.
          (c) Quote the passage if found; write "not found" if absent.
        Record which partial criteria are satisfied.
        *** This step is MANDATORY even if the solution looks complete
        or completely wrong.  Do NOT skip it. ***
Step 4. Assess the student's answer against the official solution.
        Does the student's argument cover all essential components?
        Give the student the benefit of the doubt on routine steps.
Step 5. Follow the DECISION PROCEDURE above (Steps 1 -> 2 -> 3).
        For each criterion you check, quote the specific passage from
        the student's answer that supports or refutes a match.
        For "(Partial)" criteria, be generous: if the student has
        established the mathematical substance of the criterion anywhere
        in their answer (even as a lemma or intermediate step in a
        larger failed argument), that counts.
        For "(Almost)" criteria, be strict: the answer must be
        genuinely near-complete with the correct approach, not merely
        long or detailed.
Step 6. Before finalising "correct", run the CORRECT GATE:
        (a) Verify the student addresses ALL required parts of the
            problem (both directions of iff, all cases, etc.).
        (b) Explicitly check whether ANY "(Almost)" criterion applies.
            If yes, the grade is "almost", not "correct".
        (c) Ask: "Is there a SPECIFIC, IDENTIFIABLE gap?" If yes, downgrade.
            Do NOT downgrade for vague concerns or terse presentation.
            Do NOT invent gaps -- only downgrade for concrete missing steps.
        (d) Does the student's answer go BEYOND all "(Partial)" criteria?
            If the student only satisfies a "(Partial)" criterion and does
            NOT complete the full proof, the grade is "partial", NOT "correct".
            HOWEVER: Do not invent incompleteness.  If the solution is
            genuinely complete, award "correct" even if informally written.
Step 7. Before finalising "almost", verify ALL four conditions:
        (a) The student's approach is CORRECT (not just plausible).
        (b) The solution is GENUINELY NEAR-COMPLETE (not just long).
        (c) The mistakes are TRULY MINOR (not fundamental errors).
        (d) The student reaches the correct conclusion or is within
            one minor step of it.
        SPECIAL CHECK: If the "(Almost)" criterion mentions "omitted the
        case when X" or "almost complete but omitted [sub-case]", also
        verify that the student uses the SAME approach as the official
        solution.  If the student uses a different approach, the criterion
        is NOT satisfied regardless of how long or detailed the answer is.
        If ANY condition fails, choose "partial" or "incorrect" instead.
        REMEMBER: "almost" is only ~5% of answers.  Be conservative, but
        when the grading guidelines list explicit "(Almost)" criteria and
        the student clearly satisfies one, award "almost" confidently.
        Missing a genuine "almost" costs up to 5 pts vs "partial".
Step 8. State your final grade and write it in the JSON block below.

Output your final answer as a JSON object enclosed in <json> tags.
The "response" field MUST be exactly one of: "correct", "almost",
"partial", or "incorrect" (lowercase, no quotes around the value in JSON).

Example A -- answer is fully complete -> "correct":
<json>
{{{{
    "essential_components": ["Prove the invariant mod 4.", "Show all pairs satisfying the invariant are reachable."],
    "almost_criteria": ["Applied infinite descent but did not complete the final step."],
    "partial_criteria": ["Proved c >= 3.", "Identified the mod-4 invariant."],
    "partial_scan": "Criterion 1 (Proved c >= 3): Student proved c >= 3 explicitly -- SATISFIED. Criterion 2 (mod-4 invariant): Student identified and used the mod-4 invariant -- SATISFIED. Both satisfied, but solution is complete so grade is correct.",
    "step1_correct": "All essential steps are present and the student reaches the correct conclusion with no gap.",
    "correct_gate": "No Almost criterion applies. All cases covered. No specific gap found. Student goes BEYOND all Partial criteria: they not only proved c >= 3 and identified the mod-4 invariant, but also completed the full proof of reachability.",
    "reasoning": "Solution is complete and goes beyond all partial criteria; grade is correct.",
    "response": "correct"
}}}}
</json>

Example B -- answer meets an "(Almost)" criterion -> "almost":
<json>
{{{{
    "essential_components": ["Prove the invariant mod 4.", "Show all pairs satisfying the invariant are reachable."],
    "almost_criteria": ["Applied infinite descent but did not complete the final step."],
    "partial_criteria": ["Proved c >= 3.", "Identified the mod-4 invariant."],
    "partial_scan": "Criterion 1 (Proved c >= 3): Student proved c >= 3 -- SATISFIED. Criterion 2 (mod-4 invariant): Student identified the mod-4 invariant -- SATISFIED.",
    "step1_correct": "Answer is not fully complete -- missing the final step.",
    "step2_almost": "Student applied infinite descent (matches Almost criterion 1). Approach is correct and solution is near-complete. All four conditions verified. Grade: almost.",
    "reasoning": "Student's answer satisfies Almost criterion 1; grade is almost.",
    "response": "almost"
}}}}
</json>

Example C -- answer meets a "(Partial)" criterion only -> "partial":
<json>
{{{{
    "essential_components": ["Prove the invariant mod 4.", "Show all pairs satisfying the invariant are reachable."],
    "almost_criteria": ["Applied infinite descent but did not complete the final step."],
    "partial_criteria": ["Proved c >= 3.", "Identified the mod-4 invariant."],
    "partial_scan": "Criterion 1 (Proved c >= 3): Student wrote 'We show c >= 3 by contradiction...' -- SATISFIED. Criterion 2 (mod-4 invariant): Student never mentions mod 4 -- not satisfied.",
    "step1_correct": "Answer is incomplete -- does not reach the final conclusion.",
    "step2_almost": "No Almost criterion is satisfied -- the answer is not near-complete.",
    "reasoning": "Student satisfies Partial criterion 1 but no Almost criterion; grade is partial.",
    "response": "partial"
}}}}
</json>

Example D -- answer meets no criterion -> "incorrect":
<json>
{{{{
    "essential_components": ["Prove the invariant mod 4.", "Show all pairs satisfying the invariant are reachable."],
    "almost_criteria": ["Applied infinite descent but did not complete the final step."],
    "partial_criteria": ["Proved c >= 3.", "Identified the mod-4 invariant."],
    "partial_scan": "Criterion 1 (Proved c >= 3): Student never establishes this -- not satisfied. Criterion 2 (mod-4 invariant): Student never mentions mod 4 -- not satisfied. No partial criterion satisfied.",
    "step1_correct": "Answer reaches a wrong conclusion.",
    "step2_almost": "No Almost criterion is satisfied.",
    "reasoning": "No criterion is satisfied after checking all of them; grade is incorrect.",
    "response": "incorrect"
}}}}
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
    # Verification prompts                                                     #
    # ---------------------------------------------------------------------- #

    def _build_almost_check_prompt(
        self,
        problem: str,
        official_solution: str,
        grading_guidelines: str,
        student_answer: str,
        first_pass_reasoning: str,
    ) -> str:
        """Return a focused second-pass prompt to check whether "almost" applies.

        Called when the first pass returned "correct".  A common grading
        error is awarding "correct" when the answer actually satisfies an
        "(Almost)" criterion (worth 6/7 pts, not 7/7).
        """
        return f"""You are a careful mathematics competition grader performing a VERIFICATION pass.

A first grader tentatively awarded "correct" (7 points) to the student answer below.
Your job is to decide whether the grade should instead be "almost" (6 points).

IMPORTANT: Both errors are costly.
  - Awarding "correct" when "almost" applies: costs 1 point.
  - Awarding "almost" when "correct" applies: costs 1 point.
These errors are SYMMETRIC in cost.  Apply a balanced standard.

CRITICAL RULE: If ANY "(Almost)" criterion in the grading guidelines is
satisfied, the grade MUST be "almost", NOT "correct".  The first grader
may have missed this.  Your job is to check carefully.

================================================================
KEY DISTINCTION
================================================================
"correct"  (7 pts) -- Every essential logical step is present and sound.
                       The student reaches the correct final conclusion with
                       NO significant mathematical gap.  ALL cases and
                       sub-cases are addressed.

"almost"   (6 pts) -- The student's answer is ESSENTIALLY COMPLETE but has
                       minor, non-negligible gaps or mistakes.  It satisfies
                       at least one "(Almost)" criterion in the grading
                       guidelines.  The gap between "almost" and "correct"
                       is REAL and INTENTIONAL.

WHAT COUNTS AS SATISFYING AN "(Almost)" CRITERION:
  - If the criterion says "Verification contains minor mistakes only" and
    the student's proof has minor errors but is otherwise complete, that
    criterion IS satisfied.
  - If the criterion says "Solution is almost complete, but made minor
    mistakes which are not negligible" and the student's solution is
    essentially complete with minor errors, that criterion IS satisfied.
  - If the criterion says "Applied [method] but did not complete the final
    step" and the student applied that method but missed the final step,
    that criterion IS satisfied.
  - If the criterion says "Omitted the case when X" and the student's
    solution is otherwise complete but omits that specific case, that
    criterion IS satisfied (provided the student uses the same approach
    as the official solution).

================================================================
PROBLEM
================================================================
{problem}

================================================================
OFFICIAL SOLUTION
================================================================
{official_solution}

================================================================
GRADING GUIDELINES
================================================================
{grading_guidelines}

================================================================
STUDENT'S ANSWER
================================================================
{student_answer}

================================================================
FIRST GRADER'S REASONING (tentative grade: "correct")
================================================================
{first_pass_reasoning}

================================================================
YOUR TASK
================================================================
1. List every "(Almost)" criterion from the grading guidelines verbatim.
2. For EACH "(Almost)" criterion, carefully check whether the student's
   answer satisfies it.  Quote the specific passage that supports or
   refutes a match.  Be thorough -- the first grader may have missed
   a minor gap that satisfies an "(Almost)" criterion.
3. Also check: does the student address ALL required parts of the problem?
   (Both directions of iff, all cases, all sub-cases?)
4. If ANY "(Almost)" criterion is satisfied, OR if the student's solution
   has a non-negligible gap that matches an "(Almost)" criterion, the
   grade is "almost".  Otherwise, confirm "correct".

Output your final answer as a JSON object enclosed in <json> tags.

<json>
{{
    "almost_criteria_check": "<for each criterion: criterion text -> satisfied/not satisfied + evidence>",
    "all_parts_addressed": "<yes/no + explanation>",
    "reasoning": "<your conclusion>",
    "response": "correct" or "almost"
}}
</json>

Provide your verification decision below."""

    def _build_partial_check_prompt(
        self,
        problem: str,
        official_solution: str,
        grading_guidelines: str,
        student_answer: str,
        first_pass_reasoning: str,
    ) -> str:
        """Return a focused second-pass prompt to check whether "partial" applies.

        Called when the first pass returned "incorrect".  The most common
        grading error is missing partial credit when the student has established
        a specific sub-result named in a "(Partial)" criterion.

        NOTE: This prompt intentionally does NOT check for "almost" upgrades.
        Upgrading "incorrect" to "almost" is extremely risky (6/7 pts awarded
        for what was judged worthless) and empirically causes more harm than good.
        The "almost" downgrade pass handles "almost" predictions separately.
        """
        return f"""You are a careful mathematics competition grader performing a VERIFICATION pass.

A first grader tentatively awarded "incorrect" (0 points) to the student answer below.
Your job is to decide whether the grade should instead be "partial" (1 point).

IMPORTANT CONTEXT: The first grader is frequently too strict about partial credit.
Partial credit is awarded to approximately 25% of answers.  Your job is to look
for ANY evidence that the student satisfies even ONE partial criterion.

CALIBRATION NOTE: Both errors have the same cost (1 point each):
  - Awarding "partial" to a truly "incorrect" answer: costs 1 point.
  - Confirming "incorrect" for a truly "partial" answer: costs 1 point.
Apply a GENEROUS standard: upgrade to "partial" if you can find ANY passage
that engages with the mathematical substance named in the criterion.
If in doubt, lean toward "partial" since partial credit is common (~25%).

================================================================
KEY DISTINCTIONS
================================================================
"incorrect" (0 pts) -- The student's answer is wrong, fundamentally flawed,
                        or makes no meaningful progress.  No "(Partial)"
                        criterion is satisfied.

"partial"   (1 pt)  -- The student makes genuine, non-trivial mathematical
                        progress.  They EXPLICITLY and VERIFIABLY satisfy AT
                        LEAST ONE "(Partial)" criterion in the grading guidelines.

GENEROUS STANDARD FOR "partial":
  Credit the student if the mathematical substance of a "(Partial)" criterion
  is present ANYWHERE in their answer.  This includes:
  • Deriving the exact fact as part of a larger (even failed) argument.
  • Introducing the exact object/construction named in the criterion
    (e.g., if the criterion says "Considered a prime p | xy+1", and the
    student explicitly considers such a prime, that criterion is satisfied
    even if they do not complete the argument from there).
  • Correctly identifying the answer/value named in the criterion.
  • Proving the required fact as a lemma or intermediate step.
  • Stating the fact with a plausible sketch (even if incomplete).

  *** CRITICAL: Do NOT require the student to have a complete or correct
    overall solution to earn partial credit.  Even a completely wrong
    overall solution can earn "partial" if it contains the specific
    sub-result named in a criterion. ***

  The following do NOT satisfy a partial criterion:
  • Mentioning a concept without engaging with the required mathematical fact.
  • Vague similarity to a criterion without a direct match.

================================================================
PROBLEM
================================================================
{problem}

================================================================
OFFICIAL SOLUTION
================================================================
{official_solution}

================================================================
GRADING GUIDELINES
================================================================
{grading_guidelines}

================================================================
STUDENT'S ANSWER
================================================================
{student_answer}

================================================================
FIRST GRADER'S REASONING (tentative grade: "incorrect")
================================================================
{first_pass_reasoning}

================================================================
YOUR TASK
================================================================
1. List every "(Partial)" criterion from the grading guidelines verbatim.
2. For EACH "(Partial)" criterion:
   (a) State precisely what specific fact/object/step it requires.
   (b) Search the ENTIRE student answer carefully for ANY passage that
       addresses this fact, even informally or as part of a failed argument.
       Be generous: if the student has engaged with the mathematical
       substance of the criterion anywhere in their answer, credit it.
   (c) Quote the specific passage if found.
   (d) Decide: satisfied or not satisfied?
3. If ANY "(Partial)" criterion is substantively satisfied with quoted
   evidence, the grade is "partial".  Otherwise, confirm "incorrect".
   DEFAULT: If you find any passage that engages with the required
   mathematical substance, upgrade to "partial".

Output your final answer as a JSON object enclosed in <json> tags.

<json>
{{
    "partial_criteria_check": "<for each criterion: criterion text -> (a) what it requires, (b) quoted evidence or 'not found', (c) satisfied/not satisfied>",
    "reasoning": "<your conclusion>",
    "response": "incorrect" or "partial"
}}
</json>

Provide your verification decision below."""

    # ---------------------------------------------------------------------- #
    # Main entry point                                                         #
    # ---------------------------------------------------------------------- #

    def _build_almost_from_partial_check_prompt(
        self,
        problem: str,
        official_solution: str,
        grading_guidelines: str,
        student_answer: str,
        first_pass_reasoning: str,
    ) -> str:
        """Return a focused second-pass prompt to check whether "almost" applies
        when the first pass returned "partial".

        This catches cases where the student is actually near-complete but was
        initially under-graded as partial.
        """
        return f"""You are a strict mathematics competition grader performing a VERIFICATION pass.

A first grader tentatively awarded "partial" (1 point) to the student answer below.
Your job is to decide whether the grade should instead be "almost" (6 points).

================================================================
KEY DISTINCTION
================================================================
"partial"  (1 pt)  -- The student makes genuine, non-trivial mathematical
                       progress but the answer is substantially incomplete.
                       They satisfy a "(Partial)" criterion but NOT an
                       "(Almost)" criterion.

"almost"   (6 pts) -- The student's answer is ESSENTIALLY COMPLETE but has
                       minor, non-negligible gaps or mistakes.  It satisfies
                       at least one "(Almost)" criterion in the grading
                       guidelines.  The gap between "almost" and "correct"
                       is REAL and INTENTIONAL.

RULE: If ANY "(Almost)" criterion in the grading guidelines is satisfied,
the grade MUST be "almost", NOT "partial".

IMPORTANT: "almost" is only ~5% of answers.  The cost of a false "almost"
is very high (5 extra points awarded incorrectly).  Only upgrade to "almost"
if you are CERTAIN all four conditions hold:
  (A) The student's overall approach is CORRECT (not just plausible).
  (B) The solution is GENUINELY NEAR-COMPLETE (not just long or detailed).
  (C) The mistakes are TRULY MINOR (not fundamental errors).
  (D) The answer EXPLICITLY satisfies a specific "(Almost)" criterion.
If ANY condition fails, confirm "partial".

================================================================
PROBLEM
================================================================
{problem}

================================================================
OFFICIAL SOLUTION
================================================================
{official_solution}

================================================================
GRADING GUIDELINES
================================================================
{grading_guidelines}

================================================================
STUDENT'S ANSWER
================================================================
{student_answer}

================================================================
FIRST GRADER'S REASONING (tentative grade: "partial")
================================================================
{first_pass_reasoning}

================================================================
YOUR TASK
================================================================
1. List every "(Almost)" criterion from the grading guidelines verbatim.
2. For EACH "(Almost)" criterion, carefully check whether the student's
   answer satisfies it.  Quote the specific passage that supports or
   refutes a match.
3. CRITICAL TESTS for "almost" -- ALL must pass:
   (i)  Is the student's overall approach CORRECT (not just plausible)?
   (ii) Is the solution GENUINELY NEAR-COMPLETE (not just long)?
   (iii) Are the mistakes truly MINOR (not fundamental errors)?
   (iv) Does the student reach the correct conclusion or come within
        one minor step of it?
4. If ANY "(Almost)" criterion is satisfied AND ALL four tests pass,
   the grade is "almost".  Otherwise, confirm "partial".
   DEFAULT: If uncertain, confirm "partial".

Output your final answer as a JSON object enclosed in <json> tags.

<json>
{{
    "almost_criteria_check": "<for each criterion: criterion text -> satisfied/not satisfied + evidence>",
    "approach_correct": "<yes/no + explanation>",
    "near_complete": "<yes/no + explanation>",
    "mistakes_minor": "<yes/no + explanation>",
    "correct_conclusion_reached": "<yes/no + explanation>",
    "reasoning": "<your conclusion>",
    "response": "partial" or "almost"
}}
</json>

Provide your verification decision below."""

    def _build_almost_downgrade_prompt(
        self,
        problem: str,
        official_solution: str,
        grading_guidelines: str,
        student_answer: str,
        first_pass_reasoning: str,
    ) -> str:
        """Return a focused verification prompt to check whether "almost" is
        truly warranted, or whether the grade should be "partial" or "incorrect".

        Called when the first pass returned "almost".  Empirical data shows that
        "almost" predictions are wrong the vast majority of the time (0/7 correct
        in the most recent evaluation run).  This pass must be AGGRESSIVE at
        rejecting false "almost" predictions.
        """
        return f"""You are a strict mathematics competition grader performing a VERIFICATION pass.

A first grader tentatively awarded "almost" (6 points) to the student answer below.
Your job is to RIGOROUSLY verify whether this grade is correct.

CRITICAL CONTEXT: Empirical data shows that "almost" predictions are wrong the
vast majority of the time.  The first grader is likely WRONG.  Your job is to
scrutinise this "almost" prediction very carefully and downgrade it unless you
find OVERWHELMING evidence that it is correct.

COST ANALYSIS:
  - A false "almost" (awarding "almost" to a "partial" answer) costs 5 points.
  - A false "almost" (awarding "almost" to an "incorrect" answer) costs 6 points.
  - A false "partial" (downgrading "almost" to "partial") costs 5 points.
  Given that "almost" predictions are empirically unreliable, the DEFAULT
  should be to DOWNGRADE unless the evidence is overwhelming.

================================================================
STRICT REQUIREMENTS FOR "almost"
================================================================
"almost" is awarded to only ~5% of answers.  It requires ALL four conditions:
  (A) The student's overall approach is CORRECT (not just plausible).
      The student must be using the RIGHT method, not just a related one.
  (B) The solution is GENUINELY NEAR-COMPLETE (90%+ done).
      The student must have essentially completed the full proof with only
      ONE minor gap remaining.  A student who proved one key property
      (e.g., injectivity, surjectivity, a key lemma) but did NOT complete
      the full solution is NOT "almost" -- they are "partial" at best.
  (C) The mistakes are TRULY MINOR (not fundamental errors or wrong conclusions).
      A wrong final answer, a wrong key claim, or a fundamentally flawed
      approach disqualifies "almost" entirely.
  (D) The answer EXPLICITLY satisfies a specific "(Almost)" criterion.
      You must be able to quote a specific passage from the student's answer
      that directly matches the criterion.

If ANY of (A)-(D) fails, the grade is NOT "almost".

"almost"   (6 pts) -- ALL four conditions hold simultaneously.

"partial"  (1 pt)  -- The student makes genuine progress but the answer is
                       substantially incomplete or has significant errors.
                       They satisfy a "(Partial)" criterion.

"incorrect" (0 pts) -- The student's answer is wrong or makes no meaningful
                        progress.  No criterion is satisfied.

================================================================
DOWNGRADE TRIGGERS — downgrade if ANY of these apply:
================================================================
  • The student's final answer or key conclusion is WRONG.
  • The student uses a fundamentally incorrect approach or method.
  • The student proved only ONE KEY PROPERTY (e.g., injectivity) but did
    NOT complete the full solution -- this is "partial", not "almost".
  • The student's proof has a gap that requires a non-trivial new idea.
  • The student only covers some cases but not all required cases.
  • The student's approach differs from the official solution AND the
    "(Almost)" criterion is an "omitted the case" or "almost complete" type.
  • You cannot find a SPECIFIC QUOTED PASSAGE from the student's answer
    that directly satisfies the "(Almost)" criterion.

================================================================
SPECIAL GUIDANCE FOR "OMITTED THE CASE" CRITERIA
================================================================
Some "(Almost)" criteria say things like "Omitted the case when X" or
"Solution is almost complete, but omitted [specific sub-case]".

CRITICAL INTERPRETATION: These criteria apply ONLY when the student:
  1. Uses the SAME overall approach as the official solution, AND
  2. Has completed essentially ALL of the proof EXCEPT the named omission.

A student who uses a DIFFERENT approach from the official solution does NOT
satisfy an "omitted the case" criterion, even if their approach is long and
detailed.

================================================================
PROBLEM
================================================================
{problem}

================================================================
OFFICIAL SOLUTION
================================================================
{official_solution}

================================================================
GRADING GUIDELINES
================================================================
{grading_guidelines}

================================================================
STUDENT'S ANSWER
================================================================
{student_answer}

================================================================
FIRST GRADER'S REASONING (tentative grade: "almost")
================================================================
{first_pass_reasoning}

================================================================
YOUR TASK — SCRUTINISE "almost" AND DOWNGRADE UNLESS EVIDENCE IS OVERWHELMING
================================================================
1. List every "(Almost)" criterion from the grading guidelines verbatim.
2. For EACH "(Almost)" criterion, carefully check whether the student's
   answer EXPLICITLY satisfies it.  Quote the SPECIFIC passage from the
   student's answer (not from the guidelines).
   SPECIAL CHECK for "omitted the case" or "almost complete" criteria:
   - Does the student use the SAME approach as the official solution?
     If not, the criterion is NOT satisfied.
   - Is the student's solution genuinely near-complete (90%+ done)?
3. Verify ALL four conditions for "almost":
   (A) Is the student's approach CORRECT?  Quote supporting evidence.
       Is the student using the RIGHT method, or just a related one?
   (B) Is the solution NEAR-COMPLETE (90%+ done)?  Quote supporting evidence.
       Did the student prove only one key property, or essentially the whole proof?
   (C) Are the mistakes TRULY MINOR?  Identify specific mistake(s).
       Is the final answer/conclusion correct?
   (D) Does the student EXPLICITLY satisfy a specific "(Almost)" criterion?
       Quote the SPECIFIC passage from the student's answer.
4. Decision:
   - If ALL four conditions hold with OVERWHELMING evidence -> confirm "almost".
   - If ANY condition fails or the evidence is weak -> DOWNGRADE.
     Check "(Partial)" criteria: if any is satisfied -> "partial".
     If no partial criterion is satisfied -> "incorrect".
   - DEFAULT: If you are uncertain, DOWNGRADE to "partial" or "incorrect".
     The cost of a false "almost" (5-6 pts) is very high.

Output your final answer as a JSON object enclosed in <json> tags.

<json>
{{
    "almost_criteria_check": "<for each criterion: criterion text -> satisfied/not satisfied + QUOTED evidence from student answer>",
    "approach_matches_official": "<yes/no: does student use the SAME approach as official solution? + explanation>",
    "approach_correct_evidence": "<QUOTED passage proving correct approach, or explanation of why it fails>",
    "near_complete_evidence": "<yes/no: is the solution 90%+ done? QUOTED evidence or explanation of why it fails>",
    "mistakes_assessment": "<specific mistake(s) identified; are they truly minor? yes/no + reason>",
    "explicit_criterion_match": "<QUOTED passage matching the Almost criterion, or 'not found'>",
    "downgrade_triggers_present": "<list any downgrade triggers that apply, or 'none'>",
    "partial_criteria_check": "<check each (Partial) criterion: satisfied/not satisfied + evidence>",
    "reasoning": "<your conclusion>",
    "response": "almost" or "partial" or "incorrect"
}}
</json>

Provide your verification decision below."""

    def _build_correct_downgrade_prompt(
        self,
        problem: str,
        official_solution: str,
        grading_guidelines: str,
        student_answer: str,
        first_pass_reasoning: str,
    ) -> str:
        """Return a focused verification prompt to check whether "correct" should
        be downgraded to "partial" or "almost".

        Called when the first pass returned "correct".  This pass checks for
        genuine incompleteness — but does NOT default to downgrading.  The first
        grader's "correct" verdict should be respected unless there is clear,
        concrete evidence of a significant gap.

        NOTE: This is a SEPARATE pass from the almost-check pass.  The almost-check
        pass (pass 2a) focuses on correct->almost.  This pass focuses on
        correct->partial and correct->incorrect.  Both passes run when the first
        pass returns "correct".
        """
        return f"""You are a careful mathematics competition grader performing a VERIFICATION pass.

A first grader tentatively awarded "correct" (7 points) to the student answer below.
Your job is to confirm or correct this grade.

IMPORTANT BIAS NOTE: The first grader has already reviewed this answer carefully and
awarded "correct".  Do NOT downgrade unless you find CLEAR, CONCRETE evidence of a
significant mathematical gap.  Vague doubts or stylistic concerns are NOT sufficient
to downgrade.  The burden of proof is on DOWNGRADING, not on confirming "correct".

CALIBRATION: In practice, the first grader's "correct" verdict is right the majority
of the time.  Only downgrade if you can point to a SPECIFIC missing step, wrong
conclusion, or unaddressed case.  If the solution looks complete and reaches the
right answer, confirm "correct".

================================================================
GRADE DEFINITIONS
================================================================
"correct"  (7 pts) -- Every essential logical step is present and sound.
                       The student reaches the correct final conclusion with
                       no significant mathematical gap.  ALL cases and
                       sub-cases are addressed.  The solution is complete.
                       NOTE: Minor presentational gaps or omitted trivial steps
                       do NOT disqualify "correct".  Only significant mathematical
                       gaps matter.  A well-structured proof that covers all
                       required cases and reaches the correct conclusion earns
                       "correct" even if it is not written at textbook level.

"almost"   (6 pts) -- The student's answer is essentially complete but has
                       minor, non-negligible gaps.  It satisfies an "(Almost)"
                       criterion in the grading guidelines.

"partial"  (1 pt)  -- The student makes genuine progress but the answer is
                       substantially incomplete.  They satisfy a "(Partial)"
                       criterion but NOT a "correct" or "almost" standard.
                       IMPORTANT: Only downgrade to "partial" if the student
                       clearly fails to complete a major required component of
                       the proof — not just because the write-up is terse.

"incorrect" (0 pts) -- No meaningful progress.  No criterion satisfied.

================================================================
DOWNGRADE TRIGGERS — only downgrade if you find CONCRETE evidence of these:
================================================================
  • The student's final answer or key conclusion is demonstrably WRONG.
  • The student explicitly proves only one direction of a required iff,
    and the other direction is non-trivial and clearly missing.
  • The student explicitly handles only some cases, and the missing cases
    are non-trivial and clearly absent from the answer.
  • The student's proof has a gap that requires a genuinely non-trivial
    new idea to fix — not just a routine calculation or trivial step.
  • ANY "(Almost)" criterion in the grading guidelines is clearly satisfied.
  • The student only establishes a sub-result named in a "(Partial)"
    criterion and does NOT complete the full proof.

DO NOT downgrade for:
  • Terse or informal presentation of steps that are clearly implied.
  • Using a different (but valid) approach from the official solution.
  • Omitting routine calculations that follow trivially from what is written.
  • Stylistic differences or non-standard notation.
  • Vague concerns without a specific identified gap.

================================================================
PROBLEM
================================================================
{problem}

================================================================
OFFICIAL SOLUTION
================================================================
{official_solution}

================================================================
GRADING GUIDELINES
================================================================
{grading_guidelines}

================================================================
STUDENT'S ANSWER
================================================================
{student_answer}

================================================================
FIRST GRADER'S REASONING (tentative grade: "correct")
================================================================
{first_pass_reasoning}

================================================================
YOUR TASK — CONFIRM "correct" UNLESS YOU FIND CLEAR EVIDENCE OF A GAP
================================================================
1. List every essential component a complete solution must contain.
2. For EACH essential component:
   (a) State precisely what the component requires.
   (b) Search the student's answer for a passage that establishes it.
   (c) Quote the specific passage if found; write "MISSING" only if
       the component is genuinely absent (not just tersely presented).
   (d) Is the component established (even if informally)?
3. Check the "(Almost)" criteria: does ANY clearly apply?
   If yes -> grade is "almost", NOT "correct".
4. Check the "(Partial)" criteria: does the student ONLY satisfy partial
   criteria without completing the full solution?
   If yes -> grade is "partial", NOT "correct".
5. Decision:
   - DEFAULT: Confirm "correct" unless you find CLEAR, CONCRETE evidence
     of a significant gap.  The first grader's verdict deserves respect.
   - If ALL components are present (even if informally) -> confirm "correct".
   - If ANY "(Almost)" criterion clearly applies -> grade is "almost".
   - If the student clearly fails to complete a major component -> "partial".
   - If no criterion is satisfied -> grade is "incorrect".
   REMEMBER: A false downgrade from "correct" to "partial" costs 6 points.
   Only downgrade when the evidence is clear and specific.

Output your final answer as a JSON object enclosed in <json> tags.

<json>
{{
    "essential_components": "<list of all required components>",
    "components_check": "<for each component: component -> present/absent + QUOTED evidence>",
    "almost_criteria_check": "<for each Almost criterion: satisfied/not satisfied + evidence>",
    "partial_criteria_check": "<does the student only satisfy partial criteria? yes/no + evidence>",
    "final_answer_correct": "<yes/no: does the student reach the correct final conclusion?>",
    "all_cases_covered": "<yes/no: does the student address ALL required cases?>",
    "specific_gap_found": "<describe any specific, concrete gap found, or 'none'>",
    "reasoning": "<your conclusion>",
    "response": "correct" or "almost" or "partial" or "incorrect"
}}
</json>

Provide your verification decision below."""

    def _build_partial_upgrade_prompt(
        self,
        problem: str,
        official_solution: str,
        grading_guidelines: str,
        student_answer: str,
        first_pass_reasoning: str,
    ) -> str:
        """Return a focused second-pass prompt to check whether "correct" applies
        when the first pass returned "partial".

        Called when the first pass returned "partial".  Sometimes a complete
        solution is under-graded as partial when the grader misses that all
        essential steps are present.  This pass checks whether the answer is
        actually fully correct.
        """
        return f"""You are a careful mathematics competition grader performing a VERIFICATION pass.

A first grader tentatively awarded "partial" (1 point) to the student answer below.
Your job is to decide whether the grade should instead be "correct" (7 points).

CRITICAL CALIBRATION NOTE: Empirical data shows that this verification pass
produces false positives (upgrading "partial" to "correct" incorrectly) more
often than it catches genuine under-grading.  The cost of a false upgrade is
6 points per error.  Apply a VERY STRICT standard: only upgrade to "correct"
if you can verify that EVERY essential step is present and the student reaches
the correct final conclusion with NO gap.  If in doubt, confirm "partial".

================================================================
KEY DISTINCTION
================================================================
"partial"  (1 pt)  -- The student makes genuine, non-trivial mathematical
                       progress but the answer is substantially incomplete.
                       They satisfy a "(Partial)" criterion but NOT a
                       "(Almost)" or "correct" standard.

"correct"  (7 pts) -- Every essential logical step is present and sound.
                       The student reaches the correct final conclusion with
                       NO significant mathematical gap.  ALL cases and
                       sub-cases are addressed.

IMPORTANT: Only upgrade to "correct" if you are CERTAIN the solution is
complete.  A solution that is mostly right but missing a key step or case
is NOT "correct".  When in doubt, confirm "partial".

STRICT REQUIREMENTS for upgrading to "correct" -- ALL must be true:
  (A) Every essential logical step is explicitly present (not just implied).
  (B) The student reaches the CORRECT FINAL ANSWER/CONCLUSION explicitly.
  (C) ALL required cases and sub-cases are addressed.
  (D) No "(Almost)" criterion applies (which would make it "almost", not "correct").
  (E) The solution goes BEYOND all "(Partial)" criteria -- it is a complete proof.
If ANY of (A)-(E) fails, confirm "partial".

================================================================
PROBLEM
================================================================
{problem}

================================================================
OFFICIAL SOLUTION
================================================================
{official_solution}

================================================================
GRADING GUIDELINES
================================================================
{grading_guidelines}

================================================================
STUDENT'S ANSWER
================================================================
{student_answer}

================================================================
FIRST GRADER'S REASONING (tentative grade: "partial")
================================================================
{first_pass_reasoning}

================================================================
YOUR TASK
================================================================
1. List every essential component a complete solution must contain.
2. For EACH essential component, check whether the student's answer
   contains it.  Quote the specific passage that supports or refutes
   a match.
3. Check: does the student address ALL required parts of the problem?
   (Both directions of iff, all cases, all sub-cases, correct final answer?)
4. If ALL essential components are present and the student reaches the
   correct final conclusion with no significant gap -> upgrade to "correct".
   Otherwise, confirm "partial".
   DEFAULT: If uncertain, confirm "partial".

Output your final answer as a JSON object enclosed in <json> tags.

<json>
{{
    "essential_components_check": "<for each component: component -> present/absent + evidence>",
    "all_parts_addressed": "<yes/no + explanation>",
    "correct_final_answer": "<yes/no + explanation>",
    "reasoning": "<your conclusion>",
    "response": "partial" or "correct"
}}
</json>

Provide your verification decision below."""

    def _build_partial_verify_prompt(
        self,
        problem: str,
        official_solution: str,
        grading_guidelines: str,
        student_answer: str,
        first_pass_reasoning: str,
    ) -> str:
        """Return a focused verification prompt for "partial" predictions.

        Called when the first pass returned "partial".  Verifies in two
        directions:
          (a) Is "partial" truly warranted, or should it be "incorrect"?
              False partial for true-incorrect is a common error (1 pt awarded
              for a worthless answer).
          (b) Is the answer actually fully "correct"?
              Under-grading a complete solution as partial costs 6 pts.

        NOTE: Does NOT check for upgrade to "almost" -- empirical data shows
        partial->almost upgrades cause far more false positives (5/7 pt penalty
        each) than they correct.  "almost" is only ~5% of answers.
        """
        return f"""You are a careful mathematics competition grader performing a VERIFICATION pass.

A first grader tentatively awarded "partial" (1 point) to the student answer below.
Your job is to verify this grade in TWO directions:
  (A) Is "partial" truly warranted, or should it be "incorrect" (0 points)?
  (B) Is the answer actually fully "correct" (7 points)?

IMPORTANT BIAS NOTE: The first grader has already identified partial credit.
Apply a GENEROUS standard for partial credit -- partial is awarded to ~25% of
answers.  The default is to CONFIRM "partial" unless you find clear evidence
that no partial criterion is satisfied.  Only downgrade to "incorrect" if you
are CERTAIN no partial criterion is satisfied after a thorough search.
Only upgrade to "correct" if you are CERTAIN the solution is complete.
When in doubt, confirm "partial".

CALIBRATION: Downgrading "partial" to "incorrect" costs 1 point if wrong.
Downgrading "correct" to "partial" costs 6 points if wrong.  Be more
conservative about downgrading than upgrading.

================================================================
GRADE DEFINITIONS
================================================================
"correct"  (7 pts) -- Every essential logical step is present and sound.
                       The student reaches the correct final conclusion with
                       NO significant mathematical gap.  ALL cases and
                       sub-cases are addressed.

"partial"  (1 pt)  -- The student makes genuine, non-trivial mathematical
                       progress.  They EXPLICITLY and VERIFIABLY satisfy AT
                       LEAST ONE "(Partial)" criterion in the grading guidelines.
                       The mathematical SUBSTANCE of the criterion must be
                       present -- not just related terminology or a failed
                       attempt.

"incorrect" (0 pts) -- The student's answer is wrong or makes no meaningful
                        progress.  No "(Partial)" criterion is satisfied.
                        Only downgrade to "incorrect" if you are CERTAIN that
                        no partial criterion is satisfied after a thorough search.

================================================================
GENEROUS STANDARD FOR "partial"
================================================================
The student must ENGAGE WITH the specific mathematical fact named in the
criterion.  This includes:
  - Deriving the fact as part of a larger (even failed) argument.
  - Introducing the exact object/construction named in the criterion.
  - Correctly identifying the answer/value named in the criterion.
  - Stating the fact with a plausible sketch (even if incomplete).

If you find ANY passage in the student's answer that engages with the
required mathematical fact, confirm "partial".  The standard is generous.

================================================================
PROBLEM
================================================================
{problem}

================================================================
OFFICIAL SOLUTION
================================================================
{official_solution}

================================================================
GRADING GUIDELINES
================================================================
{grading_guidelines}

================================================================
STUDENT'S ANSWER
================================================================
{student_answer}

================================================================
FIRST GRADER'S REASONING (tentative grade: "partial")
================================================================
{first_pass_reasoning}

================================================================
YOUR TASK
================================================================
DIRECTION A -- Verify "partial" vs "incorrect":
1. List every "(Partial)" criterion from the grading guidelines verbatim.
2. For EACH criterion:
   (a) State what specific fact/object/step it requires.
   (b) Search the student's answer for ANY passage that engages with this.
   (c) Quote the passage if found.
   (d) Decide: satisfied or not satisfied?
   Apply a GENEROUS standard -- if the student has engaged with the
   mathematical substance of the criterion anywhere in their answer, credit it.
3. If ANY criterion is satisfied with quoted evidence -> confirm "partial".
4. Only if NO criterion is satisfied after a thorough search -> "incorrect".
   DEFAULT: Confirm "partial" when in doubt.

DIRECTION B -- Verify "partial" vs "correct":
5. List every essential component a complete solution must contain.
6. For EACH component, check whether the student's answer contains it.
7. If ALL components are present and the student reaches the correct final
   conclusion with no significant gap -> grade is "correct".
   DEFAULT: If uncertain, confirm "partial" (not "correct").

FINAL DECISION:
- DEFAULT: Confirm "partial" unless you have clear evidence to change.
- If no (Partial) criterion is satisfied after thorough search -> "incorrect".
- If all essential components are present and complete -> "correct".
- Otherwise -> confirm "partial".
- Do NOT upgrade to "almost" -- that requires a separate, much higher bar.

Output your final answer as a JSON object enclosed in <json> tags.

<json>
{{
    "partial_criteria_check": "<for each criterion: criterion text -> (a) what it requires, (b) quoted evidence or 'not found', (c) satisfied/not satisfied>",
    "any_partial_criterion_satisfied": "<yes/no>",
    "essential_components_check": "<for each component: present/absent + evidence>",
    "all_components_present": "<yes/no>",
    "reasoning": "<your conclusion>",
    "response": "partial" or "incorrect" or "correct"
}}
</json>

Provide your verification decision below."""

    def _build_third_partial_check_prompt(
        self,
        problem: str,
        official_solution: str,
        grading_guidelines: str,
        student_answer: str,
    ) -> str:
        """Return a third-pass partial check prompt with a completely different framing.

        Called when BOTH Pass 1 AND Pass 2c return "incorrect".  Unlike Pass 2c
        (which frames the task as "verify whether partial applies"), this prompt
        frames the task as a TARGETED SUB-RESULT HUNT: for each (Partial) criterion,
        the grader must independently determine whether the specific mathematical
        object/fact/step is present ANYWHERE in the student's answer.

        Key differences from Pass 2c:
          1. No reference to the previous graders' reasoning (fresh perspective).
          2. Criterion-by-criterion structure forces independent evaluation of each.
          3. Explicit instruction to search the ENTIRE answer, not just the conclusion.
          4. Requires a QUOTED PASSAGE for each satisfied criterion (strict evidence).
          5. Does NOT use the "lean toward partial" framing -- instead uses a
             balanced "quote or it doesn't count" standard to avoid false positives.

        Empirical motivation: partial recall has declined from 0.60 (gen_30) to
        0.40 (gen_38).  Two independent graders both awarding "incorrect" is still
        wrong 10 times out of 100.  A third pass with a different framing is needed.
        """
        return f"""You are an expert mathematics competition grader performing a TARGETED SUB-RESULT HUNT.

Your task is simple: for each "(Partial)" criterion listed in the grading guidelines,
determine whether the student's answer contains the specific mathematical fact,
object, or step that the criterion requires.

================================================================
CONTEXT
================================================================
Two previous graders have both awarded "incorrect" to this student answer.
Your job is NOT to re-evaluate the overall quality of the answer.
Your job is ONLY to check, criterion by criterion, whether the student has
established the specific sub-result named in each "(Partial)" criterion.

IMPORTANT: You are looking for SPECIFIC MATHEMATICAL CONTENT, not overall quality.
A student can have a completely wrong overall solution and still earn "partial"
if they establish one specific sub-result named in a criterion.

================================================================
WHAT COUNTS AS SATISFYING A "(Partial)" CRITERION
================================================================
A criterion IS satisfied if the student's answer contains ANY of the following:
  • A direct proof or derivation of the specific fact named in the criterion.
  • The specific mathematical object or construction named in the criterion
    (e.g., if the criterion says "Considered a prime p | xy+1", and the student
    explicitly considers such a prime, that criterion IS satisfied).
  • The correct value or answer named in the criterion (e.g., "Guessed L=3").
  • The specific inequality, bound, or estimate named in the criterion.
  • The specific partition, construction, or approach named in the criterion.
  • A proof of the required fact as a lemma or intermediate step.

A criterion is NOT satisfied by:
  • Vague similarity or related terminology without the specific fact.
  • A failed attempt that never reaches the specific conclusion.
  • General mathematical knowledge without the specific sub-result.

================================================================
STANDARD OF EVIDENCE
================================================================
For each criterion you claim is satisfied, you MUST quote a specific passage
from the student's answer.  If you cannot quote a passage, the criterion is
NOT satisfied.  "Quote or it doesn't count."

================================================================
PROBLEM
================================================================
{problem}

================================================================
OFFICIAL SOLUTION
================================================================
{official_solution}

================================================================
GRADING GUIDELINES
================================================================
{grading_guidelines}

================================================================
STUDENT'S ANSWER
================================================================
{student_answer}

================================================================
YOUR TASK
================================================================
For EACH "(Partial)" criterion in the grading guidelines:
  1. State the criterion verbatim.
  2. State precisely what specific mathematical fact/object/step it requires.
  3. Search the ENTIRE student answer (beginning to end) for ANY passage
     that establishes this fact.  Do not stop at the first paragraph.
  4. If found: quote the SPECIFIC PASSAGE (verbatim or close paraphrase).
     Mark as: SATISFIED -- "[quoted passage]"
  5. If not found: mark as: NOT SATISFIED -- "no passage found"

After checking ALL criteria:
  - If ANY criterion is SATISFIED (with a quoted passage) -> grade is "partial".
  - If NO criterion is satisfied -> grade is "incorrect".

Output your final answer as a JSON object enclosed in <json> tags.

<json>
{{
    "criteria_hunt": "<for each (Partial) criterion: (1) criterion verbatim, (2) what it requires, (3) quoted passage or 'no passage found', (4) SATISFIED or NOT SATISFIED>",
    "any_criterion_satisfied": "yes" or "no",
    "reasoning": "<brief conclusion>",
    "response": "partial" or "incorrect"
}}
</json>

Provide your targeted sub-result hunt below."""

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Uses a multi-pass strategy for borderline grades:
          - Pass 1: full grading prompt -> initial grade.
          - Pass 2a (if "correct"): check for "(Almost)" criteria -- the most
            common error is awarding "correct" when "almost" applies.
          - Pass 2b (if "correct", after 2a): DISABLED -- empirical data shows
            this pass produces more false downgrades (correct->incorrect, +14 pts
            lost) than it corrects (correct->partial, +6 pts saved).
          - Pass 2c (if "incorrect"): check for "(Partial)" criteria -- the
            second most common error is missing partial credit.
          - Pass 2d (if "almost"): verify "almost" is truly warranted -- the
            most COSTLY error is over-predicting "almost" (5/7 pts per false
            positive).  Empirical data shows "almost" predictions are wrong the
            vast majority of the time; this pass must be aggressive.
          - Pass 2e (if "partial"): DISABLED -- empirical data shows this pass
            produces more false downgrades (partial->incorrect, +7 pts lost)
            than it corrects.
          - Pass 2f (if still "incorrect" after Pass 2c): ENABLED -- a third
            independent partial check using a "targeted sub-result hunt" framing.
            Unlike the disabled Pass 3 (which repeated Pass 2c), Pass 2f uses a
            completely different prompt that requires QUOTED PASSAGES as evidence.
            Empirical motivation: partial recall declined from 0.60 (gen_30) to
            0.40 (gen_38); the two-pass approach is insufficient.
          - Pass 3 (if still "incorrect" after pass 2c): DISABLED -- empirical
            analysis shows it produces more false positives than it corrects.

        EMPIRICAL PRIORITY ORDER (by net pts impact, gen_38 eval):
          1. partial->incorrect (−10 pts) -- addressed by new Pass 2f
          2. partial->correct   (+24 pts) -- pass 1 occasionally over-awards
          3. correct->partial   (−18 pts) -- pass 1 occasionally under-awards
          4. incorrect->correct (+14 pts) -- pass 1 occasionally over-awards
          5. correct->incorrect (−14 pts) -- pass 1 invents gaps
          6. incorrect->partial (+11 pts) -- pass 2c occasionally over-awards

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

        # ------------------------------------------------------------------ #
        # Pass 1: full grading                                                #
        # ------------------------------------------------------------------ #
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

        # ------------------------------------------------------------------ #
        # Pass 2: targeted verification for borderline grades                 #
        # ------------------------------------------------------------------ #
        has_almost_criteria  = "(Almost)" in grading_guidelines
        has_partial_criteria = "(Partial)" in grading_guidelines

        # Track whether pass 2c (incorrect->partial check) was run, so we
        # can decide whether to run a second partial scan in pass 3.
        ran_partial_check = False

        if prediction in _VERIFY_GRADES:

            # -------------------------------------------------------------- #
            # Pass 2a: "correct" -> check for "(Almost)" criteria             #
            # -------------------------------------------------------------- #
            if prediction == "correct" and has_almost_criteria:
                # Most common error: "correct" when "almost" should apply.
                verify_prompt = self._build_almost_check_prompt(
                    problem=problem,
                    official_solution=official_solution,
                    grading_guidelines=grading_guidelines,
                    student_answer=student_answer,
                    first_pass_reasoning=assistant_text,
                )
                try:
                    v_response, v_history, _v_info = get_response_from_llm(
                        msg=verify_prompt,
                        model=self.model,
                        msg_history=[],
                    )
                    v_text = _get_assistant_text(v_history, v_response)
                    v_prediction = self._extract_prediction(v_text)
                    if v_prediction in {"correct", "almost"}:
                        if v_prediction != prediction:
                            self.log_fn(
                                f"Pass 2a almost-check changed grade: {prediction} -> {v_prediction}"
                            )
                        prediction = v_prediction
                        msg_history = msg_history + v_history
                    else:
                        self.log_fn(
                            f"Pass 2a almost-check returned unexpected grade {v_prediction!r}; "
                            f"keeping {prediction!r}."
                        )
                except Exception as exc:
                    self.log_fn(f"Pass 2a almost-check failed: {exc}; keeping original grade.")

            # -------------------------------------------------------------- #
            # Pass 2b: "correct" -> dedicated downgrade check (correct->partial)
            # DISABLED: empirical data shows this pass produces more false
            # downgrades (correct->incorrect, +14 pts lost) than it corrects
            # (correct->partial, +6 pts saved).  Net effect is negative.
            # -------------------------------------------------------------- #
            if prediction == "correct" and _ENABLE_PASS2B_CORRECT_DOWNGRADE:
                downgrade_prompt = self._build_correct_downgrade_prompt(
                    problem=problem,
                    official_solution=official_solution,
                    grading_guidelines=grading_guidelines,
                    student_answer=student_answer,
                    first_pass_reasoning=assistant_text,
                )
                try:
                    d_response, d_history, _d_info = get_response_from_llm(
                        msg=downgrade_prompt,
                        model=self.model,
                        msg_history=[],
                    )
                    d_text = _get_assistant_text(d_history, d_response)
                    d_prediction = self._extract_prediction(d_text)
                    # Accept any valid downgrade: correct -> almost/partial/incorrect
                    if d_prediction in {"correct", "almost", "partial", "incorrect"}:
                        if d_prediction != prediction:
                            self.log_fn(
                                f"Pass 2b correct-downgrade changed grade: {prediction} -> {d_prediction}"
                            )
                        prediction = d_prediction
                        msg_history = msg_history + d_history
                    else:
                        self.log_fn(
                            f"Pass 2b correct-downgrade returned unexpected grade {d_prediction!r}; "
                            f"keeping {prediction!r}."
                        )
                except Exception as exc:
                    self.log_fn(f"Pass 2b correct-downgrade failed: {exc}; keeping original grade.")

            # -------------------------------------------------------------- #
            # Pass 2c: "incorrect" -> check for "(Partial)" criteria          #
            # -------------------------------------------------------------- #
            elif prediction == "incorrect" and (has_partial_criteria or has_almost_criteria):
                # Second most common error: "incorrect" when "partial" or "almost"
                # should apply.  Check for partial credit first; the partial check
                # prompt also catches near-complete answers.
                verify_prompt = self._build_partial_check_prompt(
                    problem=problem,
                    official_solution=official_solution,
                    grading_guidelines=grading_guidelines,
                    student_answer=student_answer,
                    first_pass_reasoning=assistant_text,
                )
                ran_partial_check = True
                try:
                    v_response, v_history, _v_info = get_response_from_llm(
                        msg=verify_prompt,
                        model=self.model,
                        msg_history=[],
                    )
                    v_text = _get_assistant_text(v_history, v_response)
                    v_prediction = self._extract_prediction(v_text)
                    # "incorrect" can be confirmed or upgraded to "partial" only.
                    # Upgrading to "almost" is too risky (6/7 pts for a first-pass
                    # "incorrect" answer) and empirically causes more harm than good.
                    if v_prediction in {"incorrect", "partial"}:
                        if v_prediction != prediction:
                            self.log_fn(
                                f"Pass 2c partial-check changed grade: {prediction} -> {v_prediction}"
                            )
                        prediction = v_prediction
                        msg_history = msg_history + v_history
                    else:
                        self.log_fn(
                            f"Pass 2c partial-check returned unexpected grade {v_prediction!r} "
                            f"for first-pass {prediction!r}; keeping original."
                        )
                except Exception as exc:
                    self.log_fn(f"Pass 2c partial-check failed: {exc}; keeping original grade.")

            # -------------------------------------------------------------- #
            # Pass 2d: "almost" -> verify "almost" is truly warranted         #
            # -------------------------------------------------------------- #
            elif prediction == "almost":
                # Most COSTLY error: over-predicting "almost" (5-6 pts per false
                # positive).  Always verify "almost" predictions.
                verify_prompt = self._build_almost_downgrade_prompt(
                    problem=problem,
                    official_solution=official_solution,
                    grading_guidelines=grading_guidelines,
                    student_answer=student_answer,
                    first_pass_reasoning=assistant_text,
                )
                try:
                    v_response, v_history, _v_info = get_response_from_llm(
                        msg=verify_prompt,
                        model=self.model,
                        msg_history=[],
                    )
                    v_text = _get_assistant_text(v_history, v_response)
                    v_prediction = self._extract_prediction(v_text)
                    # "almost" can be confirmed or downgraded to partial/incorrect
                    if v_prediction in {"almost", "partial", "incorrect"}:
                        if v_prediction != prediction:
                            self.log_fn(
                                f"Pass 2d almost-downgrade changed grade: {prediction} -> {v_prediction}"
                            )
                        prediction = v_prediction
                        msg_history = msg_history + v_history
                    else:
                        self.log_fn(
                            f"Pass 2d almost-downgrade returned unexpected grade {v_prediction!r} "
                            f"for first-pass {prediction!r}; keeping original."
                        )
                except Exception as exc:
                    self.log_fn(f"Pass 2d almost-downgrade failed: {exc}; keeping original grade.")

            # -------------------------------------------------------------- #
            # Pass 2e: "partial" -> verify in both directions                 #
            # DISABLED: empirical data shows this pass produces more false    #
            # downgrades (partial->incorrect, +7 pts lost) than it corrects.  #
            # -------------------------------------------------------------- #
            elif prediction == "partial" and _ENABLE_PASS2E_PARTIAL_VERIFY:
                # Verify "partial" predictions in both directions:
                # (a) Check whether "partial" is truly warranted vs "incorrect"
                #     (false partial for true-incorrect is a common error).
                # (b) Check whether the answer is actually fully "correct"
                #     (under-grading a complete solution as partial is costly).
                # NOTE: No upgrade to "almost" -- empirical data shows that
                # partial->almost upgrades cause far more false positives than
                # they correct.  "almost" is only ~5% of answers.
                verify_prompt = self._build_partial_verify_prompt(
                    problem=problem,
                    official_solution=official_solution,
                    grading_guidelines=grading_guidelines,
                    student_answer=student_answer,
                    first_pass_reasoning=assistant_text,
                )
                try:
                    v_response, v_history, _v_info = get_response_from_llm(
                        msg=verify_prompt,
                        model=self.model,
                        msg_history=[],
                    )
                    v_text = _get_assistant_text(v_history, v_response)
                    v_prediction = self._extract_prediction(v_text)
                    # "partial" can be confirmed, downgraded to "incorrect", or
                    # upgraded to "correct" -- but NOT upgraded to "almost"
                    # (too risky: 5/7 pt penalty per false positive).
                    if v_prediction in {"partial", "incorrect", "correct"}:
                        if v_prediction != prediction:
                            self.log_fn(
                                f"Pass 2e partial-verify changed grade: {prediction} -> {v_prediction}"
                            )
                        prediction = v_prediction
                        msg_history = msg_history + v_history
                    else:
                        self.log_fn(
                            f"Pass 2e partial-verify returned unexpected grade {v_prediction!r} "
                            f"for first-pass {prediction!r}; keeping original."
                        )
                except Exception as exc:
                    self.log_fn(f"Pass 2e partial-verify failed: {exc}; keeping original grade.")

        # ------------------------------------------------------------------ #
        # Pass 2f: third independent partial check (targeted sub-result hunt) #
        # ------------------------------------------------------------------ #
        # Run when BOTH Pass 1 AND Pass 2c return "incorrect" AND the problem
        # has (Partial) criteria.  Uses a completely different prompt framing
        # (criterion-by-criterion sub-result hunt with mandatory quoted evidence)
        # to catch partial credit that both previous passes missed.
        #
        # Unlike the disabled Pass 3 (which repeated Pass 2c verbatim), Pass 2f:
        #   - Uses no reference to previous graders' reasoning (fresh perspective).
        #   - Requires QUOTED PASSAGES as evidence (strict but thorough).
        #   - Frames the task as a targeted hunt, not a holistic re-evaluation.
        #
        # Empirical motivation: partial recall declined from 0.60 (gen_30) to
        # 0.40 (gen_38) -- a 20% drop.  10 cases of partial->incorrect in gen_38
        # survived both Pass 1 and Pass 2c.  A third pass with a different framing
        # is needed to recover this lost recall.
        if (
            _ENABLE_PASS2F_THIRD_PARTIAL_SCAN
            and prediction == "incorrect"
            and ran_partial_check
            and has_partial_criteria
        ):
            try:
                p2f_prompt = self._build_third_partial_check_prompt(
                    problem=problem,
                    official_solution=official_solution,
                    grading_guidelines=grading_guidelines,
                    student_answer=student_answer,
                )
                p2f_response, p2f_history, _p2f_info = get_response_from_llm(
                    msg=p2f_prompt,
                    model=self.model,
                    msg_history=[],
                )
                p2f_text = _get_assistant_text(p2f_history, p2f_response)
                p2f_prediction = self._extract_prediction(p2f_text)

                # Only accept "partial" from Pass 2f; do not accept "almost" or
                # "correct" (too risky for a third-pass upgrade from "incorrect").
                if p2f_prediction == "partial":
                    self.log_fn(
                        "Pass 2f sub-result hunt upgraded grade: incorrect -> partial"
                    )
                    prediction = "partial"
                    msg_history = msg_history + p2f_history
                else:
                    self.log_fn(
                        f"Pass 2f sub-result hunt confirmed: incorrect (no partial criterion found)."
                    )
            except Exception as exc:
                self.log_fn(f"Pass 2f sub-result hunt failed: {exc}; keeping grade.")

        # ------------------------------------------------------------------ #
        # Pass 3: second partial scan for persistent "incorrect" predictions  #
        # ------------------------------------------------------------------ #
        # NOTE: Pass 3 is DISABLED (_ENABLE_PASS3_SECOND_PARTIAL_SCAN = False).
        # Empirical analysis shows that after two independent graders both award
        # "incorrect", a third pass produces more false positives (incorrect->partial)
        # than it corrects.  The net effect on accuracy is negative.  The two-pass
        # approach (Pass 1 + Pass 2c) is sufficient for partial credit detection.
        # Pass 2f (above) supersedes Pass 3 with a better-designed prompt.
        if (
            _ENABLE_PASS3_SECOND_PARTIAL_SCAN
            and prediction == "incorrect"
            and ran_partial_check
            and has_partial_criteria
        ):
            try:
                p3_prompt = self._build_partial_check_prompt(
                    problem=problem,
                    official_solution=official_solution,
                    grading_guidelines=grading_guidelines,
                    student_answer=student_answer,
                    first_pass_reasoning=(
                        "Two independent graders have both tentatively awarded "
                        "'incorrect'.  Please perform a fresh, independent check "
                        "of every (Partial) criterion.  Be strict -- only upgrade "
                        "if you can quote a specific passage establishing the fact."
                    ),
                )
                p3_response, p3_history, _p3_info = get_response_from_llm(
                    msg=p3_prompt,
                    model=self.model,
                    msg_history=[],
                )
                p3_text = _get_assistant_text(p3_history, p3_response)
                p3_prediction = self._extract_prediction(p3_text)

                if p3_prediction == "partial":
                    self.log_fn(
                        "Pass 3 partial scan upgraded grade: incorrect -> partial"
                    )
                    prediction = "partial"
                    msg_history = msg_history + p3_history
                # Only accept "partial" from pass 3; do not accept "almost" or
                # "correct" (too risky for a third-pass upgrade from "incorrect").
            except Exception as exc:
                self.log_fn(f"Pass 3 partial scan failed: {exc}; keeping grade.")

        return prediction, msg_history
