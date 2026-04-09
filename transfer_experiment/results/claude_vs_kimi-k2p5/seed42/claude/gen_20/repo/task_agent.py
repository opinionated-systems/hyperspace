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
#                (over-predicting "almost" is very costly: 6/7 pts penalty per error).
# "partial"   -> verify whether "partial" might actually be "correct"
#                (catches cases where the first pass under-grades a complete solution).
_VERIFY_GRADES = {"incorrect", "correct", "almost", "partial"}

# Expected base rates for calibration (from empirical data):
# correct ~35%, incorrect ~35%, partial ~25%, almost ~5%
_BASE_RATES = {"correct": 0.35, "almost": 0.05, "partial": 0.25, "incorrect": 0.35}


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
  "almost" is VERY RARE -- only ~1 in 20 answers.  Be extremely conservative.
  If you find yourself assigning "incorrect" to more than half the
  answers you grade, you are likely being too strict on "partial".

CRITICAL WARNING ABOUT "almost" -- READ CAREFULLY:
  "almost" (6 pts) is the MOST DANGEROUS grade to award incorrectly.
  Awarding "almost" to an answer that is actually "incorrect" (0 pts) or
  "partial" (1 pt) costs 6 or 5 points respectively -- a catastrophic error.
  By contrast, awarding "correct" (7 pts) to an "almost" answer costs only 1 pt.
  THEREFORE: When in doubt between "almost" and a lower grade, ALWAYS choose
  the lower grade.  Only award "almost" when you are CERTAIN the answer is
  genuinely near-complete with the correct approach and only minor gaps.

  STRICT REQUIREMENTS for "almost" -- ALL must be true:
    (A) The student's overall approach is CORRECT (not just plausible).
    (B) The solution is GENUINELY NEAR-COMPLETE (not just long or detailed).
    (C) The mistakes are TRULY MINOR (not fundamental errors or wrong conclusions).
    (D) The answer EXPLICITLY satisfies a specific "(Almost)" criterion in the
        grading guidelines -- quote the exact passage as evidence.
  If ANY of (A)-(D) fails, do NOT award "almost".

The grading guidelines list "(Almost)" and "(Partial)" criteria.
Their intended meanings are:

  "(Almost)" criterion -- the student's answer is essentially complete but
    has minor, non-negligible gaps or mistakes that prevent a full 7 points.
    An answer satisfying an "(Almost)" criterion earns "almost" (6 pts),
    NOT "correct" (7 pts).  The gap between "almost" and "correct" is real.

  "(Partial)" criterion -- the student made genuine, non-trivial progress on
    a specific sub-task but did not come close to a full solution.
    An answer satisfying a "(Partial)" criterion (but no "(Almost)" criterion)
    earns "partial" (1 pt).

================================================================
GRADE DEFINITIONS
================================================================

"correct"  (7 pts)
  Every essential logical step is present and sound AND the student
  reaches the correct final conclusion with no significant mathematical
  gap.  Minor presentational or notational differences are acceptable,
  but there must be no mathematical gap of any significance.
  RULE: Award "correct" ONLY when the solution is genuinely complete.
  Do NOT award "correct" if the student satisfies an "(Almost)"
    criterion -- that earns "almost", not "correct".
  Do NOT award "correct" if the student only proves part of what is
    required (e.g., proves necessary conditions but not sufficiency,
    or proves one direction of an iff but not the other, or handles
    some cases but not all).

"almost"  (6 pts)
  The student's answer is essentially complete but has minor,
  non-negligible gaps or mistakes -- it satisfies at least one
  "(Almost)" criterion in the grading guidelines.
  RULE: If the student's answer satisfies ANY "(Almost)" criterion,
    the grade MUST be "almost".  STOP -- do not check further.
  Do NOT conflate "almost" with "correct": the gap is real and
    intentional.  An "(Almost)" answer is not fully correct.
  Do NOT award "almost" unless the answer is genuinely near-complete.
    A long but fundamentally flawed answer that merely mentions a related
    idea does NOT satisfy an "(Almost)" criterion.
  CRITICAL: An answer that reaches a WRONG final conclusion, or uses
    a fundamentally incorrect approach, is NOT "almost" -- it is
    "incorrect" or at most "partial".  "Almost" requires the student
    to be on the RIGHT track with only minor gaps remaining.
  When the "(Almost)" criterion says "Solution is almost complete,
    but made minor mistakes which are not negligible", this means the
    student must have a CORRECT overall approach and a NEARLY COMPLETE
    proof -- not just a long attempt.  Verify the approach is correct
    before awarding "almost".
  REMEMBER: "almost" is only ~5% of answers.  If you are considering
    "almost", ask yourself: "Is this answer truly 85%+ complete with
    the right approach?"  If not, choose "partial" or "incorrect".

"partial"  (1 pt)
  The student makes genuine, non-trivial mathematical progress but the
  answer is substantially incomplete or contains significant errors.
  Award "partial" when the student's answer EXPLICITLY and VERIFIABLY
  satisfies AT LEAST ONE "(Partial)" criterion listed in the grading
  guidelines.
  GENEROUS STANDARD: Credit the student if the mathematical substance
    of a "(Partial)" criterion is present anywhere in their answer --
    even if the write-up is informal, uses different notation, or the
    criterion is reached as a by-product of a larger (failed) argument.
  A long, detailed attempt that contains the key sub-result of a
    "(Partial)" criterion DOES earn "partial", even if the overall
    solution is wrong.
  If the student introduces the exact object, substitution, or
    construction named in a "(Partial)" criterion (e.g. "considered a
    prime p | xy+1"), that criterion is satisfied even if they do not
    complete the argument from there.
  Do NOT award "partial" based on vague similarity to a criterion,
    general mathematical knowledge, or a long but ultimately flawed
    argument that never reaches the criterion's specific conclusion.
  An answer that satisfies an "(Almost)" criterion is "almost",
    not "partial".
  IMPORTANT: A student who merely MENTIONS a concept without actually
    establishing the required mathematical fact does NOT satisfy a
    "(Partial)" criterion.  The criterion requires the specific
    mathematical substance to be present, not just the terminology.

"incorrect"  (0 pts)
  The student's answer is wrong, fundamentally flawed, or makes no
  meaningful progress toward the solution.  This includes:
  - Arriving at a clearly wrong conclusion.
  - Circular reasoning or merely restating the problem.
  - A plausible-sounding narrative with no real mathematical substance.
  - A valid approach with a critical error so early that no useful
    progress is made.
  - An answer that does NOT explicitly satisfy any "(Partial)" or
    "(Almost)" criterion in the grading guidelines.
  A long, detailed, or confident-sounding answer is NOT evidence of
    partial credit.  Only explicit criterion matches count.
  However, do NOT default to "incorrect" without carefully checking
    every "(Partial)" criterion.  Partial credit is common (~25% of
    answers); missing it is a frequent grading error.

================================================================
DECISION PROCEDURE  (follow steps in order; stop at first match)
================================================================

STEP 1 -- CORRECT CHECK
  a. Are ALL essential logical steps present and sound?
  b. Does the student reach the correct final conclusion with no
     significant mathematical gap?
  c. Does the student prove BOTH directions if the problem requires it
     (e.g., necessary AND sufficient conditions, all cases)?
  -> ALL YES -> BEFORE marking "correct", run the CORRECT GATE below.
  -> OTHERWISE -> continue to Step 2.

  CORRECT GATE (mandatory before awarding "correct"):
    Ask yourself: "Could a strict grader find any gap here?"
    - Does the student address EVERY case and sub-case?
    - Is the final answer/conclusion explicitly stated and right?
    - Does ANY "(Almost)" criterion in the guidelines apply?
      If yes -> grade is "almost", NOT "correct".
    Only if ALL checks pass -> grade is "correct".

STEP 2 -- ALMOST CHECK
  Read every "(Almost)" criterion in the grading guidelines verbatim.
  For each criterion, ask: does the student's answer EXPLICITLY satisfy
  it?  Quote the specific passage as evidence.
  CRITICAL TESTS for "almost" -- ALL must pass:
    (i)  Is the student's overall approach CORRECT (not just plausible)?
    (ii) Is the solution GENUINELY NEAR-COMPLETE (not just long)?
    (iii) Are the mistakes truly MINOR (not fundamental errors)?
    (iv) Does the student reach the CORRECT final conclusion (or come
         within one minor step of it)?
  If the criterion says "minor mistakes only" or "almost complete",
  verify that the student has the right strategy and is close to done.
  An answer that is long and detailed but uses a WRONG approach or
  reaches a WRONG conclusion does NOT satisfy an "almost" criterion.
  DEFAULT RULE: If you are uncertain whether "almost" applies, choose
  "partial" instead.  The cost of a false "almost" is very high.
  -> ALL TESTS PASS for any criterion -> grade is "almost".  STOP.
  -> NONE satisfied          -> continue to Step 3.

STEP 3 -- PARTIAL CHECK  <- THIS STEP IS CRITICAL; DO NOT SKIP IT
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
  -> ANY criterion substantively satisfied -> grade is "partial".  STOP.
  -> NONE satisfied after checking ALL criteria -> grade is "incorrect".

================================================================
COMMON GRADING MISTAKES TO AVOID
================================================================

MISTAKE 1 -- Awarding "correct" when the solution is incomplete:
  If the student proves necessary conditions but not sufficiency (or
  vice versa), or proves one case but not all cases, the grade is NOT
  "correct".  Check whether the student addresses ALL parts of the
  problem.

MISTAKE 2 -- Awarding "almost" for a fundamentally wrong answer:
  A long, detailed answer that uses the wrong approach or reaches the
  wrong conclusion is NOT "almost".  "Almost" requires the student to
  be on the right track.  If the student's final answer or key claim
  is wrong, default to "partial" or "incorrect".

MISTAKE 3 -- Missing "partial" credit:
  Carefully check EVERY "(Partial)" criterion.  A student who
  establishes even one specific sub-result deserves "partial".
  Do not skip this step even if the overall solution is wrong.

MISTAKE 4 -- Awarding "partial" for vague similarity:
  The student must establish the SPECIFIC mathematical fact named in
  the criterion, not just something vaguely related.

MISTAKE 5 -- Awarding "correct" when an "(Almost)" criterion applies:
  Before finalising "correct", explicitly verify that NONE of the
  "(Almost)" criteria in the grading guidelines are satisfied.  If
  even one applies, the grade must be "almost", not "correct".

MISTAKE 6 -- Awarding "correct" for a plausible but gapped proof:
  A confident, well-written answer that skips a key step or handles
  only some cases is NOT "correct".  The standard is strict: every
  essential step must be present and sound.
  HOWEVER: Do not invent gaps that are not there.  If a step is
  standard or follows immediately from what the student wrote, do not
  penalise the student for not spelling it out in full detail.
  A well-structured proof that covers all the required cases and
  reaches the correct conclusion should be awarded "correct" even if
  it is not written at the level of a textbook proof.

MISTAKE 7 -- Over-awarding "almost" (THE MOST COSTLY MISTAKE):
  "almost" is only ~5% of answers.  Awarding "almost" to an answer
  that is actually "incorrect" or "partial" is a catastrophic error
  (costs 5-6 points).  When in doubt, choose "partial" over "almost".
  Only award "almost" when you are CERTAIN the answer is genuinely
  near-complete with the correct approach.

MISTAKE 8 -- Downgrading "correct" to "incorrect" for invented gaps:
  Do not penalise a student for omitting steps that are trivially
  implied by what they wrote, or for using a valid approach that
  differs from the official solution.  If the student's argument is
  logically sound and reaches the correct conclusion, award "correct"
  even if the presentation is terse or uses different notation.

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
Step 3. Follow the DECISION PROCEDURE above (Steps 1 -> 2 -> 3).
        For each criterion you check, quote the specific passage from
        the student's answer that supports or refutes a match.
        For "(Partial)" criteria, be generous: if the student has
        established the mathematical substance of the criterion anywhere
        in their answer (even as a lemma or intermediate step in a
        larger failed argument), that counts.
        For "(Almost)" criteria, be strict: the answer must be
        genuinely near-complete with the correct approach, not merely
        long or detailed.
Step 4. Before finalising "correct", run the CORRECT GATE:
        (a) Verify the student addresses ALL required parts of the
            problem (both directions of iff, all cases, etc.).
        (b) Explicitly check whether ANY "(Almost)" criterion applies.
            If yes, the grade is "almost", not "correct".
        (c) Ask: "Would a strict grader find any gap?" If yes, downgrade.
Step 5. Before finalising "almost", verify ALL four conditions:
        (a) The student's approach is CORRECT (not just plausible).
        (b) The solution is GENUINELY NEAR-COMPLETE (not just long).
        (c) The mistakes are TRULY MINOR (not fundamental errors).
        (d) The student reaches the correct conclusion or is within
            one minor step of it.
        If ANY condition fails, choose "partial" or "incorrect" instead.
        REMEMBER: "almost" is only ~5% of answers.  Be very conservative.
Step 6. Before finalising "partial", verify the student has established
        the SPECIFIC mathematical substance of the criterion -- not just
        mentioned related terminology.
Step 7. State your final grade and write it in the JSON block below.

Output your final answer as a JSON object enclosed in <json> tags.
The "response" field MUST be exactly one of: "correct", "almost",
"partial", or "incorrect" (lowercase, no quotes around the value in JSON).

Example A -- answer is fully complete -> "correct":
<json>
{{{{
    "essential_components": ["Prove the invariant mod 4.", "Show all pairs satisfying the invariant are reachable."],
    "almost_criteria": ["Applied infinite descent but did not complete the final step."],
    "partial_criteria": ["Proved c >= 3.", "Identified the mod-4 invariant."],
    "step1_correct": "All essential steps are present and the student reaches the correct conclusion with no gap.",
    "correct_gate": "No Almost criterion applies. All cases covered. No gap found.",
    "reasoning": "Solution is complete; grade is correct.",
    "response": "correct"
}}}}
</json>

Example B -- answer meets an "(Almost)" criterion -> "almost":
<json>
{{{{
    "essential_components": ["Prove the invariant mod 4.", "Show all pairs satisfying the invariant are reachable."],
    "almost_criteria": ["Applied infinite descent but did not complete the final step."],
    "partial_criteria": ["Proved c >= 3.", "Identified the mod-4 invariant."],
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
    "step1_correct": "Answer is incomplete -- does not reach the final conclusion.",
    "step2_almost": "No Almost criterion is satisfied -- the answer is not near-complete.",
    "step3_partial_criterion_1": "Criterion: 'Proved c >= 3.' Student wrote: 'We show c >= 3 by contradiction...' -- criterion satisfied.",
    "step3_partial_criterion_2": "Criterion: 'Identified the mod-4 invariant.' Student never mentions mod 4 -- not satisfied.",
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
    "step1_correct": "Answer reaches a wrong conclusion.",
    "step2_almost": "No Almost criterion is satisfied.",
    "step3_partial_criterion_1": "Criterion: 'Proved c >= 3.' Student never establishes this -- not satisfied.",
    "step3_partial_criterion_2": "Criterion: 'Identified the mod-4 invariant.' Student never mentions mod 4 -- not satisfied.",
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

        Called when the first pass returned "correct".  The most common grading
        error is awarding "correct" when the answer actually satisfies an
        "(Almost)" criterion (worth 6/7 pts, not 7/7).
        """
        return f"""You are a strict mathematics competition grader performing a VERIFICATION pass.

A first grader tentatively awarded "correct" (7 points) to the student answer below.
Your job is to decide whether the grade should instead be "almost" (6 points).

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

RULE: If ANY "(Almost)" criterion in the grading guidelines is satisfied,
the grade MUST be "almost", NOT "correct".

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
   refutes a match.
3. Also check: does the student address ALL required parts of the problem?
   (Both directions of iff, all cases, all sub-cases?)
4. If ANY "(Almost)" criterion is satisfied, OR if the student's solution
   has a non-negligible gap, the grade is "almost".
   Otherwise, confirm "correct".

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
        """Return a focused second-pass prompt to check whether "partial" or
        "almost" applies.

        Called when the first pass returned "incorrect".  The most common
        grading error is missing partial credit when the student has established
        a specific sub-result named in a "(Partial)" criterion.  Also checks
        for "almost" in case the answer is near-complete.
        """
        return f"""You are a strict mathematics competition grader performing a VERIFICATION pass.

A first grader tentatively awarded "incorrect" (0 points) to the student answer below.
Your job is to decide whether the grade should instead be "partial" (1 point) or
possibly "almost" (6 points) if the answer is near-complete.

================================================================
KEY DISTINCTIONS
================================================================
"incorrect" (0 pts) -- The student's answer is wrong, fundamentally flawed,
                        or makes no meaningful progress.  No "(Partial)" or
                        "(Almost)" criterion is satisfied.

"partial"   (1 pt)  -- The student makes genuine, non-trivial mathematical
                        progress.  They EXPLICITLY and VERIFIABLY satisfy AT
                        LEAST ONE "(Partial)" criterion in the grading guidelines.

"almost"    (6 pts) -- The student's answer is ESSENTIALLY COMPLETE but has
                        minor, non-negligible gaps.  ALL four conditions hold:
                        (A) The student's overall approach is CORRECT.
                        (B) The solution is GENUINELY NEAR-COMPLETE (>85% done).
                        (C) The mistakes are TRULY MINOR.
                        (D) The answer EXPLICITLY satisfies an "(Almost)" criterion.
                        NOTE: "almost" is only ~5% of answers -- be conservative.

STANDARD FOR "partial":
  Credit the student if the mathematical SUBSTANCE of a "(Partial)" criterion
  is present ANYWHERE in their answer -- even if the write-up is informal,
  uses different notation, or the criterion is reached as a by-product of a
  larger (failed) argument.
  If the student introduces the exact object, substitution, or construction
  named in a criterion, that criterion is satisfied even if they do not
  complete the argument from there.

IMPORTANT LIMITS on "partial":
  • The student must ESTABLISH the specific mathematical fact named in the
    criterion -- merely mentioning a concept or using related terminology
    does NOT satisfy the criterion.
  • A long, detailed, confident-sounding answer that never actually proves
    the specific sub-result named in the criterion does NOT earn "partial".
  • If the student's argument for the criterion is circular, hand-wavy, or
    relies on an unproven claim, the criterion is NOT satisfied.
  • Quote the SPECIFIC passage from the student's answer that establishes
    the criterion.  If you cannot find a specific quoted passage, the
    criterion is NOT satisfied.

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
1. First, check "(Almost)" criteria: is the answer near-complete with the
   correct approach and only minor gaps?  If yes and all four conditions
   hold, the grade is "almost".
2. If not "almost", list every "(Partial)" criterion from the grading
   guidelines verbatim.
3. For EACH "(Partial)" criterion, carefully check whether the student's
   answer satisfies it.  Quote the SPECIFIC passage from the student's
   answer that establishes the required mathematical fact.  If you cannot
   find a specific quoted passage, the criterion is NOT satisfied.
4. If ANY "(Partial)" criterion is substantively satisfied with quoted
   evidence, the grade is "partial".  Otherwise, confirm "incorrect".

Output your final answer as a JSON object enclosed in <json> tags.

<json>
{{
    "almost_check": "<check if answer is near-complete with correct approach; if yes, quote evidence for all 4 conditions>",
    "partial_criteria_check": "<for each criterion: criterion text -> satisfied/not satisfied + QUOTED evidence from student answer>",
    "reasoning": "<your conclusion>",
    "response": "incorrect" or "partial" or "almost"
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

        Called when the first pass returned "almost".  Over-predicting "almost"
        is the most costly grading error (5-6 points per false positive).
        """
        return f"""You are a SKEPTICAL mathematics competition grader performing a VERIFICATION pass.

A first grader tentatively awarded "almost" (6 points) to the student answer below.
Your job is to CHALLENGE this grade.  Assume the first grader was too generous.
Downgrade to "partial" or "incorrect" unless you can PROVE all four conditions hold.

================================================================
CRITICAL CONTEXT — READ BEFORE ANYTHING ELSE
================================================================
"almost" is awarded to only ~5% of answers.  It is the MOST DANGEROUS grade
to award incorrectly: a false "almost" costs 5-6 points per error.

THE BURDEN OF PROOF IS ON "almost":
  You must find CONCRETE, QUOTED EVIDENCE for EACH of the four conditions.
  If you cannot quote specific text from the student's answer that proves
  a condition, that condition FAILS and the grade must be downgraded.

"almost"   (6 pts) -- ALL four conditions must hold simultaneously:
                       (A) The student's overall approach is CORRECT — not
                           just plausible or partially right.  The student
                           must be using the RIGHT method, not just a method
                           that happens to produce some correct sub-results.
                       (B) The solution is GENUINELY NEAR-COMPLETE — the
                           student has completed the vast majority of the
                           proof.  A long attempt that covers only 50-70%
                           of the required argument is NOT near-complete.
                       (C) The mistakes are TRULY MINOR — a single missing
                           key step, a wrong final answer, or a fundamental
                           logical gap is NOT minor.  Minor means: a small
                           computational slip, a missing edge case in an
                           otherwise complete argument, or a trivially
                           fixable notation issue.
                       (D) The answer EXPLICITLY satisfies a specific
                           "(Almost)" criterion — quote the exact passage
                           from the student's answer as evidence.

"partial"  (1 pt)  -- The student makes genuine progress but the answer is
                       substantially incomplete or has significant errors.
                       They satisfy a "(Partial)" criterion.

"incorrect" (0 pts) -- The student's answer is wrong or makes no meaningful
                        progress.  No criterion is satisfied.

DOWNGRADE TRIGGERS — if ANY of these apply, do NOT award "almost":
  • The student's final answer or key conclusion is WRONG.
  • The student uses a fundamentally incorrect approach or method.
  • The student's proof has a gap that requires a non-trivial new idea to fix.
  • The student only covers some cases but not all required cases.
  • The student's argument is long but circular or hand-wavy at a key step.
  • You cannot quote specific text proving the approach is correct.
  • You cannot quote specific text proving the solution is near-complete.

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
YOUR TASK — BE SKEPTICAL; DOWNGRADE UNLESS ALL CONDITIONS ARE MET
================================================================
1. List every "(Almost)" criterion from the grading guidelines verbatim.
2. For EACH "(Almost)" criterion, carefully check whether the student's
   answer EXPLICITLY satisfies it.  Quote the SPECIFIC passage from the
   student's answer (not from the guidelines).  If you cannot find a
   specific quoted passage, the criterion is NOT satisfied.
3. Verify ALL four conditions for "almost" with QUOTED EVIDENCE:
   (A) Quote the passage showing the student's approach is CORRECT.
       If the student's final answer or key claim is wrong, write "FAILS".
   (B) Quote the passage showing the solution is NEAR-COMPLETE (>85% done).
       If the student is missing major steps, write "FAILS".
   (C) Identify the specific mistake(s).  Are they truly minor?
       If any mistake requires a non-trivial fix, write "FAILS".
   (D) Quote the exact passage satisfying the "(Almost)" criterion.
       If no such passage exists, write "FAILS".
4. Decision:
   - If ALL four conditions hold with quoted evidence -> confirm "almost".
   - If condition (A) or (B) fails -> "incorrect" (wrong approach or too incomplete).
   - If condition (C) or (D) fails -> check "(Partial)" criteria; if any
     is satisfied -> "partial"; otherwise -> "incorrect".
   DEFAULT: When in doubt, choose "partial" or "incorrect" over "almost".

Output your final answer as a JSON object enclosed in <json> tags.

<json>
{{
    "almost_criteria_check": "<for each criterion: criterion text -> satisfied/not satisfied + QUOTED evidence from student answer>",
    "approach_correct_evidence": "<QUOTED passage proving correct approach, or FAILS + reason>",
    "near_complete_evidence": "<QUOTED passage proving near-completeness, or FAILS + reason>",
    "mistakes_assessment": "<specific mistake(s) identified; are they truly minor? yes/no + reason>",
    "explicit_criterion_match": "<QUOTED passage matching the Almost criterion, or FAILS>",
    "downgrade_triggers_present": "<list any downgrade triggers that apply>",
    "reasoning": "<your conclusion>",
    "response": "almost" or "partial" or "incorrect"
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

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Uses a multi-pass strategy for borderline grades:
          - Pass 1: full grading prompt -> initial grade.
          - Pass 2a (if "correct"): check for "(Almost)" criteria -- the most
            common error is awarding "correct" when "almost" applies.
          - Pass 2b (if "incorrect"): check for "(Partial)" criteria -- the
            second most common error is missing partial credit.
          - Pass 2c (if "almost"): verify "almost" is truly warranted -- the
            most COSTLY error is over-predicting "almost" (5-6 pts per false
            positive).  Downgrade to "partial" or "incorrect" if not warranted.
          - Pass 2d (if "partial"): check whether the answer is actually
            "correct" -- catches cases where the first pass under-grades a
            complete solution.

        NOTE: Pass 3 (partial -> almost upgrade) has been removed because
        empirical data shows it causes far more false "almost" predictions
        than it corrects.  The "almost" downgrade pass (2c) is more important.

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

        if prediction in _VERIFY_GRADES:
            verify_prompt: str | None = None

            if prediction == "correct" and has_almost_criteria:
                # Most common error: "correct" when "almost" should apply.
                verify_prompt = self._build_almost_check_prompt(
                    problem=problem,
                    official_solution=official_solution,
                    grading_guidelines=grading_guidelines,
                    student_answer=student_answer,
                    first_pass_reasoning=assistant_text,
                )

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

            elif prediction == "partial" and has_almost_criteria:
                # Check whether a "partial" grade might actually be "almost".
                # The first pass sometimes under-grades a near-complete solution.
                # This is the primary path for catching "almost" cases since the
                # model rarely predicts "almost" directly.
                verify_prompt = self._build_almost_from_partial_check_prompt(
                    problem=problem,
                    official_solution=official_solution,
                    grading_guidelines=grading_guidelines,
                    student_answer=student_answer,
                    first_pass_reasoning=assistant_text,
                )

            if verify_prompt is not None:
                try:
                    v_response, v_history, _v_info = get_response_from_llm(
                        msg=verify_prompt,
                        model=self.model,
                        msg_history=[],
                    )
                    v_text = _get_assistant_text(v_history, v_response)
                    v_prediction = self._extract_prediction(v_text)

                    # Accept the verification result only if it is a valid
                    # refinement (not a completely different grade).
                    valid_refinements = {
                        "correct":   {"correct", "almost"},
                        # "incorrect" can be upgraded to partial or almost
                        "incorrect": {"incorrect", "partial", "almost"},
                        # "almost" can be confirmed or downgraded to partial/incorrect
                        "almost":    {"almost", "partial", "incorrect"},
                        # "partial" can be confirmed or upgraded to almost
                        "partial":   {"partial", "almost"},
                    }
                    if v_prediction in valid_refinements.get(prediction, set()):
                        if v_prediction != prediction:
                            self.log_fn(
                                f"Verification changed grade: {prediction} -> {v_prediction}"
                            )
                        prediction = v_prediction
                        # Append verification exchange to the message history
                        # so callers can inspect the full reasoning chain.
                        msg_history = msg_history + v_history
                    else:
                        self.log_fn(
                            f"Verification returned unexpected grade {v_prediction!r} "
                            f"for first-pass {prediction!r}; keeping original."
                        )
                except Exception as exc:
                    self.log_fn(f"Verification pass failed: {exc}; keeping original grade.")

        return prediction, msg_history
