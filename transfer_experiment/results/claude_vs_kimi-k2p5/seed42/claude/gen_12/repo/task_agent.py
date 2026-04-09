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

# Valid grade labels (output space).
# "almost" is a distinct 4th label used by the eval harness (worth 6/7 points).
# It applies when the student's answer satisfies an "(Almost)" criterion in the
# grading guidelines — essentially complete but with minor, non-negligible gaps.
_VALID_GRADES = {"correct", "almost", "partial", "incorrect"}


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
  25% are "partial", and 5% are "almost".
  "almost" is RARE -- only ~5% of answers.  Do not award it liberally.
  If you find yourself assigning "incorrect" to more than half the
  answers you grade, you are likely being too strict on "partial".

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
  CRITICAL TESTS for "almost":
    (i)  Is the student's overall approach CORRECT (not just plausible)?
    (ii) Is the solution GENUINELY NEAR-COMPLETE (not just long)?
    (iii) Are the mistakes truly MINOR (not fundamental errors)?
  If the criterion says "minor mistakes only" or "almost complete",
  verify that the student has the right strategy and is close to done.
  An answer that is long and detailed but uses a WRONG approach or
  reaches a WRONG conclusion does NOT satisfy an "almost" criterion.
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

================================================================
PROBLEM
================================================================
{{problem}}

================================================================
OFFICIAL SOLUTION
================================================================
{{official_solution}}

================================================================
GRADING GUIDELINES
================================================================
{{grading_guidelines}}

================================================================
STUDENT'S ANSWER
================================================================
{{student_answer}}

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
Step 5. Before finalising "almost", verify: (a) the student's approach
        is correct, (b) the solution is genuinely near-complete, and
        (c) the mistakes are truly minor.  If in doubt, prefer
        "partial" or "incorrect" over "almost".
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
    "step2_almost": "Student applied infinite descent (matches Almost criterion 1). Approach is correct and solution is near-complete. Grade: almost.",
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
