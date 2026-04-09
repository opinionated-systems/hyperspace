"""
Task agent: solves a given task with a single LLM call.

Reimplemented from facebookresearch/HyperAgents task_agent.py.
Same interface, same JSON output format, same extraction logic.

This is the INITIAL task agent. The meta agent modifies this file
during self-improvement. The evaluation harness loads whatever
task_agent.py exists at the agent's repo path.

Improvements over the initial version:
- Domain-aware system prompt with explicit grading rubric guidance.
- Chain-of-thought reasoning step before emitting the final JSON answer.
- Multi-attempt JSON extraction: tagged blocks first, then bare-object
  regex fallback so partial responses are still usable.
- Configurable temperature (default 0.0 for determinism, can be raised
  for diversity when the meta-agent experiments).
- Strict label enforcement: output is always one of the 4 valid labels.
- Post-processing normalization to map common variants to valid labels.
- Improved grading prompt with explicit partial/almost criteria checking.
- Strict partial criteria checking to avoid false positives.
- Verification step to catch over-generous grading.
"""

from __future__ import annotations

import json
import logging
import re

from agent.llm_client import get_response_from_llm, EVAL_MODEL

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Valid labels
# ---------------------------------------------------------------------------

VALID_LABELS = {"incorrect", "partial", "almost", "correct"}

# ---------------------------------------------------------------------------
# JSON extraction helpers
# ---------------------------------------------------------------------------

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
        inner = text[start + 6 : end].strip()
        search_from = end + 7
        try:
            results.append(json.loads(inner))
        except json.JSONDecodeError:
            continue
    return results or None


def _extract_jsons_fallback(text: str) -> list[dict] | None:
    """Fallback: find the last bare JSON object in the text.

    Scans for '{' ... '}' pairs (outermost level) and tries to parse each
    one.  Returns the last successfully parsed object, or None.
    """
    results = []
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start != -1:
                candidate = text[start : i + 1]
                try:
                    results.append(json.loads(candidate))
                except json.JSONDecodeError:
                    pass
                start = -1
    return results or None


def _best_json(text: str) -> dict | None:
    """Return the best (last) JSON object found in *text*, or None."""
    tagged = _extract_jsons(text)
    if tagged:
        return tagged[-1]
    fallback = _extract_jsons_fallback(text)
    if fallback:
        return fallback[-1]
    return None


def _normalize_label(raw: str) -> str:
    """Normalize a raw prediction to one of the 4 valid labels.

    The evaluation harness compares predictions (lowercased) to the
    'Reward' column which contains exactly: incorrect, partial, almost, correct.

    This function maps common model output variants to the correct label.
    """
    if not isinstance(raw, str):
        raw = str(raw)

    normalized = raw.strip().lower()

    # Direct match
    if normalized in VALID_LABELS:
        return normalized

    # Common variants
    label_map = {
        # correct variants
        "correct": "correct",
        "full credit": "correct",
        "full marks": "correct",
        "complete": "correct",
        "complete solution": "correct",
        "completely correct": "correct",
        "fully correct": "correct",
        "right": "correct",
        "yes": "correct",
        "7": "correct",
        "7/7": "correct",
        "10/10": "correct",
        "10": "correct",
        # almost variants
        "almost": "almost",
        "almost correct": "almost",
        "nearly correct": "almost",
        "near correct": "almost",
        "minor error": "almost",
        "minor mistake": "almost",
        "minor errors": "almost",
        "minor mistakes": "almost",
        "6": "almost",
        "6/7": "almost",
        # partial variants
        "partial": "partial",
        "partially correct": "partial",
        "partial credit": "partial",
        "partial solution": "partial",
        "incomplete": "partial",
        "1": "partial",
        "1/7": "partial",
        "2": "partial",
        "3": "partial",
        "4": "partial",
        "5": "partial",
        # incorrect variants
        "incorrect": "incorrect",
        "wrong": "incorrect",
        "no credit": "incorrect",
        "no marks": "incorrect",
        "0": "incorrect",
        "0/7": "incorrect",
        "false": "incorrect",
        "no": "incorrect",
        "none": "incorrect",
    }

    if normalized in label_map:
        return label_map[normalized]

    # Check if any valid label appears as a substring
    for label in ["correct", "almost", "partial", "incorrect"]:
        if label in normalized:
            return label

    # Check for numeric scores
    # Try to extract a number
    num_match = re.search(r'\b(\d+(?:\.\d+)?)\b', normalized)
    if num_match:
        try:
            score = float(num_match.group(1))
            if score >= 6.5:
                return "correct"
            elif score >= 5.5:
                return "almost"
            elif score >= 0.5:
                return "partial"
            else:
                return "incorrect"
        except ValueError:
            pass

    # Default to incorrect if we can't determine
    logger.warning("Could not normalize label %r, defaulting to 'incorrect'", raw)
    return "incorrect"


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an expert mathematical grader evaluating student answers to
International Mathematical Olympiad (IMO) competition problems.

GRADING SYSTEM:
You must assign exactly one of these four grades:
- "correct": The student's solution is COMPLETE and CORRECT. All key steps
  are present and valid. The solution fully proves the required statement
  with no significant errors or gaps.
- "almost": The solution is NEARLY COMPLETE. The student has the right
  approach and nearly all key steps, but there are minor mistakes or small
  gaps. The (Almost) criteria in the guidelines are satisfied. NOTE: "almost"
  means the solution is NOT fully correct - there are real errors or gaps.
- "partial": The student has made SOME PROGRESS but the solution is
  incomplete. They satisfy AT LEAST ONE specific (Partial) criterion listed
  in the grading guidelines. The solution does NOT satisfy the (Almost)
  criteria.
- "incorrect": The student's solution does not meet ANY of the partial
  credit criteria. The solution is wrong, missing key ideas, or too
  incomplete to earn any credit.

CRITICAL GRADING RULES:

RULE 1 - For "correct":
Only give "correct" if the solution is GENUINELY COMPLETE and CORRECT.
If the solution has ANY significant errors, gaps, or missing cases → NOT "correct".
If the solution satisfies the (Almost) criteria → give "almost", NOT "correct".

RULE 2 - For "almost":
Give "almost" when the solution is nearly complete but has minor mistakes.
The (Almost) criteria in the guidelines describe what "almost" looks like.
Do NOT give "correct" for a solution that satisfies "almost" criteria.

RULE 3 - For "partial":
Give "partial" ONLY when the student's solution SPECIFICALLY satisfies one
of the listed (Partial) criteria. You must be able to point to the EXACT
criterion that is satisfied.
Do NOT give "partial" just because the student made some effort or had
some relevant ideas that don't match the specific criteria.
Do NOT give "partial" if the student's approach is completely different
from what the criteria describe.

RULE 4 - For "incorrect":
Give "incorrect" when the student's solution does not satisfy ANY of the
specific (Partial) criteria. Even if the student tried hard or had some
relevant ideas, if none of the specific criteria are met → "incorrect".

HOW TO GRADE - STEP BY STEP:

Step 1: Read the grading guidelines carefully.
        - List out each (Partial) criterion explicitly.
        - Note what the (Almost) criteria say.

Step 2: Check if the student's solution is COMPLETE and CORRECT.
        - Does it prove everything required?
        - Are there any errors, gaps, or missing cases?
        If YES (complete and correct with no significant issues) → "correct"
        If NO → continue to Step 3

Step 3: Check the (Almost) criteria.
        - Does the student's solution satisfy the (Almost) criteria?
        - This means: right approach, nearly complete, but with minor mistakes.
        If YES → "almost"
        If NO → continue to Step 4

Step 4: Check EACH (Partial) criterion INDIVIDUALLY and STRICTLY.
        For each criterion, ask: "Does the student's solution SPECIFICALLY
        contain this exact observation, step, or result?"
        - The criterion must be EXPLICITLY present in the student's work.
        - A vague or tangentially related idea does NOT count.
        - The student must have actually derived or stated the specific result.
        If the student satisfies AT LEAST ONE criterion SPECIFICALLY → "partial"
        If the student satisfies NONE → "incorrect"

Step 5: VERIFICATION - Double-check your answer.
        - If you said "correct": Are you sure there are no errors or gaps?
          Could it be "almost" instead?
        - If you said "almost": Are you sure it's not "correct" or "partial"?
        - If you said "partial": Can you point to the SPECIFIC criterion met?
          Is the criterion ACTUALLY present in the student's work?
        - If you said "incorrect": Did you check ALL partial criteria carefully?

OUTPUT FORMAT:
First, reason through your analysis step by step following the HOW TO GRADE
steps above. Then output your final grade in a JSON block:

<json>
{"response": "correct"}
</json>

Your response MUST be exactly one of: "incorrect", "partial", "almost", "correct"
"""


# ---------------------------------------------------------------------------
# TaskAgent
# ---------------------------------------------------------------------------

class TaskAgent:
    """Task agent that solves IMO grading problems."""

    def __init__(
        self,
        model: str = EVAL_MODEL,
        log_file: str = "",
        temperature: float = 0.0,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.log_fn = logger.info

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution,
                    grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Build a structured prompt that surfaces every relevant field.
        domain = inputs.get("domain", "")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = f"""{_SYSTEM_PROMPT}

---
PROBLEM:
{problem}

OFFICIAL SOLUTION:
{solution}

GRADING GUIDELINES:
{guidelines}

STUDENT'S ANSWER:
{student_answer}
---

Now grade the student's answer following the HOW TO GRADE steps:

Step 1: List each (Partial) criterion and each (Almost) criterion from the guidelines.

Step 2: Is the solution complete and correct? Check for errors, gaps, missing cases.
        → If yes: "correct". If no: continue.

Step 3: Does it satisfy the (Almost) criteria?
        → If yes: "almost". If no: continue.

Step 4: For EACH (Partial) criterion, check if it is SPECIFICALLY present in the student's work.
        Quote the relevant part of the student's answer for each criterion you check.
        → If at least one criterion is specifically met: "partial". If none: "incorrect".

Step 5: Verify your answer. If "partial", confirm you can point to the exact criterion met.
        If "correct", confirm there are no errors or gaps.

Provide your final grade in the <json> block.
"""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            temperature=self.temperature,
            msg_history=[],
        )

        # Extract prediction — try tagged blocks first, then fallback.
        prediction = "incorrect"
        try:
            last_text = msg_history[-1]["text"]
            parsed = _best_json(last_text)
            if parsed is not None and "response" in parsed:
                raw_pred = parsed["response"]
                prediction = _normalize_label(str(raw_pred))
                self.log_fn("Extracted prediction: %r -> normalized: %r", raw_pred, prediction)
            else:
                self.log_fn("No 'response' key found in extracted JSON; raw text: %s", last_text[:300])
                # Try to find a label directly in the text
                text_lower = last_text.lower()
                # Look for the label near the end of the response
                for label in ["correct", "almost", "partial", "incorrect"]:
                    # Search in last 500 chars for the label
                    tail = text_lower[-500:]
                    if label in tail:
                        prediction = label
                        self.log_fn("Found label %r in text tail", label)
                        break
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
