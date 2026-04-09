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
- "correct": The student's solution is complete and correct. All key steps
  are present and valid. The solution fully proves the required statement.
- "almost": The solution is almost complete. The student has the right
  approach and nearly all key steps, but there are minor mistakes or small
  gaps that are not negligible. The (Almost) criteria in the guidelines
  are satisfied.
- "partial": The student has made meaningful progress but the solution is
  incomplete. They satisfy at least one (but not all) of the (Partial)
  criteria listed in the grading guidelines. The solution does NOT satisfy
  the (Almost) criteria.
- "incorrect": The student's solution does not meet ANY of the partial
  credit criteria. The solution is wrong, missing key ideas, or too
  incomplete to earn any credit.

CRITICAL GRADING RULES:
1. A solution that is INCOMPLETE but contains a KEY INSIGHT from the
   (Partial) criteria list should be graded "partial", NOT "incorrect".
   Even if the overall solution is wrong or incomplete, partial credit
   is awarded for specific correct observations.

2. A solution that is NEARLY COMPLETE but has minor mistakes should be
   graded "almost", NOT "correct". Do not give "correct" if the solution
   has errors or gaps, even small ones.

3. A solution that satisfies PARTIAL criteria but NOT the ALMOST criteria
   should be graded "partial", NOT "almost" or "correct".

4. Only give "correct" if the solution is genuinely complete and correct
   with no significant errors or gaps.

HOW TO GRADE - STEP BY STEP:
Step 1: Read the grading guidelines carefully. Note the exact (Partial)
        criteria and (Almost) criteria listed.

Step 2: Check if the student's solution is COMPLETE and CORRECT.
        - Does it prove everything required?
        - Are there any errors or gaps?
        If YES (complete and correct) → "correct"
        If NO → continue to Step 3

Step 3: Check the (Almost) criteria.
        - Does the student's solution satisfy the (Almost) criteria?
        - This typically means: right approach, nearly complete, but with
          minor mistakes or small gaps.
        If YES → "almost"
        If NO → continue to Step 4

Step 4: Check EACH (Partial) criterion individually.
        - Go through each numbered item in the (Partial) list.
        - Does the student's solution contain this specific observation,
          step, or result?
        - Be generous: if the student has the key idea even if expressed
          differently, count it.
        If the student satisfies AT LEAST ONE (Partial) criterion → "partial"
        If the student satisfies NONE → "incorrect"

IMPORTANT WARNINGS:
- Do NOT give "correct" for a solution that is only partially complete.
  A solution that proves some but not all required parts is at most "partial".
- Do NOT give "incorrect" for a solution that contains a key insight from
  the (Partial) criteria, even if the overall solution is wrong.
- The (Partial) criteria are SPECIFIC CHECKBOXES. Check each one carefully.
- A solution can be "partial" even if it reaches a wrong final answer,
  as long as it contains correct intermediate steps listed in the criteria.

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

Step 1: Is the solution complete and correct? (Check for errors and gaps)
Step 2: Does it satisfy the (Almost) criteria?
Step 3: Does it satisfy ANY of the (Partial) criteria? (Check each one!)
Step 4: Final grade

Remember:
- "partial" if ANY (Partial) criterion is met (even if overall solution is wrong)
- "almost" if (Almost) criteria are met
- "correct" ONLY if the solution is genuinely complete and correct
- "incorrect" ONLY if NO criteria are met at all

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
