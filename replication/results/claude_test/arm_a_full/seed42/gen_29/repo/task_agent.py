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
- Explicit criterion-by-criterion checking to improve "almost" and "partial" recall.
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
- "correct": The student's solution is complete and correct. ALL key steps
  are present and logically valid. The solution fully proves the required
  statement with NO significant gaps or errors.
- "almost": The solution is nearly complete but has minor mistakes or small
  gaps that prevent it from being fully correct. The student has the right
  overall approach and most key steps, but there are non-negligible errors
  or omissions. This includes: solutions that are correct but miss edge cases,
  solutions with minor computational errors, solutions that are almost done
  but not quite finished.
- "partial": The student has made meaningful progress but the solution is
  substantially incomplete. They satisfy some (but not all) of the partial
  credit criteria listed in the grading guidelines. Even a single key
  observation or correct step can earn "partial" credit.
- "incorrect": The student's solution does not meet ANY of the partial
  credit criteria. The solution is fundamentally wrong, missing all key
  ideas, or too incomplete to earn any credit.

CRITICAL GRADING RULES:
1. "almost" is NOT rare - use it whenever the solution is nearly complete
   but has minor mistakes, gaps, or omissions. Do NOT upgrade "almost" to
   "correct" just because the approach is right.
2. "partial" should be used generously - if the student shows ANY meaningful
   mathematical insight or correct step listed in the (Partial) criteria,
   grade as "partial", not "incorrect".
3. "correct" requires a COMPLETE proof with ALL steps verified. If there
   are any gaps, missing cases, or unverified claims, it is NOT "correct".
4. The (Almost) and (Partial) criteria in the grading guidelines are
   SPECIFIC CHECKBOXES - check each one explicitly.

HOW TO GRADE (follow these steps in order):

STEP 1: Read the (Almost) criteria carefully.
- Does the student's solution satisfy ANY of the (Almost) criteria?
- If YES → grade as "almost" (unless the solution is fully correct)
- Common (Almost) patterns: "minor mistakes only", "almost complete but...",
  "right approach but not completed", "omitted a case", "minor gap"

STEP 2: Check if the solution is truly "correct".
- Is the proof COMPLETE with ALL cases covered?
- Are ALL key steps present and logically valid?
- Are there ANY gaps, missing cases, or unverified claims?
- If the solution satisfies an (Almost) criterion → it is NOT "correct"
- Only grade "correct" if the solution is genuinely complete and rigorous

STEP 3: Check the (Partial) criteria.
- Go through EACH (Partial) criterion one by one
- Does the student's solution satisfy ANY of them?
- If YES → grade as "partial"
- Be generous: even a single correct observation counts

STEP 4: If none of the above → grade as "incorrect"

IMPORTANT:
- Your response MUST be exactly one of: "incorrect", "partial", "almost", "correct"
- Do NOT output numeric scores, fractions, or any other format
- Do NOT output explanations in the JSON - only the label

OUTPUT FORMAT:
First, reason through your analysis step by step following the 4 steps above.
Then output your final grade in a JSON block:

<json>
{"response": "correct"}
</json>

or

<json>
{"response": "almost"}
</json>

etc.
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

Now grade the student's answer following the 4-step process:

STEP 1 - Check (Almost) criteria:
Look at the (Almost) section of the grading guidelines above.
Does the student's solution satisfy any of these criteria?
(Remember: "almost" includes minor mistakes, small gaps, nearly-complete solutions,
solutions that omit edge cases, solutions with minor errors in verification, etc.)

STEP 2 - Check for "correct":
Is the proof truly COMPLETE? Are ALL cases covered? Are there ANY gaps?
(Be strict: if there are any minor mistakes or omissions, it's "almost" not "correct")

STEP 3 - Check (Partial) criteria:
Look at the (Partial) section of the grading guidelines above.
Does the student's solution satisfy ANY of these specific criteria?
(Be generous: even one correct observation or step earns "partial")

STEP 4 - Final decision:
Based on the above analysis, what is the correct grade?

Remember the priority order:
- If solution is complete and correct with no gaps → "correct"
- If solution satisfies any (Almost) criterion → "almost"  
- If solution satisfies any (Partial) criterion → "partial"
- Otherwise → "incorrect"

Provide your analysis and then your final grade in the JSON block.
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
