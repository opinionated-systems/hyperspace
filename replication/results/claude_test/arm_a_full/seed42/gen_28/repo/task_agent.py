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
- Improved grading rubric with explicit guidance on "almost" vs "correct"
  to fix the systematic over-prediction of "correct".
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

GRADING SYSTEM — FOUR GRADES ONLY:
You must assign exactly one of these four grades:

1. "correct": The student's solution is COMPLETE and CORRECT with NO significant errors.
   - Every key step is present and mathematically valid
   - The solution fully proves the required statement
   - Minor presentation issues are acceptable, but the mathematical content must be complete
   - WARNING: Do NOT assign "correct" if there are any mathematical errors, gaps in logic,
     missing cases, or unproven claims — even if the overall approach is right

2. "almost": The solution is NEARLY COMPLETE but has minor mistakes that prevent it from
   being fully correct. The student clearly has the right approach and most key steps.
   - The (Almost) criteria in the grading guidelines are satisfied
   - Examples: a small computational error, one missing case, a gap in one step of the proof,
     an unverified claim that is non-trivial to verify
   - The solution would be correct with minor fixes
   - IMPORTANT: Use "almost" when the solution is essentially right but has a flaw that
     prevents it from being fully correct — even if the student claims it is complete

3. "partial": The student has made MEANINGFUL PROGRESS but the solution is INCOMPLETE.
   - The student satisfies one or more (Partial) criteria in the grading guidelines
   - The student has found some key ideas or observations but not enough for "almost"
   - Examples: proved one direction of an iff, found a key lemma, identified the right answer
     without full proof, made a correct key observation

4. "incorrect": The student's solution does NOT satisfy any of the partial credit criteria.
   - The solution is fundamentally wrong, missing all key ideas, or too incomplete
   - The student may have attempted the problem but made no meaningful progress

CRITICAL GRADING RULES:
- "correct" requires a COMPLETE, ERROR-FREE proof. Be skeptical — actively look for errors.
- "almost" is for solutions that are essentially right but have a specific flaw or gap.
  Do NOT skip "almost" and jump to "correct" just because the solution looks long or detailed.
- "partial" is for solutions that show some understanding but are clearly incomplete.
- The grading guidelines specify EXACTLY what counts as "partial" and "almost" — follow them.

STEP-BY-STEP GRADING PROCESS:
1. Read the problem statement carefully.
2. Study the official solution to understand what a complete proof requires.
3. Read the grading guidelines — note the specific (Partial) and (Almost) criteria.
4. Read the student's answer carefully, looking for:
   a. Does it address all required cases/parts of the problem?
   b. Are all key steps present and mathematically valid?
   c. Are there any errors, gaps, or unproven claims?
   d. Does it match the (Almost) criteria? The (Partial) criteria?
5. Make your grading decision:
   - If the proof is complete and correct → "correct"
   - If it satisfies the (Almost) criteria (nearly complete, minor flaw) → "almost"
   - If it satisfies one or more (Partial) criteria → "partial"
   - Otherwise → "incorrect"

OUTPUT FORMAT:
First, reason through your analysis step by step.
Then output your final grade in a JSON block:

<json>
{"response": "correct"}
</json>

The value must be exactly one of: "incorrect", "partial", "almost", "correct"
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

Now carefully grade the student's answer following the step-by-step process above.

REMINDER of the grading criteria:
- "correct": Complete, error-free proof. Be skeptical — actively look for errors, gaps, missing cases.
- "almost": Nearly complete but has a specific flaw/gap matching the (Almost) criteria.
  IMPORTANT: If the student's solution has a flaw that prevents it from being fully correct,
  even if they claim it is complete, use "almost" (not "correct").
- "partial": Satisfies one or more (Partial) criteria but not enough for "almost".
- "incorrect": Does not satisfy any criteria.

Your analysis should:
1. Identify what the student's approach is
2. Check each key step for correctness
3. Identify any errors, gaps, or missing cases
4. Compare against the (Almost) and (Partial) criteria in the grading guidelines
5. Make your final decision

Output your grade as:
<json>
{{"response": "YOUR_GRADE_HERE"}}
</json>

where YOUR_GRADE_HERE is one of: incorrect, partial, almost, correct
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
