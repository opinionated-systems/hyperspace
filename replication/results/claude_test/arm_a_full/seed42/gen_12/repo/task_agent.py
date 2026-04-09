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
- Structured criterion-by-criterion analysis to reduce over/under-grading.
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

GRADING SYSTEM — READ CAREFULLY:
You must assign exactly one of these four grades:

- "correct": The student's solution is COMPLETE and CORRECT. Every key step
  is present and rigorously justified. The solution fully proves the required
  statement with no significant gaps or errors. A solution that has the right
  idea but is missing crucial steps or has non-trivial errors is NOT "correct".

- "almost": The solution is nearly complete. The student has the right overall
  approach and has completed most of the proof, but there are minor mistakes
  or small gaps that prevent it from being fully correct. The (Almost) criteria
  in the grading guidelines describe exactly what qualifies.

- "partial": The student has made SOME meaningful progress. They satisfy at
  least one (but not all) of the (Partial) criteria listed in the grading
  guidelines. A solution that shows a good approach but fails to complete the
  proof typically earns "partial" credit, NOT "correct". Even a solution that
  looks sophisticated but only achieves partial criteria is "partial".

- "incorrect": The student's solution does not satisfy ANY of the (Partial)
  criteria. The solution is wrong, missing all key ideas, or too vague/
  incomplete to earn any credit.

CRITICAL DISTINCTIONS:
1. "correct" vs "partial"/"almost": A solution is "correct" ONLY if it is
   COMPLETE. If the student has a good approach but the proof is incomplete,
   missing key steps, or has significant errors, it is "partial" or "almost",
   NOT "correct". Be skeptical — ask yourself: "Is every required step actually
   proven here, or just claimed/sketched?"

2. "partial" vs "incorrect": A solution earns "partial" if it satisfies even
   ONE of the listed (Partial) criteria. Look carefully at each criterion —
   even a partial observation or a single key step can earn "partial" credit.
   Do not grade as "incorrect" if the student has made any meaningful progress
   matching the (Partial) criteria.

3. "almost" vs "partial": "almost" means the solution is nearly complete with
   only minor issues. "partial" means significant parts are missing. Use the
   (Almost) criteria in the guidelines to distinguish these.

HOW TO GRADE — FOLLOW THESE STEPS:
Step 1: Read the problem and official solution to understand what a complete
        proof requires.
Step 2: Read the grading guidelines carefully. List each (Partial) criterion
        and each (Almost) criterion.
Step 3: For each (Partial) criterion, determine: does the student's answer
        satisfy this criterion? (yes/no/partially)
Step 4: For each (Almost) criterion, determine: does the student's answer
        satisfy this criterion?
Step 5: Check completeness — is the student's proof actually complete and
        rigorous, or just an outline/sketch?
Step 6: Apply the decision rule:
        - If the proof is complete and correct → "correct"
        - If it satisfies (Almost) criteria → "almost"
        - If it satisfies ≥1 (Partial) criteria → "partial"
        - Otherwise → "incorrect"

IMPORTANT:
- Your response MUST be exactly one of: "incorrect", "partial", "almost", "correct"
- Do NOT output numeric scores, fractions, or any other format
- Do NOT output explanations in the JSON - only the label
- When in doubt between "correct" and "partial"/"almost", lean toward the
  lower grade unless the proof is genuinely complete
- When in doubt between "partial" and "incorrect", lean toward "partial" if
  the student has shown any meaningful progress matching the criteria

OUTPUT FORMAT:
First, reason through your analysis following the steps above.
Then output your final grade in a JSON block:

<json>
{"response": "correct"}
</json>
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

Now grade this student's answer by following the steps in the system prompt:

Step 1: What does a complete proof require? (summarize key steps from official solution)

Step 2: List each grading criterion:
- (Partial) criteria: [list them]
- (Almost) criteria: [list them]

Step 3: For each (Partial) criterion, does the student satisfy it? (yes/no + brief reason)

Step 4: For each (Almost) criterion, does the student satisfy it? (yes/no + brief reason)

Step 5: Completeness check — Is the student's proof actually complete and rigorous?
What key steps are MISSING or INCORRECT in the student's answer?

Step 6: Final decision:
- Complete and correct proof? → "correct"
- Satisfies (Almost) criteria? → "almost"  
- Satisfies ≥1 (Partial) criteria? → "partial"
- Satisfies none? → "incorrect"

Remember: "correct" requires a COMPLETE proof. A good approach with missing
steps is "partial" or "almost", not "correct".

<json>
{"response": "YOUR_GRADE_HERE"}
</json>
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
