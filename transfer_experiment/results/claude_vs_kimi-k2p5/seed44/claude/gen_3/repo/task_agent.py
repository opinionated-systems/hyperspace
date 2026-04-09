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

# Valid prediction labels (the harness also uses "almost" as a label,
# which we treat as "partial" since it represents near-correct work).
_VALID_LABELS = {"correct", "partial", "incorrect", "almost"}


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
    """Map a raw model response string to one of the canonical labels.

    The evaluation harness accepts: "correct", "partial", "incorrect".
    "almost" (used in some rubrics) is treated as "partial".

    The key fix here is to use exact-match checks rather than substring
    checks, which previously caused "incorrect" to match "correct".
    """
    raw = raw.strip().lower()

    # Exact match first (most reliable)
    if raw in _VALID_LABELS:
        if raw == "almost":
            return "partial"
        return raw

    # Prefix / contained-word matching as fallback, ordered carefully so
    # that "incorrect" is checked before "correct" to avoid false positives.
    if raw.startswith("incorrect") or raw == "wrong":
        return "incorrect"
    if raw.startswith("almost"):
        return "partial"
    if raw.startswith("partial"):
        return "partial"
    if raw.startswith("correct"):
        return "correct"

    # Last-resort keyword scan — check longer/more-specific words first.
    if "incorrect" in raw or "wrong" in raw:
        return "incorrect"
    if "almost" in raw:
        return "partial"
    if "partial" in raw:
        return "partial"
    if "correct" in raw:
        return "correct"

    return "incorrect"


def _extract_prediction(text: str) -> str | None:
    """Try multiple strategies to pull a label out of model output."""
    # Strategy 1: <json>...</json> block
    extracted = _extract_jsons(text)
    if extracted:
        for obj in reversed(extracted):
            if "response" in obj:
                return _normalise_label(str(obj["response"]))

    # Strategy 2: bare JSON object anywhere in the text
    for match in re.finditer(r'\{[^{}]*"response"\s*:\s*"([^"]+)"[^{}]*\}', text):
        return _normalise_label(match.group(1))

    # Strategy 3: look for the label on its own line or after a colon
    for pattern in [
        r'"response"\s*:\s*"([^"]+)"',
        r'(?i)\bgrade\s*:\s*(correct|partial|incorrect|almost)\b',
        r'(?i)\b(incorrect|almost|partial|correct)\b',
    ]:
        m = re.search(pattern, text)
        if m:
            return _normalise_label(m.group(1))

    return None


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
Then decide whether the student's answer deserves a grade of **"correct"**, **"partial"**, or **"incorrect"**.

### Grading definitions — read carefully before grading
- **correct** – The student's answer is fully correct and complete. Every key step, lemma, and logical connection required by the grading guidelines is present and sound. A minor notational slip or trivially-fixable arithmetic error is acceptable, but **any missing proof step, unjustified claim, or logical gap disqualifies "correct"**. When in doubt between "correct" and "partial", choose **"partial"**.
- **partial** – The student's answer contains meaningful correct progress (e.g. a correct key lemma, a correct invariant, a correct special case, or a nearly-complete argument with one small but non-trivial gap or error). Use "partial" for answers that are *almost* right but have a small mistake or omission that prevents full credit. **"Almost correct" is still "partial", not "correct".**
- **incorrect** – The student's answer is wrong, fundamentally flawed, trivially incomplete, or does not engage meaningfully with the core difficulty of the problem. A correct final numerical answer with no supporting argument is **incorrect** unless the guidelines explicitly award credit for the answer alone.

### Common grading pitfalls to avoid
- Do **not** award "correct" just because the student states the right final answer — the reasoning must also be complete and valid.
- Do **not** award "correct" if the student skips a non-trivial step that the grading guidelines require.
- Do **not** award "incorrect" if the student has made genuine, meaningful progress even if the solution is incomplete.
- An answer that is "almost" correct (right idea, minor error) should be graded **"partial"**, not "correct".

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

## Grading procedure
Work through the following steps explicitly before giving your verdict:

**Step 1 — Key requirements:** List the essential steps/claims the student must establish (derived from the grading guidelines and official solution).

**Step 2 — Student's achievements:** For each requirement, state whether the student has established it correctly, partially, or not at all.

**Step 3 — Gaps and errors:** Identify any missing steps, unjustified claims, or logical errors in the student's answer.

**Step 4 — Verdict:** Based on the above, assign one of: "correct", "partial", or "incorrect". Remember: if there is any non-trivial gap or error, the answer is at most "partial". If the student has made no meaningful progress, it is "incorrect".

Respond with a JSON block in the following format:
<json>
{{
    "key_requirements": "<comma-separated list of required steps>",
    "student_achievements": "<what the student got right>",
    "gaps_and_errors": "<what is missing or wrong; write 'none' if fully correct>",
    "reasoning": "<one-sentence justification of your verdict>",
    "response": "<correct | partial | incorrect>"
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
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return prediction, msg_history
