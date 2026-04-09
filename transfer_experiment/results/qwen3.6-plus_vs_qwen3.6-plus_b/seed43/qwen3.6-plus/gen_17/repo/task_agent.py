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

__version__ = "1.12.0"

VALID_LABELS = {"Correct", "Incorrect", "Partial", "Almost"}


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks.

    Uses str.find to locate each <json>...</json> block sequentially,
    avoiding the lazy .*? regex bug that truncates content with nested braces.
    Also handles case-insensitive tags and tags with attributes.
    """
    results = []
    search_from = 0
    while True:
        start = text.find("<json>", search_from)
        if start == -1:
            # Try case-insensitive fallback
            lower_text = text.lower()
            start = lower_text.find("<json>", search_from)
            if start == -1:
                break
        end = text.find("</json>", start)
        if end == -1:
            # Try case-insensitive fallback
            end = lower_text.find("</json>", start)
            if end == -1:
                break
        inner = text[start + 6:end].strip()
        search_from = end + 7
        try:
            results.append(json.loads(inner))
        except json.JSONDecodeError:
            continue
    return results or None


def _extract_json_fallback(text: str) -> list[dict] | None:
    """Fallback: try to find JSON objects in text without <json> tags.

    Searches for the last top-level JSON object by tracking brace depth.
    """
    # Find the last '{' that starts a valid JSON object
    last_start = -1
    depth = 0
    in_string = False
    escape_next = False

    for i, ch in enumerate(text):
        if escape_next:
            escape_next = False
            continue
        if ch == '\\':
            escape_next = True
            continue
        if ch == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == '{':
            if depth == 0:
                last_start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and last_start != -1:
                candidate = text[last_start:i + 1].strip()
                try:
                    obj = json.loads(candidate)
                    # Only return if it has the expected key
                    if isinstance(obj, dict) and "response" in obj:
                        return [obj]
                except json.JSONDecodeError:
                    pass
                last_start = -1
    return None


def _normalize_label(label: str) -> str:
    """Normalize a label string to one of the valid labels.

    Handles common variations like case differences, whitespace, and synonyms.
    """
    if not label or not label.strip():
        return "Incorrect"
    label = label.strip().strip('"').strip("'")
    # Case-insensitive matching
    label_lower = label.lower()
    if label_lower == "correct":
        return "Correct"
    elif label_lower in ("incorrect", "wrong", "error", "invalid"):
        return "Incorrect"
    elif label_lower in ("partial", "partially correct", "partially"):
        return "Partial"
    elif label_lower in ("almost", "almost correct", "nearly correct", "nearly"):
        return "Almost"
    return label


class TaskAgent:
    """Task agent that solves IMO grading problems."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def __repr__(self) -> str:
        return f"TaskAgent(model={self.model!r})"

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = f"""You are an expert IMO math grader. Your task is to grade a student's answer.

**Label Definitions:**
- "Correct": The student's answer is fully correct, with valid reasoning and the right conclusion. Every step is sound and the final answer matches.
- "Incorrect": The student's answer is fundamentally wrong, has major logical errors, uses an invalid approach, or reaches an incorrect conclusion.
- "Partial": The student's answer shows meaningful progress with correct intermediate steps but is incomplete, has notable errors in later steps, or doesn't reach a final conclusion. The student is on the right track but hasn't finished.
- "Almost": The student's answer is nearly complete and correct — the approach is right, most steps are valid, but there is a small gap, a minor computational/arithmetic error near the end, or a missing final justification/conclusion. The student essentially has the solution.

**Decision Guide (apply in order):**
1. Is the final answer correct AND is every step of the reasoning valid? → "Correct"
2. Is the approach fundamentally wrong, or does it reach a clearly wrong conclusion? → "Incorrect"
3. Has the student essentially solved the problem but has a tiny error, missing final step, or minor gap? → "Almost"
4. Has the student made significant correct progress but hasn't finished or has notable errors? → "Partial"
5. If none of the above clearly apply, re-evaluate: is the work mostly right with small issues (Almost) or is there substantial correct work but incomplete (Partial)?

**Key Distinction — "Almost" vs "Partial":**
- "Almost" = the student has the core solution and only needs a small fix (e.g., one missing case, a minor algebraic slip, a brief justification). If you can point to exactly what's missing in one sentence, it's "Almost".
- "Partial" = the student has made real progress but still needs substantial work to complete the solution (e.g., missing a major step, an incomplete argument, or a significant gap in reasoning).

**Important:** 
- Do NOT default to only "Correct" or "Incorrect". 
- "Partial" and "Almost" are common in IMO grading — use them when appropriate.
- "Almost" means the student is very close to a full solution (minor issue).
- "Partial" means the student has made real progress but is further from completion.

---

**Problem:**
{problem}

**Reference Solution:**
{solution}

**Grading Guidelines:**
{grading_guidelines}

**Student Answer:**
{student_answer}

---

First, think through your reasoning step by step:
1. What is the key idea/strategy needed to solve this problem?
2. Does the student's answer demonstrate understanding of this key idea?
3. Compare the student's solution to the reference: which steps match, which are missing, which are wrong?
4. Are there any errors, gaps, or missing steps in the student's reasoning? How significant are they?
5. How complete is the student's solution compared to the reference?
6. Based on the label definitions and decision guide above, which label best fits?

After your reasoning, respond with EXACTLY this format, enclosed in <json> tags:
<json>
{{"response": "Correct"}}
</json>

The "response" value must be exactly one of: "Correct", "Incorrect", "Partial", or "Almost"."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON
        prediction = "None"
        try:
            extracted = _extract_jsons(msg_history[-1]["text"])
            if extracted and "response" in extracted[-1]:
                prediction = _normalize_label(extracted[-1]["response"])
            else:
                # Fallback: try to extract JSON without <json> tags
                extracted = _extract_json_fallback(msg_history[-1]["text"])
                if extracted and "response" in extracted[-1]:
                    prediction = _normalize_label(extracted[-1]["response"])
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
