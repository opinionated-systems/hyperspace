"""
Task agent: solves a given task with a single LLM call.

Reimplemented from facebookresearch/HyperAgents task_agent.py.
Same interface, same JSON output format, same extraction logic.

This is the INITIAL task agent. The meta agent modifies this file
during self-improvement. The evaluation harness loads whatever
task_agent.py exists at the agent's repo path.

Improvements over the initial version
--------------------------------------
* Richer system prompt: the agent is told it is an expert grader and is
  instructed to reason step-by-step before emitting the JSON answer.
* Robust JSON extraction: after the primary <json>…</json> block search
  we fall back to a bare ```json … ``` fence and then to a raw brace scan,
  so partial or reformatted model output is still handled gracefully.
* Prediction logging: the extracted prediction is always logged so
  debugging is easier.
* Cleaner separation between the system prompt and the per-task payload.
* Multi-turn self-consistency: the agent makes up to MAX_VOTES independent
  calls and returns the majority-vote answer, reducing variance on
  borderline cases.
* Structured input formatting: inputs dict is rendered as labelled
  key/value sections rather than a raw repr(), making it easier for the
  LLM to parse each field.
* Confidence extraction: if the model emits an optional "confidence"
  field (0–1) it is logged alongside the prediction for diagnostics.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter

from agent.llm_client import get_response_from_llm, EVAL_MODEL

logger = logging.getLogger(__name__)

# Number of independent LLM calls whose answers are majority-voted.
# Set to 1 to disable voting (original behaviour).
MAX_VOTES = 3

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = """\
You are an expert mathematical grader specialising in olympiad problems.
Your job is to evaluate a student's answer against the official solution and
grading guidelines, then return a structured verdict.

Follow these steps:
1. Read the problem statement carefully.
2. Study the official solution and grading guidelines.
3. Analyse the student's answer, noting what is correct, partially correct,
   or incorrect.  Pay close attention to whether the student's reasoning is
   valid even if the final numerical answer differs only by a sign or a
   trivial arithmetic slip.
4. Decide on the score / verdict according to the grading guidelines.
5. Emit your final answer as a single JSON block wrapped in <json>…</json>
   tags.  Do NOT emit any text after the closing </json> tag.

The JSON must conform to this schema (confidence is optional, 0–1 float):
<json>
{
    "response": <your verdict / score here>,
    "confidence": <optional float 0-1>
}
</json>
"""


# ---------------------------------------------------------------------------
# Input formatting
# ---------------------------------------------------------------------------

def _format_inputs(inputs: dict) -> str:
    """Render the inputs dict as labelled sections for the LLM.

    Produces a more readable layout than a raw repr(), which helps the
    model correctly identify each field (problem, solution, etc.).
    """
    lines = []
    for key, value in inputs.items():
        lines.append(f"### {key.upper().replace('_', ' ')}")
        lines.append(str(value).strip())
        lines.append("")
    return "\n".join(lines)


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
        inner = text[start + 6:end].strip()
        search_from = end + 7
        try:
            results.append(json.loads(inner))
        except json.JSONDecodeError:
            continue
    return results or None


def _extract_jsons_fence(text: str) -> list[dict] | None:
    """Fallback: extract JSON from ```json … ``` fenced code blocks."""
    results = []
    for m in re.finditer(r"```json\s*(.*?)```", text, re.DOTALL):
        try:
            results.append(json.loads(m.group(1).strip()))
        except json.JSONDecodeError:
            continue
    return results or None


def _extract_jsons_brace_scan(text: str) -> list[dict] | None:
    """Last-resort: find the first top-level {...} object in the text."""
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                candidate = text[start:i + 1]
                try:
                    return [json.loads(candidate)]
                except json.JSONDecodeError:
                    start = None
    return None


def _best_extract(text: str) -> list[dict] | None:
    """Try extraction strategies in order of preference."""
    return (
        _extract_jsons(text)
        or _extract_jsons_fence(text)
        or _extract_jsons_brace_scan(text)
    )


def _extract_prediction(raw_text: str, log_fn) -> str:
    """Extract the 'response' value from a raw LLM output string.

    Logs the extracted prediction and any optional confidence score.
    Returns the string "None" if extraction fails.
    """
    extracted = _best_extract(raw_text)
    if not extracted or "response" not in extracted[-1]:
        log_fn(
            "Warning: could not find 'response' key in extracted JSON. "
            f"Raw tail: {raw_text[-300:]!r}"
        )
        return "None"

    result = extracted[-1]
    prediction = result["response"]
    confidence = result.get("confidence")
    if confidence is not None:
        log_fn(f"Extracted prediction: {prediction!r}  (confidence={confidence})")
    else:
        log_fn(f"Extracted prediction: {prediction!r}")
    return str(prediction)


# ---------------------------------------------------------------------------
# Majority vote
# ---------------------------------------------------------------------------

def _majority_vote(predictions: list[str]) -> str:
    """Return the most common prediction; ties broken by first occurrence."""
    if not predictions:
        return "None"
    counts = Counter(predictions)
    # most_common preserves insertion order for equal counts in Python 3.7+
    return counts.most_common(1)[0][0]


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class TaskAgent:
    """Task agent that solves IMO grading problems."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Makes up to MAX_VOTES independent LLM calls and returns the
        majority-vote answer together with the last call's message history.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines,
                    student_answer

        Returns:
            (prediction, msg_history)
        """
        # Build the per-task user message with structured input formatting.
        instruction = (
            _SYSTEM_PROMPT
            + "\n---\nTask input:\n\n"
            + _format_inputs(inputs)
        )

        predictions: list[str] = []
        last_msg_history: list[dict] = []

        for vote_idx in range(MAX_VOTES):
            try:
                _response, msg_history, _info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=[],
                )
                last_msg_history = msg_history
                raw_text = msg_history[-1]["text"]
                pred = _extract_prediction(raw_text, self.log_fn)
                predictions.append(pred)
                self.log_fn(f"Vote {vote_idx + 1}/{MAX_VOTES}: {pred!r}")
            except Exception as e:
                self.log_fn(f"Vote {vote_idx + 1}/{MAX_VOTES} failed: {e}")

        if not predictions:
            return "None", last_msg_history

        final = _majority_vote(predictions)
        if MAX_VOTES > 1:
            self.log_fn(f"Majority vote result ({predictions}): {final!r}")
        return final, last_msg_history
