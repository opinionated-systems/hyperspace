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
"""

from __future__ import annotations

import json
import logging
import re

from agent.llm_client import get_response_from_llm, EVAL_MODEL

logger = logging.getLogger(__name__)

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
   or incorrect.
4. Decide on the score / verdict according to the grading guidelines.
5. Emit your final answer as a single JSON block wrapped in <json>…</json>
   tags.  Do NOT emit any text after the closing </json> tag.

The JSON must conform to this schema:
<json>
{
    "response": <your verdict / score here>
}
</json>
"""


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

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines,
                    student_answer

        Returns:
            (prediction, msg_history)
        """
        # Build the per-task user message.  The system prompt is prepended
        # as the first assistant-primed turn so it works with models that
        # don't support a dedicated system role.
        instruction = (
            _SYSTEM_PROMPT
            + "\n---\nTask input:\n```\n"
            + str(inputs)
            + "\n```\n"
        )

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON — try multiple strategies
        prediction = "None"
        try:
            raw_text = msg_history[-1]["text"]
            extracted = _best_extract(raw_text)
            if extracted and "response" in extracted[-1]:
                prediction = extracted[-1]["response"]
                self.log_fn(f"Extracted prediction: {prediction!r}")
            else:
                self.log_fn(
                    "Warning: could not find 'response' key in extracted JSON. "
                    f"Raw tail: {raw_text[-300:]!r}"
                )
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
