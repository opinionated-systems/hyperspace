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
"""

from __future__ import annotations

import json
import logging
import re

from agent.llm_client import get_response_from_llm, EVAL_MODEL

logger = logging.getLogger(__name__)

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


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an expert mathematical grader evaluating student answers to
competition-style problems (e.g. IMO, USAMO, Putnam).

Your job:
1. Read the problem statement, the official solution, and the grading
   guidelines carefully.
2. Analyse the student's answer step by step, checking correctness,
   completeness, and alignment with the grading rubric.
3. Assign a score according to the grading guidelines.

Always reason through the problem before giving your final answer.
Wrap your final answer in a <json> block with the key "response".
The value of "response" must be exactly what the grading guidelines
specify (e.g. an integer score, "correct"/"incorrect", etc.).

Example output format:
<reasoning>
... your step-by-step analysis ...
</reasoning>
<json>
{"response": <your answer here>}
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
Domain: {domain}

Problem:
{problem}

Official solution:
{solution}

Grading guidelines:
{guidelines}

Student answer:
{student_answer}
---

Analyse the student's answer and respond with your final grade in the
<json> block as described above."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            temperature=self.temperature,
            msg_history=[],
        )

        # Extract prediction — try tagged blocks first, then fallback.
        prediction = "None"
        try:
            last_text = msg_history[-1]["text"]
            parsed = _best_json(last_text)
            if parsed is not None and "response" in parsed:
                prediction = parsed["response"]
            else:
                self.log_fn("No 'response' key found in extracted JSON; raw text: %s", last_text[:300])
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
