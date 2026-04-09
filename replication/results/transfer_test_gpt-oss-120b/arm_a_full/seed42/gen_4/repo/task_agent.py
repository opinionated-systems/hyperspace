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


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks.

    This function first attempts a simple tag‑based extraction using
    ``str.find`` to avoid the pitfalls of a lazy ``.*?`` regex when the
    JSON payload contains nested braces. If that fails to produce any
    results, it falls back to a more permissive regular‑expression based
    approach that captures the content between ``<json>`` and ``</json>``
    tags (including newlines). The fallback uses ``re.DOTALL`` to match
    across lines and attempts ``json.loads`` on each captured block.
    """
    results: list[dict] = []
    # Primary extraction using explicit tag indices (robust to nested braces)
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
            # Continue to next block; malformed JSON will be handled by fallback.
            continue
    # If primary method found any valid JSON, return it.
    if results:
        return results
    # Fallback: use regex to capture any content between tags (including newlines).
    pattern = re.compile(r"<json>\s*(\{.*?\})\s*</json>", re.DOTALL)
    for match in pattern.finditer(text):
        try:
            results.append(json.loads(match.group(1)))
        except json.JSONDecodeError:
            continue
    # Primary extraction completed; if any results, they will be returned later.
    if results:
        return results

    # Final fallback: simple regex to capture JSON objects (non‑nested).
    simple_pattern = re.compile(r"\{[^{}]*\}", re.DOTALL)
    for match in simple_pattern.finditer(text):
        try:
            results.append(json.loads(match.group()))
        except json.JSONDecodeError:
            continue
    return results or None


class TaskAgent:
    """Task agent that solves IMO grading problems."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        instruction = f"""You are an agent.

Task input:
```
{inputs}
```

Respond in JSON format with the following schema:
<json>
{{
    "response": ...
}}
</json>"""

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
                prediction = extracted[-1]["response"]
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
