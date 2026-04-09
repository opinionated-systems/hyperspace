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


def _extract_jsons(text: str) -> list[dict]:
    """Extract JSON objects from <json>...</json> blocks or raw JSON.
    
    First tries the original <json> tag method. If none found, falls back to
    searching for top‑level JSON objects using a regex that matches balanced
    braces. This makes the extractor more robust to variations in LLM output.
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
    if results:
        return results
    # Fallback: extract raw JSON objects
    raw_results = []
    i = 0
    n = len(text)
    while i < n:
        if text[i] == '{':
            depth = 1
            j = i + 1
            while j < n and depth > 0:
                if text[j] == '{':
                    depth += 1
                elif text[j] == '}':
                    depth -= 1
                j += 1
            if depth == 0:
                candidate = text[i:j]
                try:
                    raw_results.append(json.loads(candidate))
                except json.JSONDecodeError:
                    pass
                i = j
                continue
        i += 1
    return raw_results


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

        # Ensure msg_history has at least one entry
        if not msg_history:
            self.log_fn("msg_history is empty, returning default prediction.")
            return "None", []


        # Extract prediction from JSON
        prediction = "None"
        try:
            extracted = _extract_jsons(msg_history[-1]["text"])
            if extracted and "response" in extracted[-1]:
                prediction = extracted[-1]["response"]
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
