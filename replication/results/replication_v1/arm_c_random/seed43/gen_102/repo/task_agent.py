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

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    Also handles markdown code blocks and common formatting issues.
    """
    results = []
    search_from = 0

    # First, try to find <json>...</json> blocks
    while True:
        start = text.find("<json>", search_from)
        if start == -1:
            break
        end = text.find("</json>", start)
        if end == -1:
            break
        inner = text[start + 6:end].strip()
        search_from = end + 7

        # Try to parse the JSON, handling common formatting issues
        try:
            # Remove markdown code block markers if present
            if inner.startswith("```json"):
                inner = inner[7:]
            elif inner.startswith("```"):
                inner = inner[3:]
            if inner.endswith("```"):
                inner = inner[:-3]
            inner = inner.strip()
            results.append(json.loads(inner))
        except json.JSONDecodeError:
            # Try to extract just the first valid JSON object if there's extra text
            try:
                # Find the first { and last }
                json_start = inner.find("{")
                json_end = inner.rfind("}")
                if json_start != -1 and json_end != -1 and json_end > json_start:
                    results.append(json.loads(inner[json_start:json_end + 1]))
            except json.JSONDecodeError:
                continue

    # If no <json> blocks found, try to find JSON in markdown code blocks
    if not results:
        search_from = 0
        while True:
            start = text.find("```json", search_from)
            if start == -1:
                # Try generic code blocks
                start = text.find("```", search_from)
                if start == -1:
                    break
                code_start = start + 3
            else:
                code_start = start + 7

            end = text.find("```", code_start)
            if end == -1:
                break

            inner = text[code_start:end].strip()
            search_from = end + 3

            try:
                # Try to parse as a JSON object
                if inner.startswith("{"):
                    results.append(json.loads(inner))
                # Try to parse as a JSON array
                elif inner.startswith("["):
                    results.append(json.loads(inner))
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
