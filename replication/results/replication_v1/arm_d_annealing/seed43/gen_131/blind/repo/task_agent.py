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
    Also attempts to extract JSON from markdown code blocks as fallback.
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
            # Try to extract JSON from markdown code blocks within
            code_block_start = inner.find("```json")
            if code_block_start != -1:
                code_block_end = inner.find("```", code_block_start + 7)
                if code_block_end != -1:
                    try:
                        results.append(json.loads(inner[code_block_start + 7:code_block_end].strip()))
                    except json.JSONDecodeError:
                        continue
            continue
    
    # Fallback: try to find JSON in markdown code blocks if no <json> tags found
    if not results:
        search_from = 0
        while True:
            start = text.find("```json", search_from)
            if start == -1:
                break
            end = text.find("```", start + 7)
            if end == -1:
                break
            inner = text[start + 7:end].strip()
            search_from = end + 3
            try:
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
            last_message = msg_history[-1]["text"]
            extracted = _extract_jsons(last_message)
            if extracted:
                last_json = extracted[-1]
                if "response" in last_json:
                    prediction = last_json["response"]
                    self.log_fn(f"Successfully extracted prediction: {prediction}")
                else:
                    self.log_fn(f"Warning: JSON found but 'response' key missing. Keys: {list(last_json.keys())}")
            else:
                self.log_fn(f"Warning: No valid JSON found in response. Response preview: {last_message[:200]}...")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            import traceback
            self.log_fn(f"Traceback: {traceback.format_exc()}")

        return str(prediction), msg_history
