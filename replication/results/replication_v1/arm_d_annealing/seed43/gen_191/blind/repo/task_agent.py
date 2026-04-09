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

        # Extract prediction from JSON with enhanced error handling
        prediction = "None"
        try:
            if not msg_history:
                self.log_fn("Warning: Empty message history, cannot extract prediction")
            else:
                last_msg = msg_history[-1]
                text_content = last_msg.get("text") or last_msg.get("content", "")
                if not text_content:
                    self.log_fn("Warning: Last message has no text content")
                else:
                    extracted = _extract_jsons(text_content)
                    if extracted:
                        if "response" in extracted[-1]:
                            prediction = extracted[-1]["response"]
                            self.log_fn(f"Successfully extracted prediction: {prediction}")
                        else:
                            self.log_fn(f"Warning: JSON extracted but 'response' key missing. Keys: {list(extracted[-1].keys())}")
                    else:
                        self.log_fn(f"Warning: No valid JSON found in response. Raw text (first 200 chars): {text_content[:200]}")
        except json.JSONDecodeError as e:
            self.log_fn(f"Error: JSON decode error - {e}")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {type(e).__name__}: {e}")

        return str(prediction), msg_history
