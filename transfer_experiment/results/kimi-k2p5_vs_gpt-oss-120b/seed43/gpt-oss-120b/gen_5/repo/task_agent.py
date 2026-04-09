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
    """Extract JSON objects from a response string.
    Supports <json>...</json> blocks and any JSON object via regex.
    Returns a list of parsed JSON dictionaries, or None if none found.
    """
    
    # Attempt to extract JSON blocks first; if none found, fallback to any JSON object via regex.
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
    # If no <json> blocks were found, try to find any JSON object using a regex fallback.
    if not results:
        json_candidates = re.findall(r'\{[^{}]*\}', text, flags=re.DOTALL)
        for cand in json_candidates:
            try:
                results.append(json.loads(cand))
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
            if msg_history:
                extracted = _extract_jsons(msg_history[-1]["text"])
            else:
                extracted = None
            if extracted and "response" in extracted[-1]:
                prediction = extracted[-1]["response"]
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        # return will be after fallback handling

        # If extraction failed, use the raw response as prediction
        if prediction == "None":
            # Use the last message text as fallback
            if msg_history:
                prediction = msg_history[-1]["text"].strip()
            else:
                prediction = ""
        return str(prediction), msg_history

