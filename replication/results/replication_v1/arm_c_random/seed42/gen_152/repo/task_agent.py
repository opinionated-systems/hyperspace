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
from typing import Any

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


def _extract_json_with_retry(text: str, max_retries: int = 2) -> list[dict] | None:
    """Extract JSON objects with retry logic for malformed JSON.
    
    Attempts to fix common JSON formatting issues before giving up.
    
    Args:
        text: The text containing JSON blocks
        max_retries: Maximum number of retry attempts with fixes
        
    Returns:
        List of parsed JSON objects or None if extraction fails
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
        
        # Try parsing with retries and fixes
        for attempt in range(max_retries + 1):
            try:
                results.append(json.loads(inner))
                break
            except json.JSONDecodeError as e:
                if attempt == max_retries:
                    logger.debug(f"Failed to parse JSON after {max_retries} retries: {e}")
                    break
                # Attempt to fix common issues
                if attempt == 0:
                    # Try fixing trailing commas
                    inner = re.sub(r',\s*}', '}', inner)
                    inner = re.sub(r',\s*]', ']', inner)
                elif attempt == 1:
                    # Try fixing single quotes
                    inner = inner.replace("'", '"')
                    
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

        # Extract prediction from JSON with retry logic
        prediction = "None"
        try:
            # First try the standard extraction
            extracted = _extract_jsons(msg_history[-1]["text"])
            if not extracted:
                # Fall back to retry-enabled extraction for malformed JSON
                extracted = _extract_json_with_retry(msg_history[-1]["text"])
            
            if extracted and "response" in extracted[-1]:
                prediction = extracted[-1]["response"]
            elif extracted:
                # If no "response" key, use the first available value
                prediction = list(extracted[-1].values())[0] if extracted[-1] else "None"
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
