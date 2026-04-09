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
        except json.JSONDecodeError as e:
            logger.debug(f"JSON decode error: {e}, content: {inner[:100]}...")
            continue
    return results or None


def _extract_json_fuzzy(text: str) -> list[dict] | None:
    """Fuzzy extraction: try to find JSON objects even without <json> tags."""
    results = []
    # Try to find JSON-like structures with curly braces
    depth = 0
    start_idx = None
    for i, char in enumerate(text):
        if char == '{':
            if depth == 0:
                start_idx = i
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0 and start_idx is not None:
                try:
                    candidate = text[start_idx:i+1]
                    results.append(json.loads(candidate))
                except json.JSONDecodeError:
                    pass
                start_idx = None
    return results or None


class TaskAgent:
    """Task agent that solves IMO grading problems."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.max_retries = 2  # Number of retries for invalid JSON responses

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

        msg_history = []
        prediction = "None"
        
        for attempt in range(self.max_retries + 1):
            response, msg_history, info = get_response_from_llm(
                msg=instruction if attempt == 0 else f"{instruction}\n\nYour previous response was not valid JSON. Please respond with valid JSON only.",
                model=self.model,
                msg_history=msg_history if attempt > 0 else [],
            )

            # Extract prediction from JSON
            try:
                extracted = _extract_jsons(msg_history[-1]["text"])
                if not extracted:
                    # Try fuzzy extraction as fallback
                    extracted = _extract_json_fuzzy(msg_history[-1]["text"])
                
                if extracted and "response" in extracted[-1]:
                    prediction = extracted[-1]["response"]
                    logger.info(f"Successfully extracted prediction on attempt {attempt + 1}")
                    break
                else:
                    logger.warning(f"No valid 'response' field found on attempt {attempt + 1}")
                    if attempt < self.max_retries:
                        logger.info("Retrying with clarification prompt...")
            except Exception as e:
                logger.error(f"Error extracting prediction on attempt {attempt + 1}: {e}")
                if attempt < self.max_retries:
                    logger.info("Retrying...")
        else:
            logger.error(f"Failed to extract valid JSON after {self.max_retries + 1} attempts")

        return str(prediction), msg_history
