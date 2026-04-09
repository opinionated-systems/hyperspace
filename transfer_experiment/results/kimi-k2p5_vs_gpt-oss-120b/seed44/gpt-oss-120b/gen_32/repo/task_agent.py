"""
# Version: 1.0-modified
# Modified by MetaAgent

Task agent: solves a given task with a single LLM call.
# Version: 1.0-modified

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

# Import LLM client with fallback for testing environments
try:
    from agent.llm_client import get_response_from_llm, EVAL_MODEL
except ImportError:
    # Fallback dummy implementations for testing
    def get_response_from_llm(msg, model, msg_history=None):
        dummy_response = "<json>{\"response\": \"correct\"}</json>"
        return dummy_response, [{"text": dummy_response}], {}
    EVAL_MODEL = "dummy-model"

logger = logging.getLogger(__name__)


VERSION = "1.0-modified"


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks or raw JSON.
    
    First tries the original <json> tag method. If none found, falls back to
    searching for the first JSON object in the text using a simple brace
    matching approach. This makes the task agent more robust to variations in
    LLM output.
    """
    # First, try extracting JSON inside <json>...</json> tags.
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
    # Fallback: look for the first JSON object in the text.
    # Simple brace matching to find a balanced {...} block.
    brace_start = text.find('{')
    if brace_start == -1:
        return None
    depth = 0
    for i in range(brace_start, len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                try:
                    return [json.loads(text[brace_start:i+1])]
                except json.JSONDecodeError:
                    pass
    # If no JSON found, return None
    return None




def _sanitize_inputs(inputs: dict) -> dict:
    """Trim whitespace from string values in inputs.
    Returns a new dict with cleaned values.
    """
    cleaned = {}
    for k, v in inputs.items():
        if isinstance(v, str):
            cleaned[k] = v.strip()
        else:
            cleaned[k] = v
    return cleaned



def _missing_keys(inputs: dict) -> set:
    """Return a set of required keys that are missing from inputs."""
    required = {"domain", "problem", "solution", "grading_guidelines", "student_answer"}
    return required - inputs.keys()

def _validate_inputs(inputs: dict) -> bool:
    """Validate inputs and log missing keys using helper function."""
    missing = _missing_keys(inputs)
    if missing:
        logger.info(f"Missing keys in inputs: {missing}")
        return False
    return True



class TaskAgent:
    """Task agent that solves IMO grading problems."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model

        # Helper function moved to class method _get_last_text

        self.log_fn = logger.info


    def _get_last_text(self, msg_hist):
        """Return the most recent text or content from msg_history list of dicts."""
        for entry in reversed(msg_hist):
            if isinstance(entry, dict):
                if "text" in entry:
                    return entry["text"]
                if "content" in entry:
                    return entry["content"]
        return None

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        inputs = _sanitize_inputs(inputs)
        # utils usage removed
        if not _validate_inputs(inputs):
            return "Invalid inputs", []

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

        # Extract prediction from JSON with safety check using helper
        prediction = "None"
        try:
            last_text = self._get_last_text(msg_history)
            if last_text:
                extracted = _extract_jsons(last_text)
                if extracted:
                    for obj in extracted:
                        if isinstance(obj, dict) and "response" in obj:
                            prediction = obj["response"]
                            break
            else:
                self.log_fn("msg_history is empty or missing text/content field; cannot extract prediction.")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        # Fallback: if no JSON extracted, use raw response
        if prediction == "None":
            prediction = response
        return str(prediction), msg_history

# Modified by MetaAgent during iteration
