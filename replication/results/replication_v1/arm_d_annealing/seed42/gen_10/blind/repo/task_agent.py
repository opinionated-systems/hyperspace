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
    Also attempts to parse raw JSON objects if tags are not found.
    """
    results = []
    search_from = 0
    
    # First try to find <json>...</json> blocks
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
            logger.debug(f"Failed to parse JSON block: {e}")
            continue
    
    # If no tagged blocks found, try to find raw JSON objects
    if not results:
        # Try to find JSON objects between curly braces
        try:
            # Look for JSON-like patterns
            json_pattern = re.search(r'\{.*"response".*\}', text, re.DOTALL)
            if json_pattern:
                potential_json = json_pattern.group()
                results.append(json.loads(potential_json))
        except (json.JSONDecodeError, AttributeError) as e:
            logger.debug(f"Failed to parse raw JSON: {e}")
    
    return results or None


def _sanitize_prediction(prediction: Any) -> str:
    """Sanitize the prediction value to ensure it's a valid string.
    
    Args:
        prediction: The raw prediction value (could be any type)
        
    Returns:
        A sanitized string representation
    """
    if prediction is None:
        return "None"
    if isinstance(prediction, (dict, list)):
        return json.dumps(prediction)
    return str(prediction)


class TaskAgent:
    """Task agent that solves IMO grading problems."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.log_file = log_file

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Build a more structured prompt with clearer instructions
        instruction = f"""You are an expert grading agent. Your task is to evaluate student answers based on the provided problem, solution, and grading guidelines.

Task Input:
```json
{json.dumps(inputs, indent=2)}
```

Instructions:
1. Carefully read the problem, solution, and grading guidelines
2. Evaluate the student's answer against the criteria
3. Provide your evaluation in the exact JSON format below

Respond ONLY in JSON format with the following schema:
<json>
{{
    "response": "Your detailed evaluation here"
}}
</json>

Important: Ensure your response is valid JSON and properly enclosed in <json> tags."""

        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
        except Exception as e:
            self.log_fn(f"Error getting LLM response: {e}")
            return "Error: LLM call failed", []

        # Extract prediction from JSON with improved error handling
        prediction = "None"
        try:
            if msg_history and len(msg_history) > 0:
                last_message = msg_history[-1].get("text", "")
                extracted = _extract_jsons(last_message)
                if extracted:
                    last_extracted = extracted[-1]
                    if isinstance(last_extracted, dict) and "response" in last_extracted:
                        prediction = last_extracted["response"]
                        self.log_fn(f"Successfully extracted prediction: {prediction[:100]}...")
                    else:
                        self.log_fn(f"Extracted JSON missing 'response' key: {last_extracted}")
                else:
                    self.log_fn("No JSON blocks found in response")
            else:
                self.log_fn("Empty message history")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            # Log the full message history for debugging
            if msg_history:
                self.log_fn(f"Last message content: {msg_history[-1]}")

        return _sanitize_prediction(prediction), msg_history
