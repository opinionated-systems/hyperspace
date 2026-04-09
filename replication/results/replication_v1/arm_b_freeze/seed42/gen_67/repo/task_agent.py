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

    Args:
        text: The input text containing JSON blocks.

    Returns:
        A list of parsed JSON objects, or None if no valid JSON found.
    """
    results = []
    search_from = 0
    while True:
        start = text.find("<json>", search_from)
        if start == -1:
            break
        end = text.find("</json>", start)
        if end == -1:
            logger.warning("Found opening <json> tag but no closing </json> tag")
            break
        inner = text[start + 6:end].strip()
        search_from = end + 7
        try:
            parsed = json.loads(inner)
            if isinstance(parsed, dict):
                results.append(parsed)
            else:
                logger.warning(f"Parsed JSON is not a dict: {type(parsed)}")
        except json.JSONDecodeError as e:
            logger.debug(f"Failed to parse JSON: {e}")
            continue
    return results or None


def _format_inputs(inputs: dict) -> str:
    """Format task inputs into a structured prompt.

    Args:
        inputs: Dictionary containing task inputs.

    Returns:
        Formatted string representation of inputs.
    """
    lines = []
    for key, value in inputs.items():
        lines.append(f"{key}: {value}")
    return "\n".join(lines)


class TaskAgent:
    """Task agent that solves IMO grading problems.

    This agent uses a single LLM call to process task inputs and
    extract a structured response in JSON format.
    """

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        """Initialize the TaskAgent.

        Args:
            model: The LLM model to use for inference.
            log_file: Optional log file path (currently unused, kept for compatibility).
        """
        self.model = model
        self.log_fn = logger.info
        self.call_count = 0

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history) where prediction is the extracted response
            and msg_history contains the conversation history.
        """
        self.call_count += 1
        formatted_inputs = _format_inputs(inputs)

        instruction = f"""You are an expert grading agent. Your task is to evaluate student answers based on the provided problem, solution, and grading guidelines.

Task inputs:
{formatted_inputs}

Respond in JSON format with the following schema:
<json>
{{
    "response": "Your evaluation/grade here"
}}
</json>

Ensure your response is valid JSON within the <json> tags."""

        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return "Error: LLM call failed", []

        # Extract prediction from JSON
        prediction = "None"
        try:
            if msg_history and len(msg_history) > 0:
                extracted = _extract_jsons(msg_history[-1]["text"])
                if extracted:
                    last_json = extracted[-1]
                    if "response" in last_json:
                        prediction = last_json["response"]
                        logger.debug(f"Successfully extracted prediction: {prediction}")
                    else:
                        logger.warning("JSON extracted but 'response' key not found")
                        prediction = str(last_json)
                else:
                    logger.warning("No valid JSON found in response")
            else:
                logger.warning("Empty message history")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history

    def get_stats(self) -> dict[str, Any]:
        """Return agent statistics.

        Returns:
            Dictionary containing agent statistics.
        """
        return {
            "call_count": self.call_count,
            "model": self.model,
        }
