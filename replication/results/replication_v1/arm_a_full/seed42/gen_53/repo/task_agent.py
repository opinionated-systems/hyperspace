"""
Task agent: solves a given task with a single LLM call.

Reimplemented from facebookresearch/HyperAgents task_agent.py.
Same interface, same JSON output format, same extraction logic.

This is the INITIAL task agent. The meta agent modifies this file
during self-improvement. The evaluation harness loads whatever
task_agent.py exists at the agent's repo path.

MODIFIED: Added enhanced error handling, detailed logging, and input validation.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from agent.llm_client import get_response_from_llm, EVAL_MODEL

# Compile regex patterns for JSON extraction fallback
_MD_JSON_PATTERN = re.compile(r'```(?:json)?\s*\n?(.*?)\n?```', re.DOTALL)
_RAW_JSON_PATTERN = re.compile(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', re.DOTALL)

logger = logging.getLogger(__name__)


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.

    Args:
        text: The text containing <json>...</json> blocks.

    Returns:
        A list of parsed JSON objects, or None if no valid JSON found.
    """
    results = []
    search_from = 0
    json_count = 0
    error_count = 0

    while True:
        start = text.find("<json>", search_from)
        if start == -1:
            break
        end = text.find("</json>", start)
        if end == -1:
            logger.warning("Found unclosed <json> tag at position %d", start)
            break

        inner = text[start + 6:end].strip()
        search_from = end + 7
        json_count += 1

        try:
            parsed = json.loads(inner)
            results.append(parsed)
            logger.debug("Successfully parsed JSON block %d", json_count)
        except json.JSONDecodeError as e:
            error_count += 1
            logger.warning("Failed to parse JSON block %d: %s", json_count, e)
            continue

    if error_count > 0:
        logger.info("JSON extraction: %d found, %d parsed, %d errors", json_count, len(results), error_count)

    return results or None


def _extract_jsons_with_fallback(text: str) -> list[dict] | None:
    """Extract JSON objects with multiple fallback strategies.

    First tries standard <json>...</json> extraction, then falls back to:
    1. Extracting JSON from markdown code blocks (```json...```)
    2. Extracting raw JSON objects from text
    3. Attempting to repair common JSON syntax errors

    Args:
        text: The text containing JSON data.

    Returns:
        A list of parsed JSON objects, or None if no valid JSON found.
    """
    # First try standard extraction
    results = _extract_jsons(text)
    if results:
        return results

    # Fallback 1: Try markdown code blocks
    for match in _MD_JSON_PATTERN.finditer(text):
        try:
            inner = match.group(1).strip()
            parsed = json.loads(inner)
            results.append(parsed)
            logger.debug("Extracted JSON from markdown block")
        except json.JSONDecodeError:
            continue

    if results:
        logger.info("Used markdown fallback for JSON extraction")
        return results

    # Fallback 2: Try to find raw JSON objects
    for match in _RAW_JSON_PATTERN.finditer(text):
        try:
            parsed = json.loads(match.group(0))
            results.append(parsed)
            logger.debug("Extracted raw JSON object")
        except json.JSONDecodeError:
            continue

    if results:
        logger.info("Used raw JSON fallback for extraction")

    return results or None


class TaskAgent:
    """Task agent that solves IMO grading problems.

    This agent uses an LLM to evaluate student answers against grading guidelines.
    It expects inputs containing domain, problem, solution, grading_guidelines, and student_answer.
    """

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.call_count = 0

    def _validate_inputs(self, inputs: dict) -> None:
        """Validate that required input fields are present.

        Args:
            inputs: The input dictionary to validate.

        Raises:
            ValueError: If required fields are missing.
        """
        required_fields = ["domain", "problem", "solution", "grading_guidelines", "student_answer"]
        missing = [f for f in required_fields if f not in inputs]
        if missing:
            raise ValueError(f"Missing required input fields: {missing}")

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        self.call_count += 1
        self.log_fn(f"TaskAgent call #{self.call_count} starting")

        # Validate inputs
        try:
            self._validate_inputs(inputs)
        except ValueError as e:
            self.log_fn(f"Input validation failed: {e}")
            raise

        instruction = f"""You are an expert grader for mathematics problems.

Your task is to evaluate a student's answer against the provided solution and grading guidelines.

Task input:
```
{inputs}
```

Respond in JSON format with the following schema:
<json>
{{
    "response": "Your evaluation/grade here"
}}
</json>"""

        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
        except Exception as e:
            self.log_fn(f"LLM call failed: {e}")
            return "Error: LLM call failed", []

        # Extract prediction from JSON using fallback extraction
        prediction = "None"
        try:
            if msg_history:
                extracted = _extract_jsons_with_fallback(msg_history[-1]["text"])
                if extracted and "response" in extracted[-1]:
                    prediction = extracted[-1]["response"]
                    self.log_fn(f"Successfully extracted prediction: {prediction[:100]}...")
                else:
                    self.log_fn("No valid 'response' field found in extracted JSON")
            else:
                self.log_fn("Empty message history received from LLM")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        self.log_fn(f"TaskAgent call #{self.call_count} completed")
        return str(prediction), msg_history
