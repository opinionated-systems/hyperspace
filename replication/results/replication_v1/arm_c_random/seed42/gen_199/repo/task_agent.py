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
import time
from typing import Any

from agent.llm_client import get_response_from_llm, EVAL_MODEL

logger = logging.getLogger(__name__)

# Configuration for retry mechanism
MAX_RETRIES = 3
RETRY_DELAY = 1.0  # seconds


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


def _validate_inputs(inputs: dict) -> tuple[bool, str]:
    """Validate that required fields are present in inputs.
    
    Returns:
        (is_valid, error_message)
    """
    required_fields = ["domain", "problem", "solution", "grading_guidelines", "student_answer"]
    missing = [f for f in required_fields if f not in inputs]
    if missing:
        return False, f"Missing required fields: {missing}"
    return True, ""


def _call_llm_with_retry(instruction: str, model: str, max_retries: int = MAX_RETRIES) -> tuple[str, list[dict], dict]:
    """Call LLM with retry mechanism for transient failures.
    
    Args:
        instruction: The prompt to send to the LLM
        model: The model identifier to use
        max_retries: Maximum number of retry attempts
        
    Returns:
        (response, msg_history, info) tuple from LLM
        
    Raises:
        Exception: If all retry attempts fail
    """
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=model,
                msg_history=[],
            )
            return response, msg_history, info
        except Exception as e:
            last_exception = e
            logger.warning(f"LLM call attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))  # Exponential backoff
    
    # All retries exhausted
    raise last_exception if last_exception else Exception("LLM call failed after all retries")


class TaskAgent:
    """Task agent that solves IMO grading problems."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.call_count = 0

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Validate inputs
        is_valid, error_msg = _validate_inputs(inputs)
        if not is_valid:
            self.log_fn(f"Input validation failed: {error_msg}")
            return f"Error: {error_msg}", []

        self.call_count += 1
        
        instruction = f"""You are an expert grading agent for mathematical problems.

Your task is to evaluate a student's answer based on the provided problem, solution, and grading guidelines.

Task input:
- Domain: {inputs.get('domain', 'N/A')}
- Problem: {inputs.get('problem', 'N/A')}
- Correct Solution: {inputs.get('solution', 'N/A')}
- Grading Guidelines: {inputs.get('grading_guidelines', 'N/A')}
- Student Answer: {inputs.get('student_answer', 'N/A')}

Carefully analyze the student's answer against the correct solution and grading guidelines.
Provide your evaluation in JSON format with the following schema:
<json>
{{
    "response": "Your detailed evaluation here"
}}
</json>"""

        try:
            response, msg_history, info = _call_llm_with_retry(instruction, self.model)
        except Exception as e:
            self.log_fn(f"LLM call failed after {MAX_RETRIES} retries: {e}")
            return f"Error: LLM call failed after {MAX_RETRIES} retries - {e}", []

        # Extract prediction from JSON
        prediction = "None"
        try:
            if msg_history and len(msg_history) > 0:
                extracted = _extract_jsons(msg_history[-1]["text"])
                if extracted and "response" in extracted[-1]:
                    prediction = extracted[-1]["response"]
                else:
                    self.log_fn("No valid JSON response found in LLM output")
            else:
                self.log_fn("Empty message history from LLM")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
