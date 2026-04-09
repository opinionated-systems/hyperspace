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


class TaskAgentError(Exception):
    """Custom exception for TaskAgent errors."""
    pass


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
            logger.debug(f"JSON decode error: {e}")
            continue
    return results or None


def _validate_inputs(inputs: dict) -> None:
    """Validate that required fields are present in inputs.
    
    Args:
        inputs: Dictionary containing task inputs
        
    Raises:
        TaskAgentError: If required fields are missing
    """
    required_fields = ["domain", "problem", "solution", "grading_guidelines", "student_answer"]
    missing = [field for field in required_fields if field not in inputs]
    if missing:
        raise TaskAgentError(f"Missing required fields: {missing}")


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
            
        Raises:
            TaskAgentError: If inputs are invalid or LLM call fails
        """
        # Validate inputs
        try:
            _validate_inputs(inputs)
        except TaskAgentError as e:
            logger.error(f"Input validation failed: {e}")
            raise

        self.call_count += 1
        logger.info(f"TaskAgent call #{self.call_count} with model={self.model}")

        instruction = f"""You are an expert grading agent for mathematical olympiad problems.

Your task is to evaluate a student's answer based on the provided problem, solution, and grading guidelines.

Task input:
```
{json.dumps(inputs, indent=2)}
```

Respond in JSON format with the following schema:
<json>
{{
    "response": "Your detailed evaluation and grade here"
}}
</json>"""

        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise TaskAgentError(f"Failed to get response from LLM: {e}") from e

        # Extract prediction from JSON
        prediction = "None"
        try:
            if msg_history and len(msg_history) > 0:
                extracted = _extract_jsons(msg_history[-1]["text"])
                if extracted and len(extracted) > 0 and "response" in extracted[-1]:
                    prediction = extracted[-1]["response"]
                    logger.info(f"Successfully extracted prediction (length={len(str(prediction))})")
                else:
                    logger.warning("No valid JSON response found in LLM output")
            else:
                logger.warning("Empty message history from LLM")
        except Exception as e:
            logger.error(f"Error extracting prediction: {e}")

        return str(prediction), msg_history

    def get_stats(self) -> dict[str, Any]:
        """Get agent statistics.
        
        Returns:
            Dictionary with agent statistics
        """
        return {
            "model": self.model,
            "call_count": self.call_count,
        }

    def forward_batch(self, inputs_list: list[dict]) -> list[tuple[str, list[dict]]]:
        """Process multiple grading tasks in batch.
        
        Args:
            inputs_list: List of input dictionaries, each containing
                        domain, problem, solution, grading_guidelines, student_answer
                        
        Returns:
            List of (prediction, msg_history) tuples, one for each input.
            If an individual task fails, returns ("ERROR: <message>", []) for that item.
        """
        results = []
        logger.info(f"Starting batch processing of {len(inputs_list)} tasks")
        
        for i, inputs in enumerate(inputs_list):
            logger.info(f"Processing batch item {i+1}/{len(inputs_list)}")
            try:
                prediction, msg_history = self.forward(inputs)
                results.append((prediction, msg_history))
                logger.info(f"Batch item {i+1} completed successfully")
            except TaskAgentError as e:
                logger.error(f"Batch item {i+1} failed: {e}")
                results.append((f"ERROR: {e}", []))
            except Exception as e:
                logger.error(f"Batch item {i+1} unexpected error: {e}")
                results.append((f"ERROR: Unexpected error - {e}", []))
        
        logger.info(f"Batch processing complete. Success: {sum(1 for r in results if not r[0].startswith('ERROR'))}/{len(inputs_list)}")
        return results

    def forward_with_retry(self, inputs: dict, max_retries: int = 3) -> tuple[str, list[dict]]:
        """Run the task agent with automatic retry on failure.
        
        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer
            max_retries: Maximum number of retry attempts (default: 3)
            
        Returns:
            (prediction, msg_history)
            
        Raises:
            TaskAgentError: If all retry attempts fail
        """
        last_error = None
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Attempt {attempt + 1}/{max_retries}")
                return self.forward(inputs)
            except TaskAgentError as e:
                last_error = e
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    import time
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
        
        raise TaskAgentError(f"Failed after {max_retries} attempts. Last error: {last_error}")
