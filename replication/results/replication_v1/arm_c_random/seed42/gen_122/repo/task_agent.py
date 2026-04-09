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


# Maximum number of retry attempts for JSON extraction
MAX_JSON_RETRIES = 3


def _validate_json_response(data: dict, required_keys: list[str]) -> tuple[bool, list[str]]:
    """Validate that a JSON response contains all required keys.
    
    Args:
        data: The parsed JSON dictionary
        required_keys: List of keys that must be present
        
    Returns:
        Tuple of (is_valid, missing_keys)
    """
    missing = [key for key in required_keys if key not in data]
    return len(missing) == 0, missing


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
            logger.debug(f"JSON decode error at position {start}: {e}")
            continue
    return results or None


def _extract_json_fallback(text: str) -> dict | None:
    """Fallback method to extract JSON using regex patterns.
    
    Used when primary extraction fails to find valid JSON.
    """
    # Try to find JSON in code blocks
    code_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    matches = re.findall(code_block_pattern, text, re.DOTALL)
    
    for match in matches:
        try:
            return json.loads(match.strip())
        except json.JSONDecodeError:
            continue
    
    # Try to find JSON between curly braces
    brace_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.findall(brace_pattern, text, re.DOTALL)
    
    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue
    
    return None


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
        instruction = f"""You are an expert grading agent. Your task is to evaluate student answers based on the provided problem, solution, and grading guidelines.

Task input:
```
{inputs}
```

Respond in JSON format with the following schema:
<json>
{{
    "response": "Your evaluation/grade here"
}}
</json>

Important: Ensure your response is valid JSON wrapped in <json> tags."""

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
                last_message = msg_history[-1].get("text", "")
                extracted = _extract_jsons(last_message)
                
                if extracted:
                    # Validate the extracted JSON has required keys
                    is_valid, missing = _validate_json_response(extracted[-1], ["response"])
                    if is_valid:
                        prediction = extracted[-1]["response"]
                        logger.info(f"Successfully extracted and validated prediction using primary method")
                    else:
                        logger.warning(f"Extracted JSON missing required keys: {missing}")
                        # Try fallback extraction
                        fallback = _extract_json_fallback(last_message)
                        if fallback:
                            is_valid, missing = _validate_json_response(fallback, ["response"])
                            if is_valid:
                                prediction = fallback["response"]
                                logger.info(f"Successfully extracted and validated prediction using fallback method")
                            else:
                                logger.warning(f"Fallback JSON missing required keys: {missing}")
                        else:
                            logger.warning(f"No valid JSON found in response")
                            self.log_fn(f"Raw response: {last_message[:500]}...")
                else:
                    # Try fallback extraction
                    fallback = _extract_json_fallback(last_message)
                    if fallback:
                        is_valid, missing = _validate_json_response(fallback, ["response"])
                        if is_valid:
                            prediction = fallback["response"]
                            logger.info(f"Successfully extracted and validated prediction using fallback method")
                        else:
                            logger.warning(f"Fallback JSON missing required keys: {missing}")
                    else:
                        logger.warning(f"No valid JSON found in response")
                        self.log_fn(f"Raw response: {last_message[:500]}...")
            else:
                logger.warning("Empty message history received")
        except Exception as e:
            logger.error(f"Error extracting prediction: {e}")
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
