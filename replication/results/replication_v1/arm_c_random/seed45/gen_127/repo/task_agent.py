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
from agent.utils import truncate_text, safe_get

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY_BASE = 2  # seconds

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
            logger.debug(f"JSON decode error: {e}")
            continue
    return results or None


def _extract_response_fallback(text: str) -> str | None:
    """Fallback extraction for responses without proper JSON tags.
    
    Tries to find JSON-like structures or key-value patterns.
    """
    # Try to find JSON in code blocks
    code_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    matches = re.findall(code_block_pattern, text, re.DOTALL)
    for match in matches:
        try:
            data = json.loads(match.strip())
            if isinstance(data, dict) and "response" in data:
                return str(data["response"])
        except json.JSONDecodeError:
            continue
    
    # Try to find "response": "value" pattern
    response_pattern = r'"response"\s*:\s*"([^"]*)"'
    match = re.search(response_pattern, text)
    if match:
        return match.group(1)
    
    return None


class TaskAgent:
    """Task agent that solves IMO grading problems."""

    # Required input fields for grading tasks
    REQUIRED_FIELDS = ["domain", "problem", "solution", "grading_guidelines", "student_answer"]

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self._call_count = 0
        self._success_count = 0
        self._error_count = 0
        self._total_latency = 0.0  # Track total time for performance monitoring
        self._last_latency = 0.0

    def get_stats(self) -> dict:
        """Get agent statistics.
        
        Returns:
            Dict with call_count, success_count, error_count, success_rate,
            avg_latency_ms, and last_latency_ms
        """
        total = self._success_count + self._error_count
        success_rate = self._success_count / total if total > 0 else 0.0
        avg_latency = (self._total_latency / self._call_count * 1000) if self._call_count > 0 else 0.0
        return {
            "call_count": self._call_count,
            "success_count": self._success_count,
            "error_count": self._error_count,
            "success_rate": round(success_rate, 3),
            "avg_latency_ms": round(avg_latency, 2),
            "last_latency_ms": round(self._last_latency * 1000, 2),
        }

    def _validate_inputs(self, inputs: dict) -> tuple[bool, str]:
        """Validate that all required fields are present.
        
        Returns:
            (is_valid, error_message)
        """
        if not isinstance(inputs, dict):
            return False, f"Expected dict, got {type(inputs).__name__}"
        
        missing = [field for field in self.REQUIRED_FIELDS if field not in inputs]
        if missing:
            return False, f"Missing required fields: {missing}"
        
        # Check for empty values
        empty = [field for field in self.REQUIRED_FIELDS if not inputs.get(field)]
        if empty:
            return False, f"Empty values for fields: {empty}"
        
        return True, ""

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        self._call_count += 1
        
        # Validate inputs first
        is_valid, error_msg = self._validate_inputs(inputs)
        if not is_valid:
            logger.error(f"Input validation failed: {error_msg}")
            self._error_count += 1
            return f"ERROR_INVALID_INPUT: {error_msg}", [{"role": "system", "text": f"Validation error: {error_msg}"}]
        
        # Build structured instruction with better formatting
        instruction = self._build_instruction(inputs)
        
        self.log_fn(f"TaskAgent call #{self._call_count}: processing {inputs.get('domain', 'unknown')} problem")

        # Try LLM call with retries
        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=[],
                )
                break  # Success, exit retry loop
            except Exception as e:
                last_error = e
                logger.warning(f"LLM call failed (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
                if attempt < MAX_RETRIES - 1:
                    wait = RETRY_DELAY_BASE * (2 ** attempt)  # Exponential backoff
                    time.sleep(wait)
        else:
            # All retries exhausted
            logger.error(f"LLM call failed after {MAX_RETRIES} attempts: {last_error}")
            self._error_count += 1
            return "ERROR_LLM_CALL_FAILED", [{"role": "system", "text": f"Error: {last_error}"}]

        # Extract prediction from JSON
        prediction = self._extract_prediction(msg_history)
        
        # Track success/error
        if prediction.startswith("ERROR_") or prediction == "None":
            self._error_count += 1
        else:
            self._success_count += 1
        
        # Log usage info if available
        usage = safe_get(info, "usage", default={})
        if usage:
            self.log_fn(f"Usage: {usage}")

        return prediction, msg_history
    
    def _build_instruction(self, inputs: dict) -> str:
        """Build the instruction prompt for the LLM."""
        # Format inputs nicely
        formatted_inputs = json.dumps(inputs, indent=2, ensure_ascii=False)
        
        return f"""You are an expert grading agent. Your task is to evaluate a student's answer based on the provided problem, solution, and grading guidelines.

Task input:
```json
{formatted_inputs}
```

Instructions:
1. Carefully read the problem, solution, and grading guidelines
2. Evaluate the student's answer against the criteria
3. Provide your evaluation in the exact JSON format below

Respond in JSON format with the following schema:
<json>
{{
    "response": "Your detailed evaluation here"
}}
</json>

Important: Your response must be valid JSON inside the <json> tags."""
    
    def _extract_prediction(self, msg_history: list[dict]) -> str:
        """Extract prediction from message history."""
        if not msg_history:
            logger.warning("Empty message history")
            return "None"
        
        last_message = msg_history[-1].get("text", "")
        
        # Primary extraction method
        try:
            extracted = _extract_jsons(last_message)
            if extracted:
                response = extracted[-1].get("response")
                if response is not None:
                    return str(response)
        except Exception as e:
            self.log_fn(f"Primary extraction failed: {e}")
        
        # Fallback extraction
        fallback = _extract_response_fallback(last_message)
        if fallback is not None:
            self.log_fn(f"Used fallback extraction: {truncate_text(fallback)}")
            return fallback
        
        # Last resort: return the raw text if it's short enough
        if len(last_message) < 500:
            self.log_fn(f"Returning raw text as prediction: {truncate_text(last_message)}")
            return last_message
        
        logger.warning(f"Could not extract prediction from response: {truncate_text(last_message, 100)}")
        return "None"
