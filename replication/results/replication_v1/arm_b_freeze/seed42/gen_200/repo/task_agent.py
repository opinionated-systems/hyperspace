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
        except json.JSONDecodeError as e:
            logger.debug(f"JSON decode error at position {start}: {e}")
            continue
    return results or None


def _extract_json_fuzzy(text: str) -> list[dict] | None:
    """Fuzzy JSON extraction as fallback when strict parsing fails.
    
    Attempts to find JSON-like structures even without proper tags.
    """
    results = []
    # Try to find JSON objects between curly braces
    brace_count = 0
    start_idx = None
    for i, char in enumerate(text):
        if char == '{':
            if brace_count == 0:
                start_idx = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start_idx is not None:
                try:
                    candidate = text[start_idx:i+1]
                    results.append(json.loads(candidate))
                except json.JSONDecodeError:
                    pass
                start_idx = None
    return results or None


def _validate_grading_response(response: Any) -> tuple[bool, str]:
    """Validate that the grading response is well-formed.
    
    Returns (is_valid, error_message).
    """
    if response is None:
        return False, "Response is None"
    
    if isinstance(response, dict):
        # Check for required fields in a grading response
        if "grade" in response or "score" in response or "feedback" in response:
            return True, ""
        return True, ""  # Dict without specific fields is still valid JSON
    
    if isinstance(response, (str, int, float, bool)):
        return True, ""
    
    if isinstance(response, list):
        return True, ""
    
    return False, f"Unsupported response type: {type(response)}"


class TaskAgent:
    """Task agent that solves IMO grading problems with enhanced validation."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.max_retries = 3  # Increased from 2 for better reliability
        self.debug_mode = False

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Build a more structured prompt with clearer instructions
        instruction = self._build_grading_prompt(inputs)

        msg_history: list[dict] = []
        prediction = "None"
        last_error = ""
        
        for attempt in range(self.max_retries + 1):
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=msg_history if attempt > 0 else [],
                )

                # Extract prediction from JSON
                last_text = msg_history[-1]["text"]
                extracted = _extract_jsons(last_text)
                
                # Fallback to fuzzy extraction if strict parsing fails
                if extracted is None:
                    extracted = _extract_json_fuzzy(last_text)
                    if extracted:
                        self.log_fn(f"[Attempt {attempt + 1}] Used fuzzy JSON extraction")
                
                if extracted and "response" in extracted[-1]:
                    prediction = extracted[-1]["response"]
                    
                    # Validate the response
                    is_valid, error_msg = _validate_grading_response(prediction)
                    if is_valid:
                        self.log_fn(f"[Attempt {attempt + 1}] Successfully extracted valid response")
                        break
                    else:
                        last_error = f"Invalid response format: {error_msg}"
                        self.log_fn(f"[Attempt {attempt + 1}] {last_error}")
                else:
                    last_error = "No valid 'response' field found in JSON"
                    self.log_fn(f"[Attempt {attempt + 1}] {last_error}")
                    
            except Exception as e:
                last_error = f"Error extracting prediction: {str(e)}"
                self.log_fn(f"[Attempt {attempt + 1}] {last_error}")
                
            # Add retry instruction for next attempt with more context
            if attempt < self.max_retries:
                instruction = self._build_retry_prompt(last_error, attempt + 1)

        return str(prediction), msg_history
    
    def _build_grading_prompt(self, inputs: dict) -> str:
        """Build a structured grading prompt with clear instructions."""
        return f"""You are an expert grading agent for mathematical olympiad problems.

Your task is to evaluate a student's answer against the official solution and grading guidelines.

## Task Input:
```json
{json.dumps(inputs, indent=2, default=str)}
```

## Instructions:
1. Carefully read the problem, official solution, and grading guidelines
2. Compare the student's answer against the solution
3. Provide your grading decision in the exact JSON format below

## Response Format (REQUIRED):
You MUST respond with valid JSON wrapped in <json> tags:

<json>
{{
    "response": <your grading decision - can be a grade, score, or detailed feedback>
}}
</json>

The "response" field should contain your final grading decision. This can be:
- A numeric score (e.g., 7, 5, 0)
- A grade label (e.g., "Correct", "Partial", "Incorrect")  
- Detailed feedback about the student's work

Do not include any text outside the JSON tags."""
    
    def _build_retry_prompt(self, last_error: str, attempt_num: int) -> str:
        """Build a retry prompt with context about the previous failure."""
        return f"""Your previous response could not be processed. 

Error: {last_error}

Please respond again with valid JSON in the exact format:

<json>
{{
    "response": <your grading decision>
}}
</json>

Make sure:
1. The JSON is valid and properly formatted
2. The response is wrapped in <json>...</json> tags
3. The "response" field contains your grading decision

This is attempt {attempt_num} of {self.max_retries + 1}."""
