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

# Maximum retries for transient failures
MAX_RETRIES = 3
RETRY_DELAY_BASE = 2  # Base delay in seconds for exponential backoff

# Required input fields for grading tasks
REQUIRED_INPUT_FIELDS = {"domain", "problem", "solution", "grading_guidelines", "student_answer"}


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


def _extract_json_fallback(text: str) -> dict | None:
    """Fallback JSON extraction using regex patterns for common formats.
    
    Attempts to extract JSON from code blocks or raw JSON objects.
    """
    # Try to extract from markdown code blocks
    code_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    matches = re.findall(code_block_pattern, text, re.DOTALL)
    for match in matches:
        try:
            return json.loads(match.strip())
        except json.JSONDecodeError:
            continue
    
    # Try to find raw JSON objects
    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.findall(json_pattern, text, re.DOTALL)
    for match in matches:
        try:
            parsed = json.loads(match)
            if "response" in parsed:
                return parsed
        except json.JSONDecodeError:
            continue
    
    return None


def _validate_inputs(inputs: dict) -> tuple[bool, str]:
    """Validate that all required input fields are present.
    
    Returns:
        (is_valid, error_message)
    """
    if not isinstance(inputs, dict):
        return False, f"Expected dict, got {type(inputs).__name__}"
    
    missing = REQUIRED_INPUT_FIELDS - set(inputs.keys())
    if missing:
        return False, f"Missing required fields: {sorted(missing)}"
    
    # Check for empty values
    empty_fields = [k for k in REQUIRED_INPUT_FIELDS if not str(inputs.get(k, "")).strip()]
    if empty_fields:
        return False, f"Empty required fields: {sorted(empty_fields)}"
    
    return True, ""


class TaskAgent:
    """Task agent that solves IMO grading problems with enhanced extraction."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.stats = {
            "total_calls": 0, 
            "json_extracted": 0, 
            "fallback_used": 0, 
            "failed": 0,
            "validation_errors": 0,
            "retries": 0,
            "success": 0
        }
        self._last_inputs: dict | None = None  # For retry context

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        self.stats["total_calls"] += 1
        self._last_inputs = inputs
        
        # Validate inputs
        is_valid, error_msg = _validate_inputs(inputs)
        if not is_valid:
            self.log_fn(f"Input validation failed: {error_msg}")
            self.stats["validation_errors"] += 1
            return f"Error: {error_msg}", []
        
        # Sanitize inputs for JSON serialization
        try:
            sanitized_inputs = json.loads(json.dumps(inputs, default=str))
        except (TypeError, ValueError) as e:
            self.log_fn(f"Input sanitization failed: {e}")
            self.stats["validation_errors"] += 1
            return f"Error: Invalid input data - {e}", []
        
        instruction = f"""You are an expert grading agent. Analyze the student answer carefully.

Task input:
```
{json.dumps(sanitized_inputs, indent=2)}
```

Respond ONLY in JSON format with the following schema:
<json>
{{
    "response": "Your detailed evaluation here"
}}
</json>"""

        # Retry loop for transient failures
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
                self.log_fn(f"LLM call failed (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
                if attempt < MAX_RETRIES - 1:
                    self.stats["retries"] += 1
                    delay = RETRY_DELAY_BASE * (2 ** attempt)  # Exponential backoff
                    self.log_fn(f"Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    self.log_fn(f"All {MAX_RETRIES} attempts failed")
                    self.stats["failed"] += 1
                    return f"Error: LLM call failed after {MAX_RETRIES} attempts - {last_error}", []

        # Extract prediction from JSON
        prediction = "None"
        extraction_method = "none"
        
        try:
            if msg_history and len(msg_history) > 0:
                last_message = msg_history[-1].get("text", "")
                
                if not last_message or not last_message.strip():
                    self.log_fn("Empty response from LLM")
                    self.stats["failed"] += 1
                    return "Error: Empty response from LLM", msg_history
                
                # Primary extraction method
                extracted = _extract_jsons(last_message)
                if extracted:
                    # Try to find response in any of the extracted JSONs
                    for item in reversed(extracted):
                        if isinstance(item, dict) and "response" in item:
                            prediction = item["response"]
                            extraction_method = "primary"
                            self.stats["json_extracted"] += 1
                            break
                
                if extraction_method == "none":
                    # Fallback extraction
                    fallback = _extract_json_fallback(last_message)
                    if fallback and isinstance(fallback, dict) and "response" in fallback:
                        prediction = fallback["response"]
                        extraction_method = "fallback"
                        self.stats["fallback_used"] += 1
                        self.log_fn(f"Used fallback extraction for response")
                
                if extraction_method == "none":
                    # Last resort: use raw text (cleaned)
                    cleaned_text = last_message.strip()
                    # Remove common markdown artifacts
                    cleaned_text = re.sub(r'^```[\w]*\n?', '', cleaned_text)
                    cleaned_text = re.sub(r'\n?```$', '', cleaned_text)
                    prediction = cleaned_text[:1000]  # Limit length but allow more context
                    extraction_method = "raw"
                    self.log_fn(f"Using raw text extraction (limited to 1000 chars)")
                        
                self.log_fn(f"Extraction method: {extraction_method}, prediction length: {len(str(prediction))}")
                
                # Validate prediction is not empty
                if not str(prediction).strip():
                    self.log_fn("Extracted prediction is empty")
                    self.stats["failed"] += 1
                    return "Error: Empty prediction extracted", msg_history
                
                # Success!
                self.stats["success"] += 1
                    
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            self.stats["failed"] += 1
            return f"Error: Extraction failed - {e}", msg_history

        return str(prediction), msg_history
    
    def get_stats(self) -> dict[str, Any]:
        """Return extraction statistics."""
        return self.stats.copy()
