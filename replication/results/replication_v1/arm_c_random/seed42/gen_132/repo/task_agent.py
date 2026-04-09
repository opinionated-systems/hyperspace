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

# Required input fields for IMO grading task
_REQUIRED_FIELDS = {"domain", "problem", "solution", "grading_guidelines", "student_answer"}


def _validate_inputs(inputs: dict) -> tuple[bool, str]:
    """Validate that inputs contains all required fields.
    
    Returns:
        (is_valid, error_message)
    """
    if not isinstance(inputs, dict):
        return False, f"Expected dict, got {type(inputs).__name__}"
    
    missing = _REQUIRED_FIELDS - set(inputs.keys())
    if missing:
        return False, f"Missing required fields: {sorted(missing)}"
    
    # Check for empty values
    empty_fields = [k for k in _REQUIRED_FIELDS if not str(inputs.get(k, "")).strip()]
    if empty_fields:
        return False, f"Empty values for fields: {sorted(empty_fields)}"
    
    return True, ""


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


def _extract_json_fuzzy(text: str) -> list[dict] | None:
    """Fuzzy JSON extraction as fallback when tags are malformed.
    
    Tries to find JSON objects by looking for curly brace pairs.
    """
    results = []
    # Look for JSON-like structures outside of tags too
    # Pattern: look for { ... } with balanced braces
    i = 0
    while i < len(text):
        if text[i] == '{':
            # Try to find matching closing brace
            brace_count = 1
            j = i + 1
            while j < len(text) and brace_count > 0:
                if text[j] == '{':
                    brace_count += 1
                elif text[j] == '}':
                    brace_count -= 1
                j += 1
            
            if brace_count == 0:
                try:
                    candidate = text[i:j].strip()
                    parsed = json.loads(candidate)
                    if isinstance(parsed, dict):
                        results.append(parsed)
                except json.JSONDecodeError:
                    pass
            i = j
        else:
            i += 1
    
    return results or None


class TaskAgent:
    """Task agent that solves IMO grading problems."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Validate inputs first
        is_valid, error_msg = _validate_inputs(inputs)
        if not is_valid:
            self.log_fn(f"Input validation failed: {error_msg}")
            return f"Error: {error_msg}", []

        instruction = f"""You are an expert IMO (International Mathematical Olympiad) grader.

Your task is to evaluate a student's answer to a mathematical problem.

Task input:
```
{json.dumps(inputs, indent=2)}
```

Instructions:
1. Carefully read the problem, official solution, and grading guidelines
2. Analyze the student's answer against the solution and guidelines
3. Provide your evaluation in the exact JSON format below

Respond in JSON format with the following schema:
<json>
{{
    "response": "Your detailed evaluation here"
}}
</json>

Important: Your response MUST be valid JSON wrapped in <json>...</json> tags."""

        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
        except Exception as e:
            self.log_fn(f"LLM call failed: {e}")
            return f"Error: LLM call failed - {e}", []

        # Extract prediction from JSON
        prediction = "None"
        extraction_method = "none"
        
        try:
            if msg_history and len(msg_history) > 0:
                last_msg = msg_history[-1]
                text = last_msg.get("text", "")
                
                # Try standard extraction first
                extracted = _extract_jsons(text)
                if extracted:
                    extraction_method = "standard"
                else:
                    # Fallback to fuzzy extraction
                    extracted = _extract_json_fuzzy(text)
                    if extracted:
                        extraction_method = "fuzzy"
                
                if extracted:
                    # Find the first dict with a "response" key
                    for item in extracted:
                        if isinstance(item, dict) and "response" in item:
                            prediction = item["response"]
                            break
                    else:
                        # No "response" key found, use the last extracted item
                        prediction = str(extracted[-1])
                else:
                    self.log_fn("No JSON found in response")
                    # Fallback: use the raw text if no JSON found
                    prediction = text[:500] if text else "None"
            else:
                self.log_fn("Empty message history")
                
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            prediction = f"Error: {e}"

        self.log_fn(f"Extraction method: {extraction_method}, prediction type: {type(prediction).__name__}")
        return str(prediction), msg_history
