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

# Constants for retry logic
MAX_RETRIES = 3
RETRY_DELAY = 1.0  # seconds


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    
    Also attempts to extract JSON from markdown code blocks as fallback.
    """
    results = []
    search_from = 0
    
    # Primary: Extract from <json> tags
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
    
    # Fallback: Extract from markdown code blocks
    if not results:
        pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                results.append(json.loads(match.strip()))
            except json.JSONDecodeError:
                continue
    
    # Last resort: Try to find any JSON object in the text
    if not results:
        try:
            # Look for content between first { and last }
            start_idx = text.find('{')
            end_idx = text.rfind('}')
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                potential_json = text[start_idx:end_idx + 1]
                results.append(json.loads(potential_json))
        except (json.JSONDecodeError, ValueError):
            pass
    
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

        # Attempt LLM call with retry logic
        response = None
        msg_history = []
        info = {}
        last_error = None
        
        for attempt in range(MAX_RETRIES):
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=[],
                )
                # If we got a valid response, break out of retry loop
                if msg_history and len(msg_history) > 0:
                    break
            except Exception as e:
                last_error = e
                self.log_fn(f"LLM call attempt {attempt + 1}/{MAX_RETRIES} failed: {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY * (attempt + 1))  # Exponential backoff
                continue
        
        if response is None or not msg_history:
            error_msg = f"LLM call failed after {MAX_RETRIES} attempts"
            if last_error:
                error_msg += f": {last_error}"
            self.log_fn(error_msg)
            return f"Error: {error_msg}", []

        # Extract prediction from JSON with enhanced error handling
        prediction = "None"
        extraction_errors = []
        
        try:
            if msg_history and len(msg_history) > 0:
                llm_text = msg_history[-1].get("text", "")
                if not llm_text:
                    extraction_errors.append("Empty text in message history")
                else:
                    extracted = _extract_jsons(llm_text)
                    if extracted:
                        # Try to find response in any of the extracted JSONs
                        for json_obj in extracted:
                            if isinstance(json_obj, dict):
                                if "response" in json_obj:
                                    prediction = json_obj["response"]
                                    break
                                # Try alternative keys that might contain the evaluation
                                for key in ["evaluation", "grade", "feedback", "result", "answer"]:
                                    if key in json_obj:
                                        prediction = json_obj[key]
                                        self.log_fn(f"Found alternative key '{key}' in JSON response")
                                        break
                                if prediction != "None":
                                    break
                        else:
                            extraction_errors.append("No recognized keys found in JSON objects")
                    else:
                        extraction_errors.append("No valid JSON found in LLM output")
                        # Fallback: use raw text if no JSON found
                        if llm_text.strip():
                            prediction = llm_text.strip()
                            self.log_fn("Using raw LLM output as prediction (no JSON found)")
            else:
                extraction_errors.append("Empty message history from LLM")
        except Exception as e:
            extraction_errors.append(f"Exception during extraction: {e}")
            self.log_fn(f"Error extracting prediction: {e}")
        
        # Log any extraction issues
        if extraction_errors and prediction == "None":
            self.log_fn(f"JSON extraction issues: {'; '.join(extraction_errors)}")

        return str(prediction), msg_history
