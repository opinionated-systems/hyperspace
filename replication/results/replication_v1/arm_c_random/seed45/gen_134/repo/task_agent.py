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
from agent.utils import truncate_text, safe_get, retry_with_backoff

logger = logging.getLogger(__name__)

# Required input fields for task evaluation
REQUIRED_INPUT_FIELDS = {
    "domain",
    "problem",
    "solution",
    "grading_guidelines",
    "student_answer",
}


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


def _validate_inputs(inputs: dict) -> tuple[bool, str]:
    """Validate that all required input fields are present.
    
    Args:
        inputs: Input dictionary to validate
        
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
        return False, f"Empty values for fields: {sorted(empty_fields)}"
    
    return True, ""


class TaskAgent:
    """Task agent that solves IMO grading problems."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self._call_count = 0
        self._log_file = log_file

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        self._call_count += 1
        
        # Validate inputs
        is_valid, error_msg = _validate_inputs(inputs)
        if not is_valid:
            logger.error(f"Input validation failed: {error_msg}")
            return f"ERROR_INVALID_INPUT: {error_msg}", [{"role": "system", "text": f"Input error: {error_msg}"}]
        
        # Build structured instruction with better formatting
        instruction = self._build_instruction(inputs)
        
        self.log_fn(f"TaskAgent call #{self._call_count}: processing {inputs.get('domain', 'unknown')} problem")

        # Use retry with backoff for transient failures
        from agent.utils import retry_with_backoff
        
        @retry_with_backoff(
            max_retries=2,
            base_delay=1.0,
            max_delay=10.0,
            exceptions=(Exception,),
            on_retry=lambda e, n: logger.warning(f"LLM call attempt {n} failed: {e}. Retrying...")
        )
        def _call_llm():
            return get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
        
        try:
            response, msg_history, info = _call_llm()
        except Exception as e:
            logger.error(f"LLM call failed after retries: {e}")
            return "ERROR_LLM_CALL_FAILED", [{"role": "system", "text": f"Error: {e}"}]

        # Extract prediction from JSON
        prediction = self._extract_prediction(msg_history)
        
        # Log usage info if available
        usage = safe_get(info, "usage", default={})
        if usage:
            prompt_tokens = safe_get(usage, "prompt_tokens", default=0)
            completion_tokens = safe_get(usage, "completion_tokens", default=0)
            total_tokens = safe_get(usage, "total_tokens", default=0)
            self.log_fn(f"Usage: prompt={prompt_tokens}, completion={completion_tokens}, total={total_tokens}")

        return prediction, msg_history
    
    def _build_instruction(self, inputs: dict) -> str:
        """Build the instruction prompt for the LLM."""
        # Extract and sanitize fields
        domain = str(inputs.get("domain", "Unknown")).strip()
        problem = str(inputs.get("problem", "")).strip()
        solution = str(inputs.get("solution", "")).strip()
        grading_guidelines = str(inputs.get("grading_guidelines", "")).strip()
        student_answer = str(inputs.get("student_answer", "")).strip()
        
        # Truncate very long inputs to avoid token limits
        max_len = 8000
        problem = problem[:max_len] + "..." if len(problem) > max_len else problem
        solution = solution[:max_len] + "..." if len(solution) > max_len else solution
        grading_guidelines = grading_guidelines[:max_len] + "..." if len(grading_guidelines) > max_len else grading_guidelines
        student_answer = student_answer[:max_len] + "..." if len(student_answer) > max_len else student_answer
        
        return f"""You are an expert grading agent. Your task is to evaluate a student's answer based on the provided problem, solution, and grading guidelines.

Domain: {domain}

Problem:
{problem}

Solution:
{solution}

Grading Guidelines:
{grading_guidelines}

Student's Answer:
{student_answer}

Instructions:
1. Carefully read the problem, solution, and grading guidelines
2. Evaluate the student's answer against the criteria
3. Consider: correctness, completeness, clarity, and adherence to the solution approach
4. Provide your evaluation in the exact JSON format below

Respond in JSON format with the following schema:
<json>
{{
    "response": "Your detailed evaluation here. Include: (1) assessment of correctness, (2) partial credit if applicable, (3) specific feedback on what was done well or incorrectly."
}}
</json>

Important: Your response must be valid JSON inside the <json> tags. Do not include any text outside the JSON block."""
    
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
