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
    """Extract JSON objects from <json>...</json> blocks with enhanced robustness.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    Also attempts to parse raw JSON objects if no <json> tags are found.
    Includes additional cleanup for common LLM output issues.
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
        
        # Try to parse, with cleanup for common issues
        try:
            results.append(json.loads(inner))
        except json.JSONDecodeError:
            # Try cleaning up common issues: trailing commas, comments, etc.
            cleaned = _clean_json_string(inner)
            try:
                results.append(json.loads(cleaned))
            except json.JSONDecodeError:
                continue
    
    # Fallback: try to find JSON objects directly if no <json> tags
    if not results:
        try:
            # Look for JSON-like structures with braces
            brace_start = text.find("{")
            brace_end = text.rfind("}")
            if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
                potential_json = text[brace_start:brace_end + 1]
                try:
                    results.append(json.loads(potential_json))
                except json.JSONDecodeError:
                    # Try with cleanup
                    cleaned = _clean_json_string(potential_json)
                    results.append(json.loads(cleaned))
        except json.JSONDecodeError:
            pass
    
    return results or None


def _clean_json_string(text: str) -> str:
    """Clean up common JSON formatting issues from LLM outputs.
    
    Handles:
    - Trailing commas before closing braces/brackets
    - Single quotes instead of double quotes
    - Comments (// and /* */)
    - Extra whitespace
    """
    import re
    
    # Remove single-line comments
    text = re.sub(r'//[^\n]*', '', text)
    # Remove multi-line comments
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    
    # Remove trailing commas before } or ]
    text = re.sub(r',\s*}', '}', text)
    text = re.sub(r',\s*\]', ']', text)
    
    # Replace single quotes with double quotes (carefully)
    # Only replace quotes that appear to be delimiting strings
    text = re.sub(r"(?<=[{\s,:\[])'([^']*)'(?=[}\s,:\]])", r'"\1"', text)
    
    return text.strip()


def _format_inputs(inputs: dict) -> str:
    """Format task inputs into a structured prompt with improved formatting.
    
    Provides better structure and context for the LLM.
    Handles various input types and ensures clean formatting.
    """
    parts = []
    
    # Define priority order for common fields to ensure logical flow
    priority_fields = ["domain", "problem", "solution", "grading_guidelines", "student_answer"]
    
    # Add priority fields first in order
    for key in priority_fields:
        if key in inputs:
            value = inputs[key]
            # Clean up the value - ensure it's a string and strip excess whitespace
            if not isinstance(value, str):
                value = str(value)
            value = value.strip()
            if value:  # Only add non-empty values
                parts.append(f"## {key.replace('_', ' ').title()}\n{value}\n")
    
    # Add any remaining fields not in priority list
    for key, value in inputs.items():
        if key not in priority_fields:
            if not isinstance(value, str):
                value = str(value)
            value = value.strip()
            if value:
                parts.append(f"## {key.replace('_', ' ').title()}\n{value}\n")
    
    return "\n".join(parts)


def _validate_grading_response(response: dict) -> tuple[bool, str]:
    """Validate that the grading response has the required structure.
    
    Returns:
        (is_valid, error_message)
    """
    if not isinstance(response, dict):
        return False, "Response is not a dictionary"
    
    if "response" not in response:
        return False, "Missing 'response' key in JSON"
    
    response_value = response["response"]
    if not isinstance(response_value, (str, int, float, bool)):
        return False, f"'response' value has unsupported type: {type(response_value)}"
    
    return True, ""


class TaskAgent:
    """Task agent that solves IMO grading problems with improved error handling and retry logic."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "", max_retries: int = 2) -> None:
        self.model = model
        self.log_fn = logger.info
        self.call_count = 0
        self.success_count = 0
        self.error_count = 0
        self.retry_count = 0
        self.max_retries = max_retries

    def _build_instruction(self, formatted_inputs: str, is_retry: bool = False) -> str:
        """Build the instruction prompt for the LLM.
        
        For retries, adds stronger emphasis on JSON format compliance.
        """
        retry_note = ""
        if is_retry:
            retry_note = """
IMPORTANT: Your previous response could not be parsed. You MUST respond with valid JSON only.
Do not include any text outside the JSON tags. Ensure proper JSON syntax with double quotes."""
        
        return f"""You are an expert grading agent. Your task is to evaluate student answers based on the provided problem, solution, and grading guidelines.

{formatted_inputs}

Respond in JSON format with the following schema:
<json>
{{
    "response": "Your evaluation result here"
}}
</json>

Ensure your response is valid JSON and follows the schema exactly.{retry_note}

Think step by step:
1. First, understand the problem and what is being asked
2. Review the provided solution to understand the correct approach
3. Examine the grading guidelines carefully
4. Evaluate the student's answer against the solution and guidelines
5. Formulate your final evaluation in the required JSON format"""

    def _extract_prediction(self, msg_history: list[dict]) -> tuple[str, bool]:
        """Extract prediction from message history.
        
        Returns:
            (prediction, success)
        """
        if not msg_history:
            self.log_fn("Empty message history")
            return "None", False
        
        last_message = msg_history[-1]
        text_content = last_message.get("text", "")
        
        extracted = _extract_jsons(text_content)
        if not extracted:
            self.log_fn("No JSON found in response")
            return "None", False
        
        last_extracted = extracted[-1]
        is_valid, error_msg = _validate_grading_response(last_extracted)
        
        if not is_valid:
            self.log_fn(f"Invalid grading response: {error_msg}")
            return "None", False
        
        if not isinstance(last_extracted, dict) or "response" not in last_extracted:
            self.log_fn("Response missing 'response' key")
            return "None", False
        
        prediction = last_extracted["response"]
        self.log_fn(f"Successfully extracted prediction: {str(prediction)[:100]}")
        return str(prediction), True

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with retry logic.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        self.call_count += 1
        self.log_fn(f"TaskAgent call #{self.call_count} starting")
        
        # Format inputs more clearly
        formatted_inputs = _format_inputs(inputs)
        
        # Try initial call
        prediction, msg_history, success = self._try_call(formatted_inputs, is_retry=False)
        
        # Retry on failure if we have retries left
        attempt = 0
        while not success and attempt < self.max_retries:
            self.retry_count += 1
            attempt += 1
            self.log_fn(f"Retry attempt {attempt}/{self.max_retries}")
            prediction, msg_history, success = self._try_call(formatted_inputs, is_retry=True)
        
        # Update stats
        if success:
            self.success_count += 1
        else:
            self.error_count += 1
        
        self.log_fn(f"TaskAgent stats: {self.success_count} successes, {self.error_count} errors, {self.retry_count} retries out of {self.call_count} calls")
        return prediction, msg_history

    def _try_call(self, formatted_inputs: str, is_retry: bool = False) -> tuple[str, list[dict], bool]:
        """Make a single LLM call and extract prediction.
        
        Returns:
            (prediction, msg_history, success)
        """
        instruction = self._build_instruction(formatted_inputs, is_retry)
        
        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
            self.log_fn(f"LLM call completed, response length: {len(response)}")
        except Exception as e:
            self.log_fn(f"Error in LLM call: {e}")
            return "Error: LLM call failed", [], False

        # Extract prediction from JSON
        prediction, success = self._extract_prediction(msg_history)
        return prediction, msg_history, success

    def get_stats(self) -> dict:
        """Return agent performance statistics."""
        return {
            "total_calls": self.call_count,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "retry_count": self.retry_count,
            "success_rate": self.success_count / max(1, self.call_count),
        }
