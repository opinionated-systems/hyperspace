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
    Also attempts to parse raw JSON objects if no <json> tags are found.
    
    Enhanced to handle nested braces, multiple JSON objects, and common
    formatting issues like trailing commas and comments.
    """
    results = []
    
    def _try_parse_json(json_str: str) -> dict | None:
        """Try to parse JSON string, with cleaning fallback."""
        cleaned = _clean_json_string(json_str)
        for candidate in [cleaned, json_str]:
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                continue
        return None
    
    # First pass: extract from <json> tags
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
        
        parsed = _try_parse_json(inner)
        if parsed:
            results.append(parsed)
        else:
            logger.debug(f"Failed to parse JSON from <json> block")
    
    # Second pass: try to find JSON objects directly if no <json> tags found
    if not results:
        brace_start = 0
        while brace_start < len(text):
            brace_start = text.find("{", brace_start)
            if brace_start == -1:
                break
            
            # Find matching closing brace using a stack counter
            brace_count = 0
            brace_end = brace_start
            for i, char in enumerate(text[brace_start:], start=brace_start):
                if char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        brace_end = i
                        break
            
            if brace_end > brace_start:
                potential_json = text[brace_start:brace_end + 1]
                parsed = _try_parse_json(potential_json)
                if parsed:
                    results.append(parsed)
                brace_start = brace_end + 1
            else:
                break
    
    return results if results else None


def _clean_json_string(json_str: str) -> str:
    """Clean up common JSON formatting issues.
    
    Handles:
    - Trailing commas before closing braces/brackets
    - Single quotes instead of double quotes
    - Python-style True/False/None instead of true/false/null
    """
    import re
    
    # Remove trailing commas before } or ]
    cleaned = re.sub(r',(\s*[}\]])', r'\1', json_str)
    
    # Replace single quotes with double quotes (simple cases)
    # This is a basic fix - complex nested quotes may still fail
    cleaned = re.sub(r"'([^']*?)':", r'"\1":', cleaned)
    cleaned = re.sub(r":\s*'([^']*?)'([,}\]])", r': "\1"\2', cleaned)
    
    # Replace Python booleans/None with JSON equivalents
    cleaned = re.sub(r'\bTrue\b', 'true', cleaned)
    cleaned = re.sub(r'\bFalse\b', 'false', cleaned)
    cleaned = re.sub(r'\bNone\b', 'null', cleaned)
    
    return cleaned


def _format_inputs(inputs: dict) -> str:
    """Format task inputs into a structured prompt.
    
    Provides better structure and context for the LLM.
    """
    parts = []
    for key, value in inputs.items():
        parts.append(f"{key}:\n{value}\n")
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
    """Task agent that solves IMO grading problems with improved error handling."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.call_count = 0
        self.success_count = 0
        self.error_count = 0

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        self.call_count += 1
        self.log_fn(f"TaskAgent call #{self.call_count} starting")
        
        # Format inputs more clearly
        formatted_inputs = _format_inputs(inputs)
        
        instruction = f"""You are an expert grading agent. Your task is to evaluate student answers based on the provided problem, solution, and grading guidelines.

{formatted_inputs}

Respond in JSON format with the following schema:
<json>
{{
    "response": "Your evaluation result here"
}}
</json>

Ensure your response is valid JSON and follows the schema exactly.

Think step by step:
1. First, understand the problem and what is being asked
2. Review the provided solution to understand the correct approach
3. Examine the grading guidelines carefully
4. Evaluate the student's answer against the solution and guidelines
5. Formulate your final evaluation in the required JSON format"""

        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
            self.log_fn(f"LLM call completed, response length: {len(response)}")
        except Exception as e:
            self.log_fn(f"Error in LLM call: {e}")
            self.error_count += 1
            return "Error: LLM call failed", []

        # Extract prediction from JSON with better error handling
        prediction = "None"
        try:
            if msg_history and len(msg_history) > 0:
                last_message = msg_history[-1]
                text_content = last_message.get("text", "")
                extracted = _extract_jsons(text_content)
                if extracted:
                    last_extracted = extracted[-1]
                    is_valid, error_msg = _validate_grading_response(last_extracted)
                    if is_valid and isinstance(last_extracted, dict) and "response" in last_extracted:
                        prediction = last_extracted["response"]
                        self.success_count += 1
                        self.log_fn(f"Successfully extracted prediction: {str(prediction)[:100]}")
                    else:
                        self.error_count += 1
                        self.log_fn(f"Invalid grading response: {error_msg}")
                else:
                    self.error_count += 1
                    self.log_fn("No JSON found in response")
            else:
                self.error_count += 1
                self.log_fn("Empty message history")
        except Exception as e:
            self.error_count += 1
            self.log_fn(f"Error extracting prediction: {e}")

        self.log_fn(f"TaskAgent stats: {self.success_count} successes, {self.error_count} errors out of {self.call_count} calls")
        return str(prediction), msg_history

    def get_stats(self) -> dict:
        """Return agent performance statistics."""
        return {
            "total_calls": self.call_count,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "success_rate": self.success_count / max(1, self.call_count),
        }
