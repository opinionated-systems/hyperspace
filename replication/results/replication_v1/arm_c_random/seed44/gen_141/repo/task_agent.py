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
    Includes additional fallback strategies for robust extraction.
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
            # Try to clean up common JSON issues
            try:
                # Remove trailing commas before closing braces/brackets
                cleaned = re.sub(r',(\s*[}\]])', r'\1', inner)
                results.append(json.loads(cleaned))
            except json.JSONDecodeError:
                continue
    
    # Fallback 1: try to find JSON objects directly if no <json> tags
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
                    # Try cleaning trailing commas
                    cleaned = re.sub(r',(\s*[}\]])', r'\1', potential_json)
                    results.append(json.loads(cleaned))
        except json.JSONDecodeError:
            pass
    
    # Fallback 2: Look for code blocks with json
    if not results:
        code_block_pattern = r'```(?:json)?\s*\n(.*?)\n```'
        for match in re.finditer(code_block_pattern, text, re.DOTALL):
            try:
                content = match.group(1).strip()
                results.append(json.loads(content))
            except json.JSONDecodeError:
                try:
                    cleaned = re.sub(r',(\s*[}\]])', r'\1', content)
                    results.append(json.loads(cleaned))
                except json.JSONDecodeError:
                    continue
    
    return results or None


def _format_inputs(inputs: dict) -> str:
    """Format task inputs into a structured prompt.
    
    Provides better structure and context for the LLM.
    """
    parts = []
    # Define priority order for common fields
    priority_keys = ["domain", "problem", "solution", "grading_guidelines", "student_answer"]
    
    # Add priority keys first in order
    for key in priority_keys:
        if key in inputs:
            value = inputs[key]
            if value and str(value).strip():
                parts.append(f"=== {key.upper().replace('_', ' ')} ===\n{value}\n")
    
    # Add any remaining keys
    for key, value in inputs.items():
        if key not in priority_keys:
            if value and str(value).strip():
                parts.append(f"=== {key.upper().replace('_', ' ')} ===\n{value}\n")
    
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
    if response_value is None:
        return False, "'response' value is None"
    
    if not isinstance(response_value, (str, int, float, bool, list)):
        return False, f"'response' value has unsupported type: {type(response_value)}"
    
    # Convert non-string values to string for consistency
    if not isinstance(response_value, str):
        try:
            response_value = str(response_value)
        except Exception:
            return False, "Could not convert 'response' value to string"
    
    # Check for empty response
    if isinstance(response_value, str) and not response_value.strip():
        return False, "'response' value is empty"
    
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
        
        instruction = f"""You are an expert grading agent for mathematical olympiad problems. Your task is to evaluate student answers based on the provided problem, solution, and grading guidelines.

{formatted_inputs}

Respond in JSON format with the following schema:
<json>
{{
    "response": "Your evaluation result here"
}}
</json>

IMPORTANT: Your response MUST be valid JSON wrapped in <json> tags. The "response" field should contain your final evaluation.

Think step by step:
1. First, carefully read and understand the problem statement
2. Review the provided solution to understand the correct approach and expected answer
3. Examine the grading guidelines carefully - these define how points are awarded
4. Evaluate the student's answer against the solution:
   - Check if the answer is mathematically correct
   - Check if the reasoning is sound
   - Check if all parts of the problem are addressed
5. Formulate your final evaluation in the required JSON format

Guidelines for grading:
- Be precise and objective in your evaluation
- Consider partial credit where appropriate based on the grading guidelines
- If the answer is completely correct, state that clearly
- If there are errors, identify them specifically
- If the answer is partially correct, explain what is right and what is wrong"""

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
                
                # Try to extract from assistant message
                if last_message.get("role") == "assistant":
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
                            # Try to use raw text as prediction if JSON extraction fails
                            if text_content and len(text_content.strip()) > 0:
                                prediction = text_content.strip()[:500]
                                self.log_fn(f"Using raw text as fallback prediction")
                    else:
                        self.error_count += 1
                        self.log_fn("No JSON found in response")
                        # Try to use raw text as prediction if JSON extraction fails
                        if text_content and len(text_content.strip()) > 0:
                            prediction = text_content.strip()[:500]
                            self.log_fn(f"Using raw text as fallback prediction")
                else:
                    self.error_count += 1
                    self.log_fn(f"Last message is not from assistant: {last_message.get('role')}")
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
