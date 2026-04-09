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
    Includes multiple fallback strategies for robust extraction.
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
                # Fix single quotes to double quotes
                cleaned = cleaned.replace("'", '"')
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
                results.append(json.loads(potential_json))
        except json.JSONDecodeError:
            pass
    
    # Fallback 2: try to find JSON in code blocks
    if not results:
        code_block_pattern = r'```(?:json)?\s*(.*?)\s*```'
        matches = re.findall(code_block_pattern, text, re.DOTALL)
        for match in matches:
            try:
                results.append(json.loads(match.strip()))
            except json.JSONDecodeError:
                # Try cleaning
                try:
                    cleaned = re.sub(r',(\s*[}\]])', r'\1', match.strip())
                    results.append(json.loads(cleaned))
                except json.JSONDecodeError:
                    continue
    
    return results or None


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
        
        instruction = f"""You are an expert grading agent for mathematical problem solving. Your task is to evaluate student answers based on the provided problem, solution, and grading guidelines.

{formatted_inputs}

Respond in JSON format with the following schema:
<json>
{{
    "response": "Your evaluation result here - be specific about what the student got right/wrong"
}}
</json>

Ensure your response is valid JSON and follows the schema exactly. The "response" field should contain your complete evaluation.

Evaluation guidelines:
1. First, carefully read and understand the problem requirements
2. Study the provided solution to identify key concepts, steps, and the final answer
3. Review the grading guidelines to understand the scoring criteria
4. Compare the student's answer against the solution:
   - Check if the approach/method is correct
   - Verify calculations and reasoning steps
   - Confirm the final answer matches (or is equivalent to) the solution
5. Provide a clear, specific evaluation in the JSON response field
6. If the answer is partially correct, explain what aspects are correct and what needs improvement

Be thorough and precise in your evaluation."""

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

        # Extract prediction from JSON with better error handling and retry
        prediction = "None"
        extraction_attempts = []
        
        try:
            if msg_history and len(msg_history) > 0:
                # Try extracting from the last few messages (in case of retries)
                for msg in reversed(msg_history[-3:]):
                    text_content = msg.get("text", "")
                    if not text_content:
                        continue
                    
                    extracted = _extract_jsons(text_content)
                    if extracted:
                        for ext in reversed(extracted):
                            is_valid, error_msg = _validate_grading_response(ext)
                            if is_valid and isinstance(ext, dict) and "response" in ext:
                                prediction = ext["response"]
                                self.success_count += 1
                                self.log_fn(f"Successfully extracted prediction: {str(prediction)[:100]}")
                                break
                            else:
                                extraction_attempts.append(f"Invalid: {error_msg}")
                        if prediction != "None":
                            break
                    else:
                        extraction_attempts.append("No JSON found")
                
                if prediction == "None":
                    self.error_count += 1
                    if extraction_attempts:
                        self.log_fn(f"JSON extraction failed: {'; '.join(extraction_attempts[:3])}")
                    else:
                        self.log_fn("No valid JSON found in recent messages")
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
