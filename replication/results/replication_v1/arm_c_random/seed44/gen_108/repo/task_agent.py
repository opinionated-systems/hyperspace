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


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks with enhanced fallback strategies.

    Uses multiple extraction strategies:
    1. Primary: Extract from <json>...</json> tags
    2. Fallback 1: Extract from ```json code blocks
    3. Fallback 2: Extract from markdown code blocks
    4. Fallback 3: Find outermost JSON object by brace matching
    5. Fallback 4: Find all potential JSON objects in text
    """
    results = []
    
    # Strategy 1: Extract from <json>...</json> tags
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
    
    if results:
        return results
    
    # Strategy 2: Extract from ```json code blocks
    search_from = 0
    while True:
        start = text.find("```json", search_from)
        if start == -1:
            break
        end = text.find("```", start + 7)
        if end == -1:
            break
        inner = text[start + 7:end].strip()
        search_from = end + 3
        try:
            results.append(json.loads(inner))
        except json.JSONDecodeError:
            continue
    
    if results:
        return results
    
    # Strategy 3: Extract from generic ``` code blocks
    search_from = 0
    while True:
        start = text.find("```", search_from)
        if start == -1:
            break
        end = text.find("```", start + 3)
        if end == -1:
            break
        inner = text[start + 3:end].strip()
        search_from = end + 3
        # Try to parse as JSON
        try:
            if inner.startswith("{") or inner.startswith("["):
                results.append(json.loads(inner))
        except json.JSONDecodeError:
            continue
    
    if results:
        return results
    
    # Strategy 4: Find outermost JSON object by brace matching
    try:
        brace_start = text.find("{")
        if brace_start != -1:
            # Find matching closing brace
            brace_count = 0
            brace_end = -1
            for i, char in enumerate(text[brace_start:], start=brace_start):
                if char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        brace_end = i
                        break
            
            if brace_end != -1:
                potential_json = text[brace_start:brace_end + 1]
                results.append(json.loads(potential_json))
    except (json.JSONDecodeError, ValueError):
        pass
    
    # Strategy 5: Find all potential JSON objects using regex-like approach
    if not results:
        # Find all substrings that start with { and end with }
        json_pattern = re.compile(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}')
        for match in json_pattern.finditer(text):
            try:
                potential_json = match.group()
                parsed = json.loads(potential_json)
                if isinstance(parsed, dict):
                    results.append(parsed)
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

IMPORTANT: You must respond with ONLY a valid JSON object wrapped in <json> tags. Do not include any other text outside the JSON tags.

Required format:
<json>
{{
    "response": "Your evaluation result here - provide a clear, concise assessment"
}}
</json>

Grading Guidelines:
1. First, carefully read and understand what the problem is asking
2. Review the provided solution to understand the correct approach and expected answer
3. Examine the grading guidelines for specific criteria
4. Compare the student's answer against the correct solution:
   - Check if the answer is mathematically correct
   - Check if the reasoning follows logical steps
   - Check if the final answer matches or is equivalent to the solution
5. Provide a clear, concise evaluation in the "response" field

Remember: Your entire response must be valid JSON inside <json>...</json> tags."""

        # Retry mechanism for LLM calls
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=[],
                )
                self.log_fn(f"LLM call completed (attempt {attempt + 1}), response length: {len(response)}")
                break
            except Exception as e:
                self.log_fn(f"Error in LLM call (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    self.error_count += 1
                    return "Error: LLM call failed after retries", []
                # Brief pause before retry
                time.sleep(0.5 * (attempt + 1))

        # Extract prediction from JSON with better error handling
        prediction = "None"
        try:
            if msg_history and len(msg_history) > 0:
                # Try multiple messages in history (in case of tool calls or multiple turns)
                for message in reversed(msg_history):
                    text_content = message.get("text", "")
                    if not text_content:
                        continue
                    
                    extracted = _extract_jsons(text_content)
                    if extracted:
                        last_extracted = extracted[-1]
                        is_valid, error_msg = _validate_grading_response(last_extracted)
                        if is_valid and isinstance(last_extracted, dict) and "response" in last_extracted:
                            prediction = last_extracted["response"]
                            self.success_count += 1
                            self.log_fn(f"Successfully extracted prediction: {str(prediction)[:100]}")
                            break
                        else:
                            self.log_fn(f"Invalid grading response in message: {error_msg}")
                else:
                    # No valid JSON found in any message
                    self.error_count += 1
                    self.log_fn("No valid JSON found in any message")
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
