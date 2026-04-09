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
    Also attempts to parse raw JSON if no tags are found.
    """
    results = []
    search_from = 0
    
    # First try to find <json>...</json> blocks
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
            logger.debug(f"Failed to parse JSON block: {e}")
            continue
    
    # If no tagged blocks found, try to find raw JSON objects
    if not results:
        # Look for JSON objects between curly braces
        brace_count = 0
        start_idx = -1
        for i, char in enumerate(text):
            if char == '{':
                if brace_count == 0:
                    start_idx = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start_idx != -1:
                    try:
                        json_str = text[start_idx:i+1]
                        parsed = json.loads(json_str)
                        results.append(parsed)
                    except json.JSONDecodeError:
                        pass
                    start_idx = -1
    
    return results or None


def _sanitize_inputs(inputs: dict) -> dict:
    """Sanitize and validate task inputs."""
    required_keys = ["domain", "problem", "solution", "grading_guidelines", "student_answer"]
    sanitized = {}
    
    for key in required_keys:
        value = inputs.get(key, "")
        if not isinstance(value, str):
            value = str(value) if value is not None else ""
        # Limit length to prevent token overflow
        if len(value) > 10000:
            value = value[:10000] + "... [truncated]"
        sanitized[key] = value
    
    return sanitized


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
        self.call_count += 1
        
        # Sanitize inputs to ensure they're valid
        sanitized_inputs = _sanitize_inputs(inputs)
        
        instruction = f"""You are an expert grading agent evaluating student answers.

Your task is to analyze the student's answer and provide a grade based on the grading guidelines.

Domain: {sanitized_inputs['domain']}

Problem:
{sanitized_inputs['problem']}

Correct Solution:
{sanitized_inputs['solution']}

Grading Guidelines:
{sanitized_inputs['grading_guidelines']}

Student Answer:
{sanitized_inputs['student_answer']}

Respond in JSON format with the following schema:
<json>
{{
    "response": "Your grade/evaluation here"
}}
</json>"""

        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
        except Exception as e:
            self.log_fn(f"LLM call failed: {e}")
            return "Error: LLM call failed", []

        # Extract prediction from JSON
        prediction = "None"
        try:
            if msg_history and len(msg_history) > 0:
                last_message = msg_history[-1].get("text", "")
                extracted = _extract_jsons(last_message)
                if extracted:
                    last_extracted = extracted[-1]
                    if "response" in last_extracted:
                        prediction = last_extracted["response"]
                    else:
                        # Try to find any key that might contain the response
                        for key in ["grade", "evaluation", "result", "answer"]:
                            if key in last_extracted:
                                prediction = last_extracted[key]
                                break
                        else:
                            # If no recognized key, use the first value
                            prediction = str(list(last_extracted.values())[0]) if last_extracted else str(last_extracted)
                else:
                    self.log_fn(f"No JSON found in response. Raw response: {last_message[:200]}...")
            else:
                self.log_fn("Empty message history from LLM")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
