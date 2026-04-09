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
            logger.debug(f"JSON decode error: {e} for content: {inner[:100]}...")
            continue
    return results or None


def _format_grading_inputs(inputs: dict) -> str:
    """Format the grading inputs into a structured prompt."""
    domain = inputs.get("domain", "Unknown")
    problem = inputs.get("problem", "")
    solution = inputs.get("solution", "")
    grading_guidelines = inputs.get("grading_guidelines", "")
    student_answer = inputs.get("student_answer", "")
    
    return f"""Domain: {domain}

Problem Statement:
{problem}

Official Solution:
{solution}

Grading Guidelines:
{grading_guidelines}

Student's Answer:
{student_answer}

Your task: Evaluate the student's answer based on the problem, official solution, and grading guidelines. Provide your assessment in the response field."""


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
        self.log_fn(f"TaskAgent call #{self.call_count} with model={self.model}")
        
        # Format inputs in a more structured way
        formatted_inputs = _format_grading_inputs(inputs)
        
        instruction = f"""You are an expert mathematical grader evaluating student solutions to competition mathematics problems.

{formatted_inputs}

Respond in JSON format with the following schema:
<json>
{{
    "response": "Your detailed evaluation and grade for the student's answer"
}}
</json>

Important: Ensure your response is valid JSON enclosed in <json>...</json> tags."""

        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
            
            # Log usage info if available
            usage = info.get("usage", {})
            if usage:
                self.log_fn(f"LLM usage - prompt_tokens: {usage.get('prompt_tokens', 'N/A')}, "
                           f"completion_tokens: {usage.get('completion_tokens', 'N/A')}")
        except Exception as e:
            self.log_fn(f"Error calling LLM: {e}")
            return "Error: LLM call failed", [{"role": "error", "text": str(e)}]

        # Extract prediction from JSON
        prediction = "None"
        try:
            if msg_history and len(msg_history) > 0:
                last_message = msg_history[-1].get("text", "")
                extracted = _extract_jsons(last_message)
                if extracted:
                    last_json = extracted[-1]
                    if "response" in last_json:
                        prediction = last_json["response"]
                        self.log_fn(f"Successfully extracted prediction (length: {len(str(prediction))})")
                    else:
                        self.log_fn(f"No 'response' key in extracted JSON. Keys: {list(last_json.keys())}")
                else:
                    self.log_fn("No JSON blocks found in response")
            else:
                self.log_fn("Empty message history")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
