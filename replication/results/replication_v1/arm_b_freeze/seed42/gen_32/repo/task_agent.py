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
            logger.debug(f"Failed to parse JSON: {e}")
            continue
    return results or None


def _format_task_input(inputs: dict) -> str:
    """Format the task input dictionary into a structured prompt.
    
    Args:
        inputs: Dictionary containing task parameters
        
    Returns:
        Formatted string representation of inputs
    """
    formatted_parts = []
    for key, value in inputs.items():
        formatted_parts.append(f"{key}: {value}")
    return "\n".join(formatted_parts)


class TaskAgent:
    """Task agent that solves IMO grading problems."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.log_file = log_file

    def _build_prompt(self, inputs: dict) -> str:
        """Build the instruction prompt for the LLM.
        
        Args:
            inputs: Dictionary containing task parameters
            
        Returns:
            Formatted prompt string
        """
        formatted_input = _format_task_input(inputs)
        
        return f"""You are an expert grading agent. Your task is to evaluate student answers based on the provided problem, solution, and grading guidelines.

Task Input:
{formatted_input}

Instructions:
1. Carefully read the problem statement and official solution
2. Review the grading guidelines provided
3. Evaluate the student's answer against these criteria
4. Provide your assessment in the required JSON format

Response Format:
Respond in JSON format with the following schema:
<json>
{{
    "response": "Your detailed evaluation and grade here"
}}
</json>

Important: Ensure your response is valid JSON enclosed in <json> tags."""

    def _extract_prediction(self, msg_history: list[dict]) -> str:
        """Extract prediction from message history.
        
        Args:
            msg_history: List of message dictionaries
            
        Returns:
            Extracted prediction string or "None" if extraction fails
        """
        if not msg_history:
            logger.warning("Empty message history received")
            return "None"
            
        try:
            last_message = msg_history[-1]
            text = last_message.get("text", "")
            
            if not text:
                logger.warning("Last message has no text content")
                return "None"
            
            extracted = _extract_jsons(text)
            if extracted and "response" in extracted[-1]:
                return str(extracted[-1]["response"])
            else:
                logger.warning("No valid JSON with 'response' key found in output")
                return "None"
        except (KeyError, IndexError) as e:
            logger.error(f"Error accessing message history: {e}")
            return "None"
        except Exception as e:
            logger.error(f"Unexpected error extracting prediction: {e}")
            return "None"

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        logger.info(f"TaskAgent processing input with keys: {list(inputs.keys())}")
        
        instruction = self._build_prompt(inputs)
        
        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
            logger.info(f"LLM call completed. Response length: {len(response) if response else 0}")
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return "None", []

        # Extract prediction from JSON
        prediction = self._extract_prediction(msg_history)
        
        if prediction != "None":
            logger.info(f"Successfully extracted prediction: {prediction[:100]}...")
        
        return prediction, msg_history
