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
            logger.debug(f"JSON decode error at position {start}: {e}")
            continue
    return results or None


def _format_grading_inputs(inputs: dict) -> str:
    """Format the grading inputs into a structured prompt."""
    domain = inputs.get("domain", "Mathematics")
    problem = inputs.get("problem", "")
    solution = inputs.get("solution", "")
    guidelines = inputs.get("grading_guidelines", "")
    student_answer = inputs.get("student_answer", "")
    
    return f"""Domain: {domain}

Problem Statement:
{problem}

Official Solution:
{solution}

Grading Guidelines:
{guidelines}

Student's Answer:
{student_answer}"""


class TaskAgent:
    """Task agent that solves IMO grading problems with enhanced prompting."""

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
        start_time = time.time()
        self.call_count += 1
        
        # Format inputs in a structured way
        formatted_inputs = _format_grading_inputs(inputs)
        
        instruction = f"""You are an expert mathematics grader for the International Mathematical Olympiad (IMO).

Your task is to evaluate a student's solution to a mathematical problem. Carefully compare the student's answer against the official solution and grading guidelines.

{formatted_inputs}

Instructions:
1. Analyze the student's solution step by step
2. Check for mathematical correctness and logical reasoning
3. Award points according to the grading guidelines
4. Provide your final grade as a number (e.g., 0, 1, 2, 3, etc.)

Respond in JSON format with the following schema:
<json>
{{
    "response": <numerical_grade>
}}
</json>

The response field should contain only the numerical grade (integer or float)."""

        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return "0", [{"role": "error", "text": str(e)}]

        # Extract prediction from JSON
        prediction = "0"  # Default to 0 on failure
        try:
            extracted = _extract_jsons(msg_history[-1]["text"])
            if extracted and "response" in extracted[-1]:
                prediction = extracted[-1]["response"]
                # Validate that prediction is a number
                if not isinstance(prediction, (int, float)):
                    # Try to extract number from string
                    match = re.search(r'[-+]?[\d.]+', str(prediction))
                    if match:
                        prediction = float(match.group())
            else:
                logger.warning(f"No valid JSON found in response for call {self.call_count}")
        except Exception as e:
            self.log_fn(f"Error extracting prediction in call {self.call_count}: {e}")

        elapsed = time.time() - start_time
        logger.info(f"Call {self.call_count} completed in {elapsed:.2f}s, prediction: {prediction}")

        return str(prediction), msg_history
