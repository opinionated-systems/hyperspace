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
            logger.debug(f"JSON decode error: {e}, content: {inner[:200]}")
            continue
    return results or None


def _format_grading_inputs(inputs: dict) -> str:
    """Format the grading inputs into a structured prompt."""
    domain = inputs.get("domain", "Mathematics")
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
{student_answer}"""


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.stats = {
            "total_calls": 0,
            "successful_extractions": 0,
            "failed_extractions": 0,
        }

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        start_time = time.time()
        self.stats["total_calls"] += 1
        
        # Format inputs in a structured way
        formatted_inputs = _format_grading_inputs(inputs)
        
        instruction = f"""You are an expert IMO (International Mathematical Olympiad) grader. Your task is to evaluate a student's solution to a mathematical problem.

Carefully analyze the student's answer by:
1. Understanding the problem requirements
2. Reviewing the official solution and grading guidelines
3. Comparing the student's approach with the expected solution
4. Identifying any errors, gaps, or creative valid approaches
5. Assigning an appropriate score based on the grading rubric

{formatted_inputs}

Provide your evaluation with chain-of-thought reasoning, then give your final assessment.

Respond in JSON format with the following schema:
<json>
{{
    "reasoning": "Your detailed step-by-step analysis of the student's solution",
    "response": "Your final grade/score as a number or string"
}}
</json>

Be thorough and fair in your grading. Consider partial credit where appropriate."""

        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return "Error: LLM call failed", [{"role": "error", "text": str(e)}]

        # Extract prediction from JSON with improved error handling
        prediction = "None"
        reasoning = ""
        try:
            if msg_history and len(msg_history) > 0:
                extracted = _extract_jsons(msg_history[-1]["text"])
                if extracted:
                    last_extract = extracted[-1]
                    if "response" in last_extract:
                        prediction = last_extract["response"]
                        self.stats["successful_extractions"] += 1
                    if "reasoning" in last_extract:
                        reasoning = last_extract["reasoning"]
                else:
                    self.stats["failed_extractions"] += 1
                    logger.warning("No JSON blocks found in response")
            else:
                self.stats["failed_extractions"] += 1
                logger.warning("Empty message history")
        except Exception as e:
            self.stats["failed_extractions"] += 1
            self.log_fn(f"Error extracting prediction: {e}")

        elapsed = time.time() - start_time
        logger.info(f"TaskAgent completed in {elapsed:.2f}s | Prediction: {prediction}")
        
        return str(prediction), msg_history

    def get_stats(self) -> dict[str, Any]:
        """Return agent statistics."""
        return self.stats.copy()
