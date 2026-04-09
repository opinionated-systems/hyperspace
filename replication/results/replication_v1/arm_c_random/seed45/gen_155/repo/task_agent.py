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
        except json.JSONDecodeError:
            continue
    return results or None


def _validate_grading_response(response: any, inputs: dict) -> str:
    """Validate and normalize the grading response.
    
    For IMO grading, valid responses are typically:
    - Numeric scores (0, 1, 2, ...)
    - Special markers like "N/A", "None", "Invalid"
    - String representations of numbers
    
    Returns normalized string representation.
    """
    if response is None:
        return "None"
    
    # Convert to string and strip whitespace
    result = str(response).strip()
    
    # Handle empty responses
    if not result:
        return "None"
    
    # Try to extract numeric value for consistency
    try:
        # Check if it's a pure number
        float_val = float(result)
        # If it's a whole number, return as int string
        if float_val == int(float_val):
            return str(int(float_val))
        return result
    except ValueError:
        # Not a number, return as-is (could be "N/A", "None", etc.)
        return result


class TaskAgent:
    """Task agent that solves IMO grading problems with enhanced reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def _build_grading_prompt(self, inputs: dict) -> str:
        """Build a structured prompt for IMO grading with clear reasoning steps."""
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        prompt = f"""You are an expert grader for {domain} problems, specifically trained in IMO (International Mathematical Olympiad) grading standards.

Your task is to evaluate a student's answer based on the official solution and grading guidelines.

## Problem Statement:
{problem}

## Official Solution:
{solution}

## Grading Guidelines:
{guidelines}

## Student's Answer:
{student_answer}

## Instructions:
1. Carefully read the problem, official solution, and grading guidelines
2. Analyze the student's answer step by step
3. Compare the student's approach with the official solution
4. Award partial credit where appropriate based on the guidelines
5. Provide your final grade as a number (typically 0-7 for IMO problems, but follow the guidelines)

Respond in JSON format with the following schema:
<json>
{{
    "reasoning": "Your step-by-step analysis of the student's answer",
    "response": "The final grade (number or appropriate marker)"
}}
</json>

The "response" field must contain only the grade value (e.g., "7", "3", "0", "N/A")."""
        return prompt

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        instruction = self._build_grading_prompt(inputs)
        
        self.log_fn(f"TaskAgent: Processing grading task for domain: {inputs.get('domain', 'Unknown')}")

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON with enhanced error handling
        prediction = "None"
        try:
            extracted = _extract_jsons(msg_history[-1]["text"])
            if extracted:
                last_response = extracted[-1]
                # Try "response" field first, fallback to other common fields
                if "response" in last_response:
                    prediction = last_response["response"]
                elif "grade" in last_response:
                    prediction = last_response["grade"]
                elif "score" in last_response:
                    prediction = last_response["score"]
                elif "answer" in last_response:
                    prediction = last_response["answer"]
                else:
                    # If no recognized field, use the first value
                    prediction = list(last_response.values())[0] if last_response else "None"
                
                # Validate and normalize the response
                prediction = _validate_grading_response(prediction, inputs)
                
                self.log_fn(f"TaskAgent: Extracted grade: {prediction}")
            else:
                self.log_fn("TaskAgent: No JSON blocks found in response")
        except Exception as e:
            self.log_fn(f"TaskAgent: Error extracting prediction: {e}")

        return str(prediction), msg_history
