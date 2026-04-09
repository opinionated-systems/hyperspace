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
        except json.JSONDecodeError as e:
            # Try to extract just the response field if full JSON fails
            try:
                # Look for "response": value pattern (handles numbers, strings, booleans)
                response_match = re.search(r'"response"\s*:\s*([^,\}]+)', inner)
                if response_match:
                    value = response_match.group(1).strip()
                    # Try to parse as number or string
                    try:
                        parsed_value = json.loads(value)
                    except json.JSONDecodeError:
                        # If it's not valid JSON, treat as string
                        parsed_value = value.strip('"')
                    results.append({"response": parsed_value})
            except Exception:
                pass
            continue
    return results or None


def _extract_json_fallback(text: str) -> list[dict] | None:
    """Fallback JSON extraction using regex for unwrapped JSON objects.

    This handles cases where the model outputs valid JSON without <json> tags.
    Searches for JSON objects with a "response" key.
    """
    results = []
    # Pattern to match JSON objects (handles nested braces)
    pattern = r'\{[^{}]*"response"[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.finditer(pattern, text, re.DOTALL)

    for match in matches:
        try:
            obj = json.loads(match.group())
            if "response" in obj:
                results.append(obj)
        except json.JSONDecodeError:
            continue

    # If regex fails, try to find any JSON-like structure
    if not results:
        try:
            # Try to parse the entire text as JSON
            obj = json.loads(text.strip())
            if isinstance(obj, dict) and "response" in obj:
                results.append(obj)
        except json.JSONDecodeError:
            pass

    # Try to find JSON with single quotes instead of double quotes
    if not results:
        single_quote_pattern = r"\{[^{}]*'response'[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
        matches = re.finditer(single_quote_pattern, text, re.DOTALL)
        for match in matches:
            try:
                # Replace single quotes with double quotes for JSON parsing
                json_str = match.group().replace("'", '"')
                obj = json.loads(json_str)
                if "response" in obj:
                    results.append(obj)
            except json.JSONDecodeError:
                continue

    # Last resort: try to extract just a number or string response
    if not results:
        # Look for patterns like "The score is 5" or just a number
        number_match = re.search(r'\b(\d+(?:\.\d+)?)\b', text)
        if number_match:
            try:
                value = float(number_match.group(1))
                # If it's a whole number, convert to int
                if value == int(value):
                    value = int(value)
                results.append({"response": value})
            except ValueError:
                pass

    return results or None


class TaskAgent:
    """Task agent that solves IMO grading problems with robust JSON extraction."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Extract fields for better structured prompting
        domain = inputs.get("domain", "")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = f"""You are an expert mathematics grader specializing in {domain} problems.

Your task is to evaluate a student's answer to a mathematical problem and provide a numerical score.

## Problem Statement:
{problem}

## Correct Solution:
{solution}

## Grading Guidelines:
{grading_guidelines}

## Student's Answer:
{student_answer}

## Instructions:
1. Carefully analyze the student's answer against the correct solution
2. Check if the student has shown their work and reasoning
3. Verify that the final answer is mathematically correct
4. Consider the grading guidelines when determining the score
5. Be precise and objective in your evaluation
6. Provide a numerical score as your response

## Grading Approach:
- Compare the student's answer step-by-step with the correct solution
- Award partial credit if the student made progress but had minor errors
- Consider both the final answer and the reasoning process
- Use the grading guidelines to determine the appropriate score

Respond ONLY in JSON format with the following schema:
<json>
{{
    "response": <numerical_score>
}}
</json>

The response field must contain only the numerical score value (integer or float). Do not include any other text outside the JSON tags."""

        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
        except Exception as e:
            import traceback
            error_details = f"Error calling LLM: {e}\n{traceback.format_exc()}"
            self.log_fn(error_details)
            return "Error: LLM call failed", [{"role": "system", "text": error_details}]

        # Extract prediction from JSON using primary method
        prediction = "None"
        extraction_method = "primary"
        try:
            extracted = _extract_jsons(msg_history[-1]["text"])
            if extracted and "response" in extracted[-1]:
                prediction = extracted[-1]["response"]
            else:
                # Try fallback extraction
                extracted = _extract_json_fallback(msg_history[-1]["text"])
                if extracted and "response" in extracted[-1]:
                    prediction = extracted[-1]["response"]
                    extraction_method = "fallback"
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            # Try fallback on exception
            try:
                extracted = _extract_json_fallback(msg_history[-1]["text"])
                if extracted and "response" in extracted[-1]:
                    prediction = extracted[-1]["response"]
                    extraction_method = "fallback"
            except Exception as fallback_e:
                self.log_fn(f"Fallback extraction also failed: {fallback_e}")

        self.log_fn(f"Extraction method used: {extraction_method}")
        return str(prediction), msg_history
