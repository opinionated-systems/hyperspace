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
                # Look for "response": value pattern with more flexible matching
                response_match = re.search(r'"response"\s*:\s*([^,\}]+|"[^"]*")', inner, re.DOTALL)
                if response_match:
                    value = response_match.group(1).strip()
                    # Try to parse as number or string
                    try:
                        parsed_value = json.loads(value)
                    except json.JSONDecodeError:
                        # If it's not valid JSON, treat as string
                        parsed_value = value.strip('"').strip("'")
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
    
    # First, try to find JSON objects with balanced braces
    # This pattern looks for JSON objects containing "response" key
    # It handles nested braces by counting them
    def find_json_objects(s):
        """Find all JSON objects in string with balanced braces."""
        objects = []
        i = 0
        while i < len(s):
            if s[i] == '{':
                start = i
                brace_count = 1
                i += 1
                while i < len(s) and brace_count > 0:
                    if s[i] == '{':
                        brace_count += 1
                    elif s[i] == '}':
                        brace_count -= 1
                    i += 1
                if brace_count == 0:
                    objects.append(s[start:i])
            else:
                i += 1
        return objects
    
    # Try to find JSON objects with "response" key
    json_objects = find_json_objects(text)
    for obj_str in json_objects:
        if '"response"' in obj_str or "'response'" in obj_str:
            try:
                obj = json.loads(obj_str)
                if "response" in obj:
                    results.append(obj)
            except json.JSONDecodeError:
                # Try to extract just the response field
                try:
                    response_match = re.search(r'"response"\s*:\s*([^,\}]+|"[^"]*")', obj_str, re.DOTALL)
                    if response_match:
                        value = response_match.group(1).strip()
                        try:
                            parsed_value = json.loads(value)
                        except json.JSONDecodeError:
                            parsed_value = value.strip('"').strip("'")
                        results.append({"response": parsed_value})
                except Exception:
                    pass
                continue

    # If no results, try to parse the entire text as JSON
    if not results:
        try:
            obj = json.loads(text.strip())
            if isinstance(obj, dict) and "response" in obj:
                results.append(obj)
        except json.JSONDecodeError:
            pass

    # Last resort: try to extract just a number or string response
    if not results:
        # Look for patterns like "The score is 5" or just a number
        # Try to find a number that appears to be a score (0-10 range typically)
        number_matches = list(re.finditer(r'\b(\d+(?:\.\d+)?)\b', text))
        for number_match in number_matches:
            try:
                value = float(number_match.group(1))
                # If it's a whole number, convert to int
                if value == int(value):
                    value = int(value)
                results.append({"response": value})
                break  # Take the first valid number found
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
2. Consider the grading guidelines when determining the score
3. Provide a numerical score as your response
4. Be precise and objective in your evaluation
5. The score should be a number (integer or decimal)

Respond ONLY in the following JSON format. Do not include any other text:
<json>
{{
    "response": <numerical_score>
}}
</json>

The response field must contain only the numerical score value (e.g., 5, 7.5, 10, 0)."""

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
        response_text = msg_history[-1]["text"] if msg_history else ""
        
        try:
            extracted = _extract_jsons(response_text)
            if extracted and "response" in extracted[-1]:
                prediction = extracted[-1]["response"]
            else:
                # Try fallback extraction
                extracted = _extract_json_fallback(response_text)
                if extracted and "response" in extracted[-1]:
                    prediction = extracted[-1]["response"]
                    extraction_method = "fallback"
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            # Try fallback on exception
            try:
                extracted = _extract_json_fallback(response_text)
                if extracted and "response" in extracted[-1]:
                    prediction = extracted[-1]["response"]
                    extraction_method = "fallback"
            except Exception as fallback_e:
                self.log_fn(f"Fallback extraction also failed: {fallback_e}")

        # Validate the prediction is a valid number
        try:
            if isinstance(prediction, (int, float)):
                prediction = str(prediction)
            elif isinstance(prediction, str):
                # Try to convert to number to validate
                float(prediction)
        except (ValueError, TypeError):
            self.log_fn(f"Invalid prediction value: {prediction}")
            # Try one more extraction attempt looking for any number
            number_match = re.search(r'\b(\d+(?:\.\d+)?)\b', str(prediction))
            if number_match:
                prediction = number_match.group(1)
                extraction_method = "number_regex"
            else:
                prediction = "None"

        self.log_fn(f"Extraction method used: {extraction_method}, prediction: {prediction}")
        return str(prediction), msg_history
