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

# Priority order for extracting prediction from JSON fields
_PREDICTION_FIELDS = ["response", "grade", "answer", "result", "evaluation", "prediction", "score"]

# Additional fields to check for numeric grades
_NUMERIC_FIELDS = ["score", "grade", "points", "mark", "value"]


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


def _extract_any_json(text: str) -> list[dict] | None:
    """Fallback JSON extraction that looks for any JSON objects in text."""
    results = []
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
                    obj = json.loads(text[start_idx:i+1])
                    results.append(obj)
                except json.JSONDecodeError:
                    pass
                start_idx = -1
    return results or None


def _extract_json_with_regex(text: str) -> list[dict] | None:
    """Additional fallback using regex to find JSON-like structures."""
    results = []
    json_pattern = r'```(?:json)?\s*([\s\S]*?)```'
    matches = re.findall(json_pattern, text)
    for match in matches:
        try:
            obj = json.loads(match.strip())
            if isinstance(obj, dict):
                results.append(obj)
        except json.JSONDecodeError:
            brace_count = 0
            start_idx = -1
            for i, char in enumerate(match):
                if char == '{':
                    if brace_count == 0:
                        start_idx = i
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0 and start_idx != -1:
                        try:
                            obj = json.loads(match[start_idx:i+1])
                            results.append(obj)
                        except json.JSONDecodeError:
                            pass
                        start_idx = -1
    return results or None


def _extract_prediction_from_json(json_obj: dict) -> str | None:
    """Extract prediction value from a JSON object using priority fields.
    
    Args:
        json_obj: A dictionary containing the parsed JSON response
        
    Returns:
        The prediction string if found, None otherwise
    """
    # Check priority fields first
    for field in _PREDICTION_FIELDS:
        if field in json_obj:
            value = json_obj[field]
            if isinstance(value, str):
                return value
            elif isinstance(value, (int, float)):
                return str(value)
    
    # If no priority field found, use first string or numeric value
    for key, value in json_obj.items():
        if isinstance(value, str):
            return value
        elif isinstance(value, (int, float)):
            return str(value)
    
    return None


def _extract_numeric_grade(json_obj: dict) -> float | None:
    """Extract a numeric grade/score from a JSON object.
    
    Args:
        json_obj: A dictionary containing the parsed JSON response
        
    Returns:
        The numeric grade if found, None otherwise
    """
    # Check numeric fields first
    for field in _NUMERIC_FIELDS:
        if field in json_obj:
            value = json_obj[field]
            if isinstance(value, (int, float)):
                return float(value)
            elif isinstance(value, str):
                # Try to parse numeric value from string
                try:
                    # Remove common suffixes/prefixes
                    cleaned = value.strip().lower()
                    cleaned = re.sub(r'[^\d.]', '', cleaned)
                    if cleaned:
                        return float(cleaned)
                except (ValueError, TypeError):
                    pass
    
    # Also check the response field for numeric values
    if "response" in json_obj:
        response = json_obj["response"]
        if isinstance(response, str):
            # Try to extract numeric value from response string
            try:
                # Look for patterns like "7/7", "5 points", "score: 3"
                numeric_match = re.search(r'(\d+(?:\.\d+)?)', response)
                if numeric_match:
                    return float(numeric_match.group(1))
            except (ValueError, TypeError):
                pass
    
    return None


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self._call_count = 0

    def _log_extraction_debug(
        self,
        raw_response: str,
        extracted_jsons: list[dict] | None,
        final_prediction: str,
        domain: str = "Unknown"
    ) -> None:
        """Log debug information about the extraction process.
        
        Args:
            raw_response: The raw LLM response text
            extracted_jsons: List of extracted JSON objects
            final_prediction: The final extracted prediction
            domain: The problem domain for context
        """
        self._call_count += 1
        self.log_fn(
            f"[TaskAgent Call {self._call_count}] Domain: {domain}, "
            f"Response length: {len(raw_response)}, "
            f"JSONs found: {len(extracted_jsons) if extracted_jsons else 0}, "
            f"Prediction: {final_prediction[:50] if final_prediction else 'None'}"
        )

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Extract task components for better prompting
        domain = inputs.get("domain", "Unknown")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = f"""You are an expert mathematical grader for International Mathematical Olympiad (IMO) problems.

Your task is to evaluate a student's answer to a mathematics problem and provide a grade.

DOMAIN: {domain}

PROBLEM:
{problem}

OFFICIAL SOLUTION:
{solution}

GRADING GUIDELINES:
{grading_guidelines}

STUDENT'S ANSWER:
{student_answer}

Think step by step:
1. Analyze what the problem is asking and identify key concepts
2. Review the official solution approach and identify critical steps
3. Compare the student's answer to the official solution - check for:
   - Correctness of the final answer
   - Validity of the reasoning process
   - Completeness of the solution
   - Mathematical rigor and clarity
4. Check if the student followed the grading guidelines precisely
5. Determine the appropriate grade based on the guidelines

IMPORTANT: Your response must be valid JSON wrapped in <json> tags. Do not include any text outside the JSON tags.

Respond in this exact format:
<json>
{{
    "reasoning": "Your detailed chain-of-thought analysis here. Explain your evaluation process clearly.",
    "response": "The final grade/prediction (e.g., 'Correct', 'Incorrect', 'Partial', or a numeric score)"
}}
</json>"""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON with multiple fallback mechanisms
        prediction = "None"
        extracted = None
        try:
            last_message = msg_history[-1]["text"]
            
            # Try primary extraction method (explicit <json> tags)
            extracted = _extract_jsons(last_message)
            
            # Fallback 1: generic JSON extraction from braces
            if extracted is None:
                extracted = _extract_any_json(last_message)
            
            # Fallback 2: regex-based extraction for markdown code blocks
            if extracted is None:
                extracted = _extract_json_with_regex(last_message)
            
            if extracted:
                # Use the last JSON object and extract prediction
                last_json = extracted[-1]
                extracted_prediction = _extract_prediction_from_json(last_json)
                if extracted_prediction is not None:
                    prediction = extracted_prediction
                
                # Also try to extract numeric grade for additional context
                numeric_grade = _extract_numeric_grade(last_json)
                if numeric_grade is not None:
                    self.log_fn(f"Numeric grade extracted: {numeric_grade}")
        except (IndexError, KeyError) as e:
            self.log_fn(f"Error accessing message history: {e}")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        # Log extraction debug info
        self._log_extraction_debug(
            raw_response=msg_history[-1]["text"] if msg_history else "",
            extracted_jsons=extracted,
            final_prediction=prediction,
            domain=domain
        )

        return str(prediction), msg_history
