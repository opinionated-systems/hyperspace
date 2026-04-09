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

# Pre-compiled regex patterns for grade extraction from plain text
_GRADE_PATTERNS = [
    re.compile(r'(?:grade|result|score|verdict|assessment)[:\s]+([\w\s\-\d]+)', re.IGNORECASE),
    re.compile(r'(?:the\s+(?:grade|result|score|verdict)\s+(?:is|should\s+be))[:\s]+([\w\s\-\d]+)', re.IGNORECASE),
    re.compile(r'(?:final\s+)?(?:grade|score)[:\s]*([\w\s\-\d]+)', re.IGNORECASE),
]

logger = logging.getLogger(__name__)

# Priority order for extracting prediction from JSON fields
# Extended to handle various LLM output formats and naming conventions
_PREDICTION_FIELDS = [
    "response", "grade", "answer", "result", "evaluation", 
    "prediction", "score", "output", "verdict", "assessment",
    "decision", "conclusion", "value", "rating", "mark"
]


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
    # Check priority fields first (case-insensitive match)
    json_keys_lower = {k.lower(): k for k in json_obj.keys()}
    for field in _PREDICTION_FIELDS:
        if field in json_obj:
            value = json_obj[field]
            if isinstance(value, str):
                return value.strip()
            elif isinstance(value, (int, float)):
                return str(value)
            elif isinstance(value, bool):
                return "Correct" if value else "Incorrect"
        # Try case-insensitive match
        elif field.lower() in json_keys_lower:
            value = json_obj[json_keys_lower[field.lower()]]
            if isinstance(value, str):
                return value.strip()
            elif isinstance(value, (int, float)):
                return str(value)
            elif isinstance(value, bool):
                return "Correct" if value else "Incorrect"
    
    # If no priority field found, use first string or numeric value
    # Skip 'reasoning' and 'explanation' fields as they are not predictions
    skip_fields = {'reasoning', 'explanation', 'analysis', 'thought', 'thinking', 'chain_of_thought'}
    for key, value in json_obj.items():
        key_lower = key.lower()
        if key_lower in skip_fields:
            continue
        if isinstance(value, str):
            return value.strip()
        elif isinstance(value, (int, float)):
            return str(value)
        elif isinstance(value, bool):
            return "Correct" if value else "Incorrect"
    
    return None


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

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
   - Proper use of mathematical notation and terminology
   - Logical flow and structure of the argument
4. Check if the student followed the grading guidelines precisely
5. Determine the appropriate grade based on the guidelines

GRADING RUBRIC:
- "Correct" (or "1", "7/7", "Full Marks"): The answer is completely correct with valid reasoning
- "Incorrect" (or "0", "0/7", "No Marks"): The answer is fundamentally wrong or missing
- "Partial" (or partial credit like "3/7", "4/7"): The answer has some correct elements but is incomplete or has errors

IMPORTANT: Your response must be valid JSON wrapped in <json> tags. Do not include any text outside the JSON tags.

Respond in this exact format:
<json>
{{
    "reasoning": "Your detailed chain-of-thought analysis here. Explain your evaluation process clearly, citing specific aspects of the student's answer that led to your grade.",
    "response": "The final grade/prediction (e.g., 'Correct', 'Incorrect', 'Partial', '1', '0', '3/7', '7/7')"
}}
</json>"""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON with multiple fallback mechanisms
        prediction = "None"
        extraction_method = "none"
        try:
            if not msg_history:
                self.log_fn("Warning: Empty message history received")
            else:
                last_message = msg_history[-1].get("text", "")
                if not last_message:
                    self.log_fn("Warning: Last message has no text content")
                else:
                    # Try primary extraction method (explicit <json> tags)
                    extracted = _extract_jsons(last_message)
                    extraction_method = "json_tags"
                    
                    # Fallback 1: generic JSON extraction from braces
                    if extracted is None:
                        extracted = _extract_any_json(last_message)
                        extraction_method = "any_json"
                    
                    # Fallback 2: regex-based extraction for markdown code blocks
                    if extracted is None:
                        extracted = _extract_json_with_regex(last_message)
                        extraction_method = "regex"
                    
                    if extracted:
                        # Use the last JSON object and extract prediction
                        last_json = extracted[-1]
                        extracted_prediction = _extract_prediction_from_json(last_json)
                        if extracted_prediction is not None:
                            prediction = extracted_prediction
                            self.log_fn(f"Prediction extracted via {extraction_method}: {prediction[:100]}...")
                        else:
                            self.log_fn(f"Warning: Could not extract prediction from JSON fields: {list(last_json.keys())}")
                    else:
                        # Final fallback: try to extract any meaningful text from the response
                        # Look for common patterns like "Grade: X" or "Result: Y"
                        for pattern in _GRADE_PATTERNS:
                            match = pattern.search(last_message)
                            if match:
                                prediction = match.group(1).strip()
                                self.log_fn(f"Prediction extracted via pattern matching: {prediction[:100]}...")
                                break
                        else:
                            self.log_fn(f"Warning: No JSON found in response. Response preview: {last_message[:200]}...")
        except (IndexError, KeyError) as e:
            self.log_fn(f"Error accessing message history: {e}")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {type(e).__name__}: {e}")

        return str(prediction), msg_history
