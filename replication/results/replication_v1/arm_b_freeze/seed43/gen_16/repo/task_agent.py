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


def _extract_any_json(text: str) -> list[dict] | None:
    """Fallback JSON extraction that looks for any JSON objects in text."""
    results = []
    # Try to find JSON objects between curly braces
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


def _validate_grade(prediction: str, grading_guidelines: str) -> tuple[str, bool]:
    """Validate and normalize the grade prediction.
    
    Args:
        prediction: The raw prediction string from the LLM
        grading_guidelines: The grading guidelines to check against
        
    Returns:
        tuple of (normalized_prediction, is_valid)
    """
    if not prediction or prediction == "None":
        return "None", False
    
    prediction = prediction.strip()
    
    # Common grade patterns
    valid_grades = {
        # Binary grades
        "correct": "Correct",
        "incorrect": "Incorrect",
        "right": "Correct",
        "wrong": "Incorrect",
        "true": "Correct",
        "false": "Incorrect",
        "yes": "Correct",
        "no": "Incorrect",
        # Partial grades
        "partial": "Partial",
        "partially correct": "Partial",
        "incomplete": "Partial",
        # Numeric grades (0-10 scale)
        "0": "0",
        "1": "1",
        "2": "2",
        "3": "3",
        "4": "4",
        "5": "5",
        "6": "6",
        "7": "7",
        "8": "8",
        "9": "9",
        "10": "10",
    }
    
    # Check for exact match (case-insensitive)
    pred_lower = prediction.lower()
    if pred_lower in valid_grades:
        return valid_grades[pred_lower], True
    
    # Check if prediction contains a valid grade
    for key, value in valid_grades.items():
        if key in pred_lower:
            return value, True
    
    # Check for numeric patterns in the guidelines
    if any(char.isdigit() for char in prediction):
        # Extract the first number found
        import re as regex
        numbers = regex.findall(r'\d+', prediction)
        if numbers:
            return numbers[0], True
    
    # Return original if no normalization possible
    return prediction, True


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
1. Analyze what the problem is asking
2. Review the official solution approach
3. Compare the student's answer to the official solution
4. Check if the student followed the grading guidelines
5. Determine the appropriate grade

IMPORTANT: Your grade must be one of the following formats:
- Binary: "Correct" or "Incorrect"
- Partial credit: "Partial"
- Numeric: A number from 0-10

Respond in JSON format with the following schema:
<json>
{{
    "reasoning": "Your detailed chain-of-thought analysis here",
    "response": "The final grade/prediction (e.g., 'Correct', 'Incorrect', 'Partial', or a numeric score)"
}}
</json>"""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON with fallback mechanisms
        prediction = "None"
        extraction_method = "none"
        try:
            last_message = msg_history[-1]["text"]
            
            # Try primary extraction method
            extracted = _extract_jsons(last_message)
            extraction_method = "json_tags"
            
            # Fallback to generic JSON extraction if primary fails
            if extracted is None:
                extracted = _extract_any_json(last_message)
                extraction_method = "any_json"
            
            if extracted:
                # Prefer response field, but accept other common field names
                last_json = extracted[-1]
                if "response" in last_json:
                    prediction = last_json["response"]
                elif "grade" in last_json:
                    prediction = last_json["grade"]
                elif "answer" in last_json:
                    prediction = last_json["answer"]
                elif "result" in last_json:
                    prediction = last_json["result"]
                elif "evaluation" in last_json:
                    prediction = last_json["evaluation"]
                else:
                    # If no known field, use the first string value found
                    for key, value in last_json.items():
                        if isinstance(value, str):
                            prediction = value
                            break
                
                # Validate and normalize the grade
                normalized, is_valid = _validate_grade(prediction, grading_guidelines)
                if is_valid:
                    prediction = normalized
                    self.log_fn(f"Grade extracted via {extraction_method}: {prediction}")
                else:
                    self.log_fn(f"Warning: Invalid grade '{prediction}', using normalized: {normalized}")
                    prediction = normalized
            else:
                self.log_fn(f"Warning: No JSON found in response, using 'None'")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
