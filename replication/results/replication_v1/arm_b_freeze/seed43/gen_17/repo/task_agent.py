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
    Also handles markdown code blocks with json tag.
    """
    results = []
    search_from = 0
    
    # First try explicit <json> tags
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
    
    # Also try markdown code blocks with json
    if not results:
        search_from = 0
        while True:
            # Look for ```json or ``` blocks
            start = text.find("```json", search_from)
            if start == -1:
                start = text.find("```", search_from)
                if start == -1:
                    break
                end_marker = text.find("```", start + 3)
            else:
                end_marker = text.find("```", start + 7)
            
            if end_marker == -1:
                break
                
            # Extract content between markers
            if text[start:start+7] == "```json":
                inner = text[start + 7:end_marker].strip()
            else:
                inner = text[start + 3:end_marker].strip()
            
            search_from = end_marker + 3
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
    
    # Common grade patterns - expanded for better coverage
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
        "valid": "Correct",
        "invalid": "Incorrect",
        "accepted": "Correct",
        "rejected": "Incorrect",
        "pass": "Correct",
        "fail": "Incorrect",
        # Partial grades
        "partial": "Partial",
        "partially correct": "Partial",
        "incomplete": "Partial",
        "partial credit": "Partial",
        "half correct": "Partial",
        "mostly correct": "Partial",
        "mostly wrong": "Partial",
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
        # Letter grades
        "a": "Correct",
        "b": "Partial",
        "c": "Partial",
        "d": "Incorrect",
        "f": "Incorrect",
    }
    
    # Check for exact match (case-insensitive)
    pred_lower = prediction.lower()
    if pred_lower in valid_grades:
        return valid_grades[pred_lower], True
    
    # Check if prediction contains a valid grade as a whole word
    import re as regex
    for key, value in valid_grades.items():
        # Use word boundary matching for more precise detection
        pattern = r'\b' + regex.escape(key) + r'\b'
        if regex.search(pattern, pred_lower):
            return value, True
    
    # Check for numeric patterns - extract first valid number (0-10)
    numbers = regex.findall(r'\b(\d+)\b', prediction)
    if numbers:
        for num_str in numbers:
            num = int(num_str)
            if 0 <= num <= 10:
                return str(num), True
    
    # Check for decimal scores (e.g., "7.5", "8.5")
    decimals = regex.findall(r'\b(\d+\.\d+)\b', prediction)
    if decimals:
        for dec_str in decimals:
            num = float(dec_str)
            if 0 <= num <= 10:
                return dec_str, True
    
    # Check for fraction patterns (e.g., "1/2", "3/4")
    fractions = regex.findall(r'\b(\d+/\d+)\b', prediction)
    if fractions:
        return fractions[0], True
    
    # Check for percentage patterns (e.g., "50%", "100%")
    percentages = regex.findall(r'\b(\d+)%', prediction)
    if percentages:
        pct = int(percentages[0])
        if 0 <= pct <= 100:
            # Convert percentage to 0-10 scale
            converted = str(round(pct / 10))
            return converted, True
    
    # Return original if no normalization possible, but mark as potentially invalid
    return prediction, False


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
                    # If validation failed, try to extract from reasoning text as fallback
                    self.log_fn(f"Warning: Grade '{prediction}' not in standard format, attempting fallback extraction")
                    # Look for grade patterns in the reasoning field if available
                    if "reasoning" in last_json and isinstance(last_json["reasoning"], str):
                        reasoning = last_json["reasoning"]
                        # Try to find a grade mention in the reasoning
                        fallback, fallback_valid = _validate_grade(reasoning, grading_guidelines)
                        if fallback_valid and fallback != "None":
                            prediction = fallback
                            self.log_fn(f"Grade extracted from reasoning: {prediction}")
                        else:
                            self.log_fn(f"Warning: Could not validate grade, using: {normalized}")
                            prediction = normalized
                    else:
                        self.log_fn(f"Warning: Invalid grade '{prediction}', using: {normalized}")
                        prediction = normalized
            else:
                self.log_fn(f"Warning: No JSON found in response, using 'None'")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
