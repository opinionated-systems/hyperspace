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


def _extract_grade_from_text(text: str) -> str | None:
    """Extract grade from plain text when JSON extraction fails.
    
    Looks for common grade patterns in the text.
    """
    # Common grade patterns
    patterns = [
        r'grade[\s]*[:=][\s]*["\']?([\w\s]+)["\']?',
        r'(?:final\s+)?(?:grade|score|evaluation|result)[\s]*[:=][\s]*["\']?([\w\s\-\+]+)["\']?',
        r'(?:the\s+)?(?:student\s+(?:received|got|earned))?[\s]*["\']?([\w\s\-\+]+)["\']?',
        r'(?:mark|score)[\s]*[:=][\s]*["\']?([\w\s\-\+\d]+)["\']?',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    # Look for standalone grades (Correct, Incorrect, Partial, etc.)
    standalone_patterns = [
        r'\b(Correct|Incorrect|Partial|Partially Correct|Full Credit|No Credit|Zero|Full|Pass|Fail)\b',
        r'\b(\d+[/\s]*\d*)\b',  # Numeric scores like 7/7, 5, etc.
    ]
    
    for pattern in standalone_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    return None


def _normalize_grade(grade: str) -> str:
    """Normalize grade to a standard format."""
    grade_lower = grade.lower().strip()
    
    # Map common variations to standard forms
    correct_variations = ['correct', 'full credit', 'full', 'pass', 'right', 'true', 'yes', '1', '7/7']
    incorrect_variations = ['incorrect', 'no credit', 'zero', 'fail', 'wrong', 'false', 'no', '0', '0/7']
    partial_variations = ['partial', 'partially correct', 'partial credit', 'some credit', 'incomplete']
    
    if any(v in grade_lower for v in correct_variations):
        return 'Correct'
    elif any(v in grade_lower for v in incorrect_variations):
        return 'Incorrect'
    elif any(v in grade_lower for v in partial_variations):
        return 'Partial'
    
    return grade.strip()


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
1. Analyze what the problem is asking and identify key requirements
2. Review the official solution approach and expected outcomes
3. Compare the student's answer to the official solution line by line
4. Check if the student followed the grading guidelines precisely
5. Identify any errors, omissions, or alternative valid approaches
6. Determine the appropriate grade based on the evidence

IMPORTANT: Your grade must be one of: "Correct", "Incorrect", or "Partial".
- "Correct": The answer is fully correct, complete, and follows the solution approach
- "Incorrect": The answer is wrong, incomplete in critical ways, or fundamentally flawed
- "Partial": The answer has some correct elements but is incomplete or has significant errors

Respond in JSON format with the following schema:
<json>
{{
    "reasoning": "Your detailed chain-of-thought analysis here. Be specific about what the student did right and wrong.",
    "response": "Correct" or "Incorrect" or "Partial"
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
            else:
                # Final fallback: try to extract grade from plain text
                text_grade = _extract_grade_from_text(last_message)
                if text_grade:
                    prediction = text_grade
                    extraction_method = "text_pattern"
                    self.log_fn(f"Extracted grade from text pattern: {prediction}")
            
            # Normalize the grade to standard format
            normalized = _normalize_grade(prediction)
            if normalized != prediction:
                self.log_fn(f"Normalized grade: '{prediction}' -> '{normalized}'")
                prediction = normalized
                
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        self.log_fn(f"Final prediction: {prediction} (extracted via {extraction_method})")
        return str(prediction), msg_history
