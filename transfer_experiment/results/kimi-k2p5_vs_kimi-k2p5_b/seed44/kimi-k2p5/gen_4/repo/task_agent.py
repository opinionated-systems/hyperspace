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
    """Extract JSON objects from <json>...</json> blocks or raw JSON.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    Also tries to find raw JSON objects if tags are not present.
    """
    results = []
    search_from = 0
    
    # First try to find <json>...</json> blocks
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
    
    # If no results, try to find raw JSON objects (looking for {...})
    if not results:
        # Find JSON objects by looking for balanced braces
        i = 0
        while i < len(text):
            if text[i] == '{':
                # Try to find the matching closing brace
                brace_count = 1
                j = i + 1
                while j < len(text) and brace_count > 0:
                    if text[j] == '{':
                        brace_count += 1
                    elif text[j] == '}':
                        brace_count -= 1
                    j += 1
                
                if brace_count == 0:
                    json_str = text[i:j].strip()
                    try:
                        results.append(json.loads(json_str))
                    except json.JSONDecodeError:
                        pass
                i = j
            else:
                i += 1
    
    return results or None


def _extract_grade_from_text(text: str) -> str | None:
    """Extract grade from various formats in the text."""
    # Look for grade patterns in the text - try multiple field names
    patterns = [
        r'"grade"\s*:\s*"([^"]*)"',
        r'"response"\s*:\s*"([^"]*)"',
        r'"classification"\s*:\s*"([^"]*)"',
        r'"evaluation"\s*:\s*"([^"]*)"',
        r'"result"\s*:\s*"([^"]*)"',
        r'grade[\s:]+([\w]+)',
        r'Grade[\s:]+([\w]+)',
        r'Classification[\s:]+([\w]+)',
        r'Evaluation[\s:]+([\w]+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    return None


def _normalize_grade(grade: str) -> str:
    """Normalize grade to one of the expected values."""
    if not grade:
        return "None"
    
    grade_lower = grade.lower().strip().strip('"').strip("'")
    
    # Map to standard grades - check for exact matches first
    if grade_lower == "correct":
        return "correct"
    elif grade_lower == "incorrect":
        return "incorrect"
    elif grade_lower == "partial":
        return "partial"
    elif grade_lower == "almost":
        return "almost"
    
    # Then check for partial matches
    if "incorrect" in grade_lower:
        return "incorrect"
    elif "almost" in grade_lower:
        return "almost"
    elif "partial" in grade_lower:
        return "partial"
    elif "correct" in grade_lower:
        return "correct"
    
    # If no match, return the original (lowercased)
    return grade_lower


class TaskAgent:
    """Task agent that solves IMO grading problems."""

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
        # Extract fields from inputs for better prompt construction
        domain = inputs.get("domain", "")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = f"""You are an expert mathematical olympiad grader. Your task is to evaluate a student's solution to a competition mathematics problem.

## Problem Domain
{domain if domain else "Mathematical Olympiad"}

## Problem Statement:
{problem}

## Official Solution:
{solution}

## Grading Guidelines:
{grading_guidelines}

## Student's Answer:
{student_answer}

## Your Task:
Evaluate the student's answer and assign exactly one of the following grades:
- **correct**: The student's answer is completely correct, complete, and proves the result.
- **incorrect**: The student's answer is wrong, contains fundamental errors, or shows no meaningful progress.
- **partial**: The student made significant progress toward the solution but the answer is incomplete, has gaps, or misses key steps.
- **almost**: The solution is nearly correct with only minor mistakes, small omissions, or slight errors that don't significantly affect the overall correctness.

## Instructions:
1. First, carefully read the problem and official solution to understand what is required.
2. Study the grading guidelines carefully - they contain specific criteria for each grade level.
3. Analyze the student's answer step by step:
   - Check if the approach is correct
   - Verify calculations and reasoning
   - Identify any errors or gaps
   - Assess completeness of the solution
4. Compare the student's work against the grading guidelines to determine the appropriate grade.
5. Provide your grade in the JSON format below.

## Response Format:
Respond ONLY with a JSON object in the following format (no other text):
<json>
{{
    "grade": "correct" | "incorrect" | "partial" | "almost"
}}
</json>

Grade:"""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON
        prediction = "None"
        try:
            last_message = msg_history[-1]["text"]
            extracted = _extract_jsons(last_message)
            if extracted:
                last_json = extracted[-1]
                if isinstance(last_json, dict):
                    # Try to get grade first, then response as fallback
                    if "grade" in last_json:
                        prediction = last_json["grade"]
                    elif "response" in last_json:
                        prediction = last_json["response"]
                    else:
                        # Try to get any value from the JSON as prediction
                        prediction = str(list(last_json.values())[0]) if last_json else "None"
            else:
                # Fallback: try to extract grade from text patterns
                extracted_text = _extract_grade_from_text(last_message)
                if extracted_text:
                    prediction = extracted_text
                else:
                    # Last resort: look for the keywords directly in the text
                    text_lower = last_message.lower()
                    # Check for quoted versions first (more precise)
                    if '"almost"' in text_lower:
                        prediction = "almost"
                    elif '"partial"' in text_lower:
                        prediction = "partial"
                    elif '"incorrect"' in text_lower:
                        prediction = "incorrect"
                    elif '"correct"' in text_lower:
                        prediction = "correct"
                    # Then check for unquoted versions
                    elif 'almost' in text_lower:
                        prediction = "almost"
                    elif 'partial' in text_lower:
                        prediction = "partial"
                    elif 'incorrect' in text_lower:
                        prediction = "incorrect"
                    elif 'correct' in text_lower:
                        prediction = "correct"
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        # Normalize prediction to expected values
        prediction = _normalize_grade(prediction)

        return str(prediction), msg_history
