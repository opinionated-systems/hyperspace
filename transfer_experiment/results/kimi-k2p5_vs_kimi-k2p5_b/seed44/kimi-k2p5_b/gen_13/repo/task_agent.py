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
    Handles nested braces correctly and supports multi-field JSON objects.
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

    # Also try to find ```json code blocks
    search_from = 0
    while True:
        start = text.find("```json", search_from)
        if start == -1:
            break
        end = text.find("```", start + 7)
        if end == -1:
            break
        inner = text[start + 7:end].strip()
        search_from = end + 3
        try:
            results.append(json.loads(inner))
        except json.JSONDecodeError:
            continue

    # Also try to find ``` code blocks without json tag
    search_from = 0
    while True:
        start = text.find("```", search_from)
        if start == -1:
            break
        # Skip ```json blocks we already processed
        if text[start:start+7] == "```json":
            search_from = start + 7
            continue
        end = text.find("```", start + 3)
        if end == -1:
            break
        inner = text[start + 3:end].strip()
        search_from = end + 3
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


def _extract_all_grades_from_text(text: str) -> list[str]:
    """Extract all valid grade mentions from text, returning them in order of appearance."""
    valid_grades = ["correct", "incorrect", "partial", "almost"]
    found_grades = []
    text_lower = text.lower()
    
    # Find all occurrences of grade words
    for grade in valid_grades:
        # Look for the grade as a whole word
        for match in re.finditer(r'\b' + grade + r'\b', text_lower):
            found_grades.append((match.start(), grade))
    
    # Sort by position and return grades
    found_grades.sort(key=lambda x: x[0])
    return [g for _, g in found_grades]


def _extract_grade_from_text(text: str) -> str | None:
    """Extract grade from various formats in the text.
    
    Uses multiple strategies in order of reliability:
    1. JSON field patterns (most reliable)
    2. Explicit grade assignment patterns
    3. Last occurrence in text (conclusion usually at end)
    """
    valid_grades = ['correct', 'incorrect', 'partial', 'almost']
    text_lower = text.lower()
    
    # Priority 1: Look for JSON grade field patterns
    json_patterns = [
        r'"grade"\s*:\s*"([^"]*)"',
        r'"grade"\s*:\s*\'([^\']*)\'',
        r'"grade"\s*:\s*([a-zA-Z]+)',
        r'"response"\s*:\s*"([^"]*)"',
        r'"prediction"\s*:\s*"([^"]*)"',
        r'"result"\s*:\s*"([^"]*)"',
    ]
    
    for pattern in json_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            grade = match.group(1).strip().lower().strip('"').strip("'")
            if grade in valid_grades:
                return grade
    
    # Priority 2: Look for explicit grade assignments in text
    text_patterns = [
        r'\bfinal\s+grade\s*[:=]\s*([a-zA-Z]+)',
        r'\bgrade\s*[:=]\s*([a-zA-Z]+)',
        r'\bassigned\s+grade\s*[:=]\s*([a-zA-Z]+)',
        r'\bthe\s+grade\s+is\s+([a-zA-Z]+)',
        r'\bi\s+assign\s+([a-zA-Z]+)',
        r'\bthis\s+is\s+([a-zA-Z]+)\b',
    ]
    
    for pattern in text_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            grade = match.group(1).strip().lower()
            if grade in valid_grades:
                return grade
    
    # Priority 3: Look for the LAST occurrence of grade keywords
    # The conclusion typically appears at the end of the response
    # Search from the end for the most reliable grade
    last_grades = _extract_all_grades_from_text(text)
    if last_grades:
        # Return the last grade mentioned (most likely the conclusion)
        return last_grades[-1]
    
    return None


def _normalize_grade(grade: str) -> str:
    """Normalize grade to one of the expected lowercase values."""
    if not grade:
        return "none"
    
    grade_lower = grade.lower().strip().strip('"').strip("'")
    
    # Map to standard grades
    if grade_lower in ["correct", "incorrect", "partial", "almost"]:
        return grade_lower
    
    # Partial matches as fallback
    if "almost" in grade_lower:
        return "almost"
    elif "partial" in grade_lower:
        return "partial"
    elif "incorrect" in grade_lower:
        return "incorrect"
    elif "correct" in grade_lower:
        return "correct"
    
    return "none"


def _extract_confidence(json_obj: dict) -> str:
    """Extract confidence value from a JSON object.
    
    Returns one of: "high", "medium", "low", or "unknown"
    """
    if not isinstance(json_obj, dict):
        return "unknown"
    
    # Try common confidence field names
    for field in ["confidence", "confidence_level", "certainty", "sureness"]:
        if field in json_obj:
            val = str(json_obj[field]).lower().strip().strip('"').strip("'")
            if val in ["high", "medium", "low"]:
                return val
    
    return "unknown"


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

        instruction = f"""You are an expert mathematical olympiad grader. Evaluate the student's solution and assign exactly one grade.

## Problem Domain
{domain if domain else "Mathematical Olympiad"}

## Problem Statement:
{problem}

## Official Solution:
{solution}

## Grading Guidelines (FOLLOW THESE EXACTLY):
{grading_guidelines}

## Student's Answer:
{student_answer}

## Grade Definitions

- **correct**: Complete and fully correct solution. All steps present and logically sound.
- **incorrect**: No meaningful progress or fundamental errors. Wrong approach or no valid mathematical progress.
- **partial**: Significant progress but incomplete. Found useful invariant/lemma or established valid approach but didn't finish.
- **almost**: Nearly complete with only minor issues. Main proof structure correct, only small computational errors or omissions.

## Grading Instructions

1. Read the grading guidelines carefully - they explicitly list what constitutes each grade level
2. Check what the student proved against the guidelines:
   - Items under (Partial) proven → grade "partial"
   - Items under (Almost) proven → grade "almost"
   - Items under (Correct) proven → grade "correct"
   - No meaningful progress → grade "incorrect"
3. Your final grade must match the explicit criteria in the guidelines

## Response Format

Respond ONLY with a JSON object. The grade field must be exactly one of: "correct", "incorrect", "partial", or "almost".

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
        prediction = "none"
        confidence = "unknown"
        valid_grades = ["correct", "incorrect", "partial", "almost"]
        
        try:
            last_message = msg_history[-1]["text"]
            
            # Strategy 1: Extract from JSON blocks
            extracted = _extract_jsons(last_message)
            if extracted:
                for json_obj in reversed(extracted):  # Check from last to first
                    if isinstance(json_obj, dict):
                        # Try common grade field names
                        for field in ["grade", "response", "result", "prediction"]:
                            if field in json_obj:
                                val = str(json_obj[field]).lower().strip().strip('"').strip("'")
                                if val in valid_grades:
                                    prediction = val
                                    confidence = _extract_confidence(json_obj)
                                    break
                        if prediction in valid_grades:
                            break
                        # If no standard field found, try any value except reasoning
                        if prediction == "none":
                            for key, val in json_obj.items():
                                if key == "reasoning":
                                    continue
                                val_str = str(val).lower().strip().strip('"').strip("'")
                                if val_str in valid_grades:
                                    prediction = val_str
                                    confidence = _extract_confidence(json_obj)
                                    break
            
            # Strategy 2: Use text-based extraction if JSON didn't work
            if prediction not in valid_grades:
                extracted_text = _extract_grade_from_text(last_message)
                if extracted_text:
                    prediction = extracted_text
            
            # Strategy 3: Direct pattern matching for quoted grades
            if prediction not in valid_grades:
                text_lower = last_message.lower()
                # Look for quoted grades - check in order of specificity
                quoted_patterns = [
                    ('"almost"', "almost"), ("'almost'", "almost"),
                    ('"partial"', "partial"), ("'partial'", "partial"),
                    ('"incorrect"', "incorrect"), ("'incorrect'", "incorrect"),
                    ('"correct"', "correct"), ("'correct'", "correct"),
                ]
                for pattern, grade in quoted_patterns:
                    if pattern in text_lower:
                        prediction = grade
                        break
            
            # Log extraction results for debugging
            if prediction in valid_grades:
                self.log_fn(f"Extracted grade: {prediction}, confidence: {confidence}")
            else:
                self.log_fn(f"Failed to extract valid grade. Raw response preview: {last_message[:200]}...")
                        
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        # Normalize prediction to expected values
        prediction = _normalize_grade(prediction)

        return str(prediction), msg_history
