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
import traceback

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
    
    # First try <json>...</json> tags
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
            # Try to clean up common issues
            try:
                # Remove trailing commas before closing braces/brackets
                cleaned = re.sub(r',(\s*[}\]])', r'\1', inner)
                results.append(json.loads(cleaned))
            except json.JSONDecodeError:
                continue
    
    # Also try markdown code blocks ```json ... ```
    if not results:
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
                try:
                    cleaned = re.sub(r',(\s*[}\]])', r'\1', inner)
                    results.append(json.loads(cleaned))
                except json.JSONDecodeError:
                    continue
    
    return results or None


def _extract_json_fuzzy(text: str) -> list[dict] | None:
    """Fallback JSON extraction for malformed responses.
    
    Attempts to find JSON objects even without proper <json> tags
    by looking for curly brace patterns.
    """
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
                    json_str = text[start_idx:i+1]
                    results.append(json.loads(json_str))
                except json.JSONDecodeError:
                    pass
                start_idx = -1
    return results or None


def _normalize_grade(grade: str) -> str:
    """Normalize grade format for consistency.
    
    Handles various formats like '7/7', '7', 'Full marks', etc.
    Also handles numeric grades that may be integers or floats.
    """
    if not grade or not isinstance(grade, str):
        return str(grade) if grade else "None"
    
    grade = grade.strip()
    
    # Check for fraction format (e.g., "7/7", "3/7", "6.5/7")
    if '/' in grade:
        # Validate the fraction format
        parts = grade.split('/')
        if len(parts) == 2:
            try:
                numerator = float(parts[0].strip())
                denominator = float(parts[1].strip())
                if denominator == 7 and 0 <= numerator <= 7:
                    return f"{numerator}/7"
            except ValueError:
                pass
        return grade
    
    # Check for numeric only (assume out of 7 for IMO)
    try:
        num = float(grade)
        if 0 <= num <= 7:
            return f"{num}/7"
    except ValueError:
        pass
    
    # Check for text-based grades with more comprehensive patterns
    lower = grade.lower()
    
    # Full marks patterns
    full_patterns = ['full', 'complete', 'correct', 'perfect', 'excellent', '7/7', 'seven']
    if any(word in lower for word in full_patterns) or grade == '7':
        return '7/7'
    
    # Zero/none patterns
    zero_patterns = ['zero', 'none', 'incorrect', 'wrong', 'no credit', 'fail', '0/7']
    if any(word in lower for word in zero_patterns) or grade == '0':
        return '0/7'
    
    # Partial credit - try to extract numeric value
    partial_match = re.search(r'(\d+(?:\.\d+)?)\s*/\s*7', lower)
    if partial_match:
        return f"{partial_match.group(1)}/7"
    
    # Look for standalone numbers in text
    num_match = re.search(r'\b([0-7](?:\.\d+)?)\b', grade)
    if num_match:
        return f"{num_match.group(1)}/7"
    
    return grade


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
        # Extract key fields for structured prompting
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = f"""You are an expert IMO (International Mathematical Olympiad) grader. Your task is to evaluate a student's solution to a mathematical problem with precision and consistency.

## Domain
{domain}

## Problem Statement
{problem}

## Official Solution
{solution}

## Grading Guidelines
{grading_guidelines}

## Student's Answer
{student_answer}

## IMO Grading Scale (0-7 points)
- 7/7: Complete, correct solution with clear reasoning
- 6/7: Minor flaw or gap in an otherwise correct solution
- 5/7: Significant progress with one major gap or error
- 4/7: Multiple gaps or errors but substantial progress
- 3/7: Some meaningful progress toward solution
- 2/7: Limited progress with some correct ideas
- 1/7: Minimal progress, mostly incorrect
- 0/7: No meaningful progress or completely wrong

## Your Task
1. **Understand the Problem**: Identify what needs to be proven or solved
2. **Analyze Official Solution**: Note key steps, techniques, and critical insights
3. **Review Grading Guidelines**: Understand partial credit allocation
4. **Evaluate Student's Answer**:
   - Check if the approach is valid (even if different from official)
   - Identify all correct steps and valid insights
   - Note any errors, gaps, or logical flaws
   - Assess completeness of the solution
5. **Determine Grade**: Assign points based on demonstrated understanding and progress
6. **Provide Reasoning**: Explain your evaluation with specific references to the student's work

## Response Format
Respond ONLY in JSON format with this exact schema:
<json>
{{
    "reasoning": "Your detailed step-by-step analysis. Reference specific parts of the student's answer. Explain what they did correctly and where they went wrong. Justify the point allocation.",
    "response": "X/7"
}}
</json>

IMPORTANT: 
- The response field MUST contain ONLY the grade in "X/7" format (e.g., "5/7")
- Do not add extra text, explanations, or labels in the response field
- Be consistent with IMO grading standards"""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON with fallback strategies
        prediction = "None"
        try:
            response_text = msg_history[-1].get("text", "") if msg_history else ""
            
            if not response_text:
                self.log_fn("Empty response text in message history")
                return "None", msg_history
            
            # Try primary extraction method first
            extracted = _extract_jsons(response_text)
            extraction_method = "primary"
            
            # Fallback to fuzzy extraction if primary fails
            if not extracted:
                extracted = _extract_json_fuzzy(response_text)
                if extracted:
                    extraction_method = "fuzzy"
                    self.log_fn("Used fuzzy JSON extraction fallback")
            
            if extracted:
                last_json = extracted[-1]
                # Try multiple possible field names for the grade
                grade_fields = ["response", "grade", "score", "assessment", "result", "evaluation", "points"]
                found_field = None
                for field in grade_fields:
                    if field in last_json:
                        prediction = last_json[field]
                        found_field = field
                        break
                
                if found_field:
                    self.log_fn(f"Found grade in field '{found_field}': {prediction}")
                else:
                    self.log_fn(f"No known grade field found in JSON. Available keys: {list(last_json.keys())}")
                    # Try to use any string value that looks like a grade
                    for key, value in last_json.items():
                        if isinstance(value, str) and ('/' in value or value.isdigit()):
                            prediction = value
                            self.log_fn(f"Using value from '{key}' as grade: {prediction}")
                            break
                
                # Normalize the grade format
                prediction = _normalize_grade(prediction)
                self.log_fn(f"Extracted grade via {extraction_method}: {prediction}")
            else:
                self.log_fn("No JSON found in response, attempting text extraction")
                # Last resort: try to extract grade from text directly
                # Look for patterns like "7/7", "3/7", "6.5/7", etc.
                grade_match = re.search(r'(\d+(?:\.\d+)?)/7', response_text)
                if grade_match:
                    prediction = f"{grade_match.group(1)}/7"
                    self.log_fn(f"Extracted grade via regex: {prediction}")
                else:
                    # Try to find any number that could be a grade
                    num_match = re.search(r'\bgrade\s*(?:is|:)?\s*(\d+(?:\.\d+)?)\b', response_text, re.IGNORECASE)
                    if num_match:
                        prediction = f"{num_match.group(1)}/7"
                        self.log_fn(f"Extracted grade via text pattern: {prediction}")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            self.log_fn(f"Traceback: {traceback.format_exc()}")

        return str(prediction), msg_history
