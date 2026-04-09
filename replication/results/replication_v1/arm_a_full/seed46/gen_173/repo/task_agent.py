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

# Maximum input lengths to prevent context overflow
MAX_PROBLEM_LENGTH = 10000
MAX_SOLUTION_LENGTH = 15000
MAX_STUDENT_ANSWER_LENGTH = 15000
MAX_GRADING_GUIDELINES_LENGTH = 5000


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    Also attempts to extract JSON from markdown code blocks as fallback.
    Includes enhanced extraction for nested JSON structures.
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
            # Try to fix common JSON issues
            try:
                # Remove trailing commas before closing braces/brackets
                fixed = re.sub(r',(\s*[}\]])', r'\1', inner)
                results.append(json.loads(fixed))
            except json.JSONDecodeError:
                continue
    
    # Fallback: try to extract JSON from markdown code blocks
    if not results:
        search_from = 0
        while True:
            start = text.find("```json", search_from)
            if start == -1:
                # Try without language specifier
                start = text.find("```", search_from)
                if start == -1:
                    break
                end_marker = "```"
                start += 3
            else:
                end_marker = "```"
                start += 7
            
            end = text.find(end_marker, start)
            if end == -1:
                break
            inner = text[start:end].strip()
            search_from = end + 3
            try:
                results.append(json.loads(inner))
            except json.JSONDecodeError:
                # Try to fix common JSON issues
                try:
                    fixed = re.sub(r',(\s*[}\]])', r'\1', inner)
                    results.append(json.loads(fixed))
                except json.JSONDecodeError:
                    continue
    
    # Fallback: try to find JSON objects directly in the text
    if not results:
        # Look for JSON-like structures with curly braces
        brace_start = text.find("{")
        while brace_start != -1:
            # Try to find matching closing brace
            brace_count = 0
            for i, char in enumerate(text[brace_start:]):
                if char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        json_str = text[brace_start:brace_start + i + 1]
                        try:
                            parsed = json.loads(json_str)
                            # Only accept if it has expected fields
                            if any(key in parsed for key in ["response", "grade", "score", "analysis"]):
                                results.append(parsed)
                        except json.JSONDecodeError:
                            # Try to fix common JSON issues
                            try:
                                fixed = re.sub(r',(\s*[}\]])', r'\1', json_str)
                                parsed = json.loads(fixed)
                                if any(key in parsed for key in ["response", "grade", "score", "analysis"]):
                                    results.append(parsed)
                            except json.JSONDecodeError:
                                pass
                        break
            brace_start = text.find("{", brace_start + 1)
    
    return results or None


def _validate_and_truncate_inputs(inputs: dict, log_fn) -> dict:
    """Validate and truncate input fields to prevent context overflow.
    
    Args:
        inputs: Dictionary containing problem, solution, student_answer, grading_guidelines
        log_fn: Logging function for warnings
        
    Returns:
        Validated and potentially truncated inputs dictionary
    """
    validated = {}
    
    # Required fields
    required_fields = ["problem", "solution", "student_answer"]
    for field in required_fields:
        if field not in inputs or not inputs[field]:
            log_fn(f"Warning: Missing required field '{field}'")
            validated[field] = ""
        else:
            validated[field] = inputs[field]
    
    # Optional fields with defaults
    validated["domain"] = inputs.get("domain", "Mathematics")
    validated["grading_guidelines"] = inputs.get("grading_guidelines", "")
    
    # Truncate fields if too long
    truncation_log = []
    
    if len(validated["problem"]) > MAX_PROBLEM_LENGTH:
        truncation_log.append(f"problem: {len(validated['problem'])} -> {MAX_PROBLEM_LENGTH}")
        validated["problem"] = validated["problem"][:MAX_PROBLEM_LENGTH] + "\n... [truncated]"
    
    if len(validated["solution"]) > MAX_SOLUTION_LENGTH:
        truncation_log.append(f"solution: {len(validated['solution'])} -> {MAX_SOLUTION_LENGTH}")
        validated["solution"] = validated["solution"][:MAX_SOLUTION_LENGTH] + "\n... [truncated]"
    
    if len(validated["student_answer"]) > MAX_STUDENT_ANSWER_LENGTH:
        truncation_log.append(f"student_answer: {len(validated['student_answer'])} -> {MAX_STUDENT_ANSWER_LENGTH}")
        validated["student_answer"] = validated["student_answer"][:MAX_STUDENT_ANSWER_LENGTH] + "\n... [truncated]"
    
    if len(validated["grading_guidelines"]) > MAX_GRADING_GUIDELINES_LENGTH:
        truncation_log.append(f"grading_guidelines: {len(validated['grading_guidelines'])} -> {MAX_GRADING_GUIDELINES_LENGTH}")
        validated["grading_guidelines"] = validated["grading_guidelines"][:MAX_GRADING_GUIDELINES_LENGTH] + "\n... [truncated]"
    
    if truncation_log:
        log_fn(f"Warning: Truncated fields due to length: {', '.join(truncation_log)}")
    
    return validated


def _extract_grade_from_text_fallback(text: str) -> str | None:
    """Extract grade from plain text as a last resort fallback.
    
    Args:
        text: The text to search for grade indicators
        
    Returns:
        Extracted grade string or None if not found
    """
    text_lower = text.lower()
    
    # Look for explicit grade/score patterns
    patterns = [
        r'(?:grade|score|rating|evaluation|result)\s*[:=]\s*["\']?([^"\'\n]{1,20})["\']?',
        r'(?:final\s+)?(?:grade|score)\s+(?:is\s+)?["\']?([^"\'\n]{1,20})["\']?',
        r'(?:assigned|given|awarded)\s+(?:a\s+)?(?:grade|score)\s+(?:of\s+)?["\']?([^"\'\n]{1,20})["\']?',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text_lower, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    # Look for standalone numeric grades 0-7
    numeric_match = re.search(r'\b([0-7])\s*(?:points?|/\s*7)?\b', text_lower)
    if numeric_match:
        return numeric_match.group(1)
    
    # Look for grade keywords
    if re.search(r'\b(correct|right|true|full|complete|solved)\b', text_lower):
        return "Correct"
    if re.search(r'\b(partial|partially|incomplete|minor error|mostly)\b', text_lower):
        return "Partial"
    if re.search(r'\b(incorrect|wrong|false|none|zero|invalid|fail)\b', text_lower):
        return "Incorrect"
    
    return None


def _normalize_grade(grade: str) -> str:
    """Normalize grade to standard format.
    
    Converts various grade formats to a consistent representation.
    Handles both string grades (Correct/Partial/Incorrect) and numeric scores (0-7).
    Includes enhanced handling for edge cases and ambiguous responses.
    """
    if not grade:
        return "None"
    
    # Handle non-string types (int, float, etc.)
    if not isinstance(grade, str):
        grade = str(grade)
    
    grade = grade.strip()
    grade_lower = grade.lower()
    
    # Map common variations to standard grades (using lowercase keys)
    grade_map = {
        "correct": "Correct",
        "right": "Correct",
        "true": "Correct",
        "yes": "Correct",
        "full": "Correct",
        "full credit": "Correct",
        "complete": "Correct",
        "solved": "Correct",
        "valid": "Correct",
        "acceptable": "Correct",
        "partial": "Partial",
        "partially correct": "Partial",
        "partial credit": "Partial",
        "partially": "Partial",
        "some credit": "Partial",
        "incomplete": "Partial",
        "mostly correct": "Partial",
        "minor errors": "Partial",
        "incorrect": "Incorrect",
        "wrong": "Incorrect",
        "false": "Incorrect",
        "no": "Incorrect",
        "none": "Incorrect",
        "zero": "Incorrect",
        "invalid": "Incorrect",
        "unacceptable": "Incorrect",
        "0": "0",
        "1": "1",
        "2": "2",
        "3": "3",
        "4": "4",
        "5": "5",
        "6": "6",
        "7": "7",
    }
    
    # Check for exact match first (case-insensitive)
    if grade_lower in grade_map:
        return grade_map[grade_lower]
    
    # Try to extract numeric score from patterns like "Score: 5", "Grade: 3", "5/7", "(6)"
    # Pattern for standalone digit 0-7
    numeric_match = re.search(r'\b([0-7])\b', grade)
    if numeric_match:
        return numeric_match.group(1)
    
    # Pattern for "X points" or "X out of 7"
    points_match = re.search(r'(\d+)\s*(?:points?|pts?|/\s*7|out\s+of\s+7)', grade_lower)
    if points_match:
        score = int(points_match.group(1))
        if 0 <= score <= 7:
            return str(score)
    
    # Check for grade in parentheses or brackets
    bracket_match = re.search(r'[\(\[]([0-7]|correct|partial|incorrect)[\)\]]', grade_lower)
    if bracket_match:
        inner = bracket_match.group(1)
        if inner in grade_map:
            return grade_map[inner]
        if inner.isdigit() and 0 <= int(inner) <= 7:
            return inner
    
    # Check if grade contains keywords (order matters - check partial first)
    if "partial" in grade_lower or "partially" in grade_lower:
        return "Partial"
    if any(word in grade_lower for word in ["incomplete", "unfinished", "missing", "minor error", "mostly"]):
        return "Partial"
    if any(word in grade_lower for word in ["correct", "right", "true", "full", "complete", "solved", "valid", "acceptable"]):
        return "Correct"
    if any(word in grade_lower for word in ["incorrect", "wrong", "false", "none", "zero", "error", "invalid", "unacceptable", "fail"]):
        return "Incorrect"
    
    # Default: capitalize first letter
    return grade.capitalize()


def _log_extraction_diagnostics(log_fn, last_assistant_msg: str, extracted: list | None, prediction: str) -> None:
    """Log detailed diagnostics about grade extraction for debugging.
    
    This helps identify why grade extraction might be failing and
    provides context for improving the extraction logic.
    """
    log_fn(f"=== Grade Extraction Diagnostics ===")
    log_fn(f"Message length: {len(last_assistant_msg)} chars")
    log_fn(f"JSON blocks found: {len(extracted) if extracted else 0}")
    
    # Check for common formatting issues
    if "<json>" in last_assistant_msg and "</json>" not in last_assistant_msg:
        log_fn("WARNING: Opening <json> tag found but no closing </json> tag")
    if "</json>" in last_assistant_msg and "<json>" not in last_assistant_msg:
        log_fn("WARNING: Closing </json> tag found but no opening <json> tag")
    
    # Check for markdown code blocks
    if "```json" in last_assistant_msg:
        log_fn("Found markdown JSON code block (```json)")
    elif "```" in last_assistant_msg:
        log_fn("Found markdown code block without json specifier")
    
    # Log final prediction
    log_fn(f"Final prediction after normalization: {prediction}")
    log_fn(f"=== End Diagnostics ===")


def _validate_grading_response(extracted: dict, log_fn) -> tuple[bool, str]:
    """Validate that the extracted grading response has required fields.
    
    Returns:
        (is_valid, error_message): Tuple indicating if response is valid
        and an error message if not.
    """
    required_fields = ["response"]
    recommended_fields = ["analysis", "understanding"]
    
    # Check required fields
    for field in required_fields:
        if field not in extracted:
            return False, f"Missing required field: {field}"
    
    # Check recommended fields (log warning but don't fail)
    for field in recommended_fields:
        if field not in extracted:
            log_fn(f"Note: Missing recommended field: {field}")
    
    # Validate response field is not empty
    response_val = extracted.get("response", "")
    if not response_val or str(response_val).strip() == "":
        return False, "Empty response field"
    
    return True, ""


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with structured reasoning.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Validate and truncate inputs to prevent context overflow
        validated = _validate_and_truncate_inputs(inputs, self.log_fn)
        
        # Extract fields for structured prompting
        domain = validated["domain"]
        problem = validated["problem"]
        solution = validated["solution"]
        grading_guidelines = validated["grading_guidelines"]
        student_answer = validated["student_answer"]

        instruction = f"""You are an expert mathematical grader for International Mathematical Olympiad (IMO) problems.

Your task is to evaluate a student's answer by comparing it against the official solution and grading guidelines.

## Problem Domain
{domain}

## Problem Statement
{problem}

## Official Solution
{solution}

## Grading Guidelines
{grading_guidelines}

## Student's Answer
{student_answer}

## Instructions
Follow this structured evaluation process:

1. **Understanding Check**: Briefly summarize what the problem is asking and what the correct approach should be. Identify the key mathematical concepts and techniques required.

2. **Step-by-Step Analysis**: Go through the student's answer carefully:
   - Identify each key step or claim they make
   - Check if each step is mathematically valid
   - Note any errors, gaps, or incorrect assumptions
   - Compare their approach to the official solution
   - Check if they used the correct mathematical notation and terminology
   - Verify that logical deductions follow correctly from previous steps

3. **Partial Credit Assessment**: Based on the grading guidelines, determine:
   - Which parts of the solution they completed correctly
   - What partial credit they deserve for incomplete or partially correct work
   - Whether they demonstrated understanding of key concepts
   - Whether they made significant progress toward the solution
   - Whether errors are computational (minor) or conceptual (major)

4. **Final Grade Decision**: Assign a grade that reflects:
   - Full correctness (Correct/7) if completely solved with valid reasoning
   - Partial credit (Partial/1-6) for incomplete or partially correct solutions
   - Incorrect (Incorrect/0) for fundamentally wrong or empty answers

## IMO Grading Principles
- A solution must be COMPLETE and CORRECT to receive full marks (7 points)
- Partial credit is awarded for significant progress toward a solution
- Minor errors that don't affect the overall correctness may still allow high partial credit
- Major conceptual errors or missing key steps result in lower scores
- An incomplete solution with correct approach deserves more credit than a complete but wrong solution

## Grading Rubric Reference
- 7 points: Complete, correct solution with clear reasoning
- 6 points: Complete solution with very minor gaps or unclear reasoning
- 5 points: Correct approach but with some gaps or unclear explanations
- 4 points: Significant progress with correct approach but incomplete or with some errors
- 3 points: Good progress with correct initial steps but significant gaps
- 2 points: Some relevant work or correct initial steps
- 1 point: Minimal progress, perhaps some relevant observations
- 0 points: No meaningful progress or completely wrong approach

## Response Format
You MUST respond in JSON format with the following schema:
<json>
{{
    "understanding": "Brief summary of the problem and correct approach",
    "analysis": "Detailed step-by-step analysis of the student's answer, including what they got right and wrong",
    "partial_credit_reasoning": "Explanation of partial credit based on grading guidelines and rubric",
    "response": "Your final grade/score (use: 'Correct', 'Partial', 'Incorrect', or numeric 0-7)"
}}
</json>

Important: The "response" field must contain ONLY the grade (Correct, Partial, Incorrect, or a number 0-7). Do not include explanations in this field."""

        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
        except Exception as e:
            self.log_fn(f"Error calling LLM: {e}")
            return "None", []

        # Extract prediction from JSON with enhanced error handling
        prediction = "None"
        try:
            # Get the last assistant message
            last_assistant_msg = None
            for msg in reversed(msg_history):
                if msg.get("role") == "assistant" or "text" in msg:
                    last_assistant_msg = msg.get("text", msg.get("content", ""))
                    break
            
            if not last_assistant_msg:
                self.log_fn("Warning: No assistant message found in history")
                return str(prediction), msg_history
            
            extracted = _extract_jsons(last_assistant_msg)
            if extracted:
                # Try to get response field first, fallback to other common fields
                last_extract = extracted[-1]
                
                # Validate the grading response
                is_valid, error_msg = _validate_grading_response(last_extract, self.log_fn)
                if not is_valid:
                    self.log_fn(f"Warning: Invalid grading response - {error_msg}")
                
                # Priority order for grade extraction
                grade_fields = ["response", "grade", "score", "result", "final_grade", "evaluation"]
                for field in grade_fields:
                    if field in last_extract:
                        prediction = last_extract[field]
                        break
                
                # Normalize the grade for consistency
                prediction = _normalize_grade(prediction)
                
                # Log detailed analysis for debugging
                if "analysis" in last_extract:
                    self.log_fn(f"Analysis: {last_extract['analysis'][:200]}...")
                if "partial_credit_reasoning" in last_extract:
                    self.log_fn(f"Partial Credit: {last_extract['partial_credit_reasoning'][:200]}...")
                if "understanding" in last_extract:
                    self.log_fn(f"Understanding: {last_extract['understanding'][:200]}...")
                
                self.log_fn(f"Extracted grade: {prediction}")
                
                # Log extraction diagnostics for debugging
                _log_extraction_diagnostics(self.log_fn, last_assistant_msg, extracted, prediction)
            else:
                self.log_fn("Warning: No JSON blocks found in response")
                # Log diagnostics even when extraction fails
                _log_extraction_diagnostics(self.log_fn, last_assistant_msg, extracted, prediction)
                
                # Try to extract grade directly from text as last resort
                text_grade = _extract_grade_from_text_fallback(last_assistant_msg)
                if text_grade:
                    prediction = _normalize_grade(text_grade)
                    self.log_fn(f"Extracted grade from text fallback: {prediction}")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            import traceback
            self.log_fn(f"Traceback: {traceback.format_exc()}")

        return str(prediction), msg_history
