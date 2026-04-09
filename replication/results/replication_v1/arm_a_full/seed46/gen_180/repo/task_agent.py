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
    Also attempts to extract JSON from markdown code blocks as fallback.
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
                            pass
                        break
            brace_start = text.find("{", brace_start + 1)
    
    return results or None


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
        "perfect": "Correct",
        "excellent": "Correct",
        "good": "Correct",
        "partial": "Partial",
        "partially correct": "Partial",
        "partial credit": "Partial",
        "partially": "Partial",
        "some credit": "Partial",
        "incomplete": "Partial",
        "mostly correct": "Partial",
        "minor errors": "Partial",
        "fair": "Partial",
        "average": "Partial",
        "needs improvement": "Partial",
        "incorrect": "Incorrect",
        "wrong": "Incorrect",
        "false": "Incorrect",
        "no": "Incorrect",
        "none": "Incorrect",
        "zero": "Incorrect",
        "invalid": "Incorrect",
        "unacceptable": "Incorrect",
        "poor": "Incorrect",
        "bad": "Incorrect",
        "fail": "Incorrect",
        "failed": "Incorrect",
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
    if any(word in grade_lower for word in ["incomplete", "unfinished", "missing", "minor error", "mostly", "needs work", "could be better"]):
        return "Partial"
    if any(word in grade_lower for word in ["correct", "right", "true", "full", "complete", "solved", "valid", "acceptable", "perfect", "excellent", "good", "well done"]):
        return "Correct"
    if any(word in grade_lower for word in ["incorrect", "wrong", "false", "none", "zero", "error", "invalid", "unacceptable", "fail", "failed", "poor", "bad", "not correct"]):
        return "Incorrect"
    
    # Check for numeric ranges and extract lower bound as conservative estimate
    range_match = re.search(r'(\d+)[\s-]+(\d+)', grade_lower)
    if range_match:
        low, high = int(range_match.group(1)), int(range_match.group(2))
        if 0 <= low <= 7 and 0 <= high <= 7:
            return str(low)
    
    # Check for fraction patterns like "3/7" or "5 out of 7"
    fraction_match = re.search(r'(\d+)\s*/\s*7', grade_lower)
    if fraction_match:
        score = int(fraction_match.group(1))
        if 0 <= score <= 7:
            return str(score)
    
    # Check for percentage and convert to 0-7 scale
    percent_match = re.search(r'(\d+)\s*%', grade_lower)
    if percent_match:
        percent = int(percent_match.group(1))
        if 0 <= percent <= 100:
            # Convert percentage to 0-7 scale and round
            score = round(percent / 100 * 7)
            return str(min(7, max(0, score)))
    
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


def _extract_grade_from_text_fallback(text: str) -> str | None:
    """Extract grade from plain text when JSON extraction fails.
    
    This is a last-resort fallback that looks for grade patterns in text.
    
    Args:
        text: The text to search for grade patterns
        
    Returns:
        Normalized grade string or None if no grade found
    """
    text_lower = text.lower()
    
    # Pattern 1: "Grade: X" or "Score: X" or "Final Grade: X"
    grade_patterns = [
        r'(?:final\s+)?(?:grade|score|result|evaluation)\s*[:=]\s*["\']?([0-7]|correct|partial|incorrect)["\']?',
        r'(?:grade|score)\s+is\s+["\']?([0-7]|correct|partial|incorrect)["\']?',
        r'(?:assigned|final)\s+(?:grade|score)\s*[:=]?\s*["\']?([0-7]|correct|partial|incorrect)["\']?',
    ]
    
    for pattern in grade_patterns:
        match = re.search(pattern, text_lower, re.IGNORECASE)
        if match:
            grade = match.group(1).strip()
            return _normalize_grade(grade)
    
    # Pattern 2: Look for standalone grades at end of sentences
    sentence_end_pattern = r'[.!?]\s*([0-7]|correct|partial|incorrect)[.!?\s]*$'
    match = re.search(sentence_end_pattern, text_lower, re.IGNORECASE | re.MULTILINE)
    if match:
        return _normalize_grade(match.group(1))
    
    # Pattern 3: Check for numeric scores in parentheses like (5) or [3]
    paren_pattern = r'[\(\[]([0-7])[\)\]]'
    match = re.search(paren_pattern, text_lower)
    if match:
        return match.group(1)
    
    return None


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.max_retries = 3  # Maximum retries for LLM calls

    def _validate_inputs(self, inputs: dict) -> tuple[bool, str]:
        """Validate that all required fields are present and non-empty.
        
        Returns:
            (is_valid, error_message)
        """
        required_fields = ["problem", "solution", "grading_guidelines", "student_answer"]
        
        for field in required_fields:
            if field not in inputs:
                return False, f"Missing required field: {field}"
            if not inputs[field] or not str(inputs[field]).strip():
                return False, f"Empty required field: {field}"
        
        return True, ""

    def _call_llm_with_retry(self, instruction: str) -> tuple[str, list[dict], dict]:
        """Call LLM with retry logic for transient failures.
        
        Args:
            instruction: The prompt to send to the LLM
            
        Returns:
            (response, msg_history, info) tuple
        """
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=[],
                )
                return response, msg_history, info
            except Exception as e:
                last_error = e
                self.log_fn(f"LLM call failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    import time
                    time.sleep(2 ** attempt)  # Exponential backoff
        
        # All retries failed
        raise RuntimeError(f"LLM call failed after {self.max_retries} attempts: {last_error}")

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with structured reasoning.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Validate inputs first
        is_valid, error_msg = self._validate_inputs(inputs)
        if not is_valid:
            self.log_fn(f"Input validation failed: {error_msg}")
            return f"Error: {error_msg}", []
        
        # Extract fields for structured prompting
        domain = inputs.get("domain", "")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

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

1. **Understanding Check**: Briefly summarize what the problem is asking and what the correct approach should be.

2. **Step-by-Step Analysis**: Go through the student's answer carefully:
   - Identify each key step or claim they make
   - Check if each step is mathematically valid
   - Note any errors, gaps, or incorrect assumptions
   - Compare their approach to the official solution
   - Check if they used the correct mathematical notation and terminology

3. **Partial Credit Assessment**: Based on the grading guidelines, determine:
   - Which parts of the solution they completed correctly
   - What partial credit they deserve for incomplete or partially correct work
   - Whether they demonstrated understanding of key concepts
   - Whether they made significant progress toward the solution

4. **Final Grade Decision**: Assign a grade that reflects:
   - Full correctness (Correct/7) if completely solved with valid reasoning
   - Partial credit (Partial/1-6) for incomplete or partially correct solutions
   - Incorrect (Incorrect/0) for fundamentally wrong or empty answers

## Grading Rubric Reference
- 7 points: Complete, correct solution with clear reasoning
- 5-6 points: Minor gaps or unclear reasoning, but essentially correct approach
- 3-4 points: Significant progress, correct approach but incomplete or with errors
- 1-2 points: Some relevant work or correct initial steps
- 0 points: No meaningful progress or completely wrong approach

Respond in JSON format with the following schema:
<json>
{{
    "understanding": "Brief summary of the problem and correct approach",
    "analysis": "Detailed step-by-step analysis of the student's answer, including what they got right and wrong",
    "partial_credit_reasoning": "Explanation of partial credit based on grading guidelines and rubric",
    "response": "Your final grade/score (use: 'Correct', 'Partial', 'Incorrect', or numeric 0-7)"
}}
</json>"""

        try:
            response, msg_history, info = self._call_llm_with_retry(instruction)
        except RuntimeError as e:
            self.log_fn(f"LLM call failed completely: {e}")
            return "Error: LLM unavailable", []

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
                fallback_grade = _extract_grade_from_text_fallback(last_assistant_msg)
                if fallback_grade:
                    prediction = fallback_grade
                    self.log_fn(f"Extracted grade from text fallback: {prediction}")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            import traceback
            self.log_fn(f"Traceback: {traceback.format_exc()}")

        return str(prediction), msg_history
