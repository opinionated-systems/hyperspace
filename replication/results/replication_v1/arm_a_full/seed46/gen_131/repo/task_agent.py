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
import time
from typing import Any, Callable

from agent.llm_client import get_response_from_llm, EVAL_MODEL

logger = logging.getLogger(__name__)

# Maximum retries for LLM calls when JSON extraction fails
MAX_LLM_RETRIES = 2
# Delay between retries (seconds)
RETRY_DELAY = 0.5


def _safe_json_loads(text: str, log_fn: Callable = logger.debug) -> dict | None:
    """Safely parse JSON with detailed error logging.
    
    Args:
        text: JSON string to parse
        log_fn: Logging function for errors
        
    Returns:
        Parsed dict or None if parsing fails
    """
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError as e:
        log_fn(f"JSON parse error at position {e.pos}: {e.msg}")
        log_fn(f"Problematic text: {text[max(0, e.pos-20):e.pos+20]}")
        return None
    except Exception as e:
        log_fn(f"Unexpected JSON error: {e}")
        return None


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    Also attempts to extract JSON from markdown code blocks as fallback.
    
    Args:
        text: The text to extract JSON from.
        
    Returns:
        A list of parsed JSON objects, or None if no valid JSON found.
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


def _normalize_grade_response(response: str) -> str:
    """Normalize a grade response to a standard format.
    
    Handles various grade formats including:
    - Numeric scores (0-7)
    - Text grades (Correct, Partial, Incorrect)
    - Mixed formats (e.g., "Partial/3", "Correct/7")
    - Case-insensitive matching
    - Whitespace trimming
    
    Returns a standardized grade string.
    """
    if not response:
        return "None"
    
    # Clean up the response
    cleaned = response.strip()
    cleaned_lower = cleaned.lower()
    
    # Handle mixed formats like "Partial/3" or "Correct/7"
    if "/" in cleaned:
        parts = cleaned.split("/")
        grade_part = parts[0].strip().lower()
        
        # Check if it's a valid grade prefix
        if grade_part in ("correct", "partial", "incorrect"):
            return parts[0].strip()
        
        # Check if second part is numeric and first part might be a score
        try:
            score = int(parts[1].strip())
            if 0 <= score <= 7:
                if score == 7:
                    return "Correct"
                elif score >= 4:
                    return "Partial"
                elif score >= 1:
                    return "Partial"
                else:
                    return "Incorrect"
        except ValueError:
            pass
    
    # Direct text grade matching (case-insensitive)
    if cleaned_lower == "correct" or cleaned_lower == "correct/7":
        return "Correct"
    if cleaned_lower == "incorrect" or cleaned_lower == "incorrect/0":
        return "Incorrect"
    if cleaned_lower.startswith("partial"):
        return "Partial"
    
    # Try to parse as a numeric score
    try:
        score = int(cleaned)
        if score == 7:
            return "Correct"
        elif 4 <= score <= 6:
            return "Partial"
        elif 1 <= score <= 3:
            return "Partial"
        elif score == 0:
            return "Incorrect"
    except ValueError:
        pass
    
    # Check for grade embedded in text (e.g., "The grade is: Correct")
    for grade in ["Correct", "Partial", "Incorrect"]:
        if grade.lower() in cleaned_lower:
            return grade
    
    # Return original if no normalization applied
    return cleaned


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


def _get_last_assistant_message(msg_history: list[dict]) -> str | None:
    """Extract the last assistant message from message history.
    
    Handles different message formats from various LLM providers.
    """
    for msg in reversed(msg_history):
        role = msg.get("role", "")
        if role == "assistant" or "text" in msg:
            return msg.get("text", msg.get("content", ""))
    return None


def _extract_prediction_from_message(
    last_assistant_msg: str, 
    log_fn,
    extracted_cache: list | None = None
) -> tuple[str, list | None]:
    """Extract prediction from assistant message with comprehensive fallback strategies.
    
    Returns:
        tuple of (prediction, extracted_json_list)
    """
    prediction = "None"
    extracted = extracted_cache if extracted_cache is not None else _extract_jsons(last_assistant_msg)
    
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
    else:
        # Try to extract grade directly from text as last resort
        text_lower = last_assistant_msg.lower()
        if "grade:" in text_lower or "score:" in text_lower:
            match = re.search(r'(?:grade|score):\s*(.+?)(?:\n|$)', text_lower, re.IGNORECASE)
            if match:
                prediction = _normalize_grade(match.group(1).strip())
                log_fn(f"Extracted grade from text: {prediction}")
    
    return prediction, extracted


def _normalize_grade(grade: str) -> str:
    """Normalize grade to standard format.
    
    Converts various grade formats to a consistent representation.
    Handles both string grades (Correct/Partial/Incorrect) and numeric scores (0-7).
    Includes enhanced handling for edge cases and ambiguous responses.
    
    This function now delegates to _normalize_grade_response for improved
    handling of mixed formats and edge cases.
    """
    if not grade:
        return "None"
    
    # Handle non-string types (int, float, etc.)
    if not isinstance(grade, str):
        grade = str(grade)
    
    # First try the new comprehensive normalizer
    normalized = _normalize_grade_response(grade)
    if normalized != grade:  # If normalization changed something
        return normalized
    
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
    # Pattern for standalone digit 0-7 (word boundary to avoid matching parts of larger numbers)
    numeric_match = re.search(r'(?:^|\s|\()[0-7](?:$|\s|\))', grade)
    if numeric_match:
        # Extract just the digit
        digit_match = re.search(r'[0-7]', numeric_match.group(0))
        if digit_match:
            return digit_match.group(0)
    
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
    
    # Default: return the original grade (don't capitalize to preserve numeric values)
    return grade


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

        # Make initial LLM call with retry logic for failed JSON extraction
        prediction = "None"
        msg_history = []
        
        for attempt in range(MAX_LLM_RETRIES + 1):
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=msg_history if msg_history else [],
                )
                
                # Get the last assistant message
                last_assistant_msg = _get_last_assistant_message(msg_history)
                
                if not last_assistant_msg:
                    self.log_fn(f"Warning: No assistant message found in history (attempt {attempt + 1})")
                    if attempt < MAX_LLM_RETRIES:
                        time.sleep(RETRY_DELAY)
                        continue
                    return str(prediction), msg_history
                
                # Extract prediction from message
                prediction, extracted = _extract_prediction_from_message(
                    last_assistant_msg, self.log_fn
                )
                
                # Log detailed analysis for debugging if extraction succeeded
                if extracted:
                    last_extract = extracted[-1]
                    if "analysis" in last_extract:
                        self.log_fn(f"Analysis: {last_extract['analysis'][:200]}...")
                    if "partial_credit_reasoning" in last_extract:
                        self.log_fn(f"Partial Credit: {last_extract['partial_credit_reasoning'][:200]}...")
                    if "understanding" in last_extract:
                        self.log_fn(f"Understanding: {last_extract['understanding'][:200]}...")
                    
                    self.log_fn(f"Extracted grade: {prediction}")
                    _log_extraction_diagnostics(self.log_fn, last_assistant_msg, extracted, prediction)
                    
                    # Success! Break out of retry loop
                    break
                else:
                    self.log_fn(f"Warning: No JSON blocks found in response (attempt {attempt + 1})")
                    _log_extraction_diagnostics(self.log_fn, last_assistant_msg, extracted, prediction)
                    
                    # If we have retries left, add a reminder to the conversation
                    if attempt < MAX_LLM_RETRIES:
                        reminder = "Please respond with a valid JSON object wrapped in <json>...</json> tags containing your grade evaluation."
                        msg_history.append({"role": "user", "content": reminder})
                        time.sleep(RETRY_DELAY)
                    
            except Exception as e:
                self.log_fn(f"Error in attempt {attempt + 1}: {e}")
                import traceback
                self.log_fn(f"Traceback: {traceback.format_exc()}")
                if attempt < MAX_LLM_RETRIES:
                    time.sleep(RETRY_DELAY)

        return str(prediction), msg_history
