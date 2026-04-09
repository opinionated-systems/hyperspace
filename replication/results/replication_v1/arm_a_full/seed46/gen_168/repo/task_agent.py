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

from agent.llm_client import get_response_from_llm, EVAL_MODEL

logger = logging.getLogger(__name__)

# Maximum retries for LLM calls when JSON extraction fails
MAX_LLM_RETRIES = 2
# Delay between retries (seconds)
RETRY_DELAY = 0.5


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks with fallbacks.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    Falls back to markdown code blocks and direct brace matching if needed.
    
    Enhanced with better error handling, support for nested JSON structures,
    and improved recovery from common LLM formatting errors.
    """
    if not text or not isinstance(text, str):
        return None
    
    # Strip common leading/trailing whitespace that might interfere
    text = text.strip()
    if not text:
        return None
        
    results = []
    search_from = 0
    
    # Primary: Extract from <json>...</json> blocks
    while True:
        start = text.find("<json>", search_from)
        if start == -1:
            break
        end = text.find("</json>", start)
        if end == -1:
            # Malformed: opening tag without closing tag - try to extract anyway
            inner = text[start + 6:].strip()
            search_from = len(text)  # Move to end to prevent infinite loop
        else:
            inner = text[start + 6:end].strip()
            search_from = end + 7
        
        # Skip empty blocks
        if not inner:
            continue
            
        # Try to parse the JSON content
        parsed = _try_parse_json_with_fixes(inner)
        if parsed is not None:
            if isinstance(parsed, dict):
                results.append(parsed)
            elif isinstance(parsed, list):
                # Handle case where JSON is an array of objects
                for item in parsed:
                    if isinstance(item, dict):
                        results.append(item)
    
    # Fallback 1: Try markdown code blocks
    if not results:
        results = _extract_jsons_from_markdown(text) or []
    
    # Fallback 2: Try direct brace matching
    if not results:
        results = _extract_jsons_from_braces(text) or []
    
    return results if results else None


def _try_parse_json_with_fixes(json_str: str) -> dict | list | None:
    """Attempt to parse JSON string with multiple recovery strategies.
    
    Tries increasingly aggressive fixes for common LLM JSON formatting errors.
    Returns the parsed object or None if all attempts fail.
    """
    if not json_str:
        return None
    
    # Attempt 1: Direct parsing
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass
    
    # Attempt 2: Remove trailing commas before closing braces/brackets
    try:
        fixed = re.sub(r',\s*}', '}', json_str)
        fixed = re.sub(r',\s*]', ']', fixed)
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    
    # Attempt 3: Fix single quotes to double quotes (common LLM mistake)
    try:
        # Replace single-quoted keys with double-quoted keys
        fixed = re.sub(r"'([^']*?)'\s*:", r'"\1":', json_str)
        # Replace single-quoted string values with double-quoted values
        fixed = re.sub(r":\s*'([^']*?)'", r': "\1"', fixed)
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    
    # Attempt 4: Combined fixes (trailing commas + single quotes)
    try:
        fixed = re.sub(r',\s*}', '}', json_str)
        fixed = re.sub(r',\s*]', ']', fixed)
        fixed = re.sub(r"'([^']*?)'\s*:", r'"\1":', fixed)
        fixed = re.sub(r":\s*'([^']*?)'", r': "\1"', fixed)
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    
    # Attempt 5: Handle unescaped newlines in string values
    try:
        # Replace newlines within JSON string values with escaped newlines
        # This is a more aggressive fix for multiline string values
        fixed = json_str.replace('\n', '\\n').replace('\r', '\\r')
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    
    # All attempts failed
    return None


def _extract_jsons_from_markdown(text: str) -> list[dict] | None:
    """Extract JSON objects from markdown code blocks (```json or ```).
    
    Used as a fallback when <json> tags are not found.
    """
    if not text or not isinstance(text, str):
        return None
    
    results = []
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
        
        if not inner:
            continue
            
        # Try to parse with the same recovery strategies
        parsed = _try_parse_json_with_fixes(inner)
        if parsed is not None and isinstance(parsed, dict):
            results.append(parsed)
    
    return results if results else None


def _extract_jsons_from_braces(text: str) -> list[dict] | None:
    """Extract JSON objects directly from text by finding matching braces.
    
    Used as a last resort when other methods fail.
    Only accepts objects with expected grading-related fields.
    """
    if not text or not isinstance(text, str):
        return None
    
    results = []
    # Look for JSON-like structures with curly braces
    brace_start = text.find("{")
    while brace_start != -1:
        # Try to find matching closing brace
        brace_count = 0
        in_string = False
        escape_next = False
        
        for i, char in enumerate(text[brace_start:]):
            if escape_next:
                escape_next = False
                continue
            if char == "\\":
                escape_next = True
                continue
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
            if not in_string:
                if char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        json_str = text[brace_start:brace_start + i + 1]
                        # Try to parse with recovery strategies
                        parsed = _try_parse_json_with_fixes(json_str)
                        if parsed is not None and isinstance(parsed, dict):
                            # Only accept if it has expected fields
                            if any(key in parsed for key in ["response", "grade", "score", "analysis", "understanding"]):
                                results.append(parsed)
                        break
        brace_start = text.find("{", brace_start + 1)
    
    return results if results else None


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
    Includes enhanced handling for edge cases, decimal scores, fractions, and ambiguous responses.
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
        "partial": "Partial",
        "partially correct": "Partial",
        "partial credit": "Partial",
        "partially": "Partial",
        "some credit": "Partial",
        "incomplete": "Partial",
        "mostly correct": "Partial",
        "minor errors": "Partial",
        "good progress": "Partial",
        "significant progress": "Partial",
        "incorrect": "Incorrect",
        "wrong": "Incorrect",
        "false": "Incorrect",
        "no": "Incorrect",
        "none": "Incorrect",
        "zero": "Incorrect",
        "invalid": "Incorrect",
        "unacceptable": "Incorrect",
        "fail": "Incorrect",
        "failed": "Incorrect",
        "no credit": "Incorrect",
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
    
    # Pattern for decimal scores (e.g., "5.5", "6.0") - round to nearest integer
    decimal_match = re.search(r'\b([0-6]\.\d+)\b', grade)
    if decimal_match:
        score = float(decimal_match.group(1))
        rounded = int(round(score))
        if 0 <= rounded <= 7:
            return str(rounded)
    
    # Pattern for "X points" or "X out of 7"
    points_match = re.search(r'(\d+)\s*(?:points?|pts?|/\s*7|out\s+of\s*7)', grade_lower)
    if points_match:
        score = int(points_match.group(1))
        if 0 <= score <= 7:
            return str(score)
    
    # Pattern for fractions like "5/7" or "3 / 7"
    fraction_match = re.search(r'\b([0-7])\s*/\s*7\b', grade_lower)
    if fraction_match:
        return fraction_match.group(1)
    
    # Check for grade in parentheses or brackets
    bracket_match = re.search(r'[\(\[]([0-7]|correct|partial|incorrect)[\)\]]', grade_lower)
    if bracket_match:
        inner = bracket_match.group(1)
        if inner in grade_map:
            return grade_map[inner]
        if inner.isdigit() and 0 <= int(inner) <= 7:
            return inner
    
    # Check for numeric grades with descriptive text (e.g., "5 - Good progress")
    numeric_prefix_match = re.search(r"^([0-7])\s*[-–—]", grade)
    if numeric_prefix_match:
        return numeric_prefix_match.group(1)
    
    # Check for "Grade: X" or "Score: X" patterns (with optional decimal)
    grade_score_match = re.search(r"(?:grade|score)\s*[:=]\s*([0-7](?:\.\d+)?)", grade_lower)
    if grade_score_match:
        score_str = grade_score_match.group(1)
        if '.' in score_str:
            rounded = int(round(float(score_str)))
            return str(rounded) if 0 <= rounded <= 7 else str(int(float(score_str)))
        return score_str
    
    # Check for "X/7" or "X out of 7" patterns
    out_of_match = re.search(r'(\d+)\s*/\s*\d+', grade)
    if out_of_match:
        score = int(out_of_match.group(1))
        if 0 <= score <= 7:
            return str(score)
    
    # Check if grade contains keywords (order matters - check partial first)
    if "partial" in grade_lower or "partially" in grade_lower:
        return "Partial"
    if any(word in grade_lower for word in ["incomplete", "unfinished", "missing", "minor error", "mostly", "some progress", "good attempt"]):
        return "Partial"
    if any(word in grade_lower for word in ["correct", "right", "true", "full", "complete", "solved", "valid", "acceptable", "perfect", "excellent"]):
        return "Correct"
    if any(word in grade_lower for word in ["incorrect", "wrong", "false", "none", "zero", "error", "invalid", "unacceptable", "fail", "failed", "no credit"]):
        return "Incorrect"
    
    # Default: return the original grade (don't capitalize to preserve numeric values)
    return grade
