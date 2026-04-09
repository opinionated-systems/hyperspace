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
from typing import Callable

from agent.llm_client import get_response_from_llm, EVAL_MODEL

logger = logging.getLogger(__name__)


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks with enhanced robustness.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    Also attempts to extract JSON from markdown code blocks as fallback.
    Includes repair attempts for common JSON formatting issues.
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
            # Try to repair common JSON issues
            repaired = _repair_json(inner)
            if repaired:
                results.append(repaired)
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
                # Try to repair common JSON issues
                repaired = _repair_json(inner)
                if repaired:
                    results.append(repaired)
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
                            # Try to repair common JSON issues
                            repaired = _repair_json(json_str)
                            if repaired and any(key in repaired for key in ["response", "grade", "score", "analysis"]):
                                results.append(repaired)
                        break
            brace_start = text.find("{", brace_start + 1)
    
    return results or None


def _repair_json(text: str) -> dict | None:
    """Attempt to repair common JSON formatting issues.
    
    Handles:
    - Trailing commas in objects/arrays
    - Single quotes instead of double quotes
    - Unescaped newlines in strings
    - Missing quotes around keys
    """
    import re
    
    # Remove trailing commas before closing braces/brackets
    text = re.sub(r',(\s*[}\]])', r'\1', text)
    
    # Replace single quotes with double quotes (carefully)
    # Only replace quotes that are not inside strings
    # This is a heuristic approach
    text = re.sub(r"(?<!\\)'", '"', text)
    
    # Escape unescaped newlines in strings
    # Find strings and escape newlines within them
    def escape_newlines_in_string(match):
        content = match.group(1)
        # Escape newlines that aren't already escaped
        content = content.replace('\n', '\\n')
        content = content.replace('\r', '\\r')
        content = content.replace('\t', '\\t')
        return f'"{content}"'
    
    # Match quoted strings (simplified)
    text = re.sub(r'"([^"\\]*(?:\\.[^"\\]*)*)"', escape_newlines_in_string, text)
    
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Last resort: try to extract just the response field
        response_match = re.search(r'"response"\s*:\s*"([^"]+)"', text)
        if response_match:
            return {"response": response_match.group(1)}
        return None


def _log_extraction_details(text: str, results: list[dict] | None, logger_fn: Callable) -> None:
    """Log detailed information about JSON extraction for debugging.
    
    Args:
        text: The raw text that was processed
        results: The extracted JSON results (or None if failed)
        logger_fn: Logging function to use
    """
    if results:
        logger_fn(f"Successfully extracted {len(results)} JSON object(s)")
        for i, obj in enumerate(results):
            keys = list(obj.keys())
            logger_fn(f"  Object {i+1}: keys={keys}")
    else:
        # Log hints about why extraction might have failed
        has_json_tags = "<json>" in text and "</json>" in text
        has_code_blocks = "```" in text
        has_braces = "{" in text and "}" in text
        
        logger_fn("JSON extraction failed - diagnostic info:")
        logger_fn(f"  - Contains <json> tags: {has_json_tags}")
        logger_fn(f"  - Contains code blocks: {has_code_blocks}")
        logger_fn(f"  - Contains curly braces: {has_braces}")
        logger_fn(f"  - Text length: {len(text)} chars")
        
        # Show a snippet of the text for debugging
        snippet = text[:200].replace("\n", " ")
        logger_fn(f"  - Text preview: {snippet}...")


def _validate_grade_value(grade: str) -> tuple[bool, str]:
    """Validate that a grade value is in an acceptable format.
    
    Args:
        grade: The grade string to validate
        
    Returns:
        Tuple of (is_valid, normalized_grade)
    """
    if not grade or grade == "None":
        return False, "None"
    
    # Handle numeric grades (0-7)
    valid_grades = {"Correct", "Partial", "Incorrect", "0", "1", "2", "3", "4", "5", "6", "7"}
    
    if grade in valid_grades:
        return True, grade
    
    # Try to normalize and check again
    normalized = _normalize_grade(grade)
    if normalized in valid_grades:
        return True, normalized
    
    # Check if it's a numeric string that can be mapped
    try:
        num = int(grade)
        if 0 <= num <= 7:
            return True, str(num)
    except (ValueError, TypeError):
        pass
    
    return False, grade


def _normalize_grade(grade: str) -> str:
    """Normalize grade to standard format.
    
    Converts various grade formats to a consistent representation.
    Handles both string grades (Correct/Partial/Incorrect) and numeric scores (0-7).
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
        "perfect": "Correct",
        "excellent": "Correct",
        "partial": "Partial",
        "partially correct": "Partial",
        "partial credit": "Partial",
        "partially": "Partial",
        "some credit": "Partial",
        "incomplete": "Partial",
        "mostly correct": "Partial",
        "mostly right": "Partial",
        "incorrect": "Incorrect",
        "wrong": "Incorrect",
        "false": "Incorrect",
        "no": "Incorrect",
        "none": "Incorrect",
        "zero": "Incorrect",
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
    
    # Try to extract numeric score from strings like "Score: 5" or "Grade: 3"
    numeric_match = re.search(r'\b([0-7])\b', grade)
    if numeric_match:
        return numeric_match.group(1)
    
    # Check for numeric ranges and pick appropriate grade
    # e.g., "5-6 points" -> "5" (lower bound for partial credit)
    range_match = re.search(r'(\d+)\s*-\s*(\d+)', grade)
    if range_match:
        lower = int(range_match.group(1))
        upper = int(range_match.group(2))
        if 0 <= lower <= 7:
            return str(lower)
    
    # Check if grade contains keywords (order matters - check partial first)
    if "partial" in grade_lower:
        return "Partial"
    if any(word in grade_lower for word in ["incomplete", "unfinished", "missing", "half"]):
        return "Partial"
    if any(word in grade_lower for word in ["mostly correct", "mostly right", "nearly correct"]):
        return "Partial"
    if any(word in grade_lower for word in ["correct", "right", "true", "full", "complete", "solved", "perfect", "excellent"]):
        return "Correct"
    if any(word in grade_lower for word in ["incorrect", "wrong", "false", "none", "zero", "error", "invalid", "fail", "failed"]):
        return "Incorrect"
    
    # Default: capitalize first letter
    return grade.capitalize()


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

## IMPORTANT: Response Format
You MUST respond in valid JSON format wrapped in <json>...</json> tags.
The "response" field MUST contain exactly one of these values:
- "Correct" (for complete, correct solutions worth 7 points)
- "Partial" (for incomplete or partially correct solutions worth 1-6 points)
- "Incorrect" (for fundamentally wrong or empty answers worth 0 points)
- Or a numeric string "0" through "7" for specific point values

Example response format:
<json>
{{
    "understanding": "This problem asks to find the sum of digits...",
    "analysis": "The student correctly identified... but made an error in...",
    "partial_credit_reasoning": "The student showed understanding of... but missed...",
    "response": "Partial"
}}
</json>

Respond in JSON format with the following schema:
<json>
{{
    "understanding": "Brief summary of the problem and correct approach",
    "analysis": "Detailed step-by-step analysis of the student's answer, including what they got right and wrong",
    "partial_credit_reasoning": "Explanation of partial credit based on grading guidelines and rubric",
    "response": "Your final grade/score (use: 'Correct', 'Partial', 'Incorrect', or numeric 0-7)"
}}
</json>"""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

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
            
            # Log extraction details for debugging
            _log_extraction_details(last_assistant_msg, extracted, self.log_fn)
            
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
                
                # Validate the grade and log if invalid
                is_valid, validated_grade = _validate_grade_value(prediction)
                if not is_valid:
                    self.log_fn(f"Warning: Extracted grade '{prediction}' is not in standard format")
                    # Try to infer from analysis text if grade is invalid
                    if "analysis" in last_extract:
                        inferred = _infer_grade_from_analysis(last_extract["analysis"])
                        if inferred:
                            self.log_fn(f"Inferred grade from analysis: {inferred}")
                            prediction = inferred
                            is_valid = True
                prediction = validated_grade
                
                # Log detailed analysis for debugging
                if "analysis" in last_extract:
                    self.log_fn(f"Analysis: {last_extract['analysis'][:200]}...")
                if "partial_credit_reasoning" in last_extract:
                    self.log_fn(f"Partial Credit: {last_extract['partial_credit_reasoning'][:200]}...")
                if "understanding" in last_extract:
                    self.log_fn(f"Understanding: {last_extract['understanding'][:200]}...")
                
                self.log_fn(f"Extracted grade: {prediction} (valid: {is_valid})")
            else:
                self.log_fn("Warning: No JSON blocks found in response")
                # Try to extract grade directly from text as last resort
                prediction = _extract_grade_from_text(last_assistant_msg)
                if prediction != "None":
                    is_valid, prediction = _validate_grade_value(prediction)
                    self.log_fn(f"Extracted grade from text: {prediction} (valid: {is_valid})")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            import traceback
            self.log_fn(f"Traceback: {traceback.format_exc()}")

        return str(prediction), msg_history


def _extract_grade_from_text(text: str) -> str:
    """Extract grade from plain text when JSON parsing fails.
    
    Looks for common patterns like "Grade: X", "Score: Y", etc.
    """
    import re
    
    text_lower = text.lower()
    
    # Look for explicit grade/score patterns
    patterns = [
        r'(?:grade|score|result|evaluation)\s*[:=]\s*["\']?([^"\'\n]+)["\']?',
        r'(?:final|overall|total)\s+(?:grade|score|result)\s*[:=]\s*["\']?([^"\'\n]+)["\']?',
        r'(?:assigned|given|awarded)\s*[:=]\s*["\']?([^"\'\n]+)["\']?',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text_lower, re.IGNORECASE)
        if match:
            grade = match.group(1).strip()
            normalized = _normalize_grade(grade)
            is_valid, validated = _validate_grade_value(normalized)
            if is_valid:
                return validated
    
    # Look for standalone grade keywords
    if re.search(r'\bcorrect\b', text_lower) and not re.search(r'\bpartial(ly)?\s+correct\b', text_lower):
        return "Correct"
    if re.search(r'\bpartial(ly)?\s+correct\b', text_lower) or re.search(r'\bpartial\b', text_lower):
        return "Partial"
    if re.search(r'\bincorrect\b|\bwrong\b', text_lower):
        return "Incorrect"
    
    # Look for numeric grades
    numeric_match = re.search(r'\b([0-7])\s*(?:points?|pts?|/\s*7)?\b', text_lower)
    if numeric_match:
        return numeric_match.group(1)
    
    return "None"


def _infer_grade_from_analysis(analysis: str) -> str | None:
    """Infer grade from analysis text when explicit grade is missing or invalid.
    
    Uses keyword matching to determine the most likely grade.
    """
    if not analysis:
        return None
    
    analysis_lower = analysis.lower()
    
    # Count positive vs negative indicators
    positive_indicators = [
        "correct", "valid", "right", "properly", "accurate", "correctly",
        "complete solution", "fully solved", "correct approach"
    ]
    negative_indicators = [
        "incorrect", "wrong", "error", "invalid", "mistake", "flawed",
        "missing", "incomplete", "failed", "no progress"
    ]
    partial_indicators = [
        "partial", "partially", "some progress", "incomplete", "minor error",
        "mostly correct", "significant progress", "on the right track"
    ]
    
    pos_count = sum(1 for ind in positive_indicators if ind in analysis_lower)
    neg_count = sum(1 for ind in negative_indicators if ind in analysis_lower)
    partial_count = sum(1 for ind in partial_indicators if ind in analysis_lower)
    
    # Decision logic
    if partial_count > 0 or (pos_count > 0 and neg_count > 0):
        return "Partial"
    elif pos_count > neg_count:
        return "Correct"
    elif neg_count > pos_count:
        return "Incorrect"
    
    return None
