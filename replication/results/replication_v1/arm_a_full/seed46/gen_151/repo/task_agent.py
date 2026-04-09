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
    """Extract JSON objects from <json>...</json> blocks.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    Also handles markdown code blocks (```json...```) as fallback.
    Includes additional fallback for JSON embedded in other text.
    
    Enhanced with better handling for:
    - Nested braces and quotes
    - Common JSON formatting issues
    - Multiple JSON objects in sequence
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
        
        # Try to parse the inner content
        parsed = _try_parse_json(inner)
        if parsed is not None:
            results.append(parsed)
            continue
            
        # Try to extract JSON from within the content if it's wrapped in other text
        json_start = inner.find("{")
        json_end = inner.rfind("}")
        if json_start != -1 and json_end != -1 and json_end > json_start:
            parsed = _try_parse_json(inner[json_start:json_end + 1])
            if parsed is not None:
                results.append(parsed)
    
    # Fallback: try markdown code blocks
    if not results:
        search_from = 0
        while True:
            start = text.find("```json", search_from)
            if start == -1:
                # Also try without 'json' specifier
                start = text.find("```", search_from)
                if start == -1:
                    break
                end = text.find("```", start + 3)
            else:
                end = text.find("```", start + 7)
            if end == -1:
                break
            
            if text[start:start+7] == "```json":
                inner = text[start + 7:end].strip()
            else:
                inner = text[start + 3:end].strip()
            search_from = end + 3
            
            parsed = _try_parse_json(inner)
            if parsed is not None:
                results.append(parsed)
    
    # Fallback: try to find JSON objects directly in the text
    if not results:
        # Look for JSON-like structures with curly braces
        brace_start = text.find("{")
        while brace_start != -1:
            # Try to find matching closing brace using brace counting
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
                if char == '"' and not in_string:
                    in_string = True
                elif char == '"' and in_string:
                    in_string = False
                elif not in_string:
                    if char == "{":
                        brace_count += 1
                    elif char == "}":
                        brace_count -= 1
                        if brace_count == 0:
                            json_str = text[brace_start:brace_start + i + 1]
                            parsed = _try_parse_json(json_str)
                            if parsed is not None:
                                # Only accept if it has expected fields
                                if any(key in parsed for key in ["response", "grade", "score", "analysis", "understanding"]):
                                    results.append(parsed)
                            break
            brace_start = text.find("{", brace_start + 1)
    
    return results or None


def _try_parse_json(text: str) -> dict | None:
    """Try to parse a string as JSON, with common fixups.
    
    Args:
        text: String that might be JSON
        
    Returns:
        Parsed dict or None if parsing fails
    """
    text = text.strip()
    if not text:
        return None
    
    # Try direct parsing first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Try common fixups
    fixups = [
        # Remove trailing commas before closing braces/brackets
        (r',(\s*[}\]])', r'\1'),
        # Fix single quotes to double quotes (carefully)
        (r"(?<!\\)'", '"'),
    ]
    
    import re
    fixed_text = text
    for pattern, replacement in fixups:
        fixed_text = re.sub(pattern, replacement, fixed_text)
    
    try:
        return json.loads(fixed_text)
    except json.JSONDecodeError:
        pass
    
    # Try extracting just the object part if there's extra text
    start = fixed_text.find("{")
    end = fixed_text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(fixed_text[start:end+1])
        except json.JSONDecodeError:
            pass
    
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


def _normalize_grade(grade: str) -> str:
    """Normalize grade to standard format.
    
    Converts various grade formats to a consistent representation.
    Handles both string grades (Correct/Partial/Incorrect) and numeric scores (0-7).
    
    Mapping:
    - 7 points -> "Correct" (full credit)
    - 1-6 points -> "Partial" (partial credit)
    - 0 points -> "Incorrect" (no credit)
    
    Enhanced to better handle edge cases and ambiguous grades.
    """
    if not grade:
        return "None"
    
    # Handle non-string types (int, float, etc.)
    if not isinstance(grade, str):
        grade = str(grade)
    
    grade = grade.strip()
    grade_lower = grade.lower()
    
    # First, check for numeric grades and map to categories
    # 7 = Correct, 1-6 = Partial, 0 = Incorrect
    if grade_lower == "7":
        return "Correct"
    if grade_lower in ["1", "2", "3", "4", "5", "6"]:
        return "Partial"
    if grade_lower == "0":
        return "Incorrect"
    
    # Map common variations to standard grades (using lowercase keys)
    # Order matters: more specific patterns first
    grade_map = {
        # Correct variations
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
        "well done": "Correct",
        "success": "Correct",
        # Partial variations - more specific first
        "partially correct": "Partial",
        "partial credit": "Partial",
        "partially": "Partial",
        "some credit": "Partial",
        "incomplete": "Partial",
        "almost correct": "Partial",
        "almost": "Partial",
        "mostly correct": "Partial",
        "mostly": "Partial",
        "partial": "Partial",
        "minor gaps": "Partial",
        "minor errors": "Partial",
        "significant progress": "Partial",
        "on the right track": "Partial",
        "some understanding": "Partial",
        "partial understanding": "Partial",
        # Incorrect variations
        "incorrect": "Incorrect",
        "wrong": "Incorrect",
        "false": "Incorrect",
        "no": "Incorrect",
        "none": "Incorrect",
        "zero": "Incorrect",
        "invalid": "Incorrect",
        "unacceptable": "Incorrect",
        "failed": "Incorrect",
        "failure": "Incorrect",
        "error": "Incorrect",
        "bad": "Incorrect",
        "poor": "Incorrect",
        "no credit": "Incorrect",
    }
    
    # Check for exact match first (case-insensitive)
    if grade_lower in grade_map:
        return grade_map[grade_lower]
    
    # Try to extract numeric score from strings like "Score: 5" or "Grade: 3"
    # Look for patterns like "5/7", "score of 3", "grade: 4"
    numeric_patterns = [
        r'\b([0-7])\s*/\s*7\b',  # X/7 format
        r'\bscore\s*(?:of|:|=)?\s*([0-7])\b',  # score of X
        r'\bgrade\s*(?:of|:|=)?\s*([0-7])\b',  # grade of X
        r'\bpoints?\s*(?:of|:|=)?\s*([0-7])\b',  # points of X
        r'\b([0-7])\s*points?\b',  # X points
    ]
    
    for pattern in numeric_patterns:
        numeric_match = re.search(pattern, grade_lower)
        if numeric_match:
            score = numeric_match.group(1)
            if score == "7":
                return "Correct"
            elif score == "0":
                return "Incorrect"
            else:
                return "Partial"
    
    # Check for numeric grades (0-7) anywhere in the text
    numeric_match = re.search(r'\b([0-7])\b', grade)
    if numeric_match:
        score = numeric_match.group(1)
        if score == "7":
            return "Correct"
        elif score == "0":
            return "Incorrect"
        else:
            return "Partial"
    
    # Check if grade contains keywords (order matters - check partial first for ambiguous cases)
    # Partial indicators
    partial_indicators = [
        "partial", "incomplete", "unfinished", "missing", "almost", "mostly",
        "minor gaps", "minor errors", "significant progress", "on the right track",
        "some understanding", "partial understanding", "partially", "some credit",
        "essentially correct", "minor issues", "small errors", "nearly"
    ]
    for indicator in partial_indicators:
        if indicator in grade_lower:
            return "Partial"
    
    # Correct indicators
    correct_indicators = [
        "correct", "right", "true", "full", "complete", "solved", "valid",
        "acceptable", "perfect", "excellent", "good", "well done", "success",
        "properly", "accurate", "precise"
    ]
    for indicator in correct_indicators:
        if indicator in grade_lower:
            return "Correct"
    
    # Incorrect indicators
    incorrect_indicators = [
        "incorrect", "wrong", "false", "none", "zero", "error", "invalid",
        "unacceptable", "failed", "failure", "bad", "poor", "no credit",
        "not correct", "not valid", "not acceptable", "completely wrong"
    ]
    for indicator in incorrect_indicators:
        if indicator in grade_lower:
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
   - **Correct**: Complete, correct solution with clear reasoning (equivalent to 7 points)
   - **Partial**: Incomplete or partially correct solutions (equivalent to 1-6 points)
   - **Incorrect**: No meaningful progress or completely wrong approach (equivalent to 0 points)

## Grading Rubric Reference
- **Correct (7 points)**: Complete, correct solution with clear reasoning
- **Partial (5-6 points)**: Minor gaps or unclear reasoning, but essentially correct approach
- **Partial (3-4 points)**: Significant progress, correct approach but incomplete or with errors
- **Partial (1-2 points)**: Some relevant work or correct initial steps
- **Incorrect (0 points)**: No meaningful progress or completely wrong approach

## Important Guidelines
1. You MUST use one of these three categorical labels in your response: **Correct**, **Partial**, or **Incorrect**.
2. Do NOT output numeric scores (0-7) - the system requires categorical labels.
3. Be decisive in your grading - avoid ambiguous language in the final response field.
4. If the answer is "almost" correct but has critical gaps, classify as **Partial**.
5. If the answer shows significant understanding but is incomplete, classify as **Partial**.
6. Only classify as **Correct** if the solution is fully complete and correct.
7. Only classify as **Incorrect** if there is truly no meaningful progress.

Respond in JSON format with the following schema:
<json>
{{
    "understanding": "Brief summary of the problem and correct approach",
    "analysis": "Detailed step-by-step analysis of the student's answer, including what they got right and wrong",
    "partial_credit_reasoning": "Explanation of partial credit based on grading guidelines and rubric",
    "response": "Your final grade: MUST be one of 'Correct', 'Partial', or 'Incorrect'"
}}
</json>

Ensure your JSON is properly formatted with double quotes around all strings and no trailing commas."""

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
                
                # Log detailed analysis for debugging
                if "analysis" in last_extract:
                    self.log_fn(f"Analysis: {last_extract['analysis'][:200]}...")
                if "partial_credit_reasoning" in last_extract:
                    self.log_fn(f"Partial Credit: {last_extract['partial_credit_reasoning'][:200]}...")
                if "understanding" in last_extract:
                    self.log_fn(f"Understanding: {last_extract['understanding'][:200]}...")
                
                self.log_fn(f"Extracted grade: {prediction}")
            else:
                self.log_fn("Warning: No JSON blocks found in response")
                # Try to extract grade directly from text as last resort
                prediction = _extract_grade_from_text(last_assistant_msg)
                if prediction != "None":
                    self.log_fn(f"Extracted grade from text: {prediction}")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            import traceback
            self.log_fn(f"Traceback: {traceback.format_exc()}")

        return str(prediction), msg_history


def _extract_grade_from_text(text: str) -> str:
    """Extract grade from plain text when JSON extraction fails.
    
    Args:
        text: The text to search for grade indicators
        
    Returns:
        Normalized grade string or "None" if not found
    """
    text_lower = text.lower()
    
    # Look for explicit grade declarations
    patterns = [
        r'(?:grade|score|final grade|evaluation)\s*[:=]\s*["\']?([^"\'\n]+)["\']?',
        r'(?:the grade is|grade is|i grade this as)\s*["\']?([^"\'\n]+)["\']?',
        r'response\s*[:=]\s*["\']?(correct|partial|incorrect)["\']?',
        r'\b(correct|partial|incorrect)\b.*solution',
        r'solution is\s+(correct|partial|incorrect)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text_lower, re.IGNORECASE)
        if match:
            grade = match.group(1).strip()
            normalized = _normalize_grade(grade)
            if normalized != "None":
                return normalized
    
    # Look for standalone grade words near the end of the text (often the conclusion)
    # Check last 500 characters
    last_part = text_lower[-500:] if len(text_lower) > 500 else text_lower
    
    # Check for grades in order of specificity
    if "incorrect" in last_part and "not incorrect" not in last_part:
        return "Incorrect"
    if "partial" in last_part:
        return "Partial"
    if "correct" in last_part and "not correct" not in last_part and "incorrect" not in last_part:
        return "Correct"
    
    # Check full text if not found in last part
    if "incorrect" in text_lower and "not incorrect" not in text_lower:
        return "Incorrect"
    if "partial" in text_lower:
        return "Partial"
    if "correct" in text_lower and "not correct" not in text_lower and "incorrect" not in text_lower:
        return "Correct"
    
    return "None"
