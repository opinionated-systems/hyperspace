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
    Includes enhanced error recovery for malformed JSON.
    """
    if not text or not isinstance(text, str):
        return None
        
    results = []
    search_from = 0
    parse_errors = []  # Track errors for debugging
    
    # Primary: extract from <json>...</json> blocks
    while True:
        start = text.find("<json>", search_from)
        if start == -1:
            break
        end = text.find("</json>", start)
        if end == -1:
            parse_errors.append(f"Unclosed <json> tag at position {start}")
            break
        inner = text[start + 6:end].strip()
        search_from = end + 7
        try:
            results.append(json.loads(inner))
        except json.JSONDecodeError as e:
            parse_errors.append(f"JSON parse error at position {start}: {e}")
            # Try to fix common JSON issues
            fixed = _attempt_json_repair(inner)
            if fixed:
                try:
                    results.append(json.loads(fixed))
                except json.JSONDecodeError:
                    continue
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
            except json.JSONDecodeError as e:
                parse_errors.append(f"Markdown JSON parse error: {e}")
                # Try repair
                fixed = _attempt_json_repair(inner)
                if fixed:
                    try:
                        results.append(json.loads(fixed))
                    except json.JSONDecodeError:
                        continue
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
                        except json.JSONDecodeError as e:
                            parse_errors.append(f"Brace extraction error: {e}")
                            # Try repair
                            fixed = _attempt_json_repair(json_str)
                            if fixed:
                                try:
                                    parsed = json.loads(fixed)
                                    if any(key in parsed for key in ["response", "grade", "score", "analysis"]):
                                        results.append(parsed)
                                except json.JSONDecodeError:
                                    pass
                        break
            brace_start = text.find("{", brace_start + 1)
    
    # Store parse errors for potential debugging
    if parse_errors and results:
        results[-1]["_parse_errors"] = parse_errors
    
    return results or None


def _attempt_json_repair(json_str: str) -> str | None:
    """Attempt to repair common JSON formatting issues.
    
    Args:
        json_str: The potentially malformed JSON string
        
    Returns:
        Repaired JSON string if repairable, None otherwise
    """
    if not json_str:
        return None
    
    repaired = json_str.strip()
    
    # Fix trailing commas in objects/arrays
    repaired = re.sub(r',(\s*[}\]])', r'\1', repaired)
    
    # Fix single quotes to double quotes (common LLM mistake)
    # Only replace single quotes that are not inside strings
    repaired = re.sub(r"(?<!\\)'", '"', repaired)
    
    # Fix unescaped newlines in strings
    repaired = re.sub(r'(?<!\\)\n', '\\n', repaired)
    
    # Fix missing quotes around keys (simple heuristic)
    repaired = re.sub(r'(\{|,\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', repaired)
    
    # Ensure the string starts with { and ends with }
    if not repaired.startswith('{'):
        start_idx = repaired.find('{')
        if start_idx != -1:
            repaired = repaired[start_idx:]
    if not repaired.endswith('}'):
        end_idx = repaired.rfind('}')
        if end_idx != -1:
            repaired = repaired[:end_idx+1]
    
    return repaired if repaired else None


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


def _structured_log(log_fn, event_type: str, data: dict, level: str = "info") -> None:
    """Create a structured log entry for better observability.
    
    Args:
        log_fn: The logging function to use
        event_type: Type of event being logged
        data: Dictionary of data to include in the log
        level: Log level (info, warning, error)
    """
    import json as _json
    import time
    
    entry = {
        "timestamp": time.time(),
        "event_type": event_type,
        "level": level,
        **data
    }
    
    # Format as JSON for structured logging
    log_message = f"[STRUCTURED_LOG] {_json.dumps(entry, default=str)}"
    log_fn(log_message)


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
            return False, f"Missing required field: '{field}'"
    
    # Check for empty response
    response_val = extracted.get("response")
    if not response_val or str(response_val).strip() == "":
        return False, "Empty response field"
    
    # Log warnings for missing recommended fields
    for field in recommended_fields:
        if field not in extracted:
            log_fn(f"Warning: Missing recommended field '{field}' in grading response")
    
    # Validate response value format
    valid_grades = ["Correct", "Partial", "Incorrect", "0", "1", "2", "3", "4", "5", "6", "7"]
    normalized = _normalize_grade(str(response_val))
    if normalized not in valid_grades and normalized != "None":
        log_fn(f"Warning: Unusual grade value '{response_val}' (normalized: '{normalized}')")
    
    return True, ""


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


def _extract_grade_from_text(text: str) -> str | None:
    """Extract grade from raw text when JSON extraction fails.
    
    This is a last-resort fallback that searches for grade patterns
    in the LLM's raw text response.
    
    Args:
        text: The raw text to search for grade patterns
        
    Returns:
        Normalized grade string if found, None otherwise
    """
    if not text:
        return None
    
    text_lower = text.lower()
    
    # Look for explicit grade/score declarations
    patterns = [
        # "Grade: X" or "Score: X" patterns
        r'(?:grade|score|final grade|final score|assigned grade|assigned score)[\s:]*([0-7]|correct|partial|incorrect)',
        # "The grade is X" patterns
        r'(?:grade|score)\s+(?:is|should be|will be)\s+["\']?([0-7]|correct|partial|incorrect)',
        # "I assign X" patterns
        r'(?:assign|give|award)\s+(?:a\s+)?(?:grade|score|mark)?\s*(?:of\s+)?["\']?([0-7]|correct|partial|incorrect)',
        # "X/7" score format
        r'([0-7])\s*/\s*7',
        # "(X)" format for grades
        r'\(\s*([0-7])\s*\)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text_lower, re.IGNORECASE)
        if match:
            grade = match.group(1).strip()
            normalized = _normalize_grade(grade)
            if normalized != "None":
                return normalized
    
    # Look for standalone grade keywords near the end of the response
    # (LLMs often put their final conclusion at the end)
    lines = text_lower.split('\n')
    for line in reversed(lines[-20:]):  # Check last 20 lines
        line = line.strip()
        # Skip empty lines and common non-grade lines
        if not line or line.startswith('```') or line.startswith('<json>'):
            continue
        
        # Check for grade keywords
        if any(word in line for word in ['correct', 'partial', 'incorrect']):
            for keyword in ['correct', 'partial', 'incorrect']:
                if keyword in line:
                    return _normalize_grade(keyword)
        
        # Check for standalone digits 0-7
        digit_match = re.search(r'\b([0-7])\b', line)
        if digit_match:
            return digit_match.group(1)
    
    return None


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning.
    
    Enhanced with structured logging, better error recovery, and improved
    JSON extraction with automatic repair capabilities.
    """

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self._call_count = 0
        self._success_count = 0
        self._error_count = 0

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with structured reasoning.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        self._call_count += 1
        call_id = self._call_count
        
        # Extract fields for structured prompting
        domain = inputs.get("domain", "")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        # Log start of processing
        _structured_log(
            self.log_fn,
            "grading_start",
            {
                "call_id": call_id,
                "domain": domain,
                "problem_length": len(problem),
                "solution_length": len(solution),
                "student_answer_length": len(student_answer)
            }
        )

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
                # Try to extract grade directly from text as last resort using enhanced extraction
                fallback_grade = _extract_grade_from_text(last_assistant_msg)
                if fallback_grade:
                    prediction = fallback_grade
                    self.log_fn(f"Extracted grade from text using enhanced fallback: {prediction}")
        except Exception as e:
            self._error_count += 1
            self.log_fn(f"Error extracting prediction: {e}")
            import traceback
            self.log_fn(f"Traceback: {traceback.format_exc()}")
            _structured_log(
                self.log_fn,
                "grading_error",
                {
                    "call_id": call_id,
                    "error_type": type(e).__name__,
                    "error_message": str(e)[:200]
                },
                level="error"
            )
        else:
            # Log successful extraction
            if prediction != "None":
                self._success_count += 1
                _structured_log(
                    self.log_fn,
                    "grading_complete",
                    {
                        "call_id": call_id,
                        "prediction": prediction,
                        "success_rate": self._success_count / self._call_count if self._call_count > 0 else 0
                    }
                )

        return str(prediction), msg_history
    
    def get_stats(self) -> dict:
        """Get agent performance statistics.
        
        Returns:
            Dictionary with call counts and success rates
        """
        return {
            "total_calls": self._call_count,
            "successful_extractions": self._success_count,
            "errors": self._error_count,
            "success_rate": self._success_count / self._call_count if self._call_count > 0 else 0
        }
