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
    Also handles markdown code blocks (```json...```) as fallback.
    Includes robust nested brace counting for complex JSON structures.
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
            # Try to extract JSON from within the content if it's wrapped in other text
            try:
                # Find first { and last } with proper brace counting
                json_str = _extract_json_with_brace_counting(inner)
                if json_str:
                    results.append(json.loads(json_str))
            except (json.JSONDecodeError, ValueError):
                continue
    
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
            try:
                results.append(json.loads(inner))
            except json.JSONDecodeError:
                # Try with brace counting for nested structures
                try:
                    json_str = _extract_json_with_brace_counting(inner)
                    if json_str:
                        results.append(json.loads(json_str))
                except (json.JSONDecodeError, ValueError):
                    continue
    
    # Final fallback: try to find JSON objects directly in text
    if not results:
        try:
            json_str = _extract_json_with_brace_counting(text)
            if json_str:
                results.append(json.loads(json_str))
        except (json.JSONDecodeError, ValueError):
            pass
    
    return results or None


def _extract_json_with_brace_counting(text: str) -> str | None:
    """Extract a JSON object from text using proper brace counting.
    
    This handles nested braces correctly, ensuring we capture complete
    JSON objects even when they contain nested structures.
    
    Args:
        text: The text to search for JSON objects
        
    Returns:
        The extracted JSON string, or None if no valid JSON found
    """
    # Find the first opening brace
    start = text.find("{")
    if start == -1:
        return None
    
    # Count braces to find the matching closing brace
    brace_count = 0
    in_string = False
    escape_next = False
    
    for i, char in enumerate(text[start:]):
        if escape_next:
            escape_next = False
            continue
            
        if char == "\\" and in_string:
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
                    # Found the matching closing brace
                    return text[start:start + i + 1]
    
    # If we get here, braces weren't balanced
    return None


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
   - Almost correct (Almost/6) if the solution is nearly complete with only a minor error or gap
   - Partial credit (Partial/1-5) for incomplete or partially correct solutions with significant gaps
   - Incorrect (Incorrect/0) for fundamentally wrong or empty answers

## Grading Categories (USE THESE EXACT TERMS)
- **Correct**: Complete, correct solution with clear reasoning (7 points)
- **Almost**: Nearly complete solution with only minor errors or small gaps (6 points) - USE THIS when the student is very close to correct
- **Partial**: Significant progress but incomplete or with notable errors (1-5 points)
- **Incorrect**: No meaningful progress or completely wrong approach (0 points)

## Critical Distinctions for Grading

**Correct vs Almost:**
- Correct: The solution is 100% complete and correct. Every step is valid and the conclusion is correct.
- Almost: The solution is 90-95% correct. There's a minor gap, small calculation error, or one missing detail that prevents it from being perfect. The core approach is sound.

**Almost vs Partial (CRITICAL - PAY ATTENTION):**
- Almost (6 points): The student has essentially solved the problem but has a MINOR issue:
  * A small calculation error in an otherwise correct proof
  * A missing edge case that doesn't affect the main result
  * A gap that could be filled with one additional line of reasoning
  * The solution is "almost there" - an expert could fix it in seconds
  
- Partial (1-5 points): The student has made SIGNIFICANT progress but has MAJOR issues:
  * Missing substantial parts of the proof
  * Multiple errors that affect the conclusion
  * Incomplete solution that doesn't reach the final answer
  * Correct approach but execution has serious flaws
  * Only some initial steps are correct

**Key Decision Rule for Almost vs Partial:**
Ask yourself: "Could an expert mathematician fix this solution in under 1 minute?"
- If YES → Use "Almost" (6 points)
- If NO → Use "Partial" (1-5 points)

**Partial vs Incorrect:**
- Partial: The student demonstrated understanding of the problem and made meaningful progress toward the solution.
- Incorrect: The approach is fundamentally wrong, or there's no meaningful progress toward the solution.

## Grading Rubric Reference
- 7 points (Correct): Complete, correct solution with clear reasoning
- 6 points (Almost): Nearly correct, minor gap or small error only. The solution is essentially complete.
- 5 points: Minor gaps or unclear reasoning, but essentially correct approach
- 3-4 points: Significant progress, correct approach but incomplete or with errors
- 1-2 points: Some relevant work or correct initial steps
- 0 points (Incorrect): No meaningful progress or completely wrong approach

## Response Format
You MUST respond in valid JSON format with the following schema:
<json>
{{
    "understanding": "Brief summary of the problem and correct approach",
    "analysis": "Detailed step-by-step analysis of the student's answer, including what they got right and wrong",
    "partial_credit_reasoning": "Explanation of partial credit based on grading guidelines and rubric",
    "grade_category": "One of: 'Correct', 'Almost', 'Partial', 'Incorrect'",
    "numeric_score": "Integer 0-7",
    "response": "Your final grade: use exactly one of 'Correct', 'Almost', 'Partial', 'Incorrect', or numeric 0-7"
}}
</json>

Important: 
- Ensure your JSON is valid and properly formatted
- The "grade_category" field MUST contain exactly one of: 'Correct', 'Almost', 'Partial', 'Incorrect'
- The "numeric_score" field MUST be an integer from 0 to 7
- The "response" field MUST match the grade_category or numeric_score
- Use 'Almost' when the student is very close to a complete solution with only minor issues (6 points)
- Use 'Partial' when there are significant gaps or errors in the solution (1-5 points)
- DO NOT be overly conservative - use 'Almost' when appropriate for solutions that are nearly complete
- When in doubt between 'Almost' and 'Partial', ask: "Could an expert fix this in 1 minute?" If yes, use 'Almost'"""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction using the dedicated method
        prediction = self._extract_prediction(msg_history)
        
        return str(prediction), msg_history

    def _extract_prediction(self, msg_history: list[dict]) -> str:
        """Extract prediction from message history.
        
        Tries multiple extraction strategies and field names for robustness.
        Includes grade normalization for consistent output format.
        Logs detailed analysis for debugging purposes.
        """
        if not msg_history:
            self.log_fn("Warning: Empty message history")
            return "None"
        
        # Get the last assistant message
        last_assistant_msg = None
        for msg in reversed(msg_history):
            if msg.get("role") == "assistant" or "text" in msg:
                last_assistant_msg = msg.get("text", msg.get("content", ""))
                break
        
        if not last_assistant_msg:
            self.log_fn("Warning: No assistant message found in history")
            return "None"
        
        # Try to extract JSON blocks
        extracted = _extract_jsons(last_assistant_msg)
        
        if not extracted:
            # Fallback: try to find any JSON-like structure in the text
            self.log_fn("No JSON blocks found, trying fallback extraction")
            try:
                # Look for patterns like "response": "..." or "grade": "..."
                response_match = re.search(r'["\']response["\']\s*:\s*["\']([^"\']+)["\']', last_assistant_msg)
                if response_match:
                    grade = _normalize_grade(response_match.group(1))
                    self.log_fn(f"Extracted grade from text pattern: {grade}")
                    return grade
                grade_match = re.search(r'["\']grade["\']\s*:\s*["\']([^"\']+)["\']', last_assistant_msg)
                if grade_match:
                    grade = _normalize_grade(grade_match.group(1))
                    self.log_fn(f"Extracted grade from text pattern: {grade}")
                    return grade
                score_match = re.search(r'["\']score["\']\s*:\s*["\']?([^"\'\s,}]+)', last_assistant_msg)
                if score_match:
                    grade = _normalize_grade(score_match.group(1))
                    self.log_fn(f"Extracted grade from text pattern: {grade}")
                    return grade
                category_match = re.search(r'["\']category["\']\s*:\s*["\']([^"\']+)["\']', last_assistant_msg)
                if category_match:
                    grade = _normalize_grade(category_match.group(1))
                    self.log_fn(f"Extracted grade from category field: {grade}")
                    return grade
                # Try to find grade_category field
                grade_category_match = re.search(r'["\']grade_category["\']\s*:\s*["\']([^"\']+)["\']', last_assistant_msg)
                if grade_category_match:
                    grade = _normalize_grade(grade_category_match.group(1))
                    self.log_fn(f"Extracted grade from grade_category field: {grade}")
                    return grade
                # Try to find numeric_score field
                numeric_score_match = re.search(r'["\']numeric_score["\']\s*:\s*(\d+)', last_assistant_msg)
                if numeric_score_match:
                    grade = _normalize_grade(numeric_score_match.group(1))
                    self.log_fn(f"Extracted grade from numeric_score field: {grade}")
                    return grade
                # Try to find grade/score in plain text
                text_grade_match = re.search(r'(?:grade|score|result|category)\s*[:=]\s*["\']?([^"\'\n,}]+)', last_assistant_msg, re.IGNORECASE)
                if text_grade_match:
                    grade = _normalize_grade(text_grade_match.group(1).strip())
                    self.log_fn(f"Extracted grade from text: {grade}")
                    return grade
            except Exception as e:
                self.log_fn(f"Fallback extraction failed: {e}")
            return "None"
        
        # Try to get response from extracted JSON
        last_extract = extracted[-1]
        
        # Log detailed analysis for debugging
        if "understanding" in last_extract:
            self.log_fn(f"Understanding: {str(last_extract['understanding'])[:200]}...")
        if "analysis" in last_extract:
            self.log_fn(f"Analysis: {str(last_extract['analysis'])[:200]}...")
        if "partial_credit_reasoning" in last_extract:
            self.log_fn(f"Partial Credit: {str(last_extract['partial_credit_reasoning'])[:200]}...")
        
        # Priority order for field names - check new structured fields first
        field_priority = ["grade_category", "numeric_score", "response", "grade", "score", "result", "final_grade", "evaluation", "verdict", "category"]
        
        for field in field_priority:
            if field in last_extract:
                value = last_extract[field]
                grade = _normalize_grade(value)
                self.log_fn(f"Extracted grade from JSON field '{field}': {grade}")
                return grade
        
        # If no known field found, return the first string value
        for key, value in last_extract.items():
            if isinstance(value, str) and value:
                grade = _normalize_grade(value)
                self.log_fn(f"Extracted grade from first string value '{key}': {grade}")
                return grade
        
        self.log_fn("Warning: Could not extract grade from JSON")
        return "None"


def _normalize_grade(grade: str | int | float) -> str:
    """Normalize grade to standard format.
    
    Converts various grade formats to a consistent representation.
    Handles both string grades (Correct/Partial/Incorrect/Almost) and numeric scores (0-7).
    Maps numeric scores to grade categories for consistency.
    
    Args:
        grade: The grade value to normalize (can be string, int, or float)
        
    Returns:
        Normalized grade string
    """
    if grade is None:
        return "None"
    
    # Handle non-string types (int, float, etc.)
    if isinstance(grade, (int, float)):
        # IMO scores are integers 0-7
        score = int(grade)
        if 0 <= score <= 7:
            # Map numeric scores to grade categories
            if score == 7:
                return "Correct"
            elif score == 6:
                return "Almost"
            elif 1 <= score <= 5:
                return "Partial"
            else:  # score == 0
                return "Incorrect"
        return str(grade)
    
    if not isinstance(grade, str):
        return str(grade)
    
    grade = grade.strip()
    if not grade:
        return "None"
    
    grade_lower = grade.lower()
    
    # Map common variations to standard grades
    # IMPORTANT: "almost" is a distinct category - very close to correct but not quite
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
        "exactly right": "Correct",
        "almost": "Almost",
        "almost correct": "Almost",
        "nearly correct": "Almost",
        "very close": "Almost",
        "minor issue": "Almost",
        "small error": "Almost",
        "trivial error": "Almost",
        "essentially correct": "Almost",
        "minor gap": "Almost",
        "small gap": "Almost",
        "tiny error": "Almost",
        "minimal error": "Almost",
        "slight error": "Almost",
        "nearly complete": "Almost",
        "almost complete": "Almost",
        "just missed": "Almost",
        "nearly there": "Almost",
        "partial": "Partial",
        "partially correct": "Partial",
        "partial credit": "Partial",
        "partially": "Partial",
        "some credit": "Partial",
        "incomplete": "Partial",
        "mostly correct": "Partial",
        "minor errors": "Partial",
        "unfinished": "Partial",
        "missing": "Partial",
        "mostly": "Partial",
        "some progress": "Partial",
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
    }
    
    # Check for exact match first (case-insensitive)
    if grade_lower in grade_map:
        return grade_map[grade_lower]
    
    # CRITICAL FIX: Check if grade is a numeric string (0-7) and map to category
    # This must happen BEFORE the grade_map check to ensure proper category mapping
    if grade.isdigit():
        score = int(grade)
        if 0 <= score <= 7:
            if score == 7:
                return "Correct"
            elif score == 6:
                return "Almost"
            elif 1 <= score <= 5:
                return "Partial"
            else:  # score == 0
                return "Incorrect"
    
    # Try to extract numeric score from patterns like "Score: 5", "Grade: 3", "5/7", "(6)"
    # Pattern for standalone digit 0-7 (word boundary to avoid matching parts of larger numbers)
    numeric_match = re.search(r'(?:^|\s|\()[0-7](?:$|\s|\))', grade)
    if numeric_match:
        # Extract just the digit and map to category
        digit_match = re.search(r'[0-7]', numeric_match.group(0))
        if digit_match:
            score = int(digit_match.group(0))
            if score == 7:
                return "Correct"
            elif score == 6:
                return "Almost"
            elif 1 <= score <= 5:
                return "Partial"
            else:  # score == 0
                return "Incorrect"
    
    # Additional pattern: check for "6" as a standalone word or in common patterns
    standalone_six = re.search(r'(?:^|\s|[:=])6(?:$|\s|[:=])', grade)
    if standalone_six:
        return "Almost"
    
    # Pattern for "X points" or "X out of 7"
    points_match = re.search(r'(\d+)\s*(?:points?|pts?|/\s*7|out\s+of\s*7)', grade_lower)
    if points_match:
        score = int(points_match.group(1))
        if 0 <= score <= 7:
            if score == 7:
                return "Correct"
            elif score == 6:
                return "Almost"
            elif 1 <= score <= 5:
                return "Partial"
            else:  # score == 0
                return "Incorrect"
    
    # Check for grade in parentheses or brackets
    bracket_match = re.search(r'[\(\[]([0-7]|correct|partial|incorrect|almost)[\)\]]', grade_lower)
    if bracket_match:
        inner = bracket_match.group(1)
        if inner in grade_map:
            return grade_map[inner]
        if inner.isdigit():
            score = int(inner)
            if 0 <= score <= 7:
                if score == 7:
                    return "Correct"
                elif score == 6:
                    return "Almost"
                elif 1 <= score <= 5:
                    return "Partial"
                else:  # score == 0
                    return "Incorrect"
    
    # Check if grade contains keywords (order matters - check specific categories first)
    # Check for "almost" patterns first (highest priority after exact matches)
    if any(word in grade_lower for word in ["almost", "nearly correct", "very close", "minor issue", "small error", "trivial error", "just missed", "nearly there"]):
        return "Almost"
    if "partial" in grade_lower or "partially" in grade_lower:
        return "Partial"
    if any(word in grade_lower for word in ["incomplete", "unfinished", "missing", "mostly"]):
        return "Partial"
    if any(word in grade_lower for word in ["correct", "right", "true", "full", "complete", "solved", "valid", "acceptable"]):
        return "Correct"
    if any(word in grade_lower for word in ["incorrect", "wrong", "false", "none", "zero", "error", "invalid", "unacceptable", "fail", "fundamentally wrong"]):
        return "Incorrect"
    
    # If it's a single digit 0-7, return it mapped to category
    if grade.isdigit() and len(grade) == 1 and 0 <= int(grade) <= 7:
        score = int(grade)
        if score == 7:
            return "Correct"
        elif score == 6:
            return "Almost"
        elif 1 <= score <= 5:
            return "Partial"
        else:  # score == 0
            return "Incorrect"
    
    # Default: return the original grade
    return grade
