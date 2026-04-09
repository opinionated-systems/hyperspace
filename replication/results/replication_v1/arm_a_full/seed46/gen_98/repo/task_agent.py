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
   - Partial credit (Partial/1-6) for incomplete or partially correct solutions
   - Incorrect (Incorrect/0) for fundamentally wrong or empty answers

## Grade Category Definitions (CRITICAL - READ CAREFULLY)

**Correct**: The solution is completely correct with valid reasoning and clear presentation. ALL of the following must be true:
- All steps are mathematically sound
- The final answer matches the official solution exactly
- The reasoning is complete with no gaps
- The proof or solution is rigorous and would receive full marks in competition

**Almost**: The solution is nearly correct but has minor flaws. The student clearly understood the problem and had the right approach, but there are small issues. This includes:
- Small computational errors in an otherwise correct approach
- Missing trivial justifications that don't affect the core logic
- Slightly incomplete final answers where the method is clearly correct
- Minor notational issues that don't obscure the mathematical reasoning
- The core proof structure is correct and complete

**Partial**: The solution shows MEANINGFUL progress but has substantial gaps or errors. The student demonstrated some understanding but the solution is far from complete. This includes:
- Correct initial approach but incomplete execution (missing key steps)
- Some correct steps mixed with significant errors
- Demonstrated understanding of key concepts but failed to complete the solution
- Made meaningful progress toward the solution (not just restating the problem)

**Incorrect**: The solution shows NO MEANINGFUL PROGRESS or is fundamentally wrong. This includes:
- Completely wrong approach or fundamental misunderstanding of the problem
- No relevant work or empty answer
- Major conceptual errors throughout
- Failed to demonstrate any understanding of the problem
- Only restating the problem or making trivial observations without real progress
- Solutions that start with correct definitions but make no substantive progress

## Critical Distinctions (READ CAREFULLY)

**Correct vs Almost**: 
- "Correct" means the solution is complete and would receive full marks (7 points)
- "Almost" means the solution is essentially correct but has minor flaws that would result in 5-6 points (not 7)
- KEY DISTINCTION: "Almost" solutions have the RIGHT APPROACH and NEARLY COMPLETE execution - they just have small errors

**Almost vs Partial**:
- "Almost" means the solution is nearly complete - the core approach is correct and MOSTLY executed (5-6 points)
  * The student clearly understood the problem and had the right idea
  * Most steps are correct, with only minor gaps or small errors
  * The solution is substantially complete, just needs minor fixes
- "Partial" means significant gaps remain - the solution is incomplete and would receive 1-4 points
  * The student made some progress but the solution is far from complete
  * Major steps are missing or there are significant errors
  * The approach may be partially correct but execution is lacking

**Partial vs Incorrect**:
- "Partial" requires MEANINGFUL PROGRESS beyond just restating the problem
- If the student only restates definitions, makes trivial observations, or shows no real understanding → "Incorrect"
- If the student makes a genuine attempt at the solution with some correct steps → "Partial"

## Decision Framework for "Almost" vs "Partial"
When deciding between "Almost" and "Partial", ask yourself:
1. Did the student have the RIGHT APPROACH? (Yes → lean toward Almost, No → Partial/Incorrect)
2. Is the solution MOSTLY COMPLETE with only minor gaps? (Yes → Almost, No → Partial)
3. Would this receive 5-6 points in competition? (Yes → Almost, 1-4 points → Partial)
4. Are the errors MINOR (small calculation, missing trivial justification) or MAJOR (wrong method, missing key steps)? (Minor → Almost, Major → Partial)

## Grading Rubric Reference
- 7 points: Complete, correct solution with clear reasoning → "Correct"
- 5-6 points: Minor gaps or unclear reasoning, but essentially correct approach → "Almost"
- 3-4 points: Significant progress, correct approach but incomplete or with errors → "Partial"
- 1-2 points: Some relevant work or correct initial steps → "Partial"
- 0 points: No meaningful progress or completely wrong approach → "Incorrect"

## Response Format
You MUST respond in valid JSON format with the following schema:
<json>
{{
    "understanding": "Brief summary of the problem and correct approach",
    "analysis": "Detailed step-by-step analysis of the student's answer, including what they got right and wrong",
    "partial_credit_reasoning": "Explanation of partial credit based on grading guidelines and rubric. EXPLICITLY STATE which grade category applies and WHY. For 'Almost': explain what minor flaws exist and why the solution is nearly complete (5-6 points). For 'Partial': explain what meaningful progress was made but why significant gaps remain (1-4 points).",
    "response": "Your final grade: must be exactly one of: 'Correct', 'Almost', 'Partial', or 'Incorrect'"
}}
</json>

Important: 
- Ensure your JSON is valid and properly formatted
- The "response" field MUST contain exactly one of: 'Correct', 'Almost', 'Partial', or 'Incorrect'
- Be STRICT about the "Incorrect" category - only use it when there is truly NO MEANINGFUL PROGRESS
- When deciding between "Almost" and "Partial": "Almost" = right approach + mostly complete (5-6 points), "Partial" = some progress but significant gaps (1-4 points)
- When in doubt between categories, consider: Does this solution show meaningful progress beyond restating the problem? Would this receive >0 points in competition?"""

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
                # Try to find grade/score in plain text
                text_grade_match = re.search(r'(?:grade|score|result)\s*[:=]\s*["\']?([^"\'\n,}]+)', last_assistant_msg, re.IGNORECASE)
                if text_grade_match:
                    grade = _normalize_grade(text_grade_match.group(1).strip())
                    self.log_fn(f"Extracted grade from text: {grade}")
                    return grade
                # Try to find standalone grade words in the text
                for category in ["Correct", "Almost", "Partial", "Incorrect"]:
                    if re.search(rf'\b{category}\b', last_assistant_msg, re.IGNORECASE):
                        self.log_fn(f"Extracted grade from standalone word: {category}")
                        return category
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
        
        # Priority order for field names - response is most important
        field_priority = ["response", "grade", "score", "result", "final_grade", "evaluation", "verdict", "category", "classification"]
        
        for field in field_priority:
            if field in last_extract:
                value = last_extract[field]
                grade = _normalize_grade(value)
                self.log_fn(f"Extracted grade from JSON field '{field}': {grade}")
                return grade
        
        # If no known field found, return the first string value that looks like a grade
        for key, value in last_extract.items():
            if isinstance(value, str) and value:
                normalized = _normalize_grade(value)
                if normalized in ["Correct", "Almost", "Partial", "Incorrect", "0", "1", "2", "3", "4", "5", "6", "7"]:
                    self.log_fn(f"Extracted grade from first valid string value '{key}': {normalized}")
                    return normalized
        
        # Last resort: return the first string value
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
        if 0 <= grade <= 7:
            return str(int(grade))
        return str(grade)
    
    if not isinstance(grade, str):
        return str(grade)
    
    grade = grade.strip()
    if not grade:
        return "None"
    
    grade_lower = grade.lower()
    
    # Map common variations to standard grades
    # Priority: Check for "Almost" first to avoid matching "almost correct" as "Correct"
    grade_map = {
        # Exact matches for "Almost" category
        "almost": "Almost",
        "almost correct": "Almost",
        "nearly correct": "Almost",
        "nearly": "Almost",
        "minor errors": "Almost",
        "small errors": "Almost",
        "mostly correct": "Almost",
        "mostly": "Almost",
        "essentially correct": "Almost",
        "minor flaw": "Almost",
        "minor flaws": "Almost",
        "trivial error": "Almost",
        "trivial errors": "Almost",
        # Exact matches for "Correct" category
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
        "exact": "Correct",
        # Exact matches for "Partial" category
        "partial": "Partial",
        "partially correct": "Partial",
        "partial credit": "Partial",
        "partially": "Partial",
        "some credit": "Partial",
        "incomplete": "Partial",
        "unfinished": "Partial",
        "significant progress": "Partial",
        "some progress": "Partial",
        "partial solution": "Partial",
        # Exact matches for "Incorrect" category
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
        "error": "Incorrect",
        "errors": "Incorrect",
        "fundamentally wrong": "Incorrect",
        "no progress": "Incorrect",
        # Numeric scores
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
    points_match = re.search(r'(\d+)\s*(?:points?|pts?|/\s*7|out\s+of\s*7)', grade_lower)
    if points_match:
        score = int(points_match.group(1))
        if 0 <= score <= 7:
            return str(score)
    
    # Check for grade in parentheses or brackets
    bracket_match = re.search(r'[\(\[]([0-7]|correct|partial|incorrect|almost)[\)\]]', grade_lower)
    if bracket_match:
        inner = bracket_match.group(1)
        if inner in grade_map:
            return grade_map[inner]
        if inner.isdigit() and 0 <= int(inner) <= 7:
            return inner
    
    # Check if grade contains keywords (order matters - check specific categories first)
    # Check for "almost" category first (nearly correct with minor flaws)
    # Use word boundaries to avoid partial matches
    almost_patterns = [
        r'\balmost\b', r'\balmost correct\b', r'\bnearly correct\b', r'\bnearly\b',
        r'\bminor error\b', r'\bminor errors\b', r'\bsmall error\b', r'\bsmall errors\b',
        r'\btrivial error\b', r'\btrivial errors\b', r'\bessentially correct\b',
        r'\bmostly correct\b', r'\bmostly right\b', r'\bminor flaw\b', r'\bminor flaws\b',
        r'\bsubstantially correct\b', r'\bclose to correct\b', r'\bvery close\b',
        r'\b5[\-/]7\b', r'\b6[\-/]7\b',  # Score patterns like 5/7 or 6-7
    ]
    for pattern in almost_patterns:
        if re.search(pattern, grade_lower):
            return "Almost"
    
    # Check for "partial" category (significant progress but substantial gaps)
    partial_patterns = [
        r'\bpartial\b', r'\bpartially\b', r'\bpartially correct\b', r'\bincomplete\b',
        r'\bunfinished\b', r'\bsignificant progress\b', r'\bsome progress\b',
        r'\bpartial solution\b', r'\bpartly correct\b', r'\bpartly\b',
        r'\b3[\-/]7\b', r'\b4[\-/]7\b',  # Score patterns like 3/7 or 4/7
        r'\b1[\-/]7\b', r'\b2[\-/]7\b',  # Lower scores also indicate partial
    ]
    for pattern in partial_patterns:
        if re.search(pattern, grade_lower):
            return "Partial"
    
    # Check for "correct" category (fully correct) - but be careful not to match "almost correct"
    correct_patterns = [
        r'\bcorrect\b', r'\bright\b', r'\btrue\b', r'\bfull\b', r'\bcomplete\b',
        r'\bsolved\b', r'\bvalid\b', r'\bacceptable\b', r'\bperfect\b', r'\bexact\b',
        r'\b7[\-/]7\b',  # Full score
    ]
    for pattern in correct_patterns:
        if re.search(pattern, grade_lower):
            # Double-check it's not "almost correct" or similar
            if "almost" not in grade_lower and "nearly" not in grade_lower and "mostly" not in grade_lower:
                return "Correct"
    
    # Check for "incorrect" category (fundamentally wrong)
    incorrect_patterns = [
        r'\bincorrect\b', r'\bwrong\b', r'\bfalse\b', r'\bnone\b', r'\bzero\b',
        r'\binvalid\b', r'\bunacceptable\b', r'\bfail\b', r'\bfailed\b',
        r'\bno progress\b', r'\bfundamentally wrong\b', r'\b0[\-/]7\b',
    ]
    for pattern in incorrect_patterns:
        if re.search(pattern, grade_lower):
            return "Incorrect"
    
    # If it's a single digit 0-7, map to appropriate category or return as-is
    if grade.isdigit() and len(grade) == 1 and 0 <= int(grade) <= 7:
        score = int(grade)
        # Map numeric scores to categories for consistency
        if score == 7:
            return "Correct"
        elif score >= 5:  # 5-6 points
            return "Almost"
        elif score >= 1:  # 1-4 points
            return "Partial"
        else:  # 0 points
            return "Incorrect"
    
    # Default: return the original grade
    return grade
