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
    if not text or not isinstance(text, str):
        return None
        
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
    if not text or not isinstance(text, str):
        return None
        
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
                    json_str = text[start:start + i + 1]
                    # Validate it's actually valid JSON
                    try:
                        json.loads(json_str)
                        return json_str
                    except json.JSONDecodeError:
                        # Try next occurrence if this one isn't valid
                        next_start = text.find("{", start + 1)
                        if next_start != -1:
                            return _extract_json_with_brace_counting(text[next_start:])
                        return None
    
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

**Almost**: The solution is NEARLY CORRECT with only MINOR FLAWS. The student clearly understood the problem and executed the right approach, but made small mistakes. Think of this as "would receive 5-6 points out of 7". This includes:
- Small computational errors in an otherwise correct approach
- Missing trivial justifications that don't affect the core logic
- Slightly incomplete final answers where the method is clearly correct
- Minor notational issues that don't obscure the mathematical reasoning
- The core proof structure is correct and nearly complete
- Key insights are correct but there are minor gaps in rigor

**Partial**: The solution shows MEANINGFUL PROGRESS but has SUBSTANTIAL GAPS. The student demonstrated understanding of key concepts and made genuine progress, but the solution is incomplete or has significant errors. Think of this as "would receive 1-4 points out of 7". This includes:
- Correct initial approach but incomplete execution (missing key steps)
- Some correct steps mixed with significant errors
- Demonstrated understanding of key concepts but failed to complete the solution
- Made genuine progress toward the solution (not just restating the problem)
- Has the right idea but execution is flawed or incomplete

**Incorrect**: The solution shows NO MEANINGFUL PROGRESS or is FUNDAMENTALLY WRONG. This includes:
- Completely wrong approach or fundamental misunderstanding of the problem
- No relevant work or empty answer
- Major conceptual errors throughout
- Failed to demonstrate any understanding of the problem
- Only restating the problem or making trivial observations without real progress
- Solutions that start with correct definitions but make no substantive progress

## Critical Distinctions (READ CAREFULLY)

**Correct vs Almost**: 
- "Correct" = complete solution, would receive 7/7 points
- "Almost" = nearly complete, minor flaws only, would receive 5-6/7 points

**Almost vs Partial**:
- "Almost" = core solution is there, just needs minor fixes (5-6 points)
- "Partial" = significant work done but major gaps remain (1-4 points)
- KEY QUESTION: If the student fixed their mistakes, would they have a complete solution? If YES → "Almost", if NO → "Partial"

**Partial vs Incorrect**:
- "Partial" = genuine attempt with some correct reasoning (1-4 points)
- "Incorrect" = no meaningful progress or completely wrong (0 points)
- KEY QUESTION: Did the student demonstrate understanding of what the problem is asking and make progress toward solving it? If YES → "Partial", if NO → "Incorrect"

## Grading Rubric Reference
- 7 points: Complete, correct solution with clear reasoning → "Correct"
- 5-6 points: Minor gaps or unclear reasoning, but essentially correct approach → "Almost"
- 3-4 points: Significant progress, correct approach but incomplete or with errors → "Partial"
- 1-2 points: Some relevant work or correct initial steps → "Partial"
- 0 points: No meaningful progress or completely wrong approach → "Incorrect"

## Common Mistakes to AVOID

**DO NOT** default to "Correct" or "Incorrect" - most solutions fall in the middle categories.

**DO NOT** use "Almost" for solutions with major gaps or missing key steps.

**DO NOT** use "Partial" for solutions that are nearly complete with only minor issues.

**DO NOT** use "Incorrect" for solutions that show any meaningful understanding or progress.

**DO** use "Almost" when the student clearly understood the problem and the solution is nearly complete.

**DO** use "Partial" when the student made genuine progress but has substantial gaps remaining.

**DO** use "Incorrect" only when there is truly NO meaningful progress or the approach is fundamentally wrong.

## Response Format
You MUST respond in valid JSON format with the following schema:
<json>
{{
    "understanding": "Brief summary of the problem and correct approach",
    "analysis": "Detailed step-by-step analysis of the student's answer, including what they got right and wrong",
    "partial_credit_reasoning": "Explanation of partial credit based on grading guidelines and rubric. EXPLICITLY STATE which grade category applies and WHY. Use the KEY QUESTIONS from the Critical Distinctions section above to justify your choice.",
    "response": "Your final grade: must be exactly one of: 'Correct', 'Almost', 'Partial', or 'Incorrect'"
}}
</json>

Important: 
- Ensure your JSON is valid and properly formatted
- The "response" field MUST contain exactly one of: 'Correct', 'Almost', 'Partial', or 'Incorrect'
- Be STRICT about the "Incorrect" category - only use it when there is truly NO MEANINGFUL PROGRESS
- When deciding between categories, USE THE KEY QUESTIONS:
  1. Almost vs Partial: "If the student fixed their mistakes, would they have a complete solution?" YES → Almost, NO → Partial
  2. Partial vs Incorrect: "Did the student demonstrate understanding and make progress toward solving it?" YES → Partial, NO → Incorrect
- Remember: Almost = 5-6 points, Partial = 1-4 points, Incorrect = 0 points
- CRITICAL: Most solutions are NOT "Correct" or "Incorrect" - they fall in the "Almost" or "Partial" categories. Do not default to extremes.
- Before assigning "Correct", verify there are NO gaps or errors at all.
- Before assigning "Incorrect", verify there is truly NO meaningful progress (not just that the solution has errors).
- "Almost" and "Partial" are the most common categories - use them when appropriate!"""

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
        Validates that extracted grades are in the valid set.
        """
        VALID_GRADES = {"Correct", "Almost", "Partial", "Incorrect", "0", "1", "2", "3", "4", "5", "6", "7", "None"}
        
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
                    if grade in VALID_GRADES:
                        return grade
                grade_match = re.search(r'["\']grade["\']\s*:\s*["\']([^"\']+)["\']', last_assistant_msg)
                if grade_match:
                    grade = _normalize_grade(grade_match.group(1))
                    self.log_fn(f"Extracted grade from text pattern: {grade}")
                    if grade in VALID_GRADES:
                        return grade
                score_match = re.search(r'["\']score["\']\s*:\s*["\']?([^"\'\s,}]+)', last_assistant_msg)
                if score_match:
                    grade = _normalize_grade(score_match.group(1))
                    self.log_fn(f"Extracted grade from text pattern: {grade}")
                    if grade in VALID_GRADES:
                        return grade
                # Try to find grade/score in plain text
                text_grade_match = re.search(r'(?:grade|score|result)\s*[:=]\s*["\']?([^"\'\n,}]+)', last_assistant_msg, re.IGNORECASE)
                if text_grade_match:
                    grade = _normalize_grade(text_grade_match.group(1).strip())
                    self.log_fn(f"Extracted grade from text: {grade}")
                    if grade in VALID_GRADES:
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
        
        # Priority order for field names
        field_priority = ["response", "grade", "score", "result", "final_grade", "evaluation", "verdict", "category", "classification"]
        
        for field in field_priority:
            if field in last_extract:
                value = last_extract[field]
                grade = _normalize_grade(value)
                self.log_fn(f"Extracted grade from JSON field '{field}': {grade}")
                if grade in VALID_GRADES:
                    return grade
                else:
                    self.log_fn(f"Warning: Extracted grade '{grade}' not in valid grades, trying next field")
        
        # If no known field found, return the first string value
        for key, value in last_extract.items():
            if isinstance(value, str) and value:
                grade = _normalize_grade(value)
                self.log_fn(f"Extracted grade from first string value '{key}': {grade}")
                if grade in VALID_GRADES:
                    return grade
        
        self.log_fn("Warning: Could not extract valid grade from JSON")
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
        "close": "Almost",
        "very close": "Almost",
        "minor issues": "Almost",
        "small issues": "Almost",
        "nearly there": "Almost",
        "almost there": "Almost",
        "minor gap": "Almost",
        "minor gaps": "Almost",
        "small gap": "Almost",
        "small gaps": "Almost",
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
        "fully correct": "Correct",
        "entirely correct": "Correct",
        "completely correct": "Correct",
        "totally correct": "Correct",
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
        "half correct": "Partial",
        "halfway": "Partial",
        "mixed": "Partial",
        "mixed results": "Partial",
        "some correct": "Partial",
        "some right": "Partial",
        "partial success": "Partial",
        "in progress": "Partial",
        "started": "Partial",
        "attempted": "Partial",
        "tried": "Partial",
        "some understanding": "Partial",
        "partial understanding": "Partial",
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
        "empty": "Incorrect",
        "blank": "Incorrect",
        "nothing": "Incorrect",
        "no answer": "Incorrect",
        "no solution": "Incorrect",
        "completely wrong": "Incorrect",
        "totally wrong": "Incorrect",
        "entirely wrong": "Incorrect",
        "not correct": "Incorrect",
        "not right": "Incorrect",
        "not valid": "Incorrect",
        "not acceptable": "Incorrect",
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
    almost_keywords = [
        "almost", "nearly correct", "nearly", "minor error", "small error", "trivial error",
        "essentially correct", "mostly correct", "very close", "close to correct",
        "minor issue", "small issue", "nearly there", "almost there", "minor gap",
        "small gap", "minor flaw", "small flaw", "trivial flaw"
    ]
    if any(word in grade_lower for word in almost_keywords):
        return "Almost"
    
    # Check for "partial" category (significant progress but substantial gaps)
    if "partial" in grade_lower or "partially" in grade_lower:
        return "Partial"
    partial_keywords = [
        "incomplete", "unfinished", "missing", "significant progress", "some progress",
        "partial solution", "half correct", "halfway", "mixed", "some correct",
        "some right", "in progress", "started", "attempted", "tried",
        "some understanding", "partial understanding", "some work", "partial work"
    ]
    if any(word in grade_lower for word in partial_keywords):
        return "Partial"
    
    # Check for "correct" category (fully correct) - but be careful not to match "almost correct"
    correct_keywords = ["correct", "right", "true", "full", "complete", "solved", "valid", "acceptable", "perfect", "exact"]
    if any(word in grade_lower for word in correct_keywords):
        # Double-check it's not "almost correct" or similar
        if "almost" not in grade_lower and "nearly" not in grade_lower and "mostly" not in grade_lower:
            return "Correct"
    
    # Check for "incorrect" category (fundamentally wrong)
    incorrect_keywords = [
        "incorrect", "wrong", "false", "none", "zero", "error", "invalid",
        "unacceptable", "fail", "failed", "no progress", "fundamentally wrong",
        "empty", "blank", "nothing", "no answer", "no solution", "completely wrong"
    ]
    if any(word in grade_lower for word in incorrect_keywords):
        return "Incorrect"
    
    # If it's a single digit 0-7, return it
    if grade.isdigit() and len(grade) == 1 and 0 <= int(grade) <= 7:
        return grade
    
    # Default: return the original grade
    return grade
