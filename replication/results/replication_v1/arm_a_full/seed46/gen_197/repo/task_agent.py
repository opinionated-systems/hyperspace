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
    Now also handles multiple JSON objects and nested JSON within text.
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
        
        # Try direct parsing first
        try:
            results.append(json.loads(inner))
            continue
        except json.JSONDecodeError:
            pass
            
        # Try to extract JSON from within the content if it's wrapped in other text
        try:
            json_str = _extract_json_with_brace_counting(inner)
            if json_str:
                results.append(json.loads(json_str))
                continue
        except (json.JSONDecodeError, ValueError):
            pass
            
        # Try to find all JSON objects in the inner text
        try:
            json_objects = _extract_all_json_objects(inner)
            results.extend(json_objects)
        except Exception:
            pass
    
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
            
            # Try direct parsing first
            try:
                results.append(json.loads(inner))
                continue
            except json.JSONDecodeError:
                pass
                
            # Try with brace counting for nested structures
            try:
                json_str = _extract_json_with_brace_counting(inner)
                if json_str:
                    results.append(json.loads(json_str))
                    continue
            except (json.JSONDecodeError, ValueError):
                pass
                
            # Try to find all JSON objects
            try:
                json_objects = _extract_all_json_objects(inner)
                results.extend(json_objects)
            except Exception:
                pass
    
    # Final fallback: try to find JSON objects directly in text
    if not results:
        try:
            json_objects = _extract_all_json_objects(text)
            results.extend(json_objects)
        except Exception:
            pass
    
    return results if results else None


def _extract_all_json_objects(text: str) -> list[dict]:
    """Extract all valid JSON objects from text.
    
    Scans through the text and attempts to parse JSON objects at each
    opening brace position, returning all successfully parsed objects.
    
    Args:
        text: The text to search for JSON objects
        
    Returns:
        List of parsed JSON objects (dicts)
    """
    results = []
    i = 0
    while i < len(text):
        # Find next opening brace
        brace_pos = text.find("{", i)
        if brace_pos == -1:
            break
            
        # Try to extract JSON starting at this position
        try:
            json_str = _extract_json_at_position(text, brace_pos)
            if json_str:
                obj = json.loads(json_str)
                if isinstance(obj, dict):
                    results.append(obj)
                i = brace_pos + len(json_str)
                continue
        except (json.JSONDecodeError, ValueError):
            pass
        
        i = brace_pos + 1
    
    return results


def _extract_json_at_position(text: str, start: int) -> str | None:
    """Extract a JSON object starting at a specific position.
    
    Args:
        text: The text containing the JSON
        start: The position of the opening brace
        
    Returns:
        The extracted JSON string, or None if no valid JSON found
    """
    if start >= len(text) or text[start] != "{":
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
- The proof or solution is rigorous and would receive full marks (7/7) in competition
- No errors, omissions, or unclear reasoning

**Almost**: The solution is NEARLY CORRECT with only MINOR FLAWS. The student clearly understood the problem and had the right approach, but there are small issues that would result in 5-6 points (not 7). This is for solutions that are ESSENTIALLY CORRECT but not perfect. Examples include:
- Small computational errors in an otherwise correct approach
- Missing trivial justifications that don't affect the core logic
- Slightly incomplete final answers where the method is clearly correct
- Minor notational issues that don't obscure the mathematical reasoning
- The core proof structure is correct and complete
- Solutions that would receive 5-6 points in competition

**Partial**: The solution shows MEANINGFUL progress but has SUBSTANTIAL GAPS or ERRORS. The student demonstrated some understanding but the solution is far from complete. This includes:
- Correct initial approach but incomplete execution (missing key steps)
- Some correct steps mixed with significant errors
- Demonstrated understanding of key concepts but failed to complete the solution
- Made meaningful progress toward the solution (not just restating the problem)
- Solutions that would receive 1-4 points in competition

**Incorrect**: The solution shows NO MEANINGFUL PROGRESS or is FUNDAMENTALLY WRONG. This includes:
- Completely wrong approach or fundamental misunderstanding of the problem
- No relevant work or empty answer
- Major conceptual errors throughout
- Failed to demonstrate any understanding of the problem
- Only restating the problem or making trivial observations without real progress
- Solutions that would receive 0 points in competition

## Critical Distinctions (READ CAREFULLY - THIS IS THE KEY TO ACCURATE GRADING)

**Correct vs Almost** (THE MOST IMPORTANT DISTINCTION):
- "Correct" = FLAWLESS solution that would receive 7/7 points. NO errors, NO omissions, NO unclear reasoning.
- "Almost" = ESSENTIALLY CORRECT but with MINOR FLAWS that would result in 5-6 points (not 7).
- ASK YOURSELF: "Would this solution receive full marks (7/7)?" If NO but it's very close → "Almost"

**Almost vs Partial**:
- "Almost" = Solution is NEARLY COMPLETE. The core approach is correct and MOSTLY executed. Would receive 5-6 points.
- "Partial" = Significant gaps remain. The solution is INCOMPLETE and would receive 1-4 points.
- ASK YOURSELF: "Is this solution mostly done with just minor issues, or are there big gaps?"

**Partial vs Incorrect**:
- "Partial" = MEANINGFUL PROGRESS beyond just restating the problem. Some correct steps toward the solution.
- "Incorrect" = NO MEANINGFUL PROGRESS. Just restating definitions, trivial observations, or completely wrong approach.
- ASK YOURSELF: "Did the student make a genuine attempt with some correct steps, or is there no real progress?"

## Decision Flowchart (FOLLOW THIS ORDER)
1. Is the solution COMPLETELY FLAWLESS (would get 7/7)? → **Correct**
2. Is the solution ESSENTIALLY CORRECT with only MINOR FLAWS (would get 5-6/7)? → **Almost**
3. Does the solution show MEANINGFUL PROGRESS with some correct steps (would get 1-4/7)? → **Partial**
4. Is there NO MEANINGFUL PROGRESS or completely wrong (would get 0/7)? → **Incorrect**

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
    "partial_credit_reasoning": "Explanation of partial credit based on grading guidelines and rubric. EXPLICITLY STATE which grade category applies and WHY. For 'Almost', explain what minor flaws exist. For 'Partial', explain what meaningful progress was made. For 'Incorrect', explain why there's no meaningful progress.",
    "response": "Your final grade: must be exactly one of: 'Correct', 'Almost', 'Partial', or 'Incorrect'"
}}
</json>

Important: 
- Ensure your JSON is valid and properly formatted
- The "response" field MUST contain exactly one of: 'Correct', 'Almost', 'Partial', or 'Incorrect'
- Be CAREFUL with the "Almost" category - use it when the solution is ESSENTIALLY CORRECT but has MINOR FLAWS (5-6/7 points)
- Be STRICT about the "Incorrect" category - only use it when there is truly NO MEANINGFUL PROGRESS
- When in doubt between categories, use the Decision Flowchart above
- DOUBLE-CHECK: If you selected 'Almost', verify the solution would indeed receive 5-6 points, not 7 and not 1-4"""

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
        Now includes confidence scoring and multiple candidate extraction.
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
            return self._extract_grade_from_text(last_assistant_msg)
        
        # Try to get response from extracted JSON objects
        # Process all extracted objects and pick the best one
        candidates = []
        
        for idx, extract in enumerate(extracted):
            if not isinstance(extract, dict):
                continue
                
            # Log detailed analysis for debugging (only for the last one)
            if idx == len(extracted) - 1:
                if "understanding" in extract:
                    self.log_fn(f"Understanding: {str(extract['understanding'])[:200]}...")
                if "analysis" in extract:
                    self.log_fn(f"Analysis: {str(extract['analysis'])[:200]}...")
                if "partial_credit_reasoning" in extract:
                    self.log_fn(f"Partial Credit: {str(extract['partial_credit_reasoning'])[:200]}...")
            
            # Priority order for field names with confidence scores
            field_priority = [
                ("response", 1.0),
                ("grade", 0.95),
                ("score", 0.9),
                ("result", 0.85),
                ("final_grade", 0.85),
                ("evaluation", 0.8),
                ("verdict", 0.8),
                ("category", 0.75),
                ("classification", 0.75),
            ]
            
            for field, confidence in field_priority:
                if field in extract:
                    value = extract[field]
                    grade = _normalize_grade(value)
                    if grade and grade != "None":
                        candidates.append((grade, confidence, field, idx))
                        break
            else:
                # If no known field found, try the first string value
                for key, value in extract.items():
                    if isinstance(value, str) and value:
                        grade = _normalize_grade(value)
                        if grade and grade != "None":
                            candidates.append((grade, 0.5, key, idx))
                            break
        
        # Select the best candidate based on confidence and position
        if candidates:
            # Sort by confidence (descending) and then by position (ascending - prefer later)
            candidates.sort(key=lambda x: (-x[1], -x[3]))
            best_grade, confidence, field, idx = candidates[0]
            self.log_fn(f"Extracted grade '{best_grade}' from JSON field '{field}' (confidence: {confidence})")
            return best_grade
        
        self.log_fn("Warning: Could not extract grade from JSON")
        return "None"
    
    def _extract_grade_from_text(self, text: str) -> str:
        """Extract grade from plain text using multiple patterns.
        
        Args:
            text: The text to search for grade information
            
        Returns:
            Normalized grade string or "None" if not found
        """
        try:
            # Look for patterns like "response": "..." or "grade": "..."
            patterns = [
                (r'["\']response["\']\s*:\s*["\']([^"\']+)["\']', "response field"),
                (r'["\']grade["\']\s*:\s*["\']([^"\']+)["\']', "grade field"),
                (r'["\']score["\']\s*:\s*["\']?([^"\'\s,}]+)', "score field"),
                (r'["\']result["\']\s*:\s*["\']([^"\']+)["\']', "result field"),
                (r'["\']evaluation["\']\s*:\s*["\']([^"\']+)["\']', "evaluation field"),
                (r'(?:grade|score|result|evaluation)\s*[:=]\s*["\']?([^"\'\n,}]+)', "text pattern"),
                (r'(?:final\s+(?:grade|score|result))\s*[:=]\s*["\']?([^"\'\n,}]+)', "final pattern"),
                (r'(?:the\s+(?:grade|score|result)\s+(?:is|should\s+be))\s*[:=]?\s*["\']?([^"\'\n,}]+)', "declarative pattern"),
            ]
            
            for pattern, pattern_name in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    grade = _normalize_grade(match.group(1).strip())
                    if grade and grade != "None":
                        self.log_fn(f"Extracted grade '{grade}' from {pattern_name}")
                        return grade
            
            # Try to find standalone grade mentions
            standalone_patterns = [
                (r'\b(Correct|Almost|Partial|Incorrect)\b', "standalone category"),
                (r'\b([0-7])\s*/\s*7\b', "score out of 7"),
                (r'\(\s*([0-7])\s*\)', "score in parentheses"),
            ]
            
            for pattern, pattern_name in standalone_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    grade = _normalize_grade(match.group(1).strip())
                    if grade and grade != "None":
                        self.log_fn(f"Extracted grade '{grade}' from {pattern_name}")
                        return grade
                        
        except Exception as e:
            self.log_fn(f"Fallback extraction failed: {e}")
        
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
    if any(word in grade_lower for word in ["almost", "nearly correct", "nearly", "minor error", "small error", "trivial error", "essentially correct", "mostly correct"]):
        return "Almost"
    # Check for "partial" category (significant progress but substantial gaps)
    if "partial" in grade_lower or "partially" in grade_lower:
        return "Partial"
    if any(word in grade_lower for word in ["incomplete", "unfinished", "missing", "significant progress", "some progress", "partial solution"]):
        return "Partial"
    # Check for "correct" category (fully correct) - but be careful not to match "almost correct"
    if any(word in grade_lower for word in ["correct", "right", "true", "full", "complete", "solved", "valid", "acceptable", "perfect", "exact"]):
        # Double-check it's not "almost correct" or similar
        if "almost" not in grade_lower and "nearly" not in grade_lower and "mostly" not in grade_lower:
            return "Correct"
    # Check for "incorrect" category (fundamentally wrong)
    if any(word in grade_lower for word in ["incorrect", "wrong", "false", "none", "zero", "error", "invalid", "unacceptable", "fail", "failed", "no progress", "fundamentally wrong"]):
        return "Incorrect"
    
    # If it's a single digit 0-7, return it
    if grade.isdigit() and len(grade) == 1 and 0 <= int(grade) <= 7:
        return grade
    
    # Default: return the original grade
    return grade
