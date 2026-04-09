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
    Includes enhanced logging for debugging extraction failures.
    """
    results = []
    search_from = 0
    extraction_attempts = 0
    
    # Primary: Extract from <json> tags
    while True:
        start = text.find("<json>", search_from)
        if start == -1:
            break
        end = text.find("</json>", start)
        if end == -1:
            logger.debug("Found opening <json> tag but no closing </json> tag")
            break
        inner = text[start + 6:end].strip()
        search_from = end + 7
        extraction_attempts += 1
        try:
            parsed = json.loads(inner)
            results.append(parsed)
            logger.debug(f"Successfully extracted JSON from <json> block #{extraction_attempts}")
        except json.JSONDecodeError as e:
            logger.debug(f"JSON decode error in <json> block #{extraction_attempts}: {e}")
            # Try to clean common issues and retry
            try:
                # Remove trailing commas before closing braces/brackets
                cleaned = re.sub(r',(\s*[}\]])', r'\1', inner)
                parsed = json.loads(cleaned)
                results.append(parsed)
                logger.debug(f"Successfully extracted JSON after cleaning trailing commas")
            except json.JSONDecodeError:
                continue
    
    # Fallback 1: Extract from markdown code blocks
    if not results:
        logger.debug("No JSON found in <json> tags, trying markdown code blocks")
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
            extraction_attempts += 1
            try:
                parsed = json.loads(inner)
                results.append(parsed)
                logger.debug(f"Successfully extracted JSON from markdown block #{extraction_attempts}")
            except json.JSONDecodeError as e:
                logger.debug(f"JSON decode error in markdown block #{extraction_attempts}: {e}")
                continue
    
    # Fallback 2: Find JSON objects directly in text
    if not results:
        logger.debug("No JSON found in code blocks, trying direct JSON extraction")
        brace_start = text.find("{")
        attempt_count = 0
        while brace_start != -1:
            attempt_count += 1
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
                                logger.debug(f"Successfully extracted JSON from direct parsing (attempt {attempt_count})")
                        except json.JSONDecodeError:
                            pass
                        break
            brace_start = text.find("{", brace_start + 1)
    
    if results:
        logger.info(f"Successfully extracted {len(results)} JSON object(s) from text")
    else:
        logger.warning(f"Failed to extract any valid JSON after {extraction_attempts} attempts")
    
    return results or None


def _normalize_grade(grade: str) -> str:
    """Normalize grade to standard format.
    
    Converts various grade formats to a consistent representation.
    Handles both string grades (Correct/Partial/Incorrect) and numeric scores (0-7).
    Includes enhanced logging for debugging grade normalization.
    """
    original_grade = grade
    
    if not grade:
        logger.debug("Grade is empty/None, returning 'None'")
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
        "partial": "Partial",
        "partially correct": "Partial",
        "partial credit": "Partial",
        "partially": "Partial",
        "some credit": "Partial",
        "incomplete": "Partial",
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
        result = grade_map[grade_lower]
        logger.debug(f"Normalized grade '{original_grade}' -> '{result}' (exact match)")
        return result
    
    # Try to extract numeric score from strings like "Score: 5" or "Grade: 3"
    numeric_match = re.search(r'\b([0-7])\b', grade)
    if numeric_match:
        result = numeric_match.group(1)
        logger.debug(f"Normalized grade '{original_grade}' -> '{result}' (numeric match)")
        return result
    
    # Check if grade contains keywords (order matters - check partial first)
    if "partial" in grade_lower:
        logger.debug(f"Normalized grade '{original_grade}' -> 'Partial' (keyword match)")
        return "Partial"
    if any(word in grade_lower for word in ["incomplete", "unfinished", "missing"]):
        logger.debug(f"Normalized grade '{original_grade}' -> 'Partial' (keyword match)")
        return "Partial"
    if any(word in grade_lower for word in ["correct", "right", "true", "full", "complete", "solved"]):
        logger.debug(f"Normalized grade '{original_grade}' -> 'Correct' (keyword match)")
        return "Correct"
    if any(word in grade_lower for word in ["incorrect", "wrong", "false", "none", "zero", "error", "invalid"]):
        logger.debug(f"Normalized grade '{original_grade}' -> 'Incorrect' (keyword match)")
        return "Incorrect"
    
    # Default: capitalize first letter
    result = grade.capitalize()
    logger.debug(f"Normalized grade '{original_grade}' -> '{result}' (default capitalization)")
    return result


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.max_retries = 3  # Add retry mechanism for robustness

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

        # Retry loop for robustness
        prediction = "None"
        msg_history = []
        
        for attempt in range(self.max_retries):
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=[],
                )
                
                # Extract prediction from JSON with enhanced error handling
                last_assistant_msg = None
                for msg in reversed(msg_history):
                    if msg.get("role") == "assistant" or "text" in msg:
                        last_assistant_msg = msg.get("text", msg.get("content", ""))
                        break
                
                if not last_assistant_msg:
                    self.log_fn(f"Warning: No assistant message found in history (attempt {attempt + 1})")
                    if attempt < self.max_retries - 1:
                        continue
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
                    
                    self.log_fn(f"Extracted grade: {prediction} (attempt {attempt + 1})")
                    # Success - break out of retry loop
                    break
                else:
                    self.log_fn(f"Warning: No JSON blocks found in response (attempt {attempt + 1})")
                    # Try to extract grade directly from text as last resort
                    text_lower = last_assistant_msg.lower()
                    if "grade:" in text_lower or "score:" in text_lower:
                        match = re.search(r'(?:grade|score):\s*(.+?)(?:\n|$)', text_lower, re.IGNORECASE)
                        if match:
                            prediction = _normalize_grade(match.group(1).strip())
                            self.log_fn(f"Extracted grade from text: {prediction} (attempt {attempt + 1})")
                            break
                    
                    # If we have retries left, try again
                    if attempt < self.max_retries - 1:
                        self.log_fn(f"Retrying... ({attempt + 2}/{self.max_retries})")
                        continue
                    
            except Exception as e:
                self.log_fn(f"Error on attempt {attempt + 1}: {e}")
                if attempt < self.max_retries - 1:
                    self.log_fn(f"Retrying... ({attempt + 2}/{self.max_retries})")
                    continue
                import traceback
                self.log_fn(f"Traceback: {traceback.format_exc()}")

        return str(prediction), msg_history
