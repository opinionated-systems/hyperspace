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
    
    Improved: Better handling of nested braces, common JSON errors, and edge cases.
    """
    results = []
    search_from = 0
    extraction_attempts = 0
    
    def _clean_and_parse_json(json_str: str) -> dict | None:
        """Try to parse JSON with various cleaning strategies."""
        json_str = json_str.strip()
        
        # Strategy 1: Direct parse
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
        
        # Strategy 2: Remove trailing commas before closing braces/brackets
        try:
            cleaned = re.sub(r',(\s*[}\]])', r'\1', json_str)
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
        
        # Strategy 3: Fix single quotes to double quotes (common LLM error)
        try:
            # Replace single quotes with double quotes, but be careful with apostrophes
            cleaned = re.sub(r"(?<!\\)'", '"', json_str)
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
        
        # Strategy 4: Remove comments (// and /* */)
        try:
            cleaned = re.sub(r'//.*?\n', '\n', json_str)
            cleaned = re.sub(r'/\*.*?\*/', '', cleaned, flags=re.DOTALL)
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
        
        # Strategy 5: Fix unescaped newlines in strings
        try:
            cleaned = re.sub(r'(?<=")\n(?=")', '\\n', json_str)
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
        
        return None
    
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
        
        parsed = _clean_and_parse_json(inner)
        if parsed:
            results.append(parsed)
            logger.debug(f"Successfully extracted JSON from <json> block #{extraction_attempts}")
        else:
            logger.debug(f"JSON decode error in <json> block #{extraction_attempts}: could not parse after cleaning")
    
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
            
            parsed = _clean_and_parse_json(inner)
            if parsed:
                results.append(parsed)
                logger.debug(f"Successfully extracted JSON from markdown block #{extraction_attempts}")
            else:
                logger.debug(f"JSON decode error in markdown block #{extraction_attempts}: could not parse")
    
    # Fallback 2: Find JSON objects directly in text using brace matching
    if not results:
        logger.debug("No JSON found in code blocks, trying direct JSON extraction")
        brace_start = text.find("{")
        attempt_count = 0
        while brace_start != -1:
            attempt_count += 1
            # Try to find matching closing brace using stack counting
            brace_count = 0
            in_string = False
            escape_next = False
            
            for i, char in enumerate(text[brace_start:]):
                if escape_next:
                    escape_next = False
                    continue
                if char == '\\' and in_string:
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
                            parsed = _clean_and_parse_json(json_str)
                            if parsed:
                                # Only accept if it has expected fields
                                if any(key in parsed for key in ["response", "grade", "score", "analysis", "understanding"]):
                                    results.append(parsed)
                                    logger.debug(f"Successfully extracted JSON from direct parsing (attempt {attempt_count})")
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
    
    Improved logic: prioritizes numeric scores when present, handles edge cases better.
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
    
    # PRIORITY 1: Check for explicit numeric scores first (most reliable)
    # Look for patterns like "Score: 5", "Grade: 3/7", "5 points", "score of 4"
    numeric_patterns = [
        r'(?:score|grade|points?)[:\s]+(\d)',
        r'(\d)\s*(?:points?|/\s*7)',
        r'(?:score|grade)\s+(?:of\s+)?(\d)',
        r'\b(\d)\s*/\s*7\b',
    ]
    for pattern in numeric_patterns:
        match = re.search(pattern, grade_lower)
        if match:
            result = match.group(1)
            if result in ["0", "1", "2", "3", "4", "5", "6", "7"]:
                logger.debug(f"Normalized grade '{original_grade}' -> '{result}' (numeric pattern match)")
                return result
    
    # PRIORITY 2: Check for standalone numeric digits
    numeric_match = re.search(r'\b([0-7])\b', grade)
    if numeric_match:
        result = numeric_match.group(1)
        logger.debug(f"Normalized grade '{original_grade}' -> '{result}' (numeric match)")
        return result
    
    # PRIORITY 3: Map common variations to standard grades
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
        "half correct": "Partial",
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
    
    # PRIORITY 4: Check if grade contains keywords (order matters - check partial first)
    # Use word boundaries to avoid false matches
    if re.search(r'\bpartial\w*\b', grade_lower):
        logger.debug(f"Normalized grade '{original_grade}' -> 'Partial' (keyword match)")
        return "Partial"
    if any(re.search(rf'\b{word}\w*\b', grade_lower) for word in ["incomplete", "unfinished", "missing"]):
        logger.debug(f"Normalized grade '{original_grade}' -> 'Partial' (keyword match)")
        return "Partial"
    if any(re.search(rf'\b{word}\w*\b', grade_lower) for word in ["correct", "right", "true", "full", "complete", "solved", "perfect"]):
        logger.debug(f"Normalized grade '{original_grade}' -> 'Correct' (keyword match)")
        return "Correct"
    if any(re.search(rf'\b{word}\w*\b', grade_lower) for word in ["incorrect", "wrong", "false", "none", "zero", "error", "invalid", "fail"]):
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
   - Verify any calculations or proofs they provided

3. **Partial Credit Assessment**: Based on the grading guidelines, determine:
   - Which parts of the solution they completed correctly
   - What partial credit they deserve for incomplete or partially correct work
   - Whether they demonstrated understanding of key concepts
   - Whether they made significant progress toward the solution
   - Consider if errors are minor (computation) vs major (conceptual)

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

## Important Grading Principles
- **Be fair but rigorous**: IMO grading requires high standards
- **Reward insight**: Give credit for correct key insights even if details are missing
- **Penalize major errors**: Conceptual misunderstandings should reduce score significantly
- **Consider partial progress**: Even incomplete solutions may deserve partial credit
- **Check for plagiarism**: Ensure the student isn't just restating the problem

## Response Format (CRITICAL)
You MUST respond in valid JSON format with the following schema:
<json>
{{
    "understanding": "Brief summary of the problem and correct approach",
    "analysis": "Detailed step-by-step analysis of the student's answer, including what they got right and wrong",
    "partial_credit_reasoning": "Explanation of partial credit based on grading guidelines and rubric",
    "response": "Your final grade/score (use EXACTLY one of: 'Correct', 'Partial', 'Incorrect', or numeric 0-7)"
}}
</json>

IMPORTANT: The "response" field MUST contain ONLY one of these exact values:
- "Correct" for a complete, correct solution
- "Partial" for incomplete or partially correct work
- "Incorrect" for fundamentally wrong answers
- Or a numeric string "0" through "7" for specific point values"""

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
                
                # Priority order for grade extraction (most specific to least specific)
                grade_fields = ["response", "grade", "score", "result", "final_grade", "evaluation", "answer"]
                for field in grade_fields:
                    if field in last_extract:
                        prediction = last_extract[field]
                        self.log_fn(f"Found grade in field '{field}': {prediction}")
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
                
                self.log_fn(f"Final extracted grade: {prediction}")
            else:
                self.log_fn("Warning: No JSON blocks found in response, attempting text extraction")
                # Try to extract grade directly from text as last resort
                text_lower = last_assistant_msg.lower()
                
                # Look for various grade patterns in the text
                grade_patterns = [
                    r'(?:grade|score|result)[:\s]+([0-7]|correct|partial|incorrect)',
                    r'(?:final grade|final score)[:\s]+([0-7]|correct|partial|incorrect)',
                    r'(?:the grade is|the score is)[:\s]+([0-7]|correct|partial|incorrect)',
                    r'\b([0-7])\s*(?:points?|/\s*7)\b',
                ]
                
                for pattern in grade_patterns:
                    match = re.search(pattern, text_lower, re.IGNORECASE)
                    if match:
                        prediction = _normalize_grade(match.group(1).strip())
                        self.log_fn(f"Extracted grade from text pattern: {prediction}")
                        break
                else:
                    # Check for standalone keywords
                    if re.search(r'\bcorrect\b', text_lower) and not re.search(r'\bpartial\b', text_lower):
                        prediction = "Correct"
                        self.log_fn("Extracted grade 'Correct' from keyword match")
                    elif re.search(r'\bpartial\b', text_lower):
                        prediction = "Partial"
                        self.log_fn("Extracted grade 'Partial' from keyword match")
                    elif re.search(r'\bincorrect\b', text_lower):
                        prediction = "Incorrect"
                        self.log_fn("Extracted grade 'Incorrect' from keyword match")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            import traceback
            self.log_fn(f"Traceback: {traceback.format_exc()}")

        return str(prediction), msg_history
