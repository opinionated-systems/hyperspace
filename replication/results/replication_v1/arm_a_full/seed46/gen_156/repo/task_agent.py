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
    """
    if not text:
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
                # Find first { and last }
                json_start = inner.find("{")
                json_end = inner.rfind("}")
                if json_start != -1 and json_end != -1 and json_end > json_start:
                    results.append(json.loads(inner[json_start:json_end + 1]))
            except json.JSONDecodeError:
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
                            if any(key in parsed for key in ["response", "grade", "score", "analysis", "understanding", "partial_credit_reasoning"]):
                                results.append(parsed)
                        except json.JSONDecodeError:
                            pass
                        break
            brace_start = text.find("{", brace_start + 1)
    
    # Additional fallback: try to extract from common LLM output patterns
    if not results:
        # Try to find JSON after common prefixes like "Here is the JSON:" or "Response:"
        patterns = [
            r'\{[\s\S]*?"response"[\s\S]*?\}',
            r'\{[\s\S]*?"grade"[\s\S]*?\}',
            r'\{[\s\S]*?"score"[\s\S]*?\}',
            r'\{[\s\S]*?"understanding"[\s\S]*?\}',
            r'\{[\s\S]*?"analysis"[\s\S]*?\}',
        ]
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    parsed = json.loads(match.group())
                    if any(key in parsed for key in ["response", "grade", "score", "analysis", "understanding", "partial_credit_reasoning"]):
                        results.append(parsed)
                        break  # Only take first valid match per pattern
                except json.JSONDecodeError:
                    continue
            if results:
                break
    
    # Final fallback: try to find any JSON-like structure with expected fields
    if not results:
        # Look for the largest JSON object in the text
        best_match = None
        best_score = 0
        
        # Find all potential JSON start positions
        for start in range(len(text)):
            if text[start] == '{':
                # Try to find matching end brace
                brace_count = 0
                for i, char in enumerate(text[start:]):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            json_str = text[start:start + i + 1]
                            try:
                                parsed = json.loads(json_str)
                                # Score based on presence of expected fields
                                score = sum(1 for key in ["response", "grade", "score", "analysis", "understanding", "partial_credit_reasoning"] if key in parsed)
                                if score > best_score:
                                    best_score = score
                                    best_match = parsed
                            except json.JSONDecodeError:
                                pass
                            break
        
        if best_match and best_score > 0:
            results.append(best_match)
    
    return results or None


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
            # Log the response/grade field if present
            for key in ["response", "grade", "score", "result"]:
                if key in obj:
                    logger_fn(f"    {key}={obj[key]}")
                    break
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
        if text:
            snippet = text[:300].replace("\n", " ")
            logger_fn(f"  - Text preview: {snippet}...")


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
        "almost": "Almost",
        "almost correct": "Almost",
        "nearly correct": "Almost",
        "close": "Almost",
        "minor error": "Almost",
        "small mistake": "Almost",
        "mostly correct": "Almost",
        "essentially correct": "Almost",
        "partial": "Partial",
        "partially correct": "Partial",
        "partial credit": "Partial",
        "partially": "Partial",
        "some credit": "Partial",
        "incomplete": "Partial",
        "some progress": "Partial",
        "limited progress": "Partial",
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
    
    # Try to extract numeric score from strings like "Score: 5" or "Grade: 3" or "5/7"
    # First try to match standalone digits 0-7
    numeric_match = re.search(r'\b([0-7])\b', grade)
    if numeric_match:
        return numeric_match.group(1)
    
    # Try to match fractions like "5/7" or "3 / 7"
    fraction_match = re.search(r'\b([0-7])\s*/\s*7\b', grade)
    if fraction_match:
        return fraction_match.group(1)
    
    # Try to match "X out of 7" or "X points"
    points_match = re.search(r'\b([0-7])\s*(?:out of|points?|pts?)\s*7?\b', grade_lower)
    if points_match:
        return points_match.group(1)
    
    # Check if grade contains keywords (order matters - check specific categories first)
    # Check for "almost" category first (more specific than partial)
    # Expanded list to catch more variations
    almost_keywords = [
        "almost", "nearly", "close", "minor error", "small mistake", 
        "mostly correct", "essentially correct", "nearly correct",
        "minor issue", "small error", "trivial error", "slight error",
        "almost there", "very close", "just missed", "minor gap",
        "small gap", "nearly solved", "essentially solved", "mostly right"
    ]
    if any(word in grade_lower for word in almost_keywords):
        return "Almost"
    
    # Check for partial credit indicators
    partial_keywords = [
        "partial", "incomplete", "unfinished", "missing", "some progress", 
        "limited progress", "partially correct", "partial credit", "some credit",
        "partially", "in progress", "started", "attempted", "some work",
        "partial solution", "incomplete solution", "some correct", "partially right"
    ]
    if any(word in grade_lower for word in partial_keywords):
        return "Partial"
    
    # Check for correct indicators (be careful not to match "almost correct" etc.)
    correct_keywords = [
        "correct", "right", "true", "full", "complete", "solved", 
        "perfect", "excellent", "valid", "accurate", "proper", "sound"
    ]
    # Only match if not preceded by modifiers
    for word in correct_keywords:
        if word in grade_lower and not any(mod + " " + word in grade_lower for mod in ["almost", "nearly", "mostly", "partially", "not"]):
            return "Correct"
    
    # Check for incorrect indicators
    incorrect_keywords = [
        "incorrect", "wrong", "false", "none", "zero", "error", 
        "invalid", "fail", "failed", "unsolved", "no solution",
        "completely wrong", "totally wrong", "fundamentally wrong",
        "does not work", "not valid", "not correct", "not right"
    ]
    if any(word in grade_lower for word in incorrect_keywords):
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
   - **Correct (7)**: Complete solution with valid reasoning throughout
   - **Almost (5-6)**: Essentially correct solution with only minor gaps, unclear reasoning, or trivial errors. The student demonstrated full understanding but made small mistakes.
   - **Partial (1-4)**: Significant progress with correct approach but incomplete, OR correct initial steps but major gaps remain. The student showed understanding of key concepts but didn't complete the solution.
   - **Incorrect (0)**: Fundamentally wrong approach, no meaningful progress, or empty answer.

## Critical Distinctions
- **Almost vs Partial**: "Almost" means the solution is essentially complete with minor issues. "Partial" means significant work is missing or there are major errors.
  - Example of "Almost": Correct approach, valid reasoning, but one small calculation error or unclear explanation at the end.
  - Example of "Partial": Correct approach started but stopped halfway, OR correct method but major logical gaps.

- **Partial vs Incorrect**: "Partial" requires demonstrating understanding of the correct approach. "Incorrect" means the approach itself is wrong.
  - Example of "Partial": Student sets up the right equations but can't solve them.
  - Example of "Incorrect": Student uses completely wrong method or makes nonsense claims.

## Grading Rubric Reference
- **7 points (Correct)**: Complete, correct solution with clear reasoning
- **5-6 points (Almost)**: Minor gaps or unclear reasoning, but essentially correct approach. Student understood the problem and nearly solved it.
- **3-4 points (Partial)**: Significant progress, correct approach but incomplete or with major errors. Student showed understanding but didn't finish.
- **1-2 points (Partial)**: Some relevant work or correct initial steps, but limited progress. Student started correctly but got stuck early.
- **0 points (Incorrect)**: No meaningful progress or completely wrong approach. Student didn't understand the problem.

Respond in JSON format with the following schema:
<json>
{{
    "understanding": "Brief summary of the problem and correct approach",
    "analysis": "Detailed step-by-step analysis of the student's answer, including what they got right and wrong",
    "partial_credit_reasoning": "Explanation of partial credit based on grading guidelines and rubric",
    "response": "Your final grade/score (use: 'Correct', 'Almost', 'Partial', 'Incorrect', or numeric 0-7)"
}}
</json>"""

        # Try LLM call with retry logic for robustness
        max_retries = 2
        prediction = "None"
        msg_history = []
        
        for attempt in range(max_retries + 1):
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=[],
                )
                
                # Extract prediction from JSON with enhanced error handling
                prediction = self._extract_prediction(msg_history)
                
                # If we got a valid prediction, break out of retry loop
                if prediction != "None":
                    break
                    
                # If no valid prediction and we have retries left, try again
                if attempt < max_retries:
                    self.log_fn(f"Retry {attempt + 1}/{max_retries}: No valid prediction extracted, retrying...")
                    
            except Exception as e:
                self.log_fn(f"Error in LLM call (attempt {attempt + 1}): {e}")
                if attempt < max_retries:
                    self.log_fn(f"Retrying...")
                else:
                    self.log_fn("Max retries reached, returning None")

        return str(prediction), msg_history
    
    def _extract_prediction(self, msg_history: list[dict]) -> str:
        """Extract prediction from message history.
        
        Args:
            msg_history: List of message dictionaries
            
        Returns:
            Extracted and normalized grade/prediction
        """
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
                return prediction
            
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
                text_lower = last_assistant_msg.lower()
                
                # Try multiple patterns for grade extraction from text
                patterns = [
                    r'(?:grade|score|final grade|result|evaluation):\s*(.+?)(?:\n|$|\.\s)',
                    r'(?:the grade is|grade is|score is|result is):?\s*(.+?)(?:\n|$|\.\s)',
                    r'(?:i would grade this|my grade|final answer):?\s*(.+?)(?:\n|$|\.\s)',
                ]
                
                for pattern in patterns:
                    match = re.search(pattern, text_lower, re.IGNORECASE)
                    if match:
                        extracted_text = match.group(1).strip()
                        # Clean up common suffixes
                        extracted_text = re.sub(r'\s*(?:points?|pts?|/\s*7)\s*$', '', extracted_text, flags=re.IGNORECASE)
                        prediction = _normalize_grade(extracted_text)
                        self.log_fn(f"Extracted grade from text using pattern: {prediction}")
                        break
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            import traceback
            self.log_fn(f"Traceback: {traceback.format_exc()}")
        
        return prediction
