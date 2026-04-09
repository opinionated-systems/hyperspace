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
    Includes enhanced error reporting for debugging extraction failures.
    """
    results = []
    errors = []
    search_from = 0
    
    # Primary: Extract from <json>...</json> tags
    while True:
        start = text.find("<json>", search_from)
        if start == -1:
            break
        end = text.find("</json>", start)
        if end == -1:
            errors.append(f"Unclosed <json> tag at position {start}")
            break
        inner = text[start + 6:end].strip()
        search_from = end + 7
        try:
            parsed = json.loads(inner)
            results.append(parsed)
        except json.JSONDecodeError as e:
            errors.append(f"JSON parse error in <json> block: {e}")
            continue
    
    # Fallback 1: Extract from markdown code blocks
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
                errors.append(f"Unclosed code block at position {start}")
                break
            inner = text[start:end].strip()
            search_from = end + 3
            try:
                parsed = json.loads(inner)
                results.append(parsed)
            except json.JSONDecodeError as e:
                errors.append(f"JSON parse error in code block: {e}")
                continue
    
    # Fallback 2: Extract JSON objects directly from text using brace matching
    if not results:
        brace_start = text.find("{")
        attempt_count = 0
        max_attempts = 10  # Limit to prevent excessive processing
        
        while brace_start != -1 and attempt_count < max_attempts:
            attempt_count += 1
            # Try to find matching closing brace using stack counting
            brace_count = 0
            in_string = False
            escape_next = False
            
            for i, char in enumerate(text[brace_start:]):
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
                            json_str = text[brace_start:brace_start + i + 1]
                            try:
                                parsed = json.loads(json_str)
                                # Only accept if it has expected fields
                                if any(key in parsed for key in ["response", "grade", "score", "analysis", "evaluation", "result"]):
                                    results.append(parsed)
                            except json.JSONDecodeError as e:
                                errors.append(f"JSON parse error in raw extraction: {e}")
                            break
            brace_start = text.find("{", brace_start + 1)
    
    # Log extraction details for debugging
    if errors and logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"JSON extraction encountered {len(errors)} errors: {errors[:3]}")
    
    return results or None


def _normalize_grade(grade: str) -> str:
    """Normalize grade to standard format.
    
    Converts various grade formats to a consistent representation.
    Handles both string grades (Correct/Partial/Incorrect) and numeric scores (0-7).
    Includes enhanced handling for edge cases, ambiguous responses, numeric ranges,
    and fractional grades.
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
        "perfect": "Correct",
        "excellent": "Correct",
        "good": "Correct",
        "satisfactory": "Correct",
        "partial": "Partial",
        "partially correct": "Partial",
        "partial credit": "Partial",
        "partially": "Partial",
        "some credit": "Partial",
        "incomplete": "Partial",
        "mostly correct": "Partial",
        "minor errors": "Partial",
        "needs work": "Partial",
        "fair": "Partial",
        "average": "Partial",
        "incorrect": "Incorrect",
        "wrong": "Incorrect",
        "false": "Incorrect",
        "no": "Incorrect",
        "none": "Incorrect",
        "zero": "Incorrect",
        "invalid": "Incorrect",
        "unacceptable": "Incorrect",
        "failed": "Incorrect",
        "fail": "Incorrect",
        "poor": "Incorrect",
        "unsatisfactory": "Incorrect",
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
    
    # Try to extract numeric grade from patterns like "Grade: 5" or "Score: 3/7"
    # Look for patterns like "X/7" or "X out of 7"
    fraction_match = re.search(r'(\d+)\s*/\s*7', grade)
    if fraction_match:
        return fraction_match.group(1)
    
    out_of_match = re.search(r'(\d+)\s+out\s+of\s+7', grade_lower)
    if out_of_match:
        return out_of_match.group(1)
    
    # Look for standalone numbers 0-7 (word boundaries)
    number_match = re.search(r'\b([0-7])\b', grade)
    if number_match:
        return number_match.group(1)
    
    # Check for numeric ranges and extract best estimate
    range_match = re.search(r'(\d+)\s*-\s*(\d+)', grade)
    if range_match:
        # Return the lower bound of the range as conservative estimate
        lower = range_match.group(1)
        if int(lower) <= 7:
            return lower
    
    # Handle fractional grades (e.g., "3.5", "4/7", "5.5 points")
    fractional_match = re.search(r'(\d+\.\d+)\s*(?:points?|pts?)?', grade_lower)
    if fractional_match:
        try:
            val = float(fractional_match.group(1))
            if 0 <= val <= 7:
                return str(int(val))  # Round down to be conservative
        except ValueError:
            pass
    
    # Check for phrases containing grade keywords with negation handling
    negation_words = ["not", "no", "never", "hardly", "barely", "scarcely"]
    has_negation = any(word in grade_lower for word in negation_words)
    
    if any(word in grade_lower for word in ["correct", "right", "solved", "complete", "accurate"]):
        if has_negation or any(word in grade_lower for word in ["in", "un", "partially", "mostly"]):
            return "Partial" if "partial" in grade_lower else "Incorrect"
        return "Correct"
    
    if any(word in grade_lower for word in ["partial", "incomplete", "some", "partly", "half"]):
        return "Partial"
    
    if any(word in grade_lower for word in ["wrong", "incorrect", "error", "invalid", "mistake", "flawed"]):
        return "Incorrect"
    
    # Handle empty/blank responses
    if any(word in grade_lower for word in ["empty", "blank", "missing", "absent", "none", "null"]):
        return "Incorrect"
    
    # Default: return original if we can't normalize
    logger.debug(f"Could not normalize grade '{grade}', returning as-is")
    return grade


def _calculate_confidence(extracted: dict, raw_response: str) -> float:
    """Calculate confidence score for the grading decision.
    
    Returns a score between 0.0 and 1.0 based on:
    - Presence of structured analysis fields
    - Length and detail of reasoning
    - Clear grade assignment in response
    """
    confidence = 0.5  # Base confidence
    
    # Boost confidence for structured fields
    if extracted.get("understanding") and len(extracted["understanding"]) > 20:
        confidence += 0.1
    if extracted.get("analysis") and len(extracted["analysis"]) > 50:
        confidence += 0.15
    if extracted.get("partial_credit_reasoning") and len(extracted["partial_credit_reasoning"]) > 30:
        confidence += 0.15
    
    # Check for clear grade in response field
    response = extracted.get("response", "")
    if response:
        normalized = _normalize_grade(response)
        if normalized in ["Correct", "Partial", "Incorrect", "0", "1", "2", "3", "4", "5", "6", "7"]:
            confidence += 0.2
    
    # Penalize if response seems uncertain (contains hedge words)
    uncertainty_markers = ["maybe", "perhaps", "possibly", "unclear", "ambiguous", "uncertain", "might be"]
    response_lower = str(response).lower()
    if any(marker in response_lower for marker in uncertainty_markers):
        confidence -= 0.15
    
    return max(0.0, min(1.0, confidence))


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
        confidence = 0.0
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
                
                # Priority order for grade extraction
                grade_fields = ["response", "grade", "score", "result", "final_grade", "evaluation"]
                for field in grade_fields:
                    if field in last_extract:
                        prediction = last_extract[field]
                        break
                
                # Normalize the grade for consistency
                prediction = _normalize_grade(prediction)
                
                # Calculate confidence score
                confidence = _calculate_confidence(last_extract, last_assistant_msg)
                
                # Log detailed analysis for debugging
                if "analysis" in last_extract:
                    self.log_fn(f"Analysis: {last_extract['analysis'][:200]}...")
                if "partial_credit_reasoning" in last_extract:
                    self.log_fn(f"Partial Credit: {last_extract['partial_credit_reasoning'][:200]}...")
                if "understanding" in last_extract:
                    self.log_fn(f"Understanding: {last_extract['understanding'][:200]}...")
                
                self.log_fn(f"Extracted grade: {prediction} (confidence: {confidence:.2f})")
            else:
                self.log_fn("Warning: No JSON blocks found in response")
                # Try to extract grade directly from text as last resort
                text_lower = last_assistant_msg.lower()
                if "grade:" in text_lower or "score:" in text_lower:
                    match = re.search(r'(?:grade|score):\s*(.+?)(?:\n|$)', text_lower, re.IGNORECASE)
                    if match:
                        prediction = _normalize_grade(match.group(1).strip())
                        confidence = 0.3  # Lower confidence for text extraction
                        self.log_fn(f"Extracted grade from text: {prediction} (confidence: {confidence:.2f})")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            import traceback
            self.log_fn(f"Traceback: {traceback.format_exc()}")

        # Store confidence in msg_history for downstream use
        if msg_history and isinstance(msg_history, list):
            msg_history.append({"role": "system", "text": f"confidence_score: {confidence:.2f}"})

        return str(prediction), msg_history
