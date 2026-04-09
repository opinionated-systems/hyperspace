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
    
    Enhanced to handle nested JSON and various formatting edge cases.
    """
    results = []
    search_from = 0
    while True:
        start = text.find("<json>", search_from)
        if start == -1:
            break
        end = text.find("</json>", start)
        if end == -1:
            break
        inner = text[start + 6:end].strip()
        search_from = end + 7
        
        # Try to parse the JSON
        try:
            results.append(json.loads(inner))
        except json.JSONDecodeError:
            # Try to fix common JSON formatting issues
            # 1. Remove trailing commas before closing braces/brackets
            fixed = re.sub(r',(\s*[}\]])', r'\1', inner)
            # 2. Fix single quotes to double quotes (carefully)
            fixed = re.sub(r"(?<!\\)'", '"', fixed)
            # 3. Fix unescaped newlines in strings
            fixed = re.sub(r'(?<!\\)\n', '\\n', fixed)
            
            try:
                results.append(json.loads(fixed))
            except json.JSONDecodeError:
                # If still failing, try to extract just the response field
                response_match = re.search(r'"response"\s*:\s*"([^"]*)"', inner)
                reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]*)"', inner, re.DOTALL)
                if response_match:
                    partial = {"response": response_match.group(1)}
                    if reasoning_match:
                        partial["reasoning"] = reasoning_match.group(1)
                    results.append(partial)
                continue
    return results or None


def _extract_json_from_markdown(text: str) -> dict | None:
    """Extract JSON from markdown code blocks (```json ... ```).
    
    Enhanced to handle various code block formats and JSON edge cases.
    """
    # Try different code block patterns
    patterns = [
        r'```json\s*(.*?)\s*```',  # Standard json blocks
        r'```\s*(.*?)\s*```',       # Generic code blocks
        r'\{[^{}]*"response"[^{}]*\}',  # Raw JSON objects
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                return json.loads(match.strip())
            except json.JSONDecodeError:
                # Try fixing common issues
                fixed = re.sub(r',(\s*[}\]])', r'\1', match)
                fixed = re.sub(r"(?<!\\)'", '"', fixed)
                try:
                    return json.loads(fixed.strip())
                except json.JSONDecodeError:
                    # Try extracting just the response field
                    response_match = re.search(r'"response"\s*:\s*"([^"]*)"', match)
                    if response_match:
                        return {"response": response_match.group(1)}
                    continue
    return None


def _validate_grade(prediction: str, grading_guidelines: str) -> tuple[str, bool]:
    """Validate that the extracted grade is reasonable.
    
    Enhanced validation with support for various grade formats including
    IMO-style (0-7), percentage-based, and descriptive grades.
    
    Returns:
        (validated_grade, is_valid)
    """
    if not prediction or prediction == "None":
        return "None", False
    
    pred_clean = prediction.strip()
    pred_lower = pred_clean.lower()
    
    # Direct numeric match (0-7 for IMO problems)
    if pred_clean.isdigit():
        grade_val = int(pred_clean)
        if 0 <= grade_val <= 7:
            return pred_clean, True
    
    # Check for numeric grades with word boundaries
    numeric_match = re.search(r'\b([0-7])\b', pred_clean)
    if numeric_match:
        return numeric_match.group(1), True
    
    # Check for "X out of 7" or "X/7" patterns
    out_of_match = re.search(r'([0-7])\s*(?:out\s+of|/)\s*7', pred_lower)
    if out_of_match:
        return out_of_match.group(1), True
    
    # Check for partial credit patterns with more variations
    partial_patterns = [
        r'partial\s*(?:credit)?\s*:?\s*([0-7])',
        r'partial\s*:?\s*([0-7])',
        r'partially\s+(?:correct|right)\s*:?\s*([0-7])?',
    ]
    for pattern in partial_patterns:
        partial_match = re.search(pattern, pred_lower)
        if partial_match:
            if partial_match.group(1):
                return f"Partial credit: {partial_match.group(1)}", True
            return "Partial credit", True
    
    # Check for full credit patterns
    full_patterns = ['full credit', 'full marks', 'complete', 'perfect', '7/7', '7 out of 7']
    for pattern in full_patterns:
        if pattern in pred_lower:
            return "7", True
    
    # Check for zero/no credit patterns
    zero_patterns = ['no credit', 'zero', '0/7', '0 out of 7', 'none', 'incorrect', 'wrong']
    for pattern in zero_patterns:
        if pattern in pred_lower:
            return "0", True
    
    # Check for "correct" (implies full marks)
    if pred_lower == 'correct' or pred_lower == 'right':
        return "7", True
    
    # Check for N/A or not applicable
    if any(x in pred_lower for x in ['n/a', 'not applicable', 'invalid']):
        return "N/A", True
    
    # If prediction is too long, it might be reasoning instead of a grade
    if len(pred_clean) > 50:
        # Try to extract a grade from within the text
        # Look for patterns like "Grade: X" or "Final grade: X"
        grade_in_text = re.search(r'(?:grade|score|mark|points?)\s*:?\s*([0-7])\b', pred_lower)
        if grade_in_text:
            return grade_in_text.group(1), True
    
    # If no clear grade found, mark as invalid
    return prediction, False


class TaskAgent:
    """Task agent that solves IMO grading problems with enhanced reasoning and validation."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def _build_prompt(self, inputs: dict) -> str:
        """Build a structured prompt for IMO grading with chain-of-thought reasoning."""
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        # Truncate very long inputs to prevent token overflow
        max_len = 8000
        problem = problem[:max_len] if len(problem) > max_len else problem
        solution = solution[:max_len] if len(solution) > max_len else solution
        student_answer = student_answer[:max_len] if len(student_answer) > max_len else student_answer
        grading_guidelines = grading_guidelines[:max_len] if len(grading_guidelines) > max_len else grading_guidelines

        return f"""You are an expert IMO (International Mathematical Olympiad) grader.

Your task is to evaluate a student's solution to a mathematical problem.

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

1. First, analyze the student's answer step by step. Compare it against the official solution.
2. Identify any errors, missing steps, or creative alternative approaches.
3. Consider the grading guidelines carefully - IMO problems are typically graded 0-7 points.
4. Provide your reasoning before giving the final grade.
5. The final grade MUST be a single numeric value from 0 to 7 (inclusive).

## Response Format (IMPORTANT)

You MUST respond using EXACTLY this JSON format wrapped in <json> tags:

<json>
{{
    "reasoning": "Your detailed analysis of the student's answer...",
    "response": "X"
}}
</json>

Where X is a single digit from 0 to 7 representing the grade.

Examples of valid responses:
- "response": "7" (full marks)
- "response": "0" (no credit)
- "response": "3" (partial credit)

Do NOT include any text outside the <json> tags. The "response" field must contain ONLY the numeric grade."""

    def _extract_prediction(self, msg_history: list[dict]) -> tuple[str, str]:
        """Extract prediction and reasoning from message history.
        
        Enhanced extraction with multiple fallback strategies and better
        handling of various response formats.
        
        Returns:
            (prediction, reasoning)
        """
        prediction = "None"
        reasoning = ""
        
        try:
            if not msg_history:
                return prediction, reasoning
                
            last_msg = msg_history[-1].get("text", "") if isinstance(msg_history[-1], dict) else ""
            
            # Strategy 1: Try <json> tags first (most reliable)
            extracted = _extract_jsons(last_msg)
            if extracted:
                last_json = extracted[-1]
                if "response" in last_json:
                    prediction = str(last_json["response"]).strip()
                if "reasoning" in last_json:
                    reasoning = str(last_json["reasoning"])
                return prediction, reasoning
            
            # Strategy 2: Try markdown code blocks
            md_json = _extract_json_from_markdown(last_msg)
            if md_json:
                if "response" in md_json:
                    prediction = str(md_json["response"]).strip()
                if "reasoning" in md_json:
                    reasoning = str(md_json["reasoning"])
                return prediction, reasoning
            
            # Strategy 3: Try to find any JSON-like structure with response field
            # Use a more robust pattern that handles nested braces
            json_pattern = r'\{[^{}]*"response"[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
            json_matches = re.findall(json_pattern, last_msg, re.DOTALL)
            for json_str in json_matches:
                try:
                    fallback = json.loads(json_str)
                    if "response" in fallback:
                        prediction = str(fallback["response"]).strip()
                        if "reasoning" in fallback:
                            reasoning = str(fallback["reasoning"])
                        return prediction, reasoning
                except json.JSONDecodeError:
                    continue
            
            # Strategy 4: Look for explicit grade/score patterns
            grade_patterns = [
                r'(?:final\s+)?(?:grade|score|mark|points?)\s*:?\s*["\']?([0-7]|partial|full|incorrect|correct|zero)["\']?',
                r'(?:^|\n)\s*([0-7])\s*(?:/\s*7)?\s*(?:$|\n)',
                r'(?:grade|score)\s+is\s*:?\s*["\']?([0-7])["\']?',
            ]
            for pattern in grade_patterns:
                grade_match = re.search(pattern, last_msg, re.IGNORECASE)
                if grade_match:
                    prediction = grade_match.group(1)
                    break
            
            # Strategy 5: Extract reasoning from text if no JSON found
            if not reasoning:
                # Look for reasoning section patterns
                reasoning_patterns = [
                    r'(?:reasoning|analysis|explanation)[:\s]+(.+?)(?=\n\s*(?:grade|score|final)|$)',
                    r'(?:step\s+by\s+step|analysis)[:\s]+(.+?)(?=\n\s*(?:therefore|thus|in\s+conclusion)|$)',
                ]
                for pattern in reasoning_patterns:
                    reasoning_match = re.search(pattern, last_msg, re.IGNORECASE | re.DOTALL)
                    if reasoning_match:
                        reasoning = reasoning_match.group(1).strip()[:500]  # Limit length
                        break
                    
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
        
        return prediction, reasoning

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with enhanced reasoning and validation.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        instruction = self._build_prompt(inputs)
        grading_guidelines = inputs.get("grading_guidelines", "")
        
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=[],
                )
                break
            except Exception as e:
                self.log_fn(f"LLM call failed (attempt {attempt + 1}): {e}")
                if attempt == max_retries:
                    return "Error: LLM call failed", []
                import time
                time.sleep(2 ** attempt)  # Exponential backoff

        # Extract prediction with enhanced extraction
        prediction, reasoning = self._extract_prediction(msg_history)
        
        # Validate the grade
        validated_grade, is_valid = _validate_grade(prediction, grading_guidelines)
        
        # Log the reasoning and validation result
        if reasoning:
            self.log_fn(f"Reasoning: {reasoning[:200]}...")
        self.log_fn(f"Extracted grade: {prediction}, Validated: {validated_grade}, Is valid: {is_valid}")
        
        # If grade is invalid, try multiple fallback strategies
        if not is_valid:
            # Fallback 1: Try to extract from the full response text
            if response:
                # Try to find any numeric grade in the response
                numeric_match = re.search(r'\b([0-7])\b', response)
                if numeric_match:
                    validated_grade = numeric_match.group(1)
                    is_valid = True
                    self.log_fn(f"Fallback 1 found grade: {validated_grade}")
            
            # Fallback 2: Check all messages in history for valid grades
            if not is_valid:
                for msg in reversed(msg_history):
                    if isinstance(msg, dict) and "text" in msg:
                        text = msg["text"]
                        # Try JSON extraction
                        extracted = _extract_jsons(text)
                        if extracted:
                            for json_obj in extracted:
                                if "response" in json_obj:
                                    pred = str(json_obj["response"]).strip()
                                    val, valid = _validate_grade(pred, grading_guidelines)
                                    if valid:
                                        validated_grade = val
                                        is_valid = True
                                        self.log_fn(f"Fallback 2 found grade in history: {validated_grade}")
                                        break
                        if is_valid:
                            break
            
            # Fallback 3: Use heuristics based on grading guidelines
            if not is_valid and grading_guidelines:
                # If guidelines mention specific point values, try to match
                point_matches = re.findall(r'(\d+)\s*points?', grading_guidelines.lower())
                if point_matches:
                    # Default to middle of range if we can't determine
                    self.log_fn(f"Fallback 3: Using heuristic based on guidelines")

        return str(validated_grade), msg_history
