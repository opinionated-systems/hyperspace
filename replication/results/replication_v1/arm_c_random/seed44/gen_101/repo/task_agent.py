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
from typing import Any

from agent.llm_client import get_response_from_llm, EVAL_MODEL

logger = logging.getLogger(__name__)


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
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
        try:
            results.append(json.loads(inner))
        except json.JSONDecodeError:
            # Try to fix common JSON issues before giving up
            try:
                fixed = inner
                # Remove trailing commas before closing braces/brackets
                fixed = re.sub(r',(\s*[}\]])', r'\1', fixed)
                # Fix single quotes to double quotes (common LLM mistake)
                fixed = re.sub(r"(?<!\\)'", '"', fixed)
                # Fix unescaped newlines in strings
                fixed = re.sub(r'(?<=")\n(?=")', '\\n', fixed)
                # Fix unescaped tabs in strings
                fixed = re.sub(r'(?<=")\t(?=")', '\\t', fixed)
                # Fix unescaped carriage returns
                fixed = re.sub(r'(?<=")\r(?=")', '\\r', fixed)
                # Fix unescaped backslashes (but not already escaped ones)
                fixed = re.sub(r'(?<!\\)\\(?!\\|"|n|r|t)', '\\\\', fixed)
                # Remove comments
                fixed = re.sub(r'//.*?\n', '\n', fixed)
                fixed = re.sub(r'/\*.*?\*/', '', fixed, flags=re.DOTALL)
                results.append(json.loads(fixed))
            except json.JSONDecodeError:
                continue
    return results or None


def _extract_json_with_retry(text: str, max_retries: int = 3) -> list[dict] | None:
    """Extract JSON with multiple fallback strategies.
    
    Tries multiple extraction methods:
    1. Standard <json>...</json> blocks
    2. JSON code blocks ```json...```
    3. Raw JSON objects in text (with nested brace handling)
    4. Relaxed JSON parsing for malformed responses
    """
    # Try standard extraction first
    results = _extract_jsons(text)
    if results:
        return results
    
    # Try JSON code blocks
    json_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    matches = re.findall(json_block_pattern, text, re.DOTALL)
    for match in matches:
        try:
            results.append(json.loads(match.strip()))
        except json.JSONDecodeError:
            # Try fixing common issues
            try:
                fixed = match.strip()
                fixed = re.sub(r',(\s*[}\]])', r'\1', fixed)
                fixed = re.sub(r"(?<!\\)'", '"', fixed)
                fixed = re.sub(r'(?<=")\n(?=")', '\\n', fixed)
                fixed = re.sub(r'(?<=")\t(?=")', '\\t', fixed)
                fixed = re.sub(r'(?<=")\r(?=")', '\\r', fixed)
                fixed = re.sub(r'(?<!\\)\\(?!\\|"|n|r|t)', '\\\\', fixed)
                results.append(json.loads(fixed))
            except json.JSONDecodeError:
                continue
    
    if results:
        return results
    
    # Try to find raw JSON objects with balanced brace matching
    # This handles nested objects better than simple regex
    def find_json_objects(s: str) -> list[str]:
        """Find JSON objects using brace counting."""
        objects = []
        i = 0
        while i < len(s):
            if s[i] == '{':
                start = i
                count = 1
                i += 1
                in_string = False
                escape_next = False
                while i < len(s) and count > 0:
                    if escape_next:
                        escape_next = False
                    elif s[i] == '\\':
                        escape_next = True
                    elif s[i] == '"' and not escape_next:
                        in_string = not in_string
                    elif not in_string:
                        if s[i] == '{':
                            count += 1
                        elif s[i] == '}':
                            count -= 1
                    i += 1
                if count == 0:
                    objects.append(s[start:i])
            else:
                i += 1
        return objects
    
    for obj_str in find_json_objects(text):
        try:
            results.append(json.loads(obj_str))
        except json.JSONDecodeError:
            # Try aggressive fixing
            try:
                fixed = obj_str
                # Remove comments
                fixed = re.sub(r'//.*?\n', '\n', fixed)
                fixed = re.sub(r'/\*.*?\*/', '', fixed, flags=re.DOTALL)
                # Fix trailing commas
                fixed = re.sub(r',(\s*[}\]])', r'\1', fixed)
                # Fix single quotes
                fixed = re.sub(r"(?<!\\)'", '"', fixed)
                # Fix unescaped characters
                fixed = re.sub(r'(?<=")\n(?=")', '\\n', fixed)
                fixed = re.sub(r'(?<=")\t(?=")', '\\t', fixed)
                fixed = re.sub(r'(?<=")\r(?=")', '\\r', fixed)
                fixed = re.sub(r'(?<!\\)\\(?!\\|"|n|r|t)', '\\\\', fixed)
                results.append(json.loads(fixed))
            except json.JSONDecodeError:
                continue
    
    return results or None


class TaskAgent:
    """Task agent that solves IMO grading problems with enhanced reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.max_retries = 3

    def _validate_inputs(self, inputs: dict) -> tuple[bool, str]:
        """Validate that all required inputs are present and non-empty.
        
        Returns:
            (is_valid, error_message)
        """
        required_fields = ["domain", "problem", "solution", "grading_guidelines", "student_answer"]
        
        for field in required_fields:
            if field not in inputs:
                return False, f"Missing required field: {field}"
            if not inputs[field] or not str(inputs[field]).strip():
                return False, f"Empty required field: {field}"
        
        return True, ""

    def _build_grading_prompt(self, inputs: dict, is_retry: bool = False) -> str:
        """Build a comprehensive prompt for the grading task."""
        domain = inputs.get("domain", "")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        # Truncate very long inputs to prevent token overflow
        max_len = 8000
        problem = problem[:max_len] + "..." if len(problem) > max_len else problem
        solution = solution[:max_len] + "..." if len(solution) > max_len else solution
        student_answer = student_answer[:max_len] + "..." if len(student_answer) > max_len else student_answer
        
        if is_retry:
            return '''Your previous response was not in the correct format or I could not extract a valid grade. Please respond ONLY in the required JSON format.

<json>
{
    "reasoning": "Your detailed step-by-step analysis and reasoning",
    "response": "The final grade/score (a number or specific grade value)"
}
</json>

Important: 
- The "response" field must contain only the final grade/score (e.g., "7", "0", "3.5")
- Use the "reasoning" field to show your work
- Do not include any text outside the <json> tags
- The response value should be a simple number, not a sentence or explanation
- Example correct response: "response": "5"
- Example incorrect response: "response": "The student gets 5 points"'''
        
        return f"""You are an expert grader for {domain} problems. Your task is to evaluate a student's answer against the correct solution and grading guidelines.

## Problem
{problem}

## Correct Solution
{solution}

## Grading Guidelines
{grading_guidelines}

## Student's Answer
{student_answer}

## Instructions
1. Carefully analyze the student's answer step by step
2. Compare it against the correct solution
3. Apply the grading guidelines strictly
4. Provide your reasoning before giving the final grade
5. Respond in JSON format with the following schema:

<json>
{{
    "reasoning": "Your detailed step-by-step analysis and reasoning",
    "response": "The final grade/score (a number or specific grade value)"
}}
</json>

Important: 
- The "response" field must contain only the final grade/score (e.g., "7", "0", "3.5")
- Use the "reasoning" field to show your work
- Be precise and follow the grading guidelines exactly
- If the student answer is empty, nonsensical, or completely wrong, assign grade "0"
- If the student answer is fully correct, assign the maximum grade
- Respond ONLY with the JSON block, no additional text
- The response field should be a simple number or grade value, not a sentence"""

    def _try_extract_prediction(self, text: str) -> tuple[str, str | None]:
        """Try to extract prediction from response text.
        
        Returns:
            (prediction, reasoning)
        """
        try:
            extracted = _extract_json_with_retry(text)
            if extracted:
                last = extracted[-1]
                prediction = last.get("response", "None")
                reasoning = last.get("reasoning")
                
                # Validate the prediction - it should be a simple grade value
                if prediction and prediction != "None":
                    pred_str = str(prediction).strip()
                    # Check if it's a valid grade format (number or numeric string)
                    # Allow formats like "7", "0", "3.5", "2/7"
                    if re.match(r'^[0-9]+(?:\.[0-9]+)?(?:/[0-9]+)?$', pred_str):
                        return pred_str, reasoning
                    else:
                        # If it contains non-numeric characters, it might be a sentence
                        # Try to extract just the number
                        num_match = re.search(r'\b([0-9]+(?:\.[0-9]+)?)\b', pred_str)
                        if num_match:
                            return num_match.group(1), reasoning
                        # If no number found, return "None" to trigger fallback
                        return "None", reasoning
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
        
        return "None", None

    def _extract_fallback_grade(self, text: str) -> str | None:
        """Try to extract a grade using fallback heuristics.
        
        Looks for common grade patterns in the text.
        """
        # Look for grade/score patterns - more comprehensive patterns
        patterns = [
            # Direct grade/score assignments
            r'grade[\s]*[:=][\s]*"?([0-9]+(?:\.[0-9]+)?)"?',
            r'score[\s]*[:=][\s]*"?([0-9]+(?:\.[0-9]+)?)"?',
            r'response[\s]*[:=][\s]*"?([0-9]+(?:\.[0-9]+)?)"?',
            r'final grade[\s]*[:=][\s]*"?([0-9]+(?:\.[0-9]+)?)"?',
            r'final score[\s]*[:=][\s]*"?([0-9]+(?:\.[0-9]+)?)"?',
            # "X out of Y" patterns
            r'([0-9]+(?:\.[0-9]+)?)[\s]*out of[\s]*[0-9]+',
            # Fraction patterns like "7/7" or "3 / 7"
            r'([0-9]+(?:\.[0-9]+)?)[\s]*/[\s]*[0-9]+',
            # "Grade: X" or "Score: X" with word boundaries
            r'\bgrade\s*[:=\s]+([0-9]+(?:\.[0-9]+)?)\b',
            r'\bscore\s*[:=\s]+([0-9]+(?:\.[0-9]+)?)\b',
            # "assigned grade X" or "give grade X"
            r'(?:assigned|give|assign|is)[\s]+(?:a\s+)?grade[\s]+(?:of\s+)?([0-9]+(?:\.[0-9]+)?)',
            # Standalone numbers that could be grades (0-7 for IMO)
            r'\b([0-7](?:\.[0-9]+)?)\b',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return None

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with retry logic.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Validate inputs first
        is_valid, error_msg = self._validate_inputs(inputs)
        if not is_valid:
            self.log_fn(f"Input validation failed: {error_msg}")
            return f"Error: {error_msg}", []
        
        instruction = self._build_grading_prompt(inputs)
        msg_history = []
        prediction = "None"
        
        # Try with retries
        for attempt in range(self.max_retries):
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=msg_history if attempt > 0 else [],
                )
                
                # Extract prediction
                text = msg_history[-1]["text"] if msg_history else ""
                pred, reasoning = self._try_extract_prediction(text)
                
                if pred != "None":
                    prediction = pred
                    if reasoning:
                        self.log_fn(f"Grading reasoning: {reasoning[:200]}...")
                    self.log_fn(f"Successfully extracted grade: {prediction}")
                    break
                
                # Try fallback extraction
                fallback = self._extract_fallback_grade(text)
                if fallback:
                    self.log_fn(f"Using fallback grade extraction: {fallback}")
                    prediction = fallback
                    break
                
                # Log the failed extraction for debugging
                self.log_fn(f"Attempt {attempt + 1}: Failed to extract grade from response")
                if len(text) > 200:
                    self.log_fn(f"Response preview: {text[:200]}...")
                else:
                    self.log_fn(f"Response: {text}")
                
                # If extraction failed, add a follow-up message asking for proper format
                if attempt < self.max_retries - 1:
                    instruction = self._build_grading_prompt(inputs, is_retry=True)
                    
            except Exception as e:
                self.log_fn(f"Attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    break
        
        # Final validation of the prediction
        if prediction == "None" or prediction is None:
            self.log_fn("Warning: Could not extract a valid grade after all retries")
            # Try one more time with a simple extraction from the last response
            if msg_history:
                last_text = msg_history[-1].get("text", "")
                # Look for any number that could be a grade
                simple_match = re.search(r'\b([0-7](?:\.[0-9]+)?)\b', last_text)
                if simple_match:
                    prediction = simple_match.group(1)
                    self.log_fn(f"Last-resort extraction found grade: {prediction}")
        
        return str(prediction), msg_history
