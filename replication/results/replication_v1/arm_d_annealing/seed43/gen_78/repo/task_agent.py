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

# Valid grading responses for validation
VALID_GRADES = {"Correct", "Partial", "Incorrect"}


def _clean_json_string(json_str: str) -> str:
    """Clean common JSON formatting issues.
    
    Handles:
    - Trailing commas before closing braces/brackets
    - Single quotes to double quotes (basic cases)
    - Unescaped newlines in strings
    - Extra whitespace
    """
    # Remove trailing commas before closing braces/brackets
    cleaned = re.sub(r',(\s*[}\]])', r'\1', json_str)
    # Fix single quotes to double quotes for keys (basic cases)
    cleaned = re.sub(r"'([^']*?)':", r'"\1":', cleaned)
    # Fix single quotes to double quotes for string values (basic cases)
    cleaned = re.sub(r":\s*'([^']*?)'([,}\]])", r': "\1"\2', cleaned)
    # Replace unescaped newlines in strings with escaped newlines
    cleaned = re.sub(r'(?<=")\n(?=")', r'\\n', cleaned)
    cleaned = re.sub(r'(?<=.)\n(?=.)', r' ', cleaned)
    return cleaned.strip()


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    Also handles nested JSON objects and common formatting issues.
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
        except json.JSONDecodeError as e:
            # Try common fixes
            try:
                cleaned = _clean_json_string(inner)
                results.append(json.loads(cleaned))
            except json.JSONDecodeError:
                # Try to extract just the response field if present
                try:
                    response_match = re.search(r'"response"\s*:\s*"([^"]+)"', inner)
                    if response_match:
                        reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]*)"', inner)
                        results.append({
                            "response": response_match.group(1),
                            "reasoning": reasoning_match.group(1) if reasoning_match else ""
                        })
                except Exception:
                    continue
    return results or None


def _extract_json_with_regex(text: str) -> list[dict] | None:
    """Fallback JSON extraction using regex for malformed responses."""
    results = []
    
    # Try to find JSON objects in code blocks (with or without json label)
    code_block_patterns = [
        r'```(?:json)?\s*(\{[\s\S]*?\})\s*```',
        r'```\s*(\{[\s\S]*?\})\s*```',
    ]
    for pattern in code_block_patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                results.append(json.loads(match))
            except json.JSONDecodeError:
                # Try cleaning the match
                try:
                    cleaned = _clean_json_string(match)
                    results.append(json.loads(cleaned))
                except json.JSONDecodeError:
                    continue
    
    # Try to find any JSON-like structure with "response" key
    if not results:
        response_match = re.search(r'"response"\s*:\s*"([^"]+)"', text)
        if response_match:
            reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]*)"', text)
            results.append({
                "response": response_match.group(1),
                "reasoning": reasoning_match.group(1) if reasoning_match else ""
            })
    
    # Last resort: look for grade keywords in the text
    if not results:
        text_lower = text.lower()
        # Check for explicit grade statements first
        if any(phrase in text_lower for phrase in ['grade: correct', 'is correct', 'answer is correct']):
            results.append({"response": "Correct", "reasoning": "Extracted from text analysis"})
        elif any(phrase in text_lower for phrase in ['grade: partial', 'is partial', 'partially correct']):
            results.append({"response": "Partial", "reasoning": "Extracted from text analysis"})
        elif any(phrase in text_lower for phrase in ['grade: incorrect', 'is incorrect', 'answer is incorrect']):
            results.append({"response": "Incorrect", "reasoning": "Extracted from text analysis"})
        # Then check for general sentiment words
        elif any(word in text_lower for word in ['correct', 'right', 'accurate', 'valid', 'perfect']):
            results.append({"response": "Correct", "reasoning": "Extracted from text analysis"})
        elif any(word in text_lower for word in ['partial', 'incomplete', 'partially', 'some errors']):
            results.append({"response": "Partial", "reasoning": "Extracted from text analysis"})
        elif any(word in text_lower for word in ['incorrect', 'wrong', 'error', 'invalid', 'mistake']):
            results.append({"response": "Incorrect", "reasoning": "Extracted from text analysis"})
    
    return results or None


def _validate_grade(prediction: str) -> tuple[bool, str]:
    """Validate that the prediction is a valid grade.
    
    Returns:
        (is_valid, normalized_grade) tuple
    """
    if not prediction or prediction == "None":
        return False, "None"
    
    # Normalize the prediction
    normalized = prediction.strip()
    
    # Check for exact match
    if normalized in VALID_GRADES:
        return True, normalized
    
    # Check for case-insensitive match
    for valid_grade in VALID_GRADES:
        if normalized.lower() == valid_grade.lower():
            return True, valid_grade
    
    # Check for partial matches (e.g., "mostly correct" -> "Correct")
    normalized_lower = normalized.lower()
    if any(word in normalized_lower for word in ['correct', 'right', 'accurate', 'valid', 'true']):
        return True, "Correct"
    elif any(word in normalized_lower for word in ['partial', 'incomplete', 'somewhat', 'half']):
        return True, "Partial"
    elif any(word in normalized_lower for word in ['incorrect', 'wrong', 'error', 'invalid', 'false', 'mistake']):
        return True, "Incorrect"
    
    return False, "None"


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.max_retries = 3

    def _build_grading_prompt(self, inputs: dict, is_retry: bool = False, previous_error: str = "") -> str:
        """Build a structured prompt for the grading task with chain-of-thought."""
        domain = inputs.get("domain", "unknown")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        if is_retry and previous_error:
            error_section = f"""
## PREVIOUS ERROR
Your previous response had the following issue: {previous_error}

Please correct this and try again.
"""
        else:
            error_section = ""
        
        return f"""You are an expert grader for {domain} problems. Your task is to evaluate a student's answer against the correct solution and grading guidelines.

## Problem Statement
{problem}

## Correct Solution
{solution}

## Grading Guidelines
{guidelines}

## Student's Answer
{student_answer}
{error_section}

## Instructions
1. First, analyze the student's answer step by step. Compare it against the correct solution.
2. Identify any errors, omissions, or alternative valid approaches.
3. Consider the grading guidelines carefully.
4. Provide your reasoning for the grade you will assign.
5. Finally, provide your grade/assessment in the JSON format below.

## Grading Rubric
When assigning grades, consider:
- **Correct**: The answer matches the solution or uses an equivalent valid approach with correct reasoning and final result.
- **Partial**: The answer shows some correct reasoning but has minor errors, incomplete steps, or partially correct results.
- **Incorrect**: The answer contains fundamental errors, wrong approach, or completely wrong results.

## Response Format (REQUIRED - STRICT)
You MUST respond with ONLY valid JSON wrapped in <json>...</json> tags. Do not include any text outside the JSON tags.

The JSON must have exactly these two fields:
- "reasoning": A string containing your detailed step-by-step analysis
- "response": A string containing ONLY one of: "Correct", "Partial", or "Incorrect"

Example of a valid response:
<json>
{{
    "reasoning": "The student's answer correctly identifies the key steps...",
    "response": "Correct"
}}
</json>

IMPORTANT RULES:
1. The 'response' field MUST contain ONLY one of these exact values: "Correct", "Partial", or "Incorrect"
2. Do not add any text before or after the <json> tags
3. Ensure your JSON is valid - check for proper quotes, commas, and braces
4. The reasoning should be detailed but the response must be exactly one of the three allowed values"""

    def _extract_prediction(self, text: str) -> tuple[str, str, str]:
        """Extract prediction and reasoning from response text.
        
        Returns:
            (prediction, reasoning, error_message) tuple
        """
        prediction = "None"
        reasoning = ""
        error_message = ""
        
        # Try primary extraction method
        extracted = _extract_jsons(text)
        if extracted is None:
            # Fallback to regex extraction
            extracted = _extract_json_with_regex(text)
        
        if not extracted:
            return "None", "", "No JSON found in response"
        
        last_json = extracted[-1]
        
        # Check for required fields
        if "response" not in last_json:
            return "None", "", "Missing 'response' field in JSON"
        
        prediction = str(last_json.get("response", "None")).strip()
        reasoning = str(last_json.get("reasoning", "")).strip()
        
        # Validate the grade
        is_valid, normalized = _validate_grade(prediction)
        if not is_valid:
            return "None", reasoning, f"Invalid grade value: '{prediction}'. Must be one of: Correct, Partial, Incorrect"
        
        return normalized, reasoning, ""

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        instruction = self._build_grading_prompt(inputs)
        
        msg_history = []
        prediction = "None"
        reasoning = ""
        last_error = ""
        
        # Retry loop for robust extraction
        for attempt in range(self.max_retries):
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=msg_history if attempt > 0 else [],
                )
                
                last_text = msg_history[-1]["text"] if msg_history else ""
                prediction, reasoning, error = self._extract_prediction(last_text)
                
                if prediction != "None":
                    self.log_fn(f"Successfully extracted prediction: {prediction}")
                    if reasoning:
                        self.log_fn(f"Reasoning: {reasoning[:200]}...")
                    break
                else:
                    last_error = error
                    self.log_fn(f"Attempt {attempt + 1}: Failed to extract prediction - {error}")
                    
                    # Add feedback for retry
                    if attempt < self.max_retries - 1:
                        instruction = self._build_grading_prompt(inputs, is_retry=True, previous_error=last_error)
                    
            except Exception as e:
                last_error = str(e)
                self.log_fn(f"Error in attempt {attempt + 1}: {e}")
                if attempt < self.max_retries - 1:
                    instruction = self._build_grading_prompt(inputs, is_retry=True, previous_error=f"Exception occurred: {e}")
                else:
                    prediction = f"Error: {e}"
        
        # Final validation - if we still don't have a valid prediction, try one more time with a simplified prompt
        if prediction == "None" or prediction.startswith("Error:"):
            self.log_fn(f"All retries failed. Last error: {last_error}")
            # Try to extract any meaningful grade from the last response
            if msg_history:
                last_text = msg_history[-1]["text"]
                # Force extraction using keyword matching
                text_lower = last_text.lower()
                if any(word in text_lower for word in ['correct', 'right', 'accurate']):
                    prediction = "Correct"
                elif any(word in text_lower for word in ['partial', 'incomplete']):
                    prediction = "Partial"
                elif any(word in text_lower for word in ['incorrect', 'wrong', 'error']):
                    prediction = "Incorrect"
        
        return str(prediction), msg_history
