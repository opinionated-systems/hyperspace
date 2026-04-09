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
import time
from typing import Any

from agent.llm_client import get_response_from_llm, EVAL_MODEL

logger = logging.getLogger(__name__)

# Valid grading responses for validation
VALID_GRADES = {"Correct", "Partial", "Incorrect"}

# Grade aliases for fuzzy matching
GRADE_ALIASES = {
    "correct": "Correct",
    "right": "Correct",
    "true": "Correct",
    "accurate": "Correct",
    "valid": "Correct",
    "partial": "Partial",
    "partially": "Partial",
    "partially correct": "Partial",
    "some": "Partial",
    "incomplete": "Partial",
    "incorrect": "Incorrect",
    "wrong": "Incorrect",
    "false": "Incorrect",
    "error": "Incorrect",
    "invalid": "Incorrect",
    "none": "Incorrect",
}


def _clean_json_string(text: str) -> str:
    """Clean common JSON formatting issues with enhanced robustness."""
    if not text:
        return text
    
    # Remove trailing commas before closing braces/brackets
    cleaned = re.sub(r',(\s*[}\]])', r'\1', text)
    # Fix single quotes to double quotes (basic cases)
    cleaned = re.sub(r"'([^']*?)':", r'"\1":', cleaned)
    # Remove control characters except newlines and tabs
    cleaned = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', cleaned)
    # Fix escaped newlines that might break parsing
    cleaned = cleaned.replace('\\n', '\n')
    # Fix double-escaped quotes
    cleaned = cleaned.replace('\\"', '"')
    # Remove BOM if present
    cleaned = cleaned.lstrip('\ufeff')
    # Fix common unicode issues
    cleaned = cleaned.replace('\u0000', '')
    return cleaned


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks with enhanced robustness.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    Also handles nested JSON objects and common formatting issues.
    """
    if not text:
        return None
    
    results = []
    search_from = 0
    max_iterations = 100  # Prevent infinite loops
    iterations = 0
    
    while iterations < max_iterations:
        iterations += 1
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
                    response_match = re.search(r'"response"\s*:\s*"([^"]+)"', inner, re.IGNORECASE)
                    if response_match:
                        reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]*)"', inner, re.IGNORECASE)
                        results.append({
                            "response": response_match.group(1),
                            "reasoning": reasoning_match.group(1) if reasoning_match else ""
                        })
                except Exception:
                    # Try to find any JSON-like structure
                    try:
                        # Look for JSON object pattern
                        json_match = re.search(r'\{[^{}]*"response"[^{}]*\}', inner, re.DOTALL | re.IGNORECASE)
                        if json_match:
                            try:
                                results.append(json.loads(json_match.group(0)))
                            except json.JSONDecodeError:
                                pass
                    except Exception:
                        continue
    return results or None


def _extract_json_with_regex(text: str) -> list[dict] | None:
    """Fallback JSON extraction using regex for malformed responses with enhanced patterns."""
    if not text:
        return None
    
    results = []
    
    # Try to find JSON objects in code blocks (with or without json label)
    code_block_patterns = [
        r'```(?:json)?\s*(\{[\s\S]*?\})\s*```',
        r'```\s*(\{[\s\S]*?\})\s*```',
        r'```json\s*([\s\S]*?)```',
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
    
    # Try to find any JSON-like structure with "response" key (case-insensitive)
    if not results:
        response_match = re.search(r'"response"\s*:\s*"([^"]+)"', text, re.IGNORECASE)
        if response_match:
            reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]*)"', text, re.IGNORECASE)
            results.append({
                "response": response_match.group(1),
                "reasoning": reasoning_match.group(1) if reasoning_match else ""
            })
    
    # Try to find JSON objects without code blocks
    if not results:
        json_pattern = r'\{(?:[^{}]|"(?:[^"\\]|\\.)*")*"response"(?:[^{}]|"(?:[^"\\]|\\.)*")*\}'
        matches = re.findall(json_pattern, text, re.DOTALL | re.IGNORECASE)
        for match in matches:
            try:
                results.append(json.loads(match))
            except json.JSONDecodeError:
                try:
                    cleaned = _clean_json_string(match)
                    results.append(json.loads(cleaned))
                except json.JSONDecodeError:
                    continue
    
    # Last resort: look for the grade directly in the text with fuzzy matching
    if not results:
        text_lower = text.lower()
        for alias, grade in GRADE_ALIASES.items():
            # Use word boundaries for better matching
            pattern = r'\b' + re.escape(alias) + r'\b'
            if re.search(pattern, text_lower):
                # Extract surrounding text as reasoning
                match = re.search(pattern, text_lower)
                if match:
                    idx = match.start()
                    context = text[max(0, idx-150):min(len(text), idx+150)]
                    results.append({
                        "response": grade,
                        "reasoning": f"Grade extracted from text context: {context.strip()}"
                    })
                    break
    
    return results or None


def _normalize_grade(grade: str) -> str:
    """Normalize a grade string to one of the valid grades."""
    if not grade:
        return "Incorrect"
    
    grade_clean = grade.strip().lower()
    
    # Direct match
    if grade_clean in GRADE_ALIASES:
        return GRADE_ALIASES[grade_clean]
    
    # Check if it contains a valid grade
    for alias, normalized in GRADE_ALIASES.items():
        if alias in grade_clean:
            return normalized
    
    # Check for partial matches in valid grades
    for valid_grade in VALID_GRADES:
        if valid_grade.lower() in grade_clean:
            return valid_grade
    
    return "Incorrect"


def _validate_grading_output(data: dict) -> tuple[bool, str]:
    """Validate that the extracted grading output is valid.
    
    Returns:
        (is_valid, error_message) tuple
    """
    if not isinstance(data, dict):
        return False, "Extracted data is not a dictionary"
    
    if "response" not in data:
        return False, "Missing 'response' field"
    
    response = str(data.get("response", "")).strip()
    
    # Normalize and validate the grade
    normalized = _normalize_grade(response)
    if normalized not in VALID_GRADES:
        return False, f"Invalid response value: '{response}'. Expected one of: {VALID_GRADES}"
    
    # Update the response to normalized value
    data["response"] = normalized
    
    # Validate reasoning field exists and is non-empty
    reasoning = data.get("reasoning", "")
    if not reasoning or not str(reasoning).strip():
        data["reasoning"] = "No reasoning provided"
    
    return True, ""


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.max_retries = 3

    def _build_grading_prompt(self, inputs: dict, is_retry: bool = False, previous_error: str = "") -> str:
        """Build a structured prompt for the grading task with chain-of-thought.
        
        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer
            is_retry: Whether this is a retry attempt
            previous_error: Error message from previous attempt (if any)
        """
        domain = inputs.get("domain", "unknown")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        base_prompt = f"""You are an expert grader for {domain} problems. Your task is to evaluate a student's answer against the correct solution and grading guidelines.

## Problem Statement
{problem}

## Correct Solution
{solution}

## Grading Guidelines
{guidelines}

## Student's Answer
{student_answer}

## Instructions
1. First, analyze the student's answer step by step. Compare it against the correct solution.
2. Identify any errors, omissions, or alternative valid approaches.
3. Consider the grading guidelines carefully.
4. Provide your reasoning for the grade you will assign.
5. Finally, provide your grade/assessment in the JSON format below.

## Grading Rubric
When assigning grades, consider:
- **Correct**: The answer matches the solution or uses an equivalent valid approach with correct reasoning and final result.
  - The student's final answer is mathematically equivalent to the correct solution
  - The reasoning is sound and follows valid mathematical/logical steps
  - Alternative valid approaches are acceptable
- **Partial**: The answer shows some correct reasoning but has minor errors, incomplete steps, or partially correct results.
  - Some correct steps are present but the final answer is wrong or incomplete
  - The approach is valid but execution has minor errors
  - Partial credit scenarios per the grading guidelines
- **Incorrect**: The answer contains fundamental errors, wrong approach, or completely wrong results.
  - The approach is fundamentally flawed
  - Major conceptual errors are present
  - The answer is completely unrelated or nonsensical

## Response Format (REQUIRED - STRICT)
You MUST respond with ONLY valid JSON wrapped in <json>...</json> tags. Do not include any text outside the JSON tags.

<json>
{{
    "reasoning": "Your detailed step-by-step analysis and reasoning...",
    "response": "Correct"
}}
</json>

IMPORTANT RULES:
1. The JSON must be valid and parseable
2. The 'response' field MUST contain exactly one of: 'Correct', 'Partial', or 'Incorrect' (case-sensitive)
3. The 'reasoning' field must contain your detailed analysis
4. Do not include any text before or after the <json> tags
5. Do not use markdown formatting inside the JSON values
6. Ensure proper JSON escaping for quotes and special characters"""

        if is_retry and previous_error:
            return f"""ERROR: Your previous response was invalid: {previous_error}

You MUST fix this and respond with ONLY valid JSON wrapped in <json>...</json> tags.

Common errors to avoid:
- Missing or mismatched quotes in JSON
- Trailing commas before closing braces
- Using single quotes instead of double quotes
- Including text outside the <json> tags
- Using markdown formatting inside JSON values

Correct format example:
<json>
{{
    "reasoning": "The student correctly applied the quadratic formula and arrived at the right answer...",
    "response": "Correct"
}}
</json>

Now try again with the original task:

{base_prompt}"""
        
        return base_prompt

    def _extract_prediction(self, text: str) -> tuple[str, str, str]:
        """Extract prediction and reasoning from response text with validation.
        
        Returns:
            (prediction, reasoning, error_message) tuple
        """
        prediction = "None"
        reasoning = ""
        error_msg = ""
        
        # Try primary extraction method
        extracted = _extract_jsons(text)
        if extracted is None:
            # Fallback to regex extraction
            extracted = _extract_json_with_regex(text)
        
        if not extracted:
            return "None", "", "No JSON found in response"
        
        # Validate each extracted JSON
        for data in extracted:
            is_valid, error_msg = _validate_grading_output(data)
            if is_valid:
                prediction = str(data.get("response", "None"))
                reasoning = str(data.get("reasoning", ""))
                return prediction, reasoning, ""
        
        # If none valid, try to extract and normalize from the last one
        if extracted:
            last_json = extracted[-1]
            raw_response = str(last_json.get("response", "None"))
            prediction = _normalize_grade(raw_response)
            reasoning = str(last_json.get("reasoning", ""))
            if prediction in VALID_GRADES:
                return prediction, reasoning, ""
        
        return prediction, reasoning, error_msg or "Validation failed"

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
                prediction, reasoning, error_msg = self._extract_prediction(last_text)
                
                # Validate the prediction
                if prediction != "None" and prediction in VALID_GRADES:
                    self.log_fn(f"Successfully extracted prediction: {prediction}")
                    if reasoning:
                        self.log_fn(f"Reasoning: {reasoning[:200]}...")
                    break
                else:
                    last_error = error_msg if error_msg else f"Invalid prediction: {prediction}"
                    self.log_fn(f"Attempt {attempt + 1}: {last_error}, retrying...")
                    
                    # Add feedback for retry
                    if attempt < self.max_retries - 1:
                        instruction = self._build_grading_prompt(inputs, is_retry=True, previous_error=last_error)
                    
            except Exception as e:
                last_error = str(e)
                self.log_fn(f"Error in attempt {attempt + 1}: {e}")
                if attempt < self.max_retries - 1:
                    # Brief pause before retry
                    time.sleep(0.5)
                else:
                    prediction = f"Error: {e}"
        
        # Final validation - ensure we return a valid grade or error
        if prediction not in VALID_GRADES and not prediction.startswith("Error:"):
            self.log_fn(f"Warning: Final prediction '{prediction}' not in valid grades, defaulting to 'Incorrect'")
            prediction = "Incorrect"
        
        return str(prediction), msg_history
