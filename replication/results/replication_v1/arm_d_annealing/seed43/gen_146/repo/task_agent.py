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

# Valid grade values for validation
VALID_GRADES = {
    "correct", "partial", "incorrect",
    "true", "false", "yes", "no",
    "0", "1", "2", "3", "4", "5",
    "0.0", "0.5", "1.0", "2.0", "3.0", "4.0", "5.0",
    "pass", "fail", "none"
}


def _clean_json_string(json_str: str) -> str:
    """Clean and normalize a JSON string for parsing.
    
    Handles common formatting issues that LLMs produce.
    """
    # Remove BOM if present
    json_str = json_str.lstrip('\ufeff')
    
    # Remove leading/trailing whitespace
    json_str = json_str.strip()
    
    # Remove trailing commas before closing braces/brackets
    json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
    
    # Remove trailing commas in arrays/objects (more aggressive)
    json_str = re.sub(r',(\s*\n\s*[}\]])', r'\1', json_str)
    
    # Fix single quotes to double quotes for keys and string values
    # This is a simplified approach - handles common cases
    json_str = re.sub(r"'([^']*?)'\s*:", r'"\1":', json_str)
    json_str = re.sub(r":\s*'([^']*?)'([,}\]])", r': "\1"\2', json_str)
    
    # Fix unquoted keys (simple cases)
    json_str = re.sub(r'(\{|,\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', json_str)
    
    # Remove comments (both // and /* */)
    json_str = re.sub(r'//.*?\n', '\n', json_str)
    json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
    
    # Normalize newlines and whitespace
    json_str = re.sub(r'\n\s*\n', '\n', json_str)
    
    return json_str


def _extract_json_flexible(text: str) -> dict | None:
    """Ultra-flexible JSON extraction for severely malformed responses.
    
    This is a last-resort method that tries to extract key-value pairs
    from text that doesn't resemble valid JSON at all.
    """
    result = {}
    
    # Look for response field with maximum flexibility
    # Pattern: response followed by colon and value
    response_patterns = [
        # Standard quoted
        r'["\']?response["\']?\s*[:=]\s*["\']([^"\']+)["\']',
        # Unquoted value
        r'["\']?response["\']?\s*[:=]\s*([^\n,}\]]+)',
        # Response at start of line
        r'(?m)^\s*["\']?response["\']?\s*[:=]\s*["\']?([^\n,}\]]+)["\']?',
    ]
    
    for pattern in response_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            result["response"] = match.group(1).strip().strip('"\'')
            break
    
    # Look for reasoning field
    reasoning_patterns = [
        r'["\']?reasoning["\']?\s*[:=]\s*["\']([^"]*(?:\.[^"]*)*)["\']',
        r'["\']?reasoning["\']?\s*[:=]\s*([^\n,}\]]+)',
        r'(?m)^\s*["\']?reasoning["\']?\s*[:=]\s*["\']?([^\n]*(?:\.[^\n]*)*)["\']?',
    ]
    
    for pattern in reasoning_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            reasoning = match.group(1).strip().strip('"\'')
            # Limit reasoning length
            if len(reasoning) > 1000:
                reasoning = reasoning[:500] + "..." + reasoning[-500:]
            result["reasoning"] = reasoning
            break
    
    return result if "response" in result else None


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
            # Try cleaning the JSON
            cleaned = _clean_json_string(inner)
            try:
                results.append(json.loads(cleaned))
            except json.JSONDecodeError:
                # Try to extract fields using regex as last resort
                try:
                    response_match = re.search(r'"response"\s*:\s*"([^"]+)"', inner)
                    if not response_match:
                        # Try with single quotes
                        response_match = re.search(r"'response'\s*:\s*'([^']+)'", inner)
                    if not response_match:
                        # Try finding unquoted values
                        response_match = re.search(r'"response"\s*:\s*([^",}\]\n]+)', inner)
                    if not response_match:
                        # Try case-insensitive match
                        response_match = re.search(r'"?response"?\s*:\s*"([^"]+)"', inner, re.IGNORECASE)
                    
                    if response_match:
                        reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]*)"', inner)
                        if not reasoning_match:
                            reasoning_match = re.search(r"'reasoning'\s*:\s*'([^']*)'", inner)
                        if not reasoning_match:
                            reasoning_match = re.search(r'"?reasoning"?\s*:\s*"([^"]*)"', inner, re.IGNORECASE)
                        
                        results.append({
                            "response": response_match.group(1).strip().strip('"\''),
                            "reasoning": reasoning_match.group(1) if reasoning_match else ""
                        })
                except Exception:
                    continue
    return results or None


def _extract_json_with_regex(text: str) -> list[dict] | None:
    """Fallback JSON extraction using regex for malformed responses."""
    results = []
    
    # Try to find JSON objects in code blocks with various formats
    code_block_patterns = [
        r'```(?:json)?\s*(\{[\s\S]*?\})\s*```',
        r'```\s*(\{[\s\S]*?\})\s*```',
        r'```json\s*([\s\S]*?)```',
        r'`(\{[\s\S]*?\})`',
    ]
    for pattern in code_block_patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                results.append(json.loads(match))
            except json.JSONDecodeError:
                # Try cleaning the match
                cleaned = _clean_json_string(match)
                try:
                    results.append(json.loads(cleaned))
                except json.JSONDecodeError:
                    continue
    
    # Try to find any JSON-like structure with "response" key
    if not results:
        # Look for response field with various quote styles
        response_patterns = [
            r'"response"\s*:\s*"([^"]+)"',
            r"'response'\s*:\s*'([^']+)'",
            r'"response"\s*:\s*([^",}\]\n]+)',
            r'response\s*:\s*"([^"]+)"',
            r'response\s*:\s*([^",}\]\n]+)',
        ]
        for pattern in response_patterns:
            response_match = re.search(pattern, text, re.IGNORECASE)
            if response_match:
                reasoning_patterns = [
                    r'"reasoning"\s*:\s*"([^"]*)"',
                    r"'reasoning'\s*:\s*'([^']*)'",
                    r'reasoning\s*:\s*"([^"]*)"',
                ]
                reasoning = ""
                for r_pattern in reasoning_patterns:
                    reasoning_match = re.search(r_pattern, text, re.IGNORECASE)
                    if reasoning_match:
                        reasoning = reasoning_match.group(1)
                        break
                
                results.append({
                    "response": response_match.group(1).strip().strip('"\''),
                    "reasoning": reasoning
                })
                break
    
    # Last resort: look for grade keywords in the text
    if not results:
        text_lower = text.lower()
        # Check for explicit grade statements
        if re.search(r'\b(correct|right|accurate|valid|true|yes)\b', text_lower):
            results.append({"response": "Correct", "reasoning": "Extracted from text analysis - positive indicators found"})
        elif re.search(r'\b(partial|partially correct|incomplete|somewhat)\b', text_lower):
            results.append({"response": "Partial", "reasoning": "Extracted from text analysis - partial indicators found"})
        elif re.search(r'\b(incorrect|wrong|error|invalid|false|no)\b', text_lower):
            results.append({"response": "Incorrect", "reasoning": "Extracted from text analysis - negative indicators found"})
    
    return results or None


def _validate_grade(prediction: str, valid_grades: set[str] | None = None) -> tuple[bool, str]:
    """Validate that a grade is in the set of valid grades.
    
    Returns:
        (is_valid, normalized_grade) tuple
    """
    if valid_grades is None:
        valid_grades = VALID_GRADES
    
    if not prediction or prediction == "None":
        return False, prediction
    
    # Normalize the prediction
    normalized = prediction.strip().lower()
    
    # Check exact match
    if normalized in valid_grades:
        return True, prediction.strip()
    
    # Check if it's a numeric grade that might be valid
    try:
        float_val = float(normalized)
        # If it's a number between 0 and 5, it's likely valid
        if 0 <= float_val <= 5:
            return True, prediction.strip()
    except ValueError:
        pass
    
    # Check for common variations
    grade_mappings = {
        "correct": ["correct", "right", "accurate", "valid", "true", "yes", "pass"],
        "partial": ["partial", "partially correct", "incomplete", "somewhat"],
        "incorrect": ["incorrect", "wrong", "error", "invalid", "false", "no", "fail"],
    }
    
    for standard, variations in grade_mappings.items():
        if normalized in variations:
            return True, standard.capitalize()
    
    return False, prediction.strip()


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
            previous_error: Error message from previous attempt (if retry)
        """
        domain = inputs.get("domain", "unknown")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        # Base prompt with optimized structure
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
- **Partial**: The answer shows some correct reasoning but has minor errors, incomplete steps, or partially correct results.
- **Incorrect**: The answer contains fundamental errors, wrong approach, or completely wrong results.

## Response Format (REQUIRED - READ CAREFULLY)
You MUST respond with valid JSON wrapped in <json>...</json> tags. 

CRITICAL RULES:
1. The JSON must be the ONLY content between the tags - no markdown, no extra text
2. Use double quotes for all strings (not single quotes)
3. Do not include trailing commas
4. The "response" field must contain ONLY the grade (e.g., "Correct", "Partial", "Incorrect")
5. The "reasoning" field should contain your detailed analysis

CORRECT EXAMPLE:
<json>
{{
    "reasoning": "The student correctly identified the approach and arrived at the right answer through valid mathematical reasoning.",
    "response": "Correct"
}}
</json>

INCORRECT EXAMPLES (DO NOT DO THESE):
- <json>{{'response': 'Correct'}}</json>  (single quotes)
- <json>{{"response": "Correct",}}</json>  (trailing comma)
- <json>{{"response": "Correct because..."}}</json>  (explanation in response field)

Now provide your evaluation:"""

        # Add error feedback for retry attempts
        if is_retry and previous_error:
            retry_prompt = f"""ERROR: Your previous response was invalid. {previous_error}

You MUST fix this and respond with ONLY valid JSON wrapped in <json>...</json> tags.

REMEMBER:
1. Use double quotes ONLY (not single quotes)
2. No trailing commas
3. JSON must be valid
4. Include both "reasoning" and "response" fields

Correct format:
<json>
{{
    "reasoning": "Your detailed analysis here...",
    "response": "Correct"
}}
</json>

Now try again with the original task:

{base_prompt}"""
            return retry_prompt
        
        return base_prompt

    def _extract_prediction(self, text: str) -> tuple[str, str, str]:
        """Extract prediction and reasoning from response text.
        
        Uses a multi-tier extraction strategy:
        1. Primary: Extract from <json> tags
        2. Fallback: Extract from code blocks and JSON-like structures
        3. Last resort: Ultra-flexible field extraction
        
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
        
        if extracted:
            last_json = extracted[-1]
            if "response" in last_json:
                prediction = str(last_json["response"]).strip()
            else:
                error_message = "Missing 'response' field in JSON"
            if "reasoning" in last_json:
                reasoning = str(last_json["reasoning"]).strip()
        else:
            # Last resort: ultra-flexible extraction
            flexible_result = _extract_json_flexible(text)
            if flexible_result:
                prediction = str(flexible_result.get("response", "None")).strip()
                reasoning = str(flexible_result.get("reasoning", "")).strip()
                if prediction == "None":
                    error_message = "Could not extract 'response' field even with flexible extraction"
            else:
                error_message = "No valid JSON found in response"
        
        # Validate the grade
        if prediction != "None":
            is_valid, normalized = _validate_grade(prediction)
            if not is_valid:
                error_message = f"Invalid grade value: '{prediction}'. Expected one of: Correct, Partial, Incorrect, or numeric score"
            else:
                prediction = normalized
        
        return prediction, reasoning, error_message

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
                # Add small delay between retries to avoid rate limiting
                if attempt > 0:
                    time.sleep(0.5 * attempt)
                
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=msg_history if attempt > 0 else [],
                )
                
                last_text = msg_history[-1]["text"] if msg_history else ""
                prediction, reasoning, error_message = self._extract_prediction(last_text)
                
                if prediction != "None" and not error_message:
                    self.log_fn(f"Successfully extracted prediction: {prediction}")
                    if reasoning:
                        self.log_fn(f"Reasoning: {reasoning[:200]}...")
                    break
                else:
                    last_error = error_message or f"Attempt {attempt + 1}: Failed to extract valid prediction"
                    self.log_fn(last_error)
                    
                    # Prepare for retry with error feedback
                    if attempt < self.max_retries - 1:
                        instruction = self._build_grading_prompt(inputs, is_retry=True, previous_error=last_error)
                    
            except Exception as e:
                last_error = f"Error in attempt {attempt + 1}: {e}"
                self.log_fn(last_error)
                if attempt == self.max_retries - 1:
                    prediction = f"Error: {e}"
        
        # Final validation check
        if prediction == "None" or prediction.startswith("Error:"):
            self.log_fn(f"Failed to get valid prediction after {self.max_retries} attempts. Last error: {last_error}")
        
        return str(prediction), msg_history
