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
    Also handles nested JSON objects by tracking brace depth.
    Includes enhanced handling for common LLM output patterns.
    """
    results = []
    search_from = 0
    extraction_attempts = []
    
    while True:
        start = text.find("<json>", search_from)
        if start == -1:
            break
        end = text.find("</json>", start)
        if end == -1:
            break
        inner = text[start + 6:end].strip()
        search_from = end + 7
        
        # Try to parse the inner content as JSON
        try:
            parsed = json.loads(inner)
            results.append(parsed)
            extraction_attempts.append({"method": "direct", "success": True, "preview": str(parsed)[:100]})
        except json.JSONDecodeError as e:
            # Try to extract JSON by finding balanced braces
            json_obj = _extract_balanced_json(inner)
            if json_obj:
                results.append(json_obj)
                extraction_attempts.append({"method": "balanced_braces", "success": True, "preview": str(json_obj)[:100]})
            else:
                extraction_attempts.append({"method": "direct", "success": False, "error": str(e)[:100], "preview": inner[:200]})
    
    # Try to find JSON in markdown code blocks (even if <json> tags were found, for completeness)
    code_block_pattern = r'```(?:json)?\s*\n?(\{[\s\S]*?\})\s*```'
    matches = re.findall(code_block_pattern, text, re.DOTALL)
    for match in matches:
        try:
            parsed = json.loads(match)
            if parsed not in results:  # Avoid duplicates
                results.append(parsed)
                extraction_attempts.append({"method": "code_block", "success": True, "preview": str(parsed)[:100]})
        except json.JSONDecodeError:
            json_obj = _extract_balanced_json(match)
            if json_obj and json_obj not in results:
                results.append(json_obj)
                extraction_attempts.append({"method": "code_block_balanced", "success": True, "preview": str(json_obj)[:100]})
    
    # Try to find raw JSON objects (not in any tags) as last resort
    if not results:
        # Look for JSON-like structures that might be outside tags
        raw_json_obj = _extract_balanced_json(text)
        if raw_json_obj:
            results.append(raw_json_obj)
            extraction_attempts.append({"method": "raw_balanced", "success": True, "preview": str(raw_json_obj)[:100]})
    
    # Log extraction summary for debugging
    if extraction_attempts:
        logger.debug(f"JSON extraction attempts: {extraction_attempts}")
    
    return results or None


def _extract_balanced_json(text: str) -> dict | None:
    """Extract a JSON object by tracking brace depth.
    
    This handles cases where the JSON content contains nested braces
    that might confuse simple regex-based extraction.
    Also handles escaped characters, single-quoted strings, and common
    LLM output artifacts like trailing commas.
    """
    start_idx = -1
    brace_depth = 0
    in_string = False
    in_single_string = False
    escape_next = False
    
    for i, char in enumerate(text):
        if escape_next:
            escape_next = False
            continue
        if char == '\\' and (in_string or in_single_string):
            escape_next = True
            continue
        if char == '"' and not in_single_string:
            in_string = not in_string
            continue
        if char == "'" and not in_string:
            in_single_string = not in_single_string
            continue
        if not in_string and not in_single_string:
            if char == '{':
                if brace_depth == 0:
                    start_idx = i
                brace_depth += 1
            elif char == '}':
                brace_depth -= 1
                if brace_depth == 0 and start_idx != -1:
                    json_str = text[start_idx:i+1]
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        # Try to fix common LLM JSON issues
                        fixed = _fix_common_json_issues(json_str)
                        try:
                            return json.loads(fixed)
                        except json.JSONDecodeError:
                            continue
    return None


def _fix_common_json_issues(json_str: str) -> str:
    """Fix common JSON formatting issues from LLM outputs.
    
    Handles:
    - Trailing commas before closing braces/brackets
    - Single quotes instead of double quotes
    - Unescaped newlines in strings
    """
    import re
    
    # Remove trailing commas before } or ]
    json_str = re.sub(r',\s*}', '}', json_str)
    json_str = re.sub(r',\s*\]', ']', json_str)
    
    # Replace single quotes with double quotes (carefully)
    # This is a simple heuristic - only replace if not inside a string
    result = []
    in_string = False
    for i, char in enumerate(json_str):
        if char == '"' and (i == 0 or json_str[i-1] != '\\'):
            in_string = not in_string
            result.append(char)
        elif char == "'" and not in_string:
            result.append('"')
        else:
            result.append(char)
    
    return ''.join(result)


def _extract_json_with_regex(text: str) -> list[dict] | None:
    """Fallback JSON extraction using multiple strategies for malformed responses.
    
    Uses progressively more aggressive strategies to extract valid JSON.
    """
    results = []
    
    # Strategy 1: Try to find JSON objects in code blocks
    code_block_pattern = r'```(?:json)?\s*(\{[\s\S]*?\})\s*```'
    matches = re.findall(code_block_pattern, text, re.DOTALL)
    for match in matches:
        try:
            # Try to parse the matched content
            parsed = json.loads(match)
            if isinstance(parsed, dict):
                results.append(parsed)
        except json.JSONDecodeError:
            # Try balanced brace extraction
            balanced = _extract_balanced_json(match)
            if balanced:
                results.append(balanced)
            else:
                # Try fixing common issues
                fixed = _fix_common_json_issues(match)
                try:
                    parsed = json.loads(fixed)
                    if isinstance(parsed, dict):
                        results.append(parsed)
                except json.JSONDecodeError:
                    pass
    
    # Strategy 2: Look for any JSON-like structure with balanced braces
    if not results:
        balanced = _extract_balanced_json(text)
        if balanced:
            results.append(balanced)
    
    # Strategy 3: Extract key-value pairs for response and reasoning
    if not results:
        response_match = re.search(r'"response"\s*:\s*"([^"]*(?:\\.[^"]*)*)"', text, re.DOTALL)
        reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]*(?:\\.[^"]*)*)"', text, re.DOTALL)
        
        extracted = {}
        if response_match:
            extracted["response"] = response_match.group(1).replace('\\"', '"').replace('\\n', '\n')
        if reasoning_match:
            extracted["reasoning"] = reasoning_match.group(1).replace('\\"', '"').replace('\\n', '\n')
        
        if extracted:
            results.append(extracted)
    
    # Strategy 4: Look for response/reasoning in single quotes
    if not results:
        response_match = re.search(r"'response'\s*:\s*'([^']*(?:\\.[^']*)*)'", text, re.DOTALL)
        reasoning_match = re.search(r"'reasoning'\s*:\s*'([^']*(?:\\.[^']*)*)'", text, re.DOTALL)
        
        extracted = {}
        if response_match:
            extracted["response"] = response_match.group(1).replace("\\'", "'").replace('\\n', '\n')
        if reasoning_match:
            extracted["reasoning"] = reasoning_match.group(1).replace("\\'", "'").replace('\\n', '\n')
        
        if extracted:
            results.append(extracted)
    
    return results or None


def _normalize_prediction(prediction: str) -> str:
    """Normalize prediction string for consistent comparison.
    
    Handles common variations like case differences, extra whitespace,
    punctuation differences, and numeric grades.
    """
    if not prediction:
        return "None"
    
    # Strip whitespace and normalize case
    normalized = prediction.strip()
    
    # Handle common grade variations
    grade_map = {
        "correct": "Correct",
        "partial": "Partial",
        "partially correct": "Partial",
        "incorrect": "Incorrect",
        "wrong": "Incorrect",
        "right": "Correct",
        "true": "Correct",
        "false": "Incorrect",
        "yes": "Correct",
        "no": "Incorrect",
        "pass": "Correct",
        "fail": "Incorrect",
        "full": "Correct",
        "full marks": "Correct",
        "full credit": "Correct",
        "zero": "Incorrect",
        "0": "Incorrect",
        "1": "Correct",
        "100%": "Correct",
        "0%": "Incorrect",
        "50%": "Partial",
        "half": "Partial",
    }
    
    lower_pred = normalized.lower()
    if lower_pred in grade_map:
        return grade_map[lower_pred]
    
    # Handle numeric grades (0-100 scale)
    try:
        numeric_val = float(lower_pred)
        if numeric_val >= 90:
            return "Correct"
        elif numeric_val >= 50:
            return "Partial"
        else:
            return "Incorrect"
    except (ValueError, TypeError):
        pass
    
    # Handle fraction grades
    fraction_map = {
        "1/1": "Correct",
        "0/1": "Incorrect",
        "1/2": "Partial",
        "0.5": "Partial",
        "0.5/1": "Partial",
    }
    if lower_pred in fraction_map:
        return fraction_map[lower_pred]
    
    return normalized


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "", max_retries: int = 3) -> None:
        self.model = model
        self.log_fn = logger.info
        self.max_retries = max_retries
        self._extraction_stats = {"success": 0, "fallback": 0, "failure": 0}
        self._log_file = log_file
    
    def get_extraction_stats(self) -> dict:
        """Return the current extraction statistics."""
        return dict(self._extraction_stats)
    
    def reset_extraction_stats(self) -> None:
        """Reset extraction statistics to zero."""
        self._extraction_stats = {"success": 0, "fallback": 0, "failure": 0}
    
    def _validate_inputs(self, inputs: dict) -> tuple[bool, str]:
        """Validate the input dictionary for required fields.
        
        Returns:
            (is_valid, error_message) tuple
        """
        required_fields = ["problem", "solution", "student_answer"]
        missing_fields = []
        
        for field in required_fields:
            if field not in inputs:
                missing_fields.append(field)
            elif not inputs[field] or not str(inputs[field]).strip():
                missing_fields.append(f"{field} (empty)")
        
        if missing_fields:
            return False, f"Missing or empty required fields: {', '.join(missing_fields)}"
        
        # Validate field types and content
        for field in ["problem", "solution", "student_answer", "grading_guidelines"]:
            if field in inputs and inputs[field] is not None:
                if not isinstance(inputs[field], str):
                    return False, f"Field '{field}' must be a string, got {type(inputs[field]).__name__}"
        
        return True, ""

    def _build_grading_prompt(self, inputs: dict, is_retry: bool = False) -> str:
        """Build a structured prompt for the grading task with chain-of-thought."""
        domain = inputs.get("domain", "unknown")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        # Log prompt building for debugging
        self.log_fn(f"Building grading prompt for domain: {domain}, problem length: {len(problem)}, student answer length: {len(student_answer)}")
        
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

## Response Format (REQUIRED)
You MUST respond with valid JSON wrapped in <json>...</json> tags. Do not include any text outside the JSON tags.

<json>
{{
    "reasoning": "Your detailed step-by-step analysis and reasoning...",
    "response": "Your final grade/assessment (e.g., 'Correct', 'Partial', 'Incorrect', or a numeric score)"
}}
</json>

IMPORTANT: Ensure your JSON is valid and properly formatted. The 'response' field should contain only the grade, not the reasoning."""

        if is_retry:
            return f"""ERROR: Your previous response did not contain valid JSON with a 'response' field, or the JSON was malformed.

You MUST respond with ONLY valid JSON wrapped in <json>...</json> tags. Do not include any text before or after the JSON tags.

Correct format:
<json>
{{
    "reasoning": "Your detailed analysis here...",
    "response": "Correct"
}}
</json>

Now try again with the original task:

{base_prompt}"""
        
        return base_prompt

    def _extract_prediction(self, text: str) -> tuple[str, str, str]:
        """Extract prediction and reasoning from response text.
        
        Returns:
            (prediction, reasoning, method) tuple where method indicates extraction success
        """
        prediction = "None"
        reasoning = ""
        method = "failure"
        
        # Try primary extraction method
        extracted = _extract_jsons(text)
        if extracted:
            method = "success"
        else:
            # Fallback to regex extraction
            extracted = _extract_json_with_regex(text)
            if extracted:
                method = "fallback"
        
        if extracted:
            last_json = extracted[-1]
            if "response" in last_json:
                prediction = str(last_json["response"])
            if "reasoning" in last_json:
                reasoning = str(last_json["reasoning"])
        
        # Normalize the prediction
        prediction = _normalize_prediction(prediction)
        
        return prediction, reasoning, method

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

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
        
        instruction = self._build_grading_prompt(inputs, is_retry=False)
        
        msg_history = []
        prediction = "None"
        reasoning = ""
        
        # Track all attempts for better debugging
        attempt_details = []
        
        # Retry loop for robust extraction
        for attempt in range(self.max_retries):
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=msg_history if attempt > 0 else [],
                )
                
                last_text = msg_history[-1]["text"] if msg_history else ""
                prediction, reasoning, method = self._extract_prediction(last_text)
                
                # Update statistics
                self._extraction_stats[method] += 1
                
                # Log detailed attempt info
                attempt_info = {
                    "attempt": attempt + 1,
                    "method": method,
                    "prediction": prediction,
                    "response_length": len(last_text),
                    "has_reasoning": bool(reasoning),
                }
                attempt_details.append(attempt_info)
                
                if prediction != "None":
                    self.log_fn(f"[Attempt {attempt + 1}] Successfully extracted prediction: {prediction} (method: {method})")
                    if reasoning:
                        self.log_fn(f"[Attempt {attempt + 1}] Reasoning preview: {reasoning[:200]}...")
                    # Log extraction stats summary
                    self.log_fn(f"Extraction stats: {dict(self._extraction_stats)}")
                    break
                else:
                    self.log_fn(f"[Attempt {attempt + 1}] Failed to extract prediction (method: {method})")
                    # Log a preview of the problematic response for debugging
                    preview = last_text[:300].replace('\n', ' ')
                    self.log_fn(f"[Attempt {attempt + 1}] Response preview: {preview}...")
                    
                    # Add feedback for retry
                    if attempt < self.max_retries - 1:
                        instruction = self._build_grading_prompt(inputs, is_retry=True)
                    
            except Exception as e:
                error_msg = f"[Attempt {attempt + 1}] Error: {type(e).__name__}: {str(e)[:100]}"
                self.log_fn(error_msg)
                attempt_details.append({"attempt": attempt + 1, "error": str(e)})
                if attempt == self.max_retries - 1:
                    prediction = f"Error: {e}"
        
        # Log summary of all attempts if we failed
        if prediction == "None" or str(prediction).startswith("Error:"):
            self.log_fn(f"All {self.max_retries} attempts failed. Details: {attempt_details}")
        
        return str(prediction), msg_history
