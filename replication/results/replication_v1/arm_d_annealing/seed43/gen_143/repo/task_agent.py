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
    Includes enhanced error recovery for malformed JSON.
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
            # Log unclosed tag for debugging
            extraction_attempts.append({"error": "unclosed_json_tag", "position": start})
            break
        
        inner = text[start + 6:end].strip()
        search_from = end + 7
        
        # Skip empty content
        if not inner:
            extraction_attempts.append({"error": "empty_json_content", "position": start})
            continue
        
        # Try to parse the inner content as JSON
        parsed = False
        try:
            results.append(json.loads(inner))
            parsed = True
        except json.JSONDecodeError as e:
            extraction_attempts.append({
                "error": "json_decode_error", 
                "position": start, 
                "details": str(e)[:100]
            })
            
            # Try to extract JSON by finding balanced braces
            json_obj = _extract_balanced_json(inner)
            if json_obj:
                results.append(json_obj)
                parsed = True
            else:
                # Try to fix common JSON issues
                fixed = _fix_common_json_issues(inner)
                if fixed:
                    try:
                        results.append(json.loads(fixed))
                        parsed = True
                    except json.JSONDecodeError:
                        pass
        
        if not parsed:
            extraction_attempts.append({"error": "all_recovery_failed", "position": start})
    
    # Log extraction summary for debugging if no results
    if not results and extraction_attempts:
        logger.debug(f"JSON extraction failed. Attempts: {extraction_attempts}")
    
    return results or None


def _fix_common_json_issues(text: str) -> str | None:
    """Attempt to fix common JSON formatting issues.
    
    Returns fixed JSON string or None if cannot fix.
    Enhanced to handle more edge cases including control characters,
    unescaped quotes, and nested structure issues.
    """
    if not text or not text.strip():
        return None
    
    original_text = text
    text = text.strip()
    
    # Fix 1: Remove control characters (except common whitespace)
    text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\r\t')
    
    # Fix 2: Remove trailing commas before closing braces/brackets
    text = re.sub(r',(\s*[}\]])', r'\1', text)
    
    # Fix 3: Add quotes around unquoted keys (simple cases)
    # Match word characters followed by colon, not already in quotes
    text = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)(\s*:)', r'\1"\2"\3', text)
    
    # Fix 4: Handle unescaped quotes inside string values
    # This is a simplified fix - replace problematic patterns
    def fix_unescaped_quotes(match):
        # Get the string content and escape any unescaped quotes
        content = match.group(1)
        # Escape quotes that aren't already escaped
        content = re.sub(r'(?<!\\)"', r'\\"', content)
        return f'"{content}"'
    
    # Try to fix unescaped quotes in string values (basic pattern)
    text = re.sub(r'"([^"]*(?:[^\\"][^"]*)*)"', fix_unescaped_quotes, text)
    
    # Fix 5: Ensure the text starts with { and ends with }
    if not text.startswith('{'):
        start_idx = text.find('{')
        if start_idx != -1:
            text = text[start_idx:]
    
    if not text.endswith('}'):
        end_idx = text.rfind('}')
        if end_idx != -1:
            text = text[:end_idx+1]
    
    # Fix 6: Handle missing closing braces by finding the last complete object
    if text.startswith('{'):
        # Try to find a balanced JSON object
        brace_depth = 0
        last_valid_end = -1
        in_string = False
        escape_next = False
        
        for i, char in enumerate(text):
            if escape_next:
                escape_next = False
                continue
            if char == '\\' and in_string:
                escape_next = True
                continue
            if char == '"' and not in_string:
                in_string = True
                continue
            if char == '"' and in_string:
                in_string = False
                continue
            if not in_string:
                if char == '{':
                    brace_depth += 1
                elif char == '}':
                    brace_depth -= 1
                    if brace_depth == 0:
                        last_valid_end = i
        
        if last_valid_end != -1 and last_valid_end < len(text) - 1:
            # Truncate to the last valid JSON object
            text = text[:last_valid_end + 1]
    
    # Validate the result
    if not text.startswith('{') or not text.endswith('}'):
        return None
    
    # Try to validate by parsing
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        # If we made changes but still can't parse, return None
        if text != original_text.strip():
            logger.debug(f"JSON fix attempted but failed to parse. Text: {text[:200]}...")
        return None


def _extract_balanced_json(text: str) -> dict | None:
    """Extract a JSON object by tracking brace depth.
    
    This handles cases where the JSON content contains nested braces
    that might confuse simple regex-based extraction.
    Enhanced to handle multiple JSON objects and edge cases.
    """
    results = []
    start_idx = -1
    brace_depth = 0
    in_string = False
    escape_next = False
    
    for i, char in enumerate(text):
        if escape_next:
            escape_next = False
            continue
        if char == '\\' and in_string:
            escape_next = True
            continue
        if char == '"' and not in_string:
            in_string = True
            continue
        if char == '"' and in_string:
            in_string = False
            continue
        if not in_string:
            if char == '{':
                if brace_depth == 0:
                    start_idx = i
                brace_depth += 1
            elif char == '}':
                brace_depth -= 1
                if brace_depth == 0 and start_idx != -1:
                    try:
                        json_obj = json.loads(text[start_idx:i+1])
                        if isinstance(json_obj, dict):
                            results.append(json_obj)
                    except json.JSONDecodeError:
                        # Try to fix common issues before giving up
                        fixed = _fix_common_json_issues(text[start_idx:i+1])
                        if fixed:
                            try:
                                json_obj = json.loads(fixed)
                                if isinstance(json_obj, dict):
                                    results.append(json_obj)
                            except json.JSONDecodeError:
                                pass
                    start_idx = -1
    
    # Return the last valid JSON object found (most likely to be the response)
    if results:
        return results[-1]
    return None


def _extract_json_with_regex(text: str) -> list[dict] | None:
    """Fallback JSON extraction using multiple strategies for malformed responses.
    
    Enhanced with additional strategies for edge cases and better error handling.
    """
    results = []
    extraction_log = []
    
    # Strategy 1: Try to find JSON objects in code blocks
    code_block_pattern = r'```(?:json)?\s*(\{[\s\S]*?\})\s*```'
    matches = re.findall(code_block_pattern, text, re.DOTALL)
    for i, match in enumerate(matches):
        try:
            parsed = json.loads(match)
            if isinstance(parsed, dict):
                results.append(parsed)
                extraction_log.append(f"code_block_{i}:success")
        except json.JSONDecodeError:
            balanced = _extract_balanced_json(match)
            if balanced:
                results.append(balanced)
                extraction_log.append(f"code_block_{i}:balanced")
            else:
                extraction_log.append(f"code_block_{i}:failed")
    
    # Strategy 2: Look for any JSON-like structure with balanced braces
    if not results:
        balanced = _extract_balanced_json(text)
        if balanced:
            results.append(balanced)
            extraction_log.append("balanced_braces:success")
        else:
            extraction_log.append("balanced_braces:failed")
    
    # Strategy 3: Extract key-value pairs for response and reasoning
    if not results:
        # More flexible pattern to handle escaped quotes and multiline content
        response_patterns = [
            r'"response"\s*:\s*"([^"]*(?:\\.[^"]*)*)"',
            r'"response"\s*:\s*\'([^\']*(?:\\.[^\']*)*)\'',
            r'response["\']?\s*[:=]\s*["\']?([^"\'\n,}]+)',
        ]
        reasoning_patterns = [
            r'"reasoning"\s*:\s*"([^"]*(?:\\.[^"]*)*)"',
            r'"reasoning"\s*:\s*\'([^\']*(?:\\.[^\']*)*)\'',
            r'reasoning["\']?\s*[:=]\s*["\']?([^"\'\n,}]+)',
        ]
        
        extracted = {}
        
        for pattern in response_patterns:
            response_match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if response_match:
                extracted["response"] = response_match.group(1).replace('\\"', '"').replace('\\n', '\n').strip()
                extraction_log.append(f"regex_response:success")
                break
        
        for pattern in reasoning_patterns:
            reasoning_match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if reasoning_match:
                extracted["reasoning"] = reasoning_match.group(1).replace('\\"', '"').replace('\\n', '\n').strip()
                extraction_log.append(f"regex_reasoning:success")
                break
        
        if extracted:
            results.append(extracted)
    
    # Strategy 4: Look for plain text responses without JSON structure
    if not results:
        text_lower = text.lower()
        
        # Expanded list of grade patterns
        grade_patterns = [
            (r'\bgrade\s*[:=]\s*(correct|partial|incorrect)\b', "grade_explicit"),
            (r'\bthe\s+answer\s+is\s+(correct|partial|incorrect)\b', "answer_is"),
            (r'\bthis\s+is\s+(correct|partial|incorrect)\b', "this_is"),
            (r'\b(correct|partial|incorrect)\s+answer\b', "answer_type"),
            (r'\bmark\s*[:=]\s*(correct|partial|incorrect)\b', "mark_explicit"),
            (r'\bassessment\s*[:=]\s*(correct|partial|incorrect)\b', "assessment_explicit"),
            (r'\bverdict\s*[:=]\s*(correct|partial|incorrect)\b', "verdict_explicit"),
        ]
        
        for pattern, pattern_name in grade_patterns:
            match = re.search(pattern, text_lower)
            if match:
                grade = match.group(1).capitalize()
                results.append({"response": grade})
                extraction_log.append(f"plain_text_{pattern_name}:success")
                break
    
    # Strategy 5: Look for standalone grade words at the beginning or end of text
    if not results:
        text_clean = re.sub(r'[^\w\s]', ' ', text_lower).strip()
        words = text_clean.split()
        
        # Check first and last few words
        check_words = words[:5] + words[-5:] if len(words) > 10 else words
        
        for word in check_words:
            if word in ["correct", "partial", "incorrect"]:
                results.append({"response": word.capitalize()})
                extraction_log.append(f"standalone_word:success")
                break
    
    # Log extraction summary for debugging
    if not results:
        logger.debug(f"Regex extraction failed. Log: {extraction_log}")
    
    return results or None


def _normalize_prediction(prediction: str) -> str:
    """Normalize prediction string for consistent comparison.
    
    Handles common variations like case differences, extra whitespace,
    punctuation differences, and partial matches.
    Enhanced to handle numeric scores and more edge cases.
    """
    if not prediction:
        return "None"
    
    # Strip whitespace and normalize case
    normalized = prediction.strip()
    
    # Handle numeric scores (0-100 scale or 0-1 scale)
    try:
        # Try to parse as a number
        numeric_value = float(normalized)
        # Convert numeric score to grade
        if numeric_value >= 0.8 or numeric_value >= 80:
            return "Correct"
        elif numeric_value >= 0.4 or numeric_value >= 40:
            return "Partial"
        else:
            return "Incorrect"
    except (ValueError, TypeError):
        pass
    
    # Handle common grade variations with exact matches
    grade_map = {
        "correct": "Correct",
        "partial": "Partial",
        "partially correct": "Partial",
        "partially": "Partial",
        "incorrect": "Incorrect",
        "wrong": "Incorrect",
        "right": "Correct",
        "true": "Correct",
        "false": "Incorrect",
        "yes": "Correct",
        "no": "Incorrect",
        "full": "Correct",
        "full marks": "Correct",
        "full credit": "Correct",
        "half": "Partial",
        "half marks": "Partial",
        "half credit": "Partial",
        "zero": "Incorrect",
        "no marks": "Incorrect",
        "fail": "Incorrect",
        "pass": "Correct",
    }
    
    lower_pred = normalized.lower()
    if lower_pred in grade_map:
        return grade_map[lower_pred]
    
    # Handle partial matches and variations with punctuation
    # Remove punctuation for matching
    clean_pred = re.sub(r'[^\w\s]', '', lower_pred).strip()
    
    # Check for compound phrases first
    compound_patterns = [
        ("partially correct", "Partial"),
        ("partial credit", "Partial"),
        ("half correct", "Partial"),
        ("mostly correct", "Partial"),
        ("mostly wrong", "Partial"),
        ("partially wrong", "Partial"),
    ]
    
    for pattern, grade in compound_patterns:
        if pattern in clean_pred:
            return grade
    
    # Check for partial matches
    if "partial" in clean_pred or "partially" in clean_pred:
        return "Partial"
    if any(word in clean_pred for word in ["correct", "right", "true", "yes", "valid", "accurate"]):
        return "Correct"
    if any(word in clean_pred for word in ["incorrect", "wrong", "false", "no", "error", "invalid", "inaccurate", "mistake"]):
        return "Incorrect"
    
    # If we can't normalize, return the original (trimmed)
    return normalized if normalized else "None"


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

    def _build_grading_prompt(self, inputs: dict, is_retry: bool = False) -> str:
        """Build a structured prompt for the grading task with chain-of-thought."""
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
        
        # Validate prediction is one of the expected values
        valid_predictions = {"Correct", "Partial", "Incorrect"}
        if prediction not in valid_predictions and prediction != "None":
            # Try to map to valid prediction
            prediction = _normalize_prediction(prediction)
            if prediction not in valid_predictions:
                logger.warning(f"Unexpected prediction value: {prediction}. Setting to None.")
                prediction = "None"
        
        return prediction, reasoning, method

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
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
