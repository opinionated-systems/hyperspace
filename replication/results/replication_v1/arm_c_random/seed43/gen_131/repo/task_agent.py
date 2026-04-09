"""
Task agent: solves a given task with enhanced reasoning for IMO grading.

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


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks with enhanced robustness.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    Also attempts to extract JSON from markdown code blocks as fallback.
    Includes additional heuristics for malformed JSON common in LLM outputs.
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
            # Try to fix common JSON issues
            fixed = _attempt_json_repair(inner)
            if fixed:
                results.append(fixed)
            continue
    
    # Fallback: try to extract JSON from markdown code blocks
    if not results:
        pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                results.append(json.loads(match.strip()))
            except json.JSONDecodeError:
                fixed = _attempt_json_repair(match.strip())
                if fixed:
                    results.append(fixed)
                continue
    
    # Final fallback: try to find any JSON-like structure with reasoning/response fields
    if not results:
        try:
            # Look for patterns like {"reasoning": ..., "response": ...}
            pattern = r'\{\s*"reasoning"[^}]+"response"[^}]+\}'
            match = re.search(pattern, text, re.DOTALL)
            if match:
                try:
                    results.append(json.loads(match.group()))
                except json.JSONDecodeError:
                    fixed = _attempt_json_repair(match.group())
                    if fixed:
                        results.append(fixed)
        except (json.JSONDecodeError, AttributeError):
            pass
    
    # Ultra fallback: extract any complete JSON object
    if not results:
        # Find all potential JSON objects
        pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                parsed = json.loads(match)
                if isinstance(parsed, dict):
                    results.append(parsed)
            except json.JSONDecodeError:
                fixed = _attempt_json_repair(match)
                if fixed:
                    results.append(fixed)
    
    return results or None


def _attempt_json_repair(text: str) -> dict | None:
    """Attempt to repair common JSON formatting issues in LLM outputs.
    
    Fixes:
    - Trailing commas before closing braces/brackets
    - Single quotes instead of double quotes
    - Unescaped newlines in strings
    - Missing quotes around keys
    """
    try:
        # Try original first
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Fix 1: Remove trailing commas before } or ]
    fixed = re.sub(r',\s*(\}|\])', r'\1', text)
    
    # Fix 2: Replace single quotes with double quotes (carefully)
    # Only replace quotes that are not inside strings
    fixed = re.sub(r"(?<!\\)'", '"', fixed)
    
    # Fix 3: Escape unescaped newlines in string values
    # This is a simplified approach - replace newlines between quotes
    fixed = re.sub(r'("[^"]*?)\n([^"]*?")', r'\1\\n\2', fixed)
    
    # Fix 4: Add quotes around unquoted keys (simple heuristic)
    fixed = re.sub(r'(\{|,\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', fixed)
    
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        return None


def _validate_inputs(inputs: dict) -> tuple[bool, str]:
    """Validate that all required fields are present in inputs.
    
    Returns:
        (is_valid, error_message)
    """
    required_fields = ["domain", "problem", "solution", "grading_guidelines", "student_answer"]
    missing = [f for f in required_fields if not inputs.get(f)]
    
    if missing:
        return False, f"Missing required fields: {', '.join(missing)}"
    
    # Validate that fields are non-empty strings
    empty_fields = [f for f in required_fields if isinstance(inputs.get(f), str) and not inputs.get(f).strip()]
    if empty_fields:
        return False, f"Empty required fields: {', '.join(empty_fields)}"
    
    return True, ""


def _normalize_score(score: str | int | float) -> str:
    """Normalize a score to a standard format.
    
    Handles various score formats:
    - Numeric scores (0-7 for IMO)
    - Fractional scores (e.g., "3/7", "0.5")
    - Text descriptions with numbers
    - Returns "0" if no valid score found
    """
    if score is None:
        return "0"
    
    # If already a number, convert to string
    if isinstance(score, (int, float)):
        # Clamp to valid IMO range [0, 7]
        clamped = max(0, min(7, float(score)))
        return str(int(clamped)) if clamped == int(clamped) else str(clamped)
    
    score_str = str(score).strip()
    
    # Try to extract from fraction format like "3/7" or "3 out of 7"
    fraction_match = re.search(r'(\d+(?:\.\d+)?)\s*/\s*\d+', score_str)
    if fraction_match:
        return fraction_match.group(1)
    
    # Try to find the last number in the string (often the final score)
    numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', score_str)
    if numbers:
        # Return the last number found, clamped to [0, 7]
        try:
            val = float(numbers[-1])
            clamped = max(0, min(7, val))
            return str(int(clamped)) if clamped == int(clamped) else str(clamped)
        except ValueError:
            pass
    
    # Check for special cases
    score_lower = score_str.lower()
    if any(word in score_lower for word in ["full", "complete", "correct", "perfect", "7"]):
        return "7"
    if any(word in score_lower for word in ["zero", "none", "incorrect", "wrong", "0"]):
        return "0"
    if "partial" in score_lower:
        # Try to find partial credit amount
        partial_nums = re.findall(r'\b(\d+)\b', score_str)
        if partial_nums:
            return partial_nums[0]
    
    return "0"


class TaskAgent:
    """Task agent that solves IMO grading problems with enhanced reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "", max_retries: int = 3) -> None:
        self.model = model
        self.log_fn = logger.info
        self.max_retries = max_retries

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with chain-of-thought reasoning.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Validate inputs first
        is_valid, error_msg = _validate_inputs(inputs)
        if not is_valid:
            self.log_fn(f"Input validation failed: {error_msg}")
            return f"Error: {error_msg}", []
        
        # Extract fields for better structured prompting
        domain = inputs.get("domain", "")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = f"""You are an expert IMO (International Mathematical Olympiad) grader. Your task is to evaluate a student's solution to a mathematical problem.

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

## Your Task

Please evaluate the student's answer following these steps:

1. **Understand the Problem**: Identify what the problem is asking and the key mathematical concepts involved.

2. **Analyze the Official Solution**: Understand the expected approach and the key steps required for a correct solution.

3. **Review Grading Guidelines**: Note the specific criteria for awarding points (partial or full).

4. **Evaluate Student's Answer**: 
   - Check if the student correctly identified the approach
   - Verify each step of their reasoning
   - Identify any errors, gaps, or incorrect assumptions
   - Note any creative or alternative valid approaches

5. **Assign Score**: Based on the grading guidelines, assign an appropriate score from 0 to 7. IMO problems are scored 0-7 points.
   - 7 points: Complete, correct solution
   - 6 points: Minor flaw in an otherwise correct solution
   - 5 points: Significant progress with some gaps
   - 3-4 points: Partial progress
   - 1-2 points: Some relevant ideas but major gaps
   - 0 points: No significant progress or completely wrong

Respond ONLY in JSON format with the following schema (no markdown outside the JSON tags):
<json>
{{
    "reasoning": "Your detailed chain-of-thought analysis here. Explain your evaluation process.",
    "evaluation": "Summary of what the student did correctly/incorrectly",
    "response": "The final score as a single integer from 0 to 7"
}}
</json>

IMPORTANT: The "response" field MUST contain ONLY a single integer from 0 to 7 (e.g., "7", "3", "0"). Do not include any other text in this field."""

        all_msg_histories = []
        
        # Retry loop for robustness
        for attempt in range(self.max_retries + 1):
            try:
                # Add retry-specific guidance on subsequent attempts
                current_instruction = instruction
                if attempt > 0:
                    current_instruction += f"\n\n[Attempt {attempt + 1}/{self.max_retries + 1}] Please ensure your response follows the exact JSON format specified above."
                
                response, msg_history, info = get_response_from_llm(
                    msg=current_instruction,
                    model=self.model,
                    msg_history=[],
                )
                
                all_msg_histories.extend(msg_history)

                # Extract prediction from JSON
                prediction = self._extract_prediction(msg_history)
                
                if prediction != "None" and prediction != "Error":
                    # Normalize the score before returning
                    normalized = _normalize_score(prediction)
                    self.log_fn(f"Extracted score: {prediction} -> Normalized: {normalized}")
                    return normalized, all_msg_histories
                
                # If extraction failed and we have retries left, try again
                if attempt < self.max_retries:
                    self.log_fn(f"Attempt {attempt + 1}: Failed to extract valid prediction, retrying...")
                    time.sleep(0.5 * (attempt + 1))  # Increasing delay before retry
                    
            except Exception as e:
                self.log_fn(f"Attempt {attempt + 1}: LLM call failed: {e}")
                if attempt < self.max_retries:
                    time.sleep(1.0 * (attempt + 1))  # Increasing delay on error
                else:
                    return f"Error: {e}", all_msg_histories
        
        # All retries exhausted, try one final extraction attempt
        if all_msg_histories:
            final_prediction = self._extract_prediction(all_msg_histories)
            if final_prediction != "None":
                normalized = _normalize_score(final_prediction)
                return normalized, all_msg_histories
        
        return "0", all_msg_histories  # Return 0 as safe default instead of "None"
    
    def _extract_prediction(self, msg_history: list[dict]) -> str:
        """Extract prediction from message history with multiple fallback strategies.
        
        Args:
            msg_history: List of message dicts from LLM conversation
            
        Returns:
            Extracted prediction string or "None" if extraction fails
        """
        if not msg_history:
            return "None"
        
        try:
            # Try all messages in reverse order (most recent first)
            for msg in reversed(msg_history):
                text = msg.get("text", "")
                if not text:
                    continue
                    
                extracted = _extract_jsons(text)
                
                if extracted:
                    # Try to get response from each valid JSON block, starting from last
                    for last_json in reversed(extracted):
                        # Primary: look for "response" field
                        if "response" in last_json:
                            response_val = last_json["response"]
                            # Validate it's a reasonable score
                            normalized = _normalize_score(response_val)
                            if normalized != "0" or str(response_val) in ["0", 0, "0.0"]:
                                return str(response_val)
                        
                        # Secondary: look for "evaluation" field with score
                        if "evaluation" in last_json:
                            eval_val = last_json["evaluation"]
                            normalized = _normalize_score(eval_val)
                            if normalized != "0" or str(eval_val) in ["0", 0, "0.0"]:
                                return str(eval_val)
                        
                        # Tertiary: look for any numeric-looking field
                        for key in ["score", "grade", "points", "result", "mark", "value"]:
                            if key in last_json:
                                return str(last_json[key])
                        
                        # If no recognized field but JSON has only one value, use it
                        if len(last_json) == 1:
                            return str(list(last_json.values())[0])
                
                # No JSON found in this message, try to extract a number from the text
                # Look for patterns like "Score: 5" or "Final score: 3/7"
                score_patterns = [
                    r'[Ss]core\s*[:=]\s*(\d+(?:\.\d+)?)',
                    r'[Gg]rade\s*[:=]\s*(\d+(?:\.\d+)?)',
                    r'[Ff]inal\s*(?:score|grade)\s*[:=]\s*(\d+(?:\.\d+)?)',
                    r'[Aa]ssign\s*(?:a\s*)?(?:score|grade)\s*[:=]\s*(\d+(?:\.\d+)?)',
                    r'\b(\d+)\s*/\s*7\b',  # Pattern like "5/7"
                ]
                for pattern in score_patterns:
                    match = re.search(pattern, text)
                    if match:
                        return match.group(1)
                
                # Last resort: find any number 0-7 in the text
                numbers = re.findall(r'\b([0-7])\b', text)
                if numbers:
                    return numbers[-1]  # Return last number found (likely the score)
                
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
        
        return "None"
