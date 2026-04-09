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

from agent.llm_client import get_response_from_llm, EVAL_MODEL

logger = logging.getLogger(__name__)


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    Also attempts to extract JSON from markdown code blocks as fallback.
    Includes enhanced pattern matching for nested JSON structures.
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
                # Remove trailing commas before closing braces/brackets
                fixed = re.sub(r',(\s*[}\]])', r'\1', inner)
                # Fix single quotes to double quotes (common LLM mistake)
                fixed = re.sub(r"(?<!\\)'", '"', fixed)
                results.append(json.loads(fixed))
            except json.JSONDecodeError:
                continue
    
    # Fallback: try to extract JSON from markdown code blocks
    if not results:
        pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                results.append(json.loads(match.strip()))
            except json.JSONDecodeError:
                # Try fixing common issues
                try:
                    fixed = re.sub(r',(\s*[}\]])', r'\1', match.strip())
                    fixed = re.sub(r"(?<!\\)'", '"', fixed)
                    results.append(json.loads(fixed))
                except json.JSONDecodeError:
                    continue
    
    # Final fallback: try to find any JSON-like structure with reasoning/response fields
    if not results:
        try:
            # Look for patterns like {"reasoning": ..., "response": ...}
            # Use a more robust approach with brace counting
            pattern = r'\{\s*"reasoning"\s*:'
            match = re.search(pattern, text, re.DOTALL)
            if match:
                start_idx = match.start()
                # Find the matching closing brace by counting
                brace_count = 0
                end_idx = start_idx
                for i, char in enumerate(text[start_idx:]):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end_idx = start_idx + i + 1
                            break
                if end_idx > start_idx:
                    json_str = text[start_idx:end_idx]
                    try:
                        results.append(json.loads(json_str))
                    except json.JSONDecodeError:
                        # Try fixing common issues
                        fixed = re.sub(r',(\s*[}\]])', r'\1', json_str)
                        fixed = re.sub(r"(?<!\\)'", '"', fixed)
                        results.append(json.loads(fixed))
        except (json.JSONDecodeError, AttributeError):
            pass
    
    return results or None


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


class TaskAgent:
    """Task agent that solves IMO grading problems with enhanced reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "", max_retries: int = 2) -> None:
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

5. **Assign Score**: Based on the grading guidelines, assign an appropriate score. Be precise and justify your decision.

Respond in JSON format with the following schema:
<json>
{{
    "reasoning": "Your detailed chain-of-thought analysis here",
    "evaluation": "Summary of what the student did correctly/incorrectly",
    "response": "The final score/grade as a number or string"
}}
</json>

The "response" field should contain only the final score (e.g., "7", "3", "0", etc.) that will be used for evaluation."""

        # Retry loop for robustness
        for attempt in range(self.max_retries + 1):
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=[],
                )

                # Extract prediction from JSON
                prediction = self._extract_prediction(msg_history)
                
                if prediction != "None" and prediction != "Error":
                    return str(prediction), msg_history
                
                # If extraction failed and we have retries left, try again
                if attempt < self.max_retries:
                    self.log_fn(f"Attempt {attempt + 1}: Failed to extract valid prediction, retrying...")
                    time.sleep(0.5)  # Brief delay before retry
                    
            except Exception as e:
                self.log_fn(f"Attempt {attempt + 1}: LLM call failed: {e}")
                if attempt < self.max_retries:
                    time.sleep(1.0)  # Longer delay on error
                else:
                    return f"Error: {e}", []
        
        # All retries exhausted, return last attempt's result or error
        return "None", msg_history if 'msg_history' in locals() else []
    
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
            last_message = msg_history[-1].get("text", "")
            extracted = _extract_jsons(last_message)
            
            if extracted:
                # Try to get response from the last valid JSON block
                last_json = extracted[-1]
                
                # Primary: look for "response" field
                if "response" in last_json:
                    return str(last_json["response"])
                
                # Secondary: look for "evaluation" field
                if "evaluation" in last_json:
                    return str(last_json["evaluation"])
                
                # Tertiary: look for any numeric-looking field
                for key in ["score", "grade", "points", "result"]:
                    if key in last_json:
                        return str(last_json[key])
                
                # If no recognized field, return the whole JSON as string
                return str(last_json)
            
            # No JSON found, try to extract a number from the text
            numbers = re.findall(r'\b([0-7])\b', last_message)
            if numbers:
                return numbers[-1]  # Return last number found (likely the score)
                
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
        
        return "None"
