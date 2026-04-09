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
            continue
    return results or None


def _extract_json_with_retry(text: str, max_retries: int = 3) -> list[dict] | None:
    """Extract JSON with multiple fallback strategies.
    
    Tries multiple extraction methods:
    1. Standard <json>...</json> blocks
    2. JSON code blocks ```json...```
    3. Raw JSON objects in text (with nested brace support)
    """
    # Try standard extraction first
    results = _extract_jsons(text)
    if results:
        return results
    
    # Try JSON code blocks
    results = []
    json_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    matches = re.findall(json_block_pattern, text, re.DOTALL)
    for match in matches:
        try:
            results.append(json.loads(match.strip()))
        except json.JSONDecodeError:
            continue
    
    if results:
        return results
    
    # Try to find raw JSON objects with proper nested brace handling
    # Use a stack-based approach to find balanced braces
    results = []
    i = 0
    while i < len(text):
        if text[i] == '{':
            # Found start of potential JSON object
            start = i
            brace_count = 1
            i += 1
            while i < len(text) and brace_count > 0:
                if text[i] == '{':
                    brace_count += 1
                elif text[i] == '}':
                    brace_count -= 1
                i += 1
            if brace_count == 0:
                # Found a balanced object
                try:
                    obj = json.loads(text[start:i])
                    results.append(obj)
                except json.JSONDecodeError:
                    pass
        else:
            i += 1
    
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

    def _build_grading_prompt(self, inputs: dict) -> str:
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
- The "response" field must contain only the final grade/score
- Use the "reasoning" field to show your work
- Be precise and follow the grading guidelines exactly
- If the student answer is empty or nonsensical, assign the minimum grade"""

    def _try_extract_prediction(self, text: str) -> tuple[str, str | None]:
        """Try to extract prediction from response text.
        
        Returns:
            (prediction, reasoning)
        """
        try:
            extracted = _extract_json_with_retry(text)
            if extracted:
                # Find the first valid JSON with a "response" field
                for obj in extracted:
                    if isinstance(obj, dict) and "response" in obj:
                        prediction = obj.get("response", "None")
                        reasoning = obj.get("reasoning")
                        # Validate prediction is not empty or None
                        if prediction is not None and str(prediction).strip():
                            return str(prediction), reasoning
                # Fallback to last object if no valid response found
                last = extracted[-1]
                if isinstance(last, dict):
                    prediction = last.get("response", "None")
                    reasoning = last.get("reasoning")
                    return str(prediction) if prediction is not None else "None", reasoning
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
        
        return "None", None

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
        last_error = None
        
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
                    break
                else:
                    last_error = "Failed to extract valid prediction from response"
                    self.log_fn(f"Attempt {attempt + 1}: {last_error}")
                
                # If extraction failed, add a follow-up message asking for proper format
                if attempt < self.max_retries - 1:
                    instruction = (
                        "Your previous response did not contain a valid JSON with 'response' and 'reasoning' fields. "
                        "Please respond in the required JSON format:\n\n"
                        "<json>\n"
                        '{\n'
                        '    "reasoning": "Your detailed step-by-step analysis",\n'
                        '    "response": "The final grade/score"\n'
                        '}\n'
                        "</json>"
                    )
                    
            except Exception as e:
                last_error = str(e)
                self.log_fn(f"Attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    break
        
        # Log final result
        if prediction == "None" and last_error:
            self.log_fn(f"All {self.max_retries} attempts failed. Last error: {last_error}")
        
        return str(prediction), msg_history
