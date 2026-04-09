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

# Type aliases for clarity
PredictionType = str  # "0" or "1"
MessageHistory = list[dict[str, Any]]
InputsDict = dict[str, Any]


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


def _extract_json_flexible(text: str) -> dict | None:
    """Extract JSON with multiple fallback strategies.
    
    Tries multiple approaches in order of reliability:
    1. Standard <json>...</json> blocks (most reliable)
    2. Markdown code blocks with json
    3. Raw JSON objects in text (with nested brace handling)
    4. Look for JSON with "reasoning" and "response" keys
    5. Relaxed pattern matching for response values
    
    Args:
        text: The text to extract JSON from
        
    Returns:
        The extracted JSON dict, or None if no valid JSON found
    """
    # Strategy 1: Standard <json> tags (most reliable)
    results = _extract_jsons(text)
    if results:
        return results[-1]
    
    # Strategy 2: Markdown code blocks
    json_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    for match in re.finditer(json_block_pattern, text, re.DOTALL):
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            continue
    
    # Strategy 3: Look for JSON-like structures with balanced braces
    # This handles nested braces properly by counting
    def find_json_objects(text: str) -> list[str]:
        """Find potential JSON objects by tracking brace balance."""
        objects = []
        i = 0
        while i < len(text):
            if text[i] == '{':
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
                    objects.append(text[start:i])
            else:
                i += 1
        return objects
    
    for obj_str in find_json_objects(text):
        try:
            parsed = json.loads(obj_str)
            if isinstance(parsed, dict) and "response" in parsed:
                return parsed
        except json.JSONDecodeError:
            continue
    
    # Strategy 4: Relaxed pattern for "response" key with more flexibility
    # Matches: "response": 1, "response": 0, "response": "1", etc.
    response_pattern = r'"response"\s*[:=]\s*(\d+|"\d+"|\w+)'
    match = re.search(response_pattern, text, re.IGNORECASE)
    if match:
        # Try to extract a minimal JSON object around the response
        # Look backwards for opening brace
        start_idx = text.rfind('{', 0, match.start())
        # Look forwards for closing brace
        end_idx = text.find('}', match.end())
        if start_idx != -1 and end_idx != -1:
            try:
                candidate = text[start_idx:end_idx+1]
                parsed = json.loads(candidate)
                if "response" in parsed:
                    return parsed
            except json.JSONDecodeError:
                pass
    
    return None


def _normalize_prediction(value: Any) -> PredictionType | None:
    """Normalize a prediction value to "0" or "1".
    
    Handles various input types (int, str, bool) and validates
    that the result is a valid binary prediction.
    
    Args:
        value: The raw prediction value from JSON extraction
        
    Returns:
        Normalized "0" or "1" string, or None if invalid
    """
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        if value in (0, 1):
            return str(value)
        return None
    if isinstance(value, str):
        value = value.strip().lower()
        if value in ("0", "false", "no", "incorrect", "wrong"):
            return "0"
        if value in ("1", "true", "yes", "correct", "right"):
            return "1"
    return None


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning.
    
    This agent evaluates student answers by comparing them to correct solutions
    and following grading guidelines. It uses an LLM to perform the evaluation
    and returns a binary prediction (0 for incorrect, 1 for correct).
    """

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.max_retries = 3
        self._call_count = 0

    def forward(self, inputs: InputsDict) -> tuple[PredictionType, MessageHistory]:
        """Run the task agent on a single problem.

        Args:
            inputs: Dictionary containing:
                - domain: Subject area (e.g., "Mathematics")
                - problem: The problem statement
                - solution: The correct solution
                - grading_guidelines: Guidelines for grading
                - student_answer: The student's submitted answer

        Returns:
            Tuple of (prediction, msg_history) where:
                - prediction: "0" (incorrect) or "1" (correct)
                - msg_history: List of message dicts from the LLM conversation
        """
        self._call_count += 1
        call_id = self._call_count
        
        # Extract key fields for better prompting
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        # Truncate very long inputs to avoid token limits
        max_len = 8000
        problem = problem[:max_len] if len(problem) > max_len else problem
        solution = solution[:max_len] if len(solution) > max_len else solution
        grading_guidelines = grading_guidelines[:max_len] if len(grading_guidelines) > max_len else grading_guidelines
        student_answer = student_answer[:max_len] if len(student_answer) > max_len else student_answer

        instruction = f"""You are an expert {domain} grader evaluating student solutions.

Your task is to grade a student's answer by comparing it to the correct solution and following the grading guidelines.

PROBLEM:
{problem}

CORRECT SOLUTION:
{solution}

GRADING GUIDELINES:
{grading_guidelines}

STUDENT'S ANSWER:
{student_answer}

Think step by step:
1. Analyze what the problem is asking for
2. Review the correct solution approach
3. Compare the student's answer to the correct solution
4. Check if the student followed the grading guidelines
5. Determine if the student's answer is correct (1) or incorrect (0)

Respond in JSON format with the following schema:
<json>
{{
    "reasoning": "Your detailed step-by-step analysis here",
    "response": 1 or 0
}}
</json>

The "response" field must be either 1 (correct) or 0 (incorrect)."""

        # Try with retries for robustness
        msg_history = []
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                self.log_fn(f"[Call {call_id}] Attempt {attempt + 1}/{self.max_retries}")
                
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=msg_history if attempt > 0 else [],  # Clear history on first attempt
                )
                
                response_text = msg_history[-1]["text"] if msg_history else ""
                
                # Extract prediction from JSON using flexible extraction
                extracted = _extract_json_flexible(response_text)
                
                if extracted and "response" in extracted:
                    prediction = extracted["response"]
                    # Normalize and validate prediction
                    pred_str = _normalize_prediction(prediction)
                    if pred_str is not None:
                        self.log_fn(f"[Call {call_id}] Success on attempt {attempt + 1}: prediction={pred_str}")
                        return pred_str, msg_history
                    else:
                        last_error = f"Invalid prediction value: {prediction}"
                        self.log_fn(f"[Call {call_id}] {last_error}, retrying...")
                else:
                    last_error = "No valid JSON found in response"
                    self.log_fn(f"[Call {call_id}] {last_error}, retrying...")
                    
            except Exception as e:
                last_error = str(e)
                self.log_fn(f"[Call {call_id}] Error on attempt {attempt + 1}: {e}")
                if attempt == self.max_retries - 1:
                    break
        
        # Fallback: try to extract any numeric response from the last response
        if msg_history:
            last_response = msg_history[-1].get("text", "")
            # Look for standalone 0 or 1 in the text (but not both)
            has_one = re.search(r'\b1\b', last_response) is not None
            has_zero = re.search(r'\b0\b', last_response) is not None
            
            if has_one and not has_zero:
                self.log_fn(f"[Call {call_id}] Fallback extraction: found '1' in response")
                return "1", msg_history
            elif has_zero and not has_one:
                self.log_fn(f"[Call {call_id}] Fallback extraction: found '0' in response")
                return "0", msg_history
            
            # Try to find explicit "correct" or "incorrect" markers
            last_lower = last_response.lower()
            if "correct" in last_lower and "incorrect" not in last_lower:
                self.log_fn(f"[Call {call_id}] Fallback extraction: found 'correct' in response")
                return "1", msg_history
            elif "incorrect" in last_lower or "wrong" in last_lower:
                self.log_fn(f"[Call {call_id}] Fallback extraction: found 'incorrect' in response")
                return "0", msg_history
        
        # Final fallback: return "0" if all retries failed
        self.log_fn(f"[Call {call_id}] All retries failed ({last_error}), returning default prediction 0")
        return "0", msg_history
