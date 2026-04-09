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
    
    Tries multiple approaches in order:
    1. Standard <json>...</json> blocks
    2. Markdown code blocks with json
    3. Raw JSON objects in text
    4. JSON-like structures with common grading keys
    """
    # Strategy 1: Standard <json> tags
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
    
    # Strategy 3: Look for JSON-like structures with "response" key
    json_pattern = r'\{\s*"response"\s*:[^\}]+\}'
    for match in re.finditer(json_pattern, text, re.DOTALL):
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
    
    # Strategy 4: Look for JSON-like structures with "score" key
    score_pattern = r'\{\s*"score"\s*:[^\}]+\}'
    for match in re.finditer(score_pattern, text, re.DOTALL):
        try:
            data = json.loads(match.group(0))
            # Map score to response for compatibility
            if "score" in data and "response" not in data:
                data["response"] = 1 if data["score"] >= 0.5 else 0
            return data
        except json.JSONDecodeError:
            continue
    
    return None


def _validate_inputs(inputs: dict) -> tuple[bool, str]:
    """Validate that all required fields are present and non-empty.
    
    Args:
        inputs: Dictionary containing problem inputs
        
    Returns:
        (is_valid, error_message)
    """
    required_fields = ["problem", "solution", "grading_guidelines", "student_answer"]
    
    for field in required_fields:
        if field not in inputs:
            return False, f"Missing required field: {field}"
        if not inputs[field] or not str(inputs[field]).strip():
            return False, f"Empty required field: {field}"
    
    return True, ""


def _normalize_prediction(prediction: Any) -> str:
    """Normalize prediction to "0" or "1".
    
    Handles various input types and formats.
    
    Args:
        prediction: Raw prediction value
        
    Returns:
        Normalized prediction string ("0" or "1")
    """
    # Handle numeric types
    if isinstance(prediction, (int, float)):
        return "1" if prediction >= 0.5 else "0"
    
    # Handle string types
    if isinstance(prediction, str):
        prediction = prediction.strip().lower()
        if prediction in ["1", "true", "correct", "yes", "right"]:
            return "1"
        elif prediction in ["0", "false", "incorrect", "no", "wrong"]:
            return "0"
        # Try to parse as number
        try:
            val = float(prediction)
            return "1" if val >= 0.5 else "0"
        except ValueError:
            pass
    
    # Handle boolean
    if isinstance(prediction, bool):
        return "1" if prediction else "0"
    
    # Default to "0" for unparseable values
    return "0"


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning.
    
    Features:
    - Input validation to ensure required fields are present
    - Confidence scoring for uncertain predictions
    - Support for partial credit (0.0-1.0 scale)
    - Detailed logging of grading decisions
    - Robust JSON extraction with multiple fallback strategies
    """

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.max_retries = 3
        self.enable_confidence = True
        self.enable_partial_credit = True

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        start_time = time.time()
        
        # Validate inputs
        is_valid, error_msg = _validate_inputs(inputs)
        if not is_valid:
            self.log_fn(f"Input validation failed: {error_msg}")
            return "0", []
        
        # Extract key fields for better prompting
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        # Log problem summary
        problem_preview = problem[:100].replace('\n', ' ') if problem else ""
        self.log_fn(f"Grading problem: {problem_preview}...")

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
5. Identify any errors or misconceptions in the student's work
6. Determine the correctness of the student's answer

Respond in JSON format with the following schema:
<json>
{{
    "reasoning": "Your detailed step-by-step analysis here",
    "response": 1 or 0,
    "confidence": 0.0 to 1.0,
    "score": 0.0 to 1.0
}}
</json>

Field descriptions:
- "response": Must be 1 (correct) or 0 (incorrect) - binary correctness
- "confidence": Your confidence in this grading decision (0.0 = uncertain, 1.0 = certain)
- "score": Partial credit score from 0.0 to 1.0 (1.0 = fully correct, 0.0 = completely wrong)

Note: If confidence is low (< 0.7), the system may flag this for human review."""

        msg_history = []
        
        # Try with retries for robustness
        for attempt in range(self.max_retries):
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=msg_history if msg_history else [],
                )
                
                # Extract prediction from JSON using flexible extraction
                extracted = _extract_json_flexible(msg_history[-1]["text"])
                
                if extracted:
                    # Log the extracted data for debugging
                    confidence = extracted.get("confidence", "N/A")
                    score = extracted.get("score", "N/A")
                    self.log_fn(f"Extracted - confidence: {confidence}, score: {score}")
                    
                    # Get the prediction
                    if "response" in extracted:
                        prediction = extracted["response"]
                    elif "score" in extracted:
                        # Use score to determine response
                        prediction = 1 if extracted["score"] >= 0.5 else 0
                    else:
                        self.log_fn(f"No prediction field found, retrying...")
                        continue
                    
                    # Normalize and validate prediction
                    normalized = _normalize_prediction(prediction)
                    
                    elapsed = time.time() - start_time
                    self.log_fn(f"Grading complete in {elapsed:.2f}s - prediction: {normalized}")
                    
                    return normalized, msg_history
                else:
                    self.log_fn(f"No valid JSON found in response (attempt {attempt + 1}), retrying...")
                    
            except Exception as e:
                self.log_fn(f"Error on attempt {attempt + 1}: {e}")
                if attempt == self.max_retries - 1:
                    break
        
        # Fallback: return "0" if all retries failed
        elapsed = time.time() - start_time
        self.log_fn(f"All retries failed after {elapsed:.2f}s, returning default prediction 0")
        return "0", msg_history if msg_history else []
