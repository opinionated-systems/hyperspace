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
    4. Bracket matching for nested JSON objects
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
    
    # Strategy 4: Bracket matching for any JSON object with "reasoning" and "response" keys
    try:
        # Find all potential JSON starting points
        for start in re.finditer(r'\{', text):
            idx = start.start()
            # Try to find matching closing brace
            brace_count = 0
            for i, char in enumerate(text[idx:], start=idx):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        # Found a complete JSON object
                        try:
                            candidate = json.loads(text[idx:i+1])
                            if "response" in candidate:
                                return candidate
                        except json.JSONDecodeError:
                            break
                        break
    except Exception:
        pass
    
    return None


def _validate_prediction(prediction: Any) -> int | None:
    """Validate and normalize prediction value.
    
    Returns 0 or 1 if valid, None otherwise.
    """
    if prediction is None:
        return None
    
    # Handle string values
    if isinstance(prediction, str):
        prediction = prediction.strip().lower()
        if prediction in ('0', 'false', 'incorrect', 'wrong', 'no'):
            return 0
        if prediction in ('1', 'true', 'correct', 'right', 'yes'):
            return 1
        # Try to parse as int
        try:
            prediction = int(prediction)
        except ValueError:
            return None
    
    # Handle numeric values
    if isinstance(prediction, (int, float)):
        if prediction == 0:
            return 0
        if prediction == 1:
            return 1
    
    return None


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.max_retries = 3
        self.total_calls = 0
        self.successful_calls = 0

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        self.total_calls += 1
        
        # Extract key fields for better prompting
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        # Log problem metadata for debugging
        problem_id = inputs.get("problem_id", "unknown")
        self.log_fn(f"Processing problem {problem_id} in domain: {domain}")

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
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=msg_history,
                )
                
                # Extract prediction from JSON using flexible extraction
                response_text = msg_history[-1].get("text", "") if msg_history else ""
                extracted = _extract_json_flexible(response_text)
                
                if extracted:
                    self.log_fn(f"Successfully extracted JSON on attempt {attempt + 1}")
                    
                    if "response" in extracted:
                        prediction = extracted["response"]
                        validated = _validate_prediction(prediction)
                        
                        if validated is not None:
                            self.successful_calls += 1
                            self.log_fn(f"Valid prediction: {validated} (raw: {prediction})")
                            return str(validated), msg_history
                        else:
                            last_error = f"Invalid prediction value: {prediction}"
                            self.log_fn(f"{last_error}, retrying...")
                    else:
                        last_error = "No 'response' key in extracted JSON"
                        self.log_fn(f"{last_error}, retrying...")
                else:
                    last_error = "No valid JSON found in response"
                    self.log_fn(f"{last_error}, retrying...")
                    
            except Exception as e:
                last_error = str(e)
                self.log_fn(f"Error on attempt {attempt + 1}: {e}")
                if attempt == self.max_retries - 1:
                    break
        
        # Fallback: return "0" if all retries failed
        self.log_fn(f"All retries failed (last error: {last_error}), returning default prediction 0")
        return "0", msg_history
    
    def get_stats(self) -> dict:
        """Return agent performance statistics."""
        return {
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "success_rate": self.successful_calls / max(1, self.total_calls),
        }
