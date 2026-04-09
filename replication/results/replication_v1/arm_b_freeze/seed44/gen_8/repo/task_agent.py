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
            # Try to clean up common JSON issues
            try:
                # Remove trailing commas before closing braces/brackets
                cleaned = re.sub(r',(\s*[}\]])', r'\1', inner)
                # Fix single quotes to double quotes (common LLM mistake)
                cleaned = re.sub(r"(?<!\\)'", '"', cleaned)
                results.append(json.loads(cleaned))
            except json.JSONDecodeError:
                continue
    return results or None


def _extract_json_flexible(text: str) -> dict | None:
    """Extract JSON with multiple fallback strategies.
    
    Tries multiple approaches in order:
    1. Standard <json>...</json> blocks
    2. Markdown code blocks with json
    3. Raw JSON objects in text
    4. LLM output with reasoning + response pattern
    """
    # Strategy 1: Standard <json> tags
    results = _extract_jsons(text)
    if results:
        return results[-1]
    
    # Strategy 2: Markdown code blocks
    json_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    for match in re.finditer(json_block_pattern, text, re.DOTALL):
        content = match.group(1).strip()
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Try cleanup
            try:
                cleaned = re.sub(r',(\s*[}\]])', r'\1', content)
                cleaned = re.sub(r"(?<!\\)'", '"', cleaned)
                return json.loads(cleaned)
            except json.JSONDecodeError:
                continue
    
    # Strategy 3: Look for JSON-like structures with "response" key
    json_pattern = r'\{\s*"response"\s*:[^\}]+\}'
    for match in re.finditer(json_pattern, text, re.DOTALL):
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
    
    # Strategy 4: Look for complete JSON objects with nested structure
    # This handles cases where the LLM outputs valid JSON but not in tags
    brace_count = 0
    start_idx = None
    for i, char in enumerate(text):
        if char == '{':
            if brace_count == 0:
                start_idx = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start_idx is not None:
                try:
                    candidate = text[start_idx:i+1]
                    parsed = json.loads(candidate)
                    # Only return if it has the expected keys
                    if "response" in parsed:
                        return parsed
                except json.JSONDecodeError:
                    continue
    
    return None


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.max_retries = 3
        self._msg_history: list[dict] = []

    def _normalize_prediction(self, prediction: Any) -> str | None:
        """Normalize prediction to '0' or '1', or None if invalid."""
        if prediction is None:
            return None
        
        # Handle boolean values
        if isinstance(prediction, bool):
            return "1" if prediction else "0"
        
        # Handle numeric values
        if isinstance(prediction, (int, float)):
            if prediction == 1 or prediction == 1.0:
                return "1"
            elif prediction == 0 or prediction == 0.0:
                return "0"
            return None
        
        # Handle string values
        pred_str = str(prediction).strip().lower()
        if pred_str in ("1", "true", "correct", "yes", "right"):
            return "1"
        elif pred_str in ("0", "false", "incorrect", "no", "wrong"):
            return "0"
        
        return None

    def _build_prompt(self, inputs: dict) -> str:
        """Build the grading prompt from input fields."""
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        # Truncate very long inputs to stay within token limits
        max_len = 8000
        problem = problem[:max_len] if len(problem) > max_len else problem
        solution = solution[:max_len] if len(solution) > max_len else solution
        guidelines = grading_guidelines[:max_len] if len(grading_guidelines) > max_len else grading_guidelines
        answer = student_answer[:max_len] if len(student_answer) > max_len else student_answer

        return f"""You are an expert {domain} grader evaluating student solutions.

Your task is to grade a student's answer by comparing it to the correct solution and following the grading guidelines.

PROBLEM:
{problem}

CORRECT SOLUTION:
{solution}

GRADING GUIDELINES:
{guidelines}

STUDENT'S ANSWER:
{answer}

Think step by step:
1. Analyze what the problem is asking for
2. Review the correct solution approach
3. Compare the student's answer to the correct solution
4. Check if the student followed the grading guidelines
5. Determine if the student's answer is correct (1) or incorrect (0)

Respond ONLY in JSON format with the following schema:
<json>
{{
    "reasoning": "Your detailed step-by-step analysis here",
    "response": 1 or 0
}}
</json>

The "response" field must be either 1 (correct) or 0 (incorrect)."""

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        instruction = self._build_prompt(inputs)
        self._msg_history = []

        # Try with retries for robustness
        for attempt in range(self.max_retries):
            try:
                response, self._msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=self._msg_history if attempt > 0 else [],
                )
                
                # Extract prediction from JSON using flexible extraction
                response_text = self._msg_history[-1]["text"] if self._msg_history else ""
                extracted = _extract_json_flexible(response_text)
                
                if extracted:
                    self.log_fn(f"Extracted JSON keys: {list(extracted.keys())}")
                    
                    if "response" in extracted:
                        prediction = self._normalize_prediction(extracted["response"])
                        if prediction is not None:
                            return prediction, self._msg_history
                        else:
                            self.log_fn(f"Invalid prediction value: {extracted['response']}, retrying...")
                    else:
                        self.log_fn(f"No 'response' key in extracted JSON, retrying...")
                else:
                    self.log_fn(f"No valid JSON found in response, retrying...")
                    
            except Exception as e:
                self.log_fn(f"Error on attempt {attempt + 1}: {e}")
                if attempt == self.max_retries - 1:
                    break
        
        # Fallback: return "0" if all retries failed
        self.log_fn("All retries failed, returning default prediction 0")
        return "0", self._msg_history
