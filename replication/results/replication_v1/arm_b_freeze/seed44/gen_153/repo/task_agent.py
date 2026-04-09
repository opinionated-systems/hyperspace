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
    3. Raw JSON objects with "response" key
    4. JSON objects with "reasoning" key
    5. Any valid JSON object at the end of the text
    6. Look for JSON with numeric response values (0 or 1) anywhere in text
    7. Look for standalone 0 or 1 at the end of the response
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
    
    # Strategy 4: Look for JSON with "reasoning" key (full schema)
    full_json_pattern = r'\{\s*"reasoning"\s*:\s*"[^"]*"\s*,\s*"response"\s*:\s*(?:0|1|\d+)\s*\}'
    for match in re.finditer(full_json_pattern, text, re.DOTALL):
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
    
    # Strategy 5: Look for any JSON object at the end of text (last resort)
    # This handles cases where the model outputs JSON without any markers
    last_brace_idx = text.rfind('}')
    if last_brace_idx != -1:
        # Find the matching opening brace
        brace_count = 0
        for i in range(last_brace_idx, -1, -1):
            if text[i] == '}':
                brace_count += 1
            elif text[i] == '{':
                brace_count -= 1
                if brace_count == 0:
                    try:
                        candidate = text[i:last_brace_idx + 1]
                        parsed = json.loads(candidate)
                        if isinstance(parsed, dict) and "response" in parsed:
                            return parsed
                    except json.JSONDecodeError:
                        continue
                    break
    
    # Strategy 6: Look for simple "response": 0 or "response": 1 patterns
    # This catches cases where the JSON structure is malformed but the key info is there
    simple_response_pattern = r'"response"\s*:\s*(0|1)'
    match = re.search(simple_response_pattern, text)
    if match:
        return {"response": int(match.group(1))}
    
    # Strategy 7: Look for standalone 0 or 1 at the very end (last line)
    # This handles cases where the model just outputs the answer
    lines = text.strip().split('\n')
    for line in reversed(lines):
        line = line.strip()
        if line == '0':
            return {"response": 0}
        if line == '1':
            return {"response": 1}
        # Skip empty lines and common suffixes
        if line and not line.lower() in ['correct', 'incorrect', 'true', 'false']:
            break
    
    return None


def _validate_prediction(prediction: Any) -> int | None:
    """Validate and normalize a prediction value.
    
    Returns:
        0 or 1 if valid, None otherwise
    """
    if prediction in [0, 1]:
        return prediction
    if prediction in ["0", "1"]:
        return int(prediction)
    if isinstance(prediction, str):
        pred_lower = prediction.lower().strip()
        if pred_lower in ["0", "false", "incorrect", "no", "wrong"]:
            return 0
        if pred_lower in ["1", "true", "correct", "yes", "right"]:
            return 1
    return None


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.max_retries = 3
        self._call_count = 0
        self._success_count = 0

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        self._call_count += 1
        start_time = time.time()
        
        # Extract key fields for better prompting
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        # Truncate very long inputs to prevent token overflow
        max_len = 8000
        problem = problem[:max_len] if len(problem) > max_len else problem
        solution = solution[:max_len] if len(solution) > max_len else solution
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

        msg_history: list[dict] = []
        last_error = None
        
        # Try with retries for robustness
        for attempt in range(self.max_retries):
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=[],
                )
                
                response_text = msg_history[-1]["text"] if msg_history else ""
                
                # Extract prediction from JSON using flexible extraction
                extracted = _extract_json_flexible(response_text)
                
                if extracted and "response" in extracted:
                    prediction = extracted["response"]
                    validated = _validate_prediction(prediction)
                    
                    if validated is not None:
                        self._success_count += 1
                        elapsed = time.time() - start_time
                        self.log_fn(
                            f"Call {self._call_count}: Success in {elapsed:.2f}s "
                            f"(attempt {attempt + 1}/{self.max_retries}, "
                            f"success rate: {self._success_count}/{self._call_count})"
                        )
                        return str(validated), msg_history
                    else:
                        last_error = f"Invalid prediction value: {prediction}"
                        self.log_fn(f"Attempt {attempt + 1}: {last_error}, retrying...")
                else:
                    last_error = "No valid JSON found in response"
                    # Log a snippet of the response for debugging
                    snippet = response_text[:200].replace('\n', ' ')
                    self.log_fn(f"Attempt {attempt + 1}: {last_error} (response: {snippet}...), retrying...")
                    
            except Exception as e:
                last_error = str(e)
                self.log_fn(f"Error on attempt {attempt + 1}: {e}")
                if attempt == self.max_retries - 1:
                    break
        
        # Fallback: return "0" if all retries failed
        elapsed = time.time() - start_time
        self.log_fn(
            f"Call {self._call_count}: FAILED after {self.max_retries} attempts "
            f"({elapsed:.2f}s). Last error: {last_error}. "
            f"Returning default prediction 0."
        )
        return "0", msg_history
