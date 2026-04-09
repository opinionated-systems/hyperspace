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
from agent.utils import Timer, sanitize_string

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
    4. Relaxed JSON parsing for common LLM output errors
    5. Pattern matching for reasoning + response structure
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
    
    # Strategy 4: Relaxed parsing - look for any JSON object with response field
    # Handle common LLM errors like trailing commas, single quotes, etc.
    relaxed_pattern = r'\{[\s\S]*?"response"\s*:\s*(\d+|"[^"]*")[\s\S]*?\}'
    for match in re.finditer(relaxed_pattern, text, re.DOTALL):
        try:
            candidate = match.group(0)
            # Fix common JSON errors
            candidate = re.sub(r',\s*}', '}', candidate)  # Remove trailing commas
            candidate = re.sub(r"'", '"', candidate)     # Replace single quotes
            candidate = re.sub(r',\s*]', ']', candidate)  # Remove trailing commas in arrays
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    
    # Strategy 5: Look for reasoning + response pattern even without proper JSON
    reasoning_pattern = r'["\']?reasoning["\']?\s*[:=]\s*["\']?([^"\']+)["\']?'
    response_pattern = r'["\']?response["\']?\s*[:=]\s*(\d+|"[^"]*")'
    
    reasoning_match = re.search(reasoning_pattern, text, re.IGNORECASE)
    response_match = re.search(response_pattern, text, re.IGNORECASE)
    
    if response_match:
        response_val = response_match.group(1).strip().strip('"\'')
        reasoning_val = reasoning_match.group(1).strip() if reasoning_match else "No reasoning provided"
        return {
            "reasoning": reasoning_val,
            "response": response_val
        }
    
    return None


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.max_retries = 3
        self._call_count = 0
        self._total_latency = 0.0

    def get_stats(self) -> dict[str, Any]:
        """Return agent statistics."""
        return {
            "call_count": self._call_count,
            "total_latency": self._total_latency,
            "avg_latency": self._total_latency / self._call_count if self._call_count > 0 else 0,
        }

    def _normalize_prediction(self, prediction) -> str:
        """Normalize prediction to '0' or '1'.
        
        Handles various formats: integers, strings, booleans, etc.
        Also handles edge cases like "correct (1)" or "incorrect (0)".
        """
        if prediction is None:
            return None
        
        # Handle boolean values
        if isinstance(prediction, bool):
            return "1" if prediction else "0"
        
        # Handle numeric values
        if isinstance(prediction, (int, float)):
            return "1" if prediction == 1 else "0"
        
        # Handle string values
        pred_str = str(prediction).strip().lower()
        
        # Check for explicit "1" or "0" at start or end (e.g., "correct (1)")
        if pred_str.startswith("1") or pred_str.endswith("1") or pred_str in ("true", "correct", "yes", "right", "valid", "pass", "success"):
            return "1"
        elif pred_str.startswith("0") or pred_str.endswith("0") or pred_str in ("false", "incorrect", "no", "wrong", "invalid", "fail", "failure"):
            return "0"
        
        # Check for numeric patterns in parentheses
        paren_match = re.search(r'\((\d)\)', pred_str)
        if paren_match:
            val = paren_match.group(1)
            if val == "1":
                return "1"
            elif val == "0":
                return "0"
        
        return None

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        start_time = time.time()
        self._call_count += 1
        
        # Extract key fields for better prompting
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        # Log input summary for debugging
        self.log_fn(f"TaskAgent call #{self._call_count}: domain={domain}, "
                   f"problem_len={len(problem)}, solution_len={len(solution)}, "
                   f"student_len={len(student_answer)}")

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

IMPORTANT: You must respond ONLY with a valid JSON object wrapped in <json> tags. Do not include any other text outside the JSON tags.

Respond in this exact format:
<json>
{{
    "reasoning": "Your detailed step-by-step analysis here. Explain your thinking clearly.",
    "response": 1
}}
</json>

OR

<json>
{{
    "reasoning": "Your detailed step-by-step analysis here. Explain your thinking clearly.",
    "response": 0
}}
</json>

The "response" field MUST be either 1 (correct) or 0 (incorrect) - use the integer value, not a string.
The "reasoning" field should contain your complete analysis of why the answer is correct or incorrect."""

        # Try with retries for robustness
        msg_history = []
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                # On retry, add feedback about previous error
                msg_to_send = instruction
                if attempt > 0 and last_error:
                    msg_to_send = f"{instruction}\n\nPREVIOUS ATTEMPT FAILED: {last_error}\nPlease ensure your response follows the exact JSON format specified above."
                
                response, msg_history, info = get_response_from_llm(
                    msg=msg_to_send,
                    model=self.model,
                    msg_history=msg_history,
                )
                
                # Extract prediction from JSON using flexible extraction
                response_text = msg_history[-1]["text"] if msg_history else ""
                extracted = _extract_json_flexible(response_text)
                
                if extracted and "response" in extracted:
                    prediction = extracted["response"]
                    normalized = self._normalize_prediction(prediction)
                    
                    if normalized is not None:
                        elapsed = time.time() - start_time
                        self._total_latency += elapsed
                        self.log_fn(f"Successfully graded: response={normalized} (took {elapsed:.2f}s)")
                        return normalized, msg_history
                    else:
                        self.log_fn(f"Invalid prediction value: {prediction}, retrying...")
                        last_error = f"Invalid prediction value: {prediction}. Must be 0 or 1."
                else:
                    self.log_fn(f"No valid JSON found in response, retrying...")
                    last_error = "No valid JSON with 'response' field found. Ensure you wrap your JSON in <json> tags."
                    
            except Exception as e:
                self.log_fn(f"Error on attempt {attempt + 1}: {e}")
                last_error = str(e)
                if attempt == self.max_retries - 1:
                    break
        
        # Fallback: return "0" if all retries failed
        elapsed = time.time() - start_time
        self._total_latency += elapsed
        self.log_fn(f"All retries failed ({last_error}), returning default prediction 0 (took {elapsed:.2f}s)")
        return "0", msg_history
