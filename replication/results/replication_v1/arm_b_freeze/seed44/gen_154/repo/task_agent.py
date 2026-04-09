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


def _extract_json_flexible(text: str) -> dict | None:
    """Extract JSON with multiple fallback strategies.
    
    Tries multiple approaches in order:
    1. Standard <json>...</json> blocks
    2. Markdown code blocks with json
    3. Raw JSON objects in text (with brace balancing)
    4. Look for JSON-like structures with "response" key
    5. Extract from reasoning + response patterns
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
    
    # Strategy 3: Find JSON objects with balanced braces
    # This handles nested braces properly by counting open/close braces
    def find_json_objects(s: str) -> list[str]:
        """Find all JSON-like substrings with balanced braces."""
        results = []
        i = 0
        while i < len(s):
            if s[i] == '{':
                start = i
                brace_count = 1
                i += 1
                in_string = False
                escape_next = False
                
                while i < len(s) and brace_count > 0:
                    char = s[i]
                    if escape_next:
                        escape_next = False
                    elif char == '\\' and in_string:
                        escape_next = True
                    elif char == '"' and not escape_next:
                        in_string = not in_string
                    elif not in_string:
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                    i += 1
                
                if brace_count == 0:
                    results.append(s[start:i])
            else:
                i += 1
        return results
    
    for json_str in find_json_objects(text):
        try:
            parsed = json.loads(json_str)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue
    
    # Strategy 4: Look for JSON-like structures with "response" key
    # More permissive pattern that handles nested braces
    json_pattern = r'\{\s*"response"\s*:\s*(?:\{[^}]*\}|\[[^\]]*\]|"[^"]*"|\d+|true|false|null)\s*(?:,\s*"[^"]*"\s*:\s*(?:\{[^}]*\}|\[[^\]]*\]|"[^"]*"|\d+|true|false|null)\s*)*\}'
    for match in re.finditer(json_pattern, text, re.DOTALL):
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
    
    # Strategy 5: Look for explicit "response": 1 or "response": 0 patterns
    # This handles cases where JSON is malformed but the response value is clear
    response_match = re.search(r'"response"\s*:\s*([01])', text)
    if response_match:
        response_val = int(response_match.group(1))
        # Try to extract reasoning if available
        reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]*)"', text, re.DOTALL)
        if reasoning_match:
            return {"reasoning": reasoning_match.group(1), "response": response_val}
        return {"reasoning": "Extracted from pattern match", "response": response_val}
    
    # Strategy 6: Look for standalone 1 or 0 at the end of the response
    # This is a last resort for malformed outputs
    standalone_match = re.search(r'(?:^|\s)([01])(?:\s*$|\s*\n)', text.strip())
    if standalone_match:
        response_val = int(standalone_match.group(1))
        return {"reasoning": "Extracted from standalone value", "response": response_val}
    
    return None


def _normalize_prediction(value) -> str:
    """Normalize a prediction value to '0' or '1'.
    
    Handles various formats: integers, strings, booleans, and common variations.
    """
    if value is None:
        return "0"
    
    # Handle boolean
    if isinstance(value, bool):
        return "1" if value else "0"
    
    # Handle numeric (int, float)
    if isinstance(value, (int, float)):
        # Handle NaN and infinity
        if isinstance(value, float):
            if value != value:  # NaN check
                return "0"
            if value == float('inf') or value == float('-inf'):
                return "0"
        return "1" if value >= 0.5 else "0"
    
    # Handle string
    if isinstance(value, str):
        value = value.strip().lower()
        # Direct matches for "1"
        if value in ("1", "true", "yes", "correct", "right", "valid", "pass", "passed", 
                     "accurate", "acceptable", "satisfactory", "complete"):
            return "1"
        # Direct matches for "0"
        if value in ("0", "false", "no", "incorrect", "wrong", "invalid", "fail", "failed",
                     "inaccurate", "unacceptable", "unsatisfactory", "incomplete", "partial"):
            return "0"
        # Try to parse as number
        try:
            num = float(value)
            if num != num:  # NaN check
                return "0"
            return "1" if num >= 0.5 else "0"
        except ValueError:
            pass
    
    # Handle lists/arrays - check if any element indicates correctness
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return "0"
        # If list contains a single element, normalize that
        if len(value) == 1:
            return _normalize_prediction(value[0])
        # For multi-element lists, check if all are truthy
        return "1" if all(_normalize_prediction(v) == "1" for v in value) else "0"
    
    # Handle dictionaries - look for common keys
    if isinstance(value, dict):
        for key in ["response", "answer", "prediction", "result", "grade", "correct", "is_correct", "value"]:
            if key in value:
                return _normalize_prediction(value[key])
        return "0"
    
    # Default to 0 for unrecognizable values
    return "0"


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.max_retries = 3

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Extract key fields for better prompting
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        # Log input sizes for debugging
        self.log_fn(f"Processing problem: domain={domain}, problem_len={len(problem)}, "
                   f"solution_len={len(solution)}, guidelines_len={len(grading_guidelines)}, "
                   f"answer_len={len(student_answer)}")

        instruction = f"""You are an expert {domain} grader evaluating student solutions with high precision and consistency.

Your task is to grade a student's answer by comparing it to the correct solution and following the grading guidelines exactly.

PROBLEM:
{problem}

CORRECT SOLUTION:
{solution}

GRADING GUIDELINES:
{grading_guidelines}

STUDENT'S ANSWER:
{student_answer}

Think step by step and provide detailed reasoning:
1. PROBLEM ANALYSIS: What is the problem asking for? What are the key requirements?
2. CORRECT SOLUTION REVIEW: What is the correct approach and final answer?
3. STUDENT ANSWER ANALYSIS: What did the student submit? Break down their reasoning.
4. COMPARISON: Compare the student's answer to the correct solution point by point.
5. GUIDELINES CHECK: Did the student follow all grading guidelines explicitly?
6. FINAL DECISION: Is the student's answer fully correct (1) or incorrect (0)?
   - Award 1 ONLY if the answer is completely correct and meets all requirements
   - Award 0 if there are ANY errors, missing steps, or deviations from the correct solution

Respond in JSON format with the following schema:
<json>
{{
    "reasoning": "Your detailed step-by-step analysis here. Be thorough and explicit about why the answer is correct or incorrect.",
    "response": 1 or 0
}}
</json>

IMPORTANT: The "response" field must be either 1 (correct) or 0 (incorrect). Be conservative - when in doubt, mark as 0."""

        msg_history = []
        last_error = None
        
        # Try with retries for robustness
        for attempt in range(self.max_retries):
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=[],
                )
                
                # Extract prediction from JSON using flexible extraction
                response_text = msg_history[-1]["text"] if msg_history else ""
                
                # Log response preview for debugging
                preview = response_text[:200] if response_text else "(empty)"
                self.log_fn(f"Attempt {attempt + 1}: Response preview: {preview}...")
                
                extracted = _extract_json_flexible(response_text)
                
                if extracted:
                    self.log_fn(f"Attempt {attempt + 1}: Successfully extracted JSON with keys: {list(extracted.keys())}")
                    
                    # Try to get prediction from various possible keys
                    prediction = None
                    prediction_key = None
                    for key in ["response", "answer", "prediction", "result", "grade", "correct", "is_correct"]:
                        if key in extracted:
                            prediction = extracted[key]
                            prediction_key = key
                            break
                    
                    if prediction is not None:
                        normalized = _normalize_prediction(prediction)
                        self.log_fn(f"Attempt {attempt + 1}: Extracted prediction from key '{prediction_key}': {prediction} -> {normalized}")
                        return normalized, msg_history
                    else:
                        self.log_fn(f"Attempt {attempt + 1}: No prediction key found in extracted JSON: {extracted.keys()}, retrying...")
                        last_error = "No prediction key found"
                else:
                    self.log_fn(f"Attempt {attempt + 1}: No valid JSON found in response, retrying...")
                    last_error = "No valid JSON extracted"
                    
            except Exception as e:
                self.log_fn(f"Error on attempt {attempt + 1}: {e}")
                last_error = str(e)
                if attempt == self.max_retries - 1:
                    break
        
        # Fallback: return "0" if all retries failed
        self.log_fn(f"All {self.max_retries} retries failed (last error: {last_error}), returning default prediction 0")
        return "0", msg_history
