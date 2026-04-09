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
from typing import Any, Callable

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


def _extract_json_flexible(text: str, log_fn: Callable | None = None) -> dict | None:
    """Extract JSON with multiple fallback strategies.
    
    Tries multiple approaches in order:
    1. Standard <json>...</json> blocks
    2. Markdown code blocks with json
    3. Raw JSON objects in text
    4. Relaxed JSON pattern matching for malformed responses
    5. Line-by-line JSON object extraction
    6. Bracket matching from the end of text
    7. LLM-based JSON repair for malformed responses
    8. Single-line JSON object extraction with cleanup
    """
    # Strategy 1: Standard <json> tags
    results = _extract_jsons(text)
    if results:
        if log_fn:
            log_fn(f"JSON extracted via <json> tags: {results[-1]}")
        return results[-1]
    
    # Strategy 2: Markdown code blocks
    json_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    for match in re.finditer(json_block_pattern, text, re.DOTALL):
        try:
            parsed = json.loads(match.group(1).strip())
            if log_fn:
                log_fn(f"JSON extracted via markdown block: {parsed}")
            return parsed
        except json.JSONDecodeError:
            continue
    
    # Strategy 3: Look for JSON-like structures with "response" key
    json_pattern = r'\{\s*"response"\s*:[^\}]+\}'
    for match in re.finditer(json_pattern, text, re.DOTALL):
        try:
            parsed = json.loads(match.group(0))
            if log_fn:
                log_fn(f"JSON extracted via response pattern: {parsed}")
            return parsed
        except json.JSONDecodeError:
            continue
    
    # Strategy 4: Relaxed pattern - look for any JSON object with response field
    # Handles cases where JSON might span multiple lines with extra whitespace
    relaxed_pattern = r'\{[^}]*"response"\s*:\s*(?:1|0|"1"|"0")[^}]*\}'
    for match in re.finditer(relaxed_pattern, text, re.DOTALL):
        try:
            parsed = json.loads(match.group(0))
            if "response" in parsed:
                if log_fn:
                    log_fn(f"JSON extracted via relaxed pattern: {parsed}")
                return parsed
        except json.JSONDecodeError:
            continue
    
    # Strategy 5: Look for JSON objects with both "reasoning" and "response" keys
    # This handles cases where the JSON spans multiple lines
    full_json_pattern = r'\{[\s\S]*?"reasoning"[\s\S]*?"response"[\s\S]*?\}'
    for match in re.finditer(full_json_pattern, text):
        try:
            parsed = json.loads(match.group(0))
            if "response" in parsed:
                if log_fn:
                    log_fn(f"JSON extracted via full pattern: {parsed}")
                return parsed
        except json.JSONDecodeError:
            continue
    
    # Strategy 6: Smart bracket matching from the end of text
    # This handles cases where JSON is embedded in other text
    last_brace = text.rfind('}')
    if last_brace != -1:
        # Try to find the matching opening brace using bracket counting
        brace_count = 0
        for i in range(last_brace, -1, -1):
            if text[i] == '}':
                brace_count += 1
            elif text[i] == '{':
                brace_count -= 1
                if brace_count == 0:
                    candidate = text[i:last_brace+1]
                    try:
                        parsed = json.loads(candidate)
                        if "response" in parsed:
                            if log_fn:
                                log_fn(f"JSON extracted via bracket matching: {parsed}")
                            return parsed
                    except json.JSONDecodeError:
                        # Try to extract just the response field if full parse fails
                        try:
                            response_match = re.search(r'"response"\s*:\s*(\d)', candidate)
                            if response_match:
                                result = {"response": int(response_match.group(1))}
                                if log_fn:
                                    log_fn(f"JSON extracted via response field extraction: {result}")
                                return result
                        except Exception:
                            pass
                    break
    
    # Strategy 7: Last resort - look for any numeric response pattern
    # Matches patterns like: "response": 1 or "response":0 or response: 1
    last_resort_pattern = r'["\']?response["\']?\s*:\s*(\d)'
    match = re.search(last_resort_pattern, text, re.IGNORECASE)
    if match:
        val = match.group(1)
        if val in ('0', '1'):
            result = {"response": int(val)}
            if log_fn:
                log_fn(f"JSON extracted via last resort pattern: {result}")
            return result
    
    # Strategy 8: Try to find and clean up single-line JSON objects
    # Handles cases with trailing commas, unquoted keys, etc.
    single_line_pattern = r'\{.*\}'
    for match in re.finditer(single_line_pattern, text, re.DOTALL):
        candidate = match.group(0)
        # Clean up common JSON issues
        cleaned = candidate
        # Remove trailing commas before closing braces
        cleaned = re.sub(r',\s*}', '}', cleaned)
        # Remove trailing commas before closing brackets
        cleaned = re.sub(r',\s*\]', ']', cleaned)
        # Fix unquoted keys (simple cases)
        cleaned = re.sub(r'(\{|,\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', cleaned)
        try:
            parsed = json.loads(cleaned)
            if "response" in parsed:
                if log_fn:
                    log_fn(f"JSON extracted via cleanup: {parsed}")
                return parsed
        except json.JSONDecodeError:
            continue
    
    if log_fn:
        log_fn(f"All JSON extraction strategies failed. Text preview: {text[:200]!r}")
    return None


def _normalize_prediction(prediction: Any) -> str:
    """Normalize prediction value to '0' or '1'.
    
    Handles various formats: integers, strings, booleans.
    Returns '0' as default for invalid values.
    """
    if prediction is None:
        return "0"
    
    # Convert to string and clean
    pred_str = str(prediction).strip().lower()
    
    # Handle boolean-like values
    if pred_str in ("true", "1", "yes", "correct"):
        return "1"
    if pred_str in ("false", "0", "no", "incorrect"):
        return "0"
    
    # Try numeric conversion
    try:
        num = int(float(pred_str))
        return "1" if num == 1 else "0"
    except (ValueError, TypeError):
        pass
    
    # Default to 0 for unparseable values
    logger.warning(f"Could not normalize prediction: {prediction!r}, defaulting to 0")
    return "0"


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.max_retries = 3
        self._last_msg_history: list[dict] = []

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

        instruction = f"""You are an expert {domain} grader evaluating student solutions.

Your task is to grade a student's answer by comparing it to the correct solution and following the grading guidelines EXACTLY.

PROBLEM:
{problem}

CORRECT SOLUTION:
{solution}

GRADING GUIDELINES (FOLLOW THESE STRICTLY):
{grading_guidelines}

STUDENT'S ANSWER:
{student_answer}

Think step by step:
1. Analyze what the problem is asking for
2. Review the correct solution approach
3. Compare the student's answer to the correct solution
4. Check if the student followed the grading guidelines EXACTLY
5. Determine if the student's answer is correct (1) or incorrect (0)

IMPORTANT GRADING PRINCIPLES:
- The grading guidelines are the PRIMARY criteria for correctness
- A student's answer can be mathematically equivalent to the correct solution but still be marked INCORRECT if it violates the grading guidelines
- Conversely, a student's answer that follows the grading guidelines should be marked CORRECT even if it looks different from the expected solution
- Pay special attention to:
  * Required formats (e.g., simplified fractions, specific units)
  * Required steps or methods
  * Constraints on the solution approach
  * Presentation requirements

Respond in JSON format with the following schema:
<json>
{{
    "reasoning": "Your detailed step-by-step analysis here. Explicitly mention how the student's answer relates to the grading guidelines.",
    "response": 1 or 0
}}
</json>

The "response" field must be either 1 (correct) or 0 (incorrect)."""

        # Try with retries for robustness
        last_error = None
        current_instruction = instruction
        msg_history = []
        
        for attempt in range(self.max_retries):
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=current_instruction,
                    model=self.model,
                    msg_history=msg_history if attempt > 0 else [],
                )
                self._last_msg_history = msg_history
                
                # Extract prediction from JSON using flexible extraction
                response_text = msg_history[-1].get("text", "") if msg_history else ""
                extracted = _extract_json_flexible(response_text, log_fn=self.log_fn)
                
                if extracted and "response" in extracted:
                    prediction = extracted["response"]
                    normalized = _normalize_prediction(prediction)
                    self.log_fn(f"Successfully extracted prediction: {prediction} -> {normalized}")
                    return normalized, msg_history
                else:
                    self.log_fn(f"Attempt {attempt + 1}: No valid JSON found in response")
                    if attempt == 0:
                        # Log first few characters of response for debugging
                        preview = response_text[:500] if response_text else "(empty)"
                        self.log_fn(f"Response preview: {preview!r}")
                    
                    # Prepare feedback for retry
                    if attempt < self.max_retries - 1:
                        current_instruction = (
                            f"Your previous response did not contain valid JSON with a 'response' field.\n\n"
                            f"Please respond ONLY with JSON in this exact format:\n"
                            f"<json>\n"
                            f'{{\n    "reasoning": "Your analysis here",\n    "response": 1 or 0\n}}\n'
                            f"</json>\n\n"
                            f"The response field must be exactly 1 (correct) or 0 (incorrect)."
                        )
                    
            except Exception as e:
                last_error = e
                self.log_fn(f"Attempt {attempt + 1}: Error during LLM call or parsing: {e}")
                if attempt == self.max_retries - 1:
                    break
        
        # Fallback: return "0" if all retries failed
        if last_error:
            self.log_fn(f"All retries failed with error: {last_error}")
        else:
            self.log_fn("All retries failed - could not extract valid prediction")
        
        return "0", self._last_msg_history if self._last_msg_history else []
