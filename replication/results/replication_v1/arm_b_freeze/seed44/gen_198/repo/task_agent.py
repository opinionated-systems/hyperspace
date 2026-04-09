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


def _find_balanced_json(s: str, start_idx: int) -> str | None:
    """Find a balanced JSON object starting at start_idx."""
    if start_idx >= len(s) or s[start_idx] != '{':
        return None
    
    brace_depth = 0
    in_string = False
    escape_next = False
    end_idx = start_idx
    
    for i in range(start_idx, len(s)):
        char = s[i]
        
        if escape_next:
            escape_next = False
            continue
        
        if char == '\\':
            escape_next = True
            continue
        
        if char == '"' and not in_string:
            in_string = True
        elif char == '"' and in_string:
            in_string = False
        elif not in_string:
            if char == '{':
                brace_depth += 1
            elif char == '}':
                brace_depth -= 1
                if brace_depth == 0:
                    end_idx = i
                    break
    
    if brace_depth == 0:
        return s[start_idx:end_idx + 1]
    return None


def _extract_json_flexible(text: str) -> dict | None:
    """Extract JSON with multiple fallback strategies.
    
    Tries multiple approaches in order:
    1. Standard <json>...</json> blocks
    2. Markdown code blocks with json
    3. Bracket-balanced extraction for "response" key
    4. Any valid JSON object at the end of the text
    5. First valid JSON object anywhere in text
    6. LLM-based extraction as final fallback
    """
    # Strategy 1: Standard <json> tags
    results = _extract_jsons(text)
    if results:
        return results[-1]
    
    # Strategy 2: Markdown code blocks
    json_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    for match in re.finditer(json_block_pattern, text, re.DOTALL):
        content = match.group(1).strip()
        # Try to find JSON object within the code block
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Try to extract JSON object from within the content
            for json_match in re.finditer(r'\{', content):
                candidate = _find_balanced_json(content, json_match.start())
                if candidate:
                    try:
                        parsed = json.loads(candidate)
                        if isinstance(parsed, dict) and "response" in parsed:
                            return parsed
                    except json.JSONDecodeError:
                        continue
            continue
    
    # Strategy 3: Bracket-balanced extraction for "response" key
    for match in re.finditer(r'"response"\s*:', text):
        start = match.start()
        for i in range(start - 1, -1, -1):
            if text[i] == '{':
                candidate = _find_balanced_json(text, i)
                if candidate:
                    try:
                        parsed = json.loads(candidate)
                        if isinstance(parsed, dict) and "response" in parsed:
                            return parsed
                    except json.JSONDecodeError:
                        continue
                break
            elif text[i] in '}':
                break
    
    # Strategy 4: Look for any JSON object at the end of text (last resort)
    last_brace_idx = text.rfind('}')
    if last_brace_idx != -1:
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
    
    # Strategy 5: Find first valid JSON object with "response" field anywhere
    for match in re.finditer(r'\{', text):
        candidate = _find_balanced_json(text, match.start())
        if candidate:
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict) and "response" in parsed:
                    return parsed
            except json.JSONDecodeError:
                continue
    
    # Strategy 6: Try to find any valid JSON object and check for response-like fields
    for match in re.finditer(r'\{', text):
        candidate = _find_balanced_json(text, match.start())
        if candidate:
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    # Check for response-like fields
                    response_aliases = ['response', 'answer', 'result', 'grade', 'score', 'correct']
                    for alias in response_aliases:
                        if alias in parsed:
                            # Normalize to "response" key
                            parsed['response'] = parsed[alias]
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
        self.log_file = log_file

    def _build_prompt(self, inputs: dict) -> str:
        """Build the grading prompt from input fields."""
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        return f"""You are an expert {domain} grader evaluating student solutions with precision and care.

Your task is to grade a student's answer by comparing it to the correct solution and following the grading guidelines exactly.

PROBLEM:
{problem}

CORRECT SOLUTION:
{solution}

GRADING GUIDELINES:
{grading_guidelines}

STUDENT'S ANSWER:
{student_answer}

Think step by step:
1. Analyze what the problem is asking for - identify key requirements and expected outcomes
2. Review the correct solution approach - understand the method and final answer
3. Compare the student's answer to the correct solution - check both the method and final result
4. Check if the student followed the grading guidelines - pay special attention to any specific criteria mentioned
5. Determine if the student's answer is correct (1) or incorrect (0)

Important grading principles:
- The student's answer must match the correct solution in substance, not just form
- Partial credit is not awarded unless explicitly specified in the grading guidelines
- If the grading guidelines specify particular requirements (e.g., "answer must be in simplest form"), verify those strictly
- Numerical answers should be mathematically equivalent to the correct solution

Respond in JSON format with the following schema:
<json>
{{
    "reasoning": "Your detailed step-by-step analysis here. Explain your reasoning clearly, citing specific aspects of the student's answer that led to your conclusion.",
    "response": 1 or 0
}}
</json>

The "response" field must be either 1 (correct) or 0 (incorrect). Be conservative - only mark as correct (1) if you are confident the answer fully satisfies the requirements."""

    def _validate_prediction(self, prediction: Any) -> str | None:
        """Validate and normalize prediction value.
        
        Returns normalized string value or None if invalid.
        Enhanced to handle more edge cases and provide detailed logging.
        """
        if prediction is None:
            self.log_fn("Prediction validation: received None value")
            return None
        
        original_prediction = prediction
        
        # Handle boolean types (explicitly convert before numeric check)
        if isinstance(prediction, bool):
            result = "1" if prediction else "0"
            self.log_fn(f"Prediction validation: converted bool {original_prediction} -> {result}")
            return result
        
        # Handle numeric types
        if isinstance(prediction, (int, float)):
            # Round floats to nearest integer for robustness
            if isinstance(prediction, float):
                prediction = round(prediction)
                self.log_fn(f"Prediction validation: rounded float {original_prediction} -> {prediction}")
            if prediction in (0, 1):
                result = str(int(prediction))
                self.log_fn(f"Prediction validation: accepted numeric {original_prediction} -> {result}")
                return result
            self.log_fn(f"Prediction validation: rejected numeric {original_prediction} (not 0 or 1)")
            return None
        
        # Handle string types
        if isinstance(prediction, str):
            prediction_stripped = prediction.strip()
            prediction_lower = prediction_stripped.lower()
            
            # Direct match for "0" or "1"
            if prediction_stripped in ("0", "1"):
                self.log_fn(f"Prediction validation: accepted direct string '{original_prediction}'")
                return prediction_stripped
            
            # Boolean-like strings
            if prediction_lower in ("0", "false", "incorrect", "wrong", "no", "fail", "failed"):
                self.log_fn(f"Prediction validation: converted string '{original_prediction}' -> 0")
                return "0"
            if prediction_lower in ("1", "true", "correct", "right", "yes", "pass", "passed", "success"):
                self.log_fn(f"Prediction validation: converted string '{original_prediction}' -> 1")
                return "1"
            
            # Try numeric conversion with more flexibility
            try:
                # Handle cases like "1.0", "0.0", " 1 ", etc.
                val = float(prediction_stripped)
                val_int = round(val)
                if val_int in (0, 1):
                    result = str(int(val_int))
                    self.log_fn(f"Prediction validation: converted numeric string '{original_prediction}' -> {result}")
                    return result
                self.log_fn(f"Prediction validation: rejected numeric string '{original_prediction}' (not 0 or 1)")
            except ValueError:
                self.log_fn(f"Prediction validation: rejected non-numeric string '{original_prediction}'")
                pass
        
        self.log_fn(f"Prediction validation: rejected unsupported type {type(original_prediction)} with value {original_prediction!r}")
        return None

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        instruction = self._build_prompt(inputs)
        msg_history: list[dict] = []
        
        self.log_fn(f"TaskAgent starting: domain={inputs.get('domain', 'unknown')}, max_retries={self.max_retries}")
        
        # Try with retries for robustness
        for attempt in range(self.max_retries):
            try:
                self.log_fn(f"Attempt {attempt + 1}/{self.max_retries}: calling LLM...")
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=msg_history if msg_history else [],
                )
                
                # Log token usage if available
                usage = info.get("usage", {})
                if usage:
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    completion_tokens = usage.get("completion_tokens", 0)
                    self.log_fn(f"Token usage: prompt={prompt_tokens}, completion={completion_tokens}")
                
                # Extract prediction from JSON using flexible extraction
                if msg_history and len(msg_history) > 0:
                    last_msg = msg_history[-1]
                    text = last_msg.get("text", "") if isinstance(last_msg, dict) else str(last_msg)
                    extracted = _extract_json_flexible(text)
                else:
                    extracted = _extract_json_flexible(response)
                
                if extracted:
                    self.log_fn(f"Extracted JSON keys: {list(extracted.keys())}")
                    if "response" in extracted:
                        prediction = extracted["response"]
                        self.log_fn(f"Raw prediction value: {prediction!r} (type: {type(prediction).__name__})")
                        validated = self._validate_prediction(prediction)
                        if validated is not None:
                            self.log_fn(f"Successfully validated prediction: {validated}")
                            return validated, msg_history
                        else:
                            self.log_fn(f"Invalid prediction value: {prediction!r}, retrying...")
                    else:
                        self.log_fn(f"JSON missing 'response' key. Available keys: {list(extracted.keys())}, retrying...")
                else:
                    self.log_fn(f"No valid JSON found in response, retrying...")
                    # Log a snippet of the response for debugging
                    response_preview = response[:200] if response else "(empty)"
                    self.log_fn(f"Response preview: {response_preview!r}")
                    
            except Exception as e:
                self.log_fn(f"Error on attempt {attempt + 1}: {type(e).__name__}: {e}")
                if attempt == self.max_retries - 1:
                    break
        
        # Fallback: return "0" if all retries failed
        self.log_fn(f"All {self.max_retries} retries failed, returning default prediction 0")
        return "0", msg_history
