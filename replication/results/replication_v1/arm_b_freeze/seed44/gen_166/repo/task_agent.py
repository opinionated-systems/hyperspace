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
    5. Direct JSON parsing of the entire text
    6. Extract from single-quoted or malformed JSON
    """
    if not text or not isinstance(text, str):
        return None
        
    text = text.strip()
    if not text:
        return None
    
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
    
    # Strategy 5: Try to parse the entire text as JSON
    try:
        parsed = json.loads(text.strip())
        if isinstance(parsed, dict) and "response" in parsed:
            return parsed
    except json.JSONDecodeError:
        pass
    
    # Strategy 6: Try to fix common JSON errors and re-parse
    # Handle single quotes, trailing commas, etc.
    try:
        # Replace single quotes with double quotes (carefully)
        fixed = text.replace("'", '"')
        # Remove trailing commas before closing braces
        fixed = re.sub(r',\s*}', '}', fixed)
        fixed = re.sub(r',\s*]', ']', fixed)
        parsed = json.loads(fixed)
        if isinstance(parsed, dict) and "response" in parsed:
            return parsed
    except (json.JSONDecodeError, Exception):
        pass
    
    # Strategy 7: Regex-based extraction for response field as last resort
    # Look for patterns like "response": 1 or "response": 0 or "response": true/false
    response_patterns = [
        r'"response"\s*:\s*([01])\b',
        r'"response"\s*:\s*(true|false)\b',
        r'"response"\s*:\s*"([01])"',
    ]
    for pattern in response_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            val = match.group(1).lower()
            if val in ('1', 'true'):
                return {"response": 1, "reasoning": "Extracted via regex fallback"}
            elif val in ('0', 'false'):
                return {"response": 0, "reasoning": "Extracted via regex fallback"}
    
    return None


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.max_retries = 3

    def _normalize_prediction(self, prediction: Any) -> str | None:
        """Normalize prediction to '0' or '1', or None if invalid."""
        if prediction is None:
            return None
        
        # Handle boolean (must check before int since bool is subclass of int)
        if isinstance(prediction, bool):
            return "1" if prediction else "0"
        
        # Handle numeric types
        if isinstance(prediction, (int, float)):
            # Use exact comparison for integers
            if isinstance(prediction, int):
                if prediction == 0:
                    return "0"
                if prediction == 1:
                    return "1"
            # For floats, check if they're close to 0 or 1
            if abs(prediction - 0.0) < 0.0001:
                return "0"
            if abs(prediction - 1.0) < 0.0001:
                return "1"
            return None
        
        # Handle string types
        if isinstance(prediction, str):
            pred_clean = prediction.strip().lower()
            # Direct matches
            if pred_clean in ("0", "false", "incorrect", "wrong", "no", "fail", "failed", "f"):
                return "0"
            if pred_clean in ("1", "true", "correct", "right", "yes", "pass", "passed", "t"):
                return "1"
            # Try numeric conversion
            try:
                val = float(pred_clean)
                if abs(val - 0.0) < 0.0001:
                    return "0"
                if abs(val - 1.0) < 0.0001:
                    return "1"
            except ValueError:
                pass
        
        return None

    def _build_prompt(self, inputs: dict) -> str:
        """Build the grading prompt from inputs."""
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        return f"""You are an expert {domain} grader evaluating student solutions with high precision and consistency.

Your task is to grade a student's answer by carefully comparing it to the correct solution and strictly following the grading guidelines.

PROBLEM:
{problem}

CORRECT SOLUTION:
{solution}

GRADING GUIDELINES:
{grading_guidelines}

STUDENT'S ANSWER:
{student_answer}

GRADE THE ANSWER WITH EXTREME CARE:

Step 1 - Problem Analysis:
- Identify the core question and what constitutes a correct answer
- Note any specific requirements (format, units, precision, etc.)

Step 2 - Solution Review:
- Understand the correct approach and final answer
- Identify key elements that must be present for correctness

Step 3 - Student Answer Evaluation:
- Compare the student's answer to the correct solution line by line
- Check for mathematical equivalence (e.g., 0.5 = 1/2 = 2/4)
- Verify the student followed any formatting requirements
- Look for partial understanding vs complete correctness

Step 4 - Guideline Application:
- Apply the grading guidelines strictly and consistently
- Consider if the guidelines specify any special scoring rules

Step 5 - Final Determination:
- Award 1 (correct) ONLY if the answer is fully correct according to both the solution AND guidelines
- Award 0 (incorrect) if there are ANY errors, omissions, or guideline violations
- When in doubt, provide detailed reasoning for your decision

CRITICAL INSTRUCTIONS:
1. Your response MUST be valid JSON inside <json>...</json> tags
2. The "response" field MUST be exactly 1 or 0 (integer, not string, not boolean)
3. The "reasoning" field must contain your detailed analysis
4. Do not include any text outside the <json>...</json> tags
5. Ensure the JSON is properly formatted with double quotes

Respond in this EXACT format:
<json>
{{
    "reasoning": "Your detailed step-by-step analysis explaining why the answer is correct or incorrect",
    "response": 1
}}
</json>

OR

<json>
{{
    "reasoning": "Your detailed step-by-step analysis explaining why the answer is correct or incorrect",
    "response": 0
}}
</json>

The "response" field MUST be either 1 (correct) or 0 (incorrect) - no other values are accepted."""

    def _extract_from_history(self, msg_history: list[dict]) -> dict | None:
        """Extract JSON from message history, trying multiple sources."""
        # Try the last assistant message first
        for msg in reversed(msg_history):
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                text = msg.get("text", "")
                if text:
                    extracted = _extract_json_flexible(text)
                    if extracted and "response" in extracted:
                        return extracted
        
        # Try all messages as a last resort
        for msg in reversed(msg_history):
            if isinstance(msg, dict):
                text = msg.get("text", "") or msg.get("content", "")
                if text:
                    extracted = _extract_json_flexible(text)
                    if extracted and "response" in extracted:
                        return extracted
        
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
        last_error: str = ""
        
        # Try with retries for robustness, with feedback to the model
        for attempt in range(self.max_retries):
            try:
                # On retry, add feedback about previous errors
                msg_to_send = instruction
                if attempt > 0 and last_error:
                    msg_to_send = f"{instruction}\n\nPREVIOUS ATTEMPT FAILED: {last_error}\nPlease ensure your response is valid JSON with exactly 1 or 0 in the 'response' field."
                
                response, msg_history, info = get_response_from_llm(
                    msg=msg_to_send,
                    model=self.model,
                    msg_history=msg_history if msg_history else [],
                )
                
                # Extract prediction from JSON using flexible extraction
                # First try from history (more reliable)
                extracted = self._extract_from_history(msg_history)
                
                # Fallback to direct response extraction
                if not extracted:
                    extracted = _extract_json_flexible(response)
                
                if extracted and "response" in extracted:
                    prediction = extracted["response"]
                    normalized = self._normalize_prediction(prediction)
                    if normalized is not None:
                        self.log_fn(f"Successfully graded: response={normalized} (attempt {attempt + 1})")
                        return normalized, msg_history
                    else:
                        last_error = f"Invalid prediction value: {prediction} (type: {type(prediction).__name__})"
                        self.log_fn(f"{last_error}, retrying... (attempt {attempt + 1}/{self.max_retries})")
                else:
                    last_error = "No valid JSON with 'response' field found"
                    # Log a preview of what we received for debugging
                    preview = response[:200] if response else "(empty response)"
                    self.log_fn(f"{last_error}. Preview: {preview!r}, retrying... (attempt {attempt + 1}/{self.max_retries})")
                    
            except Exception as e:
                last_error = str(e)
                self.log_fn(f"Error on attempt {attempt + 1}: {e}")
                if attempt == self.max_retries - 1:
                    break
        
        # Fallback: return "0" if all retries failed
        self.log_fn(f"All {self.max_retries} retries failed, returning default prediction 0")
        return "0", msg_history
