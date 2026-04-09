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
    4. Relaxed JSON pattern matching for malformed responses
    5. Line-by-line JSON object extraction
    6. Bracket matching from the end of text
    7. LLM-based JSON repair for malformed responses
    8. Direct response value extraction from malformed JSON
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
    
    # Strategy 4: Relaxed pattern - look for any JSON object with response field
    # Handles cases where JSON might span multiple lines with extra whitespace
    relaxed_pattern = r'\{[^}]*"response"\s*:\s*(?:1|0|"1"|"0")[^}]*\}'
    for match in re.finditer(relaxed_pattern, text, re.DOTALL):
        try:
            parsed = json.loads(match.group(0))
            if "response" in parsed:
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
                            return parsed
                    except json.JSONDecodeError:
                        # Try to extract just the response field if full parse fails
                        try:
                            response_match = re.search(r'"response"\s*:\s*(\d)', candidate)
                            if response_match:
                                return {"response": int(response_match.group(1))}
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
            return {"response": int(val)}
    
    # Strategy 8: Look for standalone 0 or 1 at the end of the response
    # This handles cases where the model just outputs the answer
    standalone_pattern = r'(?:^|\s)([01])(?:\s*$|\s*\.)'
    match = re.search(standalone_pattern, text.strip())
    if match:
        val = match.group(1)
        if val in ('0', '1'):
            return {"response": int(val)}
    
    # Strategy 9: Look for "correct" or "incorrect" keywords
    text_lower = text.lower()
    if '"correct"' in text_lower or 'correct": true' in text_lower or 'correct": 1' in text_lower:
        return {"response": 1}
    if '"incorrect"' in text_lower or 'correct": false' in text_lower or 'correct": 0' in text_lower:
        return {"response": 0}
    
    # Strategy 10: LLM-based JSON repair for severely malformed responses
    # This is a last resort when all other strategies fail
    try:
        repair_prompt = f"""The following text contains a grading response that should be in JSON format with a "response" field (1 for correct, 0 for incorrect), but it's malformed. Extract the intended response value.

Malformed text:
{text[:2000]}

Respond with ONLY a JSON object in this exact format:
<json>
{{
    "response": 1 or 0
}}
</json>"""
        
        repair_response, _, _ = get_response_from_llm(
            msg=repair_prompt,
            model=EVAL_MODEL,
            msg_history=[],
        )
        extracted_repair = _extract_jsons(repair_response)
        if extracted_repair:
            last_entry = extracted_repair[-1]
            if "response" in last_entry:
                return last_entry
    except Exception:
        pass
    
    return None


def _normalize_prediction(prediction: Any) -> str:
    """Normalize prediction value to '0' or '1'.
    
    Handles various formats: integers, strings, booleans.
    Uses strict validation to ensure only valid 0/1 values are accepted.
    Returns '0' as default for invalid values.
    """
    if prediction is None:
        return "0"
    
    # Handle boolean directly
    if isinstance(prediction, bool):
        return "1" if prediction else "0"
    
    # Handle numeric types directly - only exactly 1 is correct
    if isinstance(prediction, (int, float)):
        return "1" if prediction == 1 else "0"
    
    # Convert to string and clean
    pred_str = str(prediction).strip().lower()
    
    # Handle boolean-like values - these indicate correctness
    if pred_str in ("true", "1", "yes", "correct", "right"):
        return "1"
    if pred_str in ("false", "0", "no", "incorrect", "wrong"):
        return "0"
    
    # Try numeric conversion for string numbers
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
- The grading guidelines are the PRIMARY and FINAL criteria for correctness
- A student's answer can be mathematically equivalent to the correct solution but still be marked INCORRECT if it violates the grading guidelines
- Conversely, a student's answer that follows the grading guidelines should be marked CORRECT even if it looks different from the expected solution
- Pay special attention to:
  * Required formats (e.g., simplified fractions, specific units)
  * Required steps or methods
  * Constraints on the solution approach
  * Presentation requirements
  * Any specific constraints mentioned in the guidelines
- For partial credit problems: award full credit (1) only if ALL requirements are met
- For proof-based problems: check logical structure, not just the final conclusion
- For computational problems: verify both the method and the final answer

CRITICAL INSTRUCTIONS:
- You MUST respond with ONLY a JSON object wrapped in <json> tags
- Do NOT include any text before or after the JSON
- The "response" field MUST be exactly 1 (correct) or 0 (incorrect)
- Do NOT use quotes around the number (use 1, not "1")
- Do NOT add any additional fields to the JSON
- The reasoning should be detailed and reference specific aspects of the grading guidelines

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
        failed_responses = []  # Track failed responses for better debugging
        
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
                extracted = _extract_json_flexible(response_text)
                
                if extracted and "response" in extracted:
                    prediction = extracted["response"]
                    normalized = _normalize_prediction(prediction)
                    self.log_fn(f"Successfully extracted prediction: {prediction} -> {normalized}")
                    return normalized, msg_history
                else:
                    self.log_fn(f"Attempt {attempt + 1}: No valid JSON found in response")
                    # Store failed response for context in next retry
                    preview = response_text[:800] if response_text else "(empty)"
                    failed_responses.append(preview)
                    self.log_fn(f"Response preview: {preview!r}")
                    
                    # Prepare feedback for retry with more specific guidance
                    if attempt < self.max_retries - 1:
                        retry_guidance = [
                            "Your previous response did not contain valid JSON with a 'response' field.",
                            "",
                            "What you responded with:",
                            "---",
                            preview[:500],
                            "---",
                            "",
                            "Please respond ONLY with JSON in this exact format:",
                            "<json>",
                            "{",
                            '    "reasoning": "Your analysis here",',
                            '    "response": 1',
                            "}",
                            "</json>",
                            "",
                            "IMPORTANT:",
                            "- The response field must be exactly 1 (correct) or 0 (incorrect)",
                            '- Do NOT use quotes around the number (use 1, not "1")',
                            "- Do NOT include any text before or after the JSON",
                            "- The JSON must be wrapped in <json> tags",
                            "- Common mistakes to avoid:",
                            "  * Using 'response': '1' (with quotes) instead of 'response': 1",
                            "  * Adding extra fields beyond 'reasoning' and 'response'",
                            "  * Forgetting the <json> tags around the JSON object",
                        ]
                        current_instruction = "\n".join(retry_guidance)
                    
            except Exception as e:
                last_error = e
                self.log_fn(f"Attempt {attempt + 1}: Error during LLM call or parsing: {e}")
                if attempt == self.max_retries - 1:
                    break
        
        # Fallback: return "0" if all retries failed
        if last_error:
            self.log_fn(f"All retries failed with error: {last_error}")
        else:
            self.log_fn(f"All retries failed - could not extract valid prediction. Failed responses: {len(failed_responses)}")
        
        return "0", self._last_msg_history if self._last_msg_history else []
