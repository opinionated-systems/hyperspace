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


def _extract_jsons(text: str) -> list[dict]:
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
    return results


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
    3. Bracket-balanced extraction for any JSON with "response" key
    4. Any valid JSON object at the end of the text
    5. Find any JSON object containing "response" field
    """
    # Strategy 1: Standard <json> tags (most reliable)
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

    # Strategy 3: Bracket-balanced extraction for JSON with "response" key
    for match in re.finditer(r'"response"\s*:', text):
        # Walk back to find the opening brace
        for i in range(match.start() - 1, -1, -1):
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
            elif text[i] == '}':
                break

    # Strategy 4: Last resort - find any JSON object at the end
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

    # Strategy 5: Find any JSON object containing "response" field (broader search)
    # Look for patterns like {"response": ...} or {"reasoning": ..., "response": ...}
    json_pattern = r'\{[^{}]*"response"[^{}]*\}'
    for match in re.finditer(json_pattern, text, re.DOTALL):
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict) and "response" in parsed:
                return parsed
        except json.JSONDecodeError:
            continue

    # Strategy 6: Try to find and parse any JSON-like structure
    # Look for the last occurrence of a pattern that looks like JSON
    brace_pattern = r'\{[^{}]*\}'
    for match in re.finditer(brace_pattern, text, re.DOTALL):
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict):
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

    def _build_prompt(self, inputs: dict) -> str:
        """Build the grading prompt from input fields."""
        domain = inputs.get("domain", "Mathematics")
        return f"""You are an expert {domain} grader evaluating student solutions.

Your task is to grade a student's answer by comparing it to the correct solution and following the grading guidelines.

PROBLEM:
{inputs.get("problem", "")}

CORRECT SOLUTION:
{inputs.get("solution", "")}

GRADING GUIDELINES:
{inputs.get("grading_guidelines", "")}

STUDENT'S ANSWER:
{inputs.get("student_answer", "")}

Think step by step:
1. Analyze what the problem is asking for
2. Review the correct solution approach
3. Compare the student's answer to the correct solution
4. Check if the student followed the grading guidelines
5. Determine if the student's answer is correct (1) or incorrect (0)

IMPORTANT: Be strict in your evaluation. The student's answer must be fully correct to receive a 1. Partial credit is not given.

Respond in JSON format with the following schema:
<json>
{{
    "reasoning": "Your detailed step-by-step analysis here",
    "response": 1 or 0
}}
</json>

The "response" field must be either 1 (correct) or 0 (incorrect). Do not include any other text outside the JSON block."""

    def _validate_prediction(self, value) -> str | None:
        """Validate and normalize prediction value.
        
        Handles various input types including integers, strings, booleans,
        and converts them to standardized "0" or "1" strings.
        """
        # Handle boolean values
        if isinstance(value, bool):
            return "1" if value else "0"
        
        # Handle numeric values (int, float)
        if isinstance(value, (int, float)):
            if value == 0:
                return "0"
            if value == 1:
                return "1"
            return None
        
        # Handle string values
        if isinstance(value, str):
            value = value.strip().lower()
            if value in ("0", "false", "no", "incorrect", "wrong"):
                return "0"
            if value in ("1", "true", "yes", "correct", "right"):
                return "1"
            # Try to parse as number
            try:
                num = float(value)
                if num == 0:
                    return "0"
                if num == 1:
                    return "1"
            except ValueError:
                pass
        
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

        for attempt in range(self.max_retries):
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=[],
                )

                # Get the last assistant message
                last_msg = msg_history[-1] if msg_history else None
                if not last_msg or last_msg.get("role") != "assistant":
                    self.log_fn(f"Attempt {attempt + 1}: No assistant message found, retrying...")
                    continue

                response_text = last_msg.get("text", "")
                extracted = _extract_json_flexible(response_text)

                if extracted:
                    if "response" in extracted:
                        prediction = self._validate_prediction(extracted["response"])
                        if prediction is not None:
                            self.log_fn(f"Attempt {attempt + 1}: Successfully extracted prediction {prediction}")
                            return prediction, msg_history
                        self.log_fn(f"Attempt {attempt + 1}: Invalid prediction value: {extracted['response']}, retrying...")
                    else:
                        self.log_fn(f"Attempt {attempt + 1}: JSON found but no 'response' key: {list(extracted.keys())}, retrying...")
                else:
                    self.log_fn(f"Attempt {attempt + 1}: No valid JSON found in response, retrying...")
                    # Log a snippet of the response for debugging
                    snippet = response_text[:200] if response_text else "(empty)"
                    self.log_fn(f"Response snippet: {snippet!r}")

            except Exception as e:
                self.log_fn(f"Error on attempt {attempt + 1}: {e}")
                if attempt == self.max_retries - 1:
                    break

        self.log_fn("All retries failed, returning default prediction 0")
        return "0", msg_history
