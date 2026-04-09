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

# Constants for JSON extraction
JSON_TAG_START = "<json>"
JSON_TAG_END = "</json>"
MAX_RETRIES = 3
VALID_PREDICTIONS = {0, 1, "0", "1"}


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    """
    results = []
    search_from = 0
    tag_start_len = len(JSON_TAG_START)
    tag_end_len = len(JSON_TAG_END)
    
    while True:
        start = text.find(JSON_TAG_START, search_from)
        if start == -1:
            break
        end = text.find(JSON_TAG_END, start)
        if end == -1:
            break
        inner = text[start + tag_start_len:end].strip()
        search_from = end + tag_end_len
        try:
            results.append(json.loads(inner))
        except json.JSONDecodeError:
            continue
    return results or None


def _extract_from_markdown_blocks(text: str) -> dict | None:
    """Strategy 2: Extract JSON from markdown code blocks."""
    json_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    for match in re.finditer(json_block_pattern, text, re.DOTALL):
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            continue
    return None


def _extract_from_response_pattern(text: str) -> dict | None:
    """Strategy 3: Extract JSON-like structures with 'response' key."""
    json_pattern = r'\{\s*"response"\s*:[^\}]+\}'
    for match in re.finditer(json_pattern, text, re.DOTALL):
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
    return None


def _extract_from_full_schema(text: str) -> dict | None:
    """Strategy 4: Extract JSON with 'reasoning' and 'response' keys."""
    full_json_pattern = r'\{\s*"reasoning"\s*:\s*"[^"]*"\s*,\s*"response"\s*:\s*(?:0|1|\d+)\s*\}'
    for match in re.finditer(full_json_pattern, text, re.DOTALL):
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
    return None


def _extract_from_brace_matching(text: str) -> dict | None:
    """Strategy 5: Extract JSON by matching braces from the end of text."""
    last_brace_idx = text.rfind('}')
    if last_brace_idx == -1:
        return None
    
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
    return None


def _extract_from_relaxed_pattern(text: str) -> dict | None:
    """Strategy 6: Extract using relaxed pattern matching for malformed JSON."""
    relaxed_pattern = r'[\{\s]*["\']reasoning["\']\s*:\s*["\']([^"\']*)["\']\s*,\s*["\']response["\']\s*:\s*(\d+)\s*[\}\s]*'
    match = re.search(relaxed_pattern, text, re.DOTALL)
    if match:
        return {
            "reasoning": match.group(1),
            "response": int(match.group(2))
        }
    return None


def _extract_from_numeric_response(text: str) -> dict | None:
    """Strategy 7: Extract standalone numeric response (0 or 1) as last resort."""
    response_pattern = r'(?:^|\s|[\:\,\(\[])\s*(0|1)\s*(?:$|\s|[\,\;\.\)\]])'
    matches = list(re.finditer(response_pattern, text))
    if matches:
        # Use the last match as it's likely the final answer
        last_match = matches[-1]
        return {
            "reasoning": "Extracted from text pattern matching",
            "response": int(last_match.group(1))
        }
    return None


def _extract_json_flexible(text: str) -> dict | None:
    """Extract JSON with multiple fallback strategies.
    
    Tries multiple approaches in order:
    1. Standard <json>...</json> blocks
    2. Markdown code blocks with json
    3. Raw JSON objects with "response" key
    4. JSON objects with "reasoning" key
    5. Any valid JSON object at the end of the text
    6. Relaxed pattern matching for malformed JSON
    7. Standalone numeric response (0 or 1)
    """
    extraction_strategies = [
        # Strategy 1: Standard <json> tags
        lambda t: _extract_jsons(t)[-1] if _extract_jsons(t) else None,
        # Strategy 2: Markdown code blocks
        _extract_from_markdown_blocks,
        # Strategy 3: JSON with "response" key
        _extract_from_response_pattern,
        # Strategy 4: Full schema with "reasoning" and "response"
        _extract_from_full_schema,
        # Strategy 5: Brace matching from end of text
        _extract_from_brace_matching,
        # Strategy 6: Relaxed pattern matching
        _extract_from_relaxed_pattern,
        # Strategy 7: Numeric response fallback
        _extract_from_numeric_response,
    ]
    
    for strategy in extraction_strategies:
        try:
            result = strategy(text)
            if result is not None:
                return result
        except Exception:
            continue
    
    return None


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.max_retries = MAX_RETRIES

    def _build_instruction(self, inputs: dict) -> str:
        """Build the grading instruction prompt from input fields.
        
        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer
            
        Returns:
            Formatted instruction string for the LLM
        """
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        return f"""You are an expert {domain} grader evaluating student solutions.

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

IMPORTANT: You must respond with valid JSON in the following format:
<json>
{{
    "reasoning": "Your detailed step-by-step analysis here",
    "response": 1 or 0
}}
</json>

The "response" field must be either 1 (correct) or 0 (incorrect)."""

    def _extract_prediction(self, msg_history: list[dict]) -> tuple[str | None, str]:
        """Extract prediction from LLM response message history.
        
        Args:
            msg_history: List of message dicts from LLM conversation
            
        Returns:
            Tuple of (prediction_value, response_text)
            prediction_value is None if extraction failed
        """
        response_text = msg_history[-1].get("text", "") if msg_history else ""
        extracted = _extract_json_flexible(response_text)
        
        if extracted and "response" in extracted:
            prediction = extracted["response"]
            if prediction in VALID_PREDICTIONS:
                return str(prediction), response_text
        
        return None, response_text

    def _log_response_snippet(self, response_text: str, max_len: int = 200) -> None:
        """Log a snippet of the response for debugging purposes.
        
        Args:
            response_text: Full response text from LLM
            max_len: Maximum length of snippet to log
        """
        if response_text:
            snippet = response_text[:max_len] + "..." if len(response_text) > max_len else response_text
            self.log_fn(f"Response snippet: {snippet}")

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        instruction = self._build_instruction(inputs)
        msg_history: list[dict] = []
        
        # Try with retries for robustness
        for attempt in range(self.max_retries):
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=[],
                )
                
                # Extract prediction from JSON using flexible extraction
                prediction, response_text = self._extract_prediction(msg_history)
                
                if prediction is not None:
                    self.log_fn(f"Successfully extracted prediction: {prediction}")
                    return prediction, msg_history
                else:
                    self.log_fn(f"No valid JSON found in response (attempt {attempt + 1}), retrying...")
                    self._log_response_snippet(response_text)
                    
            except Exception as e:
                self.log_fn(f"Error on attempt {attempt + 1}: {e}")
                if attempt == self.max_retries - 1:
                    break
        
        # Fallback: return "0" if all retries failed
        self.log_fn("All retries failed, returning default prediction 0")
        return "0", msg_history
