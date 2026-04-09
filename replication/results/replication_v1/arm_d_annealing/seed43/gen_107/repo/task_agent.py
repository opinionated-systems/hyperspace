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
    Also handles nested JSON objects by tracking brace depth.
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
        
        # Try to parse the inner content as JSON
        try:
            results.append(json.loads(inner))
        except json.JSONDecodeError:
            # Try to extract JSON by finding balanced braces
            json_obj = _extract_balanced_json(inner)
            if json_obj:
                results.append(json_obj)
            continue
    return results or None


def _extract_balanced_json(text: str) -> dict | None:
    """Extract a JSON object by tracking brace depth.
    
    This handles cases where the JSON content contains nested braces
    that might confuse simple regex-based extraction.
    """
    start_idx = -1
    brace_depth = 0
    in_string = False
    escape_next = False
    
    for i, char in enumerate(text):
        if escape_next:
            escape_next = False
            continue
        if char == '\\' and in_string:
            escape_next = True
            continue
        if char == '"' and not in_string:
            in_string = True
            continue
        if char == '"' and in_string:
            in_string = False
            continue
        if not in_string:
            if char == '{':
                if brace_depth == 0:
                    start_idx = i
                brace_depth += 1
            elif char == '}':
                brace_depth -= 1
                if brace_depth == 0 and start_idx != -1:
                    try:
                        return json.loads(text[start_idx:i+1])
                    except json.JSONDecodeError:
                        continue
    return None


def _extract_json_with_regex(text: str) -> list[dict] | None:
    """Fallback JSON extraction using multiple strategies for malformed responses."""
    results = []
    
    # Strategy 1: Try to find JSON objects in code blocks
    code_block_pattern = r'```(?:json)?\s*(\{[\s\S]*?\})\s*```'
    matches = re.findall(code_block_pattern, text, re.DOTALL)
    for match in matches:
        try:
            # Try to parse the matched content
            parsed = json.loads(match)
            if isinstance(parsed, dict):
                results.append(parsed)
        except json.JSONDecodeError:
            # Try balanced brace extraction
            balanced = _extract_balanced_json(match)
            if balanced:
                results.append(balanced)
    
    # Strategy 2: Look for any JSON-like structure with balanced braces
    if not results:
        balanced = _extract_balanced_json(text)
        if balanced:
            results.append(balanced)
    
    # Strategy 3: Extract key-value pairs for response and reasoning
    if not results:
        response_match = re.search(r'"response"\s*:\s*"([^"]*(?:\\.[^"]*)*)"', text, re.DOTALL)
        reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]*(?:\\.[^"]*)*)"', text, re.DOTALL)
        
        extracted = {}
        if response_match:
            extracted["response"] = response_match.group(1).replace('\\"', '"').replace('\\n', '\n')
        if reasoning_match:
            extracted["reasoning"] = reasoning_match.group(1).replace('\\"', '"').replace('\\n', '\n')
        
        if extracted:
            results.append(extracted)
    
    # Strategy 4: Look for plain text grade mentions when JSON is missing
    if not results:
        extracted = _extract_grade_from_text(text)
        if extracted:
            results.append(extracted)
    
    return results or None


def _extract_grade_from_text(text: str) -> dict | None:
    """Extract grade from plain text when JSON is not available.
    
    This handles cases where the model provides a grade in plain text
    without proper JSON formatting.
    """
    text_lower = text.lower()
    extracted = {}
    
    # Look for grade keywords with context
    grade_patterns = [
        (r'(?:grade|score|assessment|evaluation|rating)\s*[:=]\s*(correct|partial|incorrect|wrong|right|true|false)', 'grade_with_label'),
        (r'(?:the\s+answer\s+is|this\s+is|i\s+rate\s+this\s+as)\s+(correct|partial|incorrect|wrong|right)', 'grade_with_phrase'),
        (r'\b(correct|partial|incorrect)\b', 'standalone_grade'),
    ]
    
    for pattern, pattern_type in grade_patterns:
        match = re.search(pattern, text_lower, re.IGNORECASE)
        if match:
            grade = match.group(1).lower()
            # Normalize the grade
            if grade in ('correct', 'right', 'true'):
                extracted["response"] = "Correct"
            elif grade in ('partial', 'partially'):
                extracted["response"] = "Partial"
            elif grade in ('incorrect', 'wrong', 'false'):
                extracted["response"] = "Incorrect"
            
            # Extract reasoning from surrounding context
            if not extracted.get("reasoning"):
                # Get text around the match for reasoning
                start = max(0, match.start() - 200)
                end = min(len(text), match.end() + 200)
                context = text[start:end].strip()
                extracted["reasoning"] = f"[Extracted from plain text] {context}"
            break
    
    return extracted if extracted else None


def _normalize_prediction(prediction: str) -> str:
    """Normalize prediction string for consistent comparison.
    
    Handles common variations like case differences, extra whitespace,
    and punctuation differences.
    """
    if not prediction:
        return "None"
    
    # Strip whitespace and normalize case
    normalized = prediction.strip()
    
    # Handle common grade variations
    grade_map = {
        "correct": "Correct",
        "partial": "Partial",
        "partially correct": "Partial",
        "incorrect": "Incorrect",
        "wrong": "Incorrect",
        "right": "Correct",
        "true": "Correct",
        "false": "Incorrect",
    }
    
    lower_pred = normalized.lower()
    if lower_pred in grade_map:
        return grade_map[lower_pred]
    
    return normalized


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "", max_retries: int = 3) -> None:
        self.model = model
        self.log_fn = logger.info
        self.max_retries = max_retries
        self._extraction_stats = {"success": 0, "fallback": 0, "plaintext": 0, "failure": 0}
        self._log_file = log_file
    
    def get_extraction_stats(self) -> dict:
        """Return the current extraction statistics."""
        return dict(self._extraction_stats)
    
    def reset_extraction_stats(self) -> None:
        """Reset extraction statistics to zero."""
        self._extraction_stats = {"success": 0, "fallback": 0, "plaintext": 0, "failure": 0}

    def _build_grading_prompt(self, inputs: dict, is_retry: bool = False) -> str:
        """Build a structured prompt for the grading task with chain-of-thought."""
        domain = inputs.get("domain", "unknown")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        base_prompt = f"""You are an expert grader for {domain} problems. Your task is to evaluate a student's answer against the correct solution and grading guidelines.

## Problem Statement
{problem}

## Correct Solution
{solution}

## Grading Guidelines
{guidelines}

## Student's Answer
{student_answer}

## Instructions
1. First, analyze the student's answer step by step. Compare it against the correct solution.
2. Identify any errors, omissions, or alternative valid approaches.
3. Consider the grading guidelines carefully.
4. Provide your reasoning for the grade you will assign.
5. Finally, provide your grade/assessment in the JSON format below.

## Grading Rubric
When assigning grades, consider:
- **Correct**: The answer matches the solution or uses an equivalent valid approach with correct reasoning and final result.
- **Partial**: The answer shows some correct reasoning but has minor errors, incomplete steps, or partially correct results.
- **Incorrect**: The answer contains fundamental errors, wrong approach, or completely wrong results.

## Response Format (REQUIRED)
You MUST respond with valid JSON wrapped in <json>...</json> tags. Do not include any text outside the JSON tags.

<json>
{{
    "reasoning": "Your detailed step-by-step analysis and reasoning...",
    "response": "Your final grade/assessment (e.g., 'Correct', 'Partial', 'Incorrect', or a numeric score)"
}}
</json>

IMPORTANT: Ensure your JSON is valid and properly formatted. The 'response' field should contain only the grade, not the reasoning."""

        if is_retry:
            return f"""ERROR: Your previous response did not contain valid JSON with a 'response' field, or the JSON was malformed.

You MUST respond with ONLY valid JSON wrapped in <json>...</json> tags. Do not include any text before or after the JSON tags.

Correct format:
<json>
{{
    "reasoning": "Your detailed analysis here...",
    "response": "Correct"
}}
</json>

Now try again with the original task:

{base_prompt}"""
        
        return base_prompt

    def _extract_prediction(self, text: str) -> tuple[str, str, str]:
        """Extract prediction and reasoning from response text.
        
        Returns:
            (prediction, reasoning, method) tuple where method indicates extraction success
        """
        prediction = "None"
        reasoning = ""
        method = "failure"
        
        # Try primary extraction method
        extracted = _extract_jsons(text)
        if extracted:
            method = "success"
        else:
            # Fallback to regex extraction
            extracted = _extract_json_with_regex(text)
            if extracted:
                method = "fallback"
                # Check if we used plain text extraction
                if any("[Extracted from plain text]" in str(v) for v in extracted[-1].values()):
                    method = "plaintext"
        
        if extracted:
            last_json = extracted[-1]
            if "response" in last_json:
                prediction = str(last_json["response"])
            if "reasoning" in last_json:
                reasoning = str(last_json["reasoning"])
        
        # Normalize the prediction
        prediction = _normalize_prediction(prediction)
        
        return prediction, reasoning, method

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        instruction = self._build_grading_prompt(inputs, is_retry=False)
        
        msg_history = []
        prediction = "None"
        reasoning = ""
        
        # Track all attempts for better debugging
        attempt_details = []
        
        # Retry loop for robust extraction
        for attempt in range(self.max_retries):
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=msg_history if attempt > 0 else [],
                )
                
                last_text = msg_history[-1]["text"] if msg_history else ""
                prediction, reasoning, method = self._extract_prediction(last_text)
                
                # Update statistics
                self._extraction_stats[method] += 1
                
                # Log detailed attempt info
                attempt_info = {
                    "attempt": attempt + 1,
                    "method": method,
                    "prediction": prediction,
                    "response_length": len(last_text),
                    "has_reasoning": bool(reasoning),
                }
                attempt_details.append(attempt_info)
                
                if prediction != "None":
                    self.log_fn(f"[Attempt {attempt + 1}] Successfully extracted prediction: {prediction} (method: {method})")
                    if reasoning:
                        self.log_fn(f"[Attempt {attempt + 1}] Reasoning preview: {reasoning[:200]}...")
                    # Log extraction stats summary
                    self.log_fn(f"Extraction stats: {dict(self._extraction_stats)}")
                    break
                else:
                    self.log_fn(f"[Attempt {attempt + 1}] Failed to extract prediction (method: {method})")
                    # Log a preview of the problematic response for debugging
                    preview = last_text[:300].replace('\n', ' ')
                    self.log_fn(f"[Attempt {attempt + 1}] Response preview: {preview}...")
                    
                    # Add feedback for retry
                    if attempt < self.max_retries - 1:
                        instruction = self._build_grading_prompt(inputs, is_retry=True)
                    
            except Exception as e:
                error_msg = f"[Attempt {attempt + 1}] Error: {type(e).__name__}: {str(e)[:100]}"
                self.log_fn(error_msg)
                attempt_details.append({"attempt": attempt + 1, "error": str(e)})
                if attempt == self.max_retries - 1:
                    prediction = f"Error: {e}"
        
        # Log summary of all attempts if we failed
        if prediction == "None" or str(prediction).startswith("Error:"):
            self.log_fn(f"All {self.max_retries} attempts failed. Details: {attempt_details}")
        
        return str(prediction), msg_history
