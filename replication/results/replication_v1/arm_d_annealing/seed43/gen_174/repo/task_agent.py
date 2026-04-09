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
    
    # Also try to find JSON in code blocks as a secondary source
    if not results:
        code_block_pattern = r'```(?:json)?\s*(\{[\s\S]*?\})\s*```'
        matches = re.findall(code_block_pattern, text, re.DOTALL)
        for match in matches:
            try:
                parsed = json.loads(match)
                if isinstance(parsed, dict):
                    results.append(parsed)
            except json.JSONDecodeError:
                json_obj = _extract_balanced_json(match)
                if json_obj:
                    results.append(json_obj)
    
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
    candidates = []
    
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
                if brace_depth > 0:
                    brace_depth -= 1
                    if brace_depth == 0 and start_idx != -1:
                        candidates.append((start_idx, i + 1))
                        start_idx = -1
    
    # Try candidates from largest to smallest (prefer complete JSON objects)
    candidates.sort(key=lambda x: x[1] - x[0], reverse=True)
    
    for start, end in candidates:
        try:
            result = json.loads(text[start:end])
            if isinstance(result, dict):
                return result
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
    
    # Strategy 4: Look for single-quoted JSON or unquoted keys
    if not results:
        # Try to find patterns like response: "value" or 'response': 'value'
        single_quote_pattern = r"'response'\s*:\s*'([^']*)'"
        single_match = re.search(single_quote_pattern, text, re.DOTALL)
        if single_match:
            results.append({"response": single_match.group(1)})
        
        # Try unquoted key pattern
        unquoted_pattern = r'response\s*:\s*"([^"]*)"'
        unquoted_match = re.search(unquoted_pattern, text, re.DOTALL)
        if unquoted_match:
            results.append({"response": unquoted_match.group(1)})
    
    # Strategy 5: Look for grade/assessment at end of response
    if not results:
        # Common patterns at end of text
        end_patterns = [
            r'(?:grade|assessment|score|verdict)\s*[:=]\s*["\']?([^"\'\n]+)["\']?',
            r'(?:the answer is|i would say|final grade)[:\s]+["\']?([^"\'\n]+)["\']?',
        ]
        for pattern in end_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                results.append({"response": match.group(1).strip()})
                break
    
    return results or None


def _normalize_prediction(prediction: str) -> str:
    """Normalize prediction string for consistent comparison.
    
    Handles common variations like case differences, extra whitespace,
    and punctuation differences. Also handles numeric scores (0-100)
    by mapping them to grade categories.
    """
    if not prediction:
        return "None"
    
    # Strip whitespace and normalize case
    normalized = prediction.strip()
    
    # Try to parse as numeric score (0-100)
    try:
        # Remove any non-numeric characters except decimal point and minus
        numeric_str = ''.join(c for c in normalized if c.isdigit() or c in '.-')
        if numeric_str:
            score = float(numeric_str)
            # Map numeric score to grade category
            if score >= 80:
                return "Correct"
            elif score >= 40:
                return "Partial"
            else:
                return "Incorrect"
    except (ValueError, TypeError):
        pass
    
    # Handle common grade variations
    grade_map = {
        "correct": "Correct",
        "partial": "Partial",
        "partially correct": "Partial",
        "partially": "Partial",
        "mostly correct": "Partial",
        "mostly": "Partial",
        "incorrect": "Incorrect",
        "wrong": "Incorrect",
        "right": "Correct",
        "true": "Correct",
        "false": "Incorrect",
        "full": "Correct",
        "full credit": "Correct",
        "no credit": "Incorrect",
        "zero": "Incorrect",
        "half": "Partial",
        "half credit": "Partial",
        "satisfactory": "Correct",
        "unsatisfactory": "Incorrect",
        "complete": "Correct",
        "incomplete": "Partial",
        "accept": "Correct",
        "accepted": "Correct",
        "reject": "Incorrect",
        "rejected": "Incorrect",
        "pass": "Correct",
        "fail": "Incorrect",
        "yes": "Correct",
        "no": "Incorrect",
        "good": "Correct",
        "bad": "Incorrect",
        "excellent": "Correct",
        "poor": "Incorrect",
        "valid": "Correct",
        "invalid": "Incorrect",
    }
    
    lower_pred = normalized.lower()
    
    # Check for exact match first
    if lower_pred in grade_map:
        return grade_map[lower_pred]
    
    # Check for multi-word matches (longer keys first for better matching)
    sorted_keys = sorted(grade_map.keys(), key=len, reverse=True)
    for key in sorted_keys:
        if key in lower_pred:
            return grade_map[key]
    
    # Check for partial match patterns
    if "partial" in lower_pred or "part" in lower_pred or "mostly" in lower_pred or "some" in lower_pred:
        return "Partial"
    if "correct" in lower_pred or "right" in lower_pred or "accept" in lower_pred or "valid" in lower_pred:
        return "Correct"
    if "incorrect" in lower_pred or "wrong" in lower_pred or "error" in lower_pred or "reject" in lower_pred or "invalid" in lower_pred:
        return "Incorrect"
    
    return normalized


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.max_retries = 3
        self._extraction_stats = {"success": 0, "fallback": 0, "failure": 0}

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
When assigning grades, use EXACTLY one of these three categories:
- **Correct**: The answer matches the solution or uses an equivalent valid approach with correct reasoning and final result. Award this for fully correct answers.
- **Partial**: The answer shows some correct reasoning but has minor errors, incomplete steps, or partially correct results. Award this for answers that are on the right track but not fully correct.
- **Incorrect**: The answer contains fundamental errors, wrong approach, or completely wrong results. Award this for answers that are substantially wrong.

## Response Format (REQUIRED - STRICT)
You MUST respond with valid JSON wrapped in <json>...</json> tags. Do not include any text outside the JSON tags.

<json>
{{
    "reasoning": "Your detailed step-by-step analysis and reasoning...",
    "response": "Correct"
}}
</json>

IMPORTANT RULES:
1. The 'response' field MUST contain ONLY one of: "Correct", "Partial", or "Incorrect" (exactly as written, with capital first letter)
2. Do NOT use any other values like "True", "False", numbers, or variations like "correct" (lowercase)
3. The 'reasoning' field should contain your detailed analysis
4. Ensure your JSON is valid - check for proper quotes, commas, and braces
5. Do not include any text before <json> or after </json>"""

        if is_retry:
            return f"""ERROR: Your previous response did not contain valid JSON with a 'response' field, or the JSON was malformed, or the response value was not one of the allowed values.

You MUST respond with ONLY valid JSON wrapped in <json>...</json> tags. Do not include any text before or after the JSON tags.

The 'response' field MUST be exactly one of: "Correct", "Partial", or "Incorrect"

Correct format example:
<json>
{{
    "reasoning": "The student correctly applied the quadratic formula and arrived at the right answer. All steps are valid.",
    "response": "Correct"
}}
</json>

Another example for partial credit:
<json>
{{
    "reasoning": "The student started with the right approach but made an arithmetic error in the final step.",
    "response": "Partial"
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
        
        if extracted:
            last_json = extracted[-1]
            if "response" in last_json:
                prediction = str(last_json["response"])
            if "reasoning" in last_json:
                reasoning = str(last_json["reasoning"])
        
        # Normalize the prediction
        prediction = _normalize_prediction(prediction)
        
        return prediction, reasoning, method

    def get_extraction_stats(self) -> dict:
        """Return statistics about extraction methods used."""
        total = sum(self._extraction_stats.values())
        if total == 0:
            return {"total": 0, "success_rate": 0.0}
        return {
            "total": total,
            "success": self._extraction_stats["success"],
            "fallback": self._extraction_stats["fallback"],
            "failure": self._extraction_stats["failure"],
            "success_rate": (self._extraction_stats["success"] + self._extraction_stats["fallback"]) / total,
        }

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
                
                if prediction != "None":
                    self.log_fn(f"Successfully extracted prediction: {prediction} (method: {method})")
                    if reasoning:
                        self.log_fn(f"Reasoning: {reasoning[:200]}...")
                    break
                else:
                    self.log_fn(f"Attempt {attempt + 1}: Failed to extract prediction, retrying...")
                    # Add feedback for retry
                    if attempt < self.max_retries - 1:
                        instruction = self._build_grading_prompt(inputs, is_retry=True)
                    
            except Exception as e:
                self.log_fn(f"Error in attempt {attempt + 1}: {e}")
                if attempt == self.max_retries - 1:
                    prediction = f"Error: {e}"
        
        # Final validation: ensure prediction is one of the allowed values
        allowed = {"Correct", "Partial", "Incorrect"}
        if prediction not in allowed and not prediction.startswith("Error:"):
            self.log_fn(f"Warning: Invalid prediction '{prediction}', defaulting to 'Incorrect'")
            prediction = "Incorrect"
        
        return str(prediction), msg_history
