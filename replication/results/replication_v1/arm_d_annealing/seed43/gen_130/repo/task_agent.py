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
            # Try fixing common JSON errors
            fixed = _fix_malformed_json(match)
            if fixed:
                results.append(fixed)
    
    # Strategy 2: Look for any JSON-like structure with balanced braces
    if not results:
        balanced = _extract_balanced_json(text)
        if balanced:
            results.append(balanced)
        # Try fixing common JSON errors in the whole text
        fixed = _fix_malformed_json(text)
        if fixed:
            results.append(fixed)
    
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
    
    # Strategy 4: Look for simple text patterns like "Grade: Correct" or "Answer: Partial"
    if not results:
        grade_patterns = [
            r'[Gg]rade[:\s]+([Cc]orrect|[Pp]artial|[Ii]ncorrect)',
            r'[Aa]nswer[:\s]+([Cc]orrect|[Pp]artial|[Ii]ncorrect)',
            r'[Ss]core[:\s]+([Cc]orrect|[Pp]artial|[Ii]ncorrect)',
            r'[Rr]esult[:\s]+([Cc]orrect|[Pp]artial|[Ii]ncorrect)',
            r'[Aa]ssessment[:\s]+([Cc]orrect|[Pp]artial|[Ii]ncorrect)',
            r'[Ee]valuation[:\s]+([Cc]orrect|[Pp]artial|[Ii]ncorrect)',
        ]
        for pattern in grade_patterns:
            match = re.search(pattern, text)
            if match:
                results.append({"response": match.group(1).capitalize()})
                break
    
    return results or None


def _fix_malformed_json(text: str) -> dict | None:
    """Attempt to fix common JSON formatting errors.
    
    Handles:
    - Unescaped quotes within string values
    - Trailing commas
    - Missing quotes around keys
    - Single quotes instead of double quotes
    """
    # Try to extract a JSON object first
    json_match = re.search(r'\{[\s\S]*\}', text)
    if not json_match:
        return None
    
    json_str = json_match.group(0)
    
    # Fix 1: Remove trailing commas before closing braces/brackets
    json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
    
    # Fix 2: Replace single quotes with double quotes (carefully)
    # Only replace single quotes that appear to be used as JSON delimiters
    json_str = re.sub(r"(?<!\\)'", '"', json_str)
    
    # Fix 3: Try to fix unescaped quotes in string values
    # This is a heuristic approach - look for patterns like "key": "value"with"quotes"
    def fix_unescaped_quotes(match):
        key = match.group(1)
        value = match.group(2)
        # Escape unescaped double quotes in the value
        # Replace " that are not preceded by \ with \"
        fixed_value = re.sub(r'(?<!\\)"', r'\\"', value)
        return f'"{key}": "{fixed_value}"'
    
    # Try to fix string values with unescaped quotes
    json_str = re.sub(r'"(\w+)"\s*:\s*"([^"]*)"(?=\s*,|\s*\})', fix_unescaped_quotes, json_str)
    
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass
    
    # Last resort: try to extract just the response and reasoning fields manually
    result = {}
    
    # Extract response field
    response_patterns = [
        r'"response"\s*:\s*"([^"]*(?:\\.[^"]*)*)"',
        r'"response"\s*:\s*([^,\}]+)',
    ]
    for pattern in response_patterns:
        match = re.search(pattern, json_str, re.DOTALL)
        if match:
            result["response"] = match.group(1).strip().strip('"').replace('\\"', '"')
            break
    
    # Extract reasoning field
    reasoning_patterns = [
        r'"reasoning"\s*:\s*"([^"]*(?:\\.[^"]*)*)"',
        r'"reasoning"\s*:\s*"([\s\S]*?)"(?=\s*,\s*"|\s*\})',
    ]
    for pattern in reasoning_patterns:
        match = re.search(pattern, json_str, re.DOTALL)
        if match:
            result["reasoning"] = match.group(1).replace('\\"', '"').replace('\\n', '\n')
            break
    
    return result if result else None


def _normalize_prediction(prediction: str) -> str:
    """Normalize prediction string for consistent comparison.
    
    Handles common variations like case differences, extra whitespace,
    punctuation differences, and numeric scores.
    """
    if not prediction:
        return "None"
    
    # Strip whitespace and normalize case
    normalized = prediction.strip()
    
    # Handle common grade variations (including partial word matches)
    grade_map = {
        "correct": "Correct",
        "partial": "Partial",
        "partially correct": "Partial",
        "partially": "Partial",
        "incorrect": "Incorrect",
        "wrong": "Incorrect",
        "right": "Correct",
        "true": "Correct",
        "false": "Incorrect",
        "yes": "Correct",
        "no": "Incorrect",
        "pass": "Correct",
        "fail": "Incorrect",
        "full": "Correct",
        "full credit": "Correct",
        "half": "Partial",
        "half credit": "Partial",
        "no credit": "Incorrect",
        "zero": "Incorrect",
        "0": "Incorrect",
        "100": "Correct",
        "100%": "Correct",
    }
    
    lower_pred = normalized.lower()
    
    # Check for exact match first
    if lower_pred in grade_map:
        return grade_map[lower_pred]
    
    # Check for numeric scores (0-100 scale)
    try:
        # Remove % sign and whitespace, then try to parse as float
        numeric_str = lower_pred.replace('%', '').replace(' ', '').strip()
        score = float(numeric_str)
        if score >= 80:
            return "Correct"
        elif score >= 40:
            return "Partial"
        else:
            return "Incorrect"
    except (ValueError, TypeError):
        pass
    
    # Check for partial/fractional scores like "1/2", "2/3", etc.
    fraction_match = re.match(r'^(\d+)\s*/\s*(\d+)$', lower_pred)
    if fraction_match:
        try:
            numerator = int(fraction_match.group(1))
            denominator = int(fraction_match.group(2))
            if denominator > 0:
                ratio = numerator / denominator
                if ratio >= 0.8:
                    return "Correct"
                elif ratio >= 0.4:
                    return "Partial"
                else:
                    return "Incorrect"
        except (ValueError, ZeroDivisionError):
            pass
    
    # Check if the prediction contains any of the grade keywords
    for keyword, grade in grade_map.items():
        if keyword in lower_pred and len(keyword) > 3:  # Avoid matching short words like "no" in "know"
            return grade
    
    return normalized


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.max_retries = 3
        self._extraction_stats = {"success": 0, "fallback": 0, "failure": 0}
        self._confidence_stats = {"High": 0, "Medium": 0, "Low": 0, "Unknown": 0}
        self._total_calls = 0

    def get_extraction_stats(self) -> dict:
        """Return extraction and confidence statistics for monitoring and debugging."""
        total = sum(self._extraction_stats.values())
        confidence_total = sum(self._confidence_stats.values())
        
        result = {
            "total_calls": self._total_calls,
            "extraction": {
                "total": total,
                "success_rate": self._extraction_stats["success"] / total if total > 0 else 0.0,
                "fallback_rate": self._extraction_stats["fallback"] / total if total > 0 else 0.0,
                "failure_rate": self._extraction_stats["failure"] / total if total > 0 else 0.0,
                "breakdown": dict(self._extraction_stats),
            },
            "confidence": {
                "total": confidence_total,
                "breakdown": dict(self._confidence_stats),
                "high_rate": self._confidence_stats["High"] / confidence_total if confidence_total > 0 else 0.0,
            },
        }
        return result

    def reset_extraction_stats(self) -> None:
        """Reset extraction and confidence statistics."""
        self._extraction_stats = {"success": 0, "fallback": 0, "failure": 0}
        self._confidence_stats = {"High": 0, "Medium": 0, "Low": 0, "Unknown": 0}
        self._total_calls = 0

    def _build_grading_prompt(self, inputs: dict, is_retry: bool = False) -> str:
        """Build a structured prompt for the grading task with chain-of-thought and confidence scoring."""
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
5. Assess your confidence in this grading decision (High/Medium/Low).
6. Finally, provide your grade/assessment in the JSON format below.

## Grading Rubric
When assigning grades, consider:
- **Correct**: The answer matches the solution or uses an equivalent valid approach with correct reasoning and final result.
- **Partial**: The answer shows some correct reasoning but has minor errors, incomplete steps, or partially correct results.
- **Incorrect**: The answer contains fundamental errors, wrong approach, or completely wrong results.

## Confidence Levels
- **High**: You are very confident in your assessment (clear match or clear mismatch)
- **Medium**: The answer has some ambiguity but you can make a reasonable assessment
- **Low**: The answer is unclear, incomplete, or requires domain expertise beyond the provided context

## Response Format (REQUIRED)
You MUST respond with valid JSON wrapped in <json>...</json> tags. Do not include any text outside the JSON tags.

<json>
{{
    "reasoning": "Your detailed step-by-step analysis and reasoning...",
    "response": "Your final grade/assessment (e.g., 'Correct', 'Partial', 'Incorrect', or a numeric score)",
    "confidence": "High/Medium/Low"
}}
</json>

IMPORTANT: Ensure your JSON is valid and properly formatted. The 'response' field should contain only the grade, not the reasoning. The 'confidence' field helps track grading reliability."""

        if is_retry:
            return f"""ERROR: Your previous response did not contain valid JSON with a 'response' field, or the JSON was malformed.

You MUST respond with ONLY valid JSON wrapped in <json>...</json> tags. Do not include any text before or after the JSON tags.

Correct format:
<json>
{{
    "reasoning": "Your detailed analysis here...",
    "response": "Correct",
    "confidence": "High"
}}
</json>

Now try again with the original task:

{base_prompt}"""
        
        return base_prompt

    def _extract_prediction(self, text: str) -> tuple[str, str, str, str]:
        """Extract prediction, reasoning, and confidence from response text.
        
        Returns:
            (prediction, reasoning, confidence, method) tuple where method indicates extraction success
        """
        prediction = "None"
        reasoning = ""
        confidence = "Unknown"
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
            if "confidence" in last_json:
                confidence = str(last_json["confidence"])
        
        # Normalize the prediction
        prediction = _normalize_prediction(prediction)
        
        return prediction, reasoning, confidence, method

    def forward(self, inputs: dict) -> tuple[str, list[dict], dict]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history, metadata) where metadata includes confidence and extraction info
        """
        self._total_calls += 1
        instruction = self._build_grading_prompt(inputs, is_retry=False)
        
        msg_history = []
        prediction = "None"
        reasoning = ""
        confidence = "Unknown"
        
        # Retry loop for robust extraction
        for attempt in range(self.max_retries):
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=msg_history if attempt > 0 else [],
                )
                
                last_text = msg_history[-1]["text"] if msg_history else ""
                prediction, reasoning, confidence, method = self._extract_prediction(last_text)
                
                # Update statistics
                self._extraction_stats[method] += 1
                
                if prediction != "None":
                    # Track confidence statistics
                    if confidence in self._confidence_stats:
                        self._confidence_stats[confidence] += 1
                    else:
                        self._confidence_stats["Unknown"] += 1
                    
                    self.log_fn(f"Successfully extracted prediction: {prediction} (method: {method}, confidence: {confidence})")
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
        
        metadata = {
            "confidence": confidence,
            "reasoning": reasoning,
            "extraction_method": method if 'method' in dir() else "unknown",
            "attempts": attempt + 1,
        }
        
        return str(prediction), msg_history, metadata
