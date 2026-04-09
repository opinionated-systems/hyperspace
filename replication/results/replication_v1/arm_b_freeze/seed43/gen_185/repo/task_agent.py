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
from agent.monitoring import get_metrics_collector

logger = logging.getLogger(__name__)


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    Includes robust error recovery and nested structure handling.
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
        
        # Try multiple parsing strategies
        parsed = _try_parse_json_with_recovery(inner)
        if parsed is not None:
            results.append(parsed)
    
    return results or None


def _try_parse_json_with_recovery(text: str) -> dict | None:
    """Attempt to parse JSON with multiple recovery strategies.
    
    Tries increasingly aggressive fixes to handle malformed JSON
    from LLM outputs.
    """
    # Strategy 1: Direct parsing
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Strategy 2: Fix trailing commas
    try:
        fixed = re.sub(r',(\s*[}\]])', r'\1', text)
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    
    # Strategy 3: Fix single quotes (carefully, only outside strings)
    try:
        fixed = _fix_single_quotes(text)
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    
    # Strategy 4: Extract just the first complete JSON object
    try:
        fixed = _extract_first_json_object(text)
        return json.loads(fixed)
    except (json.JSONDecodeError, ValueError):
        pass
    
    return None


def _fix_single_quotes(text: str) -> str:
    """Fix single quotes to double quotes, being careful about apostrophes."""
    result = []
    in_string = False
    string_char = None
    
    for i, char in enumerate(text):
        if not in_string:
            if char in ('"', "'"):
                in_string = True
                string_char = char
                result.append('"')
            else:
                result.append(char)
        else:
            if char == string_char:
                # Check if this is an escaped quote
                if i > 0 and text[i-1] == '\\':
                    result.append(char)
                else:
                    in_string = False
                    string_char = None
                    result.append('"')
            elif char == '"' and string_char == "'":
                # Escape double quotes inside single-quoted string
                result.append('\\"')
            else:
                result.append(char)
    
    return ''.join(result)


def _extract_first_json_object(text: str) -> str:
    """Extract the first complete JSON object from text using brace counting."""
    brace_count = 0
    start_idx = -1
    
    for i, char in enumerate(text):
        if char == '{':
            if brace_count == 0:
                start_idx = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start_idx != -1:
                return text[start_idx:i+1]
    
    raise ValueError("No complete JSON object found")


def _normalize_prediction(prediction: str) -> str:
    """Normalize prediction string for consistent comparison.
    
    Strips whitespace, converts to lowercase, and handles common
    variations of grade labels.
    """
    if not prediction:
        return "none"
    
    normalized = prediction.strip().lower()
    
    # Map common variations to standard labels
    grade_map = {
        "correct": "correct",
        "right": "correct",
        "true": "correct",
        "yes": "correct",
        "incorrect": "incorrect",
        "wrong": "incorrect",
        "false": "incorrect",
        "no": "incorrect",
        "partial": "partial",
        "partially correct": "partial",
        "incomplete": "partial",
    }
    
    return grade_map.get(normalized, normalized)


def _extract_any_json(text: str) -> list[dict] | None:
    """Fallback JSON extraction that looks for any JSON objects in text.
    
    Uses a robust brace-counting approach to find complete JSON objects,
    handling nested braces correctly. Leverages the same recovery strategies
    as _extract_jsons for consistency.
    """
    results = []
    brace_count = 0
    start_idx = -1
    
    for i, char in enumerate(text):
        if char == '{':
            if brace_count == 0:
                start_idx = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start_idx != -1:
                json_str = text[start_idx:i+1]
                # Use the same recovery logic as primary extraction
                parsed = _try_parse_json_with_recovery(json_str)
                if parsed is not None:
                    results.append(parsed)
                start_idx = -1
    
    return results or None


def _extract_from_markdown_code_blocks(text: str) -> list[dict] | None:
    """Extract JSON from markdown code blocks (```json ... ```).
    
    Handles cases where the model outputs JSON in markdown format.
    Uses the same recovery strategies as other extraction methods.
    """
    results = []
    # Find markdown code blocks with various language tags
    pattern = r'```(?:json|python|text)?\s*(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    
    for match in matches:
        match = match.strip()
        if not match.startswith('{'):
            continue
        # Use the same recovery logic as primary extraction
        parsed = _try_parse_json_with_recovery(match)
        if parsed is not None:
            results.append(parsed)
        else:
            # Try nested extraction if direct parsing fails
            nested = _extract_any_json(match)
            if nested:
                results.extend(nested)
    
    return results or None


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    # Priority order for field extraction (most reliable first)
    _PREDICTION_FIELDS = [
        "grade",      # Most semantically correct for grading tasks
        "response",   # Original field name
        "evaluation",
        "result",
        "answer",
        "score",
    ]

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self._extraction_stats = {
            "total_calls": 0,
            "successful_extractions": 0,
            "fallback_used": 0,
            "recovery_used": 0,
        }

    def _update_extraction_stats(self, method_used: str, success: bool, recovery_used: bool = False) -> None:
        """Track extraction method usage and success rates."""
        self._extraction_stats["total_calls"] += 1
        if success:
            self._extraction_stats["successful_extractions"] += 1
        if method_used != "json_tags":
            self._extraction_stats["fallback_used"] += 1
        if recovery_used:
            self._extraction_stats["recovery_used"] += 1
        
        # Also record to global metrics collector
        metrics = get_metrics_collector()
        metrics.record_extraction(
            method=method_used,
            success=success,
            recovery_used=recovery_used,
        )

    def get_extraction_stats(self) -> dict:
        """Return extraction statistics for monitoring."""
        stats = dict(self._extraction_stats)
        if stats["total_calls"] > 0:
            stats["success_rate"] = stats["successful_extractions"] / stats["total_calls"]
            stats["fallback_rate"] = stats["fallback_used"] / stats["total_calls"]
            stats["recovery_rate"] = stats["recovery_used"] / stats["total_calls"]
        return stats

    def _extract_prediction_from_json(self, json_obj: dict) -> str | None:
        """Extract prediction from a JSON object using priority field order.
        
        Returns the first valid string value found in priority order.
        """
        # Try priority fields first
        for field in self._PREDICTION_FIELDS:
            if field in json_obj:
                value = json_obj[field]
                if isinstance(value, str) and value.strip():
                    return value.strip()
                elif isinstance(value, (int, float)):
                    return str(value)
        
        # Fallback: find first short string value
        for key, value in json_obj.items():
            if isinstance(value, str) and len(value) < 100 and value.strip():
                return value.strip()
        
        return None

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Extract task components for better prompting
        domain = inputs.get("domain", "Unknown")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = f"""You are an expert mathematical grader for International Mathematical Olympiad (IMO) problems.

Your task is to evaluate a student's answer to a mathematics problem and provide a grade with detailed reasoning.

DOMAIN: {domain}

PROBLEM:
{problem}

OFFICIAL SOLUTION:
{solution}

GRADING GUIDELINES:
{grading_guidelines}

STUDENT'S ANSWER:
{student_answer}

Think step by step and provide a thorough analysis:
1. PROBLEM ANALYSIS: Identify the key mathematical concepts, theorems, and techniques required
2. SOLUTION REVIEW: Analyze the official solution's approach, key steps, and expected answer format
3. STUDENT WORK ANALYSIS: 
   - Identify what approach the student took
   - Note any correct steps or valid insights
   - Identify errors, gaps, or misconceptions
4. GRADING CRITERIA CHECK:
   - Verify if the student met each criterion in the grading guidelines
   - Note partial credit for incomplete but valid reasoning
5. FINAL DETERMINATION: Assign grade based on completeness, correctness, and adherence to guidelines

Respond ONLY in JSON format wrapped in <json> tags with the following exact schema:
<json>
{{
    "reasoning": "Your detailed chain-of-thought analysis covering all 5 steps above",
    "response": "The final grade/prediction (e.g., 'Correct', 'Incorrect', 'Partial', or a numeric score)"
}}
</json>

IMPORTANT FORMATTING RULES:
- Use double quotes for all JSON strings, never single quotes
- Do not include trailing commas
- Ensure the JSON is valid and parseable
- The response field must contain a concise grade label

Important: Be objective and consistent. Award partial credit when the student shows valid reasoning even if the final answer is incorrect.

SELF-CORRECTION CHECK: Before finalizing your grade, verify that:
- You haven't missed any subtle errors in the student's work
- You haven't overlooked partial credit opportunities
- Your grade aligns with the official solution's requirements
- Your reasoning is consistent with the grading guidelines
- Your JSON output follows the exact format specified above"""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON with fallback mechanisms
        prediction = "None"
        extraction_method = "none"
        extraction_success = False
        recovery_used = False
        
        try:
            last_message = msg_history[-1]["text"]
            
            # Try primary extraction method (from <json> tags)
            extracted = _extract_jsons(last_message)
            if extracted:
                extraction_method = "json_tags"
                # Check if recovery was needed
                for item in extracted:
                    # Recovery is used when the JSON needed fixing
                    pass  # Recovery is tracked in the parsing functions
            
            # Fallback 1: generic JSON extraction from braces
            if extracted is None:
                extracted = _extract_any_json(last_message)
                if extracted:
                    extraction_method = "any_json"
                    recovery_used = True
            
            # Fallback 2: markdown code blocks
            if extracted is None:
                extracted = _extract_from_markdown_code_blocks(last_message)
                if extracted:
                    extraction_method = "markdown"
                    recovery_used = True
            
            if extracted:
                extraction_success = True
                last_json = extracted[-1]
                
                # Use the new extraction method with priority fields
                extracted_prediction = self._extract_prediction_from_json(last_json)
                if extracted_prediction:
                    prediction = extracted_prediction
                
                # Normalize the prediction for consistency
                prediction = _normalize_prediction(prediction)
                self.log_fn(f"Extracted prediction via {extraction_method}: {prediction}")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
        
        # Track extraction statistics
        self._update_extraction_stats(extraction_method, extraction_success, recovery_used)

        return str(prediction), msg_history
