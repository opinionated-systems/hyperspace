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


def _extract_json_regex(text: str) -> list[dict] | None:
    """Extract JSON objects using regex as a fallback method.
    
    Looks for JSON objects with "response" field.
    """
    results = []
    # Pattern to match JSON objects
    pattern = r'\{[^{}]*"response"[^{}]*\}'
    matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
    
    for match in matches:
        try:
            results.append(json.loads(match))
        except json.JSONDecodeError:
            continue
    
    # Also try to find JSON code blocks
    if not results:
        code_block_pattern = r'```json\s*(.*?)```'
        code_matches = re.findall(code_block_pattern, text, re.DOTALL)
        for match in code_matches:
            try:
                results.append(json.loads(match.strip()))
            except json.JSONDecodeError:
                continue
    
    return results or None


def _extract_response_from_text(text: str) -> str | None:
    """Extract the response value from various possible formats.
    
    Tries multiple extraction strategies:
    1. <json>...</json> blocks
    2. JSON code blocks ```json...```
    3. Raw JSON objects
    4. Direct text matching for "correct", "incorrect", "partial"
    """
    text_lower = text.lower()
    
    # Strategy 1: Try <json>...</json> blocks
    json_results = _extract_jsons(text)
    if json_results:
        for result in json_results:
            if isinstance(result, dict) and "response" in result:
                return str(result["response"]).lower().strip()
    
    # Strategy 2: Try regex extraction
    json_results = _extract_json_regex(text)
    if json_results:
        for result in json_results:
            if isinstance(result, dict) and "response" in result:
                return str(result["response"]).lower().strip()
    
    # Strategy 3: Try ```json...``` code blocks
    json_code_pattern = r'```json\s*(.*?)\s*```'
    matches = re.findall(json_code_pattern, text, re.DOTALL | re.IGNORECASE)
    for match in matches:
        try:
            data = json.loads(match.strip())
            if isinstance(data, dict) and "response" in data:
                return str(data["response"]).lower().strip()
        except json.JSONDecodeError:
            continue
    
    # Strategy 4: Try raw JSON objects with response field
    json_object_pattern = r'\{[^}]*"response"[^}]*\}'
    matches = re.findall(json_object_pattern, text, re.DOTALL | re.IGNORECASE)
    for match in matches:
        try:
            data = json.loads(match)
            if isinstance(data, dict) and "response" in data:
                return str(data["response"]).lower().strip()
        except json.JSONDecodeError:
            continue
    
    # Strategy 5: Direct text matching (look for the keywords)
    # Check for partial first (more specific than correct)
    if re.search(r'\bpartial\b', text_lower):
        return "partial"
    # Check for incorrect/wrong
    if re.search(r'\b(incorrect|wrong|false)\b', text_lower):
        return "incorrect"
    # Check for correct (but not incorrect)
    if re.search(r'\bcorrect\b', text_lower) and "incorrect" not in text_lower:
        return "correct"
    
    return None


def _normalize_prediction(prediction: str) -> str:
    """Normalize prediction to one of the three valid labels."""
    pred_str = str(prediction).lower().strip()
    
    # Direct match
    if pred_str in ["correct", "incorrect", "partial"]:
        return pred_str
    
    # Check for partial first (more specific)
    if "partial" in pred_str:
        return "partial"
    
    # Check for incorrect/wrong/false
    if "incorrect" in pred_str or "wrong" in pred_str or "false" in pred_str:
        return "incorrect"
    
    # Check for correct (but not incorrect)
    if "correct" in pred_str and "incorrect" not in pred_str:
        return "correct"
    
    # Default fallback
    return "incorrect"


class TaskAgent:
    """Task agent that solves IMO grading problems."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Extract fields from inputs with defaults
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = f"""You are an expert mathematical olympiad grader. Your task is to evaluate a student's answer to an IMO-level problem.

## Problem Statement:
{problem}

## Official Solution:
{solution}

## Grading Guidelines:
{grading_guidelines}

## Student's Answer:
{student_answer}

## Classification Criteria:

**CORRECT**: The student has provided a complete, correct solution with:
- Correct final answer
- Valid mathematical reasoning
- All key steps properly justified
- No significant gaps or errors

**INCORRECT**: The student's answer is wrong, including:
- Wrong final answer
- Critical logical or mathematical errors
- Fundamental misunderstanding of the problem
- No meaningful progress toward the solution

**PARTIAL**: The student made meaningful progress but the solution is incomplete or has minor gaps:
- Correct approach but incomplete execution
- Valid insights but missing key steps
- Minor errors that don't invalidate the main approach
- Significant progress toward the solution but not fully complete

## Your Task:
1. First, identify what the problem is asking and what the correct answer should be.
2. Analyze the student's answer step by step.
3. Compare against the official solution and grading guidelines.
4. Determine if the student made MEANINGFUL PROGRESS (partial) or if the answer is essentially WRONG (incorrect).

Respond with ONLY a JSON object in this exact format:
<json>
{{
    "response": "correct" | "incorrect" | "partial"
}}
</json>"""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from the response
        prediction = "incorrect"  # Default fallback
        try:
            # Get the raw text from the last message
            raw_text = msg_history[-1]["text"] if msg_history else ""
            
            # Try to extract the response
            extracted = _extract_response_from_text(raw_text)
            
            if extracted:
                prediction = _normalize_prediction(extracted)
            else:
                self.log_fn(f"No valid response found in: {raw_text[:200]}...")
                prediction = "incorrect"
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            prediction = "incorrect"

        return str(prediction), msg_history
