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

## Classification Criteria - READ CAREFULLY:

**CORRECT**: The student has provided a complete, correct solution with ALL of the following:
- Correct final answer that matches the official solution
- Valid mathematical reasoning throughout
- All key steps properly justified with rigorous proofs
- No significant gaps, errors, or logical flaws
- The solution would receive full marks in a competition

**INCORRECT**: The student's answer has ONE OR MORE of the following:
- Wrong final answer (different from official solution)
- Critical logical or mathematical errors that invalidate the approach
- Fundamental misunderstanding of the problem statement
- No meaningful progress toward the solution (just restating the problem or trivial observations)
- Major gaps that make the solution incomplete and unfixable
- The solution would receive 0 or minimal marks

**PARTIAL**: The student made MEANINGFUL PROGRESS but the solution is incomplete. This requires:
- Correct approach or strategy identified
- Valid insights or lemmas proven
- Significant progress toward the solution (not just trivial observations)
- Missing some key steps OR has minor errors that could be fixed
- The solution would receive partial credit (not 0, not full marks)

## CRITICAL DISTINCTION RULES:

1. If the student has the CORRECT FINAL ANSWER with mostly correct reasoning → CORRECT
2. If the student has the WRONG FINAL ANSWER with no meaningful progress → INCORRECT
3. If the student has WRONG FINAL ANSWER but made significant progress (correct approach, valid lemmas, good structure) → PARTIAL
4. If the student has CORRECT FINAL ANSWER but with critical gaps in reasoning → PARTIAL (not correct)
5. If the student only restates the problem or makes trivial observations → INCORRECT (not partial)

## Your Task:
1. Identify the correct final answer from the official solution.
2. Check if the student's final answer matches.
3. Analyze the reasoning quality and completeness.
4. Apply the CRITICAL DISTINCTION RULES above.
5. Classify as CORRECT, INCORRECT, or PARTIAL based on the criteria.

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
