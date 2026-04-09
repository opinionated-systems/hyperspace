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


def _extract_points_from_text(text: str) -> str | None:
    """Extract a points value (0, 1, 6, or 7) from text.
    
    Looks for explicit point mentions or numeric scores.
    """
    if not text:
        return None
    
    # Look for explicit point values
    # Pattern: "X points" or "score: X" or "Points: X"
    patterns = [
        r'(?:points?|score)[\s:=]+(\d+)',
        r'^(\d+)\s*$',
        r'\b(\d+)\s*points?\b',
        r'grade[d\s]*[:=]+\s*(\d+)',
        r'(?:earned|awarded|given)[\s:]+(\d+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            val = match.group(1)
            if val in ['0', '1', '6', '7']:
                return val
    
    # Check for keywords indicating score
    text_lower = text.lower()
    if any(word in text_lower for word in ['full credit', 'complete', 'correct', '7/7', 'perfect']):
        return '7'
    if any(word in text_lower for word in ['partial', 'partially correct', 'some progress', '1/7']):
        return '1'
    if any(word in text_lower for word in ['incorrect', 'wrong', 'no credit', '0/7', 'zero', 'none']):
        return '0'
    if '6' in text and ('nearly' in text_lower or 'almost' in text_lower or 'minor' in text_lower):
        return '6'
    
    return None


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
        # Extract fields from inputs for better prompt formatting
        domain = inputs.get("domain", "")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        instruction = f"""You are an expert mathematics grader for IMO (International Mathematical Olympiad) problems. Your task is to evaluate a student's answer and assign a point score.

## Problem:
{problem}

## Correct Solution:
{solution}

## Grading Guidelines:
{grading_guidelines}

## Student's Answer:
{student_answer}

## Your Task:
Evaluate the student's answer based on the grading guidelines. The possible scores are:
- 7 points: Complete, correct solution
- 6 points: Minor flaw in an otherwise correct solution  
- 1 point: Significant progress but not a complete solution
- 0 points: No significant progress or incorrect solution

Analyze the student's work carefully and assign the appropriate score.

Respond ONLY in JSON format with the following schema:
<json>
{{
    "response": "X"
}}
</json>
Where X is exactly one of: 0, 1, 6, or 7"""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON
        prediction = None
        try:
            # Try to extract from JSON blocks
            extracted = _extract_jsons(msg_history[-1]["text"])
            if extracted and "response" in extracted[-1]:
                prediction = str(extracted[-1]["response"]).strip()
        except Exception as e:
            self.log_fn(f"Error extracting JSON prediction: {e}")
        
        # If JSON extraction failed, try to extract from raw text
        if prediction is None or prediction == "None":
            try:
                raw_text = msg_history[-1].get("text", "")
                prediction = _extract_points_from_text(raw_text)
            except Exception as e:
                self.log_fn(f"Error extracting text prediction: {e}")
        
        # Validate prediction is one of the allowed values
        if prediction not in ['0', '1', '6', '7']:
            self.log_fn(f"Invalid prediction '{prediction}', defaulting to 0")
            prediction = '0'

        return str(prediction), msg_history
