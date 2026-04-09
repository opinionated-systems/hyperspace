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


def _extract_response_direct(text: str) -> str | None:
    """Extract response by looking for keywords directly in text."""
    text_lower = text.lower()
    
    # Look for explicit classification statements with higher priority patterns
    patterns = [
        # Direct response field extraction
        r'"response"\s*:\s*"?(correct|incorrect|partial)"?',
        r'response[\s:]+"?(correct|incorrect|partial)"?',
        r'classification[\s:]+"?(correct|incorrect|partial)"?',
        r'classify[\s:]+"?(correct|incorrect|partial)"?',
        r'grade[\s:]+"?(correct|incorrect|partial)"?',
        # Statement patterns
        r'"?(correct|incorrect|partial)"?\s*is\s*the\s*correct\s*classification',
        r'the\s*answer\s*is\s*"?(correct|incorrect|partial)"?',
        r'should\s*be\s*classified\s*as\s*"?(correct|incorrect|partial)"?',
        r'classification\s*is\s*"?(correct|incorrect|partial)"?',
        r'classify\s*this\s*as\s*"?(correct|incorrect|partial)"?',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            return match.group(1).lower()
    
    # Count occurrences of each label (only count standalone words)
    correct_count = len(re.findall(r'\bcorrect\b', text_lower))
    incorrect_count = len(re.findall(r'\bincorrect\b', text_lower))
    partial_count = len(re.findall(r'\bpartial\b', text_lower))
    
    # If only one appears, use that
    total = correct_count + incorrect_count + partial_count
    if total > 0:
        if correct_count > 0 and incorrect_count == 0 and partial_count == 0:
            return "correct"
        if incorrect_count > 0 and correct_count == 0 and partial_count == 0:
            return "incorrect"
        if partial_count > 0 and correct_count == 0 and incorrect_count == 0:
            return "partial"
    
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
    
    # Check for incorrect/wrong/false (but not "not incorrect")
    if ("incorrect" in pred_str or "wrong" in pred_str or "false" in pred_str) and "not" not in pred_str:
        return "incorrect"
    
    # Check for correct (but not incorrect)
    if "correct" in pred_str and "incorrect" not in pred_str:
        return "correct"
    
    # Default fallback - use "incorrect" as the safest default
    return "incorrect"


class TaskAgent:
    """Task agent that solves IMO grading problems."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Extract fields from inputs for LLM-based classification
        problem = inputs.get("problem", "") or inputs.get("Problem", "")
        solution = inputs.get("solution", "") or inputs.get("Solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "") or inputs.get("Grading guidelines", "")
        student_answer = inputs.get("student_answer", "") or inputs.get("Response", "")
        
        instruction = f"""You are an expert mathematical olympiad grader evaluating IMO-level solutions. Your task is to classify a student's answer into exactly one of three categories: "correct", "incorrect", or "partial".

## Problem Statement:
{problem}

## Official Solution:
{solution}

## Grading Guidelines:
{grading_guidelines}

## Student's Answer:
{student_answer}

## Classification Rules (read carefully):

**CORRECT** - Award this ONLY if:
- The student arrives at the correct final answer
- The solution contains valid mathematical reasoning throughout
- The approach is complete and sound (minor notation issues are acceptable)
- The student demonstrates full understanding of the problem

**INCORRECT** - Award this if:
- The final answer is wrong AND there's no meaningful progress toward the solution
- The solution contains critical logical or mathematical errors that invalidate the approach
- The answer is essentially empty or shows no understanding
- The student made no useful partial progress

**PARTIAL** - Award this if the student made MEANINGFUL PROGRESS but the solution is incomplete:
- The student proved correct lemmas or intermediate results
- They identified the right approach or key insight even if execution was incomplete
- They showed understanding of key concepts even if the final answer is wrong or missing
- The grading guidelines mention "Partial" or "Almost"
- There is significant valid work toward the solution, even if incomplete

## Critical Decision Guidelines:

1. First, check if the grading guidelines explicitly mention "Partial" or "Almost" - this strongly suggests "partial" classification.

2. Look for MEANINGFUL PROGRESS: Did the student prove anything useful? Did they identify the right approach? Did they make significant headway even if the final answer is wrong? If yes → "partial".

3. Only classify as "correct" if the solution is essentially complete and correct.

4. Only classify as "incorrect" if there's essentially no valid progress toward the solution.

## Your Response:

You MUST respond with a JSON object in this exact format:
<json>
{{
    "reasoning": "Your detailed analysis here. Explain what the student got right, what they got wrong, and specifically why you chose this classification over the alternatives.",
    "response": "correct" or "incorrect" or "partial"
}}
</json>

Do not include any text outside the JSON block. Be decisive - choose exactly one classification."""

        # Get LLM response
        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON
        prediction = "incorrect"  # Default fallback
        
        try:
            response_text = msg_history[-1]["text"]
            
            # First try: extract from <json> tags
            extracted = _extract_jsons(response_text)
            
            if extracted and isinstance(extracted, list) and len(extracted) > 0:
                last_json = extracted[-1]
                if isinstance(last_json, dict) and "response" in last_json:
                    prediction = _normalize_prediction(last_json["response"])
                else:
                    self.log_fn(f"JSON found but no 'response' field: {last_json}")
            else:
                # Try direct extraction
                direct = _extract_response_direct(response_text)
                if direct:
                    prediction = direct
                else:
                    self.log_fn(f"No valid JSON found in response: {response_text[:200]}...")
                
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
        
        return str(prediction), msg_history
