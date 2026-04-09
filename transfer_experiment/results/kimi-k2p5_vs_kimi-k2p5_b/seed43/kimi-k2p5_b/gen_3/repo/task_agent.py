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
            # Try with more permissive matching for nested braces
            continue
    
    # Also try to find JSON blocks with nested content
    if not results:
        # Look for JSON code blocks
        code_block_pattern = r'```json\s*(.*?)```'
        code_matches = re.findall(code_block_pattern, text, re.DOTALL)
        for match in code_matches:
            try:
                results.append(json.loads(match.strip()))
            except json.JSONDecodeError:
                continue
    
    return results or None


def _normalize_prediction(prediction: str) -> str:
    """Normalize prediction to one of the three valid labels."""
    pred_str = str(prediction).lower().strip()
    
    # Direct match
    if pred_str in ["correct", "incorrect", "partial"]:
        return pred_str
    
    # Check for partial first (more specific)
    if "partial" in pred_str:
        return "partial"
    
    # Check for incorrect/wrong
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
        # Extract key fields from inputs for better prompting
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

## Classification Guidelines:

**"correct"** - The student provided a complete, correct solution with proper reasoning. The answer:
- Arrives at the correct final answer
- Contains valid mathematical reasoning throughout
- May have minor notation issues or small gaps that don't affect correctness

**"incorrect"** - The student's answer is wrong or fundamentally flawed. The answer:
- Has an incorrect final answer with no valid path to solution
- Contains critical logical or mathematical errors
- Shows no meaningful progress toward the solution
- Is incomplete with no useful partial results

**"partial"** - The student made meaningful progress but the solution is incomplete. The answer:
- Contains valid and significant progress toward the solution
- Has correct lemmas, propositions, or intermediate steps
- May have the right approach but incomplete execution
- Shows understanding of key concepts even if final answer is wrong or missing
- Contains "Partial" or "Almost" markers in grading guidelines

## Your Task:
First, analyze the student's answer step by step:
1. What correct progress did the student make?
2. What are the gaps or errors in their reasoning?
3. Does the grading guideline mention "Partial" or "Almost"?

Then classify into EXACTLY ONE category: "correct", "incorrect", or "partial".

You MUST respond with a JSON object in this exact format:
<json>
{{
    "response": "correct"
}}
</json>

OR

<json>
{{
    "response": "incorrect"
}}
</json>

OR

<json>
{{
    "response": "partial"
}}
</json>

Replace the value with your classification. Do not include any other text outside the JSON block."""

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
            
            # Second try: regex extraction
            if not extracted:
                extracted = _extract_json_regex(response_text)
            
            # Third try: direct JSON parsing of the whole text
            if not extracted:
                try:
                    # Try to parse the entire response as JSON
                    parsed = json.loads(response_text.strip())
                    if isinstance(parsed, dict) and "response" in parsed:
                        extracted = [parsed]
                except json.JSONDecodeError:
                    pass
            
            if extracted and isinstance(extracted, list) and len(extracted) > 0:
                last_json = extracted[-1]
                if isinstance(last_json, dict) and "response" in last_json:
                    prediction = _normalize_prediction(last_json["response"])
                else:
                    self.log_fn(f"JSON found but no 'response' field: {last_json}")
            else:
                self.log_fn(f"No valid JSON found in response: {response_text[:200]}...")
                
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
