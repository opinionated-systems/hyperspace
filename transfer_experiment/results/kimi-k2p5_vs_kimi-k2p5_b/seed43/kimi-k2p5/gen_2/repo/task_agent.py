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
    
    # Strategy 2: Try ```json...``` code blocks
    json_code_pattern = r'```json\s*(.*?)\s*```'
    matches = re.findall(json_code_pattern, text, re.DOTALL | re.IGNORECASE)
    for match in matches:
        try:
            data = json.loads(match.strip())
            if isinstance(data, dict) and "response" in data:
                return str(data["response"]).lower().strip()
        except json.JSONDecodeError:
            continue
    
    # Strategy 3: Try raw JSON objects
    json_object_pattern = r'\{[^}]*"response"[^}]*\}'
    matches = re.findall(json_object_pattern, text, re.DOTALL | re.IGNORECASE)
    for match in matches:
        try:
            data = json.loads(match)
            if isinstance(data, dict) and "response" in data:
                return str(data["response"]).lower().strip()
        except json.JSONDecodeError:
            continue
    
    # Strategy 4: Direct text matching (look for the keywords)
    # Check for exact matches first
    for keyword in ["correct", "incorrect", "partial"]:
        # Look for the keyword as a standalone word
        pattern = r'\b' + keyword + r'\b'
        if re.search(pattern, text_lower):
            return keyword
    
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
        # Extract fields from inputs with defaults
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = f"""You are an expert mathematical grader evaluating student solutions to competition mathematics problems.

Your task is to evaluate the student's answer and determine if it is:
- "correct": The student has provided a complete and correct solution.
- "incorrect": The student has provided an incorrect solution or the answer is wrong.
- "partial": The student has made partial progress but the solution is incomplete or has significant gaps.

## Problem Statement:
{problem}

## Official Solution:
{solution}

## Grading Guidelines:
{grading_guidelines}

## Student's Answer:
{student_answer}

## Your Task:
Carefully analyze the student's answer against the official solution and grading guidelines. Determine whether the student's answer should be classified as "correct", "incorrect", or "partial".

Respond in JSON format with the following schema:
<json>
{{
    "response": "correct" | "incorrect" | "partial"
}}
</json>

Provide only the JSON response with your classification."""

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
                # Normalize the prediction to expected values
                pred_str = extracted.lower().strip()
                if pred_str in ["correct", "incorrect", "partial"]:
                    prediction = pred_str
                elif "correct" in pred_str and "partial" not in pred_str and "incorrect" not in pred_str:
                    prediction = "correct"
                elif "partial" in pred_str:
                    prediction = "partial"
                elif "incorrect" in pred_str or "wrong" in pred_str:
                    prediction = "incorrect"
                else:
                    prediction = "incorrect"  # Default fallback
            else:
                self.log_fn(f"No valid response found in: {raw_text[:200]}...")
                prediction = "incorrect"
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            prediction = "incorrect"

        return str(prediction), msg_history
