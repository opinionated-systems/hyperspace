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


def _extract_json_flexible(text: str) -> list[dict] | None:
    """Extract JSON objects using multiple strategies.
    
    Tries multiple extraction methods in order of preference:
    1. <json>...</json> tags
    2. ```json...``` code blocks
    3. Bare JSON objects (starting with { and ending with })
    """
    # Try <json> tags first
    results = _extract_jsons(text)
    if results:
        return results
    
    # Try ```json code blocks
    results = []
    pattern = r'```json\s*(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    for match in matches:
        try:
            results.append(json.loads(match.strip()))
        except json.JSONDecodeError:
            continue
    if results:
        return results
    
    # Try bare JSON objects (simple heuristic)
    # Look for outermost JSON objects
    results = []
    depth = 0
    start = -1
    for i, char in enumerate(text):
        if char == '{':
            if depth == 0:
                start = i
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0 and start != -1:
                try:
                    json_str = text[start:i+1]
                    results.append(json.loads(json_str))
                except json.JSONDecodeError:
                    pass
                start = -1
    
    return results or None


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
        # Build a more structured prompt with clear instructions
        domain = inputs.get('domain', 'Mathematics')
        problem = inputs.get('problem', '')
        solution = inputs.get('solution', '')
        grading_guidelines = inputs.get('grading_guidelines', '')
        student_answer = inputs.get('student_answer', '')
        
        instruction = f"""You are an expert grader for {domain} problems. Your task is to evaluate a student's answer against the correct solution and grading guidelines.

## Problem:
{problem}

## Correct Solution:
{solution}

## Grading Guidelines:
{grading_guidelines}

## Student's Answer:
{student_answer}

## Your Task:
Evaluate the student's answer and provide your assessment. The response should be one of:
- "correct" - if the answer is fully correct (7 points)
- "almost" - if the answer is nearly complete with only minor mistakes (6 points)
- "partial" - if the answer has some correct elements but significant gaps remain (1 point)
- "incorrect" - if the answer is wrong or significantly flawed (0 points)

Use these guidelines:
- "correct": The solution is complete and correct with at most negligible errors
- "almost": The solution is nearly complete but has minor mistakes that are not negligible
- "partial": The solution shows meaningful progress but has significant gaps or errors
- "incorrect": The solution is essentially wrong or shows no meaningful progress

Respond ONLY in the following JSON format:
<json>
{{
    "response": "correct" | "almost" | "partial" | "incorrect"
}}</json>"""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON using flexible extraction
        prediction = "None"
        try:
            extracted = _extract_json_flexible(msg_history[-1]["text"])
            if extracted:
                last_json = extracted[-1]
                if isinstance(last_json, dict) and "response" in last_json:
                    prediction = last_json["response"]
                    # Normalize the response
                    if isinstance(prediction, str):
                        prediction_lower = prediction.lower().strip()
                        if prediction_lower in ["correct", "almost", "partial", "incorrect"]:
                            prediction = prediction_lower
                        elif "almost" in prediction_lower:
                            prediction = "almost"
                        elif "partial" in prediction_lower:
                            prediction = "partial"
                        elif "correct" in prediction_lower and "incorrect" not in prediction_lower:
                            prediction = "correct"
                        elif "incorrect" in prediction_lower or "wrong" in prediction_lower:
                            prediction = "incorrect"
                self.log_fn(f"Extracted prediction: {prediction}")
            else:
                self.log_fn(f"No JSON extracted from response: {msg_history[-1]['text'][:200]}...")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
