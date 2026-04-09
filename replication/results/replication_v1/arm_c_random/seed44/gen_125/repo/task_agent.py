"""
Task agent: solves a given task with chain-of-thought reasoning.

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


def _extract_json_from_text(text: str) -> list[dict] | None:
    """Fallback: Extract JSON objects from raw text without <json> tags."""
    results = []
    # Try to find JSON objects in code blocks
    code_block_pattern = re.compile(r'```(?:json)?\s*(.*?)```', re.DOTALL)
    for match in code_block_pattern.finditer(text):
        try:
            results.append(json.loads(match.group(1).strip()))
        except json.JSONDecodeError:
            continue
    
    # Try to find JSON objects between curly braces
    if not results:
        # Look for patterns like {"key": "value"}
        brace_pattern = re.compile(r'\{[^{}]*"[^"]+"[^{}]*\}', re.DOTALL)
        for match in brace_pattern.finditer(text):
            try:
                results.append(json.loads(match.group(0)))
            except json.JSONDecodeError:
                continue
    
    return results or None


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with chain-of-thought reasoning.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Extract fields for better prompting
        domain = inputs.get("domain", "")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = f"""You are an expert grader for {domain} problems. Your task is to evaluate a student's answer against the correct solution and grading guidelines.

## Problem Statement
{problem}

## Correct Solution
{solution}

## Grading Guidelines
{grading_guidelines}

## Student's Answer
{student_answer}

## Grading Instructions

You must classify the student's answer into ONE of these three categories:

1. **"correct"** - The student's answer is fully correct, matches the solution, and follows all guidelines.
2. **"incorrect"** - The student's answer is wrong, contains significant errors, or fails to meet the requirements.
3. **"partial"** - The student's answer is partially correct but incomplete, has minor errors, or only partially meets the requirements.

## Analysis Steps

1. **Understand the Problem**: What is being asked? What is the expected answer format?
2. **Review the Correct Solution**: What should the correct answer contain?
3. **Analyze the Student's Answer**: Compare each part of the student's answer to the correct solution.
4. **Check Against Guidelines**: Does the answer meet all the grading criteria?
5. **Determine the Grade**: Based on your analysis, assign one of: "correct", "incorrect", or "partial".

## Examples of Good Reasoning

- If the student got the right answer but missed a step: "partial"
- If the student got everything right: "correct"
- If the student made a fundamental error: "incorrect"

## Response Format

You MUST respond in the following JSON format:

<json>
{{
    "reasoning": "Your detailed step-by-step analysis comparing the student's answer to the correct solution...",
    "response": "correct" | "incorrect" | "partial"
}}
</json>

IMPORTANT: The "response" field MUST be exactly one of: "correct", "incorrect", or "partial" (all lowercase)."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON
        prediction = "None"
        try:
            # First try <json> tags
            extracted = _extract_jsons(msg_history[-1]["text"])
            
            # Fallback to raw text extraction if no <json> tags found
            if not extracted:
                extracted = _extract_json_from_text(msg_history[-1]["text"])
            
            if extracted:
                # Prefer "response" field, but fallback to other common fields
                last_json = extracted[-1]
                if "response" in last_json:
                    prediction = last_json["response"]
                elif "grade" in last_json:
                    prediction = last_json["grade"]
                elif "answer" in last_json:
                    prediction = last_json["answer"]
                elif "result" in last_json:
                    prediction = last_json["result"]
                elif "evaluation" in last_json:
                    prediction = last_json["evaluation"]
                else:
                    # If no recognized field, use the whole JSON as string
                    prediction = json.dumps(last_json)
                
                # Normalize the prediction to one of the three valid labels
                prediction_lower = str(prediction).lower().strip()
                if "correct" in prediction_lower and "incorrect" not in prediction_lower and "partial" not in prediction_lower:
                    prediction = "correct"
                elif "incorrect" in prediction_lower or "wrong" in prediction_lower or "error" in prediction_lower:
                    prediction = "incorrect"
                elif "partial" in prediction_lower or "incomplete" in prediction_lower or "some" in prediction_lower:
                    prediction = "partial"
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
