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
    Also handles markdown code blocks as fallback.
    """
    results = []
    search_from = 0
    
    # First try <json>...</json> blocks
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
    
    # Fallback: try markdown code blocks with json
    if not results:
        search_from = 0
        while True:
            start = text.find("```json", search_from)
            if start == -1:
                break
            end = text.find("```", start + 7)
            if end == -1:
                break
            inner = text[start + 7:end].strip()
            search_from = end + 3
            try:
                results.append(json.loads(inner))
            except json.JSONDecodeError:
                continue
    
    # Fallback: try to find any JSON object in the text
    if not results:
        try:
            # Look for content between first { and last }
            json_start = text.find("{")
            json_end = text.rfind("}")
            if json_start != -1 and json_end != -1 and json_end > json_start:
                potential_json = text[json_start:json_end + 1]
                results.append(json.loads(potential_json))
        except json.JSONDecodeError:
            pass
    
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
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = f"""You are an expert {domain} grader evaluating student answers.

Your task is to carefully analyze the student's answer against the correct solution and grading guidelines, then provide your evaluation.

## Problem:
{problem}

## Correct Solution:
{solution}

## Grading Guidelines:
{grading_guidelines}

## Student's Answer:
{student_answer}

## Instructions:
1. First, think step-by-step about the student's answer. Compare it to the correct solution.
2. Identify any errors, omissions, or correct steps in the student's work.
3. Consider the grading guidelines when determining the final assessment.
4. Provide your reasoning in the "thinking" field.
5. Provide your final evaluation/grade in the "response" field.

## Important:
- Be thorough in your analysis - check each step of the student's work.
- Consider partial credit if the student made progress but didn't fully solve.
- If the answer is completely correct, state that clearly.
- If there are errors, explain what went wrong and what the correct approach would be.
- Your response should be a clear, concise evaluation or grade.

Respond in JSON format with the following schema:
<json>
{{
    "thinking": "Your detailed step-by-step analysis here...",
    "response": "Your final evaluation/grade here..."
}}
</json>"""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON
        prediction = "None"
        try:
            extracted = _extract_jsons(msg_history[-1]["text"])
            if extracted:
                # Prefer "response" field, fallback to other common fields
                last_json = extracted[-1]
                if "response" in last_json:
                    prediction = last_json["response"]
                elif "grade" in last_json:
                    prediction = last_json["grade"]
                elif "evaluation" in last_json:
                    prediction = last_json["evaluation"]
                elif "answer" in last_json:
                    prediction = last_json["answer"]
                elif "result" in last_json:
                    prediction = last_json["result"]
                elif "output" in last_json:
                    prediction = last_json["output"]
                else:
                    # If no recognized field, use the whole JSON as string
                    prediction = json.dumps(last_json)
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        # Ensure prediction is a string and not empty
        if prediction is None or (isinstance(prediction, str) and prediction.strip() == ""):
            prediction = "None"
        
        return str(prediction), msg_history
