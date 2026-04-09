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


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

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
        # Extract fields for better structured prompting
        domain = inputs.get("domain", "Unknown")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = f"""You are an expert mathematical grader for the International Mathematical Olympiad (IMO).

Your task is to evaluate a student's answer to a mathematical problem and assign a score based on the official solution and grading guidelines.

## Problem Domain: {domain}

## Problem Statement:
{problem}

## Official Solution:
{solution}

## Grading Guidelines:
{grading_guidelines}

## Student's Answer:
{student_answer}

## Instructions:
1. First, analyze the student's answer step by step. Compare it against the official solution.
2. Identify what the student got correct and what they got wrong or missed.
3. Consider the grading guidelines carefully - partial credit may be awarded.
4. Determine the appropriate score based on the grading rubric.

## Response Format:
Respond in JSON format with the following schema:
<json>
{{
    "reasoning": "Your detailed step-by-step analysis of the student's answer",
    "response": "The final score/grade (e.g., '7', '5', '0', 'Partial credit: 3/7')"
}}
</json>

Provide your reasoning first, then the final score in the response field."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON with better error handling
        prediction = "None"
        try:
            # Try to extract from the last assistant message
            last_msg = msg_history[-1]["text"] if msg_history else ""
            extracted = _extract_jsons(last_msg)
            
            if extracted:
                # Prefer "response" field, but fallback to other common fields
                last_json = extracted[-1]
                if "response" in last_json:
                    prediction = last_json["response"]
                elif "score" in last_json:
                    prediction = str(last_json["score"])
                elif "grade" in last_json:
                    prediction = str(last_json["grade"])
                elif "answer" in last_json:
                    prediction = str(last_json["answer"])
                else:
                    # If no recognized field, use the whole JSON as string
                    prediction = json.dumps(last_json)
            else:
                # Fallback: try to extract any numeric score from the response
                numbers = re.findall(r'\b([0-9]+(?:\.[0-9]+)?)\s*/\s*([0-9]+)\b', last_msg)
                if numbers:
                    prediction = f"{numbers[-1][0]}/{numbers[-1][1]}"
                else:
                    single_nums = re.findall(r'\b([0-7])\b', last_msg)
                    if single_nums:
                        prediction = single_nums[-1]
                        
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            # Last resort: return the raw response
            try:
                prediction = msg_history[-1]["text"] if msg_history else "None"
            except:
                prediction = "None"

        return str(prediction), msg_history
