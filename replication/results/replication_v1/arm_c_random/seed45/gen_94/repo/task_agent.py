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


def _format_grading_prompt(inputs: dict) -> str:
    """Format a structured grading prompt for IMO problems.
    
    This helps the model understand the grading task better by explicitly
    structuring the different components of the problem.
    """
    domain = inputs.get("domain", "Mathematics")
    problem = inputs.get("problem", "")
    solution = inputs.get("solution", "")
    guidelines = inputs.get("grading_guidelines", "")
    student_answer = inputs.get("student_answer", "")
    
    prompt = f"""You are an expert IMO (International Mathematical Olympiad) grader. Your task is to evaluate a student's solution to a mathematics problem.

## Problem Domain
{domain}

## Problem Statement
{problem}

## Official Solution
{solution}

## Grading Guidelines
{guidelines}

## Student's Answer
{student_answer}

## Your Task
Carefully analyze the student's answer against the official solution and grading guidelines. Provide your evaluation in the following JSON format:

<json>
{{
    "response": "Your grade/evaluation here. Be specific about what the student did correctly or incorrectly."
}}
</json>

Important: 
- Compare the student's reasoning step-by-step with the official solution
- Note any missing steps, errors, or alternative valid approaches
- Be precise and objective in your grading"""
    
    return prompt


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
        # Use structured formatting for grading tasks
        instruction = _format_grading_prompt(inputs)

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON with retry logic
        prediction = "None"
        extraction_attempts = 0
        max_attempts = 2
        
        while extraction_attempts < max_attempts:
            try:
                # Try to extract from the last assistant message
                last_msg = msg_history[-1]["text"] if msg_history else ""
                extracted = _extract_jsons(last_msg)
                
                if extracted and "response" in extracted[-1]:
                    prediction = extracted[-1]["response"]
                    break
                else:
                    # Try to find any JSON-like structure as fallback
                    json_match = re.search(r'\{[^}]*"response"[^}]*\}', last_msg, re.DOTALL)
                    if json_match:
                        try:
                            data = json.loads(json_match.group())
                            prediction = data.get("response", "None")
                            break
                        except json.JSONDecodeError:
                            pass
                    
                    extraction_attempts += 1
                    if extraction_attempts >= max_attempts:
                        # Use raw response if JSON extraction fails
                        prediction = last_msg[:500] if last_msg else "None"
                        self.log_fn(f"JSON extraction failed, using truncated raw response")
                        
            except Exception as e:
                extraction_attempts += 1
                self.log_fn(f"Error extracting prediction (attempt {extraction_attempts}): {e}")
                if extraction_attempts >= max_attempts:
                    prediction = "None"

        return str(prediction), msg_history
