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
            # Try to extract JSON from markdown code blocks within
            try:
                if "```json" in inner:
                    json_start = inner.find("```json") + 7
                    json_end = inner.find("```", json_start)
                    if json_end > json_start:
                        results.append(json.loads(inner[json_start:json_end].strip()))
            except json.JSONDecodeError:
                continue
    
    # Fallback: try markdown code blocks if no <json> blocks found
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
        # Extract structured fields for better prompting
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        # Validate required inputs
        if not problem:
            return "Error: No problem statement provided.", []
        if not student_answer:
            return "Error: No student answer provided.", []

        instruction = f"""You are an expert IMO (International Mathematical Olympiad) grader. Your task is to evaluate a student's solution to a mathematical problem.

## Problem Domain
{domain}

## Problem Statement
{problem}

## Official Solution
{solution}

## Grading Guidelines
{grading_guidelines}

## Student's Answer
{student_answer}

## Your Task
Carefully evaluate the student's answer against the official solution and grading guidelines. Consider:
1. Mathematical correctness and rigor
2. Completeness of the solution
3. Logical reasoning and proof structure
4. Whether the student addressed all parts of the problem
5. Common error patterns and misconceptions

Provide your evaluation as a JSON object with the following schema:
<json>
{{
    "response": "Your detailed grading feedback here. Include: (1) whether the solution is correct/partially correct/incorrect, (2) specific points where the student succeeded or failed, (3) a numerical score if applicable based on the grading guidelines, (4) constructive feedback for improvement."
}}
</json>

Important: Ensure your response is valid JSON within the <json> tags."""

        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
        except Exception as e:
            self.log_fn(f"LLM call failed: {e}")
            return f"Error: LLM call failed - {str(e)}", []

        # Extract prediction from JSON with multiple fallback strategies
        prediction = "None"
        try:
            if msg_history and len(msg_history) > 0:
                last_message = msg_history[-1].get("text", "")
                extracted = _extract_jsons(last_message)
                if extracted:
                    last_extracted = extracted[-1]
                    if isinstance(last_extracted, dict) and "response" in last_extracted:
                        prediction = last_extracted["response"]
                    elif isinstance(last_extracted, dict):
                        # Try to find any meaningful content
                        prediction = str(last_extracted)
                else:
                    # Fallback: use raw response if JSON extraction fails
                    prediction = last_message[:2000] if last_message else "None"
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            # Use raw response as fallback
            try:
                if msg_history and len(msg_history) > 0:
                    prediction = msg_history[-1].get("text", "None")[:2000]
            except:
                prediction = "Error: Failed to extract response"

        return str(prediction), msg_history
