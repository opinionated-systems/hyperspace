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
    """Extract JSON objects from <json>...</json> blocks or standalone JSON.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    Also handles standalone JSON objects and arrays without tags.
    """
    results = []
    search_from = 0
    
    # First, try to extract from <json> tags
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
            # Try to find JSON object boundaries if the content is malformed
            try:
                # Look for outermost braces
                brace_start = inner.find('{')
                brace_end = inner.rfind('}')
                if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
                    results.append(json.loads(inner[brace_start:brace_end + 1]))
            except json.JSONDecodeError:
                continue
    
    # If no tagged JSON found, try to extract standalone JSON objects
    if not results:
        # Look for JSON objects (starting with { and ending with })
        brace_count = 0
        start_idx = -1
        for i, char in enumerate(text):
            if char == '{':
                if brace_count == 0:
                    start_idx = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start_idx != -1:
                    try:
                        json_str = text[start_idx:i + 1]
                        results.append(json.loads(json_str))
                    except json.JSONDecodeError:
                        pass
                    start_idx = -1
    
    return results or None


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def _build_grading_prompt(self, inputs: dict) -> str:
        """Build a structured grading prompt with chain-of-thought reasoning."""
        domain = inputs.get("domain", "")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        return f"""You are an expert mathematical grader for IMO (International Mathematical Olympiad) problems.

Your task is to evaluate a student's answer by carefully analyzing:
1. The problem statement and domain
2. The official solution
3. The grading guidelines
4. The student's submitted answer

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

## Instructions
Think step by step:
1. First, understand what the problem is asking
2. Review the official solution to know the correct approach
3. Examine the grading guidelines for scoring criteria
4. Analyze the student's answer for correctness, completeness, and clarity
5. Compare the student's approach with the official solution
6. Determine if the answer is correct, partially correct, or incorrect

Respond in JSON format with the following schema:
<json>
{{
    "reasoning": "Your detailed chain-of-thought analysis here",
    "response": "Your final grading decision (e.g., 'Correct', 'Partially Correct', 'Incorrect', or a numeric score)"
}}
</json>"""

    def _extract_prediction(self, msg_history: list[dict]) -> str:
        """Extract prediction from message history with validation."""
        if not msg_history:
            return "None"
        
        last_message = msg_history[-1].get("text", "")
        
        # Try to extract JSON
        extracted = _extract_jsons(last_message)
        if not extracted:
            return "None"
        
        # Get the last JSON object
        result = extracted[-1]
        
        # Validate and return response
        if isinstance(result, dict) and "response" in result:
            response = result["response"]
            # Ensure response is a string
            if not isinstance(response, str):
                response = str(response)
            return response.strip()
        
        return "None"

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with improved reasoning.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        instruction = self._build_grading_prompt(inputs)

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction with improved error handling
        prediction = self._extract_prediction(msg_history)
        
        if prediction == "None":
            self.log_fn("Warning: Failed to extract valid prediction from response")
            # Log the raw response for debugging
            if msg_history:
                self.log_fn(f"Raw response: {msg_history[-1].get('text', '')[:500]}")

        return prediction, msg_history
