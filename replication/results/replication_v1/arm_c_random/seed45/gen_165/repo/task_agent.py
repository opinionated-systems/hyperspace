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
    Also handles markdown code blocks with json language specifier.
    """
    results = []
    search_from = 0
    
    # First try to find <json>...</json> blocks
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
            # Try to extract just the first valid JSON object
            try:
                # Find the first { and matching }
                brace_start = inner.find('{')
                if brace_start != -1:
                    brace_count = 0
                    for i, char in enumerate(inner[brace_start:], start=brace_start):
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                results.append(json.loads(inner[brace_start:i+1]))
                                break
            except (json.JSONDecodeError, ValueError):
                continue
    
    # Also try markdown code blocks ```json ... ```
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
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with step-by-step reasoning.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Extract fields for structured prompting
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = f"""You are an expert {domain} grader evaluating student solutions.

Your task is to grade a student's answer to a problem by comparing it against the official solution and following the grading guidelines.

## Problem Statement:
{problem}

## Official Solution:
{solution}

## Grading Guidelines:
{grading_guidelines}

## Student's Answer:
{student_answer}

## Instructions:
1. First, analyze the student's answer step by step. Identify what they did correctly and incorrectly.
2. Compare their approach to the official solution.
3. Consider the grading guidelines carefully.
4. Determine if the answer is correct, partially correct, or incorrect.
5. Provide your final assessment in the JSON format below.

Respond in JSON format with the following schema:
<json>
{{
    "reasoning": "Your detailed step-by-step analysis of the student's answer",
    "response": "Your final grading decision (e.g., 'Correct', 'Partially Correct', 'Incorrect', or a numeric score)"
}}
</json>

Be thorough in your reasoning before providing the final response."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON
        prediction = "None"
        try:
            extracted = _extract_jsons(msg_history[-1]["text"])
            if extracted and "response" in extracted[-1]:
                prediction = extracted[-1]["response"]
                reasoning = extracted[-1].get("reasoning", "No reasoning provided")
                self.log_fn(f"Successfully extracted prediction: {repr(prediction)[:100]}")
                self.log_fn(f"Reasoning: {repr(reasoning)[:200]}...")
            elif extracted:
                self.log_fn(f"Warning: JSON extracted but 'response' key missing. Keys: {list(extracted[-1].keys())}")
                # Try to use whatever key is available
                if extracted:
                    prediction = str(extracted[-1])
            else:
                self.log_fn("Warning: No JSON blocks found in response")
                # Fallback: try to extract any meaningful text
                text = msg_history[-1]["text"]
                # Look for common patterns like "Answer: X" or "Final answer: X"
                for pattern in [r"[Ff]inal answer:\s*(.+?)(?:\n|$)", r"[Rr]esponse:\s*(.+?)(?:\n|$)", r"[Cc]onclusion:\s*(.+?)(?:\n|$)"]:
                    match = re.search(pattern, text)
                    if match:
                        prediction = match.group(1).strip()
                        self.log_fn(f"Fallback extraction found: {repr(prediction)[:100]}")
                        break
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {type(e).__name__}: {e}")

        return str(prediction), msg_history
