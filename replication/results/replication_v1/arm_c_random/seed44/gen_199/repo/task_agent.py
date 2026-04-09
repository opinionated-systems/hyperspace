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
    Also handles markdown code blocks (```json...```) as fallback.
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
            # Try to extract just the JSON object if there's extra text
            try:
                # Find the first { and last }
                json_start = inner.find("{")
                json_end = inner.rfind("}")
                if json_start != -1 and json_end != -1 and json_end > json_start:
                    results.append(json.loads(inner[json_start:json_end + 1]))
            except json.JSONDecodeError:
                continue
    
    # Fallback: try markdown code blocks if no <json> blocks found
    if not results:
        search_from = 0
        while True:
            start = text.find("```json", search_from)
            if start == -1:
                # Also try without "json" specifier
                start = text.find("```", search_from)
                if start == -1:
                    break
                end = text.find("```", start + 3)
                if end == -1:
                    break
                inner = text[start + 3:end].strip()
                search_from = end + 3
            else:
                end = text.find("```", start + 7)
                if end == -1:
                    break
                inner = text[start + 7:end].strip()
                search_from = end + 3
            
            try:
                results.append(json.loads(inner))
            except json.JSONDecodeError:
                # Try to extract just the JSON object
                try:
                    json_start = inner.find("{")
                    json_end = inner.rfind("}")
                    if json_start != -1 and json_end != -1 and json_end > json_start:
                        results.append(json.loads(inner[json_start:json_end + 1]))
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
        # Extract fields for structured prompting
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

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

Please evaluate the student's answer following these steps:

1. **Understand the Problem**: Briefly summarize what the problem is asking and the key mathematical concepts involved.

2. **Analyze the Official Solution**: Identify the critical steps, theorems, and reasoning required for a correct solution.

3. **Evaluate the Student's Answer**: 
   - Check if the student correctly identified the approach
   - Verify each step against the official solution
   - Note any errors, omissions, or creative valid alternatives
   - Consider partial credit according to the grading guidelines

4. **Determine the Score**: Based on your analysis, assign a numerical score that reflects the student's performance.

Respond in JSON format with the following schema:
<json>
{{
    "reasoning": "Your detailed chain-of-thought analysis here...",
    "score_breakdown": {{
        "correctness": "analysis of mathematical correctness",
        "completeness": "analysis of solution completeness", 
        "clarity": "analysis of presentation clarity"
    }},
    "response": <numerical_score>
}}
</json>

The "response" field must contain only the numerical score (e.g., 7, 3.5, 0, etc.)."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON
        prediction = "None"
        try:
            # Try to extract from the last assistant message
            last_msg = msg_history[-1] if msg_history else None
            if last_msg and last_msg.get("role") == "assistant":
                text = last_msg.get("text", "")
                extracted = _extract_jsons(text)
                if extracted:
                    last_json = extracted[-1]
                    if "response" in last_json:
                        prediction = last_json["response"]
                        # Ensure prediction is a valid number or string representation
                        if isinstance(prediction, (int, float)):
                            prediction = str(prediction)
                        elif isinstance(prediction, str):
                            # Try to clean up common formatting issues
                            prediction = prediction.strip()
                    # Log reasoning if available for debugging
                    if "reasoning" in last_json:
                        reasoning = last_json["reasoning"]
                        if isinstance(reasoning, str):
                            self.log_fn(f"Agent reasoning: {reasoning[:200]}...")
                        else:
                            self.log_fn(f"Agent reasoning: {str(reasoning)[:200]}...")
                    # Log score breakdown if available
                    if "score_breakdown" in last_json:
                        breakdown = last_json["score_breakdown"]
                        self.log_fn(f"Score breakdown: {breakdown}")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
