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

    def _build_grading_prompt(self, inputs: dict) -> str:
        """Build the structured grading prompt with chain-of-thought reasoning."""
        domain = inputs.get("domain", "Unknown")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        max_score = inputs.get("max_score", 7)  # IMO problems are typically out of 7

        return f"""You are an expert IMO (International Mathematical Olympiad) grader. Your task is to evaluate a student's solution to a mathematical problem with rigorous analysis.

## Domain
{domain}

## Problem Statement
{problem}

## Official Solution
{solution}

## Grading Guidelines
{grading_guidelines}

## Student's Answer
{student_answer}

## Maximum Score
{max_score} points

## Your Task
Follow this structured evaluation process:

### Step 1: Key Steps Analysis
Identify the key logical steps in the official solution. For each step, determine:
- Is this step present in the student's answer?
- Is it correctly executed?
- Are there any gaps or errors?

### Step 2: Error Analysis
If the student's answer contains errors, classify them:
- Conceptual errors (misunderstanding of problem/theorem)
- Computational errors (arithmetic/algebraic mistakes)
- Logical gaps (missing reasoning steps)
- Notation/communication issues

### Step 3: Partial Credit Assessment
Based on the grading guidelines, determine what partial credit is warranted:
- Which key insights did the student demonstrate?
- How far did they progress toward a complete solution?
- What is the appropriate numerical score (0-{max_score})?

### Step 4: Final Evaluation
Provide your complete evaluation in the following JSON format:

<json>
{{
    "numerical_score": <integer between 0 and {max_score}>,
    "reasoning": "Brief summary of your step-by-step analysis",
    "key_strengths": ["List of what the student did correctly"],
    "errors_found": ["List of specific errors or gaps"],
    "partial_credit_justification": "Explanation of why this score is appropriate",
    "response": "Your detailed feedback for the student. Include: (1) overall assessment, (2) specific errors with corrections, (3) what was done well, (4) the final score ({max_score} points max), (5) constructive suggestions for improvement"
}}
</json>

Be thorough, objective, and consistent with IMO grading standards. Remember that IMO grading rewards correct reasoning over perfect presentation."""

    def _validate_grading_output(self, result: dict, max_score: int) -> dict:
        """Validate and sanitize the grading output."""
        # Ensure numerical_score is valid
        score = result.get("numerical_score")
        if not isinstance(score, int) or score < 0 or score > max_score:
            result["numerical_score"] = 0
            result["validation_note"] = "Invalid score detected, defaulting to 0"
        
        # Ensure required fields exist
        required_fields = ["response", "reasoning", "key_strengths", "errors_found"]
        for field in required_fields:
            if field not in result:
                result[field] = "Not provided" if field in ["response", "reasoning", "partial_credit_justification"] else []
        
        return result

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with chain-of-thought reasoning.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        instruction = self._build_grading_prompt(inputs)
        max_score = inputs.get("max_score", 7)

        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
        except Exception as e:
            self.log_fn(f"LLM call failed: {e}")
            return f"Error: LLM call failed - {e}", []

        # Extract and validate prediction from JSON
        prediction = "None"
        try:
            extracted = _extract_jsons(msg_history[-1]["text"])
            if extracted:
                result = self._validate_grading_output(extracted[-1], max_score)
                
                # Build comprehensive prediction string
                score = result.get("numerical_score", 0)
                reasoning = result.get("reasoning", "")
                strengths = result.get("key_strengths", [])
                errors = result.get("errors_found", [])
                justification = result.get("partial_credit_justification", "")
                response_text = result.get("response", "")
                
                prediction = f"""Score: {score}/{max_score}

Reasoning: {reasoning}

Key Strengths:
{chr(10).join(f'- {s}' for s in strengths) if strengths else '- None identified'}

Errors Found:
{chr(10).join(f'- {e}' for e in errors) if errors else '- None'}

Partial Credit Justification: {justification}

Detailed Feedback:
{response_text}"""
                
                self.log_fn(f"Graded solution: {score}/{max_score}")
            else:
                self.log_fn("No valid JSON found in output")
                prediction = msg_history[-1].get("text", "No response")[:500]
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            prediction = f"Error parsing result: {e}"

        return str(prediction), msg_history
