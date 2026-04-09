"""
Task agent: solves a given task with chain-of-thought reasoning and verification.

Enhanced version with multi-step reasoning for better IMO grading accuracy.

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


def _extract_final_answer(text: str) -> str | None:
    """Extract final answer from various formats as fallback."""
    # Try to find explicit answer markers
    patterns = [
        r'["\']?final[_\s]?answer["\']?\s*[:=]\s*["\']?([^"\']+)["\']?',
        r'["\']?answer["\']?\s*[:=]\s*["\']?([^"\']+)["\']?',
        r'["\']?result["\']?\s*[:=]\s*["\']?([^"\']+)["\']?',
        r'["\']?score["\']?\s*[:=]\s*["\']?([^"\']+)["\']?',
        r'["\']?grade["\']?\s*[:=]\s*["\']?([^"\']+)["\']?',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None


class TaskAgent:
    """Task agent that solves IMO grading problems with enhanced reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def _build_grading_prompt(self, inputs: dict) -> str:
        """Build a comprehensive grading prompt with chain-of-thought instructions."""
        domain = inputs.get("domain", "")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        return f"""You are an expert grader for {domain} problems.

Your task is to evaluate a student's answer against the official solution using the provided grading guidelines.

## Problem Statement
{problem}

## Official Solution
{solution}

## Grading Guidelines
{grading_guidelines}

## Student's Answer
{student_answer}

## Instructions

Follow this step-by-step process:

1. **Understand the Problem**: Identify what the problem is asking and the key concepts involved.

2. **Analyze the Official Solution**: Note the key steps, techniques, and final answer in the official solution.

3. **Evaluate the Student's Answer**:
   - Check if the approach is correct
   - Verify calculations and reasoning
   - Identify any errors or gaps
   - Note any creative or alternative valid approaches

4. **Apply Grading Guidelines**: Use the specific criteria provided to determine the appropriate score/grade.

5. **Provide Your Assessment**: Give a clear, justified grade/score.

Respond in JSON format with the following schema:
<json>
{{
    "reasoning": "Your detailed step-by-step analysis and reasoning process",
    "evaluation": "Summary of the student's performance - what they did right and wrong",
    "response": "The final grade/score (a number or specific grade value)"
}}
</json>

Be thorough in your reasoning. The response field should contain only the final numeric grade or score."""

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with enhanced reasoning.

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

        # Extract prediction from JSON with multiple fallback strategies
        prediction = "None"
        try:
            # First try: extract from <json> blocks
            extracted = _extract_jsons(msg_history[-1]["text"])
            if extracted:
                last_json = extracted[-1]
                if "response" in last_json:
                    prediction = last_json["response"]
                elif "grade" in last_json:
                    prediction = last_json["grade"]
                elif "score" in last_json:
                    prediction = last_json["score"]
                elif "answer" in last_json:
                    prediction = last_json["answer"]
            
            # Second try: look for explicit answer markers in the full text
            if prediction == "None":
                fallback = _extract_final_answer(msg_history[-1]["text"])
                if fallback:
                    prediction = fallback
                    
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            # Third try: extract any numeric value that looks like a grade
            try:
                text = msg_history[-1]["text"]
                # Look for patterns like "grade: 7" or "score is 5" or "final answer: 3"
                numeric_match = re.search(r'(?:grade|score|answer|result)\s*[:=\s]+\*?(\d+(?:\.\d+)?)', text, re.IGNORECASE)
                if numeric_match:
                    prediction = numeric_match.group(1)
            except Exception:
                pass

        return str(prediction), msg_history
