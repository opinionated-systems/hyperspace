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


def _extract_grade_from_text(text: str) -> str | None:
    """Extract grade from plain text if JSON parsing fails."""
    valid_grades = {"correct", "almost", "partial", "incorrect"}
    text_lower = text.lower()
    
    # Look for explicit grade mentions
    for grade in valid_grades:
        # Check for grade in quotes or as standalone word
        patterns = [
            rf'"{grade}"',
            rf"'{grade}'",
            rf'\b{grade}\b',
        ]
        for pattern in patterns:
            if re.search(pattern, text_lower):
                return grade
    return None


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

## Instructions
1. First, analyze the student's answer step by step. Compare it to the correct solution.
2. Identify what the student did correctly and what errors they made.
3. Consider the grading guidelines carefully - these define the specific criteria for each grade level.
4. Provide your reasoning for the grade you will assign, citing specific evidence from the student's answer.
5. Finally, provide your grade assessment in the JSON format below.

## Grade Definitions
- "correct": The student's answer is fully correct, complete, and matches the solution. All key steps and reasoning are present and accurate.
- "almost": The student's answer is nearly correct with only minor gaps or small errors. The core approach is right but missing minor details.
- "partial": The student's answer has significant gaps or errors but contains some correct elements or partial progress toward the solution.
- "incorrect": The student's answer is fundamentally wrong, does not address the problem, or shows no meaningful understanding.

CRITICAL: The "response" field MUST contain EXACTLY one of these four values: "correct", "almost", "partial", or "incorrect". No other text, no explanations, just the exact grade label.

Respond in JSON format with the following schema:
<json>
{{
    "reasoning": "Your detailed step-by-step analysis and reasoning...",
    "response": "correct" | "almost" | "partial" | "incorrect"
}}
</json>"""

        # First attempt
        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON
        prediction = self._extract_prediction(msg_history)
        
        # If extraction failed or invalid grade, try once more with a clearer prompt
        valid_grades = {"correct", "almost", "partial", "incorrect"}
        if prediction not in valid_grades:
            self.log_fn(f"First attempt failed, got: {prediction}. Retrying...")
            retry_instruction = f"""Your previous response did not provide a valid grade. Please re-evaluate and provide ONLY a valid JSON response.

The student's answer was:
{student_answer}

You MUST respond with EXACTLY this JSON format (no other text):
<json>
{{
    "reasoning": "Brief analysis of why the answer deserves this grade...",
    "response": "correct" | "almost" | "partial" | "incorrect"
}}
</json>

Remember: the "response" field must be EXACTLY one of: "correct", "almost", "partial", or "incorrect"."""
            
            response, msg_history, info = get_response_from_llm(
                msg=retry_instruction,
                model=self.model,
                msg_history=msg_history,
            )
            prediction = self._extract_prediction(msg_history)

        return str(prediction), msg_history

    def _extract_prediction(self, msg_history: list[dict]) -> str:
        """Extract and validate the grade prediction from message history."""
        valid_grades = {"correct", "almost", "partial", "incorrect"}
        
        try:
            # Get the last assistant message
            last_msg = None
            for msg in reversed(msg_history):
                if msg.get("role") == "assistant":
                    last_msg = msg
                    break
            
            if not last_msg:
                return "None"
            
            text = last_msg.get("text", "")
            
            # Try to extract JSON
            extracted = _extract_jsons(text)
            if extracted:
                last_json = extracted[-1]
                raw_response = None
                
                # Try multiple possible field names
                for field in ["response", "grade", "answer", "result", "evaluation"]:
                    if field in last_json:
                        raw_response = last_json[field]
                        break
                
                if raw_response and isinstance(raw_response, str):
                    normalized = raw_response.strip().lower()
                    # Exact match first
                    if normalized in valid_grades:
                        return normalized
                    # Substring match as fallback
                    for grade in valid_grades:
                        if grade in normalized:
                            return grade
                    # Return the raw response if no match
                    return raw_response
            
            # Fallback: try to extract grade from plain text
            grade_from_text = _extract_grade_from_text(text)
            if grade_from_text:
                return grade_from_text
                
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
        
        return "None"
