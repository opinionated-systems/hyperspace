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
from typing import Any

from agent.llm_client import get_response_from_llm, EVAL_MODEL

logger = logging.getLogger(__name__)

# Valid grade labels for IMO grading
VALID_GRADES = {"correct", "incorrect", "partial", "almost"}


def _validate_inputs(inputs: dict) -> tuple[bool, str]:
    """Validate that all required fields are present in inputs.
    
    Returns:
        (is_valid, error_message)
    """
    required_fields = ["domain", "problem", "solution", "grading_guidelines", "student_answer"]
    missing = [f for f in required_fields if f not in inputs or not inputs[f]]
    if missing:
        return False, f"Missing required fields: {', '.join(missing)}"
    return True, ""


def _normalize_grade(grade: str) -> str:
    """Normalize a grade string to a standard format.
    
    Maps various grade formats to standard labels.
    """
    if not grade:
        return "None"
    
    grade_lower = str(grade).lower().strip()
    
    # Direct matches
    if grade_lower in VALID_GRADES:
        return grade_lower
    
    # Numeric grades (0-7 scale common in IMO)
    if grade_lower in ("0", "1", "2"):
        return "incorrect"
    if grade_lower in ("3", "4"):
        return "partial"
    if grade_lower in ("5", "6"):
        return "almost"
    if grade_lower == "7":
        return "correct"
    
    # Text patterns
    if any(word in grade_lower for word in ["correct", "right", "true", "valid", "7"]):
        return "correct"
    if any(word in grade_lower for word in ["incorrect", "wrong", "false", "invalid", "0"]):
        return "incorrect"
    if any(word in grade_lower for word in ["partial", "partly", "some", "incomplete"]):
        return "partial"
    if any(word in grade_lower for word in ["almost", "nearly", "close"]):
        return "almost"
    
    return grade_lower


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    Also handles markdown code blocks (```json...```) as fallback.
    """
    results = []
    search_from = 0
    
    # First try <json>...</json> tags
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
            # Try to extract JSON from within the content if it's wrapped in other text
            try:
                # Find first { and last }
                json_start = inner.find("{")
                json_end = inner.rfind("}")
                if json_start != -1 and json_end != -1 and json_end > json_start:
                    results.append(json.loads(inner[json_start:json_end + 1]))
            except json.JSONDecodeError:
                continue
    
    # Fallback: try markdown code blocks
    if not results:
        search_from = 0
        while True:
            start = text.find("```json", search_from)
            if start == -1:
                # Also try without 'json' specifier
                start = text.find("```", search_from)
                if start == -1:
                    break
                end = text.find("```", start + 3)
            else:
                end = text.find("```", start + 7)
            if end == -1:
                break
            
            if text[start:start+7] == "```json":
                inner = text[start + 7:end].strip()
            else:
                inner = text[start + 3:end].strip()
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
        """Run the task agent on a single problem with structured reasoning.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Validate inputs
        is_valid, error_msg = _validate_inputs(inputs)
        if not is_valid:
            logger.error(f"Input validation failed: {error_msg}")
            return "None", [{"role": "system", "text": f"Error: {error_msg}"}]
        
        # Extract fields for structured prompting
        domain = inputs.get("domain", "")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = f"""You are an expert mathematical grader for International Mathematical Olympiad (IMO) problems.

Your task is to evaluate a student's answer by comparing it against the official solution and grading guidelines.

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
1. First, analyze the student's answer step by step. Identify what they got right and what they got wrong.
2. Compare their reasoning against the official solution.
3. Check if they followed the grading guidelines.
4. Provide your reasoning in the "analysis" field.
5. Provide your final grade in the "response" field using ONLY one of these exact labels: "correct", "incorrect", "partial", or "almost".

## Grading Label Definitions:
- **correct**: The student's answer is completely correct, with valid reasoning and correct final answer.
- **incorrect**: The student's answer is completely wrong, with fundamental errors or no valid reasoning.
- **partial**: The student made some progress but the solution is incomplete or has significant gaps.
- **almost**: The student was very close to the correct solution, with minor errors or missing small details.

Respond in JSON format with the following schema:
<json>
{{
    "analysis": "Your detailed step-by-step analysis of the student's answer...",
    "response": "correct" | "incorrect" | "partial" | "almost"
}}
</json>"""

        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return "None", [{"role": "system", "text": f"Error: LLM call failed - {e}"}]

        # Extract prediction from JSON
        prediction = self._extract_prediction(msg_history)
        
        # Normalize the grade to standard format
        normalized_prediction = _normalize_grade(prediction)
        
        # Log if normalization changed the value
        if prediction != normalized_prediction:
            logger.info(f"Grade normalized: '{prediction}' -> '{normalized_prediction}'")

        return str(normalized_prediction), msg_history

    def _extract_prediction(self, msg_history: list[dict]) -> str:
        """Extract prediction from message history.
        
        Tries multiple extraction strategies and field names for robustness.
        """
        if not msg_history:
            self.log_fn("No message history available")
            return "None"
        
        # Get the last assistant message
        last_msg = msg_history[-1]
        text = last_msg.get("text", "")
        
        if not text:
            self.log_fn("Last message has no text content")
            return "None"
        
        # Try to extract JSON blocks
        extracted = _extract_jsons(text)
        
        if not extracted:
            # Fallback: try to find any JSON-like structure in the text
            self.log_fn("No JSON blocks found, trying fallback extraction")
            try:
                # Look for patterns like "response": "..." or "grade": "..."
                response_match = re.search(r'["\']response["\']\s*:\s*["\']([^"\']+)["\']', text, re.IGNORECASE)
                if response_match:
                    return response_match.group(1)
                grade_match = re.search(r'["\']grade["\']\s*:\s*["\']([^"\']+)["\']', text, re.IGNORECASE)
                if grade_match:
                    return grade_match.group(1)
                score_match = re.search(r'["\']score["\']\s*:\s*["\']?([^"\'\s,}]+)', text, re.IGNORECASE)
                if score_match:
                    return score_match.group(1)
                # Try to find standalone grade words in the text
                for grade in VALID_GRADES:
                    if re.search(rf'\b{grade}\b', text, re.IGNORECASE):
                        return grade
            except Exception as e:
                self.log_fn(f"Fallback extraction failed: {e}")
            return "None"
        
        # Try to get response from extracted JSON
        last_extract = extracted[-1]
        
        # Priority order for field names
        field_priority = ["response", "grade", "score", "result", "evaluation", "verdict", "label"]
        
        for field in field_priority:
            if field in last_extract:
                value = last_extract[field]
                # Log the analysis if available for debugging
                if "analysis" in last_extract:
                    self.log_fn(f"Analysis: {str(last_extract['analysis'])[:200]}...")
                return str(value)
        
        # If no known field found, return the first string value
        for key, value in last_extract.items():
            if isinstance(value, str) and value:
                return value
            # Also check for numeric values
            if isinstance(value, (int, float)):
                return str(value)
        
        self.log_fn("Could not extract any valid prediction from JSON")
        return "None"
