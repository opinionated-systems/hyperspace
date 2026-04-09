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
from agent.utils import safe_json_extract, log_execution_time, format_error_for_display

logger = logging.getLogger(__name__)


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    Also handles markdown code blocks (```json...```) as fallback.
    
    Enhanced with safe_json_extract utility for better error handling.
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
    
    # If no <json> blocks found, try markdown code blocks
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
                # Try to extract just the JSON object
                try:
                    json_start = inner.find("{")
                    json_end = inner.rfind("}")
                    if json_start != -1 and json_end != -1 and json_end > json_start:
                        results.append(json.loads(inner[json_start:json_end + 1]))
                except json.JSONDecodeError:
                    continue
    
    # Fallback: use safe_json_extract utility for additional extraction attempts
    if not results:
        extracted = safe_json_extract(text, max_attempts=4)
        if extracted:
            results.append(extracted)
    
    return results or None


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    @log_execution_time
    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Extract fields for structured prompting
        domain = inputs.get("domain", "")
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
1. Analyze the student's answer step by step
2. Compare it against the official solution and grading guidelines
3. Identify any errors, missing steps, or correct approaches
4. Determine the appropriate grade/score

IMPORTANT: You must provide your final evaluation in the exact JSON format shown below. The JSON must be wrapped in <json> tags.

<json>
{{
    "reasoning": "Your detailed step-by-step analysis here",
    "score": "The numerical score or grade",
    "response": "The final grade/score as a number or string"
}}
</json>

The "response" field should contain the final numerical score/grade that will be used for evaluation."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON
        prediction = "None"
        try:
            extracted = _extract_jsons(msg_history[-1]["text"])
            if extracted:
                # Try to get response field first, then score, then any numeric value
                last_json = extracted[-1]
                if "response" in last_json:
                    prediction = last_json["response"]
                elif "score" in last_json:
                    prediction = last_json["score"]
                elif "grade" in last_json:
                    prediction = last_json["grade"]
                elif "evaluation" in last_json:
                    prediction = last_json["evaluation"]
                elif "result" in last_json:
                    prediction = last_json["result"]
                else:
                    # Fallback: use the first value that's not "reasoning"
                    for key, value in last_json.items():
                        if key != "reasoning" and isinstance(value, (str, int, float)):
                            prediction = value
                            break
            else:
                # No JSON found - try to extract a numeric score from the text
                text = msg_history[-1]["text"]
                # Look for patterns like "Score: 7" or "Grade: 7/7" or "The score is 5"
                score_patterns = [
                    r'[Ss]core[:\s]+(\d+(?:\.\d+)?)',
                    r'[Gg]rade[:\s]+(\d+(?:\.\d+)?)',
                    r'[Ff]inal score[:\s]+(\d+(?:\.\d+)?)',
                    r'[Tt]he score is[:\s]+(\d+(?:\.\d+)?)',
                    r'[Tt]he grade is[:\s]+(\d+(?:\.\d+)?)',
                    r'[Aa]warded[:\s]+(\d+(?:\.\d+)?)',
                    r'[Tt]otal[:\s]+(\d+(?:\.\d+)?)',
                ]
                for pattern in score_patterns:
                    match = re.search(pattern, text)
                    if match:
                        prediction = match.group(1)
                        break
        except Exception as e:
            error_msg = format_error_for_display(e, context="prediction extraction")
            self.log_fn(f"Error extracting prediction: {error_msg}")

        return str(prediction), msg_history
