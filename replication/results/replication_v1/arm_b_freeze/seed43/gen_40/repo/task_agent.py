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
import time
from typing import Any, Callable

from agent.llm_client import get_response_from_llm, EVAL_MODEL

logger = logging.getLogger(__name__)


class JSONExtractor:
    """Handles JSON extraction from LLM responses with multiple fallback strategies."""

    # Priority order of fields to extract for the prediction
    PREDICTION_FIELDS = [
        "response", "grade", "answer", "result", 
        "evaluation", "prediction", "score"
    ]

    @classmethod
    def extract(cls, text: str, log_fn: Callable = None) -> tuple[list[dict] | None, str]:
        """
        Extract JSON objects from text using multiple strategies.
        
        Returns:
            Tuple of (extracted_jsons, method_used)
        """
        # Strategy 1: Explicit <json> tags
        result = cls._extract_from_tags(text)
        if result:
            if log_fn:
                log_fn(f"JSON extracted using tag strategy: {len(result)} object(s)")
            return result, "tags"

        # Strategy 2: Markdown code blocks
        result = cls._extract_from_markdown(text)
        if result:
            if log_fn:
                log_fn(f"JSON extracted using markdown strategy: {len(result)} object(s)")
            return result, "markdown"

        # Strategy 3: Generic brace matching
        result = cls._extract_from_braces(text)
        if result:
            if log_fn:
                log_fn(f"JSON extracted using brace strategy: {len(result)} object(s)")
            return result, "braces"

        return None, "none"

    @classmethod
    def _extract_from_tags(cls, text: str) -> list[dict] | None:
        """Extract JSON objects from <json>...</json> blocks."""
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

    @classmethod
    def _extract_from_braces(cls, text: str) -> list[dict] | None:
        """Extract JSON objects by matching curly braces."""
        results = []
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
                        obj = json.loads(text[start_idx:i+1])
                        if isinstance(obj, dict):
                            results.append(obj)
                    except json.JSONDecodeError:
                        pass
                    start_idx = -1
        return results or None

    @classmethod
    def _extract_from_markdown(cls, text: str) -> list[dict] | None:
        """Extract JSON from markdown code blocks."""
        results = []
        json_pattern = r'```(?:json)?\s*([\s\S]*?)```'
        matches = re.findall(json_pattern, text)
        for match in matches:
            try:
                obj = json.loads(match.strip())
                if isinstance(obj, dict):
                    results.append(obj)
            except json.JSONDecodeError:
                # Try nested objects within the match
                nested = cls._extract_from_braces(match)
                if nested:
                    results.extend(nested)
        return results or None

    @classmethod
    def get_prediction(cls, json_objects: list[dict]) -> str | None:
        """
        Extract prediction from JSON objects using priority field order.
        
        Returns:
            The prediction string or None if no valid prediction found.
        """
        if not json_objects:
            return None

        for obj in json_objects:
            # Check priority fields first
            for field in cls.PREDICTION_FIELDS:
                if field in obj:
                    value = obj[field]
                    if isinstance(value, str):
                        return value
                    elif isinstance(value, (int, float)):
                        return str(value)

            # Fallback: use first string or numeric value found
            for key, value in obj.items():
                if isinstance(value, str):
                    return value
                elif isinstance(value, (int, float)):
                    return str(value)

        return None


# Backward compatibility: module-level functions
def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks."""
    return JSONExtractor._extract_from_tags(text)


def _extract_any_json(text: str) -> list[dict] | None:
    """Fallback JSON extraction that looks for any JSON objects in text."""
    return JSONExtractor._extract_from_braces(text)


def _extract_json_with_regex(text: str) -> list[dict] | None:
    """Additional fallback using regex to find JSON-like structures."""
    return JSONExtractor._extract_from_markdown(text)


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.max_retries = 3
        self.base_delay = 1.0

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Extract task components for better prompting
        domain = inputs.get("domain", "Unknown")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = f"""You are an expert mathematical grader for International Mathematical Olympiad (IMO) problems.

Your task is to evaluate a student's answer to a mathematics problem and provide a grade.

DOMAIN: {domain}

PROBLEM:
{problem}

OFFICIAL SOLUTION:
{solution}

GRADING GUIDELINES:
{grading_guidelines}

STUDENT'S ANSWER:
{student_answer}

Think step by step:
1. Analyze what the problem is asking and identify key concepts
2. Review the official solution approach and identify critical steps
3. Compare the student's answer to the official solution - check for:
   - Correctness of the final answer
   - Validity of the reasoning process
   - Completeness of the solution
   - Mathematical rigor and clarity
4. Check if the student followed the grading guidelines precisely
5. Determine the appropriate grade based on the guidelines

IMPORTANT: Your response must be valid JSON wrapped in <json> tags. Do not include any text outside the JSON tags.

Respond in this exact format:
<json>
{{
    "reasoning": "Your detailed chain-of-thought analysis here. Explain your evaluation process clearly.",
    "response": "The final grade/prediction (e.g., 'Correct', 'Incorrect', 'Partial', or a numeric score)"
}}
</json>"""

        # Retry loop with exponential backoff for improved reliability
        last_exception = None
        for attempt in range(self.max_retries):
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=[],
                )
                break
            except Exception as e:
                last_exception = e
                self.log_fn(f"LLM call failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    delay = self.base_delay * (2 ** attempt)
                    self.log_fn(f"Retrying in {delay:.1f}s...")
                    time.sleep(delay)
                else:
                    self.log_fn("Max retries reached, returning error prediction")
                    return "Error: LLM call failed", []

        # Extract prediction from JSON using the unified extractor
        prediction = "None"
        try:
            last_message = msg_history[-1]["text"]
            
            # Use the unified JSON extractor with logging
            extracted, method = JSONExtractor.extract(last_message, self.log_fn)
            
            if extracted:
                self.log_fn(f"JSON extraction successful using {method} method")
                pred_value = JSONExtractor.get_prediction(extracted)
                if pred_value:
                    prediction = pred_value
                    self.log_fn(f"Extracted prediction: {prediction}")
                else:
                    self.log_fn("Warning: No valid prediction field found in extracted JSON")
            else:
                self.log_fn("Warning: No JSON objects found in LLM response")
                
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
