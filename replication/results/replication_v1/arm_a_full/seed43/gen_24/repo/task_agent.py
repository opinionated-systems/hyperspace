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
    Also handles nested JSON objects within the tags.
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
        
        # Try to parse the inner content as JSON
        try:
            results.append(json.loads(inner))
        except json.JSONDecodeError:
            # Try to find valid JSON by progressively parsing
            # This handles cases where there might be extra text
            for i in range(len(inner), 0, -1):
                try:
                    candidate = inner[:i]
                    parsed = json.loads(candidate)
                    results.append(parsed)
                    break
                except json.JSONDecodeError:
                    continue
    return results or None


def _extract_json_fallback(text: str) -> dict | None:
    """Fallback JSON extraction for non-tagged JSON or markdown code blocks.

    Tries to find JSON in markdown code blocks or raw JSON objects.
    """
    patterns = [
        r'```json\s*(.*?)\s*```',  # Markdown JSON blocks
        r'```\s*(\{.*?\})\s*```',  # Generic code blocks with JSON
        r'\{\s*"reasoning".*?\}',  # Raw JSON with reasoning field
        r'\{\s*"response".*?\}',   # Raw JSON with response field
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
    return None


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
        domain = inputs.get("domain", "")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = f"""You are an expert grader for the International Mathematical Olympiad (IMO). Your task is to evaluate a student's answer to a mathematical problem.

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

## Your Task

Please evaluate the student's answer following these steps:

1. **Understand the Problem**: Briefly summarize what the problem is asking and what the key mathematical concepts are.

2. **Analyze the Official Solution**: Identify the key steps and insights in the official solution.

3. **Evaluate the Student's Answer**: 
   - Check if the student correctly understood the problem
   - Identify which parts of the solution the student got right
   - Identify any errors, gaps, or misconceptions
   - Compare against the grading guidelines

4. **Determine the Grade**: Based on the grading guidelines, assign an appropriate grade/score.

Respond in JSON format with the following schema:
<json>
{{
    "reasoning": "Your detailed chain-of-thought analysis...",
    "response": "The final grade/score as specified in the grading guidelines"
}}
</json>

IMPORTANT: 
- The "response" field must contain ONLY the grade/score (e.g., "7", "2", "0", "Correct", "Incorrect", etc.) as specified in the grading guidelines.
- Do not include any additional text, explanations, or formatting in the "response" field.
- The "reasoning" field should contain your full analysis.
- Ensure the JSON is valid and properly formatted."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON
        prediction = "None"
        last_text = msg_history[-1]["text"]
        try:
            extracted = _extract_jsons(last_text)
            if extracted:
                # Try to get response field, fall back to other common fields
                last_json = extracted[-1]
                prediction = self._extract_prediction_from_json(last_json)
            else:
                # Try fallback extraction for non-tagged JSON
                fallback = _extract_json_fallback(last_text)
                if fallback:
                    prediction = self._extract_prediction_from_json(fallback)
                    self.log_fn(f"Used fallback JSON extraction: {prediction}")
                else:
                    # Last resort: try to find any JSON-like structure in the text
                    prediction = self._extract_prediction_heuristic(last_text)
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history

    def _extract_prediction_from_json(self, data: dict) -> str:
        """Extract prediction from a JSON dict, trying multiple field names."""
        # Priority order for field names
        field_names = ["response", "grade", "score", "answer", "result", "evaluation"]
        
        for field in field_names:
            if field in data:
                value = data[field]
                # If it's a string, return it directly
                if isinstance(value, str):
                    return value.strip()
                # If it's a number, convert to string
                elif isinstance(value, (int, float)):
                    return str(value)
                # If it's a dict or list, serialize it
                else:
                    return json.dumps(value)
        
        # If no recognized field, use the whole JSON as string
        return json.dumps(data)

    def _extract_prediction_heuristic(self, text: str) -> str:
        """Last resort heuristic extraction for when JSON parsing fails."""
        # Look for common patterns like "Grade: X" or "Score: X"
        patterns = [
            r'[Gg]rade[:\s]+([\w\-\+]+)',
            r'[Ss]core[:\s]+([\w\-\+]+)',
            r'[Rr]esponse[:\s]+([\w\-\+]+)',
            r'[Aa]nswer[:\s]+([\w\-\+]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()
        
        # If all else fails, return a truncated version of the text
        return text[:100] if len(text) > 100 else text
