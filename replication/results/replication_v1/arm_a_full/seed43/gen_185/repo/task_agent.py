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
    Also handles nested JSON objects within the content.
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
            # Try to extract JSON from markdown code blocks within the content
            code_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', inner, re.DOTALL)
            if code_block_match:
                try:
                    results.append(json.loads(code_block_match.group(1)))
                except json.JSONDecodeError:
                    continue
            else:
                continue
    return results or None


def _extract_json_fallback(text: str) -> dict | None:
    """Fallback JSON extraction for non-tagged JSON or markdown code blocks.

    Tries to find JSON in markdown code blocks or raw JSON objects.
    Uses a more robust approach to handle nested braces.
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
                # Try to find the first valid JSON object by bracket matching
                try:
                    start = match.find('{')
                    if start != -1:
                        brace_count = 0
                        for i, char in enumerate(match[start:], start):
                            if char == '{':
                                brace_count += 1
                            elif char == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    candidate = match[start:i+1]
                                    return json.loads(candidate)
                except (json.JSONDecodeError, ValueError):
                    continue
                continue
    
    # Last resort: try to find any JSON-like structure with brace matching
    try:
        start = text.find('{')
        while start != -1:
            brace_count = 0
            for i, char in enumerate(text[start:], start):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        candidate = text[start:i+1]
                        try:
                            parsed = json.loads(candidate)
                            # Only return if it has expected fields
                            if any(k in parsed for k in ["reasoning", "response", "grade", "score"]):
                                return parsed
                        except json.JSONDecodeError:
                            pass
                        break
            start = text.find('{', start + 1)
    except Exception:
        pass
    
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

        instruction = f"""You are an expert grader for the International Mathematical Olympiad (IMO). Your task is to evaluate a student's answer to a mathematical problem with precision and consistency.

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

Please evaluate the student's answer following this structured approach:

1. **Problem Analysis**: Summarize the problem's requirements and key mathematical concepts (2-3 sentences).

2. **Solution Mapping**: Break down the official solution into key steps and identify which are essential vs. optional for a complete solution.

3. **Student Work Evaluation**:
   - **Correct Elements**: List specific correct steps, theorems applied, or insights demonstrated
   - **Errors/Gaps**: Identify any mathematical errors, logical gaps, or missing steps
   - **Partial Credit Analysis**: For each missing element, assess if partial credit applies per guidelines

4. **Grade Justification**: Explicitly map your grade to the grading guidelines criteria. State which criteria are met and which are not.

5. **Final Grade**: Assign the grade that best matches the student's demonstrated understanding.

Respond STRICTLY in JSON format with this exact schema:
<json>
{{
    "reasoning": "Your detailed chain-of-thought analysis following the 5 steps above...",
    "response": "The final grade/score as specified in the grading guidelines"
}}
</json>

IMPORTANT: The "response" field must contain ONLY the grade/score value (e.g., "7", "2", "0", "Correct", "Incorrect") exactly as specified in the grading guidelines. Do not add explanations in this field."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON
        prediction = "None"
        last_text = msg_history[-1]["text"]
        extraction_method = "none"
        
        try:
            # First try: extract from <json> tags
            extracted = _extract_jsons(last_text)
            if extracted:
                extraction_method = "json_tags"
                # Try to get response field, fall back to other common fields
                last_json = extracted[-1]
                prediction = self._extract_prediction_from_dict(last_json)
            else:
                # Second try: fallback extraction for non-tagged JSON
                fallback = _extract_json_fallback(last_text)
                if fallback:
                    extraction_method = "fallback"
                    prediction = self._extract_prediction_from_dict(fallback)
                    self.log_fn(f"Used fallback JSON extraction: {prediction}")
                else:
                    # Third try: look for grade/score in plain text
                    text_prediction = self._extract_prediction_from_text(last_text)
                    if text_prediction:
                        extraction_method = "text_heuristic"
                        prediction = text_prediction
                        self.log_fn(f"Used text heuristic extraction: {prediction}")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        # Log extraction method for debugging
        self.log_fn(f"Prediction extraction method: {extraction_method}, result: {prediction}")
        
        return str(prediction), msg_history

    def _extract_prediction_from_dict(self, data: dict) -> str:
        """Extract prediction value from a dictionary with priority ordering."""
        priority_fields = ["response", "grade", "score", "answer", "result", "value"]
        for field in priority_fields:
            if field in data:
                value = data[field]
                # Handle nested structures
                if isinstance(value, (str, int, float, bool)):
                    return str(value)
                elif isinstance(value, dict):
                    # Try to extract from nested dict
                    nested = self._extract_prediction_from_dict(value)
                    if nested != "None":
                        return nested
        # If no recognized field, use the whole JSON as string
        return json.dumps(data)

    def _extract_prediction_from_text(self, text: str) -> str | None:
        """Try to extract a grade/score from plain text using heuristics."""
        # Look for common grade patterns with priority ordering
        patterns = [
            # IMO-specific patterns (highest priority)
            r'[Ff]inal grade[:\s]+([0-7])\s*(?:/\s*7)?',
            r'[Gg]rade[:\s]+([0-7])\s*(?:/\s*7)?',
            r'[Ss]core[:\s]+([0-7])\s*(?:/\s*7)?',
            r'[Ff]inal score[:\s]+([0-7])\s*(?:/\s*7)?',
            # General numeric grades
            r'[Ff]inal grade[:\s]+([0-9]+(?:\.[0-9]+)?)',
            r'[Gg]rade[:\s]+([0-9]+(?:\.[0-9]+)?)',
            r'[Ss]core[:\s]+([0-9]+(?:\.[0-9]+)?)',
            r'[Ff]inal score[:\s]+([0-9]+(?:\.[0-9]+)?)',
            # Letter grades
            r'[Ff]inal grade[:\s]+([A-F][+-]?)',
            r'[Gg]rade[:\s]+([A-F][+-]?)',
            # Binary results
            r'[Rr]esult[:\s]+(Correct|Incorrect|Pass|Fail)',
            r'(?:^|\n)\s*([0-7])\s*(?:/\s*7)?\s*(?:$|\n)',  # Standalone IMO grades
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()
        
        # Last resort: look for standalone numbers 0-7 (common IMO range)
        # but only if they appear near grade-related words
        grade_context = re.search(r'(?:grade|score|mark)[^\n]{0,50}([0-7])', text, re.IGNORECASE)
        if grade_context:
            return grade_context.group(1).strip()
            
        return None
