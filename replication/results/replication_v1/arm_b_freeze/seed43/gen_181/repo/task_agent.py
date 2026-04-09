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
    """Extract JSON objects from <json>...</json> blocks or markdown code blocks.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    Handles both <json> tags and markdown code blocks (```json...```).
    """
    results = []
    
    # Try <json>...</json> tags first
    results.extend(_extract_from_tags(text, "<json>", "</json>", 6, 7))
    
    # Try markdown code blocks
    results.extend(_extract_from_markdown(text))
    
    return results if results else None


def _extract_from_tags(text: str, start_tag: str, end_tag: str, start_len: int, end_len: int) -> list[dict]:
    """Extract JSON objects from tagged blocks."""
    results = []
    search_from = 0
    
    while True:
        start = text.find(start_tag, search_from)
        if start == -1:
            break
        end = text.find(end_tag, start)
        if end == -1:
            break
        
        inner = text[start + start_len:end].strip()
        search_from = end + end_len
        
        # Try direct parsing first
        parsed = _try_parse_json(inner)
        if parsed is not None:
            results.append(parsed)
        else:
            # Try extracting from within braces
            parsed = _extract_from_braces(inner)
            if parsed is not None:
                results.append(parsed)
    
    return results


def _extract_from_markdown(text: str) -> list[dict]:
    """Extract JSON objects from markdown code blocks."""
    results = []
    search_from = 0
    
    while True:
        # Look for ```json or ```
        json_start = text.find("```json", search_from)
        plain_start = text.find("```", search_from)
        
        if json_start != -1 and (plain_start == -1 or json_start < plain_start):
            start = json_start
            offset = 7
        elif plain_start != -1:
            start = plain_start
            offset = 3
        else:
            break
        
        end_marker = text.find("```", start + offset)
        if end_marker == -1:
            break
        
        inner = text[start + offset:end_marker].strip()
        search_from = end_marker + 3
        
        # Try direct parsing first
        parsed = _try_parse_json(inner)
        if parsed is not None:
            results.append(parsed)
        else:
            # Try extracting from within braces
            parsed = _extract_from_braces(inner)
            if parsed is not None:
                results.append(parsed)
    
    return results


def _try_parse_json(text: str) -> dict | None:
    """Try to parse text as JSON."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _extract_from_braces(text: str) -> dict | None:
    """Extract JSON object from text by finding outermost braces."""
    brace_start = text.find('{')
    brace_end = text.rfind('}')
    if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
        try:
            return json.loads(text[brace_start:brace_end+1])
        except json.JSONDecodeError:
            return None
    return None


def _extract_any_json(text: str) -> list[dict] | None:
    """Fallback JSON extraction that looks for any JSON objects in text."""
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
                    results.append(obj)
                except json.JSONDecodeError:
                    pass
                start_idx = -1
    
    return results if results else None


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    # Field names to check for prediction, in order of preference
    PREDICTION_FIELDS = ["response", "grade", "answer", "result", "evaluation", "prediction"]

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
        # Extract task components for better prompting
        domain = inputs.get("domain", "Unknown")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = self._build_instruction(
            domain, problem, solution, grading_guidelines, student_answer
        )

        self.log_fn(f"Processing task in domain: {domain}")
        
        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON with fallback mechanisms
        prediction = self._extract_prediction(msg_history)
        
        self.log_fn(f"Extracted prediction: {prediction}")

        return str(prediction), msg_history

    def _build_instruction(
        self,
        domain: str,
        problem: str,
        solution: str,
        grading_guidelines: str,
        student_answer: str
    ) -> str:
        """Build the prompt instruction for the LLM."""
        return f"""You are an expert mathematical grader for International Mathematical Olympiad (IMO) problems.

Your task is to evaluate a student's answer to a mathematics problem and provide a grade with detailed reasoning.

DOMAIN: {domain}

PROBLEM:
{problem}

OFFICIAL SOLUTION:
{solution}

GRADING GUIDELINES:
{grading_guidelines}

STUDENT'S ANSWER:
{student_answer}

Think step by step and provide a thorough analysis:
1. PROBLEM ANALYSIS: Identify the key mathematical concepts, theorems, and techniques required
2. SOLUTION REVIEW: Analyze the official solution's approach, key steps, and expected answer format
3. STUDENT WORK ANALYSIS: 
   - Identify what approach the student took
   - Note any correct steps or valid insights
   - Identify errors, gaps, or misconceptions
4. GRADING CRITERIA CHECK:
   - Verify if the student met each criterion in the grading guidelines
   - Note partial credit for incomplete but valid reasoning
5. FINAL DETERMINATION: Assign grade based on completeness, correctness, and adherence to guidelines

Respond in JSON format with the following schema:
<json>
{{
    "reasoning": "Your detailed chain-of-thought analysis covering all 5 steps above",
    "response": "The final grade/prediction (e.g., 'Correct', 'Incorrect', 'Partial', or a numeric score)"
}}
</json>

Important: Be objective and consistent. Award partial credit when the student shows valid reasoning even if the final answer is incorrect."""

    def _extract_prediction(self, msg_history: list[dict]) -> str:
        """Extract prediction from message history with fallback mechanisms."""
        if not msg_history:
            self.log_fn("Warning: Empty message history")
            return "None"
        
        try:
            last_message = msg_history[-1].get("text", "")
            
            if not last_message:
                self.log_fn("Warning: Last message has no text content")
                return "None"
            
            # Try primary extraction method
            extracted = _extract_jsons(last_message)
            
            # Fallback to generic JSON extraction if primary fails
            if extracted is None:
                self.log_fn("Primary extraction failed, trying fallback")
                extracted = _extract_any_json(last_message)
            
            if not extracted:
                self.log_fn("Warning: No JSON found in response")
                return "None"
            
            # Get the last JSON object (most recent response)
            last_json = extracted[-1]
            
            # Try known field names in order of preference
            for field in self.PREDICTION_FIELDS:
                if field in last_json:
                    value = last_json[field]
                    if isinstance(value, str):
                        return value
                    # Convert non-string values to string
                    return str(value)
            
            # If no known field, use the first string value found
            for key, value in last_json.items():
                if isinstance(value, str):
                    self.log_fn(f"Using first string field '{key}' for prediction")
                    return value
            
            # If no string values, convert first value to string
            first_key = list(last_json.keys())[0] if last_json else None
            if first_key:
                self.log_fn(f"Using first available field '{first_key}' for prediction")
                return str(last_json[first_key])
            
            self.log_fn("Warning: JSON object has no fields")
            return "None"
            
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            return "None"
