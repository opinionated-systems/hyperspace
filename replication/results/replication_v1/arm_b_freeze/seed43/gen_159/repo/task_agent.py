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


def _extract_brace_json(text: str) -> list[dict] | None:
    """Extract JSON objects by tracking brace balance.
    
    More robust than regex for nested structures.
    """
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


def _extract_markdown_json(text: str) -> list[dict] | None:
    """Extract JSON from markdown code blocks."""
    results = []
    # Match ```json or ``` blocks
    pattern = r'```(?:json)?\s*([\s\S]*?)```'
    matches = re.findall(pattern, text)
    
    for match in matches:
        # Try parsing the whole block first
        try:
            obj = json.loads(match.strip())
            if isinstance(obj, dict):
                results.append(obj)
                continue
        except json.JSONDecodeError:
            pass
        
        # If that fails, try extracting JSON objects from within
        nested = _extract_brace_json(match)
        if nested:
            results.extend(nested)
    
    return results or None


def _extract_all_json(text: str) -> list[dict]:
    """Extract all possible JSON objects using multiple strategies.
    
    Returns combined results from all extraction methods.
    """
    all_results = []
    seen = set()
    
    # Helper to add unique results
    def add_unique(results: list[dict] | None) -> None:
        if not results:
            return
        for obj in results:
            # Use string representation for deduplication
            obj_str = json.dumps(obj, sort_keys=True)
            if obj_str not in seen:
                seen.add(obj_str)
                all_results.append(obj)
    
    # Try all extraction methods
    add_unique(_extract_jsons(text))
    add_unique(_extract_markdown_json(text))
    add_unique(_extract_brace_json(text))
    
    return all_results


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    # Priority order for extracting prediction from JSON fields
    PREDICTION_FIELDS = [
        "response", "grade", "answer", "result", 
        "evaluation", "prediction", "score", "verdict"
    ]

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def _extract_prediction(self, text: str) -> str:
        """Extract prediction from text using all available JSON extraction methods.
        
        Args:
            text: The text to extract prediction from
            
        Returns:
            The extracted prediction string, or "None" if extraction fails
        """
        # Try all extraction methods
        extracted = _extract_all_json(text)
        
        if not extracted:
            self.log_fn("No JSON objects found in response")
            return "None"
        
        # Use the last JSON object (most likely to be the final answer)
        last_json = extracted[-1]
        
        # Try known field names in priority order
        for field in self.PREDICTION_FIELDS:
            if field in last_json:
                value = last_json[field]
                if isinstance(value, str):
                    return value
                elif isinstance(value, (int, float, bool)):
                    return str(value)
        
        # If no known field, look for any string or numeric value
        for key, value in last_json.items():
            if isinstance(value, str):
                return value
            elif isinstance(value, (int, float)):
                return str(value)
        
        # Final fallback: if JSON is empty or has no extractable values, return "None"
        self.log_fn(f"Could not extract prediction from JSON: {last_json}")
        return "None"

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

GRADING RUBRIC:
- "Correct" or "1": The student's answer is fully correct, complete, and follows proper mathematical reasoning
- "Incorrect" or "0": The student's answer is wrong, incomplete, or contains critical errors
- "Partial": The student made progress but didn't fully solve the problem (use only when guidelines allow)

IMPORTANT: Your response must be valid JSON wrapped in <json> tags. Do not include any text outside the JSON tags.

Respond in this exact format:
<json>
{{
    "reasoning": "Your detailed chain-of-thought analysis here. Explain your evaluation process clearly.",
    "response": "The final grade/prediction (e.g., 'Correct', 'Incorrect', 'Partial', or a numeric score)"
}}
</json>"""

        # Try up to 3 times to get a valid JSON response
        max_retries = 3
        prediction = "None"
        msg_history = []
        current_instruction = instruction
        
        for attempt in range(max_retries):
            response, msg_history, info = get_response_from_llm(
                msg=current_instruction,
                model=self.model,
                msg_history=msg_history,
            )

            # Extract prediction from the last assistant message
            try:
                if msg_history and len(msg_history) > 0:
                    last_message = msg_history[-1].get("text", "")
                    prediction = self._extract_prediction(last_message)
                    
                    # If we got a valid prediction (not "None"), break the retry loop
                    if prediction != "None":
                        break
                    
                    # If this wasn't the last attempt, add feedback for retry
                    if attempt < max_retries - 1:
                        self.log_fn(f"Attempt {attempt + 1}: Failed to extract valid JSON, retrying...")
                        current_instruction = "NOTE: Your previous response did not contain valid JSON in <json> tags. Please ensure your response follows the exact format specified above with proper JSON syntax."
            except Exception as e:
                self.log_fn(f"Error extracting prediction on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    prediction = "None"

        return str(prediction), msg_history
