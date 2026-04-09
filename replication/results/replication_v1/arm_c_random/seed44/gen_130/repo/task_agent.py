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


def _extract_json_with_retry(text: str, max_retries: int = 3) -> list[dict] | None:
    """Extract JSON with multiple fallback strategies.
    
    First tries standard extraction, then attempts to fix common
    JSON formatting issues like trailing commas, missing quotes, etc.
    """
    # First attempt: standard extraction
    result = _extract_jsons(text)
    if result:
        return result
    
    # Retry with progressively more aggressive fixes
    fixes = [
        # Fix 1: Remove trailing commas
        lambda t: re.sub(r',(\s*[}\]])', r'\1', t),
        # Fix 2: Fix single quotes to double quotes
        lambda t: re.sub(r"'([^']*?)'", r'"\1"', t),
        # Fix 3: Remove comments
        lambda t: re.sub(r'//.*?\n', '\n', t),
    ]
    
    for attempt, fix in enumerate(fixes[:max_retries]):
        try:
            fixed_text = fix(text)
            result = _extract_jsons(fixed_text)
            if result:
                logger.debug(f"JSON extraction succeeded after fix attempt {attempt + 1}")
                return result
        except Exception:
            pass
    
    return None


def _validate_grading_response(response: dict) -> tuple[bool, str]:
    """Validate that the grading response has the expected format.
    
    Returns (is_valid, error_message).
    """
    if not isinstance(response, dict):
        return False, "Response is not a dictionary"
    
    if "response" not in response:
        return False, "Missing 'response' key in JSON"
    
    return True, ""


class TaskAgent:
    """Task agent that solves IMO grading problems with structured reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def _build_grading_prompt(self, inputs: dict) -> str:
        """Build a structured prompt for the IMO grading task.
        
        This prompt encourages step-by-step reasoning before providing
        the final grade, which improves accuracy on complex grading tasks.
        """
        domain = inputs.get("domain", "Unknown")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        return f"""You are an expert grader for {domain} problems. Your task is to evaluate a student's answer against the official solution and grading guidelines.

## Problem Statement
{problem}

## Official Solution
{solution}

## Grading Guidelines
{guidelines}

## Student's Answer
{student_answer}

## Instructions
Please evaluate the student's answer following these steps:

1. **Analyze the Problem**: Understand what the problem is asking and identify key concepts.

2. **Review the Official Solution**: Note the correct approach and expected answer format.

3. **Evaluate the Student's Answer**: 
   - Check if the student understood the problem correctly
   - Verify the mathematical/logical reasoning
   - Check for calculation errors
   - Assess completeness of the solution

4. **Apply Grading Guidelines**: Use the provided guidelines to determine the appropriate grade.

5. **Provide Your Assessment**: Give your final grade based on the evaluation.

Respond in JSON format with the following schema:
<json>
{{
    "reasoning": "Your step-by-step reasoning process...",
    "response": "Your final grade/assessment"
}}
</json>

The "response" field should contain only the final grade (e.g., "Correct", "Incorrect", "Partial", a number, etc. depending on the grading scheme)."""

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

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

        # Extract prediction from JSON with retry mechanism
        prediction = "None"
        try:
            extracted = _extract_json_with_retry(msg_history[-1]["text"])
            if extracted:
                # Validate the response format
                is_valid, error_msg = _validate_grading_response(extracted[-1])
                if is_valid:
                    prediction = extracted[-1]["response"]
                    self.log_fn(f"Successfully extracted prediction: {prediction}")
                else:
                    self.log_fn(f"Invalid response format: {error_msg}")
                    # Try to extract any string that looks like a grade
                    text = msg_history[-1]["text"]
                    # Look for common grade patterns
                    grade_patterns = [
                        r'"response"\s*:\s*"([^"]+)"',
                        r'grade[\s]*[:=][\s]*"?([^"\n]+)"?',
                        r'(Correct|Incorrect|Partial|0|1|2|3|4|5|6|7)',
                    ]
                    for pattern in grade_patterns:
                        match = re.search(pattern, text, re.IGNORECASE)
                        if match:
                            prediction = match.group(1).strip()
                            self.log_fn(f"Extracted grade via pattern matching: {prediction}")
                            break
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
