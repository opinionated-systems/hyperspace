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
        # Fix 2: Fix single quotes to double quotes (simple cases)
        lambda t: re.sub(r"(?<!\\)'([^']*?)'(?=\s*[:}\],])", r'"\1"', t),
        # Fix 3: Remove comments
        lambda t: re.sub(r'//.*?\n|/\*.*?\*/', '', t, flags=re.DOTALL),
    ]
    
    for i, fix in enumerate(fixes[:max_retries]):
        try:
            fixed_text = fix(text)
            result = _extract_jsons(fixed_text)
            if result:
                logger.debug(f"JSON extraction succeeded after fix attempt {i + 1}")
                return result
        except Exception:
            continue
    
    # Final attempt: try to extract any JSON-like structure
    try:
        # Look for content between braces
        match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
        if match:
            return [json.loads(match.group(0))]
    except Exception:
        pass
    
    return None


def _validate_inputs(inputs: dict) -> tuple[bool, str]:
    """Validate that inputs contains required fields for IMO grading.
    
    Returns:
        (is_valid, error_message)
    """
    required_fields = ["domain", "problem", "solution", "grading_guidelines", "student_answer"]
    missing = [f for f in required_fields if f not in inputs]
    if missing:
        return False, f"Missing required fields: {missing}"
    return True, ""


def _format_grading_prompt(inputs: dict) -> str:
    """Format a structured prompt for IMO grading tasks."""
    domain = inputs.get("domain", "Unknown")
    problem = inputs.get("problem", "")
    solution = inputs.get("solution", "")
    guidelines = inputs.get("grading_guidelines", "")
    student_answer = inputs.get("student_answer", "")
    
    return f"""You are an expert mathematics grader for International Mathematical Olympiad (IMO) problems.

Your task is to evaluate a student's answer to a mathematics problem and provide a grade.

PROBLEM DOMAIN: {domain}

PROBLEM STATEMENT:
{problem}

OFFICIAL SOLUTION:
{solution}

GRADING GUIDELINES:
{guidelines}

STUDENT'S ANSWER TO EVALUATE:
{student_answer}

INSTRUCTIONS:
1. Carefully read the problem, official solution, and grading guidelines
2. Analyze the student's answer for correctness and completeness
3. Compare the student's approach with the official solution
4. Assign an appropriate grade based on the grading guidelines
5. Provide your grade in the JSON format below

Respond in JSON format with the following schema:
<json>
{{
    "response": "Your grade here (e.g., '7', '6', '5', '0', 'Partial credit', etc.)"
}}
</json>

The response field should contain only the grade value as specified in the grading guidelines."""


class TaskAgent:
    """Task agent that solves IMO grading problems."""

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
        # Validate inputs
        is_valid, error_msg = _validate_inputs(inputs)
        if not is_valid:
            logger.warning(f"Input validation failed: {error_msg}")
            # Still proceed with a basic prompt, but log the issue
        
        # Use structured prompt for IMO grading
        instruction = _format_grading_prompt(inputs)

        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return "Error: LLM call failed", [{"role": "system", "text": f"Error: {e}"}]

        # Extract prediction from JSON with retry mechanism
        prediction = "None"
        try:
            if msg_history and len(msg_history) > 0:
                last_msg = msg_history[-1]
                text_content = last_msg.get("text", "")
                extracted = _extract_json_with_retry(text_content)
                if extracted and len(extracted) > 0:
                    last_extracted = extracted[-1]
                    if isinstance(last_extracted, dict) and "response" in last_extracted:
                        prediction = last_extracted["response"]
                    else:
                        # Try to find any grade-like value in the response
                        prediction = str(last_extracted)
                else:
                    # Fallback: try to extract grade directly from text
                    grade_match = re.search(r'\b([0-7]|Partial credit|Full credit|No credit)\b', text_content, re.IGNORECASE)
                    if grade_match:
                        prediction = grade_match.group(1)
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
