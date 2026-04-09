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
    Also attempts to extract JSON from markdown code blocks as fallback.
    """
    results = []
    search_from = 0
    
    # Primary: Extract from <json> tags
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
        except json.JSONDecodeError as e:
            logger.debug(f"Failed to parse JSON from <json> block: {e}")
            continue
    
    # Fallback: Extract from markdown code blocks
    if not results:
        pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                results.append(json.loads(match.strip()))
            except json.JSONDecodeError:
                continue
    
    # Final fallback: Try to find any JSON-like structure with "response" key
    if not results:
        try:
            # Look for patterns like {"response": "..."}
            pattern = r'\{\s*"response"\s*:\s*"([^"]*)"\s*\}'
            match = re.search(pattern, text)
            if match:
                results.append({"response": match.group(1)})
        except Exception:
            pass
    
    return results or None


def _validate_grading_response(response: dict) -> tuple[bool, str]:
    """Validate that the grading response has the required structure.
    
    Returns:
        (is_valid, error_message)
    """
    if not isinstance(response, dict):
        return False, "Response is not a dictionary"
    
    if "response" not in response:
        return False, "Missing 'response' key in output"
    
    if not isinstance(response["response"], str):
        return False, "'response' value must be a string"
    
    if len(response["response"].strip()) == 0:
        return False, "'response' value is empty"
    
    return True, ""


class TaskAgent:
    """Task agent that solves IMO grading problems."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.max_retries = 2  # Number of retries for invalid responses

    def _build_grading_prompt(
        self,
        domain: str,
        problem: str,
        solution: str,
        grading_guidelines: str,
        student_answer: str,
        previous_error: str = "",
    ) -> str:
        """Build the grading prompt with optional error feedback for retries."""
        error_feedback = ""
        if previous_error:
            error_feedback = f"""
## Previous Attempt Error
The previous response was invalid: {previous_error}
Please ensure your response follows the exact JSON format specified below.
"""
        
        return f"""You are an expert IMO (International Mathematical Olympiad) grader. Your task is to evaluate a student's solution to a mathematical problem.

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
{error_feedback}
## Your Task
Carefully evaluate the student's answer against the official solution and grading guidelines. Consider:
1. Mathematical correctness and rigor
2. Completeness of the solution
3. Logical reasoning and proof structure
4. Whether the student addressed all parts of the problem

Provide your evaluation as a JSON object with the following schema:
<json>
{{
    "response": "Your detailed grading feedback here. Include: (1) whether the solution is correct/partially correct/incorrect, (2) specific points where the student succeeded or failed, (3) a numerical score if applicable based on the grading guidelines."
}}
</json>

IMPORTANT: Your response MUST be valid JSON wrapped in <json>...</json> tags. The "response" field must contain your complete grading evaluation."""

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Extract structured fields for better prompting
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        msg_history = []
        prediction = "None"
        previous_error = ""
        
        for attempt in range(self.max_retries + 1):
            instruction = self._build_grading_prompt(
                domain=domain,
                problem=problem,
                solution=solution,
                grading_guidelines=grading_guidelines,
                student_answer=student_answer,
                previous_error=previous_error,
            )

            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=msg_history if attempt > 0 else [],
            )

            # Extract prediction from JSON
            try:
                extracted = _extract_jsons(msg_history[-1]["text"])
                if extracted:
                    # Validate the response structure
                    is_valid, error_msg = _validate_grading_response(extracted[-1])
                    if is_valid:
                        prediction = extracted[-1]["response"]
                        self.log_fn(f"Successfully extracted valid grading response on attempt {attempt + 1}")
                        break
                    else:
                        previous_error = error_msg
                        self.log_fn(f"Invalid response structure on attempt {attempt + 1}: {error_msg}")
                        if attempt < self.max_retries:
                            self.log_fn("Retrying with error feedback...")
                else:
                    previous_error = "No JSON found in response"
                    self.log_fn(f"No JSON extracted on attempt {attempt + 1}")
                    if attempt < self.max_retries:
                        self.log_fn("Retrying with error feedback...")
            except Exception as e:
                previous_error = f"Extraction error: {str(e)}"
                self.log_fn(f"Error extracting prediction on attempt {attempt + 1}: {e}")
                if attempt < self.max_retries:
                    self.log_fn("Retrying with error feedback...")
        
        # If all retries failed, use the last response text as fallback
        if prediction == "None" and msg_history:
            # Try to use the raw response as a fallback
            last_response = msg_history[-1].get("text", "")
            if last_response and len(last_response.strip()) > 0:
                prediction = f"[UNPARSED] {last_response[:500]}"
                self.log_fn("Using raw response as fallback (unparsed)")

        return str(prediction), msg_history
