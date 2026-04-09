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
    Includes additional heuristics for handling common LLM output issues.
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
        
        # Try to parse the JSON
        try:
            results.append(json.loads(inner))
        except json.JSONDecodeError as e:
            # Attempt to fix common JSON issues
            fixed_inner = _attempt_json_fix(inner)
            if fixed_inner:
                try:
                    results.append(json.loads(fixed_inner))
                    logger.debug(f"Successfully parsed JSON after fixing: {e}")
                except json.JSONDecodeError:
                    logger.debug(f"Failed to parse JSON from <json> block: {e}")
                    continue
            else:
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
                # Try fixing common issues
                fixed = _attempt_json_fix(match.strip())
                if fixed:
                    try:
                        results.append(json.loads(fixed))
                    except json.JSONDecodeError:
                        continue
                else:
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


def _attempt_json_fix(text: str) -> str | None:
    """Attempt to fix common JSON formatting issues from LLM outputs.
    
    Returns fixed string if successful, None if cannot fix.
    """
    if not text:
        return None
    
    # Remove leading/trailing whitespace and newlines
    text = text.strip()
    
    # Fix trailing commas before closing braces/brackets
    import re
    text = re.sub(r',(\s*[}\]])', r'\1', text)
    
    # Fix single quotes to double quotes (common LLM mistake)
    # Only replace if the text uses single quotes as JSON delimiters
    if '"' not in text and "'" in text:
        # Simple replacement - may not handle all edge cases
        text = text.replace("'", '"')
    
    # Ensure the text starts with { and ends with }
    if not text.startswith('{'):
        # Try to find the first {
        start_idx = text.find('{')
        if start_idx != -1:
            text = text[start_idx:]
    
    if not text.endswith('}'):
        # Try to find the last }
        end_idx = text.rfind('}')
        if end_idx != -1:
            text = text[:end_idx+1]
    
    # Validate that we have a complete object
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        return None


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
    
    response_text = response["response"].strip()
    if len(response_text) == 0:
        return False, "'response' value is empty"
    
    # Check for minimum meaningful content (at least 20 characters)
    if len(response_text) < 20:
        return False, "'response' value is too short to be a meaningful grading evaluation"
    
    # Check that the response contains some grading-related keywords
    grading_keywords = [
        "correct", "incorrect", "partial", "score", "point", "mark",
        "solution", "answer", "proof", "error", "valid", "invalid",
        "rigorous", "complete", "incomplete", "good", "bad", "issue",
        "problem", "student", "evaluation", "assessment", "grade"
    ]
    response_lower = response_text.lower()
    has_grading_content = any(keyword in response_lower for keyword in grading_keywords)
    
    if not has_grading_content:
        return False, "'response' does not appear to contain grading evaluation content"
    
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
        
        return f"""You are an expert IMO (International Mathematical Olympiad) grader. Your task is to evaluate a student's solution to a mathematical problem with precision and consistency.

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
Carefully evaluate the student's answer against the official solution and grading guidelines. Consider these criteria in order of importance:

1. **Mathematical Correctness**: Are all mathematical statements, calculations, and proofs correct?
2. **Logical Structure**: Is the reasoning clear, well-organized, and logically sound?
3. **Completeness**: Did the student address all parts of the problem? Are there any gaps?
4. **Rigor**: Are proofs sufficiently rigorous for an IMO-level solution?
5. **Novelty**: Does the student present an innovative approach that differs from but is equivalent to the official solution?

## Scoring Guidelines
- If the solution is completely correct and rigorous: Award full marks with clear justification.
- If the solution has minor flaws but the core idea is correct: Award partial credit and explain what was missed.
- If the solution is partially correct but incomplete: Award proportional credit based on what was proven.
- If the solution is fundamentally incorrect: Explain the error clearly and award minimal or no credit.

## Response Format (CRITICAL)
You MUST provide your evaluation as a valid JSON object wrapped in <json>...</json> tags. The JSON must contain exactly one key:

<json>
{{
    "response": "Your detailed grading feedback here. Structure your response as follows:\n\n1. Overall Assessment: State whether the solution is correct/partially correct/incorrect.\n2. Score: Provide a numerical score if applicable based on the grading guidelines.\n3. Detailed Analysis: Explain specific points where the student succeeded or failed.\n4. Suggestions: If applicable, suggest how the solution could be improved."
}}
</json>

IMPORTANT: 
- Your response MUST be valid JSON wrapped in <json>...</json> tags.
- The "response" field must contain your complete grading evaluation as a single string.
- Do not include any text outside the JSON tags.
- Ensure all quotes within the response are properly escaped."""

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

        # Validate required inputs
        if not problem:
            logger.warning("No problem statement provided in inputs")
        if not student_answer:
            logger.warning("No student answer provided in inputs")
            return "[ERROR] No student answer provided", []

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

            self.log_fn(f"Sending grading request to LLM (attempt {attempt + 1}/{self.max_retries + 1})")
            
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=msg_history if attempt > 0 else [],
                )
            except Exception as e:
                logger.error(f"LLM call failed on attempt {attempt + 1}: {e}")
                previous_error = f"LLM call failed: {str(e)}"
                if attempt < self.max_retries:
                    self.log_fn("Retrying after LLM failure...")
                continue

            # Extract prediction from JSON
            try:
                # Get the last assistant message
                last_message = None
                for msg in reversed(msg_history):
                    if msg.get("role") == "assistant":
                        last_message = msg.get("text", "")
                        break
                
                if not last_message:
                    previous_error = "No assistant response found in message history"
                    self.log_fn(f"No assistant response on attempt {attempt + 1}")
                    if attempt < self.max_retries:
                        self.log_fn("Retrying...")
                    continue
                
                extracted = _extract_jsons(last_message)
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
            last_response = ""
            for msg in reversed(msg_history):
                if msg.get("role") == "assistant":
                    last_response = msg.get("text", "")
                    break
            
            if last_response and len(last_response.strip()) > 0:
                # Try to extract any meaningful text even without JSON
                prediction = f"[UNPARSED] {last_response[:1000]}"
                self.log_fn("Using raw response as fallback (unparsed)")
            else:
                prediction = "[ERROR] Failed to generate any valid grading response after all retries"
                self.log_fn("All attempts failed, no valid response available")

        return str(prediction), msg_history
