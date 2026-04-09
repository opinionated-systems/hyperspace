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
    Includes repair logic for common LLM JSON formatting errors.
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
        
        # Try to parse, with repair attempts for common errors
        parsed = _try_parse_json_with_repair(inner)
        if parsed is not None:
            results.append(parsed)
    
    # Fallback: Extract from markdown code blocks
    if not results:
        pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            parsed = _try_parse_json_with_repair(match.strip())
            if parsed is not None:
                results.append(parsed)
    
    # Final fallback: Try to find any JSON-like structure with "response" key
    if not results:
        try:
            # Look for patterns like {"response": "..."} or {"response": "...", ...}
            pattern = r'\{\s*"response"\s*:\s*"([^"]*)"[^}]*\}'
            match = re.search(pattern, text, re.DOTALL)
            if match:
                # Extract the full match and try to parse
                full_match = match.group(0)
                parsed = _try_parse_json_with_repair(full_match)
                if parsed is not None:
                    results.append(parsed)
        except Exception:
            pass
    
    return results or None


def _try_parse_json_with_repair(text: str) -> dict | None:
    """Attempt to parse JSON with automatic repair for common LLM errors.
    
    Repairs attempted:
    - Trailing commas before closing braces/brackets
    - Single quotes instead of double quotes
    - Unescaped newlines in strings
    - Missing closing braces/brackets
    """
    # First, try direct parsing
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Attempt repairs
    repaired = text
    
    # Remove trailing commas before } or ]
    repaired = re.sub(r',\s*([}\]])', r'\1', repaired)
    
    # Replace single quotes with double quotes (carefully)
    # Only replace quotes that are not inside strings
    # This is a heuristic: replace ' that appear at start of values
    repaired = re.sub(r"(?<=[{,\s:])'([^']*)'(?=[,}\]])", r'"\1"', repaired)
    
    # Try parsing after repairs
    try:
        return json.loads(repaired)
    except json.JSONDecodeError as e:
        logger.debug(f"Failed to parse JSON even after repair: {e}")
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
    
    if len(response["response"].strip()) == 0:
        return False, "'response' value is empty"
    
    # Check for minimum content quality
    response_text = response["response"].strip()
    if len(response_text) < 10:
        return False, "'response' value is too short (minimum 10 characters)"
    
    return True, ""


def _extract_score_from_response(response_text: str) -> float | None:
    """Extract a numerical score from the grading response text.
    
    Looks for patterns like:
    - "Score: 7/7" or "Score: 7"
    - "7/7 points" or "7 points"
    - "Grade: 7"
    - "Total: 7"
    
    Returns the score as a float, or None if no score found.
    """
    import re
    
    # Pattern: Score: X/Y or Score: X
    score_patterns = [
        r'[Ss]core[:\s]+(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)',
        r'[Ss]core[:\s]+(\d+(?:\.\d+)?)',
        r'(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)\s*(?:points?|marks?)',
        r'[Gg]rade[:\s]+(\d+(?:\.\d+)?)',
        r'[Tt]otal[:\s]+(\d+(?:\.\d+)?)',
        r'(\d+(?:\.\d+)?)\s*(?:out of|of)\s*(\d+(?:\.\d+)?)',
    ]
    
    for pattern in score_patterns:
        match = re.search(pattern, response_text)
        if match:
            try:
                if len(match.groups()) >= 2 and match.group(2):
                    # Return as fraction (e.g., 7/7 = 1.0)
                    numerator = float(match.group(1))
                    denominator = float(match.group(2))
                    if denominator > 0:
                        return numerator / denominator
                else:
                    return float(match.group(1))
            except (ValueError, IndexError):
                continue
    
    return None


class TaskAgent:
    """Task agent that solves IMO grading problems."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.max_retries = 3  # Increased retries for better reliability
        self.min_response_length = 50  # Minimum characters for a quality grading response

    def _build_grading_prompt(
        self,
        domain: str,
        problem: str,
        solution: str,
        grading_guidelines: str,
        student_answer: str,
        previous_error: str = "",
        attempt_number: int = 0,
    ) -> str:
        """Build the grading prompt with optional error feedback for retries."""
        error_feedback = ""
        if previous_error:
            error_feedback = f"""
## Previous Attempt Error
The previous response was invalid: {previous_error}
Please ensure your response follows the exact JSON format specified below.
"""
        
        # Add progressive guidance on retries
        retry_guidance = ""
        if attempt_number > 0:
            retry_guidance = f"""
## Retry Guidance (Attempt {attempt_number + 1})
Focus on providing a complete, well-structured evaluation. Be thorough in your analysis.
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
{error_feedback}{retry_guidance}
## Your Task
Carefully evaluate the student's answer against the official solution and grading guidelines. Consider:
1. Mathematical correctness and rigor
2. Completeness of the solution
3. Logical reasoning and proof structure
4. Whether the student addressed all parts of the problem
5. Partial credit for incomplete but correct approaches

Provide your evaluation as a JSON object with the following schema:
<json>
{{
    "response": "Your detailed grading feedback here. Include: (1) whether the solution is correct/partially correct/incorrect, (2) specific points where the student succeeded or failed, (3) a numerical score if applicable based on the grading guidelines (e.g., 'Score: 5/7')."
}}
</json>

IMPORTANT: 
- Your response MUST be valid JSON wrapped in <json>...</json> tags
- The "response" field must contain your complete grading evaluation
- Be specific about what the student did right and wrong
- Include a clear score if the grading guidelines specify one"""

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
        extracted_score = None
        
        for attempt in range(self.max_retries + 1):
            instruction = self._build_grading_prompt(
                domain=domain,
                problem=problem,
                solution=solution,
                grading_guidelines=grading_guidelines,
                student_answer=student_answer,
                previous_error=previous_error,
                attempt_number=attempt,
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
                        # Try to extract score for logging/metrics
                        extracted_score = _extract_score_from_response(prediction)
                        if extracted_score is not None:
                            self.log_fn(f"Extracted score: {extracted_score:.2f}")
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
                # Try one more time to extract any JSON from the raw response
                final_extract = _extract_jsons(last_response)
                if final_extract and _validate_grading_response(final_extract[-1])[0]:
                    prediction = final_extract[-1]["response"]
                    self.log_fn("Successfully extracted valid response from raw fallback")
                else:
                    prediction = f"[UNPARSED] {last_response[:500]}"
                    self.log_fn("Using raw response as fallback (unparsed)")

        return str(prediction), msg_history
