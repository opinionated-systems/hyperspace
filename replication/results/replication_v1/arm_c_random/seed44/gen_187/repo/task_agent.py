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

# Expected input fields for IMO grading task
_REQUIRED_INPUT_FIELDS = {"domain", "problem", "solution", "grading_guidelines", "student_answer"}
_OPTIONAL_INPUT_FIELDS = {"rubric", "max_score"}


def _validate_inputs(inputs: dict) -> tuple[bool, str]:
    """Validate that inputs contains required fields for IMO grading.
    
    Returns:
        (is_valid, error_message)
    """
    if not isinstance(inputs, dict):
        return False, f"Expected dict, got {type(inputs).__name__}"
    
    missing = _REQUIRED_INPUT_FIELDS - set(inputs.keys())
    if missing:
        return False, f"Missing required fields: {missing}"
    
    # Check for empty values
    empty_fields = [k for k in _REQUIRED_INPUT_FIELDS if not str(inputs.get(k, "")).strip()]
    if empty_fields:
        return False, f"Empty values for fields: {empty_fields}"
    
    return True, ""


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


def _extract_json_with_retry(text: str, max_retries: int = 2) -> list[dict] | None:
    """Extract JSON with multiple fallback strategies.
    
    First tries standard extraction, then attempts to fix common
    JSON formatting issues like trailing commas.
    """
    # First attempt: standard extraction
    result = _extract_jsons(text)
    if result:
        return result
    
    # Retry with common fixes
    for attempt in range(max_retries):
        try:
            # Try to find and fix JSON with trailing commas
            fixed_text = re.sub(r',(\s*[}\]])', r'\1', text)
            result = _extract_jsons(fixed_text)
            if result:
                logger.debug(f"JSON extraction succeeded after fix attempt {attempt + 1}")
                return result
        except Exception:
            pass
    
    return None


def _format_grading_prompt(inputs: dict) -> str:
    """Format the grading prompt with structured sections."""
    domain = inputs.get("domain", "Mathematics")
    problem = inputs.get("problem", "")
    solution = inputs.get("solution", "")
    guidelines = inputs.get("grading_guidelines", "")
    student_answer = inputs.get("student_answer", "")
    rubric = inputs.get("rubric", "")
    max_score = inputs.get("max_score", "")
    
    prompt_parts = [
        "You are an expert grader for mathematics competitions.",
        "",
        f"Domain: {domain}",
        "",
        "=== PROBLEM ===",
        problem,
        "",
        "=== OFFICIAL SOLUTION ===",
        solution,
        "",
        "=== GRADING GUIDELINES ===",
        guidelines,
    ]
    
    if rubric:
        prompt_parts.extend(["", "=== RUBRIC ===", rubric])
    
    if max_score:
        prompt_parts.extend(["", f"=== MAXIMUM SCORE ===", str(max_score)])
    
    prompt_parts.extend([
        "",
        "=== STUDENT ANSWER TO GRADE ===",
        student_answer,
        "",
        "Provide your grading assessment in JSON format with the following structure:",
        "<json>",
        json.dumps({
            "response": "Your detailed grading assessment here",
            "score": "Numerical score (e.g., 7, 3.5, 0)",
            "max_score": str(max_score) if max_score else "7",
            "reasoning": "Brief explanation of scoring rationale"
        }, indent=2),
        "</json>",
        "",
        "Important: Include both the detailed assessment in 'response' and a numerical 'score' field.",
    ])
    
    return "\n".join(prompt_parts)


class TaskAgent:
    """Task agent that solves IMO grading problems with enhanced validation."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self._call_count = 0

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        self._call_count += 1
        
        # Validate inputs
        is_valid, error_msg = _validate_inputs(inputs)
        if not is_valid:
            logger.error(f"Input validation failed: {error_msg}")
            return f"Error: {error_msg}", []
        
        # Log task info
        domain = inputs.get("domain", "Unknown")
        problem_preview = str(inputs.get("problem", ""))[:100]
        logger.info(f"Task #{self._call_count}: domain={domain}, problem_preview={problem_preview!r}")
        
        # Build structured instruction
        instruction = _format_grading_prompt(inputs)

        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return f"Error: LLM call failed - {e}", []

        # Extract prediction from JSON with retry mechanism
        prediction = "None"
        extraction_method = "none"
        score = None
        
        try:
            if msg_history and len(msg_history) > 0:
                last_msg = msg_history[-1]
                text_content = last_msg.get("text", "") if isinstance(last_msg, dict) else str(last_msg)
                
                extracted = _extract_json_with_retry(text_content)
                if extracted:
                    extraction_method = "json_block"
                    last_json = extracted[-1]
                    if isinstance(last_json, dict):
                        # Try to get structured response with score
                        if "response" in last_json:
                            prediction = last_json["response"]
                            # Also extract score if available
                            score = last_json.get("score")
                            if score:
                                prediction = f"[Score: {score}] {prediction}"
                        elif "score" in last_json:
                            # If only score is present, use it with reasoning
                            score = last_json.get("score")
                            reasoning = last_json.get("reasoning", "No reasoning provided")
                            prediction = f"[Score: {score}] {reasoning}"
                        else:
                            prediction = json.dumps(last_json)
                            extraction_method = "json_fallback"
                    else:
                        prediction = json.dumps(last_json)
                        extraction_method = "json_fallback"
                else:
                    # Fallback: use raw text if no JSON found
                    prediction = text_content[:500]  # Truncate long responses
                    extraction_method = "raw_text"
            
            logger.info(f"Extraction method: {extraction_method}, prediction length: {len(str(prediction))}, score: {score}")
            
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            prediction = f"Error: {e}"

        return str(prediction), msg_history
