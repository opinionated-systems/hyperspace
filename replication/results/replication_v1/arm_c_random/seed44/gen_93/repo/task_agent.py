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
    Also attempts to parse raw JSON objects if no <json> tags are found.
    Includes improved handling for nested braces, malformed JSON, and common LLM output patterns.
    
    Args:
        text: The text to extract JSON from.
        
    Returns:
        A list of parsed JSON objects, or None if no valid JSON found.
    """
    results = []
    search_from = 0
    extraction_errors = []
    
    # Primary: Extract from <json>...</json> tags
    while True:
        start = text.find("<json>", search_from)
        if start == -1:
            break
        end = text.find("</json>", start)
        if end == -1:
            extraction_errors.append(f"Unclosed <json> tag at position {start}")
            break
        inner = text[start + 6:end].strip()
        search_from = end + 7
        try:
            results.append(json.loads(inner))
        except json.JSONDecodeError as e:
            extraction_errors.append(f"JSON parse error at position {start}: {e}")
            continue
    
    # Fallback 1: Try to find JSON objects directly if no <json> tags
    if not results:
        try:
            # Look for JSON-like structures with braces
            brace_start = text.find("{")
            brace_end = text.rfind("}")
            if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
                potential_json = text[brace_start:brace_end + 1]
                results.append(json.loads(potential_json))
        except json.JSONDecodeError as e:
            extraction_errors.append(f"Fallback JSON parse error: {e}")
    
    # Fallback 2: Try to extract JSON from markdown code blocks
    if not results:
        json_code_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
        matches = re.findall(json_code_block_pattern, text, re.DOTALL)
        for match in matches:
            try:
                results.append(json.loads(match.strip()))
            except json.JSONDecodeError as e:
                extraction_errors.append(f"Code block JSON parse error: {e}")
    
    # Fallback 3: Try to find JSON arrays as well
    if not results:
        try:
            bracket_start = text.find("[")
            bracket_end = text.rfind("]")
            if bracket_start != -1 and bracket_end != -1 and bracket_end > bracket_start:
                potential_json = text[bracket_start:bracket_end + 1]
                results.append(json.loads(potential_json))
        except json.JSONDecodeError as e:
            extraction_errors.append(f"Array fallback JSON parse error: {e}")
    
    # Fallback 4: Try to fix common JSON issues and re-parse
    if not results:
        try:
            # Look for JSON with common issues like trailing commas
            brace_start = text.find("{")
            brace_end = text.rfind("}")
            if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
                potential_json = text[brace_start:brace_end + 1]
                # Fix common issues
                # Remove trailing commas before closing braces/brackets
                fixed = re.sub(r',(\s*[}\]])', r'\1', potential_json)
                # Fix single quotes to double quotes (simple cases)
                fixed = re.sub(r"'([^']*?)':", r'"\1":', fixed)
                results.append(json.loads(fixed))
        except (json.JSONDecodeError, re.error) as e:
            extraction_errors.append(f"Fixed JSON parse error: {e}")
    
    # Fallback 5: Try to extract from <answer>...</answer> or similar tags
    if not results:
        answer_pattern = r'<answer>\s*(.*?)\s*</answer>'
        matches = re.findall(answer_pattern, text, re.DOTALL | re.IGNORECASE)
        for match in matches:
            try:
                # Try to parse as JSON first
                results.append(json.loads(match.strip()))
            except json.JSONDecodeError:
                # If not JSON, wrap in a response object
                results.append({"response": match.strip()})
    
    # Log extraction details for debugging
    if extraction_errors and logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"JSON extraction errors: {extraction_errors}")
    
    return results or None


def _format_inputs(inputs: dict) -> str:
    """Format task inputs into a structured prompt.
    
    Provides better structure and context for the LLM.
    Handles edge cases like empty inputs and non-string values.
    Organizes inputs by category for better comprehension.
    """
    if not inputs:
        return "No inputs provided."
    
    # Define priority order for better context understanding
    priority_keys = ["domain", "problem", "solution", "grading_guidelines", "student_answer"]
    
    parts = []
    processed = set()
    
    # First, add priority keys in order
    for key in priority_keys:
        if key in inputs:
            value = inputs[key]
            if not isinstance(value, str):
                value = str(value)
            parts.append(f"{'='*60}")
            parts.append(f"{key.upper().replace('_', ' ')}")
            parts.append(f"{'='*60}")
            if value.strip():
                parts.append(value)
            else:
                parts.append("(empty)")
            parts.append("")
            processed.add(key)
    
    # Then add any remaining keys
    for key, value in inputs.items():
        if key not in processed:
            if not isinstance(value, str):
                value = str(value)
            parts.append(f"{'='*60}")
            parts.append(f"{key.upper().replace('_', ' ')}")
            parts.append(f"{'='*60}")
            if value.strip():
                parts.append(value)
            else:
                parts.append("(empty)")
            parts.append("")
    
    return "\n".join(parts) if parts else "No valid inputs provided."


def _validate_response_schema(response: dict) -> tuple[bool, str]:
    """Validate that the response follows the expected schema.
    
    Returns:
        (is_valid, error_message)
    """
    if not isinstance(response, dict):
        return False, f"Response is not a dict, got {type(response).__name__}"
    
    if "response" not in response:
        return False, "Response missing required 'response' key"
    
    if not isinstance(response["response"], str):
        return False, f"'response' value is not a string, got {type(response['response']).__name__}"
    
    if len(response["response"].strip()) == 0:
        return False, "'response' value is empty"
    
    return True, ""


class TaskAgent:
    """Task agent that solves IMO grading problems with improved error handling."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "", max_retries: int = 2) -> None:
        self.model = model
        self.log_fn = logger.info
        self.call_count = 0
        self.max_retries = max_retries

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        self.call_count += 1
        self.log_fn(f"TaskAgent call #{self.call_count} starting")
        
        # Format inputs more clearly
        formatted_inputs = _format_inputs(inputs)
        
        instruction = f"""You are an expert grading agent for mathematical and scientific problems. Your task is to evaluate student answers with precision and consistency.

{formatted_inputs}

## GRADING INSTRUCTIONS

1. **Understand the Problem**: Read the problem statement carefully to understand what is being asked.

2. **Review the Solution**: Study the provided solution to understand the correct approach and expected answer.

3. **Analyze the Student's Answer**: Compare the student's answer against the solution, considering:
   - Is the final answer correct?
   - Is the reasoning sound and logical?
   - Are there any partial credits to be awarded?
   - Did the student follow the grading guidelines?

4. **Apply Grading Guidelines**: Use the provided grading guidelines to determine the appropriate evaluation.

5. **Provide Clear Evaluation**: Your response should clearly state whether the answer is correct, partially correct, or incorrect, with brief justification.

## RESPONSE FORMAT

Respond in JSON format with the following schema:
<json>
{{
    "response": "Your detailed evaluation here. Include: (1) correctness assessment, (2) brief reasoning, (3) final verdict"
}}
</json>

IMPORTANT:
- Ensure your response is valid JSON
- The "response" field must contain a non-empty string
- Be objective and consistent in your grading
- If the answer is partially correct, explain what was right and what was wrong"""

        # Retry loop with validation
        for attempt in range(self.max_retries + 1):
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=[],
                )
                self.log_fn(f"LLM call completed (attempt {attempt + 1}), response length: {len(response)}")
            except Exception as e:
                self.log_fn(f"Error in LLM call (attempt {attempt + 1}): {e}")
                if attempt == self.max_retries:
                    return "Error: LLM call failed", []
                continue

            # Extract prediction from JSON with better error handling
            prediction = "None"
            validation_passed = False
            
            try:
                if msg_history and len(msg_history) > 0:
                    last_message = msg_history[-1]
                    text_content = last_message.get("text", "")
                    extracted = _extract_jsons(text_content)
                    if extracted:
                        last_extracted = extracted[-1]
                        is_valid, error_msg = _validate_response_schema(last_extracted)
                        if is_valid:
                            prediction = last_extracted["response"]
                            validation_passed = True
                            self.log_fn(f"Successfully extracted and validated prediction: {str(prediction)[:100]}")
                        else:
                            self.log_fn(f"Response validation failed: {error_msg}")
                            if attempt < self.max_retries:
                                # Add validation feedback to the instruction for retry
                                instruction += f"\n\nPrevious response was invalid: {error_msg}. Please fix and respond with valid JSON."
                    else:
                        self.log_fn("No JSON found in response")
                        if attempt < self.max_retries:
                            instruction += "\n\nPrevious response did not contain valid JSON. Please wrap your response in <json>...</json> tags."
                else:
                    self.log_fn("Empty message history")
            except Exception as e:
                self.log_fn(f"Error extracting prediction (attempt {attempt + 1}): {e}")
            
            if validation_passed:
                return str(prediction), msg_history
            
            if attempt == self.max_retries:
                self.log_fn(f"Max retries ({self.max_retries}) reached, returning best effort result")
                return str(prediction), msg_history
        
        return "Error: Unexpected end of forward method", []
