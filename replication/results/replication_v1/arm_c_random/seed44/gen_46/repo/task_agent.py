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
    Includes robust error recovery for malformed JSON.
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
            # Try to fix common JSON issues
            try:
                # Remove trailing commas before closing braces/brackets
                fixed = re.sub(r',(\s*[}\]])', r'\1', inner)
                # Fix single quotes to double quotes (common LLM mistake)
                fixed = re.sub(r"(?<!\\)'", '"', fixed)
                results.append(json.loads(fixed))
            except json.JSONDecodeError:
                continue
    
    # Fallback: try to find JSON objects directly if no <json> tags
    if not results:
        try:
            # Look for JSON-like structures with braces
            brace_start = text.find("{")
            brace_end = text.rfind("}")
            if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
                potential_json = text[brace_start:brace_end + 1]
                try:
                    results.append(json.loads(potential_json))
                except json.JSONDecodeError:
                    # Try common fixes
                    fixed = re.sub(r',(\s*[}\]])', r'\1', potential_json)
                    fixed = re.sub(r"(?<!\\)'", '"', fixed)
                    results.append(json.loads(fixed))
        except json.JSONDecodeError:
            pass
    
    return results or None


def _format_inputs(inputs: dict) -> str:
    """Format task inputs into a structured prompt.
    
    Provides better structure and context for the LLM.
    Organizes inputs by category for clearer reasoning.
    Handles missing or empty fields gracefully.
    """
    # Define the order and grouping of inputs for better context
    input_order = [
        ("domain", "Domain"),
        ("problem", "Problem Statement"),
        ("solution", "Reference Solution"),
        ("grading_guidelines", "Grading Guidelines"),
        ("student_answer", "Student Answer"),
    ]
    
    parts = []
    for key, label in input_order:
        if key in inputs and inputs[key]:
            value = str(inputs[key]).strip()
            if value and value.lower() not in ('none', 'null', 'n/a', ''):
                parts.append(f"{'='*60}")
                parts.append(f"{label}:")
                parts.append(f"{'='*60}")
                parts.append(f"{value}")
                parts.append("")
    
    # Add any remaining inputs not in the standard order
    processed_keys = {k for k, _ in input_order}
    for key, value in inputs.items():
        if key not in processed_keys and value:
            val_str = str(value).strip()
            if val_str and val_str.lower() not in ('none', 'null', 'n/a', ''):
                parts.append(f"{'='*60}")
                parts.append(f"{key}:")
                parts.append(f"{'='*60}")
                parts.append(f"{val_str}")
                parts.append("")
    
    return "\n".join(parts) if parts else "[No input data provided]"


def _validate_response_schema(response: dict) -> tuple[bool, str]:
    """Validate that the response follows the expected schema.
    
    Returns:
        (is_valid, error_message)
    """
    if not isinstance(response, dict):
        return False, f"Response is not a dict, got {type(response).__name__}"
    
    if "response" not in response:
        # Check for common misspellings or variations
        for key in response.keys():
            if key.lower() in ("response", "answer", "evaluation", "result"):
                return False, f"Response has key '{key}' but expected 'response' (case-sensitive)"
        return False, "Response missing required 'response' key"
    
    if not isinstance(response["response"], str):
        # Try to convert non-string values to string
        if isinstance(response["response"], (int, float, bool)):
            return True, ""  # Allow numeric/boolean values, will be converted
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
        
        instruction = f"""You are an expert grading agent for mathematical and scientific problems. Your task is to evaluate student answers based on the provided problem, solution, and grading guidelines.

{formatted_inputs}

INSTRUCTIONS FOR GRADING:
1. Carefully read the Problem Statement and Reference Solution
2. Review the Grading Guidelines to understand the evaluation criteria
3. Analyze the Student Answer against the Reference Solution step by step
4. Provide a detailed evaluation that includes:
   - What the student did correctly (specific steps, methods, or insights)
   - Any errors, misconceptions, or missing steps
   - Assessment of mathematical/scientific rigor and notation
   - Comparison to the reference solution's approach
   - Final grading decision with clear justification

GRADING PRINCIPLES:
- Be precise and specific in your evaluation
- Consider partial credit for correct approaches with minor errors
- Distinguish between conceptual misunderstandings and calculation errors
- Note any alternative valid approaches that differ from the reference

Respond ONLY in JSON format with the following schema:
<json>
{{
    "response": "Your detailed evaluation and grading decision here. Include: (1) Summary of student's approach, (2) Correct aspects identified, (3) Errors or gaps found, (4) Final assessment with justification."
}}
</json>

IMPORTANT: 
- Ensure your response is valid JSON with proper escaping
- The "response" field must contain your complete evaluation as a single string
- Do not include any text outside the <json>...</json> tags"""

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
                            self.log_fn(f"Successfully extracted and validated prediction: {str(prediction)[:100]}...")
                        else:
                            self.log_fn(f"Response validation failed: {error_msg}")
                            if attempt < self.max_retries:
                                # Add validation feedback to the instruction for retry
                                feedback = f"""

VALIDATION ERROR (attempt {attempt + 1}): {error_msg}

Your previous response was invalid. Please fix the following issues and respond again:
- Ensure your response is wrapped in <json>...</json> tags
- The JSON must have a single "response" field containing a string
- Check for proper JSON syntax (no trailing commas, proper quotes)

Respond with valid JSON only:"""
                                instruction = instruction + feedback
                    else:
                        self.log_fn("No JSON found in response")
                        if attempt < self.max_retries:
                            feedback = f"""

VALIDATION ERROR (attempt {attempt + 1}): No valid JSON found in your response.

Your previous response did not contain properly formatted JSON. Please:
- Wrap your entire response in <json>...</json> tags
- Ensure the content inside is valid JSON with a "response" field
- Example: <json>{{"response": "your evaluation here"}}</json>

Respond with valid JSON only:"""
                            instruction = instruction + feedback
                else:
                    self.log_fn("Empty message history")
            except Exception as e:
                self.log_fn(f"Error extracting prediction (attempt {attempt + 1}): {e}")
            
            if validation_passed:
                return str(prediction), msg_history
            
            if attempt == self.max_retries:
                self.log_fn(f"Max retries ({self.max_retries}) reached, returning best effort result")
                # Try to extract any meaningful content even if JSON parsing failed
                if prediction == "None" and msg_history and len(msg_history) > 0:
                    last_text = msg_history[-1].get("text", "")
                    if last_text and len(last_text.strip()) > 0:
                        # Return the raw text as fallback
                        return f"[Unstructured Response] {last_text[:500]}", msg_history
                return str(prediction), msg_history
        
        return "Error: Unexpected end of forward method", []
