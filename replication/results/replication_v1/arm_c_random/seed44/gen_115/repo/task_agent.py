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
    Includes robust fallback strategies for malformed JSON.
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
            # Try to fix common JSON issues before giving up
            try:
                # Remove trailing commas before closing braces/brackets
                fixed = re.sub(r',(\s*[}\]])', r'\1', inner)
                # Fix single quotes to double quotes (carefully)
                fixed = re.sub(r"(?<!\\)'", '"', fixed)
                results.append(json.loads(fixed))
            except json.JSONDecodeError:
                continue
    
    # Fallback 1: try to find JSON objects directly if no <json> tags
    if not results:
        try:
            # Look for JSON-like structures with braces
            brace_start = text.find("{")
            brace_end = text.rfind("}")
            if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
                potential_json = text[brace_start:brace_end + 1]
                results.append(json.loads(potential_json))
        except json.JSONDecodeError:
            pass
    
    # Fallback 2: try to extract any valid JSON object from the text
    if not results:
        # Use regex to find potential JSON objects
        json_pattern = re.compile(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}')
        for match in json_pattern.finditer(text):
            try:
                potential = match.group()
                parsed = json.loads(potential)
                if isinstance(parsed, dict):
                    results.append(parsed)
            except json.JSONDecodeError:
                continue
    
    return results or None


def _format_inputs(inputs: dict) -> str:
    """Format task inputs into a structured prompt with clear sections.
    
    Provides better structure and context for the LLM with explicit
    section headers for improved clarity.
    """
    sections = []
    
    # Define the order of sections for consistent formatting
    section_order = [
        ("domain", "DOMAIN"),
        ("problem", "PROBLEM"),
        ("solution", "SOLUTION"),
        ("grading_guidelines", "GRADING GUIDELINES"),
        ("student_answer", "STUDENT ANSWER"),
    ]
    
    for key, header in section_order:
        if key in inputs:
            value = inputs[key]
            sections.append(f"{'='*60}")
            sections.append(f"{header}")
            sections.append(f"{'='*60}")
            sections.append(str(value))
            sections.append("")
    
    # Add any remaining inputs not in the standard order
    processed_keys = {k for k, _ in section_order}
    for key, value in inputs.items():
        if key not in processed_keys:
            sections.append(f"{'='*60}")
            sections.append(f"{key.upper().replace('_', ' ')}")
            sections.append(f"{'='*60}")
            sections.append(str(value))
            sections.append("")
    
    return "\n".join(sections)


def _validate_grading_response(response: dict) -> tuple[bool, str, Any]:
    """Validate that the grading response has the required structure.
    
    Returns:
        (is_valid, error_message, extracted_value)
    """
    if not isinstance(response, dict):
        return False, "Response is not a dictionary", None
    
    if "response" not in response:
        return False, "Missing 'response' key in JSON", None
    
    response_value = response["response"]
    if not isinstance(response_value, (str, int, float, bool)):
        return False, f"'response' value has unsupported type: {type(response_value)}", None
    
    return True, "", response_value


class TaskAgent:
    """Task agent that solves IMO grading problems with improved error handling
    and enhanced prompting for better accuracy."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.call_count = 0
        self.success_count = 0
        self.error_count = 0
        self.retry_count = 0

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with retry logic.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        self.call_count += 1
        self.log_fn(f"TaskAgent call #{self.call_count} starting")
        
        # Format inputs with improved structure
        formatted_inputs = _format_inputs(inputs)
        
        instruction = f"""You are an expert grading agent specializing in mathematical problem evaluation. Your task is to carefully evaluate student answers based on the provided problem, solution, and grading guidelines.

{formatted_inputs}

INSTRUCTIONS FOR GRADING:
1. Carefully read and understand the problem requirements
2. Study the provided solution to understand the correct approach and expected answer format
3. Review the grading guidelines to understand evaluation criteria
4. Compare the student's answer against the solution:
   - Check if the final answer matches (numerical equality or equivalent forms)
   - Verify the reasoning approach aligns with the solution
   - Identify any errors or omissions in the student's work
5. Provide your evaluation in the exact JSON format below

RESPONSE FORMAT:
You MUST respond with valid JSON wrapped in <json> tags:

<json>
{{
    "response": "Your evaluation result here - be specific about correctness"
}}
</json>

IMPORTANT:
- Ensure your JSON is valid (no trailing commas, proper quotes)
- The "response" field should contain your complete evaluation
- Be precise in your assessment - indicate if the answer is correct, partially correct, or incorrect
- If numerical, verify the exact value matches the solution"""

        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=[],
                )
                self.log_fn(f"LLM call completed (attempt {attempt + 1}), response length: {len(response)}")
            except Exception as e:
                self.log_fn(f"Error in LLM call (attempt {attempt + 1}): {e}")
                if attempt == max_retries:
                    self.error_count += 1
                    return "Error: LLM call failed after retries", []
                continue

            # Extract prediction from JSON with enhanced error handling
            prediction = None
            try:
                if msg_history and len(msg_history) > 0:
                    last_message = msg_history[-1]
                    text_content = last_message.get("text", "")
                    extracted = _extract_jsons(text_content)
                    if extracted:
                        last_extracted = extracted[-1]
                        is_valid, error_msg, extracted_value = _validate_grading_response(last_extracted)
                        if is_valid:
                            prediction = extracted_value
                            self.success_count += 1
                            self.log_fn(f"Successfully extracted prediction: {str(prediction)[:100]}")
                            return str(prediction), msg_history
                        else:
                            self.log_fn(f"Invalid grading response: {error_msg}")
                            if attempt < max_retries:
                                self.retry_count += 1
                                instruction = f"""Your previous response was invalid: {error_msg}

Please provide a valid JSON response in the exact format:

<json>
{{
    "response": "Your evaluation result here"
}}
</json>

Original task:
{formatted_inputs}"""
                                continue
                    else:
                        self.log_fn("No JSON found in response")
                        if attempt < max_retries:
                            self.retry_count += 1
                            instruction = f"""Your previous response did not contain valid JSON. 

Please respond with valid JSON wrapped in <json> tags:

<json>
{{
    "response": "Your evaluation result here"
}}
</json>

Original task:
{formatted_inputs}"""
                            continue
                else:
                    self.log_fn("Empty message history")
            except Exception as e:
                self.log_fn(f"Error extracting prediction: {e}")
            
            if attempt == max_retries:
                break
        
        # If we get here, all retries failed
        self.error_count += 1
        self.log_fn(f"TaskAgent stats: {self.success_count} successes, {self.error_count} errors, {self.retry_count} retries out of {self.call_count} calls")
        return str(prediction) if prediction is not None else "None", msg_history if 'msg_history' in locals() else []

    def get_stats(self) -> dict:
        """Return agent performance statistics."""
        return {
            "total_calls": self.call_count,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "retry_count": self.retry_count,
            "success_rate": self.success_count / max(1, self.call_count),
        }
