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
    Includes additional heuristics for common LLM output patterns.
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
                try:
                    results.append(json.loads(potential_json))
                except json.JSONDecodeError:
                    # Try fixing common issues
                    fixed = re.sub(r',(\s*[}\]])', r'\1', potential_json)
                    results.append(json.loads(fixed))
        except json.JSONDecodeError:
            pass
    
    # Fallback 2: look for code blocks with json
    if not results:
        code_block_pattern = r'```(?:json)?\s*([\s\S]*?)```'
        matches = re.findall(code_block_pattern, text)
        for match in matches:
            try:
                results.append(json.loads(match.strip()))
            except json.JSONDecodeError:
                try:
                    fixed = re.sub(r',(\s*[}\]])', r'\1', match.strip())
                    results.append(json.loads(fixed))
                except json.JSONDecodeError:
                    continue
    
    return results or None


def _format_inputs(inputs: dict) -> str:
    """Format task inputs into a structured prompt.
    
    Provides better structure and context for the LLM with clear section ordering.
    """
    # Define preferred order for grading context
    key_order = [
        "domain",
        "problem", 
        "solution",
        "grading_guidelines",
        "student_answer"
    ]
    
    parts = []
    # First add keys in preferred order if they exist
    for key in key_order:
        if key in inputs:
            value = inputs[key]
            parts.append(f"{'='*60}")
            parts.append(f"{key.upper().replace('_', ' ')}")
            parts.append(f"{'='*60}")
            parts.append(f"{value}")
            parts.append("")
    
    # Then add any remaining keys
    for key, value in inputs.items():
        if key not in key_order:
            parts.append(f"{'='*60}")
            parts.append(f"{key.upper().replace('_', ' ')}")
            parts.append(f"{'='*60}")
            parts.append(f"{value}")
            parts.append("")
    
    return "\n".join(parts)


def _validate_response_schema(response: dict) -> tuple[bool, str]:
    """Validate that the response follows the expected schema.
    
    Returns:
        (is_valid, error_message)
    """
    if not isinstance(response, dict):
        return False, f"Response is not a dict, got {type(response).__name__}"
    
    if "response" not in response:
        # Check for common alternative keys
        alternatives = ["answer", "result", "evaluation", "grade", "output"]
        for alt in alternatives:
            if alt in response:
                return False, f"Response uses '{alt}' instead of required 'response' key. Please use 'response' as the key name."
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
        
        # Build comprehensive grading instruction
        instruction = f"""You are an expert grading agent for mathematical olympiad problems. Your task is to evaluate student answers with precision and consistency.

{formatted_inputs}

GRADING INSTRUCTIONS:
1. Carefully read the problem and official solution first
2. Review the grading guidelines to understand the scoring criteria
3. Analyze the student's answer step by step
4. Compare the student's approach with the official solution
5. Provide a clear, concise evaluation

IMPORTANT RULES:
- Be objective and consistent with the grading guidelines
- Award partial credit where appropriate based on the guidelines
- If the answer is completely correct, state that clearly
- If there are errors, identify what is wrong and what credit should be given
- Your response should be a professional evaluation suitable for an olympiad grader

OUTPUT FORMAT:
You MUST respond in valid JSON format wrapped in <json> tags:
<json>
{{
    "response": "Your detailed evaluation here. Include: (1) whether the answer is correct/partially correct/incorrect, (2) specific points awarded if applicable, (3) brief reasoning for your decision."
}}
</json>

The "response" field must contain your complete evaluation as a single string."""

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
                                instruction += f"\n\n[VALIDATION ERROR - ATTEMPT {attempt + 1}]: {error_msg}\n\nPlease fix this issue and respond with valid JSON using the exact schema specified above."
                    else:
                        self.log_fn("No JSON found in response")
                        if attempt < self.max_retries:
                            instruction += f"\n\n[VALIDATION ERROR - ATTEMPT {attempt + 1}]: No valid JSON found. You MUST wrap your response in <json>...</json> tags with a valid JSON object containing a 'response' key."
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
