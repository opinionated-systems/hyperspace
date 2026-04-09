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
                fixed = re.sub(r"'([^']*?)'", r'"\1"', fixed)
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
                    fixed = re.sub(r"'([^']*?)'", r'"\1"', fixed)
                    results.append(json.loads(fixed))
        except json.JSONDecodeError:
            pass
    
    return results or None


def _format_inputs(inputs: dict) -> str:
    """Format task inputs into a structured prompt.
    
    Provides better structure and context for the LLM.
    """
    parts = []
    for key, value in inputs.items():
        parts.append(f"{key}:\n{value}\n")
    return "\n".join(parts)


def _validate_grading_response(response: dict) -> tuple[bool, str]:
    """Validate that the grading response has the required structure.
    
    Returns:
        (is_valid, error_message)
    """
    if not isinstance(response, dict):
        return False, "Response is not a dictionary"
    
    if "response" not in response:
        return False, "Missing 'response' key in JSON"
    
    response_value = response["response"]
    if not isinstance(response_value, (str, int, float, bool)):
        return False, f"'response' value has unsupported type: {type(response_value)}"
    
    return True, ""


class TaskAgent:
    """Task agent that solves IMO grading problems with improved error handling and retry logic."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.call_count = 0
        self.success_count = 0
        self.error_count = 0
        self.retry_count = 0

    def forward(self, inputs: dict, max_retries: int = 2) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with retry logic.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer
            max_retries: maximum number of retry attempts on failure

        Returns:
            (prediction, msg_history)
        """
        self.call_count += 1
        self.log_fn(f"TaskAgent call #{self.call_count} starting")
        
        # Format inputs more clearly
        formatted_inputs = _format_inputs(inputs)
        
        instruction = f"""You are an expert grading agent for mathematical problem solving. Your task is to evaluate student answers with precision and consistency.

{formatted_inputs}

Respond in JSON format with the following schema:
<json>
{{
    "response": "Your evaluation result here"
}}
</json>

CRITICAL INSTRUCTIONS:
1. You MUST wrap your JSON response in <json>...</json> tags
2. The "response" field must contain your final evaluation
3. Ensure valid JSON - no trailing commas, use double quotes

Step-by-step evaluation process:
1. PROBLEM ANALYSIS: Identify what the problem is asking, key concepts, and expected solution approach
2. SOLUTION REVIEW: Study the provided solution to understand the correct answer and reasoning
3. GUIDELINE INTERPRETATION: Parse the grading criteria - note any partial credit rules, specific requirements, or common error patterns
4. STUDENT ANSWER EVALUATION:
   - Check if the final answer matches the solution (numerical or symbolic)
   - Assess the reasoning/method shown by the student
   - Identify any errors: conceptual, computational, or procedural
   - Note any missing steps or incomplete work
5. GRADING DECISION:
   - Apply the grading guidelines strictly but fairly
   - Consider partial credit if guidelines allow
   - Be consistent with standard mathematical notation and conventions
6. FINAL RESPONSE: Provide a clear, concise evaluation in the JSON format

Remember: Your evaluation should be objective, based solely on the provided materials, and follow the grading guidelines precisely."""

        # Retry loop with temperature variation
        for attempt in range(max_retries + 1):
            try:
                # Vary temperature on retries to get different responses
                temperature = 0.0 if attempt == 0 else 0.3
                
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=[],
                    temperature=temperature,
                )
                self.log_fn(f"LLM call attempt {attempt + 1} completed, response length: {len(response)}")
            except Exception as e:
                self.log_fn(f"Error in LLM call attempt {attempt + 1}: {e}")
                if attempt == max_retries:
                    self.error_count += 1
                    return "Error: LLM call failed", []
                continue

            # Extract prediction from JSON with better error handling
            prediction = "None"
            try:
                if msg_history and len(msg_history) > 0:
                    last_message = msg_history[-1]
                    text_content = last_message.get("text", "")
                    extracted = _extract_jsons(text_content)
                    if extracted:
                        last_extracted = extracted[-1]
                        is_valid, error_msg = _validate_grading_response(last_extracted)
                        if is_valid and isinstance(last_extracted, dict) and "response" in last_extracted:
                            prediction = last_extracted["response"]
                            self.success_count += 1
                            if attempt > 0:
                                self.retry_count += 1
                            self.log_fn(f"Successfully extracted prediction (attempt {attempt + 1}): {str(prediction)[:100]}")
                            self.log_fn(f"TaskAgent stats: {self.success_count} successes, {self.error_count} errors, {self.retry_count} retries out of {self.call_count} calls")
                            return str(prediction), msg_history
                        else:
                            self.log_fn(f"Invalid grading response on attempt {attempt + 1}: {error_msg}")
                    else:
                        self.log_fn(f"No JSON found in response on attempt {attempt + 1}")
                else:
                    self.log_fn(f"Empty message history on attempt {attempt + 1}")
            except Exception as e:
                self.log_fn(f"Error extracting prediction on attempt {attempt + 1}: {e}")
            
            # If we get here, extraction failed - retry if attempts remain
            if attempt < max_retries:
                self.log_fn(f"Retrying with temperature={0.3}...")
                continue
        
        # All retries exhausted
        self.error_count += 1
        self.log_fn(f"TaskAgent stats: {self.success_count} successes, {self.error_count} errors, {self.retry_count} retries out of {self.call_count} calls")
        return "None", msg_history

    def get_stats(self) -> dict:
        """Return agent performance statistics."""
        return {
            "total_calls": self.call_count,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "retry_count": self.retry_count,
            "success_rate": self.success_count / max(1, self.call_count),
            "retry_rate": self.retry_count / max(1, self.success_count) if self.success_count > 0 else 0.0,
        }
