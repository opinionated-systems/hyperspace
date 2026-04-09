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
    Includes robust error recovery for malformed JSON with multiple fallback strategies.
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
        
        # Try parsing with multiple fallback strategies
        parsed = _try_parse_json(inner)
        if parsed is not None:
            results.append(parsed)
    
    # Fallback: try to find JSON objects directly if no <json> tags
    if not results:
        parsed = _try_parse_json(text)
        if parsed is not None:
            results.append(parsed)
    
    return results or None


def _try_parse_json(text: str) -> dict | None:
    """Attempt to parse JSON with multiple fallback strategies.
    
    Returns the parsed dict or None if all strategies fail.
    """
    # Strategy 1: Direct JSON parsing
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Strategy 2: Extract JSON from braces
    try:
        brace_start = text.find("{")
        brace_end = text.rfind("}")
        if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
            potential_json = text[brace_start:brace_end + 1]
            return json.loads(potential_json)
    except json.JSONDecodeError:
        pass
    
    # Strategy 3: Apply common fixes and retry
    try:
        fixed = text
        # Remove trailing commas before closing braces/brackets
        fixed = re.sub(r',(\s*[}\]])', r'\1', fixed)
        # Fix single quotes to double quotes (common LLM mistake)
        fixed = re.sub(r"'([^']*?)'", r'"\1"', fixed)
        # Remove comments
        fixed = re.sub(r'//.*?\n', '\n', fixed)
        fixed = re.sub(r'/\*.*?\*/', '', fixed, flags=re.DOTALL)
        # Extract from braces again after cleaning
        brace_start = fixed.find("{")
        brace_end = fixed.rfind("}")
        if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
            fixed = fixed[brace_start:brace_end + 1]
        return json.loads(fixed)
    except (json.JSONDecodeError, ValueError):
        pass
    
    # Strategy 4: Try to extract just the response field if it exists
    try:
        response_match = re.search(r'"response"\s*:\s*"([^"]*)"', text)
        if response_match:
            return {"response": response_match.group(1)}
        # Try with single quotes
        response_match = re.search(r"'response'\s*:\s*'([^']*)'", text)
        if response_match:
            return {"response": response_match.group(1)}
    except Exception:
        pass
    
    return None


def _format_inputs(inputs: dict) -> str:
    """Format task inputs into a structured prompt.
    
    Provides better structure and context for the LLM with clear section headers.
    """
    # Define the order of sections for better readability
    section_order = [
        "domain",
        "problem", 
        "solution",
        "grading_guidelines",
        "student_answer"
    ]
    
    parts = []
    
    # Add sections in preferred order
    for key in section_order:
        if key in inputs:
            value = inputs[key]
            # Format section header with clear visual separation
            header = key.replace("_", " ").upper()
            parts.append(f"{'='*60}")
            parts.append(f"{header}")
            parts.append(f"{'='*60}")
            parts.append(str(value))
            parts.append("")  # Empty line for separation
    
    # Add any remaining keys not in the preferred order
    for key, value in inputs.items():
        if key not in section_order:
            header = key.replace("_", " ").upper()
            parts.append(f"{'='*60}")
            parts.append(f"{header}")
            parts.append(f"{'='*60}")
            parts.append(str(value))
            parts.append("")
    
    return "\n".join(parts)


def _validate_grading_response(response: dict) -> tuple[bool, str]:
    """Validate that the grading response has the required structure.
    
    Returns:
        (is_valid, error_message)
    """
    if not isinstance(response, dict):
        return False, "Response is not a dictionary"
    
    if "response" not in response:
        # Check for common alternative keys
        alt_keys = ["answer", "evaluation", "result", "grade", "feedback"]
        for key in alt_keys:
            if key in response:
                # Auto-convert to standard format
                response["response"] = response[key]
                return True, ""
        return False, "Missing 'response' key in JSON"
    
    response_value = response["response"]
    if response_value is None:
        return False, "'response' value is None"
    
    if not isinstance(response_value, (str, int, float, bool)):
        # Try to convert to string
        try:
            response["response"] = str(response_value)
            return True, ""
        except Exception:
            return False, f"'response' value has unsupported type: {type(response_value)}"
    
    # Ensure string responses are not empty
    if isinstance(response_value, str) and not response_value.strip():
        return False, "'response' value is empty string"
    
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

    def forward(self, inputs: dict, max_retries: int = 3) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with retry logic.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer
            max_retries: maximum number of retry attempts on failure

        Returns:
            (prediction, msg_history)
        """
        self.call_count += 1
        self.log_fn(f"TaskAgent call #{self.call_count} starting")
        
        # Validate inputs
        if not isinstance(inputs, dict):
            self.error_count += 1
            return "Error: Invalid inputs - must be a dictionary", []
        
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
3. Ensure valid JSON - use double quotes, not single quotes
4. Do not include any text outside the JSON tags

Step-by-step evaluation process:
1. PROBLEM ANALYSIS: Identify what the problem is asking and key concepts involved
2. SOLUTION REVIEW: Understand the correct approach and expected answer format
3. GUIDELINE MAPPING: Map each grading criterion to specific aspects of the solution
4. STUDENT ANSWER EVALUATION:
   - Check if the answer addresses the core question
   - Verify mathematical correctness (if applicable)
   - Assess completeness and clarity
   - Compare against the provided solution
5. REASONING SYNTHESIS: Formulate a clear, justified evaluation
6. FINAL RESPONSE: Output your evaluation in the required JSON format

Remember: Be objective, consistent, and provide specific feedback in your evaluation."""

        # Retry loop with progressive temperature variation
        for attempt in range(max_retries + 1):
            try:
                # Progressive temperature: 0.0 -> 0.2 -> 0.4 -> 0.6
                temperature = min(0.6, attempt * 0.2)
                
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
                    
                    # Log a snippet of the response for debugging
                    snippet = text_content[:200].replace('\n', ' ')
                    self.log_fn(f"Response snippet: {snippet}...")
                    
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
                            # Log the problematic response for debugging
                            self.log_fn(f"Problematic response: {str(last_extracted)[:200]}")
                    else:
                        self.log_fn(f"No JSON found in response on attempt {attempt + 1}")
                else:
                    self.log_fn(f"Empty message history on attempt {attempt + 1}")
            except Exception as e:
                self.log_fn(f"Error extracting prediction on attempt {attempt + 1}: {e}")
            
            # If we get here, extraction failed - retry if attempts remain
            if attempt < max_retries:
                self.log_fn(f"Retrying with temperature={min(0.6, (attempt + 1) * 0.2)}...")
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
