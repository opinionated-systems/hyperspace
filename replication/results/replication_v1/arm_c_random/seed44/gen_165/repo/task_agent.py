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
    Includes additional heuristics for malformed JSON with improved robustness.
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
            # Try to fix common JSON issues with multiple strategies
            fixed = inner
            # Strategy 1: Remove trailing commas before closing braces/brackets
            fixed = re.sub(r',(\s*[}\]])', r'\1', fixed)
            # Strategy 2: Fix single quotes to double quotes (carefully)
            fixed = re.sub(r"(?<!\\)'([^']*?)'(?<!\\)", r'"\1"', fixed)
            # Strategy 3: Remove comments
            fixed = re.sub(r'//.*?\n', '\n', fixed)
            fixed = re.sub(r'/\*.*?\*/', '', fixed, flags=re.DOTALL)
            # Strategy 4: Fix unquoted keys (simple cases)
            fixed = re.sub(r'(\{|,\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', fixed)
            try:
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
                    # Try fixing common issues
                    fixed = re.sub(r',(\s*[}\]])', r'\1', potential_json)
                    fixed = re.sub(r"(?<!\\)'([^']*?)'(?<!\\)", r'"\1"', fixed)
                    fixed = re.sub(r'//.*?\n', '\n', fixed)
                    fixed = re.sub(r'/\*.*?\*/', '', fixed, flags=re.DOTALL)
                    fixed = re.sub(r'(\{|,\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', fixed)
                    results.append(json.loads(fixed))
        except (json.JSONDecodeError, ValueError):
            pass
    
    return results or None


def _format_inputs(inputs: dict) -> str:
    """Format task inputs into a structured prompt.
    
    Provides better structure and context for the LLM with clear section headers.
    """
    parts = []
    # Order matters for grading - present in logical sequence
    key_order = ["domain", "problem", "solution", "grading_guidelines", "student_answer"]
    
    for key in key_order:
        if key in inputs:
            value = inputs[key]
            # Format the key name nicely
            display_key = key.replace("_", " ").title()
            separator = "=" * 60
            parts.append(f"{separator}\n{display_key}\n{separator}\n{value}\n")
    
    # Add any remaining keys not in the standard order
    for key, value in inputs.items():
        if key not in key_order:
            display_key = key.replace("_", " ").title()
            separator = "=" * 60
            parts.append(f"{separator}\n{display_key}\n{separator}\n{value}\n")
    
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
    
    # Additional validation: check for empty responses
    if isinstance(response_value, str) and not response_value.strip():
        return False, "'response' value is empty string"
    
    return True, ""


def _extract_response_heuristic(text: str) -> str | None:
    """Extract a response using heuristics when JSON parsing fails.
    
    Looks for patterns like "Answer: X", "Response: X", "The answer is X", etc.
    Enhanced with multi-line support, content quality checks, and markdown cleanup.
    """
    # Pre-process: remove common code block markers and JSON artifacts
    cleaned_text = re.sub(r'```[\s\S]*?```', '', text)  # Remove code blocks
    cleaned_text = re.sub(r'<json>|</json>', '', cleaned_text)  # Remove JSON tags
    
    # Common patterns for answer extraction (with multi-line support)
    # Improved patterns with better boundary detection
    patterns = [
        r'(?:answer|response|result|evaluation)[:\s]+([\s\S]+?)(?:\n\n(?=[A-Z])|\Z)',
        r'(?:the answer is|the response is|the result is)[:\s]+([\s\S]+?)(?:\n\n(?=[A-Z])|\Z)',
        r'(?:conclusion|verdict|assessment)[:\s]+([\s\S]+?)(?:\n\n(?=[A-Z])|\Z)',
        r'["\']([^"\']{10,})["\']\s*(?:is the answer|is correct|is incorrect)',
        r'(?:feedback|commentary)[:\s]+([\s\S]+?)(?:\n\n(?=[A-Z])|\Z)',
        r'(?:grade|score|rating)[:\s]+([\s\S]+?)(?:\n\n(?=[A-Z])|\Z)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, cleaned_text, re.IGNORECASE | re.DOTALL)
        if match:
            result = match.group(1).strip()
            # Clean up markdown formatting
            result = re.sub(r'\*\*?|__?', '', result)  # Remove bold/italic markers
            result = re.sub(r'\n+', ' ', result)  # Normalize newlines to spaces
            result = re.sub(r'\s+', ' ', result)  # Normalize whitespace
            # Ensure substantial content (at least 15 chars for better quality)
            if len(result) >= 15:
                return result
    
    # Fallback 1: look for substantial paragraphs with quality filtering
    paragraphs = [p.strip() for p in cleaned_text.split('\n\n') if len(p.strip()) > 50]
    # Filter out paragraphs that look like headers or metadata
    filtered = [p for p in paragraphs if not p.endswith(':') and ':' not in p[:30]]
    if filtered:
        # Return the longest substantial paragraph
        return max(filtered, key=len)
    elif paragraphs:
        return max(paragraphs, key=len)
    
    # Fallback 2: extract sentences from remaining text
    sentences = re.split(r'[.!?]+', cleaned_text)
    substantial = [s.strip() for s in sentences if len(s.strip()) > 40]
    if substantial:
        return max(substantial, key=len)
    
    return None


class TaskAgent:
    """Task agent that solves IMO grading problems with improved error handling."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.call_count = 0
        self.success_count = 0
        self.error_count = 0
        self.heuristic_extraction_count = 0

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
        
        instruction = f"""You are an expert grading agent specializing in mathematical problem evaluation. Your task is to evaluate student answers based on the provided problem, solution, and grading guidelines.

{formatted_inputs}

Your evaluation must be provided in the following JSON format:
<json>
{{
    "response": "Your detailed evaluation result here"
}}
</json>

Important Instructions:
1. First, carefully read and understand the problem statement
2. Study the provided solution to understand the correct approach and expected answer
3. Review the grading guidelines to understand the criteria for evaluation
4. Compare the student's answer against the solution:
   - Check if the final answer matches
   - Evaluate the reasoning process if visible
   - Identify any errors or misconceptions
   - Note any partial credit considerations
5. Provide a clear, detailed evaluation in the "response" field
6. Ensure your JSON is valid - use double quotes, no trailing commas

Think step by step before providing your final evaluation."""

        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
            self.log_fn(f"LLM call completed, response length: {len(response)}")
        except Exception as e:
            self.log_fn(f"Error in LLM call: {e}")
            self.error_count += 1
            return "Error: LLM call failed", []

        # Extract prediction from JSON with better error handling
        prediction = "None"
        extraction_method = "none"
        
        try:
            if msg_history and len(msg_history) > 0:
                last_message = msg_history[-1]
                text_content = last_message.get("text", "")
                
                # Try JSON extraction first
                extracted = _extract_jsons(text_content)
                if extracted:
                    last_extracted = extracted[-1]
                    is_valid, error_msg = _validate_grading_response(last_extracted)
                    if is_valid and isinstance(last_extracted, dict) and "response" in last_extracted:
                        prediction = last_extracted["response"]
                        extraction_method = "json"
                        self.success_count += 1
                        self.log_fn(f"Successfully extracted prediction via JSON: {str(prediction)[:100]}")
                    else:
                        self.log_fn(f"Invalid grading response from JSON: {error_msg}")
                        # Try heuristic extraction as fallback
                        heuristic = _extract_response_heuristic(text_content)
                        if heuristic:
                            prediction = heuristic
                            extraction_method = "heuristic"
                            self.heuristic_extraction_count += 1
                            self.success_count += 1
                            self.log_fn(f"Extracted prediction via heuristic: {str(prediction)[:100]}")
                        else:
                            self.error_count += 1
                            self.log_fn("No valid response found via JSON or heuristics")
                else:
                    # Try heuristic extraction when no JSON found
                    heuristic = _extract_response_heuristic(text_content)
                    if heuristic:
                        prediction = heuristic
                        extraction_method = "heuristic"
                        self.heuristic_extraction_count += 1
                        self.success_count += 1
                        self.log_fn(f"Extracted prediction via heuristic (no JSON): {str(prediction)[:100]}")
                    else:
                        self.error_count += 1
                        self.log_fn("No JSON found in response and heuristic extraction failed")
            else:
                self.error_count += 1
                self.log_fn("Empty message history")
        except Exception as e:
            self.error_count += 1
            self.log_fn(f"Error extracting prediction: {e}")

        self.log_fn(f"TaskAgent stats: {self.success_count} successes, {self.error_count} errors, {self.heuristic_extraction_count} heuristic extractions out of {self.call_count} calls (method: {extraction_method})")
        return str(prediction), msg_history

    def get_stats(self) -> dict:
        """Return agent performance statistics."""
        return {
            "total_calls": self.call_count,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "heuristic_extraction_count": self.heuristic_extraction_count,
            "success_rate": self.success_count / max(1, self.call_count),
        }
