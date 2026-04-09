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
    Includes additional heuristics for malformed JSON with enhanced recovery.
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
            # Try to fix common JSON issues with enhanced recovery
            try:
                fixed = _fix_json_string(inner)
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
                    try:
                        fixed = _fix_json_string(potential_json)
                        results.append(json.loads(fixed))
                    except json.JSONDecodeError:
                        pass
        except (json.JSONDecodeError, ValueError):
            pass
    
    return results or None


def _fix_json_string(json_str: str) -> str:
    """Fix common JSON formatting issues.
    
    Applies multiple fixes to handle malformed JSON from LLM outputs:
    - Remove trailing commas before closing braces/brackets
    - Fix single quotes to double quotes (carefully)
    - Handle unescaped newlines in strings
    - Fix missing quotes around keys
    - Handle escaped quotes properly
    """
    fixed = json_str
    
    # Remove trailing commas before closing braces/brackets
    fixed = re.sub(r',(\s*[}\]])', r'\1', fixed)
    
    # Fix single quotes to double quotes (carefully, not inside already-quoted strings)
    # This regex handles simple cases: 'key' or 'value' -> "key" or "value"
    fixed = re.sub(r"(?<!\\)'([^']*?)'(?<!\\)", r'"\1"', fixed)
    
    # Handle unescaped newlines in string values by replacing with \n
    # This is a simplified approach - replace newlines between quotes
    fixed = re.sub(r'("[^"]*?)\n([^"]*?")', r'\1\\n\2', fixed)
    
    # Fix missing quotes around unquoted keys (e.g., {key: "value"} -> {"key": "value"})
    # Match word characters followed by colon, not preceded by quote
    fixed = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)(\s*:)', r'\1"\2"\3', fixed)
    
    return fixed


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


def _sanitize_grading_output(text: str) -> str:
    """Sanitize and normalize grading output for consistent processing.
    
    Handles common formatting issues that can occur in LLM grading responses:
    - Removes excessive whitespace and newlines
    - Normalizes Unicode characters
    - Truncates overly long responses to prevent context overflow
    - Ensures consistent line endings
    
    Args:
        text: Raw grading response text
        
    Returns:
        Sanitized text ready for downstream processing
    """
    if not text:
        return ""
    
    # Normalize line endings to Unix style
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    
    # Remove excessive blank lines (more than 2 consecutive)
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    
    # Normalize Unicode whitespace to regular space
    import unicodedata
    text = "".join(
        " " if unicodedata.category(c) == "Zs" else c
        for c in text
    )
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    # Truncate if extremely long (preserve first and last portions)
    max_len = 50000
    if len(text) > max_len:
        head = max_len // 2
        tail = max_len // 2
        text = text[:head] + f"\n\n... [content truncated: {len(text)} chars total] ...\n\n" + text[-tail:]
    
    return text


def _extract_response_heuristic(text: str) -> str | None:
    """Extract a response using heuristics when JSON parsing fails.
    
    Looks for patterns like "Answer: X", "Response: X", "The answer is X", etc.
    Enhanced with more patterns and better multi-line handling.
    """
    # Common patterns for answer extraction (ordered by priority)
    patterns = [
        # Direct response patterns with explicit labels
        r'["\']?response["\']?\s*[:=]\s*["\']?([^"\']{3,500})["\']?(?:\n|$)',
        r'["\']?answer["\']?\s*[:=]\s*["\']?([^"\']{3,500})["\']?(?:\n|$)',
        # Evaluation/grading specific patterns
        r'(?:evaluation|assessment|grade)[:\s]+(.{10,500}?)(?:\n\n|\Z)',
        r'(?:the student[\'\'s]? answer is)\s+(.{10,500}?)(?:\n\n|\Z)',
        # Conclusion patterns
        r'(?:conclusion|verdict|summary)[:\s]+(.{10,500}?)(?:\n\n|\Z)',
        r'(?:in conclusion|to summarize|overall)[,:]?\s+(.{10,500}?)(?:\n\n|\Z)',
        # Standard answer patterns
        r'(?:the answer is|the response is|the result is)[:\s]+(.{3,500}?)(?:\n|$)',
        r'(?:answer|response|result)[:\s]+(.{3,500}?)(?:\n|$)',
        # Quote-based patterns
        r'["\']([^"\']{10,500})["\']\s*(?:is the answer|is correct|is incorrect|is the evaluation)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            extracted = match.group(1).strip()
            # Clean up the extracted text
            extracted = re.sub(r'\s+', ' ', extracted)  # Normalize whitespace
            if len(extracted) >= 3:  # Minimum meaningful length
                return extracted
    
    # Last resort: look for substantial text after common markers
    last_resort_patterns = [
        r'(?:therefore|thus|hence)[,:]?\s+(.{20,500}?)(?:\n\n|\Z)',
        r'(?:final answer|final response)[,:]?\s+(.{10,500}?)(?:\n\n|\Z)',
    ]
    
    for pattern in last_resort_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            extracted = match.group(1).strip()
            extracted = re.sub(r'\s+', ' ', extracted)
            if len(extracted) >= 10:
                return extracted
    
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

CRITICAL FORMATTING REQUIREMENTS:
1. Your response MUST be valid JSON inside the <json>...</json> tags
2. Use double quotes (") for all strings, never single quotes (')
3. Do NOT include trailing commas before closing braces or brackets
4. Escape any double quotes inside the response value with backslash (\\")
5. The "response" field must contain a string value with your complete evaluation

Grading Instructions:
1. First, carefully read and understand the problem statement
2. Study the provided solution to understand the correct approach and expected answer
3. Review the grading guidelines to understand the criteria for evaluation
4. Compare the student's answer against the solution:
   - Check if the final answer matches numerically or symbolically
   - Evaluate the reasoning process if visible
   - Identify any errors or misconceptions
   - Note any partial credit considerations
   - Consider alternative valid approaches
5. Provide a clear, detailed evaluation in the "response" field that explains your reasoning
6. Include whether the answer is correct, partially correct, or incorrect with justification

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
