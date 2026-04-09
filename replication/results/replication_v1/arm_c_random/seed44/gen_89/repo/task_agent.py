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
    
    Enhanced to handle nested braces, multiple JSON objects, and common
    formatting issues like trailing commas and comments.
    """
    if not text or not isinstance(text, str):
        logger.debug("Invalid input to _extract_jsons: empty or non-string text")
        return None
        
    results = []
    search_from = 0
    json_tag_count = 0
    
    # First pass: extract from <json> tags
    while True:
        start = text.find("<json>", search_from)
        if start == -1:
            break
        end = text.find("</json>", start)
        if end == -1:
            logger.debug(f"Found opening <json> at {start} but no closing </json>")
            break
        
        json_tag_count += 1
        inner = text[start + 6:end].strip()
        search_from = end + 7
        
        if not inner:
            logger.debug(f"Empty <json> block at position {start}")
            continue
        
        # Try to clean up common JSON issues before parsing
        cleaned = _clean_json_string(inner)
        
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                results.append(parsed)
                logger.debug(f"Successfully parsed JSON from <json> block #{json_tag_count}")
            else:
                logger.debug(f"Parsed JSON from <json> block is not a dict: {type(parsed)}")
        except json.JSONDecodeError as e:
            # Try original if cleaning failed
            try:
                parsed = json.loads(inner)
                if isinstance(parsed, dict):
                    results.append(parsed)
                    logger.debug(f"Successfully parsed JSON from <json> block #{json_tag_count} (without cleaning)")
                else:
                    logger.debug(f"Parsed JSON from <json> block is not a dict: {type(parsed)}")
            except json.JSONDecodeError as e2:
                logger.debug(f"Failed to parse JSON from <json> block #{json_tag_count}: {e2}")
                continue
    
    # Second pass: try to find JSON objects directly if no <json> tags found or no valid results
    if not results:
        logger.debug("No valid JSON found in <json> tags, searching for raw JSON objects")
        # Try to find JSON objects by looking for balanced braces
        brace_start = 0
        raw_json_count = 0
        while brace_start < len(text):
            brace_start = text.find("{", brace_start)
            if brace_start == -1:
                break
            
            # Find matching closing brace
            brace_count = 0
            brace_end = brace_start
            for i, char in enumerate(text[brace_start:], start=brace_start):
                if char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        brace_end = i
                        break
            
            if brace_end > brace_start:
                raw_json_count += 1
                potential_json = text[brace_start:brace_end + 1]
                # Clean up common JSON issues
                cleaned = _clean_json_string(potential_json)
                try:
                    parsed = json.loads(cleaned)
                    if isinstance(parsed, dict):
                        results.append(parsed)
                        logger.debug(f"Successfully parsed raw JSON object #{raw_json_count}")
                except json.JSONDecodeError:
                    # Try original if cleaning failed
                    try:
                        parsed = json.loads(potential_json)
                        if isinstance(parsed, dict):
                            results.append(parsed)
                            logger.debug(f"Successfully parsed raw JSON object #{raw_json_count} (without cleaning)")
                    except json.JSONDecodeError:
                        pass
                brace_start = brace_end + 1
            else:
                break
    
    if results:
        logger.debug(f"_extract_jsons found {len(results)} valid JSON object(s)")
    else:
        logger.debug("_extract_jsons found no valid JSON objects")
    
    return results if results else None


def _clean_json_string(json_str: str) -> str:
    """Clean up common JSON formatting issues.
    
    Handles: trailing commas, Python-style booleans/None, comments,
    and extra whitespace. Also handles nested structures better.
    """
    if not json_str or not isinstance(json_str, str):
        return json_str
    
    cleaned = json_str.strip()
    
    # Remove single-line comments (// ...)
    cleaned = re.sub(r'//[^\n]*', '', cleaned)
    
    # Remove multi-line comments (/* ... */)
    cleaned = re.sub(r'/\*.*?\*/', '', cleaned, flags=re.DOTALL)
    
    # Remove trailing commas before closing braces/brackets
    cleaned = re.sub(r',(\s*[}\]])', r'\1', cleaned)
    
    # Convert Python-style booleans and None to JSON equivalents
    cleaned = re.sub(r'\bTrue\b', 'true', cleaned)
    cleaned = re.sub(r'\bFalse\b', 'false', cleaned)
    cleaned = re.sub(r'\bNone\b', 'null', cleaned)
    
    # Handle unquoted keys (simple cases only)
    # This regex looks for word characters followed by colon at start of object or after comma
    cleaned = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', cleaned)
    
    return cleaned


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


def _extract_text_fallback(text: str) -> str | None:
    """Extract meaningful text when JSON parsing fails.
    
    Looks for patterns like:
    - "The answer is..." or "Answer: ..."
    - "Evaluation: ..." or "Result: ..."
    - The last sentence/paragraph that might contain the answer
    
    Returns the extracted text or None if no clear answer found.
    """
    if not text or not isinstance(text, str):
        return None
    
    text = text.strip()
    
    # Look for explicit answer patterns
    patterns = [
        r'(?:the\s+)?(?:answer|evaluation|result|grade|score)(?:\s+is)?[:\s]+([^\n.]+)',
        r'(?:therefore|thus|so|conclusion)[:\s,]+([^\n.]+)',
        r'(?:student\s+(?:answer|response))(?:\s+is)?[:\s]+([^\n.]+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            result = match.group(1).strip()
            if result:
                return result
    
    # Look for the last substantial sentence (might be the conclusion)
    sentences = re.split(r'[.!?]\s+', text)
    for sentence in reversed(sentences):
        sentence = sentence.strip()
        if len(sentence) > 10 and not sentence.lower().startswith(('here', 'this', 'the')):
            return sentence
    
    # If nothing else, return the last non-empty line
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    if lines:
        return lines[-1]
    
    return None


class TaskAgent:
    """Task agent that solves IMO grading problems with improved error handling."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.call_count = 0
        self.success_count = 0
        self.error_count = 0
        self.fallback_count = 0

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
        
        instruction = f"""You are an expert grading agent. Your task is to evaluate student answers based on the provided problem, solution, and grading guidelines.

{formatted_inputs}

Respond in JSON format with the following schema:
<json>
{{
    "response": "Your evaluation result here"
}}
</json>

Ensure your response is valid JSON and follows the schema exactly.

Think step by step:
1. First, understand the problem and what is being asked
2. Review the provided solution to understand the correct approach
3. Examine the grading guidelines carefully
4. Evaluate the student's answer against the solution and guidelines
5. Formulate your final evaluation in the required JSON format"""

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

        # Extract prediction from JSON with better error handling and fallback
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
                        self.log_fn(f"Successfully extracted prediction: {str(prediction)[:100]}")
                    else:
                        # Try fallback extraction from raw text
                        fallback = _extract_text_fallback(text_content)
                        if fallback:
                            prediction = fallback
                            self.fallback_count += 1
                            self.success_count += 1
                            self.log_fn(f"Used fallback extraction: {str(prediction)[:100]}")
                        else:
                            self.error_count += 1
                            self.log_fn(f"Invalid grading response: {error_msg}")
                else:
                    # No JSON found, try fallback extraction
                    fallback = _extract_text_fallback(text_content)
                    if fallback:
                        prediction = fallback
                        self.fallback_count += 1
                        self.success_count += 1
                        self.log_fn(f"Used fallback extraction (no JSON): {str(prediction)[:100]}")
                    else:
                        self.error_count += 1
                        self.log_fn("No JSON found in response and fallback failed")
            else:
                self.error_count += 1
                self.log_fn("Empty message history")
        except Exception as e:
            self.error_count += 1
            self.log_fn(f"Error extracting prediction: {e}")

        self.log_fn(f"TaskAgent stats: {self.success_count} successes ({self.fallback_count} fallback), {self.error_count} errors out of {self.call_count} calls")
        return str(prediction), msg_history

    def get_stats(self) -> dict:
        """Return agent performance statistics."""
        return {
            "total_calls": self.call_count,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "fallback_count": self.fallback_count,
            "success_rate": self.success_count / max(1, self.call_count),
            "fallback_rate": self.fallback_count / max(1, self.call_count),
        }

    def reset_stats(self) -> None:
        """Reset agent performance statistics."""
        self.call_count = 0
        self.success_count = 0
        self.error_count = 0
        self.fallback_count = 0
        self.log_fn("TaskAgent stats reset")
