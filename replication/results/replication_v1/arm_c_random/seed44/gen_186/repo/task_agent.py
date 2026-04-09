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
    Includes multiple fallback strategies for robust extraction.
    """
    if not text or not isinstance(text, str):
        return None
        
    results = []
    search_from = 0
    
    # Strategy 1: Extract from <json>...</json> tags
    while True:
        start = text.find("<json>", search_from)
        if start == -1:
            break
        end = text.find("</json>", start)
        if end == -1:
            break
        inner = text[start + 6:end].strip()
        search_from = end + 7
        
        # Try parsing with multiple cleanup strategies
        parsed = _try_parse_json_with_cleanups(inner)
        if parsed:
            results.append(parsed)
    
    # Strategy 2: Try to find JSON objects directly if no <json> tags
    if not results:
        parsed = _try_extract_brace_json(text)
        if parsed:
            results.append(parsed)
    
    # Strategy 3: Try to find JSON in code blocks
    if not results:
        results.extend(_try_extract_from_code_blocks(text))
    
    # Strategy 4: Try to find JSON with regex pattern matching
    if not results:
        results.extend(_try_extract_with_regex(text))
    
    return results or None


def _try_parse_json_with_cleanups(text: str) -> dict | None:
    """Try to parse JSON with various cleanup strategies."""
    if not text:
        return None
        
    # Strategy 1: Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Strategy 2: Remove trailing commas before closing braces/brackets
    try:
        cleaned = re.sub(r',(\s*[}\]])', r'\1', text)
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    
    # Strategy 3: Fix single quotes to double quotes (carefully)
    try:
        # Only replace single quotes that are likely JSON string delimiters
        # This is a simplified approach - replace ' with " except inside words
        cleaned = re.sub(r"(?<!\w)'(?!\w)", '"', text)
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    
    # Strategy 4: Remove control characters and normalize whitespace
    try:
        cleaned = re.sub(r'[\x00-\x1F\x7F]', '', text)
        cleaned = re.sub(r'\s+', ' ', cleaned)
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    
    # Strategy 5: Try to extract just the first valid JSON object
    try:
        # Find the first { and try to balance braces
        start = text.find('{')
        if start != -1:
            brace_count = 0
            for i, char in enumerate(text[start:], start):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        potential_json = text[start:i+1]
                        return json.loads(potential_json)
    except (json.JSONDecodeError, ValueError):
        pass
    
    return None


def _try_extract_brace_json(text: str) -> dict | None:
    """Try to extract JSON from brace-delimited content."""
    try:
        # Look for JSON-like structures with braces
        brace_start = text.find('{')
        brace_end = text.rfind('}')
        if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
            potential_json = text[brace_start:brace_end + 1]
            return _try_parse_json_with_cleanups(potential_json)
    except Exception:
        pass
    return None


def _try_extract_from_code_blocks(text: str) -> list[dict]:
    """Try to extract JSON from markdown code blocks."""
    results = []
    code_block_pattern = r'```(?:json)?\s*(.*?)\s*```'
    matches = re.findall(code_block_pattern, text, re.DOTALL)
    for match in matches:
        parsed = _try_parse_json_with_cleanups(match.strip())
        if parsed:
            results.append(parsed)
    return results


def _try_extract_with_regex(text: str) -> list[dict]:
    """Try to extract JSON objects using regex patterns."""
    results = []
    # Pattern to match JSON-like objects with nested structures
    # This is a best-effort approach for malformed responses
    try:
        # Look for patterns like {"key": value} or {"key": "value"}
        pattern = r'\{\s*"[^"]+"\s*:\s*[^}]+\}'
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            parsed = _try_parse_json_with_cleanups(match)
            if parsed:
                results.append(parsed)
    except Exception:
        pass
    return results


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
    """Task agent that solves IMO grading problems with improved error handling."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.call_count = 0
        self.success_count = 0
        self.error_count = 0

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
        
        instruction = f"""You are an expert grading agent for mathematical problem solving. Your task is to evaluate student answers based on the provided problem, solution, and grading guidelines.

{formatted_inputs}

Respond in JSON format with the following schema:
<json>
{{
    "response": "Your evaluation result here - be specific about what the student got right/wrong"
}}
</json>

CRITICAL: Your response MUST be valid JSON wrapped in <json>...</json> tags. Do not include any text outside the JSON tags.

Evaluation guidelines:
1. First, carefully read and understand the problem requirements
2. Study the provided solution to identify key concepts, steps, and the final answer
3. Review the grading guidelines to understand the scoring criteria
4. Compare the student's answer against the solution:
   - Check if the approach/method is correct
   - Verify calculations and reasoning steps
   - Confirm the final answer matches (or is equivalent to) the solution
5. Provide a clear, specific evaluation in the JSON response field
6. If the answer is partially correct, explain what aspects are correct and what needs improvement

Grading best practices:
- Be objective and consistent with the provided solution
- Award full credit if the student's reasoning is sound and the answer is correct
- Award partial credit if the approach is correct but there are minor errors
- Award no credit if the approach is fundamentally wrong or the answer is incorrect
- Consider equivalent forms of the same answer (e.g., 1/2 vs 0.5) as correct
- Check for common student misconceptions and address them in your evaluation

Be thorough, precise, and fair in your evaluation."""

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

        # Extract prediction from JSON with better error handling and retry
        prediction = "None"
        extraction_attempts = []
        
        try:
            if msg_history and len(msg_history) > 0:
                # Try extracting from the last few messages (in case of retries)
                for msg in reversed(msg_history[-3:]):
                    text_content = msg.get("text", "")
                    if not text_content:
                        continue
                    
                    extracted = _extract_jsons(text_content)
                    if extracted:
                        for ext in reversed(extracted):
                            is_valid, error_msg = _validate_grading_response(ext)
                            if is_valid and isinstance(ext, dict) and "response" in ext:
                                prediction = ext["response"]
                                self.success_count += 1
                                self.log_fn(f"Successfully extracted prediction: {str(prediction)[:100]}")
                                break
                            else:
                                extraction_attempts.append(f"Invalid: {error_msg}")
                        if prediction != "None":
                            break
                    else:
                        extraction_attempts.append("No JSON found")
                
                if prediction == "None":
                    self.error_count += 1
                    if extraction_attempts:
                        self.log_fn(f"JSON extraction failed: {'; '.join(extraction_attempts[:3])}")
                    else:
                        self.log_fn("No valid JSON found in recent messages")
                    
                    # Fallback: try to extract any meaningful text from the response
                    for msg in reversed(msg_history[-2:]):
                        text_content = msg.get("text", "")
                        if text_content and len(text_content) > 10:
                            # Try to find a sentence that looks like an evaluation
                            sentences = re.split(r'[.!?]+', text_content)
                            for sentence in sentences:
                                sentence = sentence.strip()
                                if len(sentence) > 20 and any(keyword in sentence.lower() for keyword in ['correct', 'incorrect', 'answer', 'solution', 'student']):
                                    prediction = f"[Extracted from text] {sentence}"
                                    self.log_fn(f"Fallback extraction used: {str(prediction)[:100]}")
                                    break
                            if prediction != "None":
                                break
            else:
                self.error_count += 1
                self.log_fn("Empty message history")
        except Exception as e:
            self.error_count += 1
            self.log_fn(f"Error extracting prediction: {e}")

        self.log_fn(f"TaskAgent stats: {self.success_count} successes, {self.error_count} errors out of {self.call_count} calls")
        return str(prediction), msg_history

    def get_stats(self) -> dict:
        """Return agent performance statistics."""
        return {
            "total_calls": self.call_count,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "success_rate": self.success_count / max(1, self.call_count),
        }
