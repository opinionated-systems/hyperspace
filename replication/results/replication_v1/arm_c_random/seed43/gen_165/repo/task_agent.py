"""
Task agent: solves a given task with a single LLM call.

Reimplemented from facebookresearch/HyperAgents task_agent.py.
Same interface, same JSON output format, same extraction logic.

This is the INITIAL task agent. The meta agent modifies this file
during self-improvement. The evaluation harness loads whatever
task_agent.py exists at the agent's repo path.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from functools import lru_cache

from agent.llm_client import get_response_from_llm, EVAL_MODEL

logger = logging.getLogger(__name__)

# Simple in-memory cache for LLM responses to avoid redundant calls
# Key: hash of (model, prompt), Value: (response, msg_history, info)
_llm_response_cache: dict[str, tuple[str, list, dict]] = {}
_MAX_CACHE_SIZE = 100


def _get_cache_key(model: str, prompt: str) -> str:
    """Generate a cache key from model and prompt."""
    content = f"{model}:{prompt}"
    return hashlib.sha256(content.encode()).hexdigest()[:32]


def get_cache_stats() -> dict:
    """Get cache statistics for debugging.
    
    Returns:
        dict with cache_size, max_size, and hit/miss info if tracked
    """
    return {
        "cache_size": len(_llm_response_cache),
        "max_size": _MAX_CACHE_SIZE,
        "cache_keys": list(_llm_response_cache.keys())[:5],  # First 5 keys for debugging
    }


def clear_cache() -> None:
    """Clear the LLM response cache."""
    global _llm_response_cache
    _llm_response_cache.clear()


def _safe_json_loads(text: str) -> dict | None:
    """Safely parse JSON with error handling and recovery.
    
    Attempts to parse the text as JSON. If that fails, tries to find
    a JSON object within the text by looking for outermost braces.
    
    Args:
        text: The text to parse
        
    Returns:
        Parsed dict or None if parsing fails
    """
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find valid JSON within the content
        json_start = text.find('{')
        json_end = text.rfind('}')
        if json_start != -1 and json_end != -1 and json_end > json_start:
            try:
                return json.loads(text[json_start:json_end+1])
            except json.JSONDecodeError:
                pass
    return None


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
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
        parsed = _safe_json_loads(inner)
        if parsed is not None:
            results.append(parsed)
    return results or None


def _extract_json_from_markdown(text: str) -> dict | None:
    """Extract JSON from markdown code blocks (```json ... ```)."""
    pattern = r'```json\s*(.*?)\s*```'
    matches = re.findall(pattern, text, re.DOTALL)
    for match in matches:
        parsed = _safe_json_loads(match)
        if parsed is not None:
            return parsed
    return None


def _validate_grade(prediction: str, grading_guidelines: str) -> tuple[str, bool]:
    """Validate that the extracted grade is reasonable.
    
    Enhanced validation with strict support for IMO 0-7 point scale.
    Only accepts single digit grades 0-7 for consistency.
    
    Returns:
        (validated_grade, is_valid)
    """
    if not prediction or prediction == "None":
        return "None", False
    
    pred_clean = prediction.strip()
    pred_lower = pred_clean.lower()
    
    # Strict validation: only accept single digit 0-7
    # This ensures consistent output format
    if pred_clean in ["0", "1", "2", "3", "4", "5", "6", "7"]:
        return pred_clean, True
    
    # Check for fractional grades like "3/7" or "5 / 7" - extract just the numerator
    fractional_match = re.search(r'\b([0-7])\s*/\s*7\b', pred_clean)
    if fractional_match:
        return fractional_match.group(1), True
    
    # Check for numeric grades embedded in text (0-7 for IMO problems)
    numeric_match = re.search(r'\b([0-7])\b', pred_clean)
    if numeric_match:
        return numeric_match.group(1), True
    
    # Check for full credit patterns -> 7
    full_patterns = [
        r'\bfull\s*(?:credit|points?|score)?\b',
        r'\bcomplete\s*(?:solution|answer|credit)?\b',
        r'\ball\s*(?:points?|credit|marks?)?\b',
        r'\bperfect\s*(?:score|solution)?\b',
        r'\bcorrect\s*(?:solution|answer)?\b',
    ]
    for pattern in full_patterns:
        if re.search(pattern, pred_lower):
            return "7", True
    
    # Check for zero/no credit patterns -> 0
    zero_patterns = [
        r'\bno\s*(?:credit|points?|score|marks?)?\b',
        r'\bzero\s*(?:credit|points?|score|marks?)?\b',
        r'\b0\s*(?:points?|credit|score|marks?)?\b',
        r'\bincorrect\s*(?:solution|answer)?\b',
        r'\bwrong\s*(?:solution|answer)?\b',
        r'\bnone\b',
    ]
    for pattern in zero_patterns:
        if re.search(pattern, pred_lower):
            return "0", True
    
    # If no clear grade found, mark as invalid
    return pred_clean, False


def _log_structured(log_fn, event: str, data: dict) -> None:
    """Log structured data as JSON for better observability.
    
    Args:
        log_fn: Logging function (e.g., logger.info)
        event: Event name/type
        data: Dictionary of data to log
    """
    entry = {"event": event, **data}
    log_fn(json.dumps(entry, default=str))


class TaskAgent:
    """Task agent that solves IMO grading problems with enhanced reasoning and validation.
    
    This agent uses an LLM to evaluate student answers against official solutions
    for IMO (International Mathematical Olympiad) problems. It supports:
    - Chain-of-thought reasoning
    - Multiple JSON extraction strategies
    - Grade validation (0-7 scale)
    - Fallback extraction from raw text
    
    Attributes:
        model: The LLM model to use for grading
        log_fn: Logging function for observability
    """

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def _build_prompt(self, inputs: dict) -> str:
        """Build a structured prompt for IMO grading with chain-of-thought reasoning."""
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        return f"""You are an expert IMO (International Mathematical Olympiad) grader.

Your task is to evaluate a student's solution to a mathematical problem.

## Problem Domain
{domain}

## Problem Statement
{problem}

## Official Solution
{solution}

## Grading Guidelines
{grading_guidelines}

## Student's Answer
{student_answer}

## Instructions

1. First, analyze the student's answer step by step. Compare it against the official solution.
2. Identify any errors, missing steps, or creative alternative approaches.
3. Consider the grading guidelines carefully - IMO problems are typically graded 0-7 points.
4. Provide your reasoning before giving the final grade.
5. The final grade must be a single numeric value from 0 to 7 (inclusive).

## Output Format

You MUST respond with a valid JSON object wrapped in <json> tags. The JSON must have exactly these two fields:

<json>
{{
    "reasoning": "Your detailed analysis of the student's answer, comparing it to the official solution and explaining your evaluation...",
    "response": "X"
}}
</json>

IMPORTANT:
- The "response" field MUST contain ONLY a single digit from 0 to 7 (e.g., "7", "5", "0")
- Do NOT include any other text, explanations, or formatting in the response field
- The "reasoning" field should contain your full analysis
- Valid grades are: 0, 1, 2, 3, 4, 5, 6, 7
- 7 = full credit (complete correct solution)
- 0 = no credit (completely wrong or blank)
- 1-6 = partial credit based on progress made"""

    def _extract_prediction(self, msg_history: list[dict]) -> tuple[str, str]:
        """Extract prediction and reasoning from message history.
        
        Returns:
            (prediction, reasoning)
        """
        prediction = "None"
        reasoning = ""
        
        try:
            # Handle different message formats
            last_msg = ""
            if msg_history:
                last_entry = msg_history[-1]
                if isinstance(last_entry, dict):
                    # Try common keys for message content
                    last_msg = last_entry.get("text") or last_entry.get("content", "")
                    if not last_msg and "message" in last_entry:
                        msg_obj = last_entry["message"]
                        if isinstance(msg_obj, dict):
                            last_msg = msg_obj.get("content", "")
            
            if not last_msg:
                return prediction, reasoning
            
            # Try <json> tags first
            extracted = _extract_jsons(last_msg)
            if extracted:
                last_json = extracted[-1]
                if "response" in last_json:
                    prediction = str(last_json["response"]).strip()
                if "reasoning" in last_json:
                    reasoning = str(last_json["reasoning"])
                return prediction, reasoning
            
            # Try markdown code blocks
            md_json = _extract_json_from_markdown(last_msg)
            if md_json:
                if "response" in md_json:
                    prediction = str(md_json["response"]).strip()
                if "reasoning" in md_json:
                    reasoning = str(md_json["reasoning"])
                return prediction, reasoning
            
            # Fallback: try to find any JSON-like structure with response field
            json_match = re.search(r'\{[^}]*"response"[^}]*\}', last_msg)
            if json_match:
                try:
                    fallback = json.loads(json_match.group())
                    prediction = str(fallback.get("response", "None")).strip()
                    if "reasoning" in fallback:
                        reasoning = str(fallback["reasoning"])
                except json.JSONDecodeError:
                    pass
            
            # Last resort: look for grade patterns in text
            if prediction == "None":
                # Look for "Grade: X" or "Final grade: X" patterns
                grade_match = re.search(r'(?:grade|score|mark|final grade|final score)\s*:?\s*([0-7])', last_msg, re.IGNORECASE)
                if grade_match:
                    prediction = grade_match.group(1)
                    
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
        
        return prediction, reasoning

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with enhanced reasoning and validation.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        instruction = self._build_prompt(inputs)
        grading_guidelines = inputs.get("grading_guidelines", "")

        # Check cache first to avoid redundant LLM calls
        cache_key = _get_cache_key(self.model, instruction)
        if cache_key in _llm_response_cache:
            self.log_fn(f"Cache hit for key {cache_key[:8]}...")
            response, msg_history, info = _llm_response_cache[cache_key]
            # Still need to extract prediction from cached response
            prediction, reasoning = self._extract_prediction(msg_history)
            validated_grade, is_valid = _validate_grade(prediction, grading_guidelines)
            return str(validated_grade), msg_history

        # Retry mechanism for LLM calls with exponential backoff
        max_retries = 3
        response = None
        msg_history = []
        info = {}
        
        for attempt in range(max_retries):
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=[],
                )
                break  # Success, exit retry loop
            except Exception as e:
                self.log_fn(f"LLM call failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    # Final attempt failed
                    return "None", []
                # Wait before retry with exponential backoff
                import time
                time.sleep(2 ** attempt)
        
        # Cache the successful response (with LRU eviction)
        if len(_llm_response_cache) >= _MAX_CACHE_SIZE:
            # Remove oldest entry (simple FIFO eviction)
            oldest_key = next(iter(_llm_response_cache))
            del _llm_response_cache[oldest_key]
        _llm_response_cache[cache_key] = (response, msg_history, info)
        self.log_fn(f"Cached response with key {cache_key[:8]}...")
        
        if not msg_history:
            self.log_fn("No message history returned from LLM")
            return "None", []

        # Extract prediction with enhanced extraction
        prediction, reasoning = self._extract_prediction(msg_history)
        
        # Validate the grade
        validated_grade, is_valid = _validate_grade(prediction, grading_guidelines)
        
        # Log the reasoning and validation result
        if reasoning:
            self.log_fn(f"Reasoning: {reasoning[:200]}...")
        self.log_fn(f"Extracted grade: {prediction}, Validated: {validated_grade}, Is valid: {is_valid}")
        
        # Structured logging for better observability
        _log_structured(self.log_fn, "grade_extraction", {
            "prediction": prediction,
            "validated_grade": validated_grade,
            "is_valid": is_valid,
            "has_reasoning": bool(reasoning),
        })
        
        # If grade is invalid, try to extract from the full response text
        if not is_valid and response:
            # Try to find any numeric grade in the response
            numeric_match = re.search(r'\b([0-7])\b', response)
            if numeric_match:
                validated_grade = numeric_match.group(1)
                is_valid = True
                self.log_fn(f"Fallback extraction found grade: {validated_grade}")
            else:
                # Try to find grade patterns in the full response
                grade_patterns = [
                    r'(?:grade|score|mark|final grade|final score)\s*:?\s*([0-7])',
                    r'\bfull\s*(?:credit|points?|score)?\b',
                    r'\bcorrect\s*(?:solution|answer)?\b',
                    r'\bno\s*(?:credit|points?|score|marks?)?\b',
                    r'\bzero\s*(?:credit|points?|score|marks?)?\b',
                    r'\bincorrect\s*(?:solution|answer)?\b',
                ]
                for pattern in grade_patterns:
                    match = re.search(pattern, response, re.IGNORECASE)
                    if match:
                        if 'full' in pattern or 'correct' in pattern:
                            validated_grade = "7"
                        elif 'no' in pattern or 'zero' in pattern or 'incorrect' in pattern:
                            validated_grade = "0"
                        else:
                            validated_grade = match.group(1)
                        is_valid = True
                        self.log_fn(f"Pattern-based fallback found grade: {validated_grade}")
                        break

        # Structured logging for final result
        _log_structured(self.log_fn, "grade_final", {
            "final_grade": str(validated_grade),
            "is_valid": is_valid,
            "fallback_used": not is_valid and prediction == "None",
        })

        return str(validated_grade), msg_history
