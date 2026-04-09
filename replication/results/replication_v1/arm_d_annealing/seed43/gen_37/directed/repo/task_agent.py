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
_response_cache: dict[str, tuple[str, list[dict]]] = {}

# Pre-compile regex patterns for JSON fuzzy extraction (performance optimization)
_TRAILING_COMMA_RE = re.compile(r',(\s*[}\]])')
_SINGLE_QUOTE_RE = re.compile(r"(?<!\\)'")
_COMMENT_RE = re.compile(r'//[^\n]*')
_BLOCK_COMMENT_RE = re.compile(r'/\*.*?\*/', re.DOTALL)


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
        try:
            results.append(json.loads(inner))
        except json.JSONDecodeError:
            continue
    return results or None


def _extract_json_fuzzy(text: str) -> list[dict] | None:
    """Fuzzy JSON extraction for when standard extraction fails.
    
    Enhanced to handle nested braces, common JSON errors, and
    attempts to fix common formatting issues before parsing.
    """
    results = []
    # Try to find JSON-like structures with curly braces
    brace_count = 0
    start_idx = None
    for i, char in enumerate(text):
        if char == '{':
            if brace_count == 0:
                start_idx = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start_idx is not None:
                try:
                    json_str = text[start_idx:i+1]
                    # Try to parse as-is first
                    try:
                        results.append(json.loads(json_str))
                    except json.JSONDecodeError:
                        # Try common fixes using pre-compiled patterns
                        # Fix 1: Remove trailing commas before closing braces/brackets
                        fixed = _TRAILING_COMMA_RE.sub(r'\1', json_str)
                        # Fix 2: Replace single quotes with double quotes (carefully)
                        fixed = _SINGLE_QUOTE_RE.sub('"', fixed)
                        # Fix 3: Remove comments
                        fixed = _COMMENT_RE.sub('', fixed)
                        fixed = _BLOCK_COMMENT_RE.sub('', fixed)
                        try:
                            results.append(json.loads(fixed))
                        except json.JSONDecodeError:
                            pass
                except Exception:
                    pass
                start_idx = None
    return results or None


def _compute_cache_key(inputs: dict, model: str) -> str:
    """Compute a cache key from inputs and model name.
    
    Uses MD5 hash of sorted JSON for deterministic keys.
    """
    cache_data = {
        "model": model,
        "inputs": {k: inputs.get(k, "") for k in ["problem", "solution", "grading_guidelines", "student_answer", "domain"]},
    }
    json_str = json.dumps(cache_data, sort_keys=True, separators=(',', ':'))
    return hashlib.md5(json_str.encode()).hexdigest()


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "", use_cache: bool = True) -> None:
        self.model = model
        self.log_fn = logger.info
        self.max_retries = 3
        self.use_cache = use_cache

    def _build_prompt(self, inputs: dict) -> str:
        """Build a structured prompt with chain-of-thought instructions."""
        domain = inputs.get("domain", "unknown")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        return f"""You are an expert mathematical grader for {domain} problems.

Your task is to evaluate a student's answer to a mathematical problem.

PROBLEM:
{problem}

OFFICIAL SOLUTION:
{solution}

GRADING GUIDELINES:
{grading_guidelines}

STUDENT'S ANSWER:
{student_answer}

INSTRUCTIONS:
1. First, analyze the student's answer step by step. Compare it to the official solution.
2. Check if the student has the correct final answer.
3. Verify if the student's reasoning is sound and follows logical steps.
4. Consider partial credit based on the grading guidelines.
5. Provide your final grade in the JSON format below.

Respond in JSON format with the following schema:
<json>
{{
    "reasoning": "Your detailed step-by-step analysis here",
    "response": "Your final grade/assessment here"
}}
</json>

IMPORTANT: The "response" field must contain ONLY the final grade as a single number (0-7) or a simple word like "Correct" or "Incorrect". Do not include any explanation, punctuation, or additional text in the response field."""

    def _validate_grade(self, prediction: str) -> tuple[bool, str]:
        """Validate that the prediction is a valid grade format.
        
        Enhanced to handle more edge cases including decimal grades,
        percentage formats, and grades with punctuation.
        
        Returns:
            (is_valid, cleaned_grade) tuple
        """
        if not prediction or prediction == "None":
            return False, "None"
        
        prediction = prediction.strip()
        
        # Remove common punctuation that might surround grades
        prediction_clean = prediction.strip(".!?,:;\"'()[]{}<>")
        
        # Check for numeric grades (0-7 for IMO problems)
        if prediction_clean.isdigit():
            grade = int(prediction_clean)
            if 0 <= grade <= 7:
                return True, str(grade)
            return False, "None"
        
        # Check for decimal grades (e.g., "3.5", "6.0")
        try:
            grade_float = float(prediction_clean)
            if 0 <= grade_float <= 7:
                # Round to nearest valid grade
                grade_int = round(grade_float)
                return True, str(min(7, max(0, grade_int)))
        except ValueError:
            pass
        
        # Check for common grade formats
        valid_non_numeric = ["correct", "incorrect", "partial", "full", "zero", 
                            "pass", "fail", "true", "false", "yes", "no", 
                            "accepted", "rejected", "valid", "invalid"]
        lower_pred = prediction_clean.lower()
        
        if lower_pred in valid_non_numeric:
            return True, prediction_clean
        
        # Check for fractional grades (e.g., "3/7", "5/7")
        if "/" in prediction_clean:
            parts = prediction_clean.split("/")
            if len(parts) == 2 and parts[0].strip().isdigit() and parts[1].strip().isdigit():
                numerator = int(parts[0].strip())
                denominator = int(parts[1].strip())
                if 0 <= numerator <= denominator and denominator <= 7:
                    return True, prediction_clean
        
        # Check for percentage grades (e.g., "50%", "100%")
        if "%" in prediction_clean:
            try:
                pct_str = prediction_clean.replace("%", "").strip()
                pct = float(pct_str)
                if 0 <= pct <= 100:
                    # Convert percentage to 0-7 scale
                    grade = round((pct / 100) * 7)
                    return True, str(min(7, max(0, grade)))
            except ValueError:
                pass
        
        # If it looks like a number but has extra text, try to extract
        numeric_match = re.search(r'\b([0-7])\b', prediction_clean)
        if numeric_match:
            return True, numeric_match.group(1)
        
        # Try to extract any number and validate it
        any_num_match = re.search(r'\b(\d+(?:\.\d+)?)\b', prediction_clean)
        if any_num_match:
            try:
                num = float(any_num_match.group(1))
                if 0 <= num <= 7:
                    return True, str(int(num))
            except ValueError:
                pass
        
        return False, "None"

    def _extract_prediction(self, msg_history: list[dict]) -> tuple[str, str]:
        """Extract prediction and reasoning from message history.
        
        Returns:
            (prediction, reasoning) tuple
        """
        if not msg_history:
            return "None", ""
        
        last_text = msg_history[-1].get("text", "")
        
        # Try standard extraction first
        extracted = _extract_jsons(last_text)
        if not extracted:
            # Try fuzzy extraction as fallback
            extracted = _extract_json_fuzzy(last_text)
        
        if not extracted:
            return "None", ""
        
        last_json = extracted[-1]
        prediction = last_json.get("response", "None")
        reasoning = last_json.get("reasoning", "")
        
        # Clean up prediction
        if isinstance(prediction, (int, float)):
            prediction = str(prediction)
        elif isinstance(prediction, str):
            prediction = prediction.strip()
        else:
            prediction = str(prediction)
        
        # Validate the grade format
        is_valid, cleaned_prediction = self._validate_grade(prediction)
        if not is_valid:
            self.log_fn(f"Warning: Invalid grade format '{prediction}', using 'None'")
        
        return cleaned_prediction, reasoning

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with retry logic.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Validate required inputs
        required_keys = ["problem", "solution", "grading_guidelines", "student_answer"]
        missing_keys = [k for k in required_keys if not inputs.get(k)]
        if missing_keys:
            self.log_fn(f"Warning: Missing required inputs: {missing_keys}")
        
        # Check cache for existing result
        if self.use_cache:
            cache_key = _compute_cache_key(inputs, self.model)
            if cache_key in _response_cache:
                cached_prediction, cached_history = _response_cache[cache_key]
                self.log_fn(f"Cache hit: returning cached prediction '{cached_prediction}'")
                return cached_prediction, cached_history
        
        instruction = self._build_prompt(inputs)
        
        prediction = "None"
        reasoning = ""
        msg_history = []
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=[],
                )
                
                prediction, reasoning = self._extract_prediction(msg_history)
                
                # Validate that we got a meaningful prediction
                if prediction != "None" and prediction.strip():
                    self.log_fn(f"Successfully extracted prediction: {prediction} (attempt {attempt + 1})")
                    if reasoning:
                        self.log_fn(f"Reasoning length: {len(reasoning)} chars")
                    break
                else:
                    self.log_fn(f"Attempt {attempt + 1}: Failed to extract valid prediction, retrying...")
                    # Add a hint for the next attempt
                    instruction += "\n\nIMPORTANT: Make sure to include the 'response' field in your JSON output."
                    
            except Exception as e:
                last_error = str(e)
                self.log_fn(f"Attempt {attempt + 1}: Error during LLM call: {e}")
                if attempt == self.max_retries - 1:
                    # Last attempt failed, return what we have
                    break
        
        # Log final result
        if prediction == "None" or not prediction.strip():
            if last_error:
                self.log_fn(f"Warning: Could not extract valid prediction after all retries. Last error: {last_error}")
            else:
                self.log_fn("Warning: Could not extract valid prediction after all retries")
        
        # Store successful result in cache
        if self.use_cache and prediction != "None" and prediction.strip():
            cache_key = _compute_cache_key(inputs, self.model)
            _response_cache[cache_key] = (str(prediction), msg_history)
            self.log_fn(f"Cached result for key {cache_key[:8]}...")
        
        return str(prediction), msg_history
