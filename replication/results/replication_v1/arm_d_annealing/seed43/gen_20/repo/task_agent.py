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

# Simple in-memory cache for similar problems
_problem_cache: dict[str, tuple[str, str]] = {}


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
                        # Try common fixes
                        # Fix 1: Remove trailing commas before closing braces/brackets
                        fixed = re.sub(r',(\s*[}\]])', r'\1', json_str)
                        # Fix 2: Replace single quotes with double quotes (carefully)
                        fixed = re.sub(r"(?<!\\)'", '"', fixed)
                        # Fix 3: Remove comments
                        fixed = re.sub(r'//[^\n]*', '', fixed)
                        fixed = re.sub(r'/\*.*?\*/', '', fixed, flags=re.DOTALL)
                        # Fix 4: Handle unescaped newlines in strings
                        fixed = re.sub(r'(?<=")\n(?=")', '\\n', fixed)
                        # Fix 5: Handle multiple consecutive newlines
                        fixed = re.sub(r'\n+', ' ', fixed)
                        try:
                            results.append(json.loads(fixed))
                        except json.JSONDecodeError:
                            pass
                except Exception:
                    pass
                start_idx = None
    return results or None


def _compute_problem_hash(inputs: dict) -> str:
    """Compute a hash for problem caching based on key fields."""
    key_parts = [
        inputs.get("problem", ""),
        inputs.get("solution", ""),
        inputs.get("grading_guidelines", ""),
        inputs.get("student_answer", ""),
    ]
    key_str = "||".join(key_parts)
    return hashlib.md5(key_str.encode()).hexdigest()


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "", use_cache: bool = True) -> None:
        self.model = model
        self.log_fn = logger.info
        self.max_retries = 3
        self.use_cache = use_cache
        self._cache_hits = 0
        self._cache_misses = 0

    def _build_prompt(self, inputs: dict) -> str:
        """Build a structured prompt with chain-of-thought instructions."""
        domain = inputs.get("domain", "unknown")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        return f"""You are an expert mathematical grader for {domain} problems.

Your task is to evaluate a student's answer to a mathematical problem with precision and consistency.

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
2. Check if the student has the correct final answer. For IMO problems, the final answer must match exactly.
3. Verify if the student's reasoning is sound and follows logical steps. Look for:
   - Correct application of mathematical theorems and formulas
   - Logical flow from one step to the next
   - Proper justification of key claims
   - No circular reasoning or logical gaps
4. Consider partial credit based on the grading guidelines:
   - Full marks (7): Complete, correct solution with proper reasoning
   - Partial marks (1-6): Progress toward solution, partial results, or minor errors
   - No marks (0): No meaningful progress or completely wrong approach
5. Be consistent with IMO grading standards - award points only for demonstrated mathematical progress.
6. Provide your final grade in the JSON format below.

Respond in JSON format with the following schema:
<json>
{{
    "reasoning": "Your detailed step-by-step analysis here. Explain what the student did correctly and incorrectly, and justify your grade.",
    "response": "Your final grade here - a single number 0-7 for IMO problems"
}}
</json>

IMPORTANT: The "response" field must contain ONLY a single integer from 0 to 7 (for IMO problems). Do not include any text, explanation, or punctuation - just the number."""

    def _validate_grade(self, prediction: str) -> tuple[bool, str]:
        """Validate that the prediction is a valid IMO grade format (0-7).
        
        IMO problems use a 0-7 grading scale where:
        - 0: No meaningful progress
        - 1-6: Partial credit based on progress
        - 7: Complete, correct solution
        
        Returns:
            (is_valid, cleaned_grade) tuple where cleaned_grade is 0-7 or "None"
        """
        if not prediction or prediction == "None":
            return False, "None"
        
        prediction = prediction.strip()
        
        # Remove common punctuation and whitespace
        prediction_clean = prediction.strip(".!?,:;\"'()[]{}<> ")
        
        # Primary check: numeric grades 0-7 (most common for IMO)
        if prediction_clean.isdigit():
            grade = int(prediction_clean)
            if 0 <= grade <= 7:
                return True, str(grade)
            # Clamp out-of-range values
            if grade < 0:
                return True, "0"
            if grade > 7:
                return True, "7"
        
        # Check for decimal grades (e.g., "3.5", "6.0") - round to nearest integer
        try:
            grade_float = float(prediction_clean)
            if 0 <= grade_float <= 7:
                grade_int = round(grade_float)
                return True, str(min(7, max(0, grade_int)))
            # Clamp out-of-range
            if grade_float < 0:
                return True, "0"
            if grade_float > 7:
                return True, "7"
        except ValueError:
            pass
        
        # Map common text grades to IMO scale
        grade_mapping = {
            "correct": "7",
            "full": "7",
            "complete": "7",
            "right": "7",
            "true": "7",
            "yes": "7",
            "accepted": "7",
            "valid": "7",
            "incorrect": "0",
            "wrong": "0",
            "false": "0",
            "no": "0",
            "rejected": "0",
            "invalid": "0",
            "fail": "0",
            "zero": "0",
            "none": "0",
            "partial": "3",  # Middle of partial credit range
            "pass": "4",     # Just above passing threshold
        }
        
        lower_pred = prediction_clean.lower()
        if lower_pred in grade_mapping:
            return True, grade_mapping[lower_pred]
        
        # Check for fractional grades (e.g., "3/7", "5/7") - extract numerator
        if "/" in prediction_clean:
            parts = prediction_clean.split("/")
            if len(parts) == 2:
                try:
                    numerator = int(parts[0].strip())
                    denominator = int(parts[1].strip())
                    if denominator > 0 and 0 <= numerator <= denominator:
                        # Scale to 0-7 range
                        scaled = round((numerator / denominator) * 7)
                        return True, str(min(7, max(0, scaled)))
                except ValueError:
                    pass
        
        # Check for percentage grades (e.g., "50%", "100%")
        if "%" in prediction_clean:
            try:
                pct_str = prediction_clean.replace("%", "").strip()
                pct = float(pct_str)
                if 0 <= pct <= 100:
                    grade = round((pct / 100) * 7)
                    return True, str(min(7, max(0, grade)))
            except ValueError:
                pass
        
        # Extract single digit 0-7 from text (e.g., "Grade: 5" or "The answer is 3")
        numeric_match = re.search(r'\b([0-7])\b', prediction_clean)
        if numeric_match:
            return True, numeric_match.group(1)
        
        # Try to extract any number and clamp to valid range
        any_num_match = re.search(r'-?\b(\d+(?:\.\d+)?)\b', prediction_clean)
        if any_num_match:
            try:
                num = float(any_num_match.group(1))
                if num < 0:
                    return True, "0"
                if num > 7:
                    return True, "7"
                return True, str(int(num))
            except ValueError:
                pass
        
        return False, "None"

    def _extract_prediction(self, msg_history: list[dict]) -> tuple[str, str]:
        """Extract prediction and reasoning from message history.
        
        Enhanced to handle multiple JSON extraction strategies and
        validate grades according to IMO standards.
        
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
            # Last resort: try to find any number that looks like a grade
            # This handles cases where the model outputs just a number
            number_match = re.search(r'\b([0-7])\b', last_text.strip())
            if number_match:
                return number_match.group(1), "Grade extracted from raw text"
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
        """Run the task agent on a single problem with retry logic and caching.

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
        
        # Check cache first
        if self.use_cache:
            problem_hash = _compute_problem_hash(inputs)
            if problem_hash in _problem_cache:
                cached_prediction, cached_reasoning = _problem_cache[problem_hash]
                self._cache_hits += 1
                self.log_fn(f"Cache hit! Using cached prediction: {cached_prediction} (hits: {self._cache_hits}, misses: {self._cache_misses})")
                # Return a synthetic msg_history with the cached result
                msg_history = [
                    {"role": "user", "text": self._build_prompt(inputs)},
                    {"role": "assistant", "text": f"<json>\n{{'reasoning': '{cached_reasoning}', 'response': '{cached_prediction}'}}\n</json>"}
                ]
                return cached_prediction, msg_history
            self._cache_misses += 1
        
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
                    self.log_fn(f"Successfully extracted prediction: {prediction} (attempt {attempt + 1}/{self.max_retries})")
                    if reasoning:
                        self.log_fn(f"Reasoning length: {len(reasoning)} chars")
                    # Cache the result
                    if self.use_cache:
                        problem_hash = _compute_problem_hash(inputs)
                        _problem_cache[problem_hash] = (prediction, reasoning)
                    break
                else:
                    self.log_fn(f"Attempt {attempt + 1}/{self.max_retries}: Failed to extract valid prediction, retrying...")
                    # Add increasingly specific hints for the next attempt
                    if attempt == 0:
                        instruction += "\n\nIMPORTANT: You must include a 'response' field in your JSON output with a grade from 0-7."
                    elif attempt == 1:
                        instruction += "\n\nCRITICAL: The response field must contain ONLY a single digit from 0 to 7. Example: \"response\": \"5\""
                    else:
                        instruction += "\n\nFINAL ATTEMPT: Output ONLY this exact format:\n<json>\n{\"reasoning\": \"your analysis\", \"response\": \"5\"}\n</json>"
                    
            except Exception as e:
                last_error = str(e)
                self.log_fn(f"Attempt {attempt + 1}/{self.max_retries}: Error during LLM call: {e}")
                if attempt == self.max_retries - 1:
                    # Last attempt failed, return what we have
                    break
        
        # Log final result
        if prediction == "None" or not prediction.strip():
            if last_error:
                self.log_fn(f"Warning: Could not extract valid prediction after {self.max_retries} retries. Last error: {last_error}")
            else:
                self.log_fn(f"Warning: Could not extract valid prediction after {self.max_retries} retries")
        
        return str(prediction), msg_history
    
    def get_cache_stats(self) -> dict:
        """Return cache statistics."""
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "total": total,
            "hit_rate": hit_rate,
            "cache_size": len(_problem_cache),
        }
    
    def clear_cache(self) -> None:
        """Clear the problem cache."""
        global _problem_cache
        _problem_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
