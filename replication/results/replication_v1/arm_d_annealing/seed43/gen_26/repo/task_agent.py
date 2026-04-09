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
        
        # Extract max grade from grading guidelines if available
        max_grade = "7"  # Default for IMO problems
        if grading_guidelines:
            import re
            grade_match = re.search(r'(\d+)\s*(?:points?|marks?)', grading_guidelines.lower())
            if grade_match:
                max_grade = grade_match.group(1)
        
        return f"""You are an expert mathematical grader for {domain} problems.

Your task is to evaluate a student's answer to a mathematical problem and assign an appropriate grade.

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
2. Check if the student has the correct final answer and award points accordingly.
3. Verify if the student's reasoning is sound and follows logical steps.
4. Consider partial credit based on the grading guidelines - even incomplete solutions may deserve partial points.
5. Look for any valid mathematical insights, correct intermediate steps, or alternative valid approaches.
6. Provide your final grade in the JSON format below.

IMPORTANT: The grade should be a number between 0 and {max_grade} (inclusive), representing the points earned.

Respond in JSON format with the following schema:
<json>
{{
    "reasoning": "Your detailed step-by-step analysis here. Explain what the student did correctly and incorrectly, and justify the grade assigned.",
    "response": "Your final grade as a number (e.g., 7, 5, 3, 0)"
}}
</json>

The "response" field must contain ONLY a numeric grade between 0 and {max_grade}. Do not include any text, explanations, or punctuation - just the number."""

    def _validate_grade(self, prediction: str, max_grade: int = 7) -> tuple[bool, str]:
        """Validate that the prediction is a valid grade format.
        
        Enhanced to handle more edge cases including decimal grades,
        percentage formats, and grades with punctuation.
        
        Args:
            prediction: The grade string to validate
            max_grade: Maximum allowed grade (default 7 for IMO problems)
        
        Returns:
            (is_valid, cleaned_grade) tuple
        """
        if not prediction or prediction == "None":
            return False, "None"
        
        prediction = prediction.strip()
        
        # Remove common punctuation and whitespace that might surround grades
        prediction_clean = prediction.strip(".!?,:;\"'()[]{}<> ")
        
        # Handle empty string after cleaning
        if not prediction_clean:
            return False, "None"
        
        # Check for numeric grades (0 to max_grade)
        if prediction_clean.isdigit():
            grade = int(prediction_clean)
            if 0 <= grade <= max_grade:
                return True, str(grade)
            # If grade is too high, cap it at max_grade
            if grade > max_grade:
                return True, str(max_grade)
            return False, "None"
        
        # Check for decimal grades (e.g., "3.5", "6.0")
        try:
            grade_float = float(prediction_clean)
            if 0 <= grade_float <= max_grade:
                # Round to nearest valid grade
                grade_int = round(grade_float)
                return True, str(min(max_grade, max(0, grade_int)))
            elif grade_float > max_grade:
                # Cap at max_grade if too high
                return True, str(max_grade)
        except ValueError:
            pass
        
        # Check for common grade formats and map to numeric
        lower_pred = prediction_clean.lower()
        grade_mappings = {
            "correct": "7", "full": "7", "complete": "7", "perfect": "7",
            "incorrect": "0", "wrong": "0", "zero": "0", "none": "0",
            "partial": "3", "incomplete": "3", "half": "3",
            "pass": "4", "fail": "0",
            "true": "7", "false": "0",
            "yes": "7", "no": "0",
            "accepted": "7", "rejected": "0",
            "valid": "7", "invalid": "0",
        }
        
        if lower_pred in grade_mappings:
            return True, grade_mappings[lower_pred]
        
        # Check for fractional grades (e.g., "3/7", "5/7")
        if "/" in prediction_clean:
            parts = prediction_clean.split("/")
            if len(parts) == 2:
                try:
                    numerator = float(parts[0].strip())
                    denominator = float(parts[1].strip())
                    if denominator > 0 and 0 <= numerator <= denominator:
                        # Convert to 0-max_grade scale
                        grade = round((numerator / denominator) * max_grade)
                        return True, str(min(max_grade, max(0, grade)))
                except ValueError:
                    pass
        
        # Check for percentage grades (e.g., "50%", "100%")
        if "%" in prediction_clean:
            try:
                pct_str = prediction_clean.replace("%", "").strip()
                pct = float(pct_str)
                if 0 <= pct <= 100:
                    # Convert percentage to 0-max_grade scale
                    grade = round((pct / 100) * max_grade)
                    return True, str(min(max_grade, max(0, grade)))
            except ValueError:
                pass
        
        # Try to extract a single digit 0-max_grade from the text
        numeric_match = re.search(r'\b([0-' + str(max_grade) + r'])\b', prediction_clean)
        if numeric_match:
            return True, numeric_match.group(1)
        
        # Try to extract any number and validate it
        any_num_match = re.search(r'\b(\d+(?:\.\d+)?)\b', prediction_clean)
        if any_num_match:
            try:
                num = float(any_num_match.group(1))
                if 0 <= num <= max_grade:
                    return True, str(int(num))
                elif num > max_grade:
                    return True, str(max_grade)
            except ValueError:
                pass
        
        return False, "None"

    def _extract_prediction(self, msg_history: list[dict], max_grade: int = 7) -> tuple[str, str]:
        """Extract prediction and reasoning from message history.
        
        Args:
            msg_history: List of message dictionaries
            max_grade: Maximum allowed grade (default 7 for IMO problems)
        
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
        is_valid, cleaned_prediction = self._validate_grade(prediction, max_grade)
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
        
        # Extract max grade from grading guidelines
        grading_guidelines = inputs.get("grading_guidelines", "")
        max_grade = 7  # Default for IMO problems
        if grading_guidelines:
            grade_match = re.search(r'(\d+)\s*(?:points?|marks?)', grading_guidelines.lower())
            if grade_match:
                max_grade = int(grade_match.group(1))
        
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
                
                prediction, reasoning = self._extract_prediction(msg_history, max_grade)
                
                # Validate that we got a meaningful prediction
                if prediction != "None" and prediction.strip():
                    self.log_fn(f"Successfully extracted prediction: {prediction} (attempt {attempt + 1})")
                    if reasoning:
                        self.log_fn(f"Reasoning length: {len(reasoning)} chars")
                    # Cache the result
                    if self.use_cache:
                        problem_hash = _compute_problem_hash(inputs)
                        _problem_cache[problem_hash] = (prediction, reasoning)
                    break
                else:
                    self.log_fn(f"Attempt {attempt + 1}: Failed to extract valid prediction, retrying...")
                    # Add a hint for the next attempt
                    instruction += f"\n\nIMPORTANT: Make sure to include the 'response' field in your JSON output with a numeric grade between 0 and {max_grade}."
                    
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
