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


def _extract_json_with_retry(text: str, max_retries: int = 3) -> list[dict] | None:
    """Extract JSON with multiple fallback strategies.
    
    Tries multiple extraction methods:
    1. Standard <json>...</json> blocks
    2. JSON code blocks ```json...```
    3. Raw JSON objects in text
    """
    # Try standard extraction first
    results = _extract_jsons(text)
    if results:
        return results
    
    # Try JSON code blocks
    results = []
    json_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    matches = re.findall(json_block_pattern, text, re.DOTALL)
    for match in matches:
        try:
            results.append(json.loads(match.strip()))
        except json.JSONDecodeError:
            continue
    
    if results:
        return results
    
    # Try to find raw JSON objects (objects with curly braces)
    # Use a more robust pattern that handles nested braces
    results = []
    # Find all potential JSON objects by tracking brace depth
    potential_objects = []
    start_indices = []
    brace_depth = 0
    in_string = False
    escape_next = False
    
    for i, char in enumerate(text):
        if escape_next:
            escape_next = False
            continue
        if char == '\\':
            escape_next = True
            continue
        if char == '"' and not in_string:
            in_string = True
        elif char == '"' and in_string:
            in_string = False
        elif not in_string:
            if char == '{':
                if brace_depth == 0:
                    start_indices.append(i)
                brace_depth += 1
            elif char == '}':
                brace_depth -= 1
                if brace_depth == 0 and start_indices:
                    start = start_indices.pop()
                    potential_objects.append(text[start:i+1])
    
    for obj_str in potential_objects:
        try:
            results.append(json.loads(obj_str.strip()))
        except json.JSONDecodeError:
            continue
    
    return results or None


class TaskAgent:
    """Task agent that solves IMO grading problems with enhanced reasoning and caching."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.max_retries = 3
        self._cache: dict[str, tuple[str, list[dict]]] = {}
        self._cache_hits = 0
        self._cache_misses = 0

    def _get_cache_key(self, inputs: dict) -> str:
        """Generate a cache key from inputs."""
        # Create deterministic key from relevant fields
        key_data = {
            "domain": inputs.get("domain", ""),
            "problem": inputs.get("problem", ""),
            "solution": inputs.get("solution", ""),
            "grading_guidelines": inputs.get("grading_guidelines", ""),
            "student_answer": inputs.get("student_answer", ""),
            "model": self.model,
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()[:32]

    def get_cache_stats(self) -> dict:
        """Return cache statistics."""
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "hit_rate": hit_rate,
            "size": len(self._cache),
        }

    def _validate_inputs(self, inputs: dict) -> tuple[bool, str]:
        """Validate that all required inputs are present and non-empty.
        
        Returns:
            (is_valid, error_message)
        """
        required_fields = ["domain", "problem", "solution", "grading_guidelines", "student_answer"]
        
        for field in required_fields:
            if field not in inputs:
                return False, f"Missing required field: {field}"
            if not inputs[field] or not str(inputs[field]).strip():
                return False, f"Empty required field: {field}"
        
        return True, ""

    def _build_grading_prompt(self, inputs: dict) -> str:
        """Build a comprehensive prompt for the grading task."""
        domain = inputs.get("domain", "")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        # Truncate very long inputs to prevent token overflow
        max_len = 8000
        problem = problem[:max_len] + "..." if len(problem) > max_len else problem
        solution = solution[:max_len] + "..." if len(solution) > max_len else solution
        student_answer = student_answer[:max_len] + "..." if len(student_answer) > max_len else student_answer
        
        return f"""You are an expert grader for {domain} problems. Your task is to evaluate a student's answer against the correct solution and grading guidelines.

## Problem
{problem}

## Correct Solution
{solution}

## Grading Guidelines
{grading_guidelines}

## Student's Answer
{student_answer}

## Instructions
1. Carefully analyze the student's answer step by step
2. Compare it against the correct solution
3. Apply the grading guidelines strictly
4. Provide your reasoning before giving the final grade
5. Respond in JSON format with the following schema:

<json>
{{
    "reasoning": "Your detailed step-by-step analysis and reasoning",
    "response": "One of: Correct, Partial, Almost, or Incorrect"
}}
</json>

## Grade Definitions
- **Correct**: The student's answer is completely correct and matches the solution
- **Partial**: The student made significant progress but didn't complete the proof/solution
- **Almost**: The solution is nearly complete but has minor errors or gaps
- **Incorrect**: The answer is wrong or makes no meaningful progress

Important: 
- The "response" field MUST contain exactly one of: "Correct", "Partial", "Almost", or "Incorrect" (case-sensitive)
- Use the "reasoning" field to show your work
- Be precise and follow the grading guidelines exactly
- If the student answer is empty or nonsensical, respond with "Incorrect"
"""

    def _normalize_grade(self, prediction: str) -> str:
        """Normalize a grade prediction to one of the standard labels.
        
        Args:
            prediction: Raw prediction string
            
        Returns:
            Normalized grade: "Correct", "Partial", "Almost", "Incorrect", or "None"
        """
        if not prediction or not isinstance(prediction, str):
            return "None"
        
        # Clean up the prediction
        cleaned = prediction.strip().lower()
        
        # Map common variations to standard labels
        grade_map = {
            # Correct variations
            "correct": "Correct",
            "right": "Correct",
            "true": "Correct",
            "yes": "Correct",
            "full": "Correct",
            "complete": "Correct",
            "1": "Correct",
            # Partial variations
            "partial": "Partial",
            "partly": "Partial",
            "incomplete": "Partial",
            "half": "Partial",
            "some": "Partial",
            # Almost variations
            "almost": "Almost",
            "nearly": "Almost",
            "close": "Almost",
            "minor errors": "Almost",
            "minor": "Almost",
            # Incorrect variations
            "incorrect": "Incorrect",
            "wrong": "Incorrect",
            "false": "Incorrect",
            "no": "Incorrect",
            "none": "Incorrect",
            "0": "Incorrect",
            "fail": "Incorrect",
            "failed": "Incorrect",
            "error": "Incorrect",
        }
        
        # Check for exact matches first
        if cleaned in grade_map:
            return grade_map[cleaned]
        
        # Check for partial matches
        for key, value in grade_map.items():
            if key in cleaned:
                return value
        
        # If no match found, return the original with first letter capitalized
        # as a fallback
        return prediction.strip().capitalize() if prediction.strip() else "None"

    def _extract_grade_from_text(self, text: str) -> str | None:
        """Extract a grade from plain text when JSON extraction fails.
        
        Looks for grade keywords in the text.
        
        Returns:
            Grade string or None if not found
        """
        if not text:
            return None
        
        text_lower = text.lower()
        
        # Look for explicit grade statements
        grade_patterns = [
            ("correct", ["grade: correct", "final grade: correct", "is correct", 
                        "answer is correct", "solution is correct", "the correct answer"]),
            ("partial", ["grade: partial", "final grade: partial", "is partial",
                        "partial credit", "partially correct"]),
            ("almost", ["grade: almost", "final grade: almost", "is almost",
                       "almost correct", "nearly correct"]),
            ("incorrect", ["grade: incorrect", "final grade: incorrect", "is incorrect",
                          "answer is incorrect", "solution is incorrect", "is wrong"]),
        ]
        
        for grade, patterns in grade_patterns:
            for pattern in patterns:
                if pattern in text_lower:
                    return grade.capitalize()
        
        # Look for standalone grade words at the end of sentences or lines
        lines = text.split('\n')
        for line in reversed(lines):
            line_lower = line.lower().strip()
            for grade in ["correct", "partial", "almost", "incorrect"]:
                if line_lower == grade or line_lower.endswith(f" {grade}") or line_lower.endswith(f": {grade}"):
                    return grade.capitalize()
        
        return None

    def _try_extract_prediction(self, text: str) -> tuple[str, str | None]:
        """Try to extract prediction from response text.
        
        Returns:
            (prediction, reasoning)
        """
        if not text or not text.strip():
            return "None", None
            
        try:
            extracted = _extract_json_with_retry(text)
            if extracted:
                # Try to find the best JSON object with both response and reasoning
                best_match = None
                for obj in extracted:
                    if isinstance(obj, dict) and "response" in obj:
                        best_match = obj
                        break
                
                # If no object with "response" found, use the last one
                if best_match is None:
                    best_match = extracted[-1]
                
                if isinstance(best_match, dict):
                    prediction = best_match.get("response", "None")
                    reasoning = best_match.get("reasoning")
                    
                    # Handle numeric predictions
                    if prediction is None:
                        prediction = "None"
                    elif isinstance(prediction, (int, float)):
                        prediction = str(prediction)
                    elif not isinstance(prediction, str):
                        prediction = str(prediction)
                    
                    # Normalize the grade to standard labels
                    normalized = self._normalize_grade(prediction)
                    return normalized, reasoning
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
        
        # Fallback: try to extract grade from plain text
        try:
            grade = self._extract_grade_from_text(text)
            if grade:
                self.log_fn(f"Extracted grade from plain text: {grade}")
                return grade, None
        except Exception as e:
            self.log_fn(f"Error in text-based extraction: {e}")
        
        return "None", None

    def _validate_prediction(self, prediction: str) -> str:
        """Validate and normalize the final prediction.
        
        Ensures the prediction is one of the valid grade labels.
        
        Args:
            prediction: Raw prediction string
            
        Returns:
            Validated prediction string
        """
        valid_grades = {"Correct", "Partial", "Almost", "Incorrect"}
        
        if not prediction:
            return "None"
        
        # Normalize to standard format
        normalized = prediction.strip()
        
        # Check if it's already valid
        if normalized in valid_grades:
            return normalized
        
        # Try to normalize using the grade map
        normalized_lower = normalized.lower()
        grade_map = {
            "correct": "Correct",
            "partial": "Partial",
            "almost": "Almost",
            "incorrect": "Incorrect",
        }
        
        if normalized_lower in grade_map:
            return grade_map[normalized_lower]
        
        # Check for partial matches
        for valid in valid_grades:
            if valid.lower() in normalized_lower:
                return valid
        
        # If no valid grade found, return None
        self.log_fn(f"Warning: Could not validate prediction '{prediction}', returning None")
        return "None"

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with retry logic and caching.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Validate inputs first
        is_valid, error_msg = self._validate_inputs(inputs)
        if not is_valid:
            self.log_fn(f"Input validation failed: {error_msg}")
            return f"Error: {error_msg}", []
        
        # Check cache first
        cache_key = self._get_cache_key(inputs)
        if cache_key in self._cache:
            self._cache_hits += 1
            self.log_fn(f"Cache hit for key {cache_key[:8]}...")
            return self._cache[cache_key]
        
        self._cache_misses += 1
        
        instruction = self._build_grading_prompt(inputs)
        msg_history = []
        prediction = "None"
        
        # Try with retries
        for attempt in range(self.max_retries):
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=msg_history if attempt > 0 else [],
                )
                
                # Extract prediction from the last assistant message
                text = ""
                for msg in reversed(msg_history):
                    if msg.get("role") == "assistant":
                        text = msg.get("text", "")
                        break
                
                pred, reasoning = self._try_extract_prediction(text)
                
                # Validate the prediction
                validated_pred = self._validate_prediction(pred)
                
                if validated_pred != "None":
                    prediction = validated_pred
                    if reasoning:
                        self.log_fn(f"Grading reasoning: {reasoning[:200]}...")
                    break
                
                # If extraction failed, add a follow-up message asking for proper format
                if attempt < self.max_retries - 1:
                    instruction = 'Please respond in the required JSON format with \'response\' and \'reasoning\' fields. The \'response\' field MUST contain exactly one of these four values: "Correct", "Partial", "Almost", or "Incorrect". Example: <json>{"reasoning": "The student made significant progress but did not complete the proof...", "response": "Partial"}</json>'
                    
            except Exception as e:
                self.log_fn(f"Attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    break
        
        # Final validation
        prediction = self._validate_prediction(prediction)
        
        result = (str(prediction), msg_history)
        
        # Store in cache
        self._cache[cache_key] = result
        
        return result
