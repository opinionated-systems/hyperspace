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

# Pre-compiled regex for markdown JSON blocks
_MD_JSON_PATTERN = re.compile(r'```(?:json)?\s*([\s\S]*?)```')


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks.
    
    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    Also handles markdown code blocks with json language specifier.
    """
    results = []
    search_from = 0
    
    # First try explicit <json> tags
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
            # Try to extract JSON from within the content
            nested = _extract_brace_json(inner)
            if nested:
                results.extend(nested)
            continue
    
    # Also try markdown-style ```json blocks
    md_matches = _MD_JSON_PATTERN.findall(text)
    for match in md_matches:
        try:
            results.append(json.loads(match.strip()))
        except json.JSONDecodeError:
            # Try to extract JSON objects from within
            nested = _extract_brace_json(match)
            if nested:
                results.extend(nested)
            continue
    
    return results or None


def _extract_brace_json(text: str) -> list[dict] | None:
    """Extract JSON objects by tracking brace balance.
    
    More robust than regex for nested structures.
    """
    results = []
    brace_count = 0
    start_idx = -1
    
    for i, char in enumerate(text):
        if char == '{':
            if brace_count == 0:
                start_idx = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start_idx != -1:
                try:
                    obj = json.loads(text[start_idx:i+1])
                    if isinstance(obj, dict):
                        results.append(obj)
                except json.JSONDecodeError:
                    pass
                start_idx = -1
    
    return results or None


def _extract_markdown_json(text: str) -> list[dict] | None:
    """Extract JSON from markdown code blocks."""
    results = []
    # Match ```json or ``` blocks
    pattern = r'```(?:json)?\s*([\s\S]*?)```'
    matches = re.findall(pattern, text)
    
    for match in matches:
        # Try parsing the whole block first
        try:
            obj = json.loads(match.strip())
            if isinstance(obj, dict):
                results.append(obj)
                continue
        except json.JSONDecodeError:
            pass
        
        # If that fails, try extracting JSON objects from within
        nested = _extract_brace_json(match)
        if nested:
            results.extend(nested)
    
    return results or None


def _extract_all_json(text: str) -> list[dict]:
    """Extract all possible JSON objects using multiple strategies.
    
    Returns combined results from all extraction methods.
    """
    all_results = []
    seen = set()
    
    # Helper to add unique results
    def add_unique(results: list[dict] | None) -> None:
        if not results:
            return
        for obj in results:
            # Use string representation for deduplication
            obj_str = json.dumps(obj, sort_keys=True)
            if obj_str not in seen:
                seen.add(obj_str)
                all_results.append(obj)
    
    # Try all extraction methods
    add_unique(_extract_jsons(text))
    add_unique(_extract_markdown_json(text))
    add_unique(_extract_brace_json(text))
    
    return all_results


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    # Priority order for extracting prediction from JSON fields
    PREDICTION_FIELDS = [
        "response", "grade", "answer", "result", 
        "evaluation", "prediction", "score", "verdict"
    ]
    
    # Valid prediction values for normalization
    VALID_CORRECT = {"correct", "1", "true", "yes", "right", "valid", "pass", "success"}
    VALID_INCORRECT = {"incorrect", "0", "false", "no", "wrong", "invalid", "fail", "failure"}
    VALID_PARTIAL = {"partial", "0.5", "half", "incomplete", "partially correct"}

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def _normalize_prediction(self, prediction: str) -> str:
        """Normalize prediction to standard format.
        
        Args:
            prediction: Raw prediction string
            
        Returns:
            Normalized prediction: "Correct", "Incorrect", "Partial", or original
        """
        if not prediction or not isinstance(prediction, str):
            return "None"
        
        pred_lower = prediction.strip().lower()
        
        # Check for correct answers
        if pred_lower in self.VALID_CORRECT or pred_lower.startswith("correct"):
            return "Correct"
        
        # Check for incorrect answers
        if pred_lower in self.VALID_INCORRECT or pred_lower.startswith("incorrect"):
            return "Incorrect"
        
        # Check for partial credit
        if pred_lower in self.VALID_PARTIAL or pred_lower.startswith("partial"):
            return "Partial"
        
        # Try to extract numeric score
        try:
            # Check if it's a numeric value
            num_val = float(prediction.strip())
            if num_val >= 0.8:
                return "Correct"
            elif num_val <= 0.2:
                return "Incorrect"
            else:
                return "Partial"
        except (ValueError, TypeError):
            pass
        
        # Return original if no normalization applied
        return prediction.strip()

    def _validate_prediction(self, prediction: str) -> bool:
        """Validate that prediction is a meaningful grading result.
        
        Args:
            prediction: Prediction string to validate
            
        Returns:
            True if prediction is valid, False otherwise
        """
        if not prediction or prediction == "None":
            return False
        
        pred_lower = prediction.lower()
        
        # Check against valid values
        all_valid = self.VALID_CORRECT | self.VALID_INCORRECT | self.VALID_PARTIAL
        
        if pred_lower in all_valid:
            return True
        
        # Check for common prefixes
        for valid in all_valid:
            if pred_lower.startswith(valid):
                return True
        
        # Check for numeric values
        try:
            float(prediction.strip())
            return True
        except (ValueError, TypeError):
            pass
        
        return False

    def _extract_prediction(self, text: str) -> str:
        """Extract prediction from text using all available JSON extraction methods.
        
        Args:
            text: The text to extract prediction from
            
        Returns:
            The extracted prediction string, or "None" if extraction fails
        """
        # Try all extraction methods
        extracted = _extract_all_json(text)
        
        if not extracted:
            self.log_fn("No JSON objects found in response")
            return "None"
        
        # Use the last JSON object (most likely to be the final answer)
        last_json = extracted[-1]
        
        # Try known field names in priority order
        for field in self.PREDICTION_FIELDS:
            if field in last_json:
                value = last_json[field]
                if isinstance(value, str):
                    normalized = self._normalize_prediction(value)
                    if self._validate_prediction(normalized):
                        return normalized
                elif isinstance(value, (int, float, bool)):
                    normalized = self._normalize_prediction(str(value))
                    if self._validate_prediction(normalized):
                        return normalized
        
        # If no known field, look for any string or numeric value
        for key, value in last_json.items():
            if isinstance(value, str):
                normalized = self._normalize_prediction(value)
                if self._validate_prediction(normalized):
                    return normalized
            elif isinstance(value, (int, float)):
                normalized = self._normalize_prediction(str(value))
                if self._validate_prediction(normalized):
                    return normalized
        
        # Final fallback: if JSON is empty or has no extractable values, return "None"
        self.log_fn(f"Could not extract prediction from JSON: {last_json}")
        return "None"

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Validate inputs
        if not isinstance(inputs, dict):
            self.log_fn(f"Error: inputs must be a dict, got {type(inputs)}")
            return "None", []
        
        # Extract task components for better prompting
        domain = inputs.get("domain", "Unknown")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        # Validate required fields
        if not problem:
            self.log_fn("Warning: problem field is empty")
        if not student_answer:
            self.log_fn("Warning: student_answer field is empty")

        instruction = f"""You are an expert mathematical grader for International Mathematical Olympiad (IMO) problems.

Your task is to evaluate a student's answer to a mathematics problem and provide a grade.

DOMAIN: {domain}

PROBLEM:
{problem}

OFFICIAL SOLUTION:
{solution}

GRADING GUIDELINES:
{grading_guidelines}

STUDENT'S ANSWER:
{student_answer}

Think step by step:
1. Analyze what the problem is asking and identify key concepts
2. Review the official solution approach and identify critical steps
3. Compare the student's answer to the official solution - check for:
   - Correctness of the final answer
   - Validity of the reasoning process
   - Completeness of the solution
   - Mathematical rigor and clarity
4. Check if the student followed the grading guidelines precisely
5. Determine the appropriate grade based on the guidelines

GRADING RUBRIC:
- "Correct" or "1": The student's answer is fully correct, complete, and follows proper mathematical reasoning
- "Incorrect" or "0": The student's answer is wrong, incomplete, or contains critical errors
- "Partial": The student made progress but didn't fully solve the problem (use only when guidelines allow)

IMPORTANT: Your response must be valid JSON wrapped in <json> tags. Do not include any text outside the JSON tags.

Respond in this exact format:
<json>
{{
    "reasoning": "Your detailed chain-of-thought analysis here. Explain your evaluation process clearly.",
    "response": "The final grade/prediction (e.g., 'Correct', 'Incorrect', 'Partial', or a numeric score)"
}}
</json>"""

        # Try up to 3 times to get a valid JSON response
        max_retries = 3
        prediction = "None"
        msg_history = []
        last_error = None
        
        for attempt in range(max_retries):
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=[],
                )

                # Extract prediction from the last assistant message
                if msg_history and len(msg_history) > 0:
                    last_message = msg_history[-1].get("text", "")
                    
                    # Log the raw response for debugging
                    self.log_fn(f"Attempt {attempt + 1}: Raw response length: {len(last_message)}")
                    
                    prediction = self._extract_prediction(last_message)
                    
                    # If we got a valid prediction (not "None"), break the retry loop
                    if prediction != "None":
                        self.log_fn(f"Attempt {attempt + 1}: Successfully extracted prediction: {prediction}")
                        break
                    
                    last_error = "Failed to extract valid prediction from response"
                    
                    # If this wasn't the last attempt, add feedback for retry
                    if attempt < max_retries - 1:
                        self.log_fn(f"Attempt {attempt + 1}: {last_error}, retrying...")
                        instruction += f"\n\nNOTE: Your previous response did not contain valid JSON in <json> tags or the prediction was invalid. Please ensure your response follows the exact format specified above. The response field must contain one of: 'Correct', 'Incorrect', 'Partial', or a numeric score."
                else:
                    last_error = "Empty message history"
                    self.log_fn(f"Attempt {attempt + 1}: {last_error}")
                    
            except Exception as e:
                last_error = f"{type(e).__name__}: {e}"
                self.log_fn(f"Error on attempt {attempt + 1}: {last_error}")
                if attempt == max_retries - 1:
                    prediction = "None"
        
        # Final validation
        if prediction == "None":
            self.log_fn(f"All {max_retries} attempts failed. Last error: {last_error}")
        elif not self._validate_prediction(prediction):
            self.log_fn(f"Warning: Final prediction '{prediction}' may not be valid")

        return str(prediction), msg_history
