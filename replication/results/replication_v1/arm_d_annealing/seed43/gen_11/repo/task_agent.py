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


def _extract_json_fuzzy(text: str) -> list[dict] | None:
    """Fuzzy JSON extraction for when standard extraction fails.
    
    Tries to find JSON objects even without proper <json> tags.
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
                    results.append(json.loads(json_str))
                except json.JSONDecodeError:
                    pass
                start_idx = None
    return results or None


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.max_retries = 3

    def _build_prompt(self, inputs: dict, attempt: int = 0, previous_error: str = "") -> str:
        """Build a structured prompt with chain-of-thought instructions.
        
        Args:
            inputs: Dictionary containing problem data
            attempt: Current retry attempt number (0 for first attempt)
            previous_error: Description of what went wrong in previous attempt
        """
        domain = inputs.get("domain", "unknown")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        # Build base prompt with enhanced instructions
        base_prompt = f"""You are an expert mathematical grader for {domain} problems.

Your task is to evaluate a student's answer to a mathematical problem with precision and accuracy.

PROBLEM:
{problem}

OFFICIAL SOLUTION:
{solution}

GRADING GUIDELINES:
{grading_guidelines}

STUDENT'S ANSWER:
{student_answer}

INSTRUCTIONS:
1. Carefully analyze the student's answer step by step, comparing it to the official solution.
2. Check if the student has the correct final answer and valid reasoning.
3. Verify that the student's logical steps are sound and mathematically correct.
4. Award partial credit based on the grading guidelines - consider what concepts they demonstrated correctly.
5. Be precise: if the student made a minor error but showed correct methodology, award appropriate partial credit.
6. Provide your final grade in the exact JSON format below.

IMPORTANT: Your response MUST be in this exact JSON format:
<json>
{{
    "reasoning": "Your detailed step-by-step analysis here. Explain what the student did right and wrong.",
    "response": "Your final grade here - just the number or word, nothing else"
}}
</json>

The "response" field must contain ONLY the final grade with no extra text, explanation, or punctuation.
Valid formats: "7", "6", "5", "4", "3", "2", "1", "0", "Correct", "Incorrect", "Partial"
Examples of INVALID responses: "The answer is 7", "Grade: 5", "7/7 points"""
        
        # Add retry-specific guidance if this is a retry attempt
        if attempt > 0 and previous_error:
            retry_guidance = f"""

⚠️ PREVIOUS ATTEMPT FAILED: {previous_error}

Please ensure your response:
- Uses the exact JSON format shown above with <json> tags
- Includes BOTH "reasoning" and "response" fields
- The "response" field contains ONLY the grade (no extra text)
- The JSON is valid and properly formatted
"""
            base_prompt += retry_guidance
        
        return base_prompt

    def _validate_grade(self, prediction: str) -> tuple[bool, str]:
        """Validate that the prediction is a valid grade format.
        
        Returns:
            (is_valid, cleaned_grade) tuple
        """
        if not prediction or prediction == "None":
            return False, "None"
        
        prediction = prediction.strip()
        
        # Remove common prefixes/suffixes that LLMs sometimes add
        cleaned = prediction
        prefixes_to_remove = [
            "the answer is ", "grade: ", "grade is ", "score: ", "score is ",
            "final grade: ", "final answer: ", "result: ", "points: ",
            "i give ", "i would give ", "the grade is ", "the score is ",
            "answer: ", "grading: ", "assessment: ", "evaluation: ",
            "mark: ", "marks: ", "rating: ", "value: ", "output: ",
            "the student receives ", "student receives ", "student gets ",
            "the student gets ", "they receive ", "they get ", "total: ",
            "final score: ", "final: ", "grade = ", "score = ", "result = ",
            "= ", "=", ": ", "- ", "-- ", "**", "*", '"', "'"
        ]
        for prefix in prefixes_to_remove:
            if cleaned.lower().startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
        
        # Remove trailing punctuation and whitespace
        cleaned = cleaned.rstrip(".!?;:,\"'").strip()
        
        # Check for numeric grades (0-7 for IMO problems)
        if cleaned.isdigit():
            grade = int(cleaned)
            if 0 <= grade <= 7:
                return True, str(grade)
            return False, "None"
        
        # Check for common grade formats
        valid_non_numeric = ["correct", "incorrect", "partial", "full", "zero", 
                            "pass", "fail", "true", "false", "yes", "no"]
        lower_cleaned = cleaned.lower()
        
        if lower_cleaned in valid_non_numeric:
            return True, cleaned
        
        # Check for fractional grades (e.g., "3/7", "5/7")
        if "/" in cleaned:
            parts = cleaned.split("/")
            if len(parts) == 2 and parts[0].strip().isdigit() and parts[1].strip().isdigit():
                numerator = int(parts[0].strip())
                denominator = int(parts[1].strip())
                if 0 <= numerator <= denominator and denominator <= 7:
                    return True, cleaned
        
        # If it looks like a number but has extra text, try to extract
        numeric_match = re.search(r'\b([0-7])\b', cleaned)
        if numeric_match:
            return True, numeric_match.group(1)
        
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
        
        # Try each extracted JSON, starting from the last one
        for last_json in reversed(extracted):
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
            if is_valid:
                return cleaned_prediction, reasoning
        
        # If no valid prediction found, return the last one with warning
        last_json = extracted[-1]
        prediction = last_json.get("response", "None")
        reasoning = last_json.get("reasoning", "")
        
        if isinstance(prediction, (int, float)):
            prediction = str(prediction)
        elif isinstance(prediction, str):
            prediction = prediction.strip()
        else:
            prediction = str(prediction)
        
        is_valid, cleaned_prediction = self._validate_grade(prediction)
        self.log_fn(f"Warning: Invalid grade format '{prediction}', using '{cleaned_prediction}'")
        
        return cleaned_prediction, reasoning

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with retry logic.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        prediction = "None"
        reasoning = ""
        msg_history = []
        previous_error = ""
        
        for attempt in range(self.max_retries):
            # Build prompt with attempt-specific guidance
            instruction = self._build_prompt(inputs, attempt=attempt, previous_error=previous_error)
            
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=[],
                )
                
                prediction, reasoning = self._extract_prediction(msg_history)
                
                # Validate that we got a meaningful prediction
                if prediction != "None" and prediction.strip():
                    self.log_fn(f"Successfully extracted prediction: {prediction} on attempt {attempt + 1}")
                    break
                else:
                    # Determine what went wrong for better retry guidance
                    if not msg_history:
                        previous_error = "No response received from LLM"
                    else:
                        last_text = msg_history[-1].get("text", "")
                        if "<json>" not in last_text:
                            previous_error = "Response missing <json> tags"
                        elif "response" not in last_text:
                            previous_error = "JSON missing 'response' field"
                        elif "reasoning" not in last_text:
                            previous_error = "JSON missing 'reasoning' field"
                        else:
                            previous_error = "Could not parse valid JSON from response"
                    
                    self.log_fn(f"Attempt {attempt + 1}: {previous_error}, retrying...")
                    
            except Exception as e:
                previous_error = f"Error during LLM call: {str(e)}"
                self.log_fn(f"Attempt {attempt + 1}: {previous_error}")
                if attempt == self.max_retries - 1:
                    # Last attempt failed, return what we have
                    break
        
        # Log final result
        if prediction == "None" or not prediction.strip():
            self.log_fn("Warning: Could not extract valid prediction after all retries")
        
        return str(prediction), msg_history
