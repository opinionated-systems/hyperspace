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
        
        # Build base prompt
        base_prompt = f"""You are an expert mathematical grader for {domain} problems.

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

The "response" field should contain only the final grade (e.g., "7", "5", "0", "Correct", "Incorrect", etc.)."""
        
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
        
        Enhanced validation with better error logging and support for
        various grade formats commonly used in mathematical grading.
        
        Returns:
            (is_valid, cleaned_grade) tuple
        """
        if not prediction or prediction == "None":
            self.log_fn(f"Grade validation: empty or 'None' prediction received")
            return False, "None"
        
        original_prediction = prediction
        prediction = prediction.strip()
        
        # Check for numeric grades (0-7 for IMO problems)
        if prediction.isdigit():
            grade = int(prediction)
            if 0 <= grade <= 7:
                self.log_fn(f"Grade validation: valid numeric grade {grade}")
                return True, str(grade)
            self.log_fn(f"Grade validation: numeric grade {grade} out of range [0-7]")
            return False, "None"
        
        # Check for decimal grades that are effectively integers (e.g., "7.0", "3.00")
        try:
            float_val = float(prediction)
            if float_val == int(float_val):  # It's a whole number
                int_val = int(float_val)
                if 0 <= int_val <= 7:
                    self.log_fn(f"Grade validation: valid decimal grade {prediction} -> {int_val}")
                    return True, str(int_val)
        except ValueError:
            pass  # Not a valid float, continue with other checks
        
        # Check for common grade formats
        valid_non_numeric = ["correct", "incorrect", "partial", "full", "zero", 
                            "pass", "fail", "true", "false", "yes", "no"]
        lower_pred = prediction.lower()
        
        if lower_pred in valid_non_numeric:
            self.log_fn(f"Grade validation: valid non-numeric grade '{prediction}'")
            return True, prediction
        
        # Check for fractional grades (e.g., "3/7", "5/7")
        if "/" in prediction:
            parts = prediction.split("/")
            if len(parts) == 2 and parts[0].strip().isdigit() and parts[1].strip().isdigit():
                numerator = int(parts[0].strip())
                denominator = int(parts[1].strip())
                if 0 <= numerator <= denominator and denominator <= 7:
                    self.log_fn(f"Grade validation: valid fractional grade {prediction}")
                    return True, prediction
                else:
                    self.log_fn(f"Grade validation: fractional grade {prediction} out of valid range")
        
        # Check for grades with parentheses or brackets (e.g., "(7)", "[5]")
        bracket_match = re.search(r'[\(\[\{]([0-7])[\)\]\}]', prediction)
        if bracket_match:
            extracted = bracket_match.group(1)
            self.log_fn(f"Grade validation: extracted grade {extracted} from bracketed '{prediction}'")
            return True, extracted
        
        # If it looks like a number but has extra text, try to extract
        numeric_match = re.search(r'\b([0-7])\b', prediction)
        if numeric_match:
            extracted = numeric_match.group(1)
            self.log_fn(f"Grade validation: extracted numeric grade {extracted} from '{prediction}'")
            return True, extracted
        
        self.log_fn(f"Grade validation: could not validate '{original_prediction}'")
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
