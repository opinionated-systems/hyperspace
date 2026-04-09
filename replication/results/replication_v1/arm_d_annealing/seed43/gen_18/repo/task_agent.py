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

    def _build_prompt(self, inputs: dict) -> str:
        """Build a structured prompt with chain-of-thought instructions."""
        domain = inputs.get("domain", "unknown")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        return f"""You are an expert mathematical grader for {domain} problems.

Your task is to evaluate a student's answer to a mathematical problem and assign a numerical grade from 0 to 7 (IMO-style grading).

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
4. Consider partial credit based on the grading guidelines:
   - 7 points: Complete, correct solution
   - 6 points: Minor flaw in an otherwise correct solution
   - 5 points: Correct approach with significant gaps
   - 4 points: Partial progress with substantial work
   - 3 points: Some meaningful progress
   - 2 points: Limited progress
   - 1 point: Minimal progress
   - 0 points: No meaningful progress or completely wrong
5. Provide your final grade in the JSON format below.

Respond ONLY in JSON format with the following schema:
<json>
{{
    "reasoning": "Your detailed step-by-step analysis here",
    "response": "Your final grade as a number 0-7"
}}
</json>

IMPORTANT: The "response" field MUST contain ONLY a single digit from 0 to 7. Do not include any other text, explanations, or formatting in the response field."""

    def _validate_grade(self, prediction: str) -> tuple[bool, str]:
        """Validate that the prediction is a valid grade format.
        
        Enhanced to handle more edge cases and provide better grade extraction.
        
        Returns:
            (is_valid, cleaned_grade) tuple
        """
        if not prediction or prediction == "None":
            return False, "None"
        
        prediction = prediction.strip()
        
        # Check for numeric grades (0-7 for IMO problems)
        if prediction.isdigit():
            grade = int(prediction)
            if 0 <= grade <= 7:
                return True, str(grade)
            return False, "None"
        
        # Check for common grade formats with normalization
        grade_mapping = {
            "correct": "7",
            "full": "7",
            "full credit": "7",
            "right": "7",
            "true": "7",
            "yes": "7",
            "pass": "7",
            "incorrect": "0",
            "wrong": "0",
            "zero": "0",
            "fail": "0",
            "false": "0",
            "no": "0",
            "partial": "3",
            "partial credit": "3",
            "half": "3",
            "incomplete": "3",
        }
        lower_pred = prediction.lower()
        
        if lower_pred in grade_mapping:
            return True, grade_mapping[lower_pred]
        
        # Check for fractional grades (e.g., "3/7", "5/7")
        if "/" in prediction:
            parts = prediction.split("/")
            if len(parts) == 2 and parts[0].strip().isdigit() and parts[1].strip().isdigit():
                numerator = int(parts[0].strip())
                denominator = int(parts[1].strip())
                if 0 <= numerator <= denominator and denominator <= 7:
                    return True, str(numerator)
        
        # Check for decimal grades (e.g., "3.5", "6.0")
        try:
            float_val = float(prediction)
            if 0 <= float_val <= 7:
                # Round to nearest integer for IMO-style grading
                return True, str(int(round(float_val)))
        except ValueError:
            pass
        
        # If it looks like a number but has extra text, try to extract
        numeric_match = re.search(r'\b([0-7])\b', prediction)
        if numeric_match:
            return True, numeric_match.group(1)
        
        # Check for phrases containing grades
        phrase_match = re.search(r'grade\s*(?:of|:)?\s*([0-7])', lower_pred)
        if phrase_match:
            return True, phrase_match.group(1)
        
        score_match = re.search(r'score\s*(?:of|:)?\s*([0-7])', lower_pred)
        if score_match:
            return True, score_match.group(1)
        
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
        instruction = self._build_prompt(inputs)
        
        prediction = "None"
        reasoning = ""
        msg_history = []
        
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
                    self.log_fn(f"Successfully extracted prediction: {prediction}")
                    break
                else:
                    self.log_fn(f"Attempt {attempt + 1}: Failed to extract valid prediction, retrying...")
                    # Add progressively stronger hints for the next attempt
                    if attempt == 0:
                        instruction += "\n\nIMPORTANT: Your response MUST include a 'response' field with a single digit from 0-7."
                    elif attempt == 1:
                        instruction += "\n\nCRITICAL: The JSON must have exactly this format: {\"reasoning\": \"...\", \"response\": \"5\"} where 5 is the grade."
                    else:
                        instruction += "\n\nFINAL ATTEMPT: Output ONLY the JSON object with a numeric grade 0-7 in the response field."
                    
            except Exception as e:
                self.log_fn(f"Attempt {attempt + 1}: Error during LLM call: {e}")
                if attempt == self.max_retries - 1:
                    # Last attempt failed, return what we have
                    break
        
        # Log final result
        if prediction == "None" or not prediction.strip():
            self.log_fn("Warning: Could not extract valid prediction after all retries")
        
        return str(prediction), msg_history
