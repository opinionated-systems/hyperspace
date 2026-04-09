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
    Also handles common LLM output patterns like markdown code blocks.
    """
    results = []
    
    # First, try to extract from markdown code blocks
    # Pattern for ```json ... ``` blocks
    json_code_pattern = r'```(?:json)?\s*\n?(.*?)```'
    for match in re.finditer(json_code_pattern, text, re.DOTALL):
        try:
            json_str = match.group(1).strip()
            results.append(json.loads(json_str))
        except json.JSONDecodeError:
            pass
    
    # Pattern for single backtick JSON
    backtick_pattern = r'`(\{[^`]*\})`'
    for match in re.finditer(backtick_pattern, text):
        try:
            json_str = match.group(1).strip()
            results.append(json.loads(json_str))
        except json.JSONDecodeError:
            pass
    
    # Try to find JSON-like structures with curly braces (original method)
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
                    # Skip if already found via code block
                    if not any(json_str in str(r) for r in results):
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
        
        # Build base prompt with enhanced IMO-specific guidance
        base_prompt = f"""You are an expert mathematical grader for {domain} problems, specializing in IMO (International Mathematical Olympiad) grading standards.

Your task is to evaluate a student's answer to a mathematical problem with precision and consistency.

PROBLEM:
{problem}

OFFICIAL SOLUTION:
{solution}

GRADING GUIDELINES:
{grading_guidelines}

STUDENT'S ANSWER:
{student_answer}

IMO GRADING STANDARDS (0-7 scale):
- 7 points: Complete, correct solution with rigorous proof
- 6 points: Minor flaw in an otherwise correct solution
- 5 points: Significant progress with one major gap
- 4 points: Multiple substantial ideas or significant progress
- 3 points: Some meaningful progress toward solution
- 2 points: Limited progress, some relevant ideas
- 1 point: Minimal progress, basic relevant observation
- 0 points: No meaningful progress or irrelevant answer

INSTRUCTIONS:
1. Carefully analyze the student's answer step by step against the official solution.
2. Identify which key steps the student completed correctly.
3. Identify any errors, gaps, or logical flaws in the student's reasoning.
4. Determine partial credit based on the IMO 0-7 scale above.
5. Consider the grading guidelines for specific partial credit rules.
6. Provide your final grade as a single integer from 0 to 7.

Respond ONLY in the following JSON format:
<json>
{{
    "reasoning": "Your detailed step-by-step analysis comparing student answer to official solution, explaining what was correct/incorrect and why the grade was assigned",
    "response": "X"
}}
</json>

The "response" field MUST contain ONLY a single integer from 0 to 7 (no words, no explanations, just the number)."""
        
        # Add retry-specific guidance if this is a retry attempt
        if attempt > 0 and previous_error:
            retry_guidance = f"""

⚠️ PREVIOUS ATTEMPT FAILED: {previous_error}

Please ensure your response:
- Uses the exact JSON format shown above with <json> tags
- Includes BOTH "reasoning" and "response" fields
- The "response" field contains ONLY a single digit from 0-7 (no extra text)
- The JSON is valid and properly formatted
- Example correct response: "response": "5"
"""
            base_prompt += retry_guidance
        
        return base_prompt

    def _validate_grade(self, prediction: str) -> tuple[bool, str]:
        """Validate that the prediction is a valid IMO grade format (0-7).
        
        Returns:
            (is_valid, cleaned_grade) tuple where cleaned_grade is a digit 0-7 or "None"
        """
        if not prediction or prediction == "None":
            return False, "None"
        
        prediction = prediction.strip()
        
        # Direct digit check (most common case)
        if prediction.isdigit():
            grade = int(prediction)
            if 0 <= grade <= 7:
                return True, str(grade)
            return False, "None"
        
        # Check for single digit surrounded by quotes or other characters
        # Extract just the first digit found
        numeric_match = re.search(r'\b([0-7])\b', prediction)
        if numeric_match:
            return True, numeric_match.group(1)
        
        # Check for fractional grades (e.g., "3/7", "5/7") - extract numerator
        if "/" in prediction:
            parts = prediction.split("/")
            if len(parts) == 2 and parts[0].strip().isdigit() and parts[1].strip().isdigit():
                numerator = int(parts[0].strip())
                denominator = int(parts[1].strip())
                if 0 <= numerator <= 7 and denominator > 0:
                    return True, str(numerator)
        
        # Map common text responses to IMO grades
        lower_pred = prediction.lower().strip()
        grade_mapping = {
            "correct": "7",
            "full": "7",
            "complete": "7",
            "perfect": "7",
            "incorrect": "0",
            "wrong": "0",
            "fail": "0",
            "zero": "0",
            "none": "0",
            "partial": "3",  # Middle ground for partial credit
            "incomplete": "3",
            "pass": "4",  # Bare pass
        }
        
        if lower_pred in grade_mapping:
            return True, grade_mapping[lower_pred]
        
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
            prediction = str(int(prediction))
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
                
                # Validate that we got a meaningful prediction (0-7 for IMO)
                if prediction != "None" and prediction.strip() and prediction.isdigit():
                    grade = int(prediction)
                    if 0 <= grade <= 7:
                        self.log_fn(f"Successfully extracted valid IMO grade: {prediction} on attempt {attempt + 1}")
                        break
                    else:
                        previous_error = f"Grade {grade} out of valid IMO range (0-7)"
                        self.log_fn(f"Attempt {attempt + 1}: {previous_error}, retrying...")
                elif prediction != "None" and prediction.strip():
                    # Got a non-numeric prediction, try to use it
                    self.log_fn(f"Extracted non-numeric prediction: {prediction} on attempt {attempt + 1}")
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
        else:
            self.log_fn(f"Final IMO grade: {prediction}")
        
        return str(prediction), msg_history
