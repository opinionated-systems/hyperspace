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
        
        # Try to parse the JSON
        try:
            results.append(json.loads(inner))
        except json.JSONDecodeError:
            # Try to fix common JSON issues
            try:
                # Remove markdown code block markers if present
                cleaned = inner
                if cleaned.startswith("```json"):
                    cleaned = cleaned[7:]
                elif cleaned.startswith("```"):
                    cleaned = cleaned[3:]
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3]
                cleaned = cleaned.strip()
                results.append(json.loads(cleaned))
            except json.JSONDecodeError:
                continue
    return results or None


def _extract_json_fuzzy(text: str) -> list[dict] | None:
    """Fuzzy JSON extraction for when standard extraction fails.
    
    Tries to find JSON objects even without proper <json> tags.
    """
    results = []
    
    # First, try to find JSON in markdown code blocks
    import re
    code_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    for match in re.finditer(code_block_pattern, text, re.DOTALL):
        try:
            json_str = match.group(1).strip()
            results.append(json.loads(json_str))
        except json.JSONDecodeError:
            pass
    
    # Then try to find JSON-like structures with curly braces
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
                    parsed = json.loads(json_str)
                    # Only keep objects that look like our expected format
                    if isinstance(parsed, dict) and ('response' in parsed or 'reasoning' in parsed):
                        results.append(parsed)
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

IMPORTANT: You MUST respond in valid JSON format wrapped in <json> tags. The JSON must contain exactly two fields:
- "reasoning": Your detailed step-by-step analysis
- "response": ONLY the final grade (a number 0-7, or a single word like "Correct"/"Incorrect")

Example response:
<json>
{{
    "reasoning": "The student correctly identified the pattern and applied the formula. The final answer matches the official solution. The reasoning is clear and logical.",
    "response": "7"
}}
</json>

Another example:
<json>
{{
    "reasoning": "The student made an error in the calculation at step 3, leading to an incorrect final answer. However, the approach was correct.",
    "response": "3"
}}
</json>

Now provide your evaluation:"""

    def _validate_grade(self, prediction: str) -> tuple[bool, str]:
        """Validate that the prediction is a valid grade format.
        
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
        
        # Check for common grade formats
        valid_non_numeric = ["correct", "incorrect", "partial", "full", "zero", 
                            "pass", "fail", "true", "false", "yes", "no"]
        lower_pred = prediction.lower()
        
        if lower_pred in valid_non_numeric:
            return True, prediction
        
        # Check for fractional grades (e.g., "3/7", "5/7")
        if "/" in prediction:
            parts = prediction.split("/")
            if len(parts) == 2 and parts[0].strip().isdigit() and parts[1].strip().isdigit():
                numerator = int(parts[0].strip())
                denominator = int(parts[1].strip())
                if 0 <= numerator <= denominator and denominator <= 7:
                    return True, prediction
        
        # Check for decimal grades (e.g., "3.5", "5.0")
        try:
            float_val = float(prediction)
            if 0 <= float_val <= 7:
                return True, str(int(float_val)) if float_val == int(float_val) else prediction
        except ValueError:
            pass
        
        # If it looks like a number but has extra text, try to extract
        import re
        numeric_match = re.search(r'\b([0-7])\b', prediction)
        if numeric_match:
            return True, numeric_match.group(1)
        
        # Try to extract a number from patterns like "Grade: 5" or "Score: 3"
        grade_pattern = re.search(r'(?:grade|score|mark|points?)\s*[:=]?\s*([0-7])\b', lower_pred)
        if grade_pattern:
            return True, grade_pattern.group(1)
        
        return False, "None"

    def _extract_prediction(self, msg_history: list[dict]) -> tuple[str, str]:
        """Extract prediction and reasoning from message history.
        
        Returns:
            (prediction, reasoning) tuple
        """
        if not msg_history:
            return "None", ""
        
        # Search through all messages from the end to find valid JSON
        extracted = None
        for msg in reversed(msg_history):
            last_text = msg.get("text", "")
            if not last_text:
                continue
            
            # Try standard extraction first
            extracted = _extract_jsons(last_text)
            if extracted:
                break
            
            # Try fuzzy extraction as fallback
            extracted = _extract_json_fuzzy(last_text)
            if extracted:
                break
        
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
                        instruction += "\n\nIMPORTANT: Your response MUST be in valid JSON format wrapped in <json> tags with 'reasoning' and 'response' fields."
                    elif attempt == 1:
                        instruction += "\n\nCRITICAL: The 'response' field must contain ONLY the final grade (e.g., '7', '5', '0', 'Correct', 'Incorrect'). Do not include any explanation in the response field."
                    else:
                        instruction += "\n\nFINAL ATTEMPT: Please output ONLY the JSON block. Example:\n<json>\n{\"reasoning\": \"Analysis here\", \"response\": \"7\"}\n</json>"
                    
            except Exception as e:
                self.log_fn(f"Attempt {attempt + 1}: Error during LLM call: {e}")
                if attempt == self.max_retries - 1:
                    # Last attempt failed, return what we have
                    break
        
        # Log final result
        if prediction == "None" or not prediction.strip():
            self.log_fn("Warning: Could not extract valid prediction after all retries")
        
        return str(prediction), msg_history
