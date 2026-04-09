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
        except json.JSONDecodeError as e:
            logger.debug(f"JSON decode error in <json> block: {e}")
            continue
    return results or None


def _extract_json_fallback(text: str) -> list[dict] | None:
    """Fallback JSON extraction using regex for unwrapped JSON objects.

    This handles cases where the model outputs valid JSON without <json> tags.
    Searches for JSON objects with a "response" key.
    """
    results = []
    # Pattern to match JSON objects (handles nested braces)
    pattern = r'\{[^{}]*"response"[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.finditer(pattern, text, re.DOTALL)

    for match in matches:
        try:
            obj = json.loads(match.group())
            if "response" in obj:
                results.append(obj)
        except json.JSONDecodeError:
            continue

    # If regex fails, try to find any JSON-like structure
    if not results:
        try:
            # Try to parse the entire text as JSON
            obj = json.loads(text.strip())
            if isinstance(obj, dict) and "response" in obj:
                results.append(obj)
        except json.JSONDecodeError:
            pass

    return results or None


def _extract_json_brute_force(text: str) -> list[dict] | None:
    """Last-resort extraction: find any valid JSON with 'response' key.
    
    Tries multiple strategies to extract JSON from messy output.
    """
    results = []
    
    # Strategy 1: Find JSON between curly braces with relaxed matching
    # Look for patterns like {"response": ...}
    brace_pattern = r'\{[^}]*"response"[^}]*\}'
    for match in re.finditer(brace_pattern, text, re.DOTALL):
        try:
            obj = json.loads(match.group())
            if "response" in obj:
                results.append(obj)
        except json.JSONDecodeError:
            continue
    
    # Strategy 2: Try to extract from markdown code blocks
    code_block_pattern = r'```(?:json)?\s*(.*?)```'
    for match in re.finditer(code_block_pattern, text, re.DOTALL):
        try:
            obj = json.loads(match.group(1).strip())
            if isinstance(obj, dict) and "response" in obj:
                results.append(obj)
        except json.JSONDecodeError:
            continue
    
    # Strategy 3: Look for the last occurrence of a JSON-like structure
    # and try to fix common issues
    json_like_pattern = r'\{.*"response".*\}'
    match = re.search(json_like_pattern, text, re.DOTALL)
    if match:
        candidate = match.group()
        # Try to fix common JSON issues
        fixes = [
            candidate,  # Original
            candidate.replace("'", '"'),  # Fix single quotes
            re.sub(r',\s*}', '}', candidate),  # Remove trailing comma
            re.sub(r',\s*]', ']', candidate),  # Remove trailing comma in arrays
        ]
        for fixed in fixes:
            try:
                obj = json.loads(fixed)
                if isinstance(obj, dict) and "response" in obj:
                    results.append(obj)
                    break
            except json.JSONDecodeError:
                continue
    
    return results or None


def _validate_prediction(prediction: str, inputs: dict) -> tuple[bool, str]:
    """Validate that the prediction is appropriate for the task.
    
    Returns:
        (is_valid, reason) tuple where is_valid is True if prediction looks reasonable
    """
    if prediction is None or prediction == "None":
        return False, "Prediction is None"
    
    if prediction == "Error: LLM call failed":
        return False, "LLM call failed"
    
    # Check for empty or whitespace-only predictions
    if not prediction or not str(prediction).strip():
        return False, "Prediction is empty or whitespace-only"
    
    # Check if prediction is unreasonably long (might indicate an error)
    if len(str(prediction)) > 10000:
        return False, "Prediction is unreasonably long (>10000 chars)"
    
    # For IMO grading, check if prediction looks like a valid grade
    # Valid grades are typically: 0, 1, 2, 3, 4, 5, 6, 7 or partial credit like 0.5, 1.5, etc.
    prediction_str = str(prediction).strip()
    
    # Try to extract numeric value for validation
    try:
        # Remove common prefixes/suffixes that might appear
        cleaned = prediction_str.lower()
        for prefix in ["grade:", "score:", "points:", "mark:", "the answer is", "answer:", "result:"]:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
        
        # Try to parse as float
        numeric_value = float(cleaned)
        # IMO grades are typically 0-7, but allow some flexibility
        if numeric_value < 0 or numeric_value > 10:
            return False, f"Numeric prediction {numeric_value} outside reasonable range (0-10)"
    except ValueError:
        # Not a numeric prediction - this might be OK for some tasks
        # but for IMO grading we expect numbers
        pass
    
    return True, "Prediction looks valid"


class TaskAgent:
    """Task agent that solves IMO grading problems with robust JSON extraction."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Build a more structured and informative prompt
        domain = inputs.get("domain", "Unknown")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        instruction = f"""You are an expert mathematics grader for the International Mathematical Olympiad (IMO).

Your task is to grade a student's solution to a mathematical problem. You must provide a numerical grade based on the official solution and grading guidelines.

DOMAIN: {domain}

PROBLEM:
{problem}

OFFICIAL SOLUTION:
{solution}

GRADING GUIDELINES:
{grading_guidelines}

STUDENT'S ANSWER:
{student_answer}

Instructions:
1. Carefully read the problem, official solution, and grading guidelines.
2. Analyze the student's answer against the official solution.
3. Assign a grade based on the grading guidelines.
4. IMO problems are typically graded on a scale of 0-7 points.
5. Provide your grade as a number (integer or decimal for partial credit).

Respond ONLY in the following JSON format:
<json>
{{
    "response": <your numerical grade here>
}}
</json>

Example valid responses:
<json>
{{
    "response": 7
}}
</json>

<json>
{{
    "response": 3.5
}}
</json>

<json>
{{
    "response": 0
}}
</json>"""

        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
        except Exception as e:
            import traceback
            error_details = f"Error calling LLM: {e}\n{traceback.format_exc()}"
            self.log_fn(error_details)
            return "Error: LLM call failed", [{"role": "system", "text": error_details}]

        # Extract prediction from JSON using multiple methods
        prediction = "None"
        extraction_method = "none"
        
        # Try primary extraction
        try:
            extracted = _extract_jsons(msg_history[-1]["text"])
            if extracted and "response" in extracted[-1]:
                prediction = extracted[-1]["response"]
                extraction_method = "primary"
        except Exception as e:
            self.log_fn(f"Primary extraction failed: {e}")
        
        # Try fallback extraction if primary failed
        if prediction == "None":
            try:
                extracted = _extract_json_fallback(msg_history[-1]["text"])
                if extracted and "response" in extracted[-1]:
                    prediction = extracted[-1]["response"]
                    extraction_method = "fallback"
            except Exception as e:
                self.log_fn(f"Fallback extraction failed: {e}")
        
        # Try brute force extraction as last resort
        if prediction == "None":
            try:
                extracted = _extract_json_brute_force(msg_history[-1]["text"])
                if extracted and "response" in extracted[-1]:
                    prediction = extracted[-1]["response"]
                    extraction_method = "brute_force"
            except Exception as e:
                self.log_fn(f"Brute force extraction failed: {e}")

        # Validate the prediction
        is_valid, validation_reason = _validate_prediction(prediction, inputs)
        if not is_valid:
            self.log_fn(f"Prediction validation failed: {validation_reason}")
        else:
            self.log_fn(f"Prediction validation passed: {validation_reason}")

        self.log_fn(f"Extraction method used: {extraction_method}")
        return str(prediction), msg_history
