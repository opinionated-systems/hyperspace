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
    Enhanced to handle various edge cases and malformed JSON.
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
            # Try to fix common JSON issues and extract response field
            try:
                # Remove markdown code blocks if present
                cleaned_inner = re.sub(r'```json\s*|\s*```', '', inner).strip()
                
                # Try parsing cleaned version
                try:
                    results.append(json.loads(cleaned_inner))
                    continue
                except json.JSONDecodeError:
                    pass
                
                # Look for "response": value pattern with flexible matching
                # Handle numeric values, strings, and various formats
                response_patterns = [
                    r'"response"\s*:\s*(-?\d+\.?\d*)',  # Numeric value
                    r'"response"\s*:\s*"(-?\d+\.?\d*)"',  # Numeric in quotes
                    r"'response'\s*:\s*(-?\d+\.?\d*)",  # Single quotes numeric
                    r"'response'\s*:\s*'(-?\d+\.?\d*)'",  # Single quotes numeric in quotes
                    r'"response"\s*:\s*"([^"]*)"',  # String value
                    r"'response'\s*:\s*'([^']*)'",  # Single quotes string
                ]
                
                for pattern in response_patterns:
                    response_match = re.search(pattern, cleaned_inner, re.DOTALL)
                    if response_match:
                        value = response_match.group(1).strip()
                        # Try to parse as number
                        try:
                            if '.' in value:
                                parsed_value = float(value)
                            else:
                                parsed_value = int(value)
                        except ValueError:
                            # Keep as string if not a number
                            parsed_value = value
                        results.append({"response": parsed_value})
                        break
            except Exception:
                pass
            continue
    return results or None


def _extract_json_fallback(text: str) -> list[dict] | None:
    """Fallback JSON extraction using regex for unwrapped JSON objects.

    This handles cases where the model outputs valid JSON without <json> tags.
    Searches for JSON objects with a "response" key.
    Enhanced to handle various edge cases and malformed responses.
    """
    results = []
    
    # First, try to find JSON objects with balanced braces
    def find_json_objects(s):
        """Find all JSON objects in string with balanced braces."""
        objects = []
        i = 0
        while i < len(s):
            if s[i] == '{':
                start = i
                brace_count = 1
                i += 1
                while i < len(s) and brace_count > 0:
                    if s[i] == '{':
                        brace_count += 1
                    elif s[i] == '}':
                        brace_count -= 1
                    i += 1
                if brace_count == 0:
                    objects.append(s[start:i])
            else:
                i += 1
        return objects
    
    # Try to find JSON objects with "response" key
    json_objects = find_json_objects(text)
    for obj_str in json_objects:
        if '"response"' in obj_str or "'response'" in obj_str:
            try:
                obj = json.loads(obj_str)
                if "response" in obj:
                    results.append(obj)
            except json.JSONDecodeError:
                # Try to extract just the response field with multiple patterns
                response_patterns = [
                    r'"response"\s*:\s*(-?\d+\.?\d*)',
                    r'"response"\s*:\s*"(-?\d+\.?\d*)"',
                    r"'response'\s*:\s*(-?\d+\.?\d*)",
                    r"'response'\s*:\s*'(-?\d+\.?\d*)'",
                    r'"response"\s*:\s*"([^"]*)"',
                    r"'response'\s*:\s*'([^']*)'",
                ]
                for pattern in response_patterns:
                    try:
                        response_match = re.search(pattern, obj_str, re.DOTALL)
                        if response_match:
                            value = response_match.group(1).strip()
                            try:
                                if '.' in value:
                                    parsed_value = float(value)
                                else:
                                    parsed_value = int(value)
                            except ValueError:
                                parsed_value = value
                            results.append({"response": parsed_value})
                            break
                    except Exception:
                        continue
                continue

    # If no results, try to parse the entire text as JSON
    if not results:
        try:
            # Remove markdown code blocks if present
            cleaned_text = re.sub(r'```json\s*|\s*```', '', text).strip()
            obj = json.loads(cleaned_text)
            if isinstance(obj, dict) and "response" in obj:
                results.append(obj)
        except json.JSONDecodeError:
            pass

    # Last resort: try to extract just a number from the text
    # Look for patterns that suggest a score
    if not results:
        # Priority patterns for score extraction
        score_patterns = [
            r'[Ss]core\s*[:=]\s*(-?\d+\.?\d*)',  # "Score: 5" or "score = 10"
            r'[Gg]rade\s*[:=]\s*(-?\d+\.?\d*)',  # "Grade: 7.5"
            r'[Rr]esult\s*[:=]\s*(-?\d+\.?\d*)',  # "Result: 8"
            r'[Ff]inal\s+score\s*[:=]\s*(-?\d+\.?\d*)',  # "Final score: 9"
            r'\b(\d+\.?\d*)\s*points?\b',  # "5 points" or "7.5 point"
            r'\b(\d+\.?\d*)\s*/\s*\d+\b',  # "7/10" or "8.5/10"
        ]
        
        for pattern in score_patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    value_str = match.group(1)
                    if '.' in value_str:
                        value = float(value_str)
                    else:
                        value = int(value_str)
                    results.append({"response": value})
                    break
                except (ValueError, IndexError):
                    continue
        
        # If still no results, take the first standalone number
        if not results:
            number_matches = list(re.finditer(r'\b(\d+(?:\.\d+)?)\b', text))
            for number_match in number_matches:
                try:
                    value = float(number_match.group(1))
                    if value == int(value):
                        value = int(value)
                    results.append({"response": value})
                    break
                except ValueError:
                    pass

    return results or None


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
        # Extract fields for better structured prompting
        domain = inputs.get("domain", "")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = f"""You are an expert mathematics grader specializing in {domain} problems.

Your task is to evaluate a student's answer to a mathematical problem and provide a numerical score.

## Problem Statement:
{problem}

## Correct Solution:
{solution}

## Grading Guidelines:
{grading_guidelines}

## Student's Answer:
{student_answer}

## Evaluation Instructions:
1. Carefully read and understand the problem requirements
2. Compare the student's answer step-by-step with the correct solution
3. Identify what the student got right and what they got wrong
4. Consider the grading guidelines for partial credit
5. Determine the appropriate numerical score based on:
   - Correctness of the final answer
   - Validity of the reasoning/method used
   - Completeness of the solution
   - Any errors or omissions

## Scoring Guidelines:
- Full credit (typically 10 or max points): Complete and correct solution
- Partial credit: Partially correct solution with some valid steps
- No credit (0): Completely incorrect or irrelevant answer
- Be precise with decimal scores when appropriate (e.g., 7.5 for partial credit)

## Response Format:
You must respond ONLY in the following JSON format. Do not include any other text, markdown formatting, or explanations outside the JSON tags:

<json>
{{
    "response": <numerical_score>
}}
</json>

The response field must contain only the numerical score value as a number (integer or decimal).
Examples: 10, 7.5, 5, 0, 8.25

Provide your score now:"""

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

        # Extract prediction from JSON using primary method
        prediction = "None"
        extraction_method = "none"
        response_text = msg_history[-1]["text"] if msg_history else ""
        
        # Try multiple extraction strategies in order of reliability
        extraction_attempts = [
            ("primary", _extract_jsons),
            ("fallback", _extract_json_fallback),
        ]
        
        for method_name, extract_func in extraction_attempts:
            if prediction != "None" and self._is_valid_number(prediction):
                break
            
            try:
                extracted = extract_func(response_text)
                if extracted and len(extracted) > 0:
                    # Try to find a valid numeric response
                    for obj in reversed(extracted):
                        if isinstance(obj, dict) and "response" in obj:
                            pred_val = obj["response"]
                            if self._is_valid_number(pred_val):
                                prediction = pred_val
                                extraction_method = method_name
                                break
                    
                    # If no valid number found in reversed order, try forward
                    if not self._is_valid_number(prediction):
                        for obj in extracted:
                            if isinstance(obj, dict) and "response" in obj:
                                pred_val = obj["response"]
                                if self._is_valid_number(pred_val):
                                    prediction = pred_val
                                    extraction_method = method_name
                                    break
            except Exception as e:
                self.log_fn(f"Error in {method_name} extraction: {e}")
                continue
        
        # If still no valid prediction, try direct number extraction
        if not self._is_valid_number(prediction):
            direct_number = self._extract_direct_number(response_text)
            if direct_number is not None:
                prediction = direct_number
                extraction_method = "direct_number"

        # Validate and normalize the prediction
        prediction = self._normalize_prediction(prediction, response_text)

        self.log_fn(f"Extraction method used: {extraction_method}, prediction: {prediction}")
        return str(prediction), msg_history

    def _is_valid_number(self, value) -> bool:
        """Check if a value is a valid number (int or float)."""
        if isinstance(value, (int, float)):
            return True
        if isinstance(value, str):
            try:
                float(value)
                return True
            except (ValueError, TypeError):
                pass
        return False

    def _extract_direct_number(self, text: str) -> int | float | None:
        """Extract a number directly from text using score-related patterns.
        
        Returns the most likely score value or None if no valid number found.
        """
        if not text:
            return None
            
        # Priority patterns for score extraction (in order of specificity)
        score_patterns = [
            (r'[Ss]core\s*[:=]\s*(-?\d+\.?\d*)', "score_label"),
            (r'[Gg]rade\s*[:=]\s*(-?\d+\.?\d*)', "grade_label"),
            (r'[Ff]inal\s+score\s*[:=]\s*(-?\d+\.?\d*)', "final_score"),
            (r'[Rr]esult\s*[:=]\s*(-?\d+\.?\d*)', "result_label"),
            (r'\b(\d+\.?\d*)\s*points?\b', "points"),
            (r'\b(\d+\.?\d*)\s*/\s*\d+\b', "fraction"),
            (r'"response"\s*:\s*(-?\d+\.?\d*)', "json_response"),
            (r'"response"\s*:\s*"(-?\d+\.?\d*)"', "json_quoted"),
        ]
        
        for pattern, pattern_name in score_patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    value_str = match.group(1)
                    if '.' in value_str:
                        return float(value_str)
                    else:
                        return int(value_str)
                except (ValueError, IndexError):
                    continue
        
        # Last resort: find any standalone number
        # Look for numbers that appear to be scores (typically 0-100 range)
        number_matches = list(re.finditer(r'\b(\d+(?:\.\d+)?)\b', text))
        for number_match in number_matches:
            try:
                value = float(number_match.group(1))
                # Prefer numbers in typical score ranges
                if 0 <= value <= 100:
                    if value == int(value):
                        return int(value)
                    return value
            except ValueError:
                continue
        
        # If no score-range number found, return the first number
        for number_match in number_matches:
            try:
                value = float(number_match.group(1))
                if value == int(value):
                    return int(value)
                return value
            except ValueError:
                continue
                
        return None

    def _normalize_prediction(self, prediction, response_text: str) -> str:
        """Normalize and validate the prediction value."""
        # If already a number, convert to string
        if isinstance(prediction, (int, float)):
            return str(prediction)
        
        # If it's a string, try to parse as number
        if isinstance(prediction, str):
            # Handle "None" or empty string
            if prediction.strip().lower() in ("none", "", "null", "nan"):
                # Try to extract from response text
                direct = self._extract_direct_number(response_text)
                if direct is not None:
                    return str(direct)
                return "None"
            
            try:
                float(prediction)
                return prediction
            except (ValueError, TypeError):
                pass
        
        # Try to extract any number from the prediction or response text
        self.log_fn(f"Invalid prediction value: {prediction}, attempting regex extraction")
        
        # First try the prediction string itself
        if isinstance(prediction, str):
            number_match = re.search(r'-?\b(\d+(?:\.\d+)?)\b', prediction)
            if number_match:
                return number_match.group(1)
        
        # Then try the full response text
        direct = self._extract_direct_number(response_text)
        if direct is not None:
            return str(direct)
        
        return "None"
