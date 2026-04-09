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
        
        # Clean up common formatting issues
        # Remove markdown code block markers
        inner = re.sub(r'^```\w*\s*', '', inner)
        inner = re.sub(r'\s*```$', '', inner)
        
        try:
            results.append(json.loads(inner))
        except json.JSONDecodeError as e:
            # Try to extract just the response field if full JSON fails
            try:
                # Look for "response": value pattern with more flexible matching
                # Handle both numeric and string values, including negative numbers
                response_match = re.search(r'"response"\s*:\s*(-?\d+(?:\.\d+)?|"[^"]*"|\w+)', inner, re.DOTALL)
                if response_match:
                    value = response_match.group(1).strip()
                    # Try to parse as number or string
                    try:
                        parsed_value = json.loads(value)
                    except json.JSONDecodeError:
                        # If it's not valid JSON, treat as string and try to extract number
                        cleaned_value = value.strip('"').strip("'")
                        # Try to convert to number if possible
                        try:
                            if '.' in cleaned_value:
                                parsed_value = float(cleaned_value)
                            else:
                                parsed_value = int(cleaned_value)
                        except ValueError:
                            parsed_value = cleaned_value
                    results.append({"response": parsed_value})
            except Exception:
                pass
            
            # Try to fix common JSON syntax errors
            try:
                # Fix single quotes to double quotes
                fixed = inner.replace("'", '"')
                # Fix trailing commas
                fixed = re.sub(r',\s*}', '}', fixed)
                fixed = re.sub(r',\s*]', ']', fixed)
                obj = json.loads(fixed)
                if isinstance(obj, dict) and "response" in obj:
                    results.append(obj)
            except (json.JSONDecodeError, Exception):
                pass
            
            continue
    return results or None


def _extract_json_fallback(text: str) -> list[dict] | None:
    """Fallback JSON extraction using regex for unwrapped JSON objects.

    This handles cases where the model outputs valid JSON without <json> tags.
    Searches for JSON objects with a "response" key.
    Enhanced with better error recovery and edge case handling.
    """
    results = []
    
    # First, try to find JSON objects with balanced braces
    # This pattern looks for JSON objects containing "response" key
    # It handles nested braces by counting them
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
            # Try to parse as-is first
            try:
                obj = json.loads(obj_str)
                if "response" in obj:
                    results.append(obj)
                    continue
            except json.JSONDecodeError:
                pass
            
            # Try to fix common JSON syntax errors
            try:
                # Fix single quotes to double quotes
                fixed = obj_str.replace("'", '"')
                # Fix trailing commas
                fixed = re.sub(r',\s*}', '}', fixed)
                fixed = re.sub(r',\s*]', ']', fixed)
                obj = json.loads(fixed)
                if isinstance(obj, dict) and "response" in obj:
                    results.append(obj)
                    continue
            except (json.JSONDecodeError, Exception):
                pass
            
            # Try to extract just the response field
            try:
                # Enhanced pattern to handle negative numbers and decimals
                response_match = re.search(r'["\']response["\']\s*:\s*(-?\d+(?:\.\d+)?|["\'][^"\']*["\']|\w+)', obj_str, re.DOTALL)
                if response_match:
                    value = response_match.group(1).strip()
                    try:
                        parsed_value = json.loads(value)
                    except json.JSONDecodeError:
                        # Try to convert to number if possible
                        cleaned_value = value.strip('"').strip("'")
                        try:
                            if '.' in cleaned_value:
                                parsed_value = float(cleaned_value)
                            else:
                                parsed_value = int(cleaned_value)
                        except ValueError:
                            parsed_value = cleaned_value
                    results.append({"response": parsed_value})
                    continue
            except Exception:
                pass
    
    # If no results, try to parse the entire text as JSON
    if not results:
        try:
            # Clean up the text first
            cleaned_text = text.strip()
            # Remove markdown code block markers
            cleaned_text = re.sub(r'^```\w*\s*', '', cleaned_text)
            cleaned_text = re.sub(r'\s*```$', '', cleaned_text)
            obj = json.loads(cleaned_text)
            if isinstance(obj, dict) and "response" in obj:
                results.append(obj)
        except json.JSONDecodeError:
            # Try with common fixes
            try:
                fixed = text.strip().replace("'", '"')
                fixed = re.sub(r',\s*}', '}', fixed)
                fixed = re.sub(r',\s*]', ']', fixed)
                fixed = re.sub(r'^```\w*\s*', '', fixed)
                fixed = re.sub(r'\s*```$', '', fixed)
                obj = json.loads(fixed)
                if isinstance(obj, dict) and "response" in obj:
                    results.append(obj)
            except (json.JSONDecodeError, Exception):
                pass

    # Last resort: try to extract just a number or string response
    if not results:
        # Look for patterns like "The score is 5" or just a number
        # Try to find a number that appears to be a score (0-10 range typically)
        # Enhanced to handle negative numbers
        number_matches = list(re.finditer(r'-?\b(\d+(?:\.\d+)?)\b', text))
        for number_match in number_matches:
            try:
                value = float(number_match.group(0))  # Use group(0) to include negative sign
                # If it's a whole number, convert to int
                if value == int(value):
                    value = int(value)
                results.append({"response": value})
                break  # Take the first valid number found
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

## Grading Instructions:
1. Carefully analyze the student's answer against the correct solution
2. Consider the grading guidelines when determining the score
3. Award partial credit for correct reasoning even if the final answer is wrong
4. Deduct points for missing steps, incorrect logic, or computational errors
5. The score should be a number (integer or decimal) representing the points earned
6. Be objective and consistent with the grading guidelines

## Scoring Principles:
- Full credit: Complete correct solution with proper reasoning
- Partial credit: Correct approach with minor errors or missing details
- No credit: Completely wrong approach or blank answer
- Check for: Correct method, accurate calculations, proper justification, final answer correctness

## Response Format (STRICT):
You MUST respond ONLY in the following JSON format. Do not include any other text, markdown formatting, code blocks, or explanations outside the JSON tags:

<json>
{{
    "response": <numerical_score>
}}
</json>

The response field must contain ONLY the numerical score value as a number (not a string).
Examples of valid responses:
- "response": 5
- "response": 7.5
- "response": 10
- "response": 0

Do NOT wrap the number in quotes."""

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
        
        # Try multiple extraction strategies with enhanced error handling
        try:
            # Strategy 1: Primary extraction from <json> tags
            extracted = _extract_jsons(response_text)
            if extracted and len(extracted) > 0:
                # Check the last extracted JSON object first (most recent)
                for obj in reversed(extracted):
                    if isinstance(obj, dict) and "response" in obj:
                        pred_val = obj["response"]
                        if self._is_valid_number(pred_val):
                            prediction = pred_val
                            extraction_method = "primary"
                            break
                # If no valid number found, try any response value
                if prediction == "None":
                    for obj in reversed(extracted):
                        if isinstance(obj, dict) and "response" in obj:
                            prediction = obj["response"]
                            extraction_method = "primary"
                            break
            
            # Strategy 2: Fallback extraction for unwrapped JSON
            if prediction == "None" or not self._is_valid_number(prediction):
                extracted = _extract_json_fallback(response_text)
                if extracted and len(extracted) > 0:
                    for obj in reversed(extracted):
                        if isinstance(obj, dict) and "response" in obj:
                            pred_val = obj["response"]
                            if self._is_valid_number(pred_val):
                                prediction = pred_val
                                extraction_method = "fallback"
                                break
                    # If no valid number found, try any response value
                    if prediction == "None":
                        for obj in reversed(extracted):
                            if isinstance(obj, dict) and "response" in obj:
                                prediction = obj["response"]
                                extraction_method = "fallback"
                                break
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            # Try fallback on exception
            try:
                extracted = _extract_json_fallback(response_text)
                if extracted and len(extracted) > 0:
                    for obj in reversed(extracted):
                        if isinstance(obj, dict) and "response" in obj:
                            pred_val = obj["response"]
                            if self._is_valid_number(pred_val):
                                prediction = pred_val
                                extraction_method = "fallback_exception"
                                break
            except Exception as fallback_e:
                self.log_fn(f"Fallback extraction also failed: {fallback_e}")

        # Strategy 3: Direct regex extraction as last resort
        if prediction == "None" or not self._is_valid_number(prediction):
            try:
                # Look for any number in the response
                number_match = re.search(r'-?\b(\d+(?:\.\d+)?)\b', str(response_text))
                if number_match:
                    prediction = number_match.group(1)
                    extraction_method = "regex"
            except Exception as regex_e:
                self.log_fn(f"Regex extraction failed: {regex_e}")

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

    def _normalize_prediction(self, prediction, response_text: str) -> str:
        """Normalize and validate the prediction value.
        
        Enhanced to handle various edge cases and extract numbers from text.
        """
        # If already a number, convert to string
        if isinstance(prediction, (int, float)):
            return str(prediction)
        
        # If it's a string, try to parse as number
        if isinstance(prediction, str):
            # Handle string representations of numbers
            prediction = prediction.strip()
            if prediction:
                try:
                    float(prediction)
                    return prediction
                except (ValueError, TypeError):
                    pass
        
        # Try to extract any number from the prediction or response text
        self.log_fn(f"Invalid prediction value: {prediction}, attempting regex extraction")
        
        # First try the prediction string itself
        if isinstance(prediction, str) and prediction:
            # Look for patterns like "score: 5" or "The answer is 7.5"
            # Enhanced pattern to handle various formats
            patterns = [
                r'-?\b(\d+(?:\.\d+)?)\b',  # Basic number
                r'["\']?response["\']?\s*[=:]\s*["\']?(-?\d+(?:\.\d+)?)["\']?',  # response=5 or response: 5
                r'score\s*[=:]\s*(-?\d+(?:\.\d+)?)',  # score=5 or score: 5
                r'answer\s*is\s*(-?\d+(?:\.\d+)?)',  # answer is 5
            ]
            for pattern in patterns:
                number_match = re.search(pattern, prediction, re.IGNORECASE)
                if number_match:
                    return number_match.group(1)
        
        # Then try the full response text
        if response_text:
            patterns = [
                r'-?\b(\d+(?:\.\d+)?)\b',  # Basic number
                r'["\']?response["\']?\s*[=:]\s*["\']?(-?\d+(?:\.\d+)?)["\']?',  # response=5 or response: 5
                r'<json>[^<]*["\']?response["\']?\s*[=:]\s*(-?\d+(?:\.\d+)?)',  # Inside JSON tags
            ]
            for pattern in patterns:
                number_match = re.search(pattern, str(response_text), re.IGNORECASE)
                if number_match:
                    return number_match.group(1)
        
        return "None"
