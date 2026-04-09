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
    """Extract JSON objects from <json>...</json> blocks or markdown code blocks.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    Also handles markdown code blocks (```json...```).
    """
    results = []
    search_from = 0
    
    # First, try to extract from <json> tags
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
            # Try to extract just the response field if full JSON fails
            try:
                # Look for "response": value pattern with more flexible matching
                # Handle both numeric and string values
                response_match = re.search(r'"response"\s*:\s*([^,\}]+|"[^"]*")', inner, re.DOTALL)
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
            continue
    
    # Also try markdown code blocks (```json...```)
    markdown_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    for match in re.finditer(markdown_pattern, text, re.DOTALL):
        inner = match.group(1).strip()
        try:
            obj = json.loads(inner)
            if isinstance(obj, dict) and "response" in obj:
                results.append(obj)
        except json.JSONDecodeError:
            # Try to extract just the response field
            try:
                response_match = re.search(r'"response"\s*:\s*([^,\}]+|"[^"]*")', inner, re.DOTALL)
                if response_match:
                    value = response_match.group(1).strip()
                    try:
                        parsed_value = json.loads(value)
                    except json.JSONDecodeError:
                        cleaned_value = value.strip('"').strip("'")
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
    
    return results or None


def _extract_json_fallback(text: str) -> list[dict] | None:
    """Fallback JSON extraction using regex for unwrapped JSON objects.

    This handles cases where the model outputs valid JSON without <json> tags.
    Searches for JSON objects with a "response" key.
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
            try:
                obj = json.loads(obj_str)
                if "response" in obj:
                    results.append(obj)
            except json.JSONDecodeError:
                # Try to extract just the response field
                try:
                    response_match = re.search(r'"response"\s*:\s*([^,\}]+|"[^"]*")', obj_str, re.DOTALL)
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
                except Exception:
                    pass
                continue

    # If no results, try to parse the entire text as JSON
    if not results:
        try:
            obj = json.loads(text.strip())
            if isinstance(obj, dict) and "response" in obj:
                results.append(obj)
        except json.JSONDecodeError:
            pass

    # Last resort: try to extract just a number or string response
    if not results:
        # Look for patterns like "The score is 5" or just a number
        # Try to find a number that appears to be a score (0-10 range typically)
        number_matches = list(re.finditer(r'\b(\d+(?:\.\d+)?)\b', text))
        for number_match in number_matches:
            try:
                value = float(number_match.group(1))
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

        # Parse grading guidelines to extract max score if available
        max_score = self._extract_max_score(grading_guidelines)
        
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

## Instructions:
1. Carefully analyze the student's answer against the correct solution
2. Check for: correct final answer, valid reasoning/logic, proper mathematical notation
3. Consider the grading guidelines when determining the score
4. Award partial credit for correct reasoning even if the final answer is wrong
5. Deduct points for: missing steps, incorrect logic, wrong final answer, unclear notation
6. Provide a numerical score as your response
7. Be precise and objective in your evaluation
8. The score should be a number (integer or decimal)
{f"9. Maximum possible score is {max_score}" if max_score else "9. Ensure the score is within the valid range specified in the grading guidelines"}

IMPORTANT: You must respond ONLY in the following JSON format. Do not include any other text, markdown formatting, or explanations outside the JSON tags:
<json>
{{
    "response": <numerical_score>
}}
</json>

The response field must contain only the numerical score value (e.g., 5, 7.5, 10, 0)."""

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
        extraction_method = "primary"
        response_text = msg_history[-1]["text"] if msg_history else ""
        
        # Try multiple extraction strategies
        try:
            extracted = _extract_jsons(response_text)
            if extracted and len(extracted) > 0:
                # Check the last extracted JSON object
                last_extracted = extracted[-1]
                if isinstance(last_extracted, dict) and "response" in last_extracted:
                    prediction = last_extracted["response"]
                # Also check if any extracted object has a valid response
                for obj in reversed(extracted):
                    if isinstance(obj, dict) and "response" in obj:
                        pred_val = obj["response"]
                        if self._is_valid_number(pred_val):
                            prediction = pred_val
                            break
            
            if prediction == "None":
                # Try fallback extraction
                extracted = _extract_json_fallback(response_text)
                if extracted and len(extracted) > 0:
                    for obj in reversed(extracted):
                        if isinstance(obj, dict) and "response" in obj:
                            pred_val = obj["response"]
                            if self._is_valid_number(pred_val):
                                prediction = pred_val
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
                                extraction_method = "fallback"
                                break
            except Exception as fallback_e:
                self.log_fn(f"Fallback extraction also failed: {fallback_e}")

        # If extraction failed, try one more time with a simplified prompt
        if prediction == "None" or not self._is_valid_number(prediction):
            self.log_fn("Initial extraction failed, attempting retry with simplified prompt")
            try:
                retry_instruction = f"""Based on the previous analysis, provide ONLY the numerical score.

Previous response: {response_text[:500]}

Respond ONLY with a number (e.g., 5, 7.5, 10, 0).

<json>
{{
    "response": <numerical_score>
}}
</json>"""
                
                retry_response, retry_msg_history, retry_info = get_response_from_llm(
                    msg=retry_instruction,
                    model=self.model,
                    msg_history=[],
                )
                
                retry_text = retry_msg_history[-1]["text"] if retry_msg_history else ""
                extracted_retry = _extract_jsons(retry_text)
                
                if extracted_retry and len(extracted_retry) > 0:
                    for obj in reversed(extracted_retry):
                        if isinstance(obj, dict) and "response" in obj:
                            pred_val = obj["response"]
                            if self._is_valid_number(pred_val):
                                prediction = pred_val
                                extraction_method = "retry"
                                msg_history = retry_msg_history
                                break
                
                # If still no valid prediction, try direct number extraction from retry
                if prediction == "None" or not self._is_valid_number(prediction):
                    number_match = re.search(r'-?\b(\d+(?:\.\d+)?)\b', retry_text)
                    if number_match:
                        prediction = number_match.group(1)
                        extraction_method = "retry_regex"
                        msg_history = retry_msg_history
                        
            except Exception as retry_e:
                self.log_fn(f"Retry extraction failed: {retry_e}")

        # Validate and normalize the prediction
        prediction = self._normalize_prediction(prediction, response_text, max_score)

        self.log_fn(f"Extraction method used: {extraction_method}, prediction: {prediction}")
        return str(prediction), msg_history

    def _extract_max_score(self, grading_guidelines: str) -> float | None:
        """Extract the maximum possible score from grading guidelines."""
        if not grading_guidelines:
            return None
        
        # Look for patterns like "out of X", "maximum X", "total X", "X points"
        patterns = [
            r'out of\s+(\d+(?:\.\d+)?)',
            r'maximum\s+(\d+(?:\.\d+)?)',
            r'total\s+(\d+(?:\.\d+)?)',
            r'(\d+(?:\.\d+)?)\s*points',
            r'full\s+(\d+(?:\.\d+)?)',
            r'score\s+is\s+(\d+(?:\.\d+)?)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, grading_guidelines, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue
        
        return None

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

    def _normalize_prediction(self, prediction, response_text: str, max_score: float | None = None) -> str:
        """Normalize and validate the prediction value."""
        # If already a number, convert to string
        if isinstance(prediction, (int, float)):
            # Validate against max_score if provided
            if max_score is not None and prediction > max_score:
                self.log_fn(f"Warning: prediction {prediction} exceeds max_score {max_score}")
            return str(prediction)
        
        # If it's a string, try to parse as number
        if isinstance(prediction, str):
            try:
                num_val = float(prediction)
                if max_score is not None and num_val > max_score:
                    self.log_fn(f"Warning: prediction {num_val} exceeds max_score {max_score}")
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
        
        # Then try the full response text - look for numbers in JSON context first
        json_number_match = re.search(r'"response"\s*:\s*(-?\d+(?:\.\d+)?)', str(response_text))
        if json_number_match:
            return json_number_match.group(1)
        
        # General number extraction as last resort
        number_match = re.search(r'-?\b(\d+(?:\.\d+)?)\b', str(response_text))
        if number_match:
            return number_match.group(1)
        
        return "None"
