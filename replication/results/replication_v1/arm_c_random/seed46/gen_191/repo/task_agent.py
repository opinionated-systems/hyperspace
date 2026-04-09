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
    Also handles markdown code blocks and plain JSON.
    """
    results = []
    search_from = 0
    
    # First, try to find <json> tags
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
                # Handle both numeric and string values, including negative numbers and decimals
                response_match = re.search(r'"response"\s*:\s*(-?\d+(?:\.\d+)?|"[^"]*"|\[[^\]]*\]|\{[^}]*\})', inner, re.DOTALL)
                if response_match:
                    value = response_match.group(1).strip()
                    # Try to parse as number or string
                    try:
                        parsed_value = json.loads(value)
                    except json.JSONDecodeError:
                        # If it's not valid JSON, treat as string
                        parsed_value = value.strip('"').strip("'")
                    results.append({"response": parsed_value})
            except Exception:
                pass
            continue
    
    # If no <json> tags found, try markdown code blocks
    if not results:
        # Look for ```json or ``` code blocks
        md_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
        for match in re.finditer(md_pattern, text, re.DOTALL):
            inner = match.group(1).strip()
            try:
                obj = json.loads(inner)
                if isinstance(obj, dict) and "response" in obj:
                    results.append(obj)
            except json.JSONDecodeError:
                # Try to extract response field from malformed JSON
                try:
                    response_match = re.search(r'"response"\s*:\s*(-?\d+(?:\.\d+)?|"[^"]*")', inner, re.DOTALL)
                    if response_match:
                        value = response_match.group(1).strip()
                        try:
                            parsed_value = json.loads(value)
                        except json.JSONDecodeError:
                            parsed_value = value.strip('"').strip("'")
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
                # Try to extract just the response field with improved pattern
                try:
                    # Handle negative numbers, decimals, strings, arrays, and objects
                    response_match = re.search(r'"response"\s*:\s*(-?\d+(?:\.\d+)?|"[^"]*"|\[[^\]]*\]|\{[^}]*\})', obj_str, re.DOTALL)
                    if response_match:
                        value = response_match.group(1).strip()
                        try:
                            parsed_value = json.loads(value)
                        except json.JSONDecodeError:
                            parsed_value = value.strip('"').strip("'")
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
        # Prioritize numbers that appear after keywords like "score", "grade", "points"
        score_patterns = [
            r'(?:score|grade|points|mark|rating)\s*(?:is|of|:|=)\s*(-?\d+(?:\.\d+)?)',
            r'(?:score|grade|points|mark|rating)\s*(-?\d+(?:\.\d+)?)',
            r'(-?\d+(?:\.\d+)?)\s*(?:points?|marks?|score|grade)',
        ]
        for pattern in score_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    value = float(match.group(1))
                    if value == int(value):
                        value = int(value)
                    results.append({"response": value})
                    break
                except ValueError:
                    pass
        
        # If still no results, try to find any number
        if not results:
            number_matches = list(re.finditer(r'\b(-?\d+(?:\.\d+)?)\b', text))
            for number_match in number_matches:
                try:
                    value = float(number_match.group(1))
                    if value == int(value):
                        value = int(value)
                    results.append({"response": value})
                    break  # Take the first valid number found
                except ValueError:
                    pass

    return results or None


class TaskAgent:
    """Task agent that solves IMO grading problems with robust JSON extraction and self-verification."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def _build_grading_prompt(self, domain: str, problem: str, solution: str, 
                              grading_guidelines: str, student_answer: str) -> str:
        """Build a comprehensive grading prompt with clear instructions."""
        return f"""You are an expert mathematics grader specializing in {domain} problems.

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
2. Consider the grading guidelines when determining the score
3. Check for partial credit - students may have correct reasoning even if the final answer is wrong
4. Look for common errors and apply appropriate deductions
5. The score should be a number (integer or decimal)
6. Be objective and consistent with the grading guidelines
7. If the maximum score is specified in the guidelines, do not exceed it
8. If no maximum is specified, use your judgment but typically scores range from 0 to the problem's difficulty level

## Grading Process:
Step 1 - Analysis: Identify what the student got right (method, steps, final answer)
Step 2 - Error Detection: Identify any errors, omissions, or misconceptions
Step 3 - Partial Credit: Determine what partial credit should be awarded for correct reasoning
Step 4 - Score Calculation: Apply the grading guidelines to calculate the final score
Step 5 - Verification: Double-check that your score is consistent with the guidelines

## Output Format:
Respond ONLY in the following JSON format. Do not include any other text outside the JSON tags:
<json>
{{
    "response": <numerical_score>
}}
</json>

The response field must contain only the numerical score value (e.g., 5, 7.5, 10, 0)."""

    def _extract_score(self, response_text: str) -> tuple[str, str]:
        """Extract score from response text using multiple methods.
        
        Returns:
            (prediction, extraction_method)
        """
        prediction = "None"
        extraction_method = "primary"
        
        # Try primary extraction
        try:
            extracted = _extract_jsons(response_text)
            if extracted and len(extracted) > 0 and "response" in extracted[-1]:
                prediction = extracted[-1]["response"]
                return prediction, extraction_method
        except Exception as e:
            self.log_fn(f"Primary extraction failed: {e}")
        
        # Try fallback extraction
        extraction_method = "fallback"
        try:
            extracted = _extract_json_fallback(response_text)
            if extracted and len(extracted) > 0 and "response" in extracted[-1]:
                prediction = extracted[-1]["response"]
                return prediction, extraction_method
        except Exception as e:
            self.log_fn(f"Fallback extraction failed: {e}")
        
        return prediction, "failed"

    def _validate_prediction(self, prediction) -> float | int | None:
        """Validate and normalize a prediction value.
        
        Returns:
            Validated number or None if invalid
        """
        try:
            if isinstance(prediction, (int, float)):
                # Ensure it's a valid number (not NaN or Inf)
                if isinstance(prediction, float):
                    if prediction != prediction or prediction == float('inf') or prediction == float('-inf'):
                        return None
                return prediction
            elif isinstance(prediction, str):
                # Clean up the string and try to parse as number
                cleaned = prediction.strip().strip('"').strip("'")
                if cleaned:
                    float_val = float(cleaned)
                    # Check for NaN/Inf
                    if float_val != float_val or float_val == float('inf') or float_val == float('-inf'):
                        return None
                    # Use int if it's a whole number
                    if float_val == int(float_val):
                        return int(float_val)
                    return float_val
        except (ValueError, TypeError) as e:
            self.log_fn(f"Validation error for prediction '{prediction}': {e}")
        return None

    def _emergency_extraction(self, response_text: str) -> float | int | None:
        """Emergency extraction using regex to find any number in the response."""
        # Look for patterns that suggest a score
        score_patterns = [
            r'(?:score|grade|points|mark|rating)\s*(?:is|of|:|=)\s*(-?\d+(?:\.\d+)?)',
            r'(?:score|grade|points|mark|rating)\s*(-?\d+(?:\.\d+)?)',
            r'(-?\d+(?:\.\d+)?)\s*(?:points?|marks?|score|grade)',
        ]
        for pattern in score_patterns:
            match = re.search(pattern, response_text, re.IGNORECASE)
            if match:
                try:
                    value = float(match.group(1))
                    if value == int(value):
                        return int(value)
                    return value
                except ValueError:
                    pass
        
        # Last resort: find any number
        number_match = re.search(r'\b(-?\d+(?:\.\d+)?)\b', str(response_text))
        if number_match:
            try:
                val = float(number_match.group(1))
                if val == int(val):
                    return int(val)
                return val
            except ValueError:
                pass
        return None

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with self-verification.

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

        # Build the grading prompt
        instruction = self._build_grading_prompt(
            domain, problem, solution, grading_guidelines, student_answer
        )

        # First LLM call for initial grading
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

        response_text = msg_history[-1]["text"] if msg_history else ""
        
        # Extract prediction
        prediction, extraction_method = self._extract_score(response_text)
        
        # Validate the prediction
        validated_prediction = self._validate_prediction(prediction)
        
        # If validation failed, try emergency extraction
        if validated_prediction is None:
            self.log_fn(f"Invalid prediction value: {prediction}")
            validated_prediction = self._emergency_extraction(response_text)
            if validated_prediction is not None:
                extraction_method = "emergency_regex"

        # Self-verification step: ask the model to verify its grading
        if validated_prediction is not None:
            verification_prompt = f"""You previously graded a student's answer with a score of {validated_prediction}.

Please verify this score is correct by reviewing the grading guidelines again:

## Grading Guidelines:
{grading_guidelines}

## Student's Answer:
{student_answer}

Is the score {validated_prediction} consistent with these guidelines? 
If yes, respond with the same score. If no, provide the corrected score.

Respond ONLY in the following JSON format:
<json>
{{
    "response": <numerical_score>
}}
</json>"""
            
            try:
                verify_response, verify_history, verify_info = get_response_from_llm(
                    msg=verification_prompt,
                    model=self.model,
                    msg_history=[],
                )
                verify_text = verify_history[-1]["text"] if verify_history else ""
                
                # Extract verification score
                verify_prediction, verify_method = self._extract_score(verify_text)
                verify_validated = self._validate_prediction(verify_prediction)
                
                # If verification gives a different valid score, use it (it's likely more accurate)
                if verify_validated is not None and verify_validated != validated_prediction:
                    self.log_fn(f"Verification changed score from {validated_prediction} to {verify_validated}")
                    validated_prediction = verify_validated
                    extraction_method = f"verification_{verify_method}"
                    # Append verification history to msg_history for logging
                    msg_history.extend(verify_history)
            except Exception as e:
                self.log_fn(f"Verification step failed: {e}")
                # Continue with original prediction

        # Finalize prediction
        if validated_prediction is None:
            prediction = "None"
        else:
            prediction = str(validated_prediction)

        self.log_fn(f"Extraction method used: {extraction_method}, prediction: {prediction}")
        return str(prediction), msg_history
