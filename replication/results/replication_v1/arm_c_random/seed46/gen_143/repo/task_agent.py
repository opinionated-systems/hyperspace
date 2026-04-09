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
    
    # Also look for markdown code blocks with json
    if not results:
        code_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
        for match in re.finditer(code_block_pattern, text, re.DOTALL):
            inner = match.group(1).strip()
            try:
                obj = json.loads(inner)
                if isinstance(obj, dict) and "response" in obj:
                    results.append(obj)
            except json.JSONDecodeError:
                # Try to extract response field from malformed JSON
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
    Also handles single-quoted JSON and various malformed formats.
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
                # Try to fix common JSON issues
                try:
                    # Replace single quotes with double quotes
                    fixed = obj_str.replace("'", '"')
                    obj = json.loads(fixed)
                    if "response" in obj:
                        results.append(obj)
                        continue
                except json.JSONDecodeError:
                    pass
                
                # Try to extract just the response field
                try:
                    # Handle both double and single quotes
                    response_match = re.search(r'["\']response["\']\s*:\s*([^,\}]+|["\'][^"\']*["\'])', obj_str, re.DOTALL)
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
            # Try with single quote replacement
            try:
                fixed = text.strip().replace("'", '"')
                obj = json.loads(fixed)
                if isinstance(obj, dict) and "response" in obj:
                    results.append(obj)
            except json.JSONDecodeError:
                pass

    # Look for patterns like "response": 5 or "response": "5" anywhere in text
    if not results:
        response_patterns = [
            r'["\']?response["\']?\s*[:=]\s*(\d+(?:\.\d+)?)',
            r'["\']?response["\']?\s*[:=]\s*["\'](\d+(?:\.\d+)?)["\']',
        ]
        for pattern in response_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    value = match.group(1)
                    if '.' in value:
                        parsed_value = float(value)
                    else:
                        parsed_value = int(value)
                    results.append({"response": parsed_value})
                    break
                except ValueError:
                    pass

    # Last resort: try to extract just a number or string response
    if not results:
        # Look for patterns like "The score is 5" or just a number
        # Try to find a number that appears to be a score (0-100 range typically)
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
        
        # Try to extract max score from grading guidelines
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
2. Check if the student has shown their work and reasoning
3. Identify any errors or misconceptions in the student's approach
4. Consider the grading guidelines when determining the score
5. Award partial credit for correct steps even if the final answer is wrong
6. Provide a numerical score as your response

## Scoring Rules:
- The maximum possible score is {max_score}
- Award full points ({max_score}) for a completely correct solution with proper reasoning
- Deduct points for: missing steps, calculation errors, incorrect final answers, lack of justification
- Award partial credit for demonstrating understanding of key concepts
- The minimum score is 0

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

        # Validate and normalize the prediction
        prediction = self._normalize_prediction(prediction, response_text, max_score)

        self.log_fn(f"Extraction method used: {extraction_method}, prediction: {prediction}")
        return str(prediction), msg_history

    def _extract_max_score(self, grading_guidelines: str) -> int:
        """Extract the maximum possible score from grading guidelines.
        
        Looks for patterns like "out of X", "maximum X points", "X points total",
        or simply the largest number mentioned in the context of scoring.
        """
        if not grading_guidelines:
            return 10  # Default assumption
        
        # Common patterns for max score
        patterns = [
            r'out of\s+(\d+)',
            r'maximum\s+(\d+)\s+points?',
            r'(?:total|full)\s+(\d+)\s+points?',
            r'(\d+)\s+points?\s+(?:total|maximum|possible)',
            r'worth\s+(\d+)\s+points?',
            r'\/(\d+)\s*(?:points?)?',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, grading_guidelines, re.IGNORECASE)
            if match:
                try:
                    return int(match.group(1))
                except ValueError:
                    pass
        
        # Look for the largest number that appears to be a score
        numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', grading_guidelines)
        if numbers:
            try:
                # Filter reasonable score values (1-100)
                valid_scores = [float(n) for n in numbers if 1 <= float(n) <= 100]
                if valid_scores:
                    return int(max(valid_scores))
            except ValueError:
                pass
        
        return 10  # Default

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

    def _normalize_prediction(self, prediction, response_text: str, max_score: int = 10) -> str:
        """Normalize and validate the prediction value."""
        # If already a number, convert to string
        if isinstance(prediction, (int, float)):
            # Clamp to valid range
            prediction = max(0, min(float(prediction), max_score))
            # Return as int if it's a whole number
            if prediction == int(prediction):
                return str(int(prediction))
            return str(prediction)
        
        # If it's a string, try to parse as number
        if isinstance(prediction, str):
            try:
                val = float(prediction)
                # Clamp to valid range
                val = max(0, min(val, max_score))
                # Return as int if it's a whole number
                if val == int(val):
                    return str(int(val))
                return str(val)
            except (ValueError, TypeError):
                pass
        
        # Try to extract any number from the prediction or response text
        self.log_fn(f"Invalid prediction value: {prediction}, attempting regex extraction")
        
        # First try the prediction string itself
        if isinstance(prediction, str):
            number_match = re.search(r'-?\b(\d+(?:\.\d+)?)\b', prediction)
            if number_match:
                try:
                    val = float(number_match.group(1))
                    val = max(0, min(val, max_score))
                    if val == int(val):
                        return str(int(val))
                    return str(val)
                except ValueError:
                    pass
        
        # Then try the full response text - look for numbers that could be scores
        # Prioritize numbers that appear after keywords like "score", "grade", "points"
        score_patterns = [
            r'(?:score|grade|points?|mark)[\s:=]+(\d+(?:\.\d+)?)',
            r'(?:is|equals?|[:=])\s*(\d+(?:\.\d+)?)',
            r'\b(\d+(?:\.\d+)?)\s*(?:points?|marks?|score)',
        ]
        
        for pattern in score_patterns:
            match = re.search(pattern, str(response_text), re.IGNORECASE)
            if match:
                try:
                    val = float(match.group(1))
                    val = max(0, min(val, max_score))
                    if val == int(val):
                        return str(int(val))
                    return str(val)
                except ValueError:
                    pass
        
        # Last resort: any number in the text
        number_match = re.search(r'-?\b(\d+(?:\.\d+)?)\b', str(response_text))
        if number_match:
            try:
                val = float(number_match.group(1))
                val = max(0, min(val, max_score))
                if val == int(val):
                    return str(int(val))
                return str(val)
            except ValueError:
                pass
        
        return "None"
