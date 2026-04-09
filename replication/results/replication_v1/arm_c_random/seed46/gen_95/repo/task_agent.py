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
        
        # Try to parse the full JSON first
        try:
            results.append(json.loads(inner))
            continue
        except json.JSONDecodeError:
            pass
        
        # Try to fix common JSON issues and re-parse
        # Issue 1: Trailing commas
        fixed_inner = re.sub(r',\s*}', '}', inner)
        fixed_inner = re.sub(r',\s*]', ']', fixed_inner)
        
        # Issue 2: Single quotes instead of double quotes
        fixed_inner = re.sub(r"'([^']*?)'\s*:", r'"\1":', fixed_inner)
        fixed_inner = re.sub(r":\s*'([^']*?)'", r': "\1"', fixed_inner)
        
        try:
            results.append(json.loads(fixed_inner))
            continue
        except json.JSONDecodeError:
            pass
        
        # Try to extract just the response field if full JSON fails
        try:
            # Look for "response": value pattern (handles numbers, strings, arrays)
            # More robust pattern that handles nested objects and various formats
            response_match = re.search(r'"response"\s*:\s*([^,\}]+|"[^"]*"|\[[^\]]*\]|\{[^}]*\})', inner, re.DOTALL)
            if response_match:
                value = response_match.group(1).strip()
                # Try to parse as JSON value
                try:
                    parsed_value = json.loads(value)
                except json.JSONDecodeError:
                    # If it's not valid JSON, treat as string
                    parsed_value = value.strip('"')
                results.append({"response": parsed_value})
                continue
        except Exception:
            pass
        
        # Last resort: look for any number that could be a score
        try:
            # Look for patterns like "response": 5 or "response": 5.5
            number_match = re.search(r'"response"\s*:\s*(-?\d+(?:\.\d+)?)', inner, re.IGNORECASE)
            if number_match:
                value = float(number_match.group(1))
                if value == int(value):
                    value = int(value)
                results.append({"response": value})
                continue
        except Exception:
            pass
            
    return results or None


def _detect_score_scale(grading_guidelines: str, problem: str = "") -> tuple[float, float]:
    """Detect the appropriate score scale from grading guidelines and problem.
    
    Returns:
        (min_score, max_score) tuple
    """
    text = (grading_guidelines + " " + problem).lower()
    
    # Check for IMO-style 0-7 scale (check before 0-1 to avoid false matches)
    if any(pattern in text for pattern in ['0-7', '0 to 7', 'full marks 7', 'score 7']):
        return (0, 7)
    
    # Check for IMO/competition keywords
    if 'imo' in text or 'competition' in text:
        return (0, 7)
    
    # Check for 0-100 scale (percentage) - check before smaller scales
    if any(pattern in text for pattern in ['percentage', 'percent', '0-100', '0 to 100', 'out of 100', '/100']):
        return (0, 100)
    
    # Check for 0-10 scale
    if any(pattern in text for pattern in ['0-10', '0 to 10', 'out of 10', '/10', '10 points']):
        return (0, 10)
    
    # Check for 0-5 scale
    if any(pattern in text for pattern in ['0-5', '0 to 5', 'out of 5', '/5', '5 points']):
        return (0, 5)
    
    # Check for binary 0-1 scale (check last to avoid false matches)
    if any(pattern in text for pattern in ['binary', '0-1', '0 to 1', '0 or 1', 'correct/incorrect']):
        return (0, 1)
    
    # Default to 0-1 scale for most problems
    return (0, 1)


def _validate_and_normalize_score(score: any, grading_guidelines: str = "", problem: str = "") -> any:
    """Validate and normalize the extracted score.
    
    Args:
        score: The extracted score value
        grading_guidelines: The grading guidelines to infer valid score range
        problem: The problem statement for additional context
        
    Returns:
        Validated and normalized score, or None if invalid
    """
    if score is None:
        return None
    
    # Detect the expected scale
    min_scale, max_scale = _detect_score_scale(grading_guidelines, problem)
    
    # Handle string numbers
    if isinstance(score, str):
        score_str = score.strip()
        score_lower = score_str.lower()
        
        # Check for common text patterns first
        if score_lower in ['full', 'complete', 'correct', 'all', 'perfect', 'max', 'maximum', 'true', 'yes']:
            return max_scale
        elif score_lower in ['none', 'zero', 'incorrect', 'wrong', 'no credit', 'fail', 'failed', 'false', 'no']:
            return min_scale
        elif score_lower in ['half', 'partial', 'some']:
            return (min_scale + max_scale) / 2
        
        # Try to convert to number
        try:
            # Remove any trailing text like "points" or "marks"
            cleaned = re.sub(r'\s*(?:points?|marks?|score|grade)\s*$', '', score_str, flags=re.IGNORECASE)
            score = float(cleaned)
            if score == int(score):
                score = int(score)
        except ValueError:
            # Try to extract number from string like "5 points" or "score: 3"
            number_match = re.search(r'(-?\d+(?:\.\d+)?)', score_str)
            if number_match:
                try:
                    score = float(number_match.group(1))
                    if score == int(score):
                        score = int(score)
                except ValueError:
                    return None
            else:
                return None
    
    # Handle numeric types
    if isinstance(score, (int, float)):
        # Clamp negative scores to minimum
        if score < 0:
            return min_scale
        
        # If score is within expected scale, return as-is
        if min_scale <= score <= max_scale:
            return score
        
        # If score is larger than expected max, cap it at max_scale
        if score > max_scale:
            return max_scale
        
        return score
    
    return None


def _extract_json_fallback(text: str) -> list[dict] | None:
    """Fallback JSON extraction using regex for unwrapped JSON objects.

    This handles cases where the model outputs valid JSON without <json> tags.
    Searches for JSON objects with a "response" key.
    """
    results = []
    
    # Pattern 1: Match JSON objects with response key (handles nested braces more robustly)
    # Use a stack-based approach to handle nested braces
    pattern = r'\{[^{}]*"response"[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.finditer(pattern, text, re.DOTALL)

    for match in matches:
        try:
            obj = json.loads(match.group())
            if "response" in obj:
                results.append(obj)
        except json.JSONDecodeError:
            # Try to fix common JSON issues
            fixed = match.group()
            fixed = re.sub(r',\s*}', '}', fixed)
            fixed = re.sub(r',\s*]', ']', fixed)
            try:
                obj = json.loads(fixed)
                if "response" in obj:
                    results.append(obj)
            except json.JSONDecodeError:
                continue

    # Pattern 2: Try to parse the entire text as JSON
    if not results:
        try:
            obj = json.loads(text.strip())
            if isinstance(obj, dict) and "response" in obj:
                results.append(obj)
        except json.JSONDecodeError:
            # Try fixing common issues
            try:
                fixed = text.strip()
                fixed = re.sub(r',\s*}', '}', fixed)
                fixed = re.sub(r',\s*]', ']', fixed)
                obj = json.loads(fixed)
                if isinstance(obj, dict) and "response" in obj:
                    results.append(obj)
            except json.JSONDecodeError:
                pass

    # Pattern 3: Look for JSON with response field using a more flexible pattern
    if not results:
        # Look for patterns like {"response": 5} or {"response": "5"} or {"response": 5.5}
        flexible_pattern = r'\{\s*"response"\s*:\s*([^\}]+)\s*\}'
        for match in re.finditer(flexible_pattern, text, re.DOTALL):
            try:
                # Reconstruct the JSON
                json_str = '{"response": ' + match.group(1) + '}'
                obj = json.loads(json_str)
                if "response" in obj:
                    results.append(obj)
            except json.JSONDecodeError:
                # Try extracting just the numeric value
                try:
                    value_str = match.group(1).strip()
                    # Remove quotes if present
                    if value_str.startswith('"') and value_str.endswith('"'):
                        value_str = value_str[1:-1]
                    # Try to parse as number
                    value = float(value_str)
                    if value == int(value):
                        value = int(value)
                    results.append({"response": value})
                except (ValueError, json.JSONDecodeError):
                    continue

    # Pattern 4: Look for response: value patterns (without quotes on key)
    if not results:
        unquoted_pattern = r'response\s*:\s*([^\n\},]+)'
        for match in re.finditer(unquoted_pattern, text, re.IGNORECASE):
            try:
                value_str = match.group(1).strip()
                # Try to parse as number
                try:
                    value = float(value_str)
                    if value == int(value):
                        value = int(value)
                    results.append({"response": value})
                    break
                except ValueError:
                    # Try as string (remove quotes if present)
                    value = value_str.strip('"\'')
                    results.append({"response": value})
                    break
            except Exception:
                continue

    # Pattern 5: Look for "score" or "grade" fields as alternatives
    if not results:
        score_pattern = r'(?:score|grade)\s*[=:]\s*([^\n\},]+)'
        for match in re.finditer(score_pattern, text, re.IGNORECASE):
            try:
                value_str = match.group(1).strip()
                # Try to parse as number
                try:
                    value = float(value_str)
                    if value == int(value):
                        value = int(value)
                    results.append({"response": value})
                    break
                except ValueError:
                    pass
            except Exception:
                continue

    # Pattern 6: Look for standalone numbers in code blocks or after specific markers
    if not results:
        # Look for numbers in code blocks
        code_block_pattern = r'```(?:json)?\s*\n?\s*(\d+(?:\.\d+)?)\s*\n?```'
        for match in re.finditer(code_block_pattern, text, re.IGNORECASE):
            try:
                value = float(match.group(1))
                if value == int(value):
                    value = int(value)
                results.append({"response": value})
                break
            except ValueError:
                continue

    # Pattern 7: Last resort - extract standalone numbers that look like scores
    if not results:
        # Look for patterns like "The score is 5" or just a number
        # Try to find a number that appears to be a score (0-10 range typically)
        # First, try to find numbers after common score-related words
        score_context_pattern = r'(?:score|grade|points?|mark|value)\s*(?:is|=|:)?\s*(-?\d+(?:\.\d+)?)'
        for match in re.finditer(score_context_pattern, text, re.IGNORECASE):
            try:
                value = float(match.group(1))
                if value == int(value):
                    value = int(value)
                results.append({"response": value})
                break
            except ValueError:
                continue
        
        # If still no results, take the first number in the text that looks like a score
        if not results:
            number_matches = re.finditer(r'\b(-?\d+(?:\.\d+)?)\b', text)
            for match in number_matches:
                try:
                    value = float(match.group(1))
                    # Only accept values that look like scores (0-100 range)
                    if 0 <= value <= 100:
                        if value == int(value):
                            value = int(value)
                        results.append({"response": value})
                        break  # Take the first valid number found
                except ValueError:
                    continue
    
    # Pattern 8: Look for the word "Score:" followed by a number (common in reasoning)
    if not results:
        score_line_pattern = r'^\s*Score:\s*(-?\d+(?:\.\d+)?)'
        for match in re.finditer(score_line_pattern, text, re.MULTILINE | re.IGNORECASE):
            try:
                value = float(match.group(1))
                if value == int(value):
                    value = int(value)
                results.append({"response": value})
                break
            except ValueError:
                continue

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
        
        # Detect the score scale to include in the prompt
        min_scale, max_scale = _detect_score_scale(grading_guidelines, problem)
        
        # Determine scale description for the prompt
        if max_scale == 7:
            scale_desc = "0-7 (IMO competition style)"
            scale_examples = """Example - IMO Full Credit (Score: 7):
Complete and correct proof with all necessary steps.

Example - IMO Partial Credit (Score: 1-6):
Partial progress toward solution (correct base case, key insight, etc.).

Example - IMO No Credit (Score: 0):
No meaningful progress or incorrect approach."""
        elif max_scale == 1:
            scale_desc = "0-1 (binary/correct-incorrect)"
            scale_examples = """Example - Full Credit (Score: 1):
Correct answer with valid reasoning.

Example - No Credit (Score: 0):
Incorrect answer or no valid reasoning."""
        elif max_scale == 10:
            scale_desc = "0-10"
            scale_examples = """Example - Full Credit (Score: 10):
Complete and correct solution.

Example - Partial Credit (Score: 1-9):
Partial solution with some correct elements."""
        elif max_scale == 100:
            scale_desc = "0-100 (percentage)"
            scale_examples = """Example - Full Credit (Score: 100):
Complete and correct solution.

Example - Partial Credit (Score: 1-99):
Partial solution with some correct elements."""
        else:
            scale_desc = f"{min_scale}-{max_scale}"
            scale_examples = """Example - Full Credit (Maximum score):
Complete and correct solution.

Example - Partial Credit (Intermediate score):
Partial solution with some correct elements."""

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

## Score Scale:
Use the {scale_desc} scale for this problem.

{scale_examples}

## Instructions:
1. Read the problem statement carefully to understand what is being asked
2. Review the correct solution to understand the expected approach and answer
3. Analyze the student's answer step by step:
   - Check if the student understood the problem correctly
   - Verify the mathematical reasoning and calculations
   - Identify any errors or misconceptions
   - Note any partial credit-worthy steps
4. Apply the grading guidelines to determine the appropriate score
5. Consider common grading scenarios:
   - Full credit: Correct answer with valid reasoning
   - Partial credit: Correct approach with minor errors, or incomplete solution
   - No credit: Incorrect approach or no valid mathematical reasoning

## Response Format:
You must respond with a JSON object containing only the numerical score in the "response" field.

<json>
{{
    "response": <numerical_score>
}}
</json>

CRITICAL INSTRUCTIONS:
- The response field must contain ONLY a number (integer or decimal)
- Do NOT include any text, explanations, or units in the response field
- Do NOT include quotes around the number in the response field
- Ensure the JSON is valid and properly formatted
- Use the {scale_desc} scale for this problem
- Score must be between {min_scale} and {max_scale}
- Return ONLY the JSON object, no additional text before or after"""

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
        prediction = None
        extraction_method = "primary"
        raw_response = None
        
        # Get the assistant's response text
        assistant_text = ""
        for msg in reversed(msg_history):
            if msg.get("role") == "assistant":
                assistant_text = msg.get("text", "")
                break
        
        # If no assistant message found, use the last message
        if not assistant_text and msg_history:
            assistant_text = msg_history[-1].get("text", "")
        
        try:
            extracted = _extract_jsons(assistant_text)
            if extracted and len(extracted) > 0 and "response" in extracted[-1]:
                raw_response = extracted[-1]["response"]
            else:
                # Try fallback extraction
                extracted = _extract_json_fallback(assistant_text)
                if extracted and len(extracted) > 0 and "response" in extracted[-1]:
                    raw_response = extracted[-1]["response"]
                    extraction_method = "fallback"
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            # Try fallback on exception
            try:
                extracted = _extract_json_fallback(assistant_text)
                if extracted and len(extracted) > 0 and "response" in extracted[-1]:
                    raw_response = extracted[-1]["response"]
                    extraction_method = "fallback"
            except Exception as fallback_e:
                self.log_fn(f"Fallback extraction also failed: {fallback_e}")

        # Validate and normalize the score
        if raw_response is not None:
            prediction = _validate_and_normalize_score(raw_response, grading_guidelines, problem)
            if prediction is None:
                self.log_fn(f"Invalid score extracted: {raw_response}, using raw value")
                prediction = raw_response
        
        if prediction is None:
            prediction = "None"
            
        self.log_fn(f"Extraction method used: {extraction_method}, raw: {raw_response}, normalized: {prediction}")
        return str(prediction), msg_history
