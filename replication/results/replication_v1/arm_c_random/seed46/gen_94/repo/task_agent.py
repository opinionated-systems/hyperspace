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
            except Exception:
                pass
            continue
    return results or None


def _validate_and_normalize_score(score: any, grading_guidelines: str = "") -> any:
    """Validate and normalize the extracted score.
    
    Args:
        score: The extracted score value
        grading_guidelines: The grading guidelines to infer valid score range
        
    Returns:
        Validated and normalized score, or None if invalid
    """
    if score is None:
        return None
    
    # Handle string numbers
    if isinstance(score, str):
        # Try to convert to number
        try:
            score = float(score)
            if score == int(score):
                score = int(score)
        except ValueError:
            # Check for common text patterns
            score_lower = score.lower().strip()
            if score_lower in ['full', 'complete', 'correct', 'all', 'perfect', 'max', 'maximum']:
                return 1
            elif score_lower in ['none', 'zero', 'incorrect', 'wrong', 'no credit', 'fail', 'failed']:
                return 0
            elif score_lower in ['half', 'partial', 'some']:
                return 0.5
            # Try to extract number from string like "5 points" or "score: 3"
            number_match = re.search(r'(-?\d+(?:\.\d+)?)', score)
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
        # Check if it's a valid score (typically 0-10 or 0-1 range)
        # Most IMO problems use 0-7 or 0-1 scoring
        if 0 <= score <= 10:
            return score
        # If outside typical range, might be a percentage
        if 0 <= score <= 100:
            return score / 100.0 * 7  # Convert to 0-7 scale
        # Handle negative scores (should be 0)
        if score < 0:
            return 0
    
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
            continue

    # Pattern 2: Try to parse the entire text as JSON
    if not results:
        try:
            obj = json.loads(text.strip())
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
        score_context_pattern = r'(?:score|grade|points?|mark|value)\s*(?:is|=|:)?\s*(\d+(?:\.\d+)?)'
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
            number_matches = re.finditer(r'\b(\d+(?:\.\d+)?)\b', text)
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

## Grading Examples:

Example 1 - Full Credit (Binary 0-1 scale):
Problem: What is 2 + 2?
Student Answer: 4
Score: 1
Reasoning: The student provided the correct answer.

Example 2 - Partial Credit (Binary 0-1 scale):
Problem: Solve x² = 4
Student Answer: x = 2
Score: 0.5
Reasoning: The student found one correct solution but missed x = -2.

Example 3 - No Credit (Binary 0-1 scale):
Problem: Find the derivative of x²
Student Answer: x³
Score: 0
Reasoning: The student applied the wrong rule (power rule incorrectly).

Example 4 - IMO-style Grading (0-7 scale):
Problem: Prove that for any positive integer n, n³ + 2n is divisible by 3.
Student Answer: Complete proof with base case n=1, induction hypothesis, and induction step.
Score: 7
Reasoning: Full marks for complete and correct proof.

Example 5 - Partial IMO Credit (0-7 scale):
Problem: Prove that for any positive integer n, n³ + 2n is divisible by 3.
Student Answer: Only proved the base case n=1.
Score: 1
Reasoning: Partial credit for correct base case but missing induction step.

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
- Score ranges depend on the problem type:
  * Binary/short answer problems: typically 0-1 scale
  * IMO competition problems: typically 0-7 scale
  * Other problems: 0-10 scale or as specified in grading guidelines
- If the grading guidelines specify a maximum score, use that scale
- When in doubt, use the scale implied by the problem difficulty and type"""

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
            prediction = _validate_and_normalize_score(raw_response, grading_guidelines)
            if prediction is None:
                self.log_fn(f"Invalid score extracted: {raw_response}, using raw value")
                prediction = raw_response
        
        if prediction is None:
            prediction = "None"
            
        self.log_fn(f"Extraction method used: {extraction_method}, raw: {raw_response}, normalized: {prediction}")
        return str(prediction), msg_history
