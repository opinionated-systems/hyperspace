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
    
    Enhanced to handle nested braces, common JSON errors, and
    attempts to fix common formatting issues before parsing.
    
    This function uses a brace-counting approach to find JSON-like
    structures and applies multiple repair strategies for malformed JSON.
    """
    results = []
    # Try to find JSON-like structures with curly braces
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
                    
                    # Try to parse as-is first
                    try:
                        parsed = json.loads(json_str)
                        if isinstance(parsed, dict):
                            results.append(parsed)
                        continue
                    except json.JSONDecodeError:
                        pass
                    
                    # Apply progressive fixes
                    fixes_applied = []
                    fixed = json_str
                    
                    # Fix 1: Remove trailing commas before closing braces/brackets
                    fixed = re.sub(r',(\s*[}\]])', r'\1', fixed)
                    if fixed != json_str:
                        fixes_applied.append("trailing commas")
                    
                    # Fix 2: Replace single quotes with double quotes (carefully)
                    fixed = re.sub(r"(?<!\\)'", '"', fixed)
                    if fixed != json_str:
                        fixes_applied.append("single quotes")
                    
                    # Fix 3: Remove comments
                    fixed = re.sub(r'//[^\n]*', '', fixed)
                    fixed = re.sub(r'/\*.*?\*/', '', fixed, flags=re.DOTALL)
                    if fixed != json_str:
                        fixes_applied.append("comments")
                    
                    # Fix 4: Handle unquoted keys (simple cases)
                    fixed = re.sub(r'(\{|,\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', fixed)
                    
                    # Fix 5: Handle escaped newlines in strings
                    fixed = fixed.replace('\\n', '\n').replace('\\t', '\t')
                    
                    try:
                        parsed = json.loads(fixed)
                        if isinstance(parsed, dict):
                            results.append(parsed)
                    except json.JSONDecodeError:
                        # Last resort: try to extract key-value pairs manually
                        try:
                            # Look for response/grade field patterns
                            response_match = re.search(r'["\']?(response|grade|answer)["\']?\s*[:=]\s*["\']?([^"\'}\n]+)', json_str, re.IGNORECASE)
                            reasoning_match = re.search(r'["\']?(reasoning|explanation|analysis)["\']?\s*[:=]\s*["\']([^"\']+)', json_str, re.IGNORECASE)
                            
                            if response_match:
                                manual_obj = {
                                    response_match.group(1): response_match.group(2).strip('"\' ,')
                                }
                                if reasoning_match:
                                    manual_obj[reasoning_match.group(1)] = reasoning_match.group(2).strip('"\' ,')
                                results.append(manual_obj)
                        except Exception:
                            pass
                            
                except Exception:
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

Respond in JSON format with the following schema:
<json>
{{
    "reasoning": "Your detailed step-by-step analysis here",
    "response": "Your final grade/assessment here"
}}
</json>

The "response" field should contain only the final grade (e.g., "7", "5", "0", "Correct", "Incorrect", etc.)."""

    def _validate_grade(self, prediction: str) -> tuple[bool, str]:
        """Validate that the prediction is a valid grade format.
        
        Enhanced to handle more edge cases including decimal grades,
        percentage formats, grades with punctuation, and various text formats.
        
        The IMO grading scale is 0-7, so this method normalizes various
        input formats to valid grades in this range.
        
        Args:
            prediction: The raw prediction string from the LLM
            
        Returns:
            (is_valid, cleaned_grade) tuple where cleaned_grade is the
            normalized grade string or "None" if invalid
        """
        if not prediction or prediction == "None":
            return False, "None"
        
        prediction = prediction.strip()
        
        # Remove common punctuation and whitespace that might surround grades
        prediction_clean = prediction.strip(".!?,:;\"'()[]{}<> ")
        
        # Handle empty string after cleaning
        if not prediction_clean:
            return False, "None"
        
        # Check for numeric grades (0-7 for IMO problems)
        if prediction_clean.isdigit():
            grade = int(prediction_clean)
            if 0 <= grade <= 7:
                return True, str(grade)
            # Out of range - could be a different scale, try to normalize
            if 0 <= grade <= 100:
                # Assume percentage and convert
                normalized = round((grade / 100) * 7)
                return True, str(min(7, max(0, normalized)))
            return False, "None"
        
        # Check for decimal grades (e.g., "3.5", "6.0")
        try:
            grade_float = float(prediction_clean)
            if 0 <= grade_float <= 7:
                # Round to nearest valid grade
                grade_int = round(grade_float)
                return True, str(min(7, max(0, grade_int)))
            elif 0 <= grade_float <= 100:
                # Assume percentage and convert
                grade_int = round((grade_float / 100) * 7)
                return True, str(min(7, max(0, grade_int)))
        except ValueError:
            pass
        
        # Check for common grade formats (text-based grades)
        valid_non_numeric = ["correct", "incorrect", "partial", "full", "zero", 
                            "pass", "fail", "true", "false", "yes", "no", 
                            "accepted", "rejected", "valid", "invalid"]
        lower_pred = prediction_clean.lower()
        
        if lower_pred in valid_non_numeric:
            return True, prediction_clean
        
        # Check for fractional grades (e.g., "3/7", "5/7")
        if "/" in prediction_clean:
            parts = prediction_clean.split("/")
            if len(parts) == 2 and parts[0].strip().isdigit() and parts[1].strip().isdigit():
                numerator = int(parts[0].strip())
                denominator = int(parts[1].strip())
                if denominator > 0 and 0 <= numerator <= denominator:
                    # Normalize to 0-7 scale if denominator is not 7
                    if denominator == 7:
                        return True, str(numerator)
                    else:
                        normalized = round((numerator / denominator) * 7)
                        return True, str(min(7, max(0, normalized)))
        
        # Check for percentage grades (e.g., "50%", "100%")
        if "%" in prediction_clean:
            try:
                pct_str = prediction_clean.replace("%", "").strip()
                pct = float(pct_str)
                if 0 <= pct <= 100:
                    # Convert percentage to 0-7 scale
                    grade = round((pct / 100) * 7)
                    return True, str(min(7, max(0, grade)))
            except ValueError:
                pass
        
        # If it looks like a number but has extra text, try to extract valid IMO grade
        import re
        numeric_match = re.search(r'\b([0-7])\b', prediction_clean)
        if numeric_match:
            return True, numeric_match.group(1)
        
        # Try to extract any number and validate it
        any_num_match = re.search(r'\b(\d+(?:\.\d+)?)\b', prediction_clean)
        if any_num_match:
            try:
                num = float(any_num_match.group(1))
                if 0 <= num <= 7:
                    return True, str(int(num))
                elif 0 <= num <= 100:
                    # Assume percentage
                    normalized = round((num / 100) * 7)
                    return True, str(min(7, max(0, normalized)))
            except ValueError:
                pass
        
        return False, "None"

    def _extract_prediction(self, msg_history: list[dict]) -> tuple[str, str]:
        """Extract prediction and reasoning from message history.
        
        Enhanced with better error handling, multiple extraction strategies,
        and detailed logging for debugging extraction failures.
        
        Returns:
            (prediction, reasoning) tuple
        """
        if not msg_history:
            self.log_fn("Warning: Empty message history, cannot extract prediction")
            return "None", ""
        
        last_text = msg_history[-1].get("text", "")
        
        if not last_text:
            self.log_fn("Warning: Last message has no text content")
            return "None", ""
        
        # Try standard extraction first
        extracted = _extract_jsons(last_text)
        extraction_method = "standard"
        
        if not extracted:
            # Try fuzzy extraction as fallback
            extracted = _extract_json_fuzzy(last_text)
            extraction_method = "fuzzy"
        
        if not extracted:
            self.log_fn(f"Warning: Failed to extract JSON from response using any method")
            self.log_fn(f"Response preview: {last_text[:200]}...")
            return "None", ""
        
        self.log_fn(f"Successfully extracted {len(extracted)} JSON object(s) using {extraction_method} method")
        
        last_json = extracted[-1]
        
        # Extract prediction with fallback strategies
        prediction = "None"
        for key in ["response", "grade", "answer", "result", "prediction"]:
            if key in last_json:
                prediction = last_json[key]
                break
        
        # Extract reasoning with fallback strategies  
        reasoning = ""
        for key in ["reasoning", "explanation", "analysis", "thought", "rationale"]:
            if key in last_json:
                reasoning = last_json[key]
                break
        
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
        else:
            if prediction != cleaned_prediction:
                self.log_fn(f"Grade normalized: '{prediction}' -> '{cleaned_prediction}'")
        
        return cleaned_prediction, reasoning

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with enhanced retry logic.

        This method implements a robust grading pipeline with multiple retry
        strategies, progressive error recovery, and detailed logging.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history) where prediction is the grade or "None"
        """
        # Validate required inputs
        required_keys = ["problem", "solution", "grading_guidelines", "student_answer"]
        missing_keys = [k for k in required_keys if not inputs.get(k)]
        if missing_keys:
            self.log_fn(f"Warning: Missing required inputs: {missing_keys}")
        
        instruction = self._build_prompt(inputs)
        
        prediction = "None"
        reasoning = ""
        msg_history = []
        last_error = None
        
        # Progressive hints for retry attempts
        retry_hints = [
            "\n\nIMPORTANT: Your response MUST include a JSON object with a 'response' field containing the grade.",
            "\n\nCRITICAL: Please provide your grade in this exact format:\n<json>\n{\"reasoning\": \"your analysis\", \"response\": \"GRADE_HERE\"}\n</json>",
            "\n\nFINAL ATTEMPT: The grade should be a number from 0-7. Example: <json>{\"response\": \"5\"}</json>",
        ]
        
        for attempt in range(self.max_retries):
            try:
                self.log_fn(f"Attempt {attempt + 1}/{self.max_retries}: Calling LLM for grading...")
                
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=[],
                )
                
                prediction, reasoning = self._extract_prediction(msg_history)
                
                # Validate that we got a meaningful prediction
                if prediction != "None" and prediction.strip():
                    self.log_fn(f"SUCCESS: Extracted grade '{prediction}' on attempt {attempt + 1}")
                    if reasoning:
                        self.log_fn(f"Reasoning: {len(reasoning)} characters")
                    
                    # Log usage info if available
                    usage = info.get("usage", {})
                    if usage:
                        prompt_tokens = usage.get("prompt_tokens", 0)
                        completion_tokens = usage.get("completion_tokens", 0)
                        self.log_fn(f"Token usage: {prompt_tokens} prompt + {completion_tokens} completion")
                    
                    break
                else:
                    self.log_fn(f"Attempt {attempt + 1}: No valid grade extracted, retrying...")
                    # Add progressive hint for next attempt
                    if attempt < len(retry_hints):
                        instruction += retry_hints[attempt]
                    
            except Exception as e:
                last_error = str(e)
                self.log_fn(f"Attempt {attempt + 1}: LLM call failed with error: {e}")
                if attempt == self.max_retries - 1:
                    break
        
        # Log final result with appropriate severity
        if prediction == "None" or not prediction.strip():
            if last_error:
                self.log_fn(f"FAILURE: Could not extract valid grade after {self.max_retries} attempts. Last error: {last_error}")
            else:
                self.log_fn(f"FAILURE: Could not extract valid grade after {self.max_retries} attempts")
        
        return str(prediction), msg_history
