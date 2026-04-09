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


def _extract_json_with_retry(text: str, max_retries: int = 3) -> list[dict] | None:
    """Extract JSON with multiple fallback strategies.
    
    Tries multiple extraction methods in order of reliability:
    1. Standard <json>...</json> blocks
    2. JSON code blocks ```json...```
    3. Raw JSON objects in text (with brace tracking)
    4. Fuzzy extraction for malformed JSON
    
    Added logging for debugging extraction failures.
    """
    if not text or not text.strip():
        logger.debug("JSON extraction failed: empty text input")
        return None
    
    text_preview = text[:200].replace('\n', ' ')
    logger.debug(f"Attempting JSON extraction from text: {text_preview}...")
        
    # Try standard extraction first
    results = _extract_jsons(text)
    if results:
        logger.debug(f"Standard <json> extraction succeeded, found {len(results)} objects")
        return results
    
    # Try JSON code blocks
    results = []
    json_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    matches = re.findall(json_block_pattern, text, re.DOTALL)
    logger.debug(f"Found {len(matches)} JSON code blocks")
    for match in matches:
        try:
            results.append(json.loads(match.strip()))
        except json.JSONDecodeError:
            # Try to fix common JSON issues
            fixed = _fix_common_json_issues(match.strip())
            try:
                results.append(json.loads(fixed))
                logger.debug("Fixed and parsed JSON from code block")
            except json.JSONDecodeError:
                continue
    
    if results:
        logger.debug(f"JSON code block extraction succeeded, found {len(results)} objects")
        return results
    
    # Try to find raw JSON objects (objects with curly braces)
    results = _extract_raw_json_objects(text)
    if results:
        logger.debug(f"Raw JSON object extraction succeeded, found {len(results)} objects")
        return results
    
    # Final fallback: try to extract any JSON-like structure
    results = _extract_fuzzy_json(text)
    if results:
        logger.debug(f"Fuzzy JSON extraction succeeded as last resort")
    else:
        logger.debug(f"All JSON extraction methods failed for text: {text_preview}...")
    
    return results


def _fix_common_json_issues(text: str) -> str:
    """Fix common JSON formatting issues."""
    # Remove trailing commas before closing braces/brackets
    text = re.sub(r',(\s*[}\]])', r'\1', text)
    # Fix single quotes to double quotes (simple cases)
    text = re.sub(r"(?<!\\)'", '"', text)
    return text


def _extract_raw_json_objects(text: str) -> list[dict] | None:
    """Extract raw JSON objects by tracking brace depth."""
    results = []
    potential_objects = []
    start_indices = []
    brace_depth = 0
    in_string = False
    escape_next = False
    
    for i, char in enumerate(text):
        if escape_next:
            escape_next = False
            continue
        if char == '\\':
            escape_next = True
            continue
        if char == '"' and not in_string:
            in_string = True
        elif char == '"' and in_string:
            in_string = False
        elif not in_string:
            if char == '{':
                if brace_depth == 0:
                    start_indices.append(i)
                brace_depth += 1
            elif char == '}':
                brace_depth -= 1
                if brace_depth == 0 and start_indices:
                    start = start_indices.pop()
                    potential_objects.append(text[start:i+1])
    
    for obj_str in potential_objects:
        try:
            results.append(json.loads(obj_str.strip()))
        except json.JSONDecodeError:
            # Try fixing common issues
            fixed = _fix_common_json_issues(obj_str.strip())
            try:
                results.append(json.loads(fixed))
            except json.JSONDecodeError:
                continue
    
    return results or None


def _extract_fuzzy_json(text: str) -> list[dict] | None:
    """Fuzzy extraction for malformed JSON - last resort.
    
    Enhanced to handle more edge cases including:
    - Single quotes instead of double quotes
    - Missing quotes around keys
    - Numbers with surrounding text
    - Partial JSON fragments
    """
    results = []
    
    # Look for patterns like "response": "..." or "reasoning": "..."
    # Support both single and double quotes
    response_patterns = [
        r'"response"\s*:\s*"([^"]*)"',
        r"'response'\s*:\s*'([^']*)'",
        r'"response"\s*:\s*([\d.]+)',
        r"'response'\s*:\s*([\d.]+)",
    ]
    reasoning_patterns = [
        r'"reasoning"\s*:\s*"([^"]*)"',
        r"'reasoning'\s*:\s*'([^']*)'",
    ]
    
    response_val = None
    reasoning_val = None
    
    for pattern in response_patterns:
        match = re.search(pattern, text)
        if match:
            response_val = match.group(1)
            break
    
    for pattern in reasoning_patterns:
        match = re.search(pattern, text)
        if match:
            reasoning_val = match.group(1)
            break
    
    if response_val or reasoning_val:
        obj = {}
        if response_val:
            obj["response"] = response_val
        if reasoning_val:
            obj["reasoning"] = reasoning_val
        if obj:
            results.append(obj)
    
    # Try to find any quoted string that might be a grade
    if not results:
        grade_patterns = [
            r'grade["\'\s:]+([\d.]+)',
            r'score["\'\s:]+([\d.]+)',
            r'response["\'\s:]+([\d.]+)',
            r'(?:^|\s)([\d.]+)\s*(?:points?|marks?|grade|score)',
        ]
        for pattern in grade_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                results.append({"response": match.group(1)})
                break
    
    # Last resort: look for standalone numbers that could be grades (0-10 range)
    # Be more selective - look for numbers that appear to be final answers
    if not results:
        # Look for patterns like "The grade is 7" or "Final score: 8.5"
        final_patterns = [
            r'(?:grade|score|mark|final|answer|result)(?:\s+(?:is|:|=)\s+)([\d.]+)',
            r'(?:^|\n)\s*([\d.]+)\s*$',  # Line with just a number
        ]
        for pattern in final_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                try:
                    val = float(match.group(1))
                    if 0 <= val <= 10:
                        results.append({"response": match.group(1)})
                        break
                except ValueError:
                    continue
    
    # Ultra last resort: any standalone number in valid range
    if not results:
        standalone_numbers = re.findall(r'\b([0-9](?:\.\d+)?)\b', text)
        for num in standalone_numbers:
            try:
                val = float(num)
                if 0 <= val <= 10:
                    results.append({"response": num})
                    break
            except ValueError:
                continue
    
    return results or None


def _validate_numeric_grade(grade: str) -> tuple[bool, float | None]:
    """Validate that a grade is a proper numeric value.
    
    Returns:
        (is_valid, numeric_value)
    
    Enhanced to handle edge cases like:
    - Whitespace around numbers
    - Numbers with trailing text (e.g., "7 points")
    - Scientific notation
    - Integer-only grades
    """
    if not grade or not isinstance(grade, str):
        return False, None
    
    grade = grade.strip()
    
    # Handle empty string after stripping
    if not grade:
        return False, None
    
    # Try to extract leading numeric portion if mixed with text
    # Pattern matches: optional +/-, digits, optional decimal and more digits
    numeric_match = re.match(r'^([+-]?\d+(?:\.\d+)?)', grade)
    if numeric_match:
        grade = numeric_match.group(1)
    
    # Try to parse as float
    try:
        value = float(grade)
        # Check if it's a reasonable grade (typically 0-10 for IMO)
        # Allow slight negative for partial credit edge cases, but cap at -1
        if -1 <= value <= 10:
            return True, value
        else:
            logger.debug(f"Grade {value} out of valid range [-1, 10]")
            return False, None
    except ValueError:
        logger.debug(f"Failed to parse grade as number: {grade}")
        return False, None


class TaskAgent:
    """Task agent that solves IMO grading problems with enhanced reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.max_retries = 3

    def _validate_inputs(self, inputs: dict) -> tuple[bool, str]:
        """Validate that all required inputs are present and non-empty.
        
        Returns:
            (is_valid, error_message)
        """
        required_fields = ["domain", "problem", "solution", "grading_guidelines", "student_answer"]
        
        for field in required_fields:
            if field not in inputs:
                return False, f"Missing required field: {field}"
            if not inputs[field] or not str(inputs[field]).strip():
                return False, f"Empty required field: {field}"
        
        return True, ""

    def _build_grading_prompt(self, inputs: dict) -> str:
        """Build a comprehensive prompt for the grading task."""
        domain = inputs.get("domain", "")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        # Truncate very long inputs to prevent token overflow
        max_len = 8000
        problem = problem[:max_len] + "..." if len(problem) > max_len else problem
        solution = solution[:max_len] + "..." if len(solution) > max_len else solution
        student_answer = student_answer[:max_len] + "..." if len(student_answer) > max_len else student_answer
        
        return f"""You are an expert grader for {domain} problems. Your task is to evaluate a student's answer against the correct solution and grading guidelines.

## Problem
{problem}

## Correct Solution
{solution}

## Grading Guidelines
{grading_guidelines}

## Student's Answer
{student_answer}

## Instructions
1. Carefully analyze the student's answer step by step
2. Compare it against the correct solution
3. Apply the grading guidelines strictly
4. Provide your reasoning before giving the final grade

## REQUIRED OUTPUT FORMAT
You MUST respond with a valid JSON object wrapped in <json> tags. Do not include any text outside the JSON tags.

<json>
{{
    "reasoning": "Your detailed step-by-step analysis and reasoning here. Explain how you evaluated the answer.",
    "response": "NUMERIC_GRADE_HERE"
}}
</json>

Important formatting rules:
- The "response" field MUST contain ONLY a numeric grade (e.g., "7", "3.5", "0")
- The "reasoning" field should contain your detailed analysis
- Do NOT add any text before or after the JSON tags
- Ensure the JSON is valid (no trailing commas, proper quotes)
- If the student answer is empty or nonsensical, assign the minimum grade (usually 0)"""

    def _try_extract_prediction(self, text: str) -> tuple[str, str | None]:
        """Try to extract prediction from response text with enhanced robustness.
        
        Returns:
            (prediction, reasoning)
        """
        if not text or not text.strip():
            return "None", None
            
        try:
            extracted = _extract_json_with_retry(text)
            if extracted:
                # Try to find the best JSON object with both response and reasoning
                best_match = None
                for obj in extracted:
                    if isinstance(obj, dict) and "response" in obj:
                        best_match = obj
                        break
                
                # If no object with "response" found, use the last one
                if best_match is None:
                    best_match = extracted[-1]
                
                if isinstance(best_match, dict):
                    prediction = best_match.get("response", "None")
                    reasoning = best_match.get("reasoning")
                    
                    # Handle numeric predictions
                    if prediction is None:
                        prediction = "None"
                    elif isinstance(prediction, (int, float)):
                        prediction = str(prediction)
                    elif not isinstance(prediction, str):
                        prediction = str(prediction)
                    
                    # Validate prediction is a proper numeric grade
                    is_valid, _ = _validate_numeric_grade(prediction)
                    if is_valid:
                        return prediction, reasoning
                    elif prediction and prediction.strip() and prediction != "None":
                        # Log that we found something but it's not a valid grade
                        self.log_fn(f"Found non-numeric prediction: {prediction}")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
        
        return "None", None

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with retry logic.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Validate inputs first
        is_valid, error_msg = self._validate_inputs(inputs)
        if not is_valid:
            self.log_fn(f"Input validation failed: {error_msg}")
            return f"Error: {error_msg}", []
        
        instruction = self._build_grading_prompt(inputs)
        msg_history = []
        prediction = "None"
        
        # Try with retries
        for attempt in range(self.max_retries):
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=msg_history if attempt > 0 else [],
                )
                
                # Extract prediction from the last assistant message
                text = ""
                for msg in reversed(msg_history):
                    if msg.get("role") == "assistant":
                        text = msg.get("text", "")
                        break
                
                pred, reasoning = self._try_extract_prediction(text)
                
                if pred != "None":
                    prediction = pred
                    if reasoning:
                        self.log_fn(f"Grading reasoning: {reasoning[:200]}...")
                    break
                
                # If extraction failed, add a follow-up message asking for proper format
                if attempt < self.max_retries - 1:
                    self.log_fn(f"Attempt {attempt + 1}: Failed to extract valid prediction, retrying with clearer instructions")
                    instruction = (
                        "Your previous response could not be parsed. "
                        "Please respond ONLY in the required JSON format:\n\n"
                        "<json>\n"
                        "{\n"
                        '    "reasoning": "Your detailed step-by-step analysis",\n'
                        '    "response": "The final grade/score (a number)"\n'
                        "}\n"
                        "</json>\n\n"
                        "Make sure the JSON is valid and the response field contains only the grade."
                    )
                    
            except Exception as e:
                self.log_fn(f"Attempt {attempt + 1} failed with exception: {e}")
                if attempt == self.max_retries - 1:
                    break
        
        if prediction == "None":
            self.log_fn("All attempts failed to extract a valid prediction")
        
        return str(prediction), msg_history
