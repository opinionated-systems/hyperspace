"""
Task agent: solves a given task with chain-of-thought reasoning.

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
from typing import Any

from agent.llm_client import get_response_from_llm, EVAL_MODEL

logger = logging.getLogger(__name__)

# Constants for JSON extraction
MAX_JSON_SIZE = 100000  # Maximum size for JSON string processing
MAX_RETRIES = 3  # Maximum retry attempts for LLM calls
DEFAULT_RETRY_KEYS = ["response", "grade", "answer", "result", "assessment", "evaluation", "verdict", "conclusion"]
REASONING_KEYS = ["reasoning", "analysis", "thought", "explanation", "rationale", "thinking"]


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    Also handles markdown code blocks and bare JSON objects.
    Includes improved handling for nested braces, escaped characters, and 
    common LLM formatting issues like trailing commas and comments.
    
    Args:
        text: The text to extract JSON from.
        
    Returns:
        A list of parsed JSON objects, or None if no valid JSON found.
    """
    # Input validation
    if not text or not isinstance(text, str):
        logger.debug("JSON extraction: input is empty or not a string")
        return None
    
    # Size limit to prevent memory issues with huge inputs
    original_len = len(text)
    if original_len > MAX_JSON_SIZE:
        logger.warning(f"Text exceeds maximum size ({MAX_JSON_SIZE}), truncating for JSON extraction (original size: {original_len})")
        text = text[:MAX_JSON_SIZE]
        
    results = []
    search_from = 0
    parse_errors = []  # Track errors for debugging
    extraction_attempts = 0  # Track number of extraction attempts
    
    # First try to find <json>...</json> blocks
    while True:
        start = text.find("<json>", search_from)
        if start == -1:
            break
        end = text.find("</json>", start)
        if end == -1:
            # Log unclosed tag for debugging
            logger.debug(f"Unclosed <json> tag found at position {start}")
            break
        inner = text[start + 6:end].strip()
        search_from = end + 7
        
        # Skip empty content
        if not inner:
            logger.debug(f"Empty <json> block found at position {start}, skipping")
            continue
        
        extraction_attempts += 1
        logger.debug(f"Attempting to parse <json> block #{extraction_attempts} (length: {len(inner)} chars)")
            
        try:
            results.append(json.loads(inner))
            logger.debug(f"Successfully parsed <json> block #{extraction_attempts}")
        except json.JSONDecodeError as e:
            parse_errors.append(f"Direct JSON parse failed for block #{extraction_attempts}: {e}")
            # Try to extract JSON from within the content using brace matching
            try:
                json_str = _extract_json_with_brace_matching(inner)
                if json_str:
                    results.append(json.loads(json_str))
                    logger.debug(f"Successfully parsed <json> block #{extraction_attempts} using brace matching")
                else:
                    parse_errors.append(f"Block #{extraction_attempts}: Brace matching returned None")
            except (json.JSONDecodeError, ValueError) as e2:
                parse_errors.append(f"Block #{extraction_attempts}: Brace matching failed: {e2}")
                # Try cleaning common LLM formatting issues
                try:
                    cleaned = _clean_json_string(inner)
                    if cleaned:
                        results.append(json.loads(cleaned))
                        logger.debug(f"Successfully parsed <json> block #{extraction_attempts} after cleaning")
                    else:
                        parse_errors.append(f"Block #{extraction_attempts}: Cleaning returned None")
                except (json.JSONDecodeError, ValueError) as e3:
                    parse_errors.append(f"Block #{extraction_attempts}: Cleaning failed: {e3}")
                    continue
    
    # If no <json> blocks found, try markdown code blocks
    if not results:
        logger.debug("No valid JSON found in <json> blocks, trying markdown code blocks")
        # Try ```json ... ``` blocks
        json_blocks = re.findall(r'```json\s*(.*?)```', text, re.DOTALL)
        logger.debug(f"Found {len(json_blocks)} markdown JSON blocks")
        for i, block in enumerate(json_blocks):
            if not block.strip():
                logger.debug(f"Markdown block #{i+1} is empty, skipping")
                continue
            try:
                results.append(json.loads(block.strip()))
                logger.debug(f"Successfully parsed markdown block #{i+1}")
            except json.JSONDecodeError as e:
                parse_errors.append(f"Markdown block #{i+1} parse failed: {e}")
                # Try brace matching extraction
                try:
                    json_str = _extract_json_with_brace_matching(block)
                    if json_str:
                        results.append(json.loads(json_str))
                        logger.debug(f"Successfully parsed markdown block #{i+1} using brace matching")
                except (json.JSONDecodeError, ValueError) as e2:
                    parse_errors.append(f"Markdown block #{i+1} brace matching failed: {e2}")
                    # Try cleaning common LLM formatting issues
                    try:
                        cleaned = _clean_json_string(block)
                        if cleaned:
                            results.append(json.loads(cleaned))
                            logger.debug(f"Successfully parsed markdown block #{i+1} after cleaning")
                    except (json.JSONDecodeError, ValueError) as e3:
                        parse_errors.append(f"Markdown block #{i+1} cleaning failed: {e3}")
                        continue
        
        # Try bare JSON objects as fallback with improved regex
        if not results:
            logger.debug("No valid JSON in markdown blocks, trying bare JSON objects")
            # Find JSON-like structures with nested brace support
            potential_jsons = _find_json_objects(text)
            logger.debug(f"Found {len(potential_jsons)} potential JSON objects")
            for i, pj in enumerate(potential_jsons):
                if not pj or len(pj) < 2:  # Skip empty or too short
                    continue
                try:
                    results.append(json.loads(pj))
                    logger.debug(f"Successfully parsed bare JSON object #{i+1}")
                except json.JSONDecodeError as e:
                    parse_errors.append(f"Bare JSON #{i+1} parse failed: {e}")
                    # Try cleaning common LLM formatting issues
                    try:
                        cleaned = _clean_json_string(pj)
                        if cleaned:
                            results.append(json.loads(cleaned))
                            logger.debug(f"Successfully parsed bare JSON object #{i+1} after cleaning")
                    except (json.JSONDecodeError, ValueError) as e2:
                        parse_errors.append(f"Bare JSON #{i+1} cleaning failed: {e2}")
                        continue
    
    # Log parse errors if debugging is enabled and no results found
    if not results:
        if parse_errors:
            logger.warning(f"JSON extraction failed after {extraction_attempts} attempts with {len(parse_errors)} errors")
            if logger.isEnabledFor(logging.DEBUG):
                for err in parse_errors[:10]:  # Log first 10 errors
                    logger.debug(f"  - {err}")
        else:
            logger.warning("No JSON found in text and no parse errors recorded")
    else:
        logger.info(f"Successfully extracted {len(results)} JSON object(s) from text")
    
    return results or None


def _clean_json_string(text: str) -> str | None:
    """Clean common LLM JSON formatting issues.
    
    Handles:
    - Trailing commas before closing braces/brackets
    - Single quotes instead of double quotes
    - Comments (// and /* */)
    - Extra whitespace and newlines
    - Control characters and invalid Unicode
    - BOM (Byte Order Mark) characters
    """
    if not text:
        return None
    
    # First try to extract just the JSON object using brace matching
    json_str = _extract_json_with_brace_matching(text)
    if not json_str:
        json_str = text.strip()
    
    # Remove BOM if present
    json_str = json_str.lstrip('\ufeff')
    
    # Remove control characters except tab, newline, carriage return
    cleaned_chars = []
    for char in json_str:
        code = ord(char)
        # Allow printable ASCII, extended ASCII, and common whitespace
        if code == 9 or code == 10 or code == 13 or (32 <= code <= 126) or code >= 160:
            cleaned_chars.append(char)
        # Skip control characters silently
    json_str = ''.join(cleaned_chars)
    
    # Remove comments (// style) with improved quote handling
    lines = []
    for line in json_str.split('\n'):
        # Remove // comments but preserve URLs and quoted strings
        if '//' in line:
            in_string = False
            escape_next = False
            comment_start = -1
            for i, char in enumerate(line):
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
                elif not in_string and char == '/' and i + 1 < len(line) and line[i + 1] == '/':
                    comment_start = i
                    break
            if comment_start >= 0:
                line = line[:comment_start]
        lines.append(line)
    json_str = '\n'.join(lines)
    
    # Remove /* */ comments (handle nested blocks carefully)
    max_iterations = 10  # Prevent infinite loops
    iterations = 0
    while '/*' in json_str and '*/' in json_str and iterations < max_iterations:
        start = json_str.find('/*')
        end = json_str.find('*/', start)
        if start >= 0 and end > start:
            json_str = json_str[:start] + json_str[end + 2:]
        else:
            break
        iterations += 1
    
    # Remove trailing commas before } or ] (handle multiple commas)
    json_str = re.sub(r',+\s*}', '}', json_str)
    json_str = re.sub(r',+\s*\]', ']', json_str)
    
    # Replace single quotes with double quotes (carefully)
    result = []
    in_string = False
    escape_next = False
    for char in json_str:
        if escape_next:
            result.append(char)
            escape_next = False
            continue
        if char == '\\':
            result.append(char)
            escape_next = True
            continue
        if char == '"' and not in_string:
            in_string = True
            result.append(char)
        elif char == '"' and in_string:
            in_string = False
            result.append(char)
        elif char == "'" and not in_string:
            result.append('"')
        else:
            result.append(char)
    json_str = ''.join(result)
    
    # Normalize whitespace: collapse multiple spaces outside strings
    # This is a simplified approach - just strip leading/trailing
    json_str = json_str.strip()
    
    return json_str if json_str else None


def _extract_json_with_brace_matching(text: str) -> str | None:
    """Extract a JSON object from text using brace counting.
    
    This handles nested braces correctly by counting open/close braces.
    Also handles square brackets for JSON arrays.
    
    Args:
        text: The text to extract JSON from.
        
    Returns:
        The extracted JSON string, or None if no valid JSON found.
    """
    if not text or not isinstance(text, str):
        return None
    
    # Look for both object and array starts
    obj_start = text.find("{")
    arr_start = text.find("[")
    
    # Determine which comes first
    if obj_start == -1 and arr_start == -1:
        return None
    elif obj_start == -1:
        start = arr_start
        is_array = True
    elif arr_start == -1:
        start = obj_start
        is_array = False
    else:
        start = min(obj_start, arr_start)
        is_array = (start == arr_start)
    
    brace_count = 0
    bracket_count = 0
    in_string = False
    escape_next = False
    
    for i, char in enumerate(text[start:], start):
        if escape_next:
            escape_next = False
            continue
        if char == "\\":
            escape_next = True
            continue
        if char == '"' and not in_string:
            in_string = True
        elif char == '"' and in_string:
            in_string = False
        elif not in_string:
            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                # For objects, check if we're done
                if not is_array and brace_count == 0:
                    return text[start:i+1]
            elif char == "[":
                bracket_count += 1
            elif char == "]":
                bracket_count -= 1
                # For arrays, check if we're done
                if is_array and bracket_count == 0 and brace_count == 0:
                    return text[start:i+1]
    
    # If we get here, braces weren't balanced - return None
    return None


def _find_json_objects(text: str) -> list[str]:
    """Find all potential JSON objects and arrays in text using brace matching.
    
    Args:
        text: The text to search for JSON structures.
        
    Returns:
        A list of JSON strings found in the text.
    """
    if not text or not isinstance(text, str):
        return []
    
    results = []
    i = 0
    while i < len(text):
        if text[i] == "{" or text[i] == "[":
            json_str = _extract_json_with_brace_matching(text[i:])
            if json_str:
                # For objects: must contain at least one quoted string (likely a key)
                # For arrays: accept if it looks like valid JSON array
                is_object = json_str.startswith("{")
                is_array = json_str.startswith("[")
                
                if is_object and '"' in json_str:
                    results.append(json_str)
                    i += len(json_str)
                elif is_array:
                    # Arrays are valid JSON - accept them
                    results.append(json_str)
                    i += len(json_str)
                else:
                    i += 1
            else:
                i += 1
        else:
            i += 1
    return results


def _extract_grade_from_text(text: str) -> str | None:
    """Extract a grade/assessment from plain text when JSON parsing fails.
    
    Looks for common grading patterns and keywords in the text.
    Returns the most likely grade or None if no grade is found.
    """
    if not text:
        return None
    
    text_lower = text.lower()
    
    # Define grade patterns with their associated keywords
    grade_patterns = [
        ("Correct", ["correct", "right", "accurate", "valid", "proper", "appropriate", "satisfactory"]),
        ("Incorrect", ["incorrect", "wrong", "error", "mistake", "invalid", "unsatisfactory", "fail"]),
        ("Partially Correct", ["partially correct", "partial credit", "partially right", "some correct", "incomplete"]),
    ]
    
    # Count occurrences of each grade's keywords
    grade_scores = {}
    for grade, keywords in grade_patterns:
        score = 0
        for keyword in keywords:
            score += text_lower.count(keyword)
        grade_scores[grade] = score
    
    # Also look for explicit grade statements
    explicit_patterns = [
        (r'grade\s*[:=]\s*["\']?([^"\'\n]+)["\']?', 1),
        (r'(?:the\s+)?(?:final\s+)?(?:grade|score|assessment|evaluation|verdict)\s*(?:is|[:=])\s*["\']?([^"\'\n.]+)["\']?', 1),
        (r'(?:i\s+)?(?:would\s+)?(?:grade|score|rate|assess)\s*(?:this\s+)?(?:as|at)\s*["\']?([^"\'\n.]+)["\']?', 1),
        (r'(?:answer|solution)\s+(?:is\s+)?(["\']?(?:correct|incorrect|partially correct|partial|wrong)["\']?)', 1),
    ]
    
    for pattern, group in explicit_patterns:
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        for match in matches:
            if isinstance(match, str):
                match_clean = match.strip().strip('"\'').lower()
                for grade, keywords in grade_patterns:
                    if any(kw in match_clean for kw in keywords):
                        grade_scores[grade] += 5  # Higher weight for explicit statements
    
    # Look for numeric scores (0-100 or 0-10)
    numeric_patterns = [
        r'(?:score|grade)\s*[:=]?\s*(\d+(?:\.\d+)?)\s*(?:/\s*(\d+))?',
        r'(\d+(?:\.\d+)?)\s*(?:out\s+of|/\s*)(\d+)',
    ]
    for pattern in numeric_patterns:
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        for match in matches:
            if isinstance(match, tuple):
                try:
                    score = float(match[0])
                    total = float(match[1]) if match[1] and match[1].strip() else 100.0
                    if total > 0:
                        percentage = (score / total) * 100
                        if percentage >= 80:
                            grade_scores["Correct"] += 3
                        elif percentage >= 50:
                            grade_scores["Partially Correct"] += 3
                        else:
                            grade_scores["Incorrect"] += 3
                except (ValueError, TypeError):
                    pass
            elif isinstance(match, str):
                try:
                    score = float(match)
                    if score >= 8:
                        grade_scores["Correct"] += 2
                    elif score >= 5:
                        grade_scores["Partially Correct"] += 2
                    else:
                        grade_scores["Incorrect"] += 2
                except ValueError:
                    pass
    
    # Return the grade with the highest score, if any
    if grade_scores:
        best_grade = max(grade_scores, key=grade_scores.get)
        if grade_scores[best_grade] > 0:
            return best_grade
    
    return None


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning.
    
    This agent uses an LLM to evaluate student answers against official solutions
    and grading guidelines. It includes robust JSON extraction and fallback
    mechanisms for handling various LLM response formats.
    
    Attributes:
        model: The LLM model identifier to use for evaluation.
        log_fn: Logging function for agent activity.
        max_retries: Maximum number of retry attempts for failed JSON extraction.
    """

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        """Initialize the TaskAgent.
        
        Args:
            model: The LLM model identifier to use.
            log_file: Optional path to a log file (currently unused, for interface compatibility).
        """
        self.model = model
        self.log_fn = logger.info
        self.max_retries = MAX_RETRIES

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with chain-of-thought reasoning.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Extract fields for better prompting
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = f"""You are an expert {domain} grader evaluating student solutions.

Your task is to grade a student's answer to a problem by comparing it against the official solution and following the grading guidelines.

## Problem Statement:
{problem}

## Official Solution:
{solution}

## Grading Guidelines:
{grading_guidelines}

## Student's Answer:
{student_answer}

## Instructions:
1. First, analyze the student's answer step by step. Identify what they did correctly and incorrectly.
2. Compare their approach to the official solution.
3. Consider the grading guidelines carefully.
4. Provide your reasoning before giving the final grade.
5. Respond in JSON format with the following schema:

<json>
{{
    "reasoning": "Your detailed analysis and reasoning about the student's answer...",
    "response": "The final grade/assessment (e.g., 'Correct', 'Partially Correct', 'Incorrect', or a numeric score)"
}}
</json>

Important: 
- The JSON must be valid and properly formatted.
- Wrap the JSON in <json>...</json> tags.
- The 'response' field should contain only the final grade/assessment.
- The 'reasoning' field should contain your detailed analysis.

Think carefully and provide a fair assessment based on the official solution and grading guidelines."""

        msg_history = []
        prediction = "None"
        reasoning = ""
        
        for attempt in range(self.max_retries + 1):
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=msg_history,
            )

            # Extract prediction from JSON
            extracted = None
            try:
                # Try the last assistant message
                last_msg = msg_history[-1]["text"] if msg_history else ""
                extracted = _extract_jsons(last_msg)
                
                # If not found, try searching all messages
                if not extracted:
                    for msg in reversed(msg_history):
                        text = msg.get("text", "")
                        extracted = _extract_jsons(text)
                        if extracted:
                            break
                
                if extracted:
                    result = extracted[-1]
                    # Try multiple possible keys for the response
                    for key in DEFAULT_RETRY_KEYS:
                        if key in result:
                            prediction = result[key]
                            break
                    
                    # Extract reasoning if available
                    for key in REASONING_KEYS:
                        if key in result:
                            reasoning = result[key]
                            break
                    
                    # Log reasoning if available
                    if reasoning:
                        self.log_fn(f"Reasoning: {reasoning[:200]}...")
                    
                    # Success - break out of retry loop
                    break
                else:
                    # Try to extract any text that might be a grade/assessment
                    # This is a fallback for when JSON parsing fails completely
                    fallback_prediction = _extract_grade_from_text(last_msg)
                    if fallback_prediction and fallback_prediction != "None":
                        self.log_fn(f"Using fallback grade extraction: {fallback_prediction}")
                        prediction = fallback_prediction
                        break
                    
                if attempt < self.max_retries:
                    # No JSON found - add feedback and retry
                    self.log_fn(f"No JSON found in attempt {attempt + 1}, retrying with feedback...")
                    feedback = (
                        "Your previous response did not contain valid JSON in the required format. "
                        "Please respond with a JSON object wrapped in <json>...</json> tags. "
                        "The JSON must have 'reasoning' and 'response' fields. "
                        "Example format:\n"
                        "<json>\n"
                        '{\n  "reasoning": "The student correctly identified...",\n  "response": "Correct"\n}'
                        "\n</json>"
                    )
                    msg_history.append({"role": "user", "text": feedback})
                    instruction = feedback  # Update instruction for next iteration
                else:
                    self.log_fn(f"No JSON found after {self.max_retries + 1} attempts")
                    
            except Exception as e:
                self.log_fn(f"Error extracting prediction (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries:
                    feedback = (
                        f"Error parsing your response: {e}. "
                        "Please ensure your response contains valid JSON wrapped in <json>...</json> tags. "
                        "The JSON must have 'reasoning' and 'response' fields."
                    )
                    msg_history.append({"role": "user", "text": feedback})
                    instruction = feedback

        return str(prediction), msg_history
