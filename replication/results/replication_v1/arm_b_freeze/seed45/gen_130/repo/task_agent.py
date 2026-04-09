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

from agent.llm_client import get_response_from_llm, EVAL_MODEL

logger = logging.getLogger(__name__)


def _try_parse_json(text: str, context: str = "") -> dict | None:
    """Attempt to parse a JSON string with multiple fallback strategies.
    
    Args:
        text: The text to parse as JSON
        context: Context for logging (e.g., "<json> block #1")
        
    Returns:
        Parsed JSON dict if successful, None otherwise
    """
    if not text or not text.strip():
        return None
        
    text = text.strip()
    
    # Strategy 1: Direct parsing
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Strategy 2: Brace matching extraction
    try:
        json_str = _extract_json_with_brace_matching(text)
        if json_str:
            return json.loads(json_str)
    except (json.JSONDecodeError, ValueError):
        pass
    
    # Strategy 3: Clean and parse
    try:
        cleaned = _clean_json_string(text)
        if cleaned:
            return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        pass
    
    return None


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    Also handles markdown code blocks and bare JSON objects.
    Includes improved handling for nested braces, escaped characters, and 
    common LLM formatting issues like trailing commas and comments.
    
    Enhanced with better error logging and multiple fallback strategies.
    """
    if not text or not isinstance(text, str):
        logger.debug("_extract_jsons: received empty or non-string input")
        return None
    
    results = []
    search_from = 0
    
    # First try to find <json>...</json> blocks
    while True:
        start = text.find("<json>", search_from)
        if start == -1:
            break
        end = text.find("</json>", start)
        if end == -1:
            logger.debug(f"_extract_jsons: found opening <json> at {start} but no closing tag")
            break
        inner = text[start + 6:end].strip()
        search_from = end + 7
        
        result = _try_parse_json(inner, "<json> block")
        if result:
            results.append(result)
            logger.debug(f"_extract_jsons: successfully parsed JSON from <json> block")
        else:
            logger.debug(f"_extract_jsons: failed to parse JSON from <json> block")
    
    # If no <json> blocks found, try markdown code blocks
    if not results:
        logger.debug("_extract_jsons: no valid JSON found in <json> blocks, trying markdown code blocks")
        json_blocks = re.findall(r'```json\s*(.*?)```', text, re.DOTALL)
        logger.debug(f"_extract_jsons: found {len(json_blocks)} markdown json blocks")
        
        for block in json_blocks:
            result = _try_parse_json(block, "markdown block")
            if result:
                results.append(result)
                logger.debug(f"_extract_jsons: successfully parsed JSON from markdown block")
        
        # Try bare JSON objects as fallback
        if not results:
            logger.debug("_extract_jsons: no valid JSON in markdown blocks, trying bare JSON objects")
            potential_jsons = _find_json_objects(text)
            logger.debug(f"_extract_jsons: found {len(potential_jsons)} potential JSON objects")
            
            for pj in potential_jsons:
                result = _try_parse_json(pj, "bare JSON object")
                if result:
                    results.append(result)
                    logger.debug(f"_extract_jsons: successfully parsed bare JSON object")
    
    if results:
        logger.debug(f"_extract_jsons: returning {len(results)} JSON object(s)")
    else:
        logger.debug("_extract_jsons: no JSON objects found")
    
    return results or None


def _remove_line_comments(line: str) -> str:
    """Remove // style comments from a line, preserving strings.
    
    Args:
        line: A line of text that may contain comments
        
    Returns:
        The line with comments removed
    """
    if '//' not in line:
        return line
        
    in_string = False
    escape_next = False
    
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
            return line[:i].rstrip()
    
    return line


def _remove_multiline_comments(text: str) -> str:
    """Remove /* */ style multiline comments.
    
    Args:
        text: Text that may contain multiline comments
        
    Returns:
        Text with multiline comments removed
    """
    while '/*' in text and '*/' in text:
        start = text.find('/*')
        end = text.find('*/', start)
        if start >= 0 and end > start:
            text = text[:start] + text[end + 2:]
        else:
            break
    return text


def _replace_quotes_in_json(text: str) -> str:
    """Replace single quotes with double quotes in JSON, preserving strings.
    
    Args:
        text: JSON text that may contain single quotes
        
    Returns:
        Text with single quotes replaced by double quotes
    """
    result = []
    in_string = False
    escape_next = False
    
    for char in text:
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
    
    return ''.join(result)


def _escape_special_chars(text: str) -> str:
    """Escape unescaped newlines and tabs inside JSON strings.
    
    Args:
        text: JSON text that may contain unescaped special characters
        
    Returns:
        Text with special characters properly escaped
    """
    result = []
    in_string = False
    escape_next = False
    
    for char in text:
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
        elif in_string and char == '\n':
            result.append('\\n')
        elif in_string and char == '\t':
            result.append('\\t')
        else:
            result.append(char)
    
    return ''.join(result)


def _clean_json_string(text: str) -> str | None:
    """Clean common LLM JSON formatting issues.
    
    Handles:
    - Trailing commas before closing braces/brackets
    - Single quotes instead of double quotes
    - Comments (// and /* */)
    - Extra whitespace and newlines
    - Unescaped newlines in string values
    - Unicode escape sequences
    - BOM (Byte Order Mark) characters
    
    Args:
        text: Raw text that may contain JSON with formatting issues
        
    Returns:
        Cleaned JSON string if successful, None if input is empty
    """
    if not text:
        return None
    
    # First try to extract just the JSON object using brace matching
    json_str = _extract_json_with_brace_matching(text)
    if not json_str:
        json_str = text.strip()
    
    # Remove BOM if present
    json_str = json_str.lstrip('\ufeff')
    
    # Remove // style comments line by line
    lines = [_remove_line_comments(line) for line in json_str.split('\n')]
    json_str = '\n'.join(lines)
    
    # Remove /* */ comments (multiline)
    json_str = _remove_multiline_comments(json_str)
    
    # Remove trailing commas before } or ] - handle multiple commas
    json_str = re.sub(r',\s*,\s*}', '}', json_str)
    json_str = re.sub(r',\s*,\s*\]', ']', json_str)
    json_str = re.sub(r',\s*}', '}', json_str)
    json_str = re.sub(r',\s*\]', ']', json_str)
    
    # Replace single quotes with double quotes
    json_str = _replace_quotes_in_json(json_str)
    
    # Escape unescaped newlines and tabs in strings
    json_str = _escape_special_chars(json_str)
    
    return json_str.strip()


def _extract_json_with_brace_matching(text: str) -> str | None:
    """Extract a JSON object from text using brace counting.
    
    This handles nested braces correctly by counting open/close braces.
    Also handles square brackets for JSON arrays.
    
    Args:
        text: Text that may contain a JSON object or array
        
    Returns:
        The extracted JSON string if found, None otherwise
    """
    if not text:
        return None
    
    # Find the first { or [ that starts a JSON structure
    brace_start = text.find("{")
    bracket_start = text.find("[")
    
    if brace_start == -1 and bracket_start == -1:
        return None
    
    # Determine which comes first
    if brace_start == -1:
        start = bracket_start
        is_array = True
    elif bracket_start == -1:
        start = brace_start
        is_array = False
    else:
        start = min(brace_start, bracket_start)
        is_array = (start == bracket_start)
    
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
                if not is_array and brace_count == 0:
                    return text[start:i+1]
            elif char == "[":
                bracket_count += 1
            elif char == "]":
                bracket_count -= 1
                if is_array and bracket_count == 0 and brace_count == 0:
                    return text[start:i+1]
    
    return None


def _is_valid_json_object(json_str: str) -> bool:
    """Check if a string represents a valid JSON object or array.
    
    Args:
        json_str: The string to validate
        
    Returns:
        True if the string is a valid JSON object/array, False otherwise
    """
    if not json_str or len(json_str) < 2:
        return False
    
    first_char = json_str[0]
    
    # For objects: must contain at least one quoted string
    if first_char == "{":
        return '"' in json_str
    
    # For arrays: must contain at least one element
    if first_char == "[":
        return len(json_str) > 2
    
    return False


def _find_json_objects(text: str) -> list[str]:
    """Find all potential JSON objects and arrays in text using brace matching.
    
    Args:
        text: Text to search for JSON objects
        
    Returns:
        List of extracted JSON strings
    """
    results = []
    i = 0
    while i < len(text):
        if text[i] in "{[":
            json_str = _extract_json_with_brace_matching(text[i:])
            if json_str and _is_valid_json_object(json_str):
                results.append(json_str)
                i += len(json_str)
            else:
                i += 1
        else:
            i += 1
    return results


def _extract_grade_from_text(text: str) -> str | None:
    """Extract a grade/assessment from plain text when JSON parsing fails.
    
    Looks for common grading patterns and keywords in the text.
    Returns the most likely grade or None if no grade is found.
    
    Enhanced with better pattern matching and context awareness.
    """
    if not text:
        return None
    
    text_lower = text.lower()
    
    # Define grade patterns with their associated keywords and weights
    grade_patterns = [
        ("Correct", [
            ("correct", 2), ("right", 2), ("accurate", 2), ("valid", 1), 
            ("proper", 1), ("appropriate", 1), ("satisfactory", 1),
            ("well done", 3), ("excellent", 2), ("perfect", 3), ("good", 1)
        ]),
        ("Incorrect", [
            ("incorrect", 3), ("wrong", 2), ("error", 1), ("mistake", 1), 
            ("invalid", 2), ("unsatisfactory", 2), ("fail", 2),
            ("not correct", 3), ("not right", 3), ("does not match", 2),
            ("doesn't match", 2), ("no match", 2)
        ]),
        ("Partially Correct", [
            ("partially correct", 3), ("partial credit", 3), ("partially right", 3), 
            ("some correct", 2), ("incomplete", 2), ("partial", 2),
            ("mostly correct", 2), ("mostly right", 2), ("half correct", 2),
            ("partly correct", 3), ("some errors", 1), ("minor errors", 1)
        ]),
    ]
    
    # Count weighted occurrences of each grade's keywords
    grade_scores = {}
    for grade, keywords in grade_patterns:
        score = 0
        for keyword, weight in keywords:
            # Count occurrences and apply weight
            count = text_lower.count(keyword)
            score += count * weight
        grade_scores[grade] = score
    
    # Look for explicit grade statements with higher weights
    explicit_patterns = [
        (r'\bgrade\s*[:=]\s*["\']?([^"\'\n]+)["\']?', 5),
        (r'\b(?:the\s+)?(?:final\s+)?(?:grade|score|assessment|evaluation|verdict)\s*(?:is|[:=])\s*["\']?([^"\'\n.]+)["\']?', 5),
        (r'\b(?:i\s+)?(?:would\s+)?(?:grade|score|rate|assess)\s*(?:this\s+)?(?:as|at)\s*["\']?([^"\'\n.]+)["\']?', 5),
        (r'\b(?:answer|solution)\s+(?:is\s+)?(["\']?(?:correct|incorrect|partially correct|partial|wrong)["\']?)', 6),
        (r'\b(?:this\s+is|the\s+answer\s+is)\s+(["\']?(?:correct|incorrect|partially correct|partial|wrong)["\']?)', 6),
    ]
    
    for pattern, weight in explicit_patterns:
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        for match in matches:
            if isinstance(match, str):
                match_clean = match.strip().strip('"\'').lower()
                for grade, keywords in grade_patterns:
                    # Check if any keyword is in the match
                    for keyword, _ in keywords:
                        if keyword in match_clean:
                            grade_scores[grade] += weight
                            break
    
    # Look for negation patterns (e.g., "not correct" should favor Incorrect)
    negation_patterns = [
        (r'\bnot\s+(correct|right|accurate|valid)\b', "Incorrect", 4),
        (r'\bnot\s+(incorrect|wrong|invalid)\b', "Correct", 4),
    ]
    
    for pattern, grade, weight in negation_patterns:
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        grade_scores[grade] += len(matches) * weight
    
    # Look for numeric scores (0-100 or 0-10)
    numeric_patterns = [
        r'\b(?:score|grade)\s*[:=]?\s*(\d+(?:\.\d+)?)\s*(?:/\s*(\d+))?\b',
        r'\b(\d+(?:\.\d+)?)\s*(?:out\s+of|/\s*)(\d+)\b',
        r'\b(\d+(?:\.\d+)?)\s*points?\b',
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
                            grade_scores["Correct"] += 4
                        elif percentage >= 50:
                            grade_scores["Partially Correct"] += 4
                        else:
                            grade_scores["Incorrect"] += 4
                except (ValueError, TypeError):
                    pass
            elif isinstance(match, str):
                try:
                    score = float(match)
                    # Assume 0-10 scale if score is small
                    if score <= 10:
                        if score >= 8:
                            grade_scores["Correct"] += 3
                        elif score >= 5:
                            grade_scores["Partially Correct"] += 3
                        else:
                            grade_scores["Incorrect"] += 3
                    else:
                        # Assume 0-100 scale
                        if score >= 80:
                            grade_scores["Correct"] += 3
                        elif score >= 50:
                            grade_scores["Partially Correct"] += 3
                        else:
                            grade_scores["Incorrect"] += 3
                except ValueError:
                    pass
    
    # Return the grade with the highest score, if any
    if grade_scores:
        best_grade = max(grade_scores, key=grade_scores.get)
        if grade_scores[best_grade] > 0:
            logger.debug(f"_extract_grade_from_text: selected '{best_grade}' with score {grade_scores[best_grade]}")
            return best_grade
    
    logger.debug("_extract_grade_from_text: no grade found")
    return None


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning.
    
    Enhanced with better error handling, logging, and multiple fallback strategies
    for extracting grades from LLM responses.
    """

    # Response keys to try when extracting the grade from JSON
    RESPONSE_KEYS = ["response", "grade", "answer", "result", "assessment", "evaluation", "verdict", "conclusion"]
    
    # Reasoning keys to try when extracting the reasoning from JSON
    REASONING_KEYS = ["reasoning", "analysis", "thought", "explanation", "rationale", "thinking"]
    
    # Default feedback message when JSON is not found
    JSON_FEEDBACK = (
        "Your previous response did not contain valid JSON in the required format. "
        "Please respond with a JSON object wrapped in <json>...</json> tags. "
        "The JSON must have 'reasoning' and 'response' fields. "
        "Example format:\n"
        "<json>\n"
        '{\n  "reasoning": "The student correctly identified...",\n  "response": "Correct"\n}'
        "\n</json>"
    )

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.max_retries = 3
        self._extraction_stats = {
            "json_extractions": 0,
            "fallback_extractions": 0,
            "failed_extractions": 0,
            "total_attempts": 0
        }

    def _build_instruction(self, inputs: dict) -> str:
        """Build the grading instruction from input fields.
        
        Args:
            inputs: Dictionary containing problem data
            
        Returns:
            Formatted instruction string for the LLM
        """
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        logger.debug(f"TaskAgent._build_instruction: problem length={len(problem)}, "
                    f"solution length={len(solution)}, "
                    f"student_answer length={len(student_answer)}")
        
        return f"""You are an expert {domain} grader evaluating student solutions.

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
- Do not include any text outside the <json>...</json> tags.

Think carefully and provide a fair assessment based on the official solution and grading guidelines."""

    def _extract_from_history(self, msg_history: list[dict]) -> tuple[dict | None, str]:
        """Extract JSON from message history.
        
        Args:
            msg_history: List of message dictionaries
            
        Returns:
            Tuple of (extracted JSON dict or None, source description)
        """
        if not msg_history:
            return None, "empty history"
        
        # Try the last assistant message first
        last_msg = msg_history[-1].get("text", "")
        if last_msg:
            extracted = _extract_jsons(last_msg)
            if extracted:
                return extracted[-1], "last message"
        
        # Search all messages if not found in last
        logger.debug("TaskAgent._extract_from_history: no JSON in last message, searching all messages")
        for i, msg in enumerate(reversed(msg_history)):
            text = msg.get("text", "")
            if text:
                extracted = _extract_jsons(text)
                if extracted:
                    logger.debug(f"TaskAgent._extract_from_history: found JSON in message {len(msg_history) - i - 1}")
                    return extracted[-1], f"message {len(msg_history) - i - 1}"
        
        return None, "not found"

    def _extract_response_and_reasoning(self, result: dict) -> tuple[str | None, str | None]:
        """Extract response and reasoning from a JSON result.
        
        Args:
            result: Dictionary containing the parsed JSON
            
        Returns:
            Tuple of (response/grade, reasoning) - either may be None
        """
        # Extract response/grade
        response = None
        for key in self.RESPONSE_KEYS:
            if key in result:
                response = result[key]
                logger.debug(f"TaskAgent._extract_response_and_reasoning: found response in key '{key}': {response}")
                break
        
        if not response:
            logger.warning(f"TaskAgent._extract_response_and_reasoning: no response key found in result: {list(result.keys())}")
        
        # Extract reasoning
        reasoning = None
        for key in self.REASONING_KEYS:
            if key in result:
                reasoning = result[key]
                break
        
        return response, reasoning

    def _handle_extraction_failure(self, last_msg: str, attempt: int) -> tuple[str | None, str]:
        """Handle case when JSON extraction fails.
        
        Args:
            last_msg: The last message text
            attempt: Current attempt number
            
        Returns:
            Tuple of (fallback prediction or None, extraction method)
        """
        fallback = _extract_grade_from_text(last_msg)
        if fallback and fallback != "None":
            self._extraction_stats["fallback_extractions"] += 1
            self.log_fn(f"Using fallback grade extraction: {fallback}")
            return fallback, "fallback"
        return None, "none"

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with chain-of-thought reasoning.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Validate inputs
        if not isinstance(inputs, dict):
            logger.error(f"TaskAgent.forward: expected dict, got {type(inputs)}")
            return "Error: Invalid inputs", []
        
        instruction = self._build_instruction(inputs)
        msg_history = []
        prediction = "None"
        extraction_method = None
        
        for attempt in range(self.max_retries + 1):
            self._extraction_stats["total_attempts"] += 1
            
            # Call LLM
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=msg_history,
                )
            except Exception as e:
                logger.error(f"TaskAgent.forward: LLM call failed on attempt {attempt + 1}: {e}")
                if attempt < self.max_retries:
                    continue
                return "Error: LLM call failed", msg_history

            # Extract prediction from JSON
            try:
                result, source = self._extract_from_history(msg_history)
                
                if result and isinstance(result, dict):
                    self._extraction_stats["json_extractions"] += 1
                    extraction_method = "json"
                    
                    response_val, reasoning = self._extract_response_and_reasoning(result)
                    
                    if response_val:
                        prediction = response_val
                        if reasoning:
                            self.log_fn(f"Reasoning: {reasoning[:200]}...")
                        break
                    
                    # No response key found - will retry
                    logger.warning(f"TaskAgent.forward: no response key found from {source}")
                
                else:
                    # Try fallback extraction
                    last_msg = msg_history[-1].get("text", "") if msg_history else ""
                    fallback, method = self._handle_extraction_failure(last_msg, attempt)
                    if fallback:
                        prediction = fallback
                        extraction_method = method
                        break
                
                # Retry with feedback if we have attempts left
                if attempt < self.max_retries:
                    self.log_fn(f"No JSON found in attempt {attempt + 1}, retrying with feedback...")
                    msg_history.append({"role": "user", "text": self.JSON_FEEDBACK})
                    instruction = self.JSON_FEEDBACK
                else:
                    self._extraction_stats["failed_extractions"] += 1
                    self.log_fn(f"No JSON found after {self.max_retries + 1} attempts")
                    
            except Exception as e:
                self.log_fn(f"Error extracting prediction (attempt {attempt + 1}): {e}")
                logger.exception("TaskAgent.forward: exception during extraction")
                if attempt < self.max_retries:
                    feedback = (
                        f"Error parsing your response: {e}. "
                        "Please ensure your response contains valid JSON wrapped in <json>...</json> tags. "
                        "The JSON must have 'reasoning' and 'response' fields."
                    )
                    msg_history.append({"role": "user", "text": feedback})
                    instruction = feedback

        # Log extraction stats periodically
        if self._extraction_stats["total_attempts"] % 100 == 0:
            logger.info(f"TaskAgent extraction stats: {self._extraction_stats}")
        
        logger.debug(f"TaskAgent.forward: returning prediction='{prediction}', method={extraction_method}")
        return str(prediction), msg_history
