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


def _try_parse_json(text: str, parse_errors: list[str], source: str) -> dict | None:
    """Attempt to parse JSON from text with multiple fallback strategies.
    
    Args:
        text: The text to parse.
        parse_errors: List to collect parsing error messages.
        source: Description of the source for error messages.
        
    Returns:
        Parsed JSON dict or None if all strategies fail.
    """
    text = text.strip()
    if not text:
        return None
        
    # Strategy 1: Direct JSON parse
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        parse_errors.append(f"{source} direct parse failed: {e}")
    
    # Strategy 2: Brace matching extraction
    try:
        json_str = _extract_json_with_brace_matching(text)
        if json_str:
            return json.loads(json_str)
        else:
            parse_errors.append(f"{source} brace matching returned no valid JSON")
    except (json.JSONDecodeError, ValueError) as e:
        parse_errors.append(f"{source} brace matching failed: {e}")
    
    # Strategy 3: Clean and parse
    try:
        cleaned = _clean_json_string(text)
        if cleaned:
            return json.loads(cleaned)
        else:
            parse_errors.append(f"{source} JSON cleaning returned empty result")
    except (json.JSONDecodeError, ValueError) as e:
        parse_errors.append(f"{source} cleaned parse failed: {e}")
    
    return None


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
        return None
    
    # Size limit to prevent memory issues with huge inputs
    if len(text) > MAX_JSON_SIZE:
        logger.warning(f"Text exceeds maximum size ({MAX_JSON_SIZE}), truncating for JSON extraction")
        text = text[:MAX_JSON_SIZE]
        
    results = []
    parse_errors: list[str] = []  # Track parsing errors for debugging
    
    # First try to find <json>...</json> blocks
    search_from = 0
    while True:
        start = text.find("<json>", search_from)
        if start == -1:
            break
        end = text.find("</json>", start)
        if end == -1:
            logger.debug(f"Found <json> tag at position {start} but no closing </json> tag")
            break
        inner = text[start + 6:end].strip()
        search_from = end + 7
        
        result = _try_parse_json(inner, parse_errors, "<json> block")
        if result:
            results.append(result)
    
    # If no <json> blocks found, try markdown code blocks
    if not results:
        # Try ```json ... ``` blocks
        json_blocks = re.findall(r'```json\s*(.*?)```', text, re.DOTALL)
        for block in json_blocks:
            result = _try_parse_json(block, parse_errors, "markdown json block")
            if result:
                results.append(result)
        
        # Try ``` ... ``` blocks without json specifier
        if not results:
            code_blocks = re.findall(r'```\s*(.*?)```', text, re.DOTALL)
            for block in code_blocks:
                block = block.strip()
                if not block or block.startswith('json'):
                    continue
                # Quick check if it looks like JSON
                if block.startswith('{') or block.startswith('['):
                    result = _try_parse_json(block, parse_errors, "markdown code block")
                    if result:
                        results.append(result)
        
        # Try bare JSON objects as fallback with improved regex
        if not results:
            # Find JSON-like structures with nested brace support
            potential_jsons = _find_json_objects(text)
            for pj in potential_jsons:
                result = _try_parse_json(pj, parse_errors, "bare JSON")
                if result:
                    results.append(result)
    
    # Log parsing errors if we found no results but had attempts
    if not results and parse_errors:
        logger.debug(f"JSON extraction failed. Errors: {parse_errors[:5]}")  # Log first 5 errors
    
    return results or None


def _clean_json_string(text: str) -> str | None:
    """Clean common LLM JSON formatting issues.
    
    Handles:
    - Trailing commas before closing braces/brackets
    - Single quotes instead of double quotes
    - Comments (// and /* */)
    - Extra whitespace and newlines
    
    Args:
        text: The text to clean.
        
    Returns:
        Cleaned JSON string or None if input is empty.
    """
    if not text:
        return None
    
    # First try to extract just the JSON object using brace matching
    json_str = _extract_json_with_brace_matching(text)
    if not json_str:
        json_str = text.strip()
    
    # Remove comments (// style) - preserve URLs and quoted strings
    lines = []
    for line in json_str.split('\n'):
        if '//' in line:
            comment_start = _find_comment_start(line)
            if comment_start >= 0:
                line = line[:comment_start]
        lines.append(line)
    json_str = '\n'.join(lines)
    
    # Remove /* */ comments
    json_str = _remove_block_comments(json_str)
    
    # Remove trailing commas before } or ]
    json_str = re.sub(r',\s*}', '}', json_str)
    json_str = re.sub(r',\s*\]', ']', json_str)
    
    # Replace single quotes with double quotes (carefully)
    json_str = _replace_single_quotes(json_str)
    
    return json_str.strip()


def _find_comment_start(line: str) -> int:
    """Find the start of a // comment in a line, respecting quoted strings.
    
    Args:
        line: The line to search.
        
    Returns:
        Index of comment start or -1 if not found.
    """
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
            return i
    return -1


def _remove_block_comments(text: str) -> str:
    """Remove /* */ style block comments from text.
    
    Args:
        text: The text to process.
        
    Returns:
        Text with block comments removed.
    """
    while '/*' in text and '*/' in text:
        start = text.find('/*')
        end = text.find('*/', start)
        if start >= 0 and end > start:
            text = text[:start] + text[end + 2:]
        else:
            break
    return text


def _replace_single_quotes(json_str: str) -> str:
    """Replace single quotes with double quotes, respecting quoted strings.
    
    Args:
        json_str: The JSON string to process.
        
    Returns:
        String with single quotes replaced by double quotes.
    """
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
    return ''.join(result)


def _extract_json_with_brace_matching(text: str) -> str | None:
    """Extract a JSON object from text using brace counting.
    
    This handles nested braces correctly by counting open/close braces.
    """
    start = text.find("{")
    if start == -1:
        return None
    
    brace_count = 0
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
                if brace_count == 0:
                    return text[start:i+1]
    
    return None


def _find_json_objects(text: str) -> list[str]:
    """Find all potential JSON objects in text using brace matching."""
    results = []
    i = 0
    while i < len(text):
        if text[i] == "{":
            json_str = _extract_json_with_brace_matching(text[i:])
            if json_str and '"' in json_str:  # Must contain at least one quoted string
                results.append(json_str)
                i += len(json_str)
            else:
                i += 1
        else:
            i += 1
    return results


# Grade extraction constants
GRADE_KEYWORDS: dict[str, dict[str, int]] = {
    "Correct": {
        "correct": 2, "right": 2, "accurate": 2, "valid": 1, 
        "proper": 1, "appropriate": 1, "satisfactory": 1,
        "well done": 3, "excellent": 2, "good": 1, "perfect": 3,
        "correctly": 2, "true": 1
    },
    "Incorrect": {
        "incorrect": 3, "wrong": 3, "error": 2, "mistake": 2, 
        "invalid": 2, "unsatisfactory": 2, "fail": 2,
        "incorrectly": 2, "false": 2, "not correct": 3,
        "not right": 3, "does not match": 2, "contradicts": 2
    },
    "Partially Correct": {
        "partially correct": 4, "partial credit": 3, "partially right": 3,
        "some correct": 3, "incomplete": 2, "partial": 2,
        "mostly correct": 2, "partly correct": 3, "half correct": 3,
        "some errors": 2, "minor errors": 1, "on the right track": 2
    },
}

EXPLICIT_GRADE_PATTERNS: list[tuple[str, int]] = [
    # "The grade is X" or "Grade: X"
    (r'(?:the\s+)?(?:final\s+)?(?:grade|score|assessment|evaluation|verdict)\s*(?:is|[:=])\s*["\']?([^"\'\n.]+)["\']?', 10),
    # "I would grade this as X"
    (r'(?:i\s+)?(?:would\s+)?(?:grade|score|rate|assess)\s*(?:this\s+)?(?:as|at)\s*["\']?([^"\'\n.]+)["\']?', 10),
    # "The answer is correct/incorrect"
    (r'(?:the\s+)?(?:answer|solution)\s+(?:is\s+)?(["\']?(?:correct|incorrect|partially correct|partial|wrong)["\']?)', 8),
    # "Grade: X" at start of line
    (r'^\s*(?:grade|score)\s*[:=]\s*["\']?([^"\'\n]+)["\']?', 10),
    # "X is the correct answer"
    (r'(?:is\s+)?(?:the\s+)?(correct|incorrect|right|wrong)\s+(?:answer|solution)', 8),
]

NUMERIC_SCORE_PATTERNS: list[str] = [
    # "Score: 85" or "Grade = 90"
    r'(?:score|grade)\s*[:=]\s*(\d+(?:\.\d+)?)\s*(?:/\s*(\d+))?',
    # "85/100" or "8.5 out of 10"
    r'(\d+(?:\.\d+)?)\s*(?:out\s+of|/\s*)(\d+)',
    # "85 points" or "90 percent"
    r'(\d+(?:\.\d+)?)\s*(?:points?|percent|%|pct)',
]

NEGATION_PATTERNS: list[str] = [
    r'not\s+(correct|right|accurate)',
    r'(?:is|are)\s+not\s+(correct|right)',
    r"(?:does not|doesn't)\s+(?:match|equal|correspond)",
]

# Score thresholds for numeric interpretation: (min_score, grade, points)
SCORE_THRESHOLDS_100: list[tuple[float, str, int]] = [
    (85, "Correct", 5),
    (70, "Correct", 3),
    (50, "Partially Correct", 4),
    (30, "Partially Correct", 2),
    (0, "Incorrect", 4),
]

SCORE_THRESHOLDS_10: list[tuple[float, str, int]] = [
    (8.5, "Correct", 4),
    (7, "Correct", 2),
    (5, "Partially Correct", 3),
    (3, "Partially Correct", 1),
    (0, "Incorrect", 3),
]


def _calculate_keyword_scores(text_lower: str) -> dict[str, int]:
    """Calculate scores based on keyword matching.
    
    Args:
        text_lower: Lowercase text to search.
        
    Returns:
        Dictionary mapping grades to their scores.
    """
    grade_scores: dict[str, int] = {}
    for grade, keywords in GRADE_KEYWORDS.items():
        score = 0
        for keyword, weight in keywords.items():
            count = text_lower.count(keyword)
            if count > 0:
                score += count * weight
        grade_scores[grade] = score
    return grade_scores


def _apply_explicit_patterns(text_lower: str, grade_scores: dict[str, int]) -> None:
    """Apply explicit grade statement patterns to boost scores.
    
    Args:
        text_lower: Lowercase text to search.
        grade_scores: Dictionary to update with additional scores.
    """
    for pattern, weight in EXPLICIT_GRADE_PATTERNS:
        matches = re.findall(pattern, text_lower, re.IGNORECASE | re.MULTILINE)
        for match in matches:
            if isinstance(match, str):
                match_clean = match.strip().strip('"\'').lower()
                for grade, keywords in GRADE_KEYWORDS.items():
                    for keyword in keywords:
                        if keyword in match_clean:
                            grade_scores[grade] += weight
                            break


def _apply_numeric_scores(text_lower: str, grade_scores: dict[str, int]) -> None:
    """Interpret numeric scores and update grade scores.
    
    Args:
        text_lower: Lowercase text to search.
        grade_scores: Dictionary to update with additional scores.
    """
    for pattern in NUMERIC_SCORE_PATTERNS:
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        for match in matches:
            if isinstance(match, tuple):
                try:
                    score = float(match[0])
                    total = float(match[1]) if match[1] and match[1].strip() else 100.0
                    if total > 0:
                        percentage = (score / total) * 100
                        for threshold, grade, points in SCORE_THRESHOLDS_100:
                            if percentage >= threshold:
                                grade_scores[grade] += points
                                break
                except (ValueError, TypeError):
                    pass
            elif isinstance(match, str):
                try:
                    score = float(match)
                    # Assume 0-10 scale if score is small, 0-100 otherwise
                    thresholds = SCORE_THRESHOLDS_10 if score <= 10 else SCORE_THRESHOLDS_100
                    for threshold, grade, points in thresholds:
                        if score >= threshold:
                            grade_scores[grade] += points
                            break
                except ValueError:
                    pass


def _apply_negation_patterns(text_lower: str, grade_scores: dict[str, int]) -> None:
    """Detect negation patterns and adjust scores accordingly.
    
    Args:
        text_lower: Lowercase text to search.
        grade_scores: Dictionary to update with additional scores.
    """
    for pattern in NEGATION_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            # Boost Incorrect score when negations are found
            grade_scores["Incorrect"] += 3


def _extract_grade_from_text(text: str) -> str | None:
    """Extract a grade/assessment from plain text when JSON parsing fails.
    
    Looks for common grading patterns and keywords in the text.
    Returns the most likely grade or None if no grade is found.
    
    This function uses a multi-layered approach:
    1. Direct keyword matching with weighted scoring
    2. Explicit grade statement patterns
    3. Numeric score interpretation
    4. Context-aware phrase matching
    
    Args:
        text: The text to extract a grade from.
        
    Returns:
        The extracted grade or None if no grade could be determined.
    """
    if not text:
        return None
    
    text_lower = text.lower()
    
    # Calculate weighted scores for each grade
    grade_scores = _calculate_keyword_scores(text_lower)
    
    # Apply explicit grade statement patterns
    _apply_explicit_patterns(text_lower, grade_scores)
    
    # Interpret numeric scores
    _apply_numeric_scores(text_lower, grade_scores)
    
    # Detect negation patterns
    _apply_negation_patterns(text_lower, grade_scores)
    
    # Return the grade with the highest score, if any
    if grade_scores:
        best_grade = max(grade_scores, key=grade_scores.get)
        if grade_scores[best_grade] > 0:
            logger.debug(f"Extracted grade '{best_grade}' with score {grade_scores[best_grade]}")
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
        extraction_stats: Statistics about JSON extraction success/failure.
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
        self.extraction_stats = {
            "successful_extractions": 0,
            "fallback_extractions": 0,
            "failed_extractions": 0,
            "total_attempts": 0,
        }
    
    def get_extraction_stats(self) -> dict:
        """Return statistics about JSON extraction performance.
        
        Returns:
            Dictionary with extraction statistics.
        """
        return dict(self.extraction_stats)
    
    def reset_extraction_stats(self) -> None:
        """Reset extraction statistics."""
        self.extraction_stats = {
            "successful_extractions": 0,
            "fallback_extractions": 0,
            "failed_extractions": 0,
            "total_attempts": 0,
        }

    def _build_grading_instruction(self, inputs: dict) -> str:
        """Build the grading instruction prompt from inputs.
        
        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer
            
        Returns:
            Formatted instruction string.
        """
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

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

Think carefully and provide a fair assessment based on the official solution and grading guidelines."""

    def _extract_prediction_from_response(self, msg_history: list[dict]) -> tuple[str, str, bool]:
        """Extract prediction and reasoning from LLM response.
        
        Args:
            msg_history: Message history from LLM interaction.
            
        Returns:
            Tuple of (prediction, reasoning, success_flag).
        """
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
            prediction = "None"
            reasoning = ""
            
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
            
            return str(prediction), reasoning, True
        
        return "None", "", False

    def _get_retry_feedback(self, error: Exception | None = None) -> str:
        """Generate feedback message for retry attempts.
        
        Args:
            error: Optional error that occurred during extraction.
            
        Returns:
            Feedback message string.
        """
        if error:
            return (
                f"Error parsing your response: {error}. "
                "Please ensure your response contains valid JSON wrapped in <json>...</json> tags. "
                "The JSON must have 'reasoning' and 'response' fields."
            )
        return (
            "Your previous response did not contain valid JSON in the required format. "
            "Please respond with a JSON object wrapped in <json>...</json> tags. "
            "The JSON must have 'reasoning' and 'response' fields. "
            "Example format:\n"
            "<json>\n"
            '{\n  "reasoning": "The student correctly identified...",\n  "response": "Correct"\n}'
            "\n</json>"
        )

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with chain-of-thought reasoning.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        instruction = self._build_grading_instruction(inputs)
        msg_history: list[dict] = []
        prediction = "None"
        extraction_success = False
        used_fallback = False
        
        for attempt in range(self.max_retries + 1):
            self.extraction_stats["total_attempts"] += 1
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=msg_history,
            )

            try:
                prediction, reasoning, success = self._extract_prediction_from_response(msg_history)
                
                if success:
                    extraction_success = True
                    break
                
                # Try fallback extraction
                last_msg = msg_history[-1]["text"] if msg_history else ""
                fallback_prediction = _extract_grade_from_text(last_msg)
                if fallback_prediction and fallback_prediction != "None":
                    self.log_fn(f"Using fallback grade extraction: {fallback_prediction}")
                    prediction = fallback_prediction
                    used_fallback = True
                    break
                
                # Need to retry
                if attempt < self.max_retries:
                    self.log_fn(f"No JSON found in attempt {attempt + 1}, retrying with feedback...")
                    feedback = self._get_retry_feedback()
                    msg_history.append({"role": "user", "text": feedback})
                    instruction = feedback
                else:
                    self.log_fn(f"No JSON found after {self.max_retries + 1} attempts")
                    
            except Exception as e:
                self.log_fn(f"Error extracting prediction (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries:
                    feedback = self._get_retry_feedback(e)
                    msg_history.append({"role": "user", "text": feedback})
                    instruction = feedback
        
        # Update extraction statistics
        if extraction_success:
            self.extraction_stats["successful_extractions"] += 1
        elif used_fallback:
            self.extraction_stats["fallback_extractions"] += 1
        else:
            self.extraction_stats["failed_extractions"] += 1

        return str(prediction), msg_history
