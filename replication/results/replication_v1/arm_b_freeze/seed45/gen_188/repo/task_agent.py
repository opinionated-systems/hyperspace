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
        return None
    
    # Size limit to prevent memory issues with huge inputs
    if len(text) > MAX_JSON_SIZE:
        logger.warning(f"Text exceeds maximum size ({MAX_JSON_SIZE}), truncating for JSON extraction")
        text = text[:MAX_JSON_SIZE]
    
    results = []
    parse_errors = []  # Track parsing errors for debugging
    
    # Try extraction strategies in order of preference
    extraction_strategies = [
        ("xml_tags", _extract_from_xml_tags),
        ("markdown_json", _extract_from_markdown_json),
        ("markdown_generic", _extract_from_markdown_generic),
        ("bare_json", _extract_from_bare_json),
    ]
    
    for strategy_name, strategy_func in extraction_strategies:
        if results:
            break
        try:
            strategy_results, strategy_errors = strategy_func(text)
            results.extend(strategy_results)
            parse_errors.extend(strategy_errors)
        except Exception as e:
            parse_errors.append(f"{strategy_name} strategy failed: {e}")
            logger.debug(f"Extraction strategy '{strategy_name}' failed: {e}")
    
    # Log parsing errors if we found no results but had attempts
    if not results and parse_errors:
        logger.debug(f"JSON extraction failed. Errors: {parse_errors[:5]}")  # Log first 5 errors
    
    return results or None


def _extract_from_xml_tags(text: str) -> tuple[list[dict], list[str]]:
    """Extract JSON from <json>...</json> XML-style tags.
    
    Args:
        text: The text to extract from.
        
    Returns:
        Tuple of (list of parsed JSON objects, list of error messages).
    """
    results = []
    errors = []
    search_from = 0
    
    while True:
        start = text.find("<json>", search_from)
        if start == -1:
            break
        end = text.find("</json>", start)
        if end == -1:
            errors.append(f"Found <json> at position {start} but no closing </json> tag")
            break
        
        inner = text[start + 6:end].strip()
        search_from = end + 7
        
        if not inner:
            continue
            
        parsed = _parse_json_with_fallbacks(inner, errors)
        if parsed is not None:
            results.append(parsed)
    
    return results, errors


def _extract_from_markdown_json(text: str) -> tuple[list[dict], list[str]]:
    """Extract JSON from ```json ... ``` markdown code blocks.
    
    Args:
        text: The text to extract from.
        
    Returns:
        Tuple of (list of parsed JSON objects, list of error messages).
    """
    results = []
    errors = []
    
    json_blocks = re.findall(r'```json\s*(.*?)```', text, re.DOTALL)
    for block in json_blocks:
        block = block.strip()
        if not block:
            continue
        parsed = _parse_json_with_fallbacks(block, errors)
        if parsed is not None:
            results.append(parsed)
    
    return results, errors


def _extract_from_markdown_generic(text: str) -> tuple[list[dict], list[str]]:
    """Extract JSON from generic ``` ... ``` markdown code blocks.
    
    Args:
        text: The text to extract from.
        
    Returns:
        Tuple of (list of parsed JSON objects, list of error messages).
    """
    results = []
    errors = []
    
    code_blocks = re.findall(r'```\s*(.*?)```', text, re.DOTALL)
    for block in code_blocks:
        block = block.strip()
        if not block or block.startswith('json'):
            continue
        # Quick check if it looks like JSON
        if block.startswith('{') or block.startswith('['):
            parsed = _parse_json_with_fallbacks(block, errors)
            if parsed is not None:
                results.append(parsed)
    
    return results, errors


def _extract_from_bare_json(text: str) -> tuple[list[dict], list[str]]:
    """Extract bare JSON objects from text using brace matching.
    
    Args:
        text: The text to extract from.
        
    Returns:
        Tuple of (list of parsed JSON objects, list of error messages).
    """
    results = []
    errors = []
    
    potential_jsons = _find_json_objects(text)
    for pj in potential_jsons:
        parsed = _parse_json_with_fallbacks(pj, errors)
        if parsed is not None:
            results.append(parsed)
    
    return results, errors


def _parse_json_with_fallbacks(text: str, errors: list[str]) -> dict | None:
    """Attempt to parse JSON with multiple fallback strategies.
    
    Tries direct parsing first, then brace matching, then cleaning.
    
    Args:
        text: The text to parse.
        errors: List to append error messages to.
        
    Returns:
        Parsed JSON object or None if all strategies fail.
    """
    # Try direct parsing first
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        errors.append(f"Direct JSON parse failed: {e}")
    
    # Try brace matching extraction
    try:
        json_str = _extract_json_with_brace_matching(text)
        if json_str:
            return json.loads(json_str)
        else:
            errors.append("Brace matching returned no valid JSON")
    except (json.JSONDecodeError, ValueError) as e:
        errors.append(f"Brace matching failed: {e}")
    
    # Try cleaning common LLM formatting issues
    try:
        cleaned = _clean_json_string(text)
        if cleaned:
            return json.loads(cleaned)
        else:
            errors.append("JSON cleaning returned empty result")
    except (json.JSONDecodeError, ValueError) as e:
        errors.append(f"Cleaned JSON parse failed: {e}")
    
    return None


def _clean_json_string(text: str) -> str | None:
    """Clean common LLM JSON formatting issues.
    
    Handles:
    - Trailing commas before closing braces/brackets
    - Single quotes instead of double quotes
    - Comments (// and /* */)
    - Extra whitespace and newlines
    """
    if not text:
        return None
    
    # First try to extract just the JSON object using brace matching
    json_str = _extract_json_with_brace_matching(text)
    if not json_str:
        json_str = text.strip()
    
    # Remove comments (// style)
    lines = []
    for line in json_str.split('\n'):
        # Remove // comments but preserve URLs
        if '//' in line:
            # Simple heuristic: if // is inside quotes, keep it
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
    
    # Remove /* */ comments
    while '/*' in json_str and '*/' in json_str:
        start = json_str.find('/*')
        end = json_str.find('*/', start) + 2
        if start >= 0 and end > start:
            json_str = json_str[:start] + json_str[end:]
        else:
            break
    
    # Remove trailing commas before } or ]
    json_str = re.sub(r',\s*}', '}', json_str)
    json_str = re.sub(r',\s*\]', ']', json_str)
    
    # Replace single quotes with double quotes (carefully)
    # Only replace quotes that are not inside strings
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
    
    return json_str.strip()


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


def _extract_grade_from_text(text: str) -> str | None:
    """Extract a grade/assessment from plain text when JSON parsing fails.
    
    Looks for common grading patterns and keywords in the text.
    Returns the most likely grade or None if no grade is found.
    
    This function uses a multi-layered approach:
    1. Direct keyword matching with weighted scoring
    2. Explicit grade statement patterns
    3. Numeric score interpretation
    4. Context-aware phrase matching
    """
    if not text:
        return None
    
    text_lower = text.lower()
    grade_scores: dict[str, int] = {}
    
    # Apply all scoring strategies
    _score_keyword_patterns(text_lower, grade_scores)
    _score_explicit_patterns(text_lower, grade_scores)
    _score_numeric_patterns(text_lower, grade_scores)
    _score_negation_patterns(text_lower, grade_scores)
    
    # Return the grade with the highest score, if any
    if grade_scores:
        best_grade = max(grade_scores, key=grade_scores.get)
        if grade_scores[best_grade] > 0:
            logger.debug(f"Extracted grade '{best_grade}' with score {grade_scores[best_grade]}")
            return best_grade
    
    return None


def _score_keyword_patterns(text_lower: str, grade_scores: dict[str, int]) -> None:
    """Score grades based on keyword frequency with weights.
    
    Args:
        text_lower: Lowercase text to search.
        grade_scores: Dictionary to update with scores.
    """
    # Define grade patterns with their associated keywords and weights
    grade_patterns = [
        ("Correct", {
            "correct": 2, "right": 2, "accurate": 2, "valid": 1, 
            "proper": 1, "appropriate": 1, "satisfactory": 1,
            "well done": 3, "excellent": 2, "good": 1, "perfect": 3,
            "correctly": 2, "true": 1
        }),
        ("Incorrect", {
            "incorrect": 3, "wrong": 3, "error": 2, "mistake": 2, 
            "invalid": 2, "unsatisfactory": 2, "fail": 2,
            "incorrectly": 2, "false": 2, "not correct": 3,
            "not right": 3, "does not match": 2, "contradicts": 2
        }),
        ("Partially Correct", {
            "partially correct": 4, "partial credit": 3, "partially right": 3,
            "some correct": 3, "incomplete": 2, "partial": 2,
            "mostly correct": 2, "partly correct": 3, "half correct": 3,
            "some errors": 2, "minor errors": 1, "on the right track": 2
        }),
    ]
    
    for grade, keywords in grade_patterns:
        score = 0
        for keyword, weight in keywords.items():
            count = text_lower.count(keyword)
            if count > 0:
                score += count * weight
        grade_scores[grade] = grade_scores.get(grade, 0) + score


def _score_explicit_patterns(text_lower: str, grade_scores: dict[str, int]) -> None:
    """Score grades based on explicit grade statement patterns.
    
    Args:
        text_lower: Lowercase text to search.
        grade_scores: Dictionary to update with scores.
    """
    explicit_patterns = [
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
    
    # Map of keywords to grades for explicit pattern matching
    keyword_to_grade = {
        "correct": "Correct",
        "right": "Correct",
        "accurate": "Correct",
        "incorrect": "Incorrect",
        "wrong": "Incorrect",
        "partially correct": "Partially Correct",
        "partial": "Partially Correct",
    }
    
    for pattern, weight in explicit_patterns:
        matches = re.findall(pattern, text_lower, re.IGNORECASE | re.MULTILINE)
        for match in matches:
            if isinstance(match, str):
                match_clean = match.strip().strip('"\'').lower()
                for keyword, grade in keyword_to_grade.items():
                    if keyword in match_clean:
                        grade_scores[grade] = grade_scores.get(grade, 0) + weight
                        break


def _score_numeric_patterns(text_lower: str, grade_scores: dict[str, int]) -> None:
    """Score grades based on numeric score patterns.
    
    Args:
        text_lower: Lowercase text to search.
        grade_scores: Dictionary to update with scores.
    """
    numeric_patterns = [
        # "Score: 85" or "Grade = 90"
        r'(?:score|grade)\s*[:=]\s*(\d+(?:\.\d+)?)\s*(?:/\s*(\d+))?',
        # "85/100" or "8.5 out of 10"
        r'(\d+(?:\.\d+)?)\s*(?:out\s+of|/\s*)(\d+)',
        # "85 points" or "90 percent"
        r'(\d+(?:\.\d+)?)\s*(?:points?|percent|%|pct)',
    ]
    
    for pattern in numeric_patterns:
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        for match in matches:
            if isinstance(match, tuple):
                _score_tuple_match(match, grade_scores)
            elif isinstance(match, str):
                _score_string_match(match, grade_scores)


def _score_tuple_match(match: tuple, grade_scores: dict[str, int]) -> None:
    """Score a tuple match from numeric pattern (score, total)."""
    try:
        score = float(match[0])
        total = float(match[1]) if match[1] and match[1].strip() else 100.0
        if total > 0:
            percentage = (score / total) * 100
            if percentage >= 85:
                grade_scores["Correct"] = grade_scores.get("Correct", 0) + 5
            elif percentage >= 70:
                grade_scores["Correct"] = grade_scores.get("Correct", 0) + 3
            elif percentage >= 50:
                grade_scores["Partially Correct"] = grade_scores.get("Partially Correct", 0) + 4
            elif percentage >= 30:
                grade_scores["Partially Correct"] = grade_scores.get("Partially Correct", 0) + 2
            else:
                grade_scores["Incorrect"] = grade_scores.get("Incorrect", 0) + 4
    except (ValueError, TypeError):
        pass


def _score_string_match(match: str, grade_scores: dict[str, int]) -> None:
    """Score a string match from numeric pattern (just score value)."""
    try:
        score = float(match)
        # Assume 0-10 scale if score is small, 0-100 otherwise
        if score <= 10:
            if score >= 8.5:
                grade_scores["Correct"] = grade_scores.get("Correct", 0) + 4
            elif score >= 7:
                grade_scores["Correct"] = grade_scores.get("Correct", 0) + 2
            elif score >= 5:
                grade_scores["Partially Correct"] = grade_scores.get("Partially Correct", 0) + 3
            elif score >= 3:
                grade_scores["Partially Correct"] = grade_scores.get("Partially Correct", 0) + 1
            else:
                grade_scores["Incorrect"] = grade_scores.get("Incorrect", 0) + 3
        else:  # 0-100 scale
            if score >= 85:
                grade_scores["Correct"] = grade_scores.get("Correct", 0) + 4
            elif score >= 70:
                grade_scores["Correct"] = grade_scores.get("Correct", 0) + 2
            elif score >= 50:
                grade_scores["Partially Correct"] = grade_scores.get("Partially Correct", 0) + 3
            else:
                grade_scores["Incorrect"] = grade_scores.get("Incorrect", 0) + 3
    except ValueError:
        pass


def _score_negation_patterns(text_lower: str, grade_scores: dict[str, int]) -> None:
    """Score grades based on negation patterns that indicate incorrectness.
    
    Args:
        text_lower: Lowercase text to search.
        grade_scores: Dictionary to update with scores.
    """
    negation_patterns = [
        r'not\s+(correct|right|accurate)',
        r'(?:is|are)\s+not\s+(correct|right)',
        r'(?:does not|doesn\'t)\s+(?:match|equal|correspond)',
    ]
    
    for pattern in negation_patterns:
        if re.search(pattern, text_lower, re.IGNORECASE):
            # Boost Incorrect score when negations are found
            grade_scores["Incorrect"] = grade_scores.get("Incorrect", 0) + 3


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

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with chain-of-thought reasoning.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Build the grading instruction
        instruction = self._build_grading_instruction(inputs)
        
        # Initialize state
        msg_history: list[dict] = []
        prediction = "None"
        extraction_success = False
        used_fallback = False
        
        # Retry loop for JSON extraction
        for attempt in range(self.max_retries + 1):
            self.extraction_stats["total_attempts"] += 1
            
            # Get LLM response
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=msg_history,
            )
            
            # Try to extract prediction from response
            extracted_prediction, extraction_method = self._extract_prediction(msg_history)
            
            if extracted_prediction is not None:
                prediction = extracted_prediction
                if extraction_method == "json":
                    extraction_success = True
                elif extraction_method == "fallback":
                    used_fallback = True
                break
            
            # If extraction failed and we have retries left, add feedback
            if attempt < self.max_retries:
                instruction = self._build_retry_feedback(attempt + 1)
                msg_history.append({"role": "user", "text": instruction})
            else:
                self.log_fn(f"No JSON found after {self.max_retries + 1} attempts")
        
        # Update extraction statistics
        self._update_extraction_stats(extraction_success, used_fallback)
        
        return str(prediction), msg_history
    
    def _build_grading_instruction(self, inputs: dict) -> str:
        """Build the grading instruction prompt from inputs.
        
        Args:
            inputs: Dictionary containing problem data.
            
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
    
    def _extract_prediction(self, msg_history: list[dict]) -> tuple[str | None, str]:
        """Extract prediction from message history.
        
        Args:
            msg_history: List of message dictionaries.
            
        Returns:
            Tuple of (prediction or None, extraction method: 'json', 'fallback', or 'none').
        """
        if not msg_history:
            return None, "none"
        
        # Try the last assistant message first
        last_msg = msg_history[-1].get("text", "")
        extracted = _extract_jsons(last_msg)
        
        # If not found, try searching all messages
        if not extracted:
            for msg in reversed(msg_history):
                text = msg.get("text", "")
                extracted = _extract_jsons(text)
                if extracted:
                    break
        
        # If JSON extraction succeeded, extract prediction and reasoning
        if extracted:
            result = extracted[-1]
            prediction = None
            
            # Try multiple possible keys for the response
            for key in DEFAULT_RETRY_KEYS:
                if key in result:
                    prediction = result[key]
                    break
            
            # Extract and log reasoning if available
            reasoning = ""
            for key in REASONING_KEYS:
                if key in result:
                    reasoning = result[key]
                    break
            if reasoning:
                self.log_fn(f"Reasoning: {reasoning[:200]}...")
            
            return prediction, "json"
        
        # Try fallback grade extraction from text
        fallback_prediction = _extract_grade_from_text(last_msg)
        if fallback_prediction and fallback_prediction != "None":
            self.log_fn(f"Using fallback grade extraction: {fallback_prediction}")
            return fallback_prediction, "fallback"
        
        return None, "none"
    
    def _build_retry_feedback(self, attempt_num: int) -> str:
        """Build feedback message for retry attempts.
        
        Args:
            attempt_num: The current attempt number (1-indexed).
            
        Returns:
            Feedback message string.
        """
        self.log_fn(f"No JSON found in attempt {attempt_num}, retrying with feedback...")
        return (
            "Your previous response did not contain valid JSON in the required format. "
            "Please respond with a JSON object wrapped in <json>...</json> tags. "
            "The JSON must have 'reasoning' and 'response' fields. "
            "Example format:\n"
            "<json>\n"
            '{\n  "reasoning": "The student correctly identified...",\n  "response": "Correct"\n}'
            "\n</json>"
        )
    
    def _update_extraction_stats(self, extraction_success: bool, used_fallback: bool) -> None:
        """Update extraction statistics based on outcome.
        
        Args:
            extraction_success: Whether JSON extraction succeeded.
            used_fallback: Whether fallback extraction was used.
        """
        if extraction_success:
            self.extraction_stats["successful_extractions"] += 1
        elif used_fallback:
            self.extraction_stats["fallback_extractions"] += 1
        else:
            self.extraction_stats["failed_extractions"] += 1
