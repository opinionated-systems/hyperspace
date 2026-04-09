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


def _try_parse_json(text: str) -> dict | None:
    """Try to parse a single JSON object from text with multiple fallback strategies.
    
    Returns the parsed dict or None if all strategies fail.
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
        
        parsed = _try_parse_json(inner)
        if parsed:
            results.append(parsed)
            logger.debug(f"_extract_jsons: successfully parsed JSON from <json> block")
        else:
            logger.debug(f"_extract_jsons: failed to parse <json> block content")
    
    # If no <json> blocks found, try markdown code blocks
    if not results:
        logger.debug("_extract_jsons: no valid JSON in <json> blocks, trying markdown code blocks")
        json_blocks = re.findall(r'```json\s*(.*?)```', text, re.DOTALL)
        logger.debug(f"_extract_jsons: found {len(json_blocks)} markdown json blocks")
        
        for block in json_blocks:
            parsed = _try_parse_json(block)
            if parsed:
                results.append(parsed)
                logger.debug(f"_extract_jsons: successfully parsed JSON from markdown block")
        
        # Try bare JSON objects as fallback
        if not results:
            logger.debug("_extract_jsons: no valid JSON in markdown blocks, trying bare JSON objects")
            potential_jsons = _find_json_objects(text)
            logger.debug(f"_extract_jsons: found {len(potential_jsons)} potential JSON objects")
            
            for pj in potential_jsons:
                parsed = _try_parse_json(pj)
                if parsed:
                    results.append(parsed)
                    logger.debug(f"_extract_jsons: successfully parsed bare JSON object")
    
    if results:
        logger.debug(f"_extract_jsons: returning {len(results)} JSON object(s)")
    else:
        logger.debug(f"_extract_jsons: no JSON objects found")
    
    return results or None


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
    """
    if not text:
        return None
    
    # First try to extract just the JSON object using brace matching
    json_str = _extract_json_with_brace_matching(text)
    if not json_str:
        json_str = text.strip()
    
    # Remove BOM if present
    json_str = json_str.lstrip('\ufeff')
    
    # Remove comments (// style) - more robust implementation
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
        lines.append(line.rstrip())  # Also strip trailing whitespace from each line
    json_str = '\n'.join(lines)
    
    # Remove /* */ comments (multiline)
    while '/*' in json_str and '*/' in json_str:
        start = json_str.find('/*')
        end = json_str.find('*/', start)
        if start >= 0 and end > start:
            json_str = json_str[:start] + json_str[end + 2:]
        else:
            break
    
    # Remove trailing commas before } or ] - handle multiple commas
    json_str = re.sub(r',\s*,\s*}', '}', json_str)  # Multiple trailing commas
    json_str = re.sub(r',\s*,\s*\]', ']', json_str)
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
    
    # Handle unescaped newlines in string values by replacing with \n
    # This is a more aggressive fix for malformed JSON
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
        elif in_string and char == '\n':
            result.append('\\n')  # Escape newlines inside strings
        elif in_string and char == '\t':
            result.append('\\t')  # Escape tabs inside strings
        else:
            result.append(char)
    json_str = ''.join(result)
    
    return json_str.strip()


def _extract_json_with_brace_matching(text: str) -> str | None:
    """Extract a JSON object from text using brace counting.
    
    This handles nested braces correctly by counting open/close braces.
    Also handles square brackets for JSON arrays.
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


def _find_json_objects(text: str) -> list[str]:
    """Find all potential JSON objects and arrays in text using brace matching."""
    results = []
    i = 0
    while i < len(text):
        if text[i] in "{[":
            json_str = _extract_json_with_brace_matching(text[i:])
            if json_str:
                # For objects: must contain at least one quoted string
                # For arrays: must contain at least one element
                if (text[i] == "{" and '"' in json_str) or (text[i] == "[" and len(json_str) > 2):
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
    grade_scores = _score_keywords(text_lower)
    grade_scores = _score_explicit_patterns(text_lower, grade_scores)
    grade_scores = _score_negations(text_lower, grade_scores)
    grade_scores = _score_numeric_scores(text_lower, grade_scores)
    
    # Return the grade with the highest score, if any
    if grade_scores:
        best_grade = max(grade_scores, key=grade_scores.get)
        if grade_scores[best_grade] > 0:
            logger.debug(f"_extract_grade_from_text: selected '{best_grade}' with score {grade_scores[best_grade]}")
            return best_grade
    
    logger.debug("_extract_grade_from_text: no grade found")
    return None


def _score_keywords(text: str) -> dict[str, int]:
    """Score grades based on keyword occurrences."""
    grade_keywords = {
        "Correct": [
            ("correct", 2), ("right", 2), ("accurate", 2), ("valid", 1), 
            ("proper", 1), ("appropriate", 1), ("satisfactory", 1),
            ("well done", 3), ("excellent", 2), ("perfect", 3), ("good", 1)
        ],
        "Incorrect": [
            ("incorrect", 3), ("wrong", 2), ("error", 1), ("mistake", 1), 
            ("invalid", 2), ("unsatisfactory", 2), ("fail", 2),
            ("not correct", 3), ("not right", 3), ("does not match", 2),
            ("doesn't match", 2), ("no match", 2)
        ],
        "Partially Correct": [
            ("partially correct", 3), ("partial credit", 3), ("partially right", 3), 
            ("some correct", 2), ("incomplete", 2), ("partial", 2),
            ("mostly correct", 2), ("mostly right", 2), ("half correct", 2),
            ("partly correct", 3), ("some errors", 1), ("minor errors", 1)
        ],
    }
    
    scores = {}
    for grade, keywords in grade_keywords.items():
        scores[grade] = sum(text.count(kw) * weight for kw, weight in keywords)
    return scores


def _score_explicit_patterns(text: str, scores: dict[str, int]) -> dict[str, int]:
    """Score grades based on explicit grade statements."""
    patterns = [
        (r'\bgrade\s*[:=]\s*["\']?([^"\'\n]+)["\']?', 5),
        (r'\b(?:the\s+)?(?:final\s+)?(?:grade|score|assessment|evaluation|verdict)\s*(?:is|[:=])\s*["\']?([^"\'\n.]+)["\']?', 5),
        (r'\b(?:i\s+)?(?:would\s+)?(?:grade|score|rate|assess)\s*(?:this\s+)?(?:as|at)\s*["\']?([^"\'\n.]+)["\']?', 5),
        (r'\b(?:answer|solution)\s+(?:is\s+)?(["\']?(?:correct|incorrect|partially correct|partial|wrong)["\']?)', 6),
        (r'\b(?:this\s+is|the\s+answer\s+is)\s+(["\']?(?:correct|incorrect|partially correct|partial|wrong)["\']?)', 6),
    ]
    
    grade_keywords = {
        "Correct": ["correct", "right", "accurate", "valid", "proper", "good", "excellent", "perfect"],
        "Incorrect": ["incorrect", "wrong", "error", "invalid", "fail"],
        "Partially Correct": ["partially correct", "partial", "partly correct", "incomplete"],
    }
    
    for pattern, weight in patterns:
        for match in re.findall(pattern, text, re.IGNORECASE):
            if isinstance(match, str):
                match_clean = match.strip().strip('"\'').lower()
                for grade, keywords in grade_keywords.items():
                    if any(kw in match_clean for kw in keywords):
                        scores[grade] = scores.get(grade, 0) + weight
                        break
    return scores


def _score_negations(text: str, scores: dict[str, int]) -> dict[str, int]:
    """Score grades based on negation patterns."""
    negations = [
        (r'\bnot\s+(correct|right|accurate|valid)\b', "Incorrect", 4),
        (r'\bnot\s+(incorrect|wrong|invalid)\b', "Correct", 4),
    ]
    for pattern, grade, weight in negations:
        scores[grade] = scores.get(grade, 0) + len(re.findall(pattern, text, re.IGNORECASE)) * weight
    return scores


def _score_numeric_scores(text: str, scores: dict[str, int]) -> dict[str, int]:
    """Score grades based on numeric score patterns."""
    patterns = [
        r'\b(?:score|grade)\s*[:=]?\s*(\d+(?:\.\d+)?)\s*(?:/\s*(\d+))?\b',
        r'\b(\d+(?:\.\d+)?)\s*(?:out\s+of|/\s*)(\d+)\b',
        r'\b(\d+(?:\.\d+)?)\s*points?\b',
    ]
    
    for pattern in patterns:
        for match in re.findall(pattern, text, re.IGNORECASE):
            score_val = _parse_numeric_score(match)
            if score_val is not None:
                grade = _numeric_to_grade(score_val)
                scores[grade] = scores.get(grade, 0) + 3
    return scores


def _parse_numeric_score(match) -> tuple[float, float] | None:
    """Parse a numeric score match into (score, total) tuple."""
    try:
        if isinstance(match, tuple):
            score = float(match[0])
            total = float(match[1]) if match[1] and match[1].strip() else 100.0
            return (score, total)
        elif isinstance(match, str):
            score = float(match)
            # Assume 0-10 scale if score is small, otherwise 0-100
            total = 10.0 if score <= 10 else 100.0
            return (score, total)
    except (ValueError, TypeError):
        pass
    return None


def _numeric_to_grade(score_tuple: tuple[float, float]) -> str:
    """Convert a numeric score to a grade category."""
    score, total = score_tuple
    if total <= 0:
        return "Incorrect"
    percentage = (score / total) * 100
    if percentage >= 80:
        return "Correct"
    elif percentage >= 50:
        return "Partially Correct"
    else:
        return "Incorrect"


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning.
    
    Enhanced with better error handling, logging, and multiple fallback strategies
    for extracting grades from LLM responses.
    """

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.max_retries = 3  # Increased max retries for better robustness
        self._extraction_stats = {
            "json_extractions": 0,
            "fallback_extractions": 0,
            "failed_extractions": 0,
            "total_attempts": 0
        }

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
        
        # Extract fields for better prompting
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        # Log input sizes for debugging
        logger.debug(f"TaskAgent.forward: problem length={len(problem)}, "
                    f"solution length={len(solution)}, "
                    f"student_answer length={len(student_answer)}")

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
- Do not include any text outside the <json>...</json> tags.

Think carefully and provide a fair assessment based on the official solution and grading guidelines."""

        msg_history = []
        prediction = "None"
        reasoning = ""
        extraction_method = None
        
        for attempt in range(self.max_retries + 1):
            self._extraction_stats["total_attempts"] += 1
            
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
                else:
                    return "Error: LLM call failed", msg_history

            # Extract prediction from response
            prediction, reasoning, extraction_method, should_retry = self._extract_prediction(
                msg_history, attempt
            )
            
            if prediction != "None" and not should_retry:
                break
            
            if should_retry and attempt < self.max_retries:
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
                instruction = feedback
            else:
                self._extraction_stats["failed_extractions"] += 1
                self.log_fn(f"No JSON found after {self.max_retries + 1} attempts")

        # Log extraction stats periodically
        if self._extraction_stats["total_attempts"] % 100 == 0:
            logger.info(f"TaskAgent extraction stats: {self._extraction_stats}")
        
        logger.debug(f"TaskAgent.forward: returning prediction='{prediction}', method={extraction_method}")
        return str(prediction), msg_history

    def _extract_prediction(self, msg_history: list[dict], attempt: int) -> tuple[str, str, str | None, bool]:
        """Extract prediction from message history.
        
        Returns: (prediction, reasoning, extraction_method, should_retry)
        """
        last_msg = msg_history[-1]["text"] if msg_history else ""
        
        if not last_msg:
            logger.warning(f"TaskAgent.forward: empty last message on attempt {attempt + 1}")
            return "None", "", None, True
        
        # Try to extract JSON from last message
        extracted = _extract_jsons(last_msg)
        
        # If not found, try searching all messages
        if not extracted:
            logger.debug(f"TaskAgent.forward: no JSON in last message, searching all messages")
            for msg in reversed(msg_history):
                text = msg.get("text", "")
                if text:
                    extracted = _extract_jsons(text)
                    if extracted:
                        logger.debug(f"TaskAgent.forward: found JSON in earlier message")
                        break
        
        if extracted:
            self._extraction_stats["json_extractions"] += 1
            result = extracted[-1]
            
            # Validate result is a dict
            if not isinstance(result, dict):
                logger.warning(f"TaskAgent.forward: extracted result is not a dict: {type(result)}")
                return "None", "", None, True
            
            # Try multiple possible keys for the response
            prediction = "None"
            for key in ["response", "grade", "answer", "result", "assessment", "evaluation", "verdict", "conclusion"]:
                if key in result:
                    prediction = result[key]
                    logger.debug(f"TaskAgent.forward: found response in key '{key}': {prediction}")
                    break
            
            # Extract reasoning if available
            reasoning = ""
            for key in ["reasoning", "analysis", "thought", "explanation", "rationale", "thinking"]:
                if key in result:
                    reasoning = result[key]
                    break
            
            if reasoning:
                self.log_fn(f"Reasoning: {reasoning[:200]}...")
            
            return prediction, reasoning, "json", prediction == "None"
        
        # Try fallback extraction
        fallback = _extract_grade_from_text(last_msg)
        if fallback and fallback != "None":
            self._extraction_stats["fallback_extractions"] += 1
            self.log_fn(f"Using fallback grade extraction: {fallback}")
            return fallback, "", "fallback", False
        
        return "None", "", None, True
