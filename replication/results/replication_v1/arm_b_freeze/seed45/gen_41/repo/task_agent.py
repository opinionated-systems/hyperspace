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


def _try_parse_json(text: str, source: str, index: int = 0) -> dict | None:
    """Attempt to parse JSON text with multiple fallback strategies.
    
    Args:
        text: The text to parse
        source: Description of the source (for logging)
        index: Index of the item (for logging)
        
    Returns:
        Parsed JSON dict or None if parsing fails
    """
    text = text.strip()
    if not text:
        return None
    
    # Strategy 1: Direct parsing
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            logger.debug(f"_try_parse_json: direct parse succeeded for {source} #{index}")
            return result
    except json.JSONDecodeError:
        pass
    
    # Strategy 2: Brace matching extraction
    try:
        json_str = _extract_json_with_brace_matching(text)
        if json_str:
            result = json.loads(json_str)
            if isinstance(result, dict):
                logger.debug(f"_try_parse_json: brace matching succeeded for {source} #{index}")
                return result
    except (json.JSONDecodeError, ValueError):
        pass
    
    # Strategy 3: Clean and parse
    try:
        cleaned = _clean_json_string(text)
        if cleaned:
            result = json.loads(cleaned)
            if isinstance(result, dict):
                logger.debug(f"_try_parse_json: cleaning succeeded for {source} #{index}")
                return result
    except (json.JSONDecodeError, ValueError):
        pass
    
    logger.debug(f"_try_parse_json: all strategies failed for {source} #{index}")
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
    extraction_attempts = 0
    
    # Strategy 1: Find <json>...</json> blocks
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
        extraction_attempts += 1
        
        result = _try_parse_json(inner, "<json> block", extraction_attempts)
        if result:
            results.append(result)
    
    # Strategy 2: Try markdown code blocks if no results yet
    if not results:
        logger.debug("_extract_jsons: trying markdown code blocks")
        json_blocks = re.findall(r'```json\s*(.*?)```', text, re.DOTALL)
        logger.debug(f"_extract_jsons: found {len(json_blocks)} markdown json blocks")
        
        for i, block in enumerate(json_blocks):
            result = _try_parse_json(block, "markdown block", i + 1)
            if result:
                results.append(result)
    
    # Strategy 3: Try bare JSON objects as fallback
    if not results:
        logger.debug("_extract_jsons: trying bare JSON objects")
        potential_jsons = _find_json_objects(text)
        logger.debug(f"_extract_jsons: found {len(potential_jsons)} potential JSON objects")
        
        for i, pj in enumerate(potential_jsons):
            result = _try_parse_json(pj, "bare object", i + 1)
            if result:
                results.append(result)
    
    if results:
        logger.debug(f"_extract_jsons: returning {len(results)} JSON object(s)")
    else:
        logger.debug(f"_extract_jsons: no JSON objects found after {extraction_attempts} attempts")
    
    return results or None


def _clean_json_string(text: str) -> str | None:
    """Clean common LLM JSON formatting issues.
    
    Handles:
    - Trailing commas before closing braces/brackets
    - Single quotes instead of double quotes
    - Comments (// and /* */)
    - Extra whitespace and newlines
    - Unescaped newlines in string values
    - BOM (Byte Order Mark) characters
    
    Args:
        text: The text to clean
        
    Returns:
        Cleaned JSON string or None if cleaning fails
    """
    if not text:
        return None
    
    # Extract JSON object using brace matching
    json_str = _extract_json_with_brace_matching(text) or text.strip()
    
    # Remove BOM if present
    json_str = json_str.lstrip('\ufeff')
    
    # Remove // comments (preserve strings)
    json_str = _remove_line_comments(json_str)
    
    # Remove /* */ comments
    json_str = _remove_block_comments(json_str)
    
    # Remove trailing commas
    json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
    
    # Replace single quotes with double quotes (outside strings)
    json_str = _normalize_quotes(json_str)
    
    # Escape unescaped newlines and tabs inside strings
    json_str = _escape_string_content(json_str)
    
    return json_str.strip()


def _remove_line_comments(text: str) -> str:
    """Remove // style comments while preserving strings."""
    lines = []
    for line in text.split('\n'):
        if '//' not in line:
            lines.append(line.rstrip())
            continue
        
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
        
        lines.append(line[:comment_start].rstrip() if comment_start >= 0 else line.rstrip())
    
    return '\n'.join(lines)


def _remove_block_comments(text: str) -> str:
    """Remove /* */ style block comments."""
    while '/*' in text and '*/' in text:
        start = text.find('/*')
        end = text.find('*/', start)
        if start >= 0 and end > start:
            text = text[:start] + text[end + 2:]
        else:
            break
    return text


def _normalize_quotes(text: str) -> str:
    """Replace single quotes with double quotes outside of strings."""
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


def _escape_string_content(text: str) -> str:
    """Escape unescaped newlines and tabs inside strings."""
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
    
    Uses a scoring system based on keyword matches and explicit grade statements.
    Returns the most likely grade or None if no grade is found.
    """
    if not text:
        return None
    
    text_lower = text.lower()
    grade_scores = {"Correct": 0, "Incorrect": 0, "Partially Correct": 0}
    
    # Score based on keyword matches
    _score_keywords(text_lower, grade_scores)
    
    # Score based on explicit grade statements
    _score_explicit_patterns(text_lower, grade_scores)
    
    # Score based on negation patterns
    _score_negations(text_lower, grade_scores)
    
    # Score based on numeric patterns
    _score_numeric_patterns(text_lower, grade_scores)
    
    # Return the grade with the highest score
    best_grade = max(grade_scores, key=grade_scores.get)
    if grade_scores[best_grade] > 0:
        logger.debug(f"_extract_grade_from_text: selected '{best_grade}' with score {grade_scores[best_grade]}")
        return best_grade
    
    logger.debug("_extract_grade_from_text: no grade found")
    return None


def _score_keywords(text: str, scores: dict) -> None:
    """Score grades based on keyword occurrences."""
    keywords = {
        "Correct": [
            ("correct", 2), ("right", 2), ("accurate", 2), ("valid", 1),
            ("well done", 3), ("excellent", 2), ("perfect", 3)
        ],
        "Incorrect": [
            ("incorrect", 3), ("wrong", 2), ("invalid", 2),
            ("not correct", 3), ("not right", 3), ("does not match", 2)
        ],
        "Partially Correct": [
            ("partially correct", 3), ("partial credit", 3),
            ("incomplete", 2), ("partial", 2), ("mostly correct", 2)
        ],
    }
    
    for grade, words in keywords.items():
        for keyword, weight in words:
            scores[grade] += text.count(keyword) * weight


def _score_explicit_patterns(text: str, scores: dict) -> None:
    """Score grades based on explicit grade statements."""
    patterns = [
        (r'\bgrade\s*[:=]\s*["\']?([^"\'\n]+)["\']?', 5),
        (r'\b(?:the\s+)?(?:final\s+)?(?:grade|score|assessment)\s*(?:is|[:=])\s*["\']?([^"\'\n.]+)["\']?', 5),
        (r'\b(?:answer|solution)\s+(?:is\s+)?(["\']?(?:correct|incorrect|partially correct)["\']?)', 6),
    ]
    
    grade_keywords = {
        "Correct": ["correct", "right", "accurate"],
        "Incorrect": ["incorrect", "wrong", "invalid"],
        "Partially Correct": ["partially correct", "partial"]
    }
    
    for pattern, weight in patterns:
        for match in re.findall(pattern, text, re.IGNORECASE):
            match_clean = match.strip().strip('"\'').lower()
            for grade, keywords in grade_keywords.items():
                if any(kw in match_clean for kw in keywords):
                    scores[grade] += weight


def _score_negations(text: str, scores: dict) -> None:
    """Score grades based on negation patterns."""
    negations = [
        (r'\bnot\s+(correct|right|accurate)\b', "Incorrect", 4),
        (r'\bnot\s+(incorrect|wrong)\b', "Correct", 4),
    ]
    
    for pattern, grade, weight in negations:
        scores[grade] += len(re.findall(pattern, text, re.IGNORECASE)) * weight


def _score_numeric_patterns(text: str, scores: dict) -> None:
    """Score grades based on numeric score patterns."""
    # Pattern: score/grade: X or X/Y
    for match in re.findall(r'\b(?:score|grade)\s*[:=]?\s*(\d+(?:\.\d+)?)(?:/\s*(\d+))?\b', text, re.IGNORECASE):
        try:
            score = float(match[0])
            total = float(match[1]) if match[1] else 100.0
            if total > 0:
                pct = (score / total) * 100
                if pct >= 80:
                    scores["Correct"] += 4
                elif pct >= 50:
                    scores["Partially Correct"] += 4
                else:
                    scores["Incorrect"] += 4
        except (ValueError, TypeError):
            pass
    
    # Pattern: standalone numbers (assume 0-100 scale if > 10)
    for match in re.findall(r'\b(\d+(?:\.\d+)?)\s*points?\b', text, re.IGNORECASE):
        try:
            score = float(match)
            if score <= 10:
                threshold = 8
            else:
                threshold = 80
            
            if score >= threshold:
                scores["Correct"] += 3
            elif score >= threshold * 0.6:
                scores["Partially Correct"] += 3
            else:
                scores["Incorrect"] += 3
        except ValueError:
            pass


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
        
        instruction = self._build_instruction(inputs)
        msg_history = []
        prediction = "None"
        
        for attempt in range(self.max_retries + 1):
            self._extraction_stats["total_attempts"] += 1
            
            # Get LLM response
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

            # Extract prediction from response
            prediction, success = self._extract_prediction(msg_history, attempt)
            
            if success:
                break
            
            # Retry with feedback if not successful
            if attempt < self.max_retries:
                instruction = self._build_retry_feedback()
                msg_history.append({"role": "user", "text": instruction})
                self.log_fn(f"No valid JSON found in attempt {attempt + 1}, retrying...")
            else:
                self._extraction_stats["failed_extractions"] += 1
                self.log_fn(f"No valid JSON found after {self.max_retries + 1} attempts")

        # Log extraction stats periodically
        if self._extraction_stats["total_attempts"] % 100 == 0:
            logger.info(f"TaskAgent extraction stats: {self._extraction_stats}")
        
        logger.debug(f"TaskAgent.forward: returning prediction='{prediction}'")
        return str(prediction), msg_history

    def _build_instruction(self, inputs: dict) -> str:
        """Build the grading instruction prompt."""
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        logger.debug(f"TaskAgent.forward: problem={len(problem)}, solution={len(solution)}, "
                    f"student_answer={len(student_answer)}")

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

    def _build_retry_feedback(self) -> str:
        """Build feedback message for retry attempts."""
        return (
            "Your previous response did not contain valid JSON in the required format. "
            "Please respond with a JSON object wrapped in <json>...</json> tags. "
            "The JSON must have 'reasoning' and 'response' fields. "
            "Example format:\n"
            "<json>\n"
            '{\n  "reasoning": "The student correctly identified...",\n  "response": "Correct"\n}'
            "\n</json>"
        )

    def _extract_prediction(self, msg_history: list[dict], attempt: int) -> tuple[str, bool]:
        """Extract prediction from message history.
        
        Returns:
            (prediction, success)
        """
        if not msg_history:
            return "None", False
        
        last_msg = msg_history[-1].get("text", "")
        if not last_msg:
            logger.warning(f"TaskAgent.forward: empty last message on attempt {attempt + 1}")
            return "None", False
        
        # Try JSON extraction
        extracted = _extract_jsons(last_msg)
        
        # If not found in last message, search all messages
        if not extracted:
            for msg in reversed(msg_history):
                text = msg.get("text", "")
                if text:
                    extracted = _extract_jsons(text)
                    if extracted:
                        break
        
        if extracted:
            return self._process_json_result(extracted[-1])
        
        # Try fallback text extraction
        fallback = _extract_grade_from_text(last_msg)
        if fallback and fallback != "None":
            self._extraction_stats["fallback_extractions"] += 1
            self.log_fn(f"Using fallback grade extraction: {fallback}")
            return fallback, True
        
        return "None", False

    def _process_json_result(self, result: dict) -> tuple[str, bool]:
        """Process extracted JSON result.
        
        Returns:
            (prediction, success)
        """
        if not isinstance(result, dict):
            logger.warning(f"TaskAgent.forward: extracted result is not a dict: {type(result)}")
            return "None", False
        
        self._extraction_stats["json_extractions"] += 1
        
        # Try multiple possible keys for the response
        for key in ["response", "grade", "answer", "result", "assessment", "evaluation", "verdict", "conclusion"]:
            if key in result:
                prediction = result[key]
                logger.debug(f"TaskAgent.forward: found response in key '{key}': {prediction}")
                
                # Log reasoning if available
                for reason_key in ["reasoning", "analysis", "thought", "explanation", "rationale", "thinking"]:
                    if reason_key in result:
                        self.log_fn(f"Reasoning: {result[reason_key][:200]}...")
                        break
                
                return prediction, True
        
        logger.warning(f"TaskAgent.forward: no response key found in result: {list(result.keys())}")
        return "None", False
