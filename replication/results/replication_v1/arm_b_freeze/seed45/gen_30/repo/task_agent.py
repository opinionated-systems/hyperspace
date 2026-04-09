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


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    Also handles markdown code blocks and bare JSON objects.
    Includes improved handling for nested braces, escaped characters, and 
    common LLM formatting issues like trailing commas and comments.
    """
    if not text or not isinstance(text, str):
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
            break
        inner = text[start + 6:end].strip()
        search_from = end + 7
        
        # Try multiple parsing strategies in order of preference
        parsed = _try_parse_json(inner)
        if parsed:
            results.append(parsed)
    
    # If no <json> blocks found, try markdown code blocks
    if not results:
        # Try ```json ... ``` blocks
        json_blocks = re.findall(r'```json\s*(.*?)```', text, re.DOTALL)
        for block in json_blocks:
            parsed = _try_parse_json(block)
            if parsed:
                results.append(parsed)
        
        # Try bare JSON objects as fallback with improved regex
        if not results:
            # Find JSON-like structures with nested brace support
            potential_jsons = _find_json_objects(text)
            for pj in potential_jsons:
                parsed = _try_parse_json(pj)
                if parsed:
                    results.append(parsed)
    
    return results or None


def _try_parse_json(text: str) -> dict | None:
    """Try to parse JSON using multiple strategies.
    
    Returns the parsed dict if successful, None otherwise.
    """
    if not text or not isinstance(text, str):
        return None
    
    text = text.strip()
    if not text:
        return None
    
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


def _clean_json_string(text: str) -> str | None:
    """Clean common LLM JSON formatting issues.
    
    Handles:
    - Trailing commas before closing braces/brackets
    - Single quotes instead of double quotes
    - Comments (// and /* */)
    - Extra whitespace and newlines
    - Unescaped newlines in strings
    - Control characters
    """
    if not text:
        return None
    
    # First try to extract just the JSON object using brace matching
    json_str = _extract_json_with_brace_matching(text)
    if not json_str:
        json_str = text.strip()
    
    # Remove control characters except common whitespace
    json_str = ''.join(char for char in json_str if ord(char) >= 32 or char in '\n\r\t')
    
    # Remove comments (// style) - handle properly with string detection
    lines = []
    for line in json_str.split('\n'):
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
    
    # Handle unescaped newlines in strings by escaping them
    # This is a common LLM error - putting actual newlines in JSON strings
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
        elif char == '\n' and in_string:
            result.append('\\n')  # Escape newlines inside strings
        elif char == '\r' and in_string:
            result.append('\\r')  # Escape carriage returns inside strings
        elif char == '\t' and in_string:
            result.append('\\t')  # Escape tabs inside strings
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
    """
    if not text or not isinstance(text, str):
        return None
    
    text_lower = text.lower()
    
    # Define grade patterns with their associated keywords and weights
    grade_patterns = [
        ("Correct", ["correct", "right", "accurate", "valid", "proper", "appropriate", "satisfactory", "true"], 1),
        ("Incorrect", ["incorrect", "wrong", "error", "mistake", "invalid", "unsatisfactory", "fail", "false"], 1),
        ("Partially Correct", ["partially correct", "partial credit", "partially right", "some correct", "incomplete", "partial"], 1),
    ]
    
    # Count occurrences of each grade's keywords
    grade_scores = {}
    for grade, keywords, weight in grade_patterns:
        score = 0
        for keyword in keywords:
            # Use word boundary matching for more accurate counting
            pattern = r'\b' + re.escape(keyword) + r'\b'
            matches = re.findall(pattern, text_lower)
            score += len(matches) * weight
        grade_scores[grade] = score
    
    # Look for explicit grade statements with higher weight
    explicit_patterns = [
        (r'(?:the\s+)?(?:final\s+)?(?:grade|score|assessment|evaluation|verdict)\s*(?:is|[:=])\s*["\']?([^"\'\n.]+)["\']?', 10),
        (r'(?:i\s+)?(?:would\s+)?(?:grade|score|rate|assess)\s*(?:this\s+)?(?:as|at)\s*["\']?([^"\'\n.]+)["\']?', 8),
        (r'grade\s*[:=]\s*["\']?([^"\'\n]+)["\']?', 8),
        (r'(?:answer|solution)\s+(?:is\s+)?(["\']?(?:correct|incorrect|partially correct|partial|wrong)["\']?)', 8),
        (r'(?:therefore|thus|hence|so)\s*,?\s*(?:the\s+)?(?:answer|solution|grade)\s+(?:is\s+)?["\']?([^"\'\n.]+)["\']?', 6),
    ]
    
    for pattern, weight in explicit_patterns:
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        for match in matches:
            if isinstance(match, str):
                match_clean = match.strip().strip('"\'').lower()
                for grade, keywords, _ in grade_patterns:
                    if any(kw in match_clean for kw in keywords):
                        grade_scores[grade] += weight
    
    # Look for negation patterns (e.g., "not correct" should favor Incorrect)
    negation_patterns = [
        (r'\bnot\s+(?:correct|right|accurate|valid)\b', "Incorrect", 5),
        (r'\bnot\s+(?:incorrect|wrong)\b', "Correct", 5),
    ]
    for pattern, grade, weight in negation_patterns:
        matches = re.findall(pattern, text_lower)
        grade_scores[grade] += len(matches) * weight
    
    # Look for numeric scores (0-100 or 0-10)
    numeric_patterns = [
        r'(?:score|grade)\s*[:=]?\s*(\d+(?:\.\d+)?)\s*(?:/\s*(\d+))?',
        r'(\d+(?:\.\d+)?)\s*(?:out\s+of|/\s*)(\d+)',
        r'(?:score|grade)\s*[:=]?\s*(\d+(?:\.\d+)?)\s*(?:points?)?',
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
                            grade_scores["Correct"] += 5
                        elif percentage >= 50:
                            grade_scores["Partially Correct"] += 5
                        else:
                            grade_scores["Incorrect"] += 5
                except (ValueError, TypeError):
                    pass
            elif isinstance(match, str):
                try:
                    score = float(match)
                    if score >= 8:
                        grade_scores["Correct"] += 3
                    elif score >= 5:
                        grade_scores["Partially Correct"] += 3
                    else:
                        grade_scores["Incorrect"] += 3
                except ValueError:
                    pass
    
    # Return the grade with the highest score, if any
    if grade_scores:
        best_grade = max(grade_scores, key=grade_scores.get)
        if grade_scores[best_grade] > 0:
            return best_grade
    
    return None


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.max_retries = 3  # Increased max retries for better robustness
        self._setup_logging(log_file)

    def _setup_logging(self, log_file: str) -> None:
        """Setup file logging if log_file is provided."""
        if log_file:
            import logging.handlers
            file_handler = logging.FileHandler(log_file, mode='a')
            file_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with chain-of-thought reasoning.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Validate inputs
        if not isinstance(inputs, dict):
            logger.error("Invalid inputs: expected dict, got %s", type(inputs).__name__)
            return "Error: Invalid inputs", []
        
        # Extract fields for better prompting
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        # Log input sizes for debugging
        logger.debug("Input sizes - problem: %d, solution: %d, guidelines: %d, answer: %d",
                    len(problem), len(solution), len(grading_guidelines), len(student_answer))

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
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=msg_history,
                )
            except Exception as e:
                logger.error("LLM call failed (attempt %d): %s", attempt + 1, e)
                if attempt < self.max_retries:
                    continue
                return str(prediction), msg_history

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
                    for key in ["response", "grade", "answer", "result", "assessment", "evaluation", "verdict", "conclusion"]:
                        if key in result:
                            prediction = result[key]
                            break
                    
                    # Extract reasoning if available
                    for key in ["reasoning", "analysis", "thought", "explanation", "rationale", "thinking"]:
                        if key in result:
                            reasoning = result[key]
                            break
                    
                    # Log reasoning if available
                    if reasoning:
                        self.log_fn(f"Reasoning: {reasoning[:200]}...")
                    
                    # Success - break out of retry loop
                    logger.info("Successfully extracted grade '%s' on attempt %d", prediction, attempt + 1)
                    break
                else:
                    # Try to extract any text that might be a grade/assessment
                    # This is a fallback for when JSON parsing fails completely
                    fallback_prediction = _extract_grade_from_text(last_msg)
                    if fallback_prediction and fallback_prediction != "None":
                        self.log_fn(f"Using fallback grade extraction: {fallback_prediction}")
                        prediction = fallback_prediction
                        logger.info("Successfully extracted grade via fallback '%s' on attempt %d", prediction, attempt + 1)
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
                    logger.warning("Failed to extract grade after %d attempts", self.max_retries + 1)
                    
            except Exception as e:
                self.log_fn(f"Error extracting prediction (attempt {attempt + 1}): {e}")
                logger.exception("Error extracting prediction")
                if attempt < self.max_retries:
                    feedback = (
                        f"Error parsing your response: {e}. "
                        "Please ensure your response contains valid JSON wrapped in <json>...</json> tags. "
                        "The JSON must have 'reasoning' and 'response' fields."
                    )
                    msg_history.append({"role": "user", "text": feedback})
                    instruction = feedback

        return str(prediction), msg_history
