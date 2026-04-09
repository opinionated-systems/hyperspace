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


def _extract_json_from_markdown(text: str) -> list[dict] | None:
    """Extract JSON objects from markdown code blocks.

    Fallback for when <json> tags are not used but markdown code blocks are.
    """
    results = []
    # Match ```json ... ``` or just ``` ... ``` blocks
    pattern = r'```(?:json)?\s*(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    for match in matches:
        try:
            results.append(json.loads(match.strip()))
        except json.JSONDecodeError:
            continue
    return results or None


def _normalize_grade(grade: str) -> str:
    """Normalize grade to a standard format.

    Handles various grade formats and normalizes them.
    Enhanced to handle numeric grades, IMO-style scoring, and more variations.
    """
    if not isinstance(grade, str):
        grade = str(grade)
    
    original = grade.strip()
    grade = original.lower()
    
    # First, check for numeric grades (0, 1, 2, etc.)
    # These are common in IMO-style grading
    try:
        numeric_grade = float(grade)
        if numeric_grade == 0:
            return 'Incorrect'
        elif numeric_grade >= 1:
            return 'Correct'
        elif 0 < numeric_grade < 1:
            return 'Partial'
    except ValueError:
        pass
    
    # Handle IMO-style grade formats like "0/7", "1/7", "7/7"
    imo_match = re.match(r'^(\d+)\s*/\s*\d+$', grade)
    if imo_match:
        score = int(imo_match.group(1))
        if score == 0:
            return 'Incorrect'
        elif score >= 1:
            return 'Correct'
    
    # Handle letter grades (A, B, C, D, F)
    letter_grades = {
        'a': 'Correct', 'a+': 'Correct', 'a-': 'Correct',
        'b': 'Correct', 'b+': 'Correct', 'b-': 'Partial',
        'c': 'Partial', 'c+': 'Partial', 'c-': 'Partial',
        'd': 'Incorrect', 'd+': 'Partial', 'd-': 'Incorrect',
        'f': 'Incorrect',
    }
    if grade in letter_grades:
        return letter_grades[grade]
    
    # Map common variations to standard formats
    correct_variations = [
        'correct', 'right', 'true', 'yes', 'full', 'full credit', 
        'full marks', 'complete', 'valid', 'accepted', 'pass',
        'satisfactory', 'excellent', 'good', 'perfect', 'success',
        'accurate', 'proper', 'appropriate', 'sound', 'reasonable'
    ]
    incorrect_variations = [
        'incorrect', 'wrong', 'false', 'no', 'none', 'zero',
        'invalid', 'rejected', 'fail', 'error', 'mistake',
        'unsatisfactory', 'poor', 'bad', 'unsuccessful', 'inaccurate',
        'improper', 'inappropriate', 'unsound', 'unreasonable'
    ]
    partial_variations = [
        'partial', 'partial credit', 'half', 'incomplete',
        'partially correct', 'partially right', 'some credit',
        'partially valid', 'mostly correct', 'mostly right',
        'partially accurate', 'partially accepted', 'mixed'
    ]
    
    # Check for exact matches first, then substring matches
    if grade in correct_variations:
        return 'Correct'
    if grade in incorrect_variations:
        return 'Incorrect'
    if grade in partial_variations:
        return 'Partial'
    
    # Substring matching for more flexibility
    # Check for incorrect first (more specific negative indicators)
    if any(v in grade for v in incorrect_variations):
        return 'Incorrect'
    # Then check for partial
    if any(v in grade for v in partial_variations):
        return 'Partial'
    # Finally check for correct
    if any(v in grade for v in correct_variations):
        return 'Correct'
    
    # Handle edge cases with punctuation
    clean_grade = re.sub(r'[^\w\s]', '', grade).strip()
    if clean_grade in correct_variations:
        return 'Correct'
    if clean_grade in incorrect_variations:
        return 'Incorrect'
    if clean_grade in partial_variations:
        return 'Partial'
    
    # Return original if no normalization applied
    return original


class TaskAgent:
    """Task agent that solves IMO grading problems with enhanced reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def _build_grading_prompt(self, inputs: dict) -> str:
        """Build a structured grading prompt with clear evaluation criteria."""
        domain = inputs.get("domain", "")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        return f"""You are an expert mathematical grader specializing in {domain} problems.

Your task is to evaluate a student's answer to a mathematics problem and assign a grade.

## Problem Statement:
{problem}

## Official Solution:
{solution}

## Grading Guidelines:
{grading_guidelines}

## Student's Answer:
{student_answer}

## Evaluation Instructions:

1. First, understand what the problem is asking and identify key constraints
2. Analyze the official solution to understand the correct approach
3. Evaluate the student's answer against the solution and guidelines
4. Assign a grade based on correctness, reasoning, and completeness

## Response Format:

You MUST respond in JSON format wrapped in <json> tags:

<json>
{{
    "reasoning": "Your detailed analysis of the student's answer",
    "grade": "The final grade (use: 'Correct', 'Incorrect', or 'Partial')"
}}
</json>

Important: The "grade" field must contain ONLY one of: 'Correct', 'Incorrect', or 'Partial'."""

    def _extract_prediction(self, msg_history: list[dict]) -> str:
        """Extract prediction from message history with multiple fallback strategies."""
        if not msg_history:
            self.log_fn("No message history available for prediction extraction")
            return "None"
        
        # Find the last assistant message with content
        last_message = ""
        for msg in reversed(msg_history):
            if msg.get("role") == "assistant":
                text = msg.get("text", "")
                if text:
                    last_message = text
                    break
        
        if not last_message:
            self.log_fn("No assistant message with text content found")
            return "None"
        
        # Strategy 1: Extract from <json> tags
        extracted = _extract_jsons(last_message)
        if extracted:
            self.log_fn(f"Extracted JSON from <json> tags: {len(extracted)} objects")
            for json_obj in extracted:
                result = self._get_grade_from_json(json_obj)
                if result != "None":
                    self.log_fn(f"Successfully extracted grade from JSON: {result}")
                    return result
        
        # Strategy 2: Extract from markdown code blocks
        extracted = _extract_json_from_markdown(last_message)
        if extracted:
            self.log_fn(f"Extracted JSON from markdown: {len(extracted)} objects")
            for json_obj in extracted:
                result = self._get_grade_from_json(json_obj)
                if result != "None":
                    self.log_fn(f"Successfully extracted grade from markdown: {result}")
                    return result
        
        # Strategy 3: Look for grade patterns in plain text
        self.log_fn("Falling back to plain text extraction")
        result = self._extract_grade_from_text(last_message)
        if result != "None":
            self.log_fn(f"Successfully extracted grade from text: {result}")
        else:
            self.log_fn("Failed to extract grade from any source")
        return result

    def _get_grade_from_json(self, json_obj: dict) -> str:
        """Extract grade from JSON object with field priority."""
        if not isinstance(json_obj, dict):
            self.log_fn(f"Invalid JSON object type: {type(json_obj)}")
            return "None"
        
        # Priority order for grade fields - "grade" first since that's what we ask for in prompt
        priority_fields = ["grade", "response", "answer", "result", "score", "evaluation", "prediction"]
        
        for field in priority_fields:
            if field in json_obj:
                value = json_obj[field]
                self.log_fn(f"Found grade field '{field}' with value: {value}")
                if isinstance(value, str):
                    normalized = _normalize_grade(value)
                    if normalized != value.strip().lower():
                        self.log_fn(f"Normalized grade from '{value}' to '{normalized}'")
                    return normalized
                elif isinstance(value, (int, float)):
                    return _normalize_grade(str(value))
                elif isinstance(value, bool):
                    return "Correct" if value else "Incorrect"
        
        # If no recognized field, use the first string or numeric value found
        for key, value in json_obj.items():
            if isinstance(value, str):
                self.log_fn(f"Using first string value from field '{key}': {value}")
                return _normalize_grade(value)
            elif isinstance(value, (int, float)):
                self.log_fn(f"Using first numeric value from field '{key}': {value}")
                return _normalize_grade(str(value))
            elif isinstance(value, bool):
                self.log_fn(f"Using first boolean value from field '{key}': {value}")
                return "Correct" if value else "Incorrect"
        
        self.log_fn(f"No valid grade found in JSON object with keys: {list(json_obj.keys())}")
        return "None"

    def _extract_grade_from_text(self, text: str) -> str:
        """Extract grade from plain text using pattern matching.
        
        Enhanced with multiple extraction strategies and fallback mechanisms
        to handle various response formats from different LLMs.
        """
        # Strategy 1: Look for explicit grade statements with various formats
        patterns = [
            # JSON-style grade field (highest priority since we ask for it in prompt)
            (r'"grade"\s*:\s*"([^"]+)"', 'JSON grade field'),
            (r"'grade'\s*:\s*'([^']+)'", 'JSON grade field (single quote)'),
            # Standard grade assignment patterns
            (r'grade[\s]*[:=][\s]*["\']?([^"\'\n]+)["\']?', 'grade assignment'),
            (r'response[\s]*[:=][\s]*["\']?([^"\'\n]+)["\']?', 'response assignment'),
            (r'prediction[\s]*[:=][\s]*["\']?([^"\'\n]+)["\']?', 'prediction assignment'),
            (r'final grade[\s]*[:=][\s]*["\']?([^"\'\n]+)["\']?', 'final grade'),
            (r'assign[\s]+["\']?([^"\'\n]+)["\']?', 'assign statement'),
            # IMO-style numeric grades (0, 1, 2, 3, etc.)
            (r'(?:score|points?|mark)[\s]*[:=][\s]*(\d+)', 'score/mark assignment'),
            (r'(?:the\s+)?grade\s+(?:is|of)\s+["\']?(\d+|[^"\'\n]+)["\']?', 'grade is/of'),
            # IMO-style fraction grades (e.g., "0/7", "7/7")
            (r'(?:score|points?|mark)[\s]*[:=][\s]*(\d+\s*/\s*\d+)', 'fraction score'),
            (r'(?:the\s+)?grade\s+(?:is|of)\s+["\']?(\d+\s*/\s*\d+)["\']?', 'fraction grade'),
            # Evaluation result patterns
            (r'evaluation[\s]*[:=][\s]*["\']?([^"\'\n]+)["\']?', 'evaluation assignment'),
            (r'result[\s]*[:=][\s]*["\']?([^"\'\n]+)["\']?', 'result assignment'),
            # Answer/verdict patterns
            (r'(?:the\s+)?(?:answer|verdict|decision)\s+(?:is\s+)?["\']?([^"\'\n]+)["\']?', 'answer/verdict'),
            # Letter grade patterns
            (r'(?:grade|score)[\s]*[:=]?\s*["\']?([a-f][+-]?)["\']?', 'letter grade'),
            # Additional patterns for common LLM outputs
            (r'(?:therefore|thus|hence|so)[,\s]+(?:the\s+)?(?:grade|answer|result)\s+(?:is\s+)?["\']?([^"\'\n]+)["\']?', 'conclusion grade'),
            (r'(?:i\s+)?(?:conclude|determine|assign|give)[,\s]+["\']?([^"\'\n]+)["\']?', 'conclusion statement'),
        ]
        
        for pattern, pattern_name in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                grade = _normalize_grade(match.group(1).strip())
                self.log_fn(f"Extracted grade '{grade}' using pattern: {pattern_name}")
                return grade
        
        # Strategy 2: Look for standalone numeric grades at end of text
        # Common pattern: "The grade is 2" or just "2" at the end
        end_patterns = [
            (r'(?:^|\n)\s*(\d+)\s*$', 'standalone number at end'),
            (r'grade\s*[:\-]?\s*(\d+)(?:\s|$)', 'grade with number'),
            (r'(?:score|mark)\s*[:\-]?\s*(\d+)(?:\s|$)', 'score/mark with number'),
            (r'(?:^|\n)\s*(\d+\s*/\s*\d+)\s*$', 'standalone fraction at end'),
        ]
        for pattern, pattern_name in end_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                grade = _normalize_grade(match.group(1).strip())
                self.log_fn(f"Extracted grade '{grade}' using pattern: {pattern_name}")
                return grade
        
        # Strategy 3: Look for explicit correctness indicators
        correctness_patterns = [
            (r'\b(correct|right|true|valid|accepted|full\s+credit)\b', 'Correct', 'correctness indicator'),
            (r'\b(incorrect|wrong|false|invalid|rejected|no\s+credit)\b', 'Incorrect', 'incorrectness indicator'),
            (r'\b(partial|partially\s+correct|partial\s+credit|half\s+credit)\b', 'Partial', 'partial indicator'),
        ]
        for pattern, grade, pattern_name in correctness_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                self.log_fn(f"Extracted grade '{grade}' using pattern: {pattern_name}")
                return grade
        
        # Strategy 4: Look for grade in the last line of the response
        lines = [line.strip() for line in text.strip().split('\n') if line.strip()]
        if lines:
            last_line = lines[-1].lower()
            # Check if last line contains just a grade
            if re.match(r'^[\s"\']*(\d+|correct|incorrect|partial|right|wrong|true|false|yes|no)[\s"\']*$', last_line, re.IGNORECASE):
                grade = _normalize_grade(last_line.strip('"\' '))
                self.log_fn(f"Extracted grade '{grade}' from last line")
                return grade
            # Check if last line contains "grade" followed by a value
            grade_in_last = re.search(r'grade[\s]*[:=]?\s*["\']?([^"\'\n]+)["\']?$', last_line, re.IGNORECASE)
            if grade_in_last:
                grade = _normalize_grade(grade_in_last.group(1).strip())
                self.log_fn(f"Extracted grade '{grade}' from last line grade pattern")
                return grade
        
        # Strategy 5: Look for grade in the last few lines
        if len(lines) >= 2:
            for line in lines[-3:]:  # Check last 3 lines
                line_lower = line.lower()
                if any(word in line_lower for word in ['grade', 'score', 'result', 'answer', 'correct', 'incorrect']):
                    # Try to extract a grade word from this line
                    grade_match = re.search(r'\b(correct|incorrect|partial|right|wrong)\b', line, re.IGNORECASE)
                    if grade_match:
                        grade = _normalize_grade(grade_match.group(1))
                        self.log_fn(f"Extracted grade '{grade}' from context line")
                        return grade
        
        self.log_fn("No grade patterns matched in text")
        return "None"

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with enhanced error handling.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Log input information for debugging
        domain = inputs.get("domain", "unknown")
        self.log_fn(f"Starting grading for domain: {domain}")
        
        instruction = self._build_grading_prompt(inputs)

        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
            self.log_fn(f"LLM call successful, response length: {len(response) if response else 0} chars")
        except Exception as e:
            self.log_fn(f"LLM call failed: {e}")
            return "None", []

        # Extract prediction using multiple strategies
        prediction = self._extract_prediction(msg_history)
        
        if prediction == "None":
            self.log_fn(f"Failed to extract prediction from response: {response[:200] if response else 'empty'}")
        else:
            self.log_fn(f"Final prediction: {prediction}")

        return str(prediction), msg_history
