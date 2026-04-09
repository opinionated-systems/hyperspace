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
    Enhanced to handle numeric grades, IMO-style scoring, and edge cases.
    """
    if not isinstance(grade, str):
        grade = str(grade)
    
    original_grade = grade.strip()
    grade = original_grade.lower()
    
    # First, check for numeric grades (0, 1, 2, etc.)
    # These are common in IMO-style grading
    try:
        numeric_grade = float(grade)
        # IMO problems are typically scored 0-7 or 0-1
        # 0 = Incorrect, 1+ = Correct (or partial credit for intermediate values)
        if numeric_grade == 0:
            return 'Incorrect'
        elif numeric_grade >= 1:
            # For IMO-style 0-7 scoring, any positive score indicates some correctness
            return 'Correct'
        elif 0 < numeric_grade < 1:
            return 'Partial'
    except ValueError:
        pass
    
    # Handle quoted strings that might contain numbers
    # e.g., '"0"' or "'1'" 
    unquoted = grade.strip('"\'')
    if unquoted != grade:
        try:
            numeric_grade = float(unquoted)
            if numeric_grade == 0:
                return 'Incorrect'
            elif numeric_grade >= 1:
                return 'Correct'
            elif 0 < numeric_grade < 1:
                return 'Partial'
        except ValueError:
            pass
    
    # Map common variations to standard formats
    correct_variations = [
        'correct', 'right', 'true', 'yes', 'full', 'full credit', 
        'full marks', 'complete', 'valid', 'accepted', 'pass', 'solved',
        'success', 'accurate', 'perfect', 'excellent', 'good'
    ]
    incorrect_variations = [
        'incorrect', 'wrong', 'false', 'no', 'none', 'zero',
        'invalid', 'rejected', 'fail', 'error', 'mistake', 'unsolved',
        'failure', 'bad', 'unsatisfactory', 'unacceptable'
    ]
    partial_variations = [
        'partial', 'partial credit', 'half', 'incomplete',
        'partially correct', 'partially right', 'some credit',
        'partially solved', 'in progress', 'attempted', 'tried'
    ]
    
    # Check for exact matches first, then substring matches
    if grade in correct_variations:
        return 'Correct'
    if grade in incorrect_variations:
        return 'Incorrect'
    if grade in partial_variations:
        return 'Partial'
    
    # Substring matching for more flexibility - prioritize longer matches
    # to avoid false positives (e.g., "incorrectly" contains "correct")
    def find_best_match(text: str, variations: list[str]) -> str | None:
        """Find the best matching variation (longest match)."""
        best_match = None
        best_len = 0
        for v in variations:
            if v in text and len(v) > best_len:
                best_match = v
                best_len = len(v)
        return best_match
    
    # Check for incorrect first to avoid matching "correct" inside "incorrect"
    incorrect_match = find_best_match(grade, incorrect_variations)
    correct_match = find_best_match(grade, correct_variations)
    partial_match = find_best_match(grade, partial_variations)
    
    # Prioritize: incorrect > partial > correct (to avoid false positives)
    if incorrect_match:
        return 'Incorrect'
    elif partial_match:
        return 'Partial'
    elif correct_match:
        return 'Correct'
    
    # Return original if no normalization applied
    return original_grade


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

## Evaluation Framework:

Follow this structured evaluation process:

### Step 1: Problem Understanding
- What is the problem asking for?
- What are the key constraints and conditions?
- What is the expected answer format?

### Step 2: Solution Analysis
- What is the correct approach according to the official solution?
- What are the critical steps that must be present?
- What constitutes a complete vs. incomplete solution?

### Step 3: Student Answer Evaluation
- Did the student understand the problem correctly?
- What approach did the student take?
- Are the student's steps logically valid?
- Did the student show sufficient work and reasoning?
- Is the final answer mathematically correct?

### Step 4: Grade Assignment
Based on the grading guidelines, assign the appropriate grade considering:
- Correctness of the final answer
- Validity of the reasoning process
- Completeness of the solution
- Adherence to the expected solution method

## Grade Definitions:

- **Correct**: The student's answer is completely correct with valid reasoning
- **Incorrect**: The student's answer is wrong or contains fundamental errors
- **Partial**: The student's answer has some correct elements but is incomplete or has minor errors

For IMO-style problems, you may also use numeric grades:
- **0**: Incorrect (no credit)
- **1-7**: Partial to full credit (higher numbers = more credit)

## Response Format:

You MUST respond in JSON format wrapped in <json> tags:

<json>
{{
    "reasoning": "Your detailed step-by-step analysis following the Evaluation Framework above",
    "response": "The final grade you assign - use exactly one of: 'Correct', 'Incorrect', 'Partial', or a number like '0', '1', '2', etc."
}}
</json>

Important: 
- The "response" field must contain ONLY the grade value
- Use exactly 'Correct', 'Incorrect', 'Partial', or a single number
- Do not add any explanation in the response field"""

    def _extract_prediction(self, msg_history: list[dict]) -> str:
        """Extract prediction from message history with multiple fallback strategies."""
        if not msg_history:
            self.log_fn("No message history available for prediction extraction")
            return "None"
        
        last_message = msg_history[-1].get("text", "")
        if not last_message:
            self.log_fn("Last message has no text content")
            return "None"
        
        # Log the full response for debugging (truncated)
        preview = last_message[:500] + "..." if len(last_message) > 500 else last_message
        self.log_fn(f"Processing response: {preview}")
        
        # Strategy 1: Extract from <json> tags
        extracted = _extract_jsons(last_message)
        if extracted:
            self.log_fn(f"Extracted JSON from <json> tags: {len(extracted)} objects")
            # Try each extracted JSON object, starting from the last one
            for json_obj in reversed(extracted):
                result = self._get_grade_from_json(json_obj)
                if result != "None":
                    self.log_fn(f"Successfully extracted grade from JSON: {result}")
                    return result
        
        # Strategy 2: Extract from markdown code blocks
        extracted = _extract_json_from_markdown(last_message)
        if extracted:
            self.log_fn(f"Extracted JSON from markdown: {len(extracted)} objects")
            for json_obj in reversed(extracted):
                result = self._get_grade_from_json(json_obj)
                if result != "None":
                    self.log_fn(f"Successfully extracted grade from markdown: {result}")
                    return result
        
        # Strategy 3: Look for grade patterns in plain text
        self.log_fn("Falling back to plain text extraction")
        result = self._extract_grade_from_text(last_message)
        if result != "None":
            self.log_fn(f"Successfully extracted grade from text: {result}")
            return result
        
        # Strategy 4: Last resort - check if the entire response is just a grade
        clean_response = last_message.strip().lower()
        if clean_response in ['correct', 'incorrect', 'partial', '0', '1', '2', '3', '4', '5', '6', '7']:
            grade = _normalize_grade(clean_response)
            self.log_fn(f"Extracted grade from clean response: {grade}")
            return grade
        
        self.log_fn("Failed to extract grade from any source")
        return "None"

    def _get_grade_from_json(self, json_obj: dict) -> str:
        """Extract grade from JSON object with field priority."""
        if not isinstance(json_obj, dict):
            self.log_fn(f"Invalid JSON object type: {type(json_obj)}")
            return "None"
        
        # Priority order for grade fields - most specific to least specific
        priority_fields = [
            "response",    # Most common in our prompts
            "grade",       # Direct grade field
            "answer",      # Alternative field name
            "result",      # Result field
            "score",       # Numeric score
            "evaluation",  # Evaluation field
            "verdict",     # Verdict field
            "decision",    # Decision field
            "assessment",  # Assessment field
        ]
        
        # First pass: look for exact field matches
        for field in priority_fields:
            if field in json_obj:
                value = json_obj[field]
                self.log_fn(f"Found grade field '{field}' with value: {value}")
                if isinstance(value, str):
                    normalized = _normalize_grade(value)
                    if normalized != value:  # Normalization succeeded
                        return normalized
                    # If normalization returned the same value, check if it's a valid grade
                    if normalized.lower() in ['correct', 'incorrect', 'partial', '0', '1', '2', '3', '4', '5', '6', '7']:
                        return normalized
                elif isinstance(value, (int, float)):
                    return _normalize_grade(str(value))
        
        # Second pass: look for any field that might contain a grade
        # Check for fields with "grade", "score", "result" in the name
        for key, value in json_obj.items():
            key_lower = key.lower()
            if any(term in key_lower for term in ['grade', 'score', 'result', 'answer', 'evaluation', 'verdict']):
                self.log_fn(f"Found potential grade field '{key}' with value: {value}")
                if isinstance(value, str):
                    return _normalize_grade(value)
                elif isinstance(value, (int, float)):
                    return _normalize_grade(str(value))
        
        # Third pass: use the first string or numeric value found
        for key, value in json_obj.items():
            if isinstance(value, str):
                # Skip long text fields (likely reasoning, not grade)
                if len(value) <= 50:
                    self.log_fn(f"Using first short string value from field '{key}': {value}")
                    return _normalize_grade(value)
            elif isinstance(value, (int, float)):
                self.log_fn(f"Using first numeric value from field '{key}': {value}")
                return _normalize_grade(str(value))
        
        self.log_fn(f"No valid grade found in JSON object with keys: {list(json_obj.keys())}")
        return "None"

    def _extract_grade_from_text(self, text: str) -> str:
        """Extract grade from plain text using pattern matching.
        
        Enhanced with multiple extraction strategies and fallback mechanisms
        to handle various response formats from different LLMs.
        """
        # Strategy 1: Look for explicit grade statements with various formats
        # These patterns look for explicit grade assignments
        patterns = [
            # Standard grade assignment patterns with quotes
            (r'grade[\s]*[:=][\s]*["\']([^"\']+)["\']', 'grade assignment (quoted)'),
            (r'response[\s]*[:=][\s]*["\']([^"\']+)["\']', 'response assignment (quoted)'),
            (r'final grade[\s]*[:=][\s]*["\']([^"\']+)["\']', 'final grade (quoted)'),
            # Grade assignment without quotes (single word/number)
            (r'grade[\s]*[:=][\s]*([\w\d]+)(?:\s|$|\n)', 'grade assignment (unquoted)'),
            (r'response[\s]*[:=][\s]*([\w\d]+)(?:\s|$|\n)', 'response assignment (unquoted)'),
            # IMO-style numeric grades (0, 1, 2, 3, etc.)
            (r'(?:score|points?|mark)[\s]*[:=][\s]*(\d+)', 'score/mark assignment'),
            (r'(?:the\s+)?grade\s+(?:is|of)\s+["\']?(\d+|[^"\'\n]+)["\']?', 'grade is/of'),
            # Evaluation result patterns
            (r'evaluation[\s]*[:=][\s]*["\']?([^"\'\n]+)["\']?', 'evaluation assignment'),
            (r'result[\s]*[:=][\s]*["\']?([^"\'\n]+)["\']?', 'result assignment'),
            # Answer/verdict patterns
            (r'(?:the\s+)?(?:answer|verdict|decision)\s+(?:is\s+)?["\']?([^"\'\n]+)["\']?', 'answer/verdict'),
            # Assignment patterns
            (r'assign[\s]+["\']?([^"\'\n]+)["\']?', 'assign statement'),
            # "I assign/grade/give..." patterns
            (r'(?:i\s+)?(?:assign|grade|give)[\s]+(?:a\s+)?(?:grade\s+)?(?:of\s+)?["\']?([^"\'\n]+)["\']?', 'i assign/give'),
        ]
        
        for pattern, pattern_name in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                raw_grade = match.group(1).strip()
                # Skip if the match is too long (likely not a grade)
                if len(raw_grade) > 50:
                    continue
                grade = _normalize_grade(raw_grade)
                self.log_fn(f"Extracted grade '{grade}' using pattern: {pattern_name}")
                return grade
        
        # Strategy 2: Look for standalone numeric grades at end of text
        # Common pattern: "The grade is 2" or just "2" at the end
        end_patterns = [
            (r'(?:^|\n)\s*(\d+)\s*$', 'standalone number at end'),
            (r'grade\s*[:\-]?\s*(\d+)(?:\s|$|\n)', 'grade with number'),
            (r'(?:score|mark)\s*[:\-]?\s*(\d+)(?:\s|$|\n)', 'score/mark with number'),
            # Look for "Grade: X" or similar in the last few lines
            (r'(?:^|\n)[\s]*(?:the\s+)?(?:final\s+)?grade[\s]*[:\-]?\s*(\d+)', 'grade in last lines'),
        ]
        for pattern, pattern_name in end_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                grade = _normalize_grade(match.group(1).strip())
                self.log_fn(f"Extracted grade '{grade}' using pattern: {pattern_name}")
                return grade
        
        # Strategy 3: Look for explicit correctness indicators
        # Use word boundaries to avoid matching inside other words
        correctness_patterns = [
            (r'\b(incorrect|wrong|false|invalid|rejected|no\s+credit)\b', 'Incorrect', 'incorrectness indicator'),
            (r'\b(partial|partially\s+correct|partial\s+credit|half\s+credit|partially\s+solved)\b', 'Partial', 'partial indicator'),
            (r'\b(correct|right|true|valid|accepted|full\s+credit|full\s+marks)\b', 'Correct', 'correctness indicator'),
        ]
        # Check in order: incorrect, partial, correct (to avoid false positives)
        for pattern, grade, pattern_name in correctness_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                self.log_fn(f"Extracted grade '{grade}' using pattern: {pattern_name}")
                return grade
        
        # Strategy 4: Look for grade in the last line of the response
        # Sometimes the grade is just stated at the end without explicit labeling
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        if lines:
            last_line = lines[-1]
            # Check if last line is just a number
            if re.match(r'^\s*(\d+)\s*$', last_line):
                grade = _normalize_grade(last_line.strip())
                self.log_fn(f"Extracted grade '{grade}' from last line number")
                return grade
            # Check if last line is a single word grade
            if re.match(r'^\s*(correct|incorrect|partial|right|wrong|true|false|valid|invalid)\s*$', last_line, re.IGNORECASE):
                grade = _normalize_grade(last_line.strip())
                self.log_fn(f"Extracted grade '{grade}' from last line word")
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
        problem_id = inputs.get("problem_id", "unknown")
        self.log_fn(f"Starting grading for domain: {domain}, problem: {problem_id}")
        
        instruction = self._build_grading_prompt(inputs)

        # Try up to 2 times if we fail to extract a prediction
        max_attempts = 2
        for attempt in range(max_attempts):
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=[],
                )
                self.log_fn(f"LLM call successful (attempt {attempt + 1}), response length: {len(response) if response else 0} chars")
            except Exception as e:
                self.log_fn(f"LLM call failed (attempt {attempt + 1}): {e}")
                if attempt == max_attempts - 1:
                    return "None", []
                continue

            # Extract prediction using multiple strategies
            prediction = self._extract_prediction(msg_history)
            
            if prediction != "None":
                self.log_fn(f"Final prediction: {prediction}")
                return str(prediction), msg_history
            
            # If we failed to extract and this is not the last attempt, log and retry
            if attempt < max_attempts - 1:
                self.log_fn(f"Failed to extract prediction on attempt {attempt + 1}, retrying...")
                self.log_fn(f"Response preview: {response[:300] if response else 'empty'}...")
        
        # All attempts failed
        self.log_fn("All attempts failed to extract prediction")
        return "None", msg_history if 'msg_history' in locals() else []
