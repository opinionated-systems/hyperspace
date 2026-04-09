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
    Optimized for IMO-style grading with clear numeric and categorical mappings.
    """
    if not isinstance(grade, str):
        grade = str(grade)
    
    # Clean the grade string
    grade = grade.strip().lower()
    grade_clean = grade.strip('.,;:!?"\'')
    
    # Priority 1: Handle numeric grades (0, 1, 2, etc.) - IMO standard
    # These are the most reliable indicators in mathematical grading
    try:
        numeric_grade = float(grade_clean)
        if numeric_grade == 0:
            return 'Incorrect'
        elif numeric_grade >= 1:
            return 'Correct'
        elif 0 < numeric_grade < 1:
            return 'Partial'
    except ValueError:
        pass
    
    # Priority 2: Check for numeric strings that couldn't be parsed as floats
    # Handle cases like "0/2", "1/2", "2/2" or "0 points", "1 point", etc.
    if grade_clean in ('0', '0.0', '0/2', '0 points', '0 marks', 'zero', '0/1', '0/7'):
        return 'Incorrect'
    if grade_clean in ('1', '1.0', '1/2', '1 point', '1 mark', 'one', '1/1', '1/7'):
        return 'Correct'
    if grade_clean in ('2', '2.0', '2/2', '2 points', '2 marks', 'two', '2/7'):
        return 'Correct'
    if grade_clean in ('0.5', '1/2', 'half', 'half credit'):
        return 'Partial'
    
    # Priority 3: Map common categorical variations to standard formats
    # Use sets for O(1) lookup performance
    correct_set = {
        'correct', 'right', 'true', 'yes', 'full', 'full credit', 
        'full marks', 'complete', 'valid', 'accepted', 'pass',
        'solved', 'solution correct', 'answer correct', 'perfect',
        'excellent', 'good', 'success', 'successful', 'valid solution',
        'accurate', 'proper', 'appropriate', 'satisfactory'
    }
    incorrect_set = {
        'incorrect', 'wrong', 'false', 'no', 'none', 'zero',
        'invalid', 'rejected', 'fail', 'error', 'mistake',
        'unsolved', 'solution incorrect', 'answer incorrect',
        'bad', 'failed', 'failure', 'unsuccessful', 'invalid solution',
        'inaccurate', 'improper', 'inappropriate', 'unsatisfactory'
    }
    partial_set = {
        'partial', 'partial credit', 'half', 'incomplete',
        'partially correct', 'partially right', 'some credit',
        'partially solved', 'partial solution', 'almost',
        'minor errors', 'small mistake', 'mostly correct', 'partial marks',
        'partially accurate', 'partial success'
    }
    
    # Check for exact matches first (most reliable)
    if grade in correct_set:
        return 'Correct'
    if grade in incorrect_set:
        return 'Incorrect'
    if grade in partial_set:
        return 'Partial'
    
    # Priority 4: Substring matching for compound phrases
    # Check if any variation appears as a substring
    for v in correct_set:
        if v in grade:
            return 'Correct'
    for v in incorrect_set:
        if v in grade:
            return 'Incorrect'
    for v in partial_set:
        if v in grade:
            return 'Partial'
    
    # Priority 5: Handle negation patterns
    # "not correct" -> Incorrect, "not wrong" -> Correct
    if 'not correct' in grade or 'not right' in grade or 'not valid' in grade:
        return 'Incorrect'
    if 'not incorrect' in grade or 'not wrong' in grade:
        return 'Correct'
    if 'not partial' in grade:
        return 'Incorrect'
    
    # Return cleaned original if no normalization applied
    return grade_clean


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

## Response Format:

You MUST respond in JSON format wrapped in <json> tags:

<json>
{{
    "reasoning": "Your detailed step-by-step analysis following the Evaluation Framework above",
    "response": "The final grade you assign (e.g., '0', '1', '2', 'Correct', 'Incorrect', 'Partial')"
}}
</json>

Important: The "response" field must contain ONLY the grade value, nothing else."""

    def _extract_prediction(self, msg_history: list[dict]) -> str:
        """Extract prediction from message history with multiple fallback strategies."""
        if not msg_history:
            return "None"
        
        last_message = msg_history[-1].get("text", "")
        
        # Strategy 1: Extract from <json> tags
        extracted = _extract_jsons(last_message)
        if extracted:
            return self._get_grade_from_json(extracted[-1])
        
        # Strategy 2: Extract from markdown code blocks
        extracted = _extract_json_from_markdown(last_message)
        if extracted:
            return self._get_grade_from_json(extracted[-1])
        
        # Strategy 3: Look for grade patterns in plain text
        return self._extract_grade_from_text(last_message)

    def _get_grade_from_json(self, json_obj: dict) -> str:
        """Extract grade from JSON object with field priority."""
        # Priority order for grade fields
        priority_fields = ["response", "grade", "answer", "result", "score", "evaluation"]
        
        for field in priority_fields:
            if field in json_obj:
                value = json_obj[field]
                if isinstance(value, str):
                    return _normalize_grade(value)
                elif isinstance(value, (int, float)):
                    return str(value)
        
        # If no recognized field, use the first string value found
        for key, value in json_obj.items():
            if isinstance(value, str):
                return _normalize_grade(value)
            elif isinstance(value, (int, float)):
                return str(value)
        
        return "None"

    def _extract_grade_from_text(self, text: str) -> str:
        """Extract grade from plain text using pattern matching.
        
        Optimized with prioritized patterns for robust extraction from
        various response formats including IMO-style numeric grades.
        """
        text_lower = text.lower()
        
        # Priority 1: Look for explicit grade assignments with clear context
        # These patterns indicate a definitive grade decision
        priority_patterns = [
            # Direct grade field assignments (most reliable)
            r'["\']?grade["\']?\s*[:=]\s*["\']?([^"\'\n,]{1,20})["\']?',
            r'["\']?response["\']?\s*[:=]\s*["\']?([^"\'\n,]{1,20})["\']?',
            r'final\s+grade\s*[:=]\s*["\']?([^"\'\n,]{1,20})["\']?',
            # Action-oriented grade statements
            r'(?:assign|give|award)\s*["\']?([^"\'\n,]{1,20})["\']?\s*(?:grade|points?|marks?)?',
            # Student receives statements
            r'(?:student\s+)?(?:receives?|gets?|earns?)\s*["\']?(\d+(?:\.\d+)?)["\']?\s*(?:points?|marks?)?',
        ]
        
        for pattern in priority_patterns:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                grade = match.group(1).strip()
                # Validate the extracted grade
                if grade and len(grade) <= 20:
                    normalized = _normalize_grade(grade)
                    if normalized in ('Correct', 'Incorrect', 'Partial', '0', '1', '2'):
                        return normalized
        
        # Priority 2: Look for standalone numeric grades (0, 1, 2) with word boundaries
        # These are common in IMO grading and are reliable when isolated
        standalone_numeric = re.findall(r'(?:^|\s|\()["\']?([0-2](?:\.0|\.5)?)["\']?(?:$|\s|\)|[.,])', text)
        if standalone_numeric:
            # Return the last occurrence (often the final decision)
            return _normalize_grade(standalone_numeric[-1])
        
        # Priority 3: Look for explicit grade keywords with clear context
        # Check for grade statements like "grade is X" or "the grade: X"
        grade_is_patterns = [
            r'(?:grade|score|mark|result)\s+(?:is|of|[:=])\s*["\']?([^"\'\n,.]{1,15})["\']?',
            r'(?:the\s+)?(?:answer|solution|result)\s+(?:is|should\s+be)\s*["\']?([^"\'\n,.]{1,15})["\']?',
        ]
        
        for pattern in grade_is_patterns:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                grade = match.group(1).strip()
                if grade and len(grade) <= 15:
                    normalized = _normalize_grade(grade)
                    if normalized in ('Correct', 'Incorrect', 'Partial', '0', '1', '2'):
                        return normalized
        
        # Priority 4: Look for standalone grade keywords at word boundaries
        # These are less reliable but useful as fallbacks
        keyword_patterns = [
            (r'\b(correct|right|valid|accepted|full\s+credit)\b', 'Correct'),
            (r'\b(incorrect|wrong|invalid|rejected|failed?)\b', 'Incorrect'),
            (r'\b(partial|partially\s+correct|partial\s+credit|incomplete)\b', 'Partial'),
        ]
        
        for pattern, default_grade in keyword_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return default_grade
        
        # Priority 5: Final fallback - check for numeric grades anywhere
        # This is the least reliable but catches edge cases
        numeric_matches = re.findall(r'\b([0-2])\b', text)
        if numeric_matches:
            return _normalize_grade(numeric_matches[-1])
        
        return "None"

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with enhanced error handling.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Validate required inputs
        required_fields = ["problem", "solution", "student_answer"]
        missing_fields = [f for f in required_fields if not inputs.get(f)]
        if missing_fields:
            self.log_fn(f"Missing required fields: {missing_fields}")
            return "None", []

        instruction = self._build_grading_prompt(inputs)

        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
        except Exception as e:
            self.log_fn(f"LLM call failed: {e}")
            return "None", []

        # Extract prediction using multiple strategies
        prediction = self._extract_prediction(msg_history)
        
        if prediction == "None":
            self.log_fn(f"Failed to extract prediction from response: {response[:200] if response else 'empty'}")
            # Attempt to extract any grade-like pattern as last resort
            prediction = self._last_resort_extraction(response)

        return str(prediction), msg_history

    def _last_resort_extraction(self, text: str) -> str:
        """Last resort extraction when all other methods fail.
        
        Uses aggressive pattern matching to find any grade-like value.
        """
        if not text:
            return "None"
        
        text_lower = text.lower()
        
        # Look for any occurrence of grade keywords
        if 'incorrect' in text_lower or 'wrong' in text_lower:
            return 'Incorrect'
        if 'correct' in text_lower or 'right' in text_lower:
            return 'Correct'
        if 'partial' in text_lower:
            return 'Partial'
        
        # Look for any digits that could be grades
        digits = re.findall(r'\d+', text)
        if digits:
            for d in digits:
                if d in ['0']:
                    return 'Incorrect'
                elif d in ['1', '2', '3', '4', '5']:
                    return 'Correct'
        
        return "None"
