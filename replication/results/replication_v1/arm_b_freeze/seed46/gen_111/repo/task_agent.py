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
    if grade is None:
        return 'None'
    
    if not isinstance(grade, str):
        grade = str(grade)
    
    grade = grade.strip().lower()
    
    # Handle empty strings
    if not grade:
        return 'None'
    
    # First, check for numeric grades (0, 1, 2, etc.)
    # These are common in IMO-style grading (0-7 scale)
    try:
        numeric_grade = float(grade)
        # IMO-style: 0 = incorrect, 1-7 = partial to full credit
        # For binary classification: 0 = Incorrect, >=1 = Correct
        # For partial credit: 0 = Incorrect, 1-6 = Partial, 7 = Correct
        if numeric_grade == 0:
            return 'Incorrect'
        elif numeric_grade >= 7:  # Full marks in IMO
            return 'Correct'
        elif numeric_grade >= 1:  # Partial credit (1-6)
            return 'Partial'
    except ValueError:
        pass
    
    # Map common variations to standard formats
    correct_variations = [
        'correct', 'right', 'true', 'yes', 'full', 'full credit', 
        'full marks', 'complete', 'valid', 'accepted', 'pass', 'solved',
        'success', 'accurate', 'perfect', 'excellent', 'good', 'full score'
    ]
    incorrect_variations = [
        'incorrect', 'wrong', 'false', 'no', 'none', 'zero',
        'invalid', 'rejected', 'fail', 'error', 'mistake', 'unsolved',
        'failure', 'bad', 'poor', 'unacceptable', 'no credit'
    ]
    partial_variations = [
        'partial', 'partial credit', 'half', 'incomplete',
        'partially correct', 'partially right', 'some credit',
        'partially solved', 'almost', 'nearly', 'minor errors',
        'mostly correct', 'significant progress', 'partial marks'
    ]
    
    # Check for exact matches first, then substring matches
    if grade in correct_variations:
        return 'Correct'
    if grade in incorrect_variations:
        return 'Incorrect'
    if grade in partial_variations:
        return 'Partial'
    
    # Substring matching for more flexibility
    if any(v in grade for v in correct_variations):
        return 'Correct'
    elif any(v in grade for v in incorrect_variations):
        return 'Incorrect'
    elif any(v in grade for v in partial_variations):
        return 'Partial'
    
    # Handle IMO-style letter grades (A, B, C, D, F, etc.)
    if grade in ['a', 'b']:
        return 'Correct'
    if grade in ['d', 'f']:
        return 'Incorrect'
    if grade == 'c':
        return 'Partial'
    
    # Handle common punctuation that might be attached to grades
    # e.g., "Correct." or "Incorrect!" or "Partial,"
    grade_clean = re.sub(r'[^\w\s]', '', grade).strip()
    if grade_clean in correct_variations:
        return 'Correct'
    if grade_clean in incorrect_variations:
        return 'Incorrect'
    if grade_clean in partial_variations:
        return 'Partial'
    
    # Return original if no normalization applied
    return grade.strip()


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

### Step 5: Verification (Self-Consistency Check)
Before finalizing your grade, verify:
- Re-read the grading guidelines - does your assigned grade match the criteria?
- Check for partial credit scenarios: did the student make significant progress even if the final answer is wrong?
- Consider edge cases: blank answers, answers that are completely off-topic, or answers that use alternative valid methods
- Ensure you haven't been too harsh or too lenient

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
        
        # Handle both "text" and "content" keys in message history
        last_message = msg_history[-1]
        if isinstance(last_message, dict):
            content = last_message.get("text", "") or last_message.get("content", "")
        else:
            content = str(last_message)
        
        if not content:
            return "None"
        
        # Strategy 1: Extract from <json> tags
        extracted = _extract_jsons(content)
        if extracted:
            return self._get_grade_from_json(extracted[-1])
        
        # Strategy 2: Extract from markdown code blocks
        extracted = _extract_json_from_markdown(content)
        if extracted:
            return self._get_grade_from_json(extracted[-1])
        
        # Strategy 3: Look for grade patterns in plain text
        return self._extract_grade_from_text(content)

    def _get_grade_from_json(self, json_obj: dict) -> str:
        """Extract grade from JSON object with field priority."""
        if not isinstance(json_obj, dict):
            return "None"
            
        # Priority order for grade fields
        priority_fields = ["response", "grade", "answer", "result", "score", "evaluation"]
        
        for field in priority_fields:
            if field in json_obj:
                value = json_obj[field]
                if isinstance(value, str):
                    return _normalize_grade(value)
                elif isinstance(value, (int, float)):
                    return str(value)
                elif isinstance(value, list) and value:
                    # Handle case where grade is in a list
                    first_item = value[0]
                    if isinstance(first_item, str):
                        return _normalize_grade(first_item)
                    elif isinstance(first_item, (int, float)):
                        return str(first_item)
        
        # If no recognized field, use the first string or numeric value found
        for key, value in json_obj.items():
            if isinstance(value, str):
                return _normalize_grade(value)
            elif isinstance(value, (int, float)):
                return str(value)
        
        return "None"

    def _extract_grade_from_text(self, text: str) -> str:
        """Extract grade from plain text using pattern matching.
        
        Enhanced with multiple extraction strategies and fallback mechanisms
        to handle various response formats from different LLMs.
        """
        if not text or not isinstance(text, str):
            return "None"
            
        text = text.strip()
        if not text:
            return "None"
        
        # Strategy 1: Look for explicit grade statements with various formats
        patterns = [
            # Standard grade assignment patterns
            r'grade[\s]*[:=][\s]*["\']?([^"\'\n]+)["\']?',
            r'response[\s]*[:=][\s]*["\']?([^"\'\n]+)["\']?',
            r'final grade[\s]*[:=][\s]*["\']?([^"\'\n]+)["\']?',
            r'assign[\s]+["\']?([^"\'\n]+)["\']?',
            # IMO-style numeric grades (0, 1, 2, 3, etc.)
            r'(?:score|points?|mark)[\s]*[:=][\s]*(\d+)',
            r'(?:the\s+)?grade\s+(?:is|of)\s+["\']?(\d+|[^"\'\n]+)["\']?',
            # Evaluation result patterns
            r'evaluation[\s]*[:=][\s]*["\']?([^"\'\n]+)["\']?',
            r'result[\s]*[:=][\s]*["\']?([^"\'\n]+)["\']?',
            # Answer/verdict patterns
            r'(?:the\s+)?(?:answer|verdict|decision)\s+(?:is\s+)?["\']?([^"\'\n]+)["\']?',
            # Additional patterns for common LLM outputs
            r'(?:therefore|thus|hence|so)[,\s]+(?:the\s+)?(?:grade|score)\s+(?:is\s+)?["\']?([^"\'\n]+)["\']?',
            r'(?:i\s+)?(?:assign|give|award)[\s]+["\']?([^"\'\n]+)["\']?',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return _normalize_grade(match.group(1).strip())
        
        # Strategy 2: Look for standalone numeric grades at end of text
        # Common pattern: "The grade is 2" or just "2" at the end
        end_patterns = [
            r'(?:^|\n)\s*(\d+)\s*$',  # Standalone number at end
            r'grade\s*[:\-]?\s*(\d+)(?:\s|$)',  # Grade followed by number
            r'(?:score|mark)\s*[:\-]?\s*(\d+)(?:\s|$)',
            r'(?:^|\n)\s*grade\s*:\s*(\d+)\s*$',  # "Grade: X" at end
        ]
        for pattern in end_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return _normalize_grade(match.group(1).strip())
        
        # Strategy 3: Look for explicit correctness indicators
        correctness_patterns = [
            (r'\b(correct|right|true|valid|accepted|full\s+credit)\b', 'Correct'),
            (r'\b(incorrect|wrong|false|invalid|rejected|no\s+credit)\b', 'Incorrect'),
            (r'\b(partial|partially\s+correct|partial\s+credit|half\s+credit)\b', 'Partial'),
        ]
        for pattern, grade in correctness_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return grade
        
        # Strategy 4: Look for grade in the last line of the response
        # Sometimes LLMs put the grade at the very end without explicit labels
        lines = text.split('\n')
        for line in reversed(lines):
            line = line.strip()
            if line:
                # Check if the last line is just a number (common IMO format)
                if re.match(r'^\s*(\d+)\s*$', line):
                    return _normalize_grade(line)
                # Check if the last line contains a grade word
                words = line.split()
                if words:
                    last_word = words[-1]
                    normalized = _normalize_grade(last_word)
                    if normalized in ['Correct', 'Incorrect', 'Partial']:
                        return normalized
                break  # Only check the last non-empty line
        
        # Strategy 5: Look for grade in the last few lines (broader search)
        # Sometimes the grade is in the conclusion but not the very last line
        for line in reversed(lines[-5:] if len(lines) > 5 else lines):
            line_lower = line.lower().strip()
            if 'grade' in line_lower or 'score' in line_lower or 'correct' in line_lower:
                # Try to extract from this line
                for pattern in patterns[:4]:  # Use first 4 patterns
                    match = re.search(pattern, line, re.IGNORECASE)
                    if match:
                        return _normalize_grade(match.group(1).strip())
        
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
        for field in required_fields:
            if field not in inputs or not inputs[field]:
                self.log_fn(f"Missing required field: {field}")
                return "None", []
        
        instruction = self._build_grading_prompt(inputs)
        self.log_fn(f"Processing problem in domain: {inputs.get('domain', 'unknown')}")

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
        else:
            self.log_fn(f"Extracted prediction: {prediction}")

        return str(prediction), msg_history
