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
    Also handles nested JSON objects and common formatting issues.
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
        
        # Try to parse the JSON with multiple fallback strategies
        parsed = _try_parse_json(inner)
        if parsed is not None:
            results.append(parsed)
    
    return results or None


def _try_parse_json(text: str) -> dict | None:
    """Try to parse JSON with multiple fallback strategies.
    
    Returns the parsed dict or None if all strategies fail.
    """
    # Strategy 1: Direct parsing
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Strategy 2: Remove trailing commas before closing braces/brackets
    try:
        fixed = re.sub(r',(\s*[}\]])', r'\1', text)
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    
    # Strategy 3: Extract the first valid JSON object by brace matching
    brace_start = text.find('{')
    if brace_start == -1:
        return None
    
    brace_count = 0
    for i, char in enumerate(text[brace_start:]):
        if char == '{':
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0:
                try:
                    return json.loads(text[brace_start:brace_start+i+1])
                except json.JSONDecodeError:
                    break
    
    return None


def _extract_json_from_markdown(text: str) -> list[dict] | None:
    """Extract JSON objects from markdown code blocks.

    Fallback for when <json> tags are not used but markdown code blocks are.
    Enhanced to handle nested braces and common formatting issues.
    """
    results = []
    # Match ```json ... ``` or just ``` ... ``` blocks
    pattern = r'```(?:json)?\s*(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    for match in matches:
        match = match.strip()
        parsed = _try_parse_json(match)
        if parsed is not None:
            results.append(parsed)
    return results or None


def _normalize_grade(grade: str) -> str:
    """Normalize grade to a standard format.

    Handles various grade formats and normalizes them.
    Enhanced to handle numeric grades and more variations.
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
    
    # Map common variations to standard formats
    correct_variations = [
        'correct', 'right', 'true', 'yes', 'full', 'full credit', 
        'full marks', 'complete', 'valid', 'accepted', 'pass', 'solved',
        'success', 'accurate', 'perfect', 'excellent', 'good'
    ]
    incorrect_variations = [
        'incorrect', 'wrong', 'false', 'no', 'none', 'zero',
        'invalid', 'rejected', 'fail', 'error', 'mistake', 'unsolved',
        'failure', 'bad', 'unacceptable', 'unsatisfactory'
    ]
    partial_variations = [
        'partial', 'partial credit', 'half', 'incomplete',
        'partially correct', 'partially right', 'some credit',
        'partially solved', 'almost', 'nearly', 'minor errors'
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
    
    # Return original if no normalization applied (preserve case)
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

- **Correct**: The student's answer is completely correct with valid reasoning and correct final answer.
- **Incorrect**: The student's answer is completely wrong, has no valid reasoning, or the final answer is incorrect.
- **Partial**: The student's answer has some valid elements but is incomplete or has minor errors.

For IMO-style problems with numeric grades (0, 1, 2, etc.):
- **0**: No progress or completely wrong
- **1**: Some progress but incomplete or incorrect final answer
- **2+**: Correct or nearly correct solution

## Response Format:

You MUST respond in JSON format wrapped in <json> tags:

<json>
{{
    "reasoning": "Your detailed step-by-step analysis following the Evaluation Framework above",
    "response": "The final grade you assign (e.g., '0', '1', '2', 'Correct', 'Incorrect', 'Partial')"
}}
</json>

Important: The "response" field must contain ONLY the grade value, nothing else. Do not include explanations in the response field."""

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
        
        Enhanced with multiple extraction strategies and confidence scoring
        to handle ambiguous cases more robustly.
        """
        text = text.strip()
        if not text:
            return "None"
        
        # Strategy 1: Look for explicit grade statements with high confidence
        explicit_patterns = [
            # Direct grade assignment patterns
            r'grade[\s]*[:=][\s]*["\']?([^"\'\n]{1,20})["\']?',
            r'response[\s]*[:=][\s]*["\']?([^"\'\n]{1,20})["\']?',
            r'final grade[\s]*[:=][\s]*["\']?([^"\'\n]{1,20})["\']?',
            r'["\']?grade["\']?\s*is\s*["\']?([^"\'\n]{1,20})["\']?',
            r'["\']?score["\']?\s*[:=]\s*["\']?([^"\'\n]{1,20})["\']?',
            r'["\']?result["\']?\s*[:=]\s*["\']?([^"\'\n]{1,20})["\']?',
            r'assign[\s]+["\']?([^"\'\n]{1,20})["\']?',
            # IMO-specific patterns
            r'grade\s+of\s+["\']?([^"\'\n]{1,20})["\']?',
            r'award[\s]+["\']?([^"\'\n]{1,20})["\']?',
        ]
        
        for pattern in explicit_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                candidate = match.group(1).strip()
                # Validate candidate looks like a grade
                if self._looks_like_grade(candidate):
                    return _normalize_grade(candidate)
        
        # Strategy 2: Look for standalone grades at the end of the text
        lines = text.strip().split('\n')
        
        # Check last 10 lines with decreasing priority
        for i, line in enumerate(reversed(lines[-10:])):
            line = line.strip()
            if not line:
                continue
                
            # High-priority: standalone grades on their own line
            standalone_patterns = [
                # Numbers (0-7 for IMO-style, or larger)
                r'^\s*["\']?([0-7])["\']?\s*$',
                r'^\s*["\']?(\d{1,2})["\']?\s*$',
                # Standard grade words
                r'^\s*["\']?(Correct|Incorrect|Partial)["\']?\s*$',
                r'^\s*["\']?(correct|incorrect|partial)["\']?\s*$',
                # IMO-specific grades
                r'^\s*["\']?([0-7])\s*/\s*\d["\']?\s*$',
            ]
            
            for pattern in standalone_patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    candidate = match.group(1).strip()
                    # Higher confidence for lines closer to the end
                    if i < 3 or self._looks_like_grade(candidate):
                        return _normalize_grade(candidate)
        
        # Strategy 3: Look for grade mentions in conclusion/summary sections
        conclusion_markers = ['conclusion', 'summary', 'final', 'therefore', 'thus', 'in conclusion']
        text_lower = text.lower()
        
        for marker in conclusion_markers:
            idx = text_lower.find(marker)
            if idx != -1:
                # Look in the 200 chars after the marker
                context = text[idx:idx+200]
                # Try to find a grade in this context
                grade_match = re.search(r'["\']?([0-7]|Correct|Incorrect|Partial)["\']?', context, re.IGNORECASE)
                if grade_match:
                    return _normalize_grade(grade_match.group(1).strip())
        
        # Strategy 4: Last resort - scan for any grade-like tokens
        all_tokens = re.findall(r'\b([0-7]|Correct|Incorrect|Partial|correct|incorrect|partial)\b', text, re.IGNORECASE)
        if all_tokens:
            # Return the last occurrence (usually the final decision)
            return _normalize_grade(all_tokens[-1].strip())
        
        return "None"
    
    def _looks_like_grade(self, candidate: str) -> bool:
        """Check if a string looks like a valid grade.
        
        Returns True for:
        - Numeric grades 0-7 (IMO style)
        - Standard grade words
        - Short strings that are clearly grades
        """
        if not candidate:
            return False
        
        candidate = candidate.strip().lower()
        
        # Numeric grades (IMO uses 0-7, but be flexible)
        if re.match(r'^[0-9]+$', candidate):
            return True
        
        # Fractions like "7/7"
        if re.match(r'^[0-9]+/[0-9]+$', candidate):
            return True
        
        # Standard grade words
        grade_words = ['correct', 'incorrect', 'partial', 'right', 'wrong', 
                      'true', 'false', 'yes', 'no', 'full', 'none', 'zero',
                      'pass', 'fail', 'solved', 'unsolved']
        if candidate in grade_words:
            return True
        
        # Short strings (likely grades if under 20 chars)
        if len(candidate) <= 20:
            # Check it doesn't look like a sentence
            if '.' not in candidate and len(candidate.split()) <= 3:
                return True
        
        return False

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with enhanced error handling.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
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

        return str(prediction), msg_history
