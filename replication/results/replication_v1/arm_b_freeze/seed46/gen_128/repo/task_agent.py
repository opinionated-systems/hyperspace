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
    Enhanced to handle IMO-style numeric grades (0, 1, 2, 7) and more variations.
    
    Improvements:
    - Better handling of edge cases and punctuation
    - More robust numeric parsing
    - Handles empty/None inputs gracefully
    """
    # Handle None or empty inputs
    if grade is None:
        return "None"
    
    if not isinstance(grade, str):
        grade = str(grade)
    
    grade = grade.strip()
    
    # Handle empty string after stripping
    if not grade:
        return "None"
    
    # Store original for potential return
    original = grade
    grade = grade.lower()
    
    # First, check for IMO-style numeric grades (0, 1, 2, 7)
    # These are standard in IMO grading
    try:
        numeric_grade = float(grade)
        # IMO-style grading: 0 = no progress, 1 = minor progress, 2 = substantial progress, 7 = full solution
        if numeric_grade == 0:
            return 'Incorrect'
        elif numeric_grade >= 7:
            return 'Correct'
        elif numeric_grade >= 1:
            return 'Partial'
        elif 0 < numeric_grade < 1:
            return 'Partial'
    except ValueError:
        pass
    
    # Map common variations to standard formats
    correct_variations = [
        'correct', 'right', 'true', 'yes', 'full', 'full credit', 
        'full marks', 'complete', 'valid', 'accepted', 'pass',
        'solved', 'solution correct', 'answer correct', 'perfect',
        'excellent', 'full score', '7', '7.0', '7/7', 'seven'
    ]
    incorrect_variations = [
        'incorrect', 'wrong', 'false', 'no', 'none', 'zero',
        'invalid', 'rejected', 'fail', 'error', 'mistake',
        'unsolved', 'solution incorrect', 'answer incorrect',
        '0', '0.0', '0/7', 'no progress', 'not correct', 'not right'
    ]
    partial_variations = [
        'partial', 'partial credit', 'half', 'incomplete',
        'partially correct', 'partially right', 'some credit',
        'partially solved', 'partial solution', 'minor progress',
        'substantial progress', '1', '1.0', '2', '2.0', '3', '4', '5', '6',
        '1/7', '2/7', '3/7', '4/7', '5/7', '6/7'
    ]
    
    # Check for exact matches first, then substring matches
    if grade in correct_variations:
        return 'Correct'
    if grade in incorrect_variations:
        return 'Incorrect'
    if grade in partial_variations:
        return 'Partial'
    
    # Substring matching for more flexibility (but be careful not to over-match)
    # Check for clear indicators at word boundaries
    grade_words = set(grade.split())
    
    if any(v in grade_words for v in correct_variations):
        return 'Correct'
    if any(v in grade_words for v in incorrect_variations):
        return 'Incorrect'
    if any(v in grade_words for v in partial_variations):
        return 'Partial'
    
    # Broader substring matching for compound phrases
    if any(v in grade for v in ['correct', 'right', 'true', 'full credit', 'full marks', 'solved']):
        return 'Correct'
    if any(v in grade for v in ['incorrect', 'wrong', 'false', 'invalid', 'rejected', 'fail', 'error']):
        return 'Incorrect'
    if any(v in grade for v in ['partial', 'incomplete', 'half', 'some credit', 'progress']):
        return 'Partial'
    
    # Handle edge cases with punctuation - more aggressive cleaning
    grade_clean = grade.strip('.,;:!?"\'()[]{}')
    # Remove trailing punctuation that might be part of sentence structure
    grade_clean = re.sub(r'[.,;:!?]+$', '', grade_clean).strip()
    
    try:
        numeric_clean = float(grade_clean)
        if numeric_clean == 0:
            return 'Incorrect'
        elif numeric_clean >= 7:
            return 'Correct'
        elif numeric_clean >= 1:
            return 'Partial'
        elif 0 < numeric_clean < 1:
            return 'Partial'
    except ValueError:
        pass
    
    # Return original if no normalization applied (but capitalize first letter)
    if original:
        return original[0].upper() + original[1:] if len(original) > 1 else original.upper()
    return "None"


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

## IMO-Style Grading Reference:

When assigning grades, use these standard IMO-style interpretations:
- **0**: No meaningful progress toward the solution
- **1**: Minor progress (some relevant ideas but far from complete)
- **2**: Substantial progress (significant correct work but incomplete or with major gaps)
- **7**: Complete and correct solution (all steps valid and properly justified)

For binary grading (Correct/Incorrect/Partial):
- **Correct**: Full solution with correct final answer and valid reasoning
- **Partial**: Partial credit for significant progress even if final answer is wrong
- **Incorrect**: No meaningful progress or completely wrong approach

## Response Format:

You MUST respond in JSON format wrapped in <json> tags:

<json>
{{
    "reasoning": "Your detailed step-by-step analysis following the Evaluation Framework above",
    "response": "The final grade you assign (e.g., '0', '1', '2', '7', 'Correct', 'Incorrect', 'Partial')"
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
        
        Enhanced with additional patterns for robust extraction from
        various response formats including IMO-style numeric grades (0-7).
        
        Improvements:
        - Better handling of multi-line responses
        - Priority-based pattern matching (most specific first)
        - Enhanced logging for debugging extraction failures
        """
        if not text or not isinstance(text, str):
            return "None"
        
        # Clean up the text: normalize whitespace but preserve structure
        text = text.strip()
        
        # Priority 1: Look for explicit grade statements with specific patterns
        # These are the most reliable patterns - explicit grade assignments
        priority_patterns = [
            # Explicit "Grade: X" or "Grade = X" patterns
            r'(?:^|\n|\s)grade[\s]*[:=][\s]*["\']?([^"\'\n,;]+)["\']?(?:\s|$|\n|\.|<)',
            r'(?:^|\n|\s)final[\s]+grade[\s]*[:=][\s]*["\']?([^"\'\n,;]+)["\']?(?:\s|$|\n|\.|<)',
            r'(?:^|\n|\s)response[\s]*[:=][\s]*["\']?([^"\'\n,;]+)["\']?(?:\s|$|\n|\.|<)',
            # "I assign/give/award X" patterns
            r'I\s+(?:assign|give|award)[\s:]+["\']?([^"\'\n,;]{1,20})["\']?(?:\s|$|\n|\.|<)',
            r'(?:the\s+)?student\s+(?:receives?|gets?|earns?)[\s:]+["\']?([^"\'\n,;]{1,20})["\']?(?:\s|$|\n|\.|<)',
        ]
        
        for pattern in priority_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                extracted = match.group(1).strip()
                normalized = _normalize_grade(extracted)
                if normalized and normalized != "None":
                    return normalized
        
        # Priority 2: Look for grade in specific contexts
        context_patterns = [
            # "grade/score/mark is/of X"
            r'(?:grade|score|mark|result)[\s]+(?:is|of)[\s]+["\']?([^"\'\n,.]{1,20})["\']?',
            # "a grade of X"
            r'grade\s+of\s+["\']?([^"\'\n,.]{1,20})["\']?',
            # "assign X" at start of sentence
            r'(?:^|\n)[\s]*assign[\s]+["\']?([^"\'\n,.]{1,20})["\']?',
        ]
        
        for pattern in context_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                extracted = match.group(1).strip()
                normalized = _normalize_grade(extracted)
                if normalized and normalized != "None":
                    return normalized
        
        # Priority 3: Look for standalone grade keywords
        keyword_patterns = [
            r'\b(Correct|Incorrect|Partial|Right|Wrong|Full|None)\b',
            r'\b([0-7](?:\.0|\.5)?)\b',
        ]
        
        for pattern in keyword_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                # Return the last match (usually the final conclusion)
                extracted = matches[-1] if isinstance(matches[-1], str) else matches[-1][0]
                normalized = _normalize_grade(extracted.strip())
                if normalized and normalized != "None":
                    return normalized
        
        # Priority 4: Fallback - look for any numeric grades (0-7) as standalone tokens
        numeric_matches = re.findall(r'(?:^|\s|\n)([0-7])(?:\s|$|\n|\.|<)', text)
        if numeric_matches:
            # Return the last numeric match (often the final grade)
            return _normalize_grade(numeric_matches[-1])
        
        # Log the failure for debugging
        self.log_fn(f"Could not extract grade from text (first 300 chars): {text[:300]}")
        
        return "None"

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
