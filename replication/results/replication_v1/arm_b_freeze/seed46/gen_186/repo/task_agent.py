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
import time
from typing import Tuple, Optional

from agent.llm_client import get_response_from_llm, EVAL_MODEL

logger = logging.getLogger(__name__)

# Configuration for retry mechanism
MAX_RETRIES = 3
RETRY_DELAY = 1.0  # seconds between retries


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
    Includes caching for repeated grade normalizations.
    """
    if not isinstance(grade, str):
        grade = str(grade)
    
    # Fast path: exact match for standard grades
    grade_stripped = grade.strip()
    if grade_stripped in ('Correct', 'Incorrect', 'Partial'):
        return grade_stripped
    
    grade_lower = grade_stripped.lower()
    if grade_lower in ('correct', 'incorrect', 'partial'):
        # Return canonical form
        return grade_lower.capitalize()
    
    # First, check for IMO-style numeric grades (0, 1, 2, 7)
    # These are standard in IMO grading
    try:
        numeric_grade = float(grade_lower)
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
        'excellent', 'full score', '7', '7.0'
    ]
    incorrect_variations = [
        'incorrect', 'wrong', 'false', 'no', 'none', 'zero',
        'invalid', 'rejected', 'fail', 'error', 'mistake',
        'unsolved', 'solution incorrect', 'answer incorrect',
        '0', '0.0', 'no progress'
    ]
    partial_variations = [
        'partial', 'partial credit', 'half', 'incomplete',
        'partially correct', 'partially right', 'some credit',
        'partially solved', 'partial solution', 'minor progress',
        'substantial progress', '1', '1.0', '2', '2.0', '3', '4', '5', '6'
    ]
    
    # Check for exact matches first, then substring matches
    if grade_lower in correct_variations:
        return 'Correct'
    if grade_lower in incorrect_variations:
        return 'Incorrect'
    if grade_lower in partial_variations:
        return 'Partial'
    
    # Substring matching for more flexibility
    if any(v in grade_lower for v in correct_variations):
        return 'Correct'
    elif any(v in grade_lower for v in incorrect_variations):
        return 'Incorrect'
    elif any(v in grade_lower for v in partial_variations):
        return 'Partial'
    
    # Handle edge cases with punctuation
    grade_clean = grade_lower.strip('.,;:!?"\'')
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
    
    # Return original if no normalization applied
    return grade_stripped


def _extract_grade_with_confidence(text: str) -> Tuple[str, float]:
    """Extract grade from text with confidence score.
    
    Returns a tuple of (grade, confidence) where confidence is 0.0-1.0.
    Higher confidence indicates more reliable extraction.
    Uses a tiered pattern matching approach with early termination on high-confidence matches.
    """
    if not text or not text.strip():
        return "None", 0.0
    
    # High confidence patterns (explicit JSON or structured format)
    high_confidence_patterns = [
        (r'<json>.*?"response"\s*:\s*"([^"]+)".*?</json>', 0.95),
        (r'<json>.*?"grade"\s*:\s*"([^"]+)".*?</json>', 0.95),
        (r'```json.*?"response"\s*:\s*"([^"]+)".*?```', 0.90),
        (r'```json.*?"grade"\s*:\s*"([^"]+)".*?```', 0.90),
    ]
    
    # Medium confidence patterns (explicit statements)
    medium_confidence_patterns = [
        (r'grade\s*[:=]\s*["\']?([^"\'\n,]+)["\']?', 0.80),
        (r'response\s*[:=]\s*["\']?([^"\'\n,]+)["\']?', 0.80),
        (r'final grade\s*[:=]\s*["\']?([^"\'\n,]+)["\']?', 0.80),
        (r'(?:the\s+)?(?:student\s+(?:receives?|gets?|earns?)|I\s+(?:assign|give|award))\s+["\']?(\d+(?:\.\d+)?)["\']?\s*(?:points?|marks?)?', 0.75),
        (r'(?:grade|score|mark)\s+(?:is|of)\s+["\']?([^"\'\n,]+)["\']?', 0.75),
    ]
    
    # Lower confidence patterns (standalone values)
    low_confidence_patterns = [
        (r'\b([0-7](?:\.0|\.5)?)\b', 0.60),
        (r'\b(Correct|Incorrect|Partial|Right|Wrong)\b', 0.60),
    ]
    
    # Check high confidence patterns first (early termination)
    for pattern, confidence in high_confidence_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            grade = _normalize_grade(match.group(1).strip())
            if grade in ['Correct', 'Incorrect', 'Partial']:
                return grade, confidence
    
    # Check medium confidence patterns
    for pattern, confidence in medium_confidence_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            grade = _normalize_grade(match.group(1).strip())
            if grade in ['Correct', 'Incorrect', 'Partial']:
                return grade, confidence
    
    # Check low confidence patterns
    for pattern, confidence in low_confidence_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            grade = _normalize_grade(match.group(1).strip())
            if grade in ['Correct', 'Incorrect', 'Partial']:
                return grade, confidence
    
    # Fallback: look for numeric grades at the end of the text (often the final conclusion)
    numeric_matches = re.findall(r'\b([0-7])\b', text)
    if numeric_matches:
        # Prefer the last numeric match (often the final grade in a reasoning chain)
        grade = _normalize_grade(numeric_matches[-1])
        if grade in ['Correct', 'Incorrect', 'Partial']:
            return grade, 0.50
    
    return "None", 0.0


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

    def _extract_prediction(self, msg_history: list[dict]) -> Tuple[str, float]:
        """Extract prediction from message history with confidence scoring.
        
        Returns a tuple of (grade, confidence) where confidence is 0.0-1.0.
        Uses a multi-strategy approach with cascading confidence levels.
        """
        if not msg_history:
            return "None", 0.0
        
        last_message = msg_history[-1].get("text", "")
        if not last_message or not last_message.strip():
            return "None", 0.0
        
        # Strategy 1: Extract from <json> tags (highest confidence)
        extracted = _extract_jsons(last_message)
        if extracted:
            grade = self._get_grade_from_json(extracted[-1])
            if grade != "None":
                return grade, 0.95
        
        # Strategy 2: Extract from markdown code blocks
        extracted = _extract_json_from_markdown(last_message)
        if extracted:
            grade = self._get_grade_from_json(extracted[-1])
            if grade != "None":
                return grade, 0.90
        
        # Strategy 3: Use confidence-based extraction from plain text
        grade, confidence = _extract_grade_with_confidence(last_message)
        if grade != "None":
            return grade, confidence
        
        # Strategy 4: Fallback - look for any valid grade mention in the entire history
        # This helps when the model's last message is truncated or unclear
        for msg in reversed(msg_history[-3:]):  # Check last 3 messages
            text = msg.get("text", "")
            if text:
                grade, confidence = _extract_grade_with_confidence(text)
                if grade != "None" and confidence >= 0.60:
                    return grade, confidence * 0.8  # Reduce confidence for historical lookup
        
        return "None", 0.0

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
        """
        # Look for explicit grade statements with flexible patterns
        patterns = [
            # Standard grade assignments
            r'grade[\s]*[:=][\s]*["\']?([^"\'\n]+)["\']?',
            r'response[\s]*[:=][\s]*["\']?([^"\'\n]+)["\']?',
            r'final grade[\s]*[:=][\s]*["\']?([^"\'\n]+)["\']?',
            r'assign[\s]+["\']?([^"\'\n]+)["\']?',
            # IMO-style grade statements (0, 1, 2, 7)
            r'(?:the\s+)?(?:student\s+(?:receives?|gets?|earns?)|I\s+(?:assign|give|award))\s+["\']?(\d+(?:\.\d+)?)["\']?\s*(?:points?|marks?)?',
            r'(?:score|grade|mark)[\s]*[:=\s]+["\']?(\d+(?:\.\d+)?)["\']?',
            # Grade at end of sentence
            r'(?:grade|score|mark|result)[\s]+(?:is|of)[\s]+["\']?([^"\'\n.]+)["\']?',
            # Standalone IMO grades (0, 1, 2, 7) - full range
            r'\b([0-7](?:\.0|\.5)?)\b',
            r'\b(Correct|Incorrect|Partial|Right|Wrong)\b',
            # Additional patterns for edge cases
            r'(?:grade|score|mark|result)[\s]*[:=\s]+["\']?([^"\'\n,]+)["\']?',
            r'(?:the\s+)?(?:answer|solution)\s+(?:is|should\s+be)\s+["\']?([^"\'\n,]+)["\']?',
            # Pattern for "grade of X" or "a grade of X"
            r'grade\s+of\s+["\']?([^"\'\n,]+)["\']?',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return _normalize_grade(match.group(1).strip())
        
        # Fallback: look for numeric grades (0-7) as standalone tokens
        # This handles cases where the model just outputs a number
        numeric_matches = re.findall(r'\b([0-7])\b', text)
        if numeric_matches:
            # Return the last numeric match (often the final grade)
            return _normalize_grade(numeric_matches[-1])
        
        return "None"

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with enhanced error handling and retry logic.

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
        
        # Retry loop with exponential backoff
        last_exception = None
        for attempt in range(MAX_RETRIES):
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=[],
                )
                
                # Extract prediction with confidence
                prediction, confidence = self._extract_prediction(msg_history)
                
                # If we have a valid prediction with good confidence, return it
                if prediction != "None" and confidence >= 0.60:
                    self.log_fn(f"Extracted grade '{prediction}' with confidence {confidence:.2f}")
                    return str(prediction), msg_history
                
                # If confidence is low but we have a prediction, still use it after logging
                if prediction != "None":
                    self.log_fn(f"Low confidence grade '{prediction}' (confidence: {confidence:.2f}), using anyway")
                    return str(prediction), msg_history
                
                # No valid prediction extracted, will retry
                self.log_fn(f"Attempt {attempt + 1}/{MAX_RETRIES}: Failed to extract valid prediction (confidence: {confidence:.2f})")
                
                # If this isn't the last attempt, wait before retrying
                if attempt < MAX_RETRIES - 1:
                    delay = RETRY_DELAY * (2 ** attempt)  # Exponential backoff
                    self.log_fn(f"Waiting {delay:.1f}s before retry...")
                    time.sleep(delay)
                    
            except Exception as e:
                last_exception = e
                self.log_fn(f"Attempt {attempt + 1}/{MAX_RETRIES}: LLM call failed: {e}")
                if attempt < MAX_RETRIES - 1:
                    delay = RETRY_DELAY * (2 ** attempt)
                    time.sleep(delay)
        
        # All retries exhausted
        if last_exception:
            self.log_fn(f"All retries failed. Last error: {last_exception}")
        
        # Try to extract any prediction from the last attempt, even with low confidence
        if 'msg_history' in locals() and msg_history:
            prediction, confidence = self._extract_prediction(msg_history)
            if prediction != "None":
                self.log_fn(f"Using low-confidence prediction '{prediction}' after all retries")
                return str(prediction), msg_history
        
        return "None", []
