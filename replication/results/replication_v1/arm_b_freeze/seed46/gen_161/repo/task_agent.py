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
    Enhanced to handle numeric grades and more variations.
    """
    if not isinstance(grade, str):
        grade = str(grade)
    
    # Clean up the grade string
    original_grade = grade.strip()
    grade = original_grade.lower()
    
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
        'full marks', 'complete', 'valid', 'accepted', 'pass',
        'solved', 'solution correct', 'answer correct', 'perfect',
        'excellent', 'good', 'success', 'successful', 'valid solution'
    ]
    incorrect_variations = [
        'incorrect', 'wrong', 'false', 'no', 'none', 'zero',
        'invalid', 'rejected', 'fail', 'error', 'mistake',
        'unsolved', 'solution incorrect', 'answer incorrect',
        'bad', 'failed', 'failure', 'unsuccessful', 'not correct',
        'not valid', 'not accepted'
    ]
    partial_variations = [
        'partial', 'partial credit', 'half', 'incomplete',
        'partially correct', 'partially right', 'some credit',
        'partially solved', 'partial solution', 'almost',
        'minor errors', 'small mistake', 'mostly correct',
        'half marks', 'half credit', 'partial marks'
    ]
    
    # Check for exact matches first, then substring matches
    if grade in correct_variations:
        return 'Correct'
    if grade in incorrect_variations:
        return 'Incorrect'
    if grade in partial_variations:
        return 'Partial'
    
    # Substring matching for more flexibility - but check for negations first
    # Check for "not" prefix which often negates the grade
    if 'not ' in grade and any(v in grade for v in ['correct', 'right', 'valid', 'accepted']):
        return 'Incorrect'
    
    # Check for partial first (more specific)
    if any(v in grade for v in partial_variations):
        return 'Partial'
    elif any(v in grade for v in correct_variations):
        return 'Correct'
    elif any(v in grade for v in incorrect_variations):
        return 'Incorrect'
    
    # Handle edge cases with punctuation and common suffixes
    grade_clean = grade.strip('.,;:!?"\'')
    # Remove common suffixes like "points", "marks", "out of X"
    grade_clean = re.sub(r'\s*(?:points?|marks?|out\s+of\s+\d+).*$', '', grade_clean).strip()
    
    # Handle numeric grades with proper partial credit recognition
    if grade_clean in ['0', '0.0']:
        return 'Incorrect'
    if grade_clean == '0.5':
        return 'Partial'  # 0.5 is partial credit, not incorrect
    if grade_clean in ['1', '1.0', '1.5', '2', '2.0']:
        return 'Correct'
    
    # Return original if no normalization applied
    return original_grade


def _calculate_extraction_confidence(text: str, extracted_grade: str) -> float:
    """Calculate confidence score for grade extraction.
    
    Returns a value between 0.0 and 1.0 indicating how confident we are
    in the extracted grade. Higher confidence for explicit grade statements.
    """
    if extracted_grade == "None":
        return 0.0
    
    text_lower = text.lower()
    confidence = 0.5  # Base confidence
    
    # Boost confidence for explicit grade assignment patterns
    explicit_patterns = [
        r'(?:the\s+)?(?:student\s+(?:receives?|gets?|earns?)|I\s+(?:assign|give|award))\s+["\']?\d+',
        r'["\']?(?:grade|response|score)["\']?\s*[:=]\s*["\']?(?:Correct|Incorrect|Partial|\d+)',
        r'\b(?:grade|score|mark|result)\s+(?:is|of)\s+["\']?(?:Correct|Incorrect|Partial|\d+)',
    ]
    
    for pattern in explicit_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            confidence = min(1.0, confidence + 0.3)
            break
    
    # Boost for JSON format
    if '<json>' in text or '```json' in text:
        confidence = min(1.0, confidence + 0.2)
    
    # Reduce confidence if multiple conflicting grades mentioned
    grade_keywords = ['correct', 'incorrect', 'partial', 'right', 'wrong']
    found_grades = sum(1 for kw in grade_keywords if kw in text_lower)
    if found_grades > 2:
        confidence = max(0.0, confidence - 0.2)
    
    return confidence


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

    def _extract_prediction(self, msg_history: list[dict]) -> tuple[str, float]:
        """Extract prediction from message history with multiple fallback strategies.
        
        Returns:
            Tuple of (prediction, confidence_score)
        """
        if not msg_history:
            return "None", 0.0
        
        last_message = msg_history[-1].get("text", "")
        
        # Strategy 1: Extract from <json> tags (highest confidence)
        extracted = _extract_jsons(last_message)
        if extracted:
            grade = self._get_grade_from_json(extracted[-1])
            confidence = _calculate_extraction_confidence(last_message, grade)
            return grade, confidence + 0.1  # Boost for JSON format
        
        # Strategy 2: Extract from markdown code blocks
        extracted = _extract_json_from_markdown(last_message)
        if extracted:
            grade = self._get_grade_from_json(extracted[-1])
            confidence = _calculate_extraction_confidence(last_message, grade)
            return grade, confidence + 0.05  # Small boost for markdown JSON
        
        # Strategy 3: Look for grade patterns in plain text
        grade = self._extract_grade_from_text(last_message)
        confidence = _calculate_extraction_confidence(last_message, grade)
        return grade, confidence

    def _get_grade_from_json(self, json_obj: dict) -> str:
        """Extract grade from JSON object with field priority.
        
        Enhanced to handle nested structures and more field variations.
        """
        # Priority order for grade fields - most specific first
        priority_fields = [
            "response", "grade", "answer", "result", "score", 
            "evaluation", "verdict", "decision", "assessment",
            "classification", "outcome", "status"
        ]
        
        for field in priority_fields:
            if field in json_obj:
                value = json_obj[field]
                # Handle string values
                if isinstance(value, str):
                    normalized = _normalize_grade(value)
                    if normalized in ['Correct', 'Incorrect', 'Partial']:
                        return normalized
                    return normalized
                # Handle numeric values
                elif isinstance(value, (int, float)):
                    if value == 0:
                        return 'Incorrect'
                    elif value >= 1:
                        return 'Correct'
                    elif 0 < value < 1:
                        return 'Partial'
                    return str(value)
                # Handle boolean values
                elif isinstance(value, bool):
                    return 'Correct' if value else 'Incorrect'
        
        # If no recognized field, look for any field that might contain a grade
        for key, value in json_obj.items():
            # Skip metadata fields
            if key in ['reasoning', 'explanation', 'analysis', 'notes', 'comment']:
                continue
            if isinstance(value, str):
                normalized = _normalize_grade(value)
                if normalized in ['Correct', 'Incorrect', 'Partial']:
                    return normalized
            elif isinstance(value, (int, float)):
                if value == 0:
                    return 'Incorrect'
                elif value >= 1:
                    return 'Correct'
                elif 0 < value < 1:
                    return 'Partial'
                return str(value)
            elif isinstance(value, bool):
                return 'Correct' if value else 'Incorrect'
        
        return "None"

    def _extract_grade_from_text(self, text: str) -> str:
        """Extract grade from plain text using pattern matching.
        
        Enhanced with additional patterns for robust extraction from
        various response formats including IMO-style numeric grades.
        """
        # Look for explicit grade statements with flexible patterns
        # Priority: more specific patterns first, then general ones
        patterns = [
            # IMO-style grade statements with explicit keywords
            r'(?:the\s+)?(?:student\s+(?:receives?|gets?|earns?)|I\s+(?:assign|give|award))\s+["\']?(\d+(?:\.\d+)?)["\']?\s*(?:points?|marks?)?',
            # Grade field in JSON-like or assignment format
            r'["\']?grade["\']?\s*[:=]\s*["\']?([^"\'\n,]+)["\']?',
            r'["\']?response["\']?\s*[:=]\s*["\']?([^"\'\n,]+)["\']?',
            r'["\']?final[_\s]grade["\']?\s*[:=]\s*["\']?([^"\'\n,]+)["\']?',
            # Score/mark with numeric value
            r'(?:score|mark|grade|result)\s*[:=\s]+["\']?(\d+(?:\.\d+)?)["\']?',
            # Grade at end of sentence with "is" or "of"
            r'(?:the\s+)?(?:grade|score|mark|result)\s+(?:is|of)\s+["\']?([^"\'\n.]+)["\']?',
            # Standalone numeric grades (0, 0.5, 1, 1.5, 2) - word boundaries
            r'(?:^|\s)([0-2](?:\.0|\.5)?)(?:\s|$|[^\d])',
            # Explicit grade keywords
            r'\b(Correct|Incorrect|Partial|Right|Wrong|True|False)\b',
            # Answer/solution is pattern
            r'(?:the\s+)?(?:answer|solution|grade)\s+(?:is|should\s+be)\s+["\']?([^"\'\n,]+)["\']?',
            # Assign pattern
            r'\bassign\s+["\']?([^"\'\n,]+)["\']?',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                grade_value = match.group(1).strip()
                # Validate the extracted value looks like a grade
                normalized = _normalize_grade(grade_value)
                if normalized != grade_value:  # Normalization changed something
                    return normalized
                # Check if it's a valid numeric grade
                if re.match(r'^\d+(?:\.\d+)?$', grade_value):
                    return _normalize_grade(grade_value)
                # Check if it's a valid text grade
                if any(keyword in grade_value.lower() for keyword in ['correct', 'incorrect', 'partial', 'right', 'wrong', 'true', 'false']):
                    return _normalize_grade(grade_value)
        
        # Fallback: look for numeric grades (0, 1, 2) as standalone tokens
        # This handles cases where the model just outputs a number
        numeric_matches = re.findall(r'(?:^|\s)([0-2])(?:\s|$|[^\d])', text)
        if numeric_matches:
            # Return the last numeric match (often the final grade)
            return _normalize_grade(numeric_matches[-1])
        
        # Additional fallback: look for grade keywords anywhere in text
        # But be more careful about context
        text_lower = text.lower()
        
        # Check for explicit negations first
        negation_patterns = [
            'not correct', 'not right', 'not valid', 'not accepted',
            'is incorrect', 'is wrong', 'is not', 'not a correct'
        ]
        has_negation = any(pat in text_lower for pat in negation_patterns)
        
        if has_negation:
            return 'Incorrect'
        
        # Check for positive indicators
        positive_patterns = ['correct', 'right', 'valid', 'accepted', 'full credit', 'full marks']
        negative_patterns = ['incorrect', 'wrong', 'invalid', 'rejected', 'no credit', 'zero']
        partial_patterns = ['partial', 'partial credit', 'half credit', 'incomplete', 'partially correct']
        
        has_positive = any(pat in text_lower for pat in positive_patterns)
        has_negative = any(pat in text_lower for pat in negative_patterns)
        has_partial = any(pat in text_lower for pat in partial_patterns)
        
        if has_partial:
            return 'Partial'
        if has_positive and not has_negative:
            return 'Correct'
        if has_negative:
            return 'Incorrect'
        
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

        # Extract prediction using multiple strategies with confidence scoring
        prediction, confidence = self._extract_prediction(msg_history)
        
        if prediction == "None":
            self.log_fn(f"Failed to extract prediction from response: {response[:200] if response else 'empty'}")
        else:
            self.log_fn(f"Extracted grade '{prediction}' with confidence {confidence:.2f}")

        return str(prediction), msg_history
