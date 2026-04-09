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
    Enhanced to handle numeric grades, fractions, and more variations.
    """
    if not isinstance(grade, str):
        grade = str(grade)
    
    grade = grade.strip().lower()
    
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
        'full marks', 'complete', 'valid', 'accepted', 'pass', 'perfect',
        'acceptable', 'satisfactory', 'good', 'excellent'
    ]
    incorrect_variations = [
        'incorrect', 'wrong', 'false', 'no', 'none', 'zero',
        'invalid', 'rejected', 'fail', 'error', 'mistake', 'unsatisfactory',
        'unacceptable', 'bad', 'poor', 'inadequate'
    ]
    partial_variations = [
        'partial', 'partial credit', 'half', 'incomplete',
        'partially correct', 'partially right', 'some credit', 'partially',
        'half credit', 'partial marks', 'incomplete solution'
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
    
    # Handle special cases like "0/1", "1/1", "0/2", "2/2", etc.
    if '/' in grade:
        parts = grade.split('/')
        if len(parts) == 2:
            try:
                numerator = float(parts[0].strip())
                denominator = float(parts[1].strip())
                if denominator > 0:
                    ratio = numerator / denominator
                    if ratio == 0:
                        return 'Incorrect'
                    elif ratio >= 0.5:
                        return 'Correct'
                    else:
                        return 'Partial'
            except ValueError:
                pass
    
    # Handle decimal grades like "0.5", "1.0", "2.0"
    if '.' in grade:
        try:
            decimal_grade = float(grade)
            if decimal_grade == 0:
                return 'Incorrect'
            elif decimal_grade >= 1:
                return 'Correct'
            elif 0 < decimal_grade < 1:
                return 'Partial'
        except ValueError:
            pass
    
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
        
        Enhanced with additional patterns for robust extraction from
        various response formats including IMO-style numeric grades.
        Includes multi-stage extraction with confidence scoring.
        """
        # Stage 1: Look for explicit grade statements with flexible patterns
        # Priority-ordered patterns - earlier patterns are more reliable
        priority_patterns = [
            # JSON-like grade assignments (most reliable)
            r'"grade"\s*:\s*"([^"]+)"',
            r'"response"\s*:\s*"([^"]+)"',
            r'"result"\s*:\s*"([^"]+)"',
            # Explicit grade declarations
            r'grade[\s]*[:=][\s]*["\']?([^"\'\n]+)["\']?',
            r'response[\s]*[:=][\s]*["\']?([^"\'\n]+)["\']?',
            r'final grade[\s]*[:=][\s]*["\']?([^"\'\n]+)["\']?',
            r'assign[\s]+["\']?([^"\'\n]+)["\']?',
            # IMO-style grade statements with points/marks
            r'(?:the\s+)?(?:student\s+(?:receives?|gets?|earns?)|I\s+(?:assign|give|award))\s+["\']?(\d+(?:\.\d+)?)["\']?\s*(?:points?|marks?)?',
            r'(?:score|grade|mark)[\s]*[:=\s]+["\']?(\d+(?:\.\d+)?)["\']?',
            # Grade at end of sentence
            r'(?:grade|score|mark|result)[\s]+(?:is|of)[\s]+["\']?([^"\'\n.]+)["\']?',
        ]
        
        for pattern in priority_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return _normalize_grade(match.group(1).strip())
        
        # Stage 2: Look for standalone grades in common formats
        standalone_patterns = [
            # IMO numeric grades (0, 0.5, 1, 1.5, 2, etc.)
            r'\b([0-2](?:\.0|\.5)?)\b',
            # Letter/word grades
            r'\b(Correct|Incorrect|Partial|Right|Wrong|Acceptable|Unacceptable)\b',
            # Fraction grades like 1/2, 0/1, 2/2
            r'\b([0-2]/[0-2])\b',
        ]
        
        for pattern in standalone_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return _normalize_grade(match.group(1).strip())
        
        # Stage 3: Context-aware extraction from sentences
        # Look for grade mentions in context
        context_patterns = [
            r'(?:the\s+)?(?:answer|solution)\s+(?:is|would\s+be)\s+["\']?([^"\'\n.]{3,30})["\']?',
            r'(?:therefore|thus|hence|so)\s*,?\s*(?:the\s+)?(?:grade|score|result)\s+(?:is|would\s+be)\s+["\']?([^"\'\n.]{3,30})["\']?',
            r'(?:I\s+(?:would|will)\s+)?(?:assign|give|award|grant)\s+["\']?([^"\'\n.]{3,30})["\']?',
        ]
        
        for pattern in context_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                candidate = match.group(1).strip()
                # Only accept if it looks like a grade
                if any(keyword in candidate.lower() for keyword in ['correct', 'incorrect', 'partial', 'right', 'wrong', '0', '1', '2', 'full', 'half', 'none']):
                    return _normalize_grade(candidate)
        
        # Stage 4: Fallback - look for numeric grades (0, 1, 2) as standalone tokens
        # This handles cases where the model just outputs a number
        numeric_matches = re.findall(r'\b([0-2])\b', text)
        if numeric_matches:
            # Return the last numeric match (often the final grade)
            return _normalize_grade(numeric_matches[-1])
        
        # Stage 5: Semantic analysis - look for grade-like words in context
        text_lower = text.lower()
        
        # Check for explicit negations first (to avoid false positives)
        negation_patterns = [
            'not correct', 'not right', 'not valid', 'not acceptable',
            'is incorrect', 'is wrong', 'is not', 'does not',
        ]
        has_negation = any(neg in text_lower for neg in negation_patterns)
        
        # Look for grade indicators with context
        if has_negation:
            if 'correct' in text_lower or 'right' in text_lower:
                return 'Incorrect'
        
        # Check for positive indicators
        positive_indicators = ['correct', 'right', 'valid', 'acceptable', 'full credit', 'full marks']
        negative_indicators = ['incorrect', 'wrong', 'invalid', 'unacceptable', 'no credit', 'zero']
        partial_indicators = ['partial', 'partially', 'some credit', 'half credit', 'incomplete']
        
        positive_count = sum(1 for ind in positive_indicators if ind in text_lower)
        negative_count = sum(1 for ind in negative_indicators if ind in text_lower)
        partial_count = sum(1 for ind in partial_indicators if ind in text_lower)
        
        # Determine grade based on indicator counts and context
        if partial_count > 0 and (positive_count > 0 or negative_count > 0):
            return 'Partial'
        elif negative_count > positive_count:
            return 'Incorrect'
        elif positive_count > negative_count:
            return 'Correct'
        elif partial_count > 0:
            return 'Partial'
        
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
