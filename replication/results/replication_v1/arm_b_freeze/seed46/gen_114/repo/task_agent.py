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
    
    original_grade = grade.strip()
    grade = original_grade.lower()
    
    # First, check for numeric grades (0, 0.5, 1, 2, etc.)
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
        'full marks', 'complete', 'accepted', 'pass',
        'solved', 'solution correct', 'answer correct', 'valid'
    ]
    incorrect_variations = [
        'incorrect', 'wrong', 'false', 'no', 'none', 'zero',
        'invalid', 'rejected', 'fail', 'error', 'mistake',
        'unsolved', 'solution incorrect', 'answer incorrect', 'not correct'
    ]
    partial_variations = [
        'partial', 'partial credit', 'half', 'incomplete',
        'partially correct', 'partially right', 'some credit',
        'partially solved', 'partial solution', 'partially valid'
    ]
    
    # Check for exact matches first
    if grade in correct_variations:
        return 'Correct'
    if grade in incorrect_variations:
        return 'Incorrect'
    if grade in partial_variations:
        return 'Partial'
    
    # Substring matching for more flexibility - but require word boundaries
    # to avoid matching "valid" inside "invalid"
    for v in correct_variations:
        if re.search(r'\b' + re.escape(v) + r'\b', grade):
            return 'Correct'
    for v in incorrect_variations:
        if re.search(r'\b' + re.escape(v) + r'\b', grade):
            return 'Incorrect'
    for v in partial_variations:
        if re.search(r'\b' + re.escape(v) + r'\b', grade):
            return 'Partial'
    
    # Handle edge cases with punctuation - check for numeric grades
    grade_clean = grade.strip('.,;:!?"\'')
    if grade_clean in ['0', '0.0']:
        return 'Incorrect'
    if grade_clean in ['0.5', '.5']:
        return 'Partial'
    if grade_clean in ['1', '1.0', '1.5', '2', '2.0']:
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

## Response Format:

You MUST respond in JSON format wrapped in <json> tags:

<json>
{{
    "reasoning": "Your detailed step-by-step analysis following the Evaluation Framework above",
    "response": "The final grade you assign: use 'Correct' for full credit (scores 1 or 2), 'Partial' for partial credit (score 0.5), or 'Incorrect' for no credit (score 0)"
}}
</json>

Important: 
- The "response" field must contain ONLY the grade value, nothing else.
- Use semantic grades: 'Correct' (full credit), 'Partial' (partial credit), or 'Incorrect' (no credit).
- These correspond to IMO numeric grades: Correct=1-2 points, Partial=0.5 points, Incorrect=0 points."""

    def _validate_grade(self, grade: str) -> str:
        """Validate and normalize extracted grade to standard format.
        
        Returns the normalized grade if valid, or the original if no
        standard normalization applies.
        
        This method ensures consistent output format by converting all
        numeric IMO grades (0, 0.5, 1, 2) to semantic grades 
        (Incorrect, Partial, Correct) for consistency.
        """
        if not grade or grade == "None":
            return "None"
        
        stripped = grade.strip()
        
        # Handle raw numeric IMO grades - convert to semantic grades for consistency
        if stripped == "0":
            return "Incorrect"
        if stripped in ["0.5", ".5"]:
            return "Partial"
        if stripped in ["1", "2", "1.0", "2.0"]:
            return "Correct"
        
        # Normalize the grade using the global normalizer
        normalized = _normalize_grade(grade)
        
        # Check if it's already a valid standard grade
        valid_grades = ["Correct", "Incorrect", "Partial"]
        if normalized in valid_grades:
            return normalized
        
        # Try to extract numeric grades from the normalized value
        numeric_match = re.search(r'\b([0-2](?:\.\d+)?)\b', normalized)
        if numeric_match:
            val = float(numeric_match.group(1))
            if val == 0:
                return "Incorrect"
            elif val >= 1:
                return "Correct"
            else:
                return "Partial"
        
        # Return the normalized value even if not in standard list
        # (allows for domain-specific grades)
        return normalized

    def _extract_prediction(self, msg_history: list[dict]) -> str:
        """Extract prediction from message history with multiple fallback strategies."""
        if not msg_history:
            return "None"
        
        last_message = msg_history[-1].get("text", "")
        
        # Strategy 1: Extract from <json> tags
        extracted = _extract_jsons(last_message)
        if extracted:
            grade = self._get_grade_from_json(extracted[-1])
            validated = self._validate_grade(grade)
            if validated != "None":
                return validated
        
        # Strategy 2: Extract from markdown code blocks
        extracted = _extract_json_from_markdown(last_message)
        if extracted:
            grade = self._get_grade_from_json(extracted[-1])
            validated = self._validate_grade(grade)
            if validated != "None":
                return validated
        
        # Strategy 3: Look for grade patterns in plain text
        grade = self._extract_grade_from_text(last_message)
        return self._validate_grade(grade)

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
                    # Handle numeric grades properly
                    if value == 0:
                        return "Incorrect"
                    elif value >= 1:
                        return "Correct"
                    elif 0 < value < 1:
                        return "Partial"
                    return str(int(value)) if value == int(value) else str(value)
        
        # If no recognized field, use the first string or numeric value found
        for key, value in json_obj.items():
            if isinstance(value, str):
                return _normalize_grade(value)
            elif isinstance(value, (int, float)):
                # Handle numeric grades properly
                if value == 0:
                    return "Incorrect"
                elif value >= 1:
                    return "Correct"
                elif 0 < value < 1:
                    return "Partial"
                return str(int(value)) if value == int(value) else str(value)
        
        return "None"

    def _extract_grade_from_text(self, text: str) -> str:
        """Extract grade from plain text using pattern matching.
        
        Enhanced with additional patterns for robust extraction from
        various response formats including IMO-style numeric grades.
        
        Improvements:
        - Better handling of negations (e.g., "not incorrect")
        - Priority-based extraction from multiple grade mentions
        - Context-aware grade selection from conclusion sections
        """
        # Look for explicit grade statements with flexible patterns
        # Ordered by specificity - more specific patterns first
        patterns = [
            # JSON-like field patterns (highest priority)
            r'"response"\s*:\s*"([^"]+)"',
            r'"grade"\s*:\s*"([^"]+)"',
            r'"response"\s*:\s*(\d+(?:\.\d+)?)',
            r'"grade"\s*:\s*(\d+(?:\.\d+)?)',
            # Standard grade assignments
            r'grade[\s]*[:=][\s]*["\']?([^"\'\n,]+)["\']?',
            r'response[\s]*[:=][\s]*["\']?([^"\'\n,]+)["\']?',
            r'final grade[\s]*[:=][\s]*["\']?([^"\'\n,]+)["\']?',
            r'assign[\s]+["\']?([^"\'\n,]+)["\']?',
            # IMO-style grade statements
            r'(?:the\s+)?(?:student\s+(?:receives?|gets?|earns?)|I\s+(?:assign|give|award))\s+["\']?(\d+(?:\.\d+)?)["\']?\s*(?:points?|marks?)?',
            r'(?:score|grade|mark)[\s]*[:=\s]+["\']?(\d+(?:\.\d+)?)["\']?',
            # Grade at end of sentence
            r'(?:grade|score|mark|result)[\s]+(?:is|of)[\s]+["\']?([^"\'\n.]+)["\']?',
            # Standalone grades in common formats
            r'\b([0-2](?:\.0|\.5)?)\b',
            r'\b(Correct|Incorrect|Partial|Right|Wrong)\b',
            # Additional patterns for edge cases
            r'(?:the\s+)?(?:answer|solution)\s+(?:is|should\s+be)\s+["\']?([^"\'\n,]+)["\']?',
            # Conclusion patterns
            r'(?:therefore|thus|hence|conclusion)[,:]?\s+(?:the\s+)?(?:grade|score|answer)\s+(?:is\s+)?["\']?([^"\'\n.]+)["\']?',
        ]
        
        # Collect all matches with their positions for context-aware selection
        all_matches = []
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                grade_value = match.group(1).strip()
                position = match.start()
                all_matches.append((position, grade_value))
        
        # If we found matches, prioritize those in conclusion sections
        if all_matches:
            # Sort by position
            all_matches.sort(key=lambda x: x[0])
            
            # Look for conclusion section markers
            conclusion_markers = ['conclusion', 'final grade', 'in summary', 'to summarize']
            text_lower = text.lower()
            conclusion_start = len(text)
            for marker in conclusion_markers:
                pos = text_lower.find(marker)
                if pos != -1 and pos < conclusion_start:
                    conclusion_start = pos
            
            # Prioritize matches in the conclusion section (last 30% of text or after conclusion marker)
            conclusion_threshold = max(conclusion_start, int(len(text) * 0.7))
            
            conclusion_matches = [(pos, grade) for pos, grade in all_matches if pos >= conclusion_threshold]
            other_matches = [(pos, grade) for pos, grade in all_matches if pos < conclusion_threshold]
            
            # Try conclusion matches first, then other matches
            for matches in [conclusion_matches, other_matches]:
                for pos, grade_value in matches:
                    normalized = _normalize_grade(grade_value)
                    if normalized != "None" and normalized != grade_value:
                        return normalized
            
            # If no valid normalized grade found, return the last match
            if all_matches:
                return _normalize_grade(all_matches[-1][1])
        
        # Fallback: look for numeric grades (0, 0.5, 1, 2) as standalone tokens
        # This handles cases where the model just outputs a number
        numeric_matches = re.findall(r'\b([0-2](?:\.\d+)?)\b', text)
        if numeric_matches:
            # Return the last numeric match (often the final grade)
            return _normalize_grade(numeric_matches[-1])
        
        # Final fallback: look for grade keywords anywhere in text with negation handling
        text_lower = text.lower()
        
        # Define negation patterns to check
        negation_patterns = [
            (r'not\s+incorrect', 'Incorrect'),  # "not incorrect" -> Incorrect (double negative)
            (r'not\s+wrong', 'Incorrect'),
            (r'not\s+correct', 'Incorrect'),
            (r'not\s+right', 'Incorrect'),
            (r'not\s+partial', 'Incorrect'),
        ]
        
        # Check for negated patterns first
        for pattern, result in negation_patterns:
            if re.search(pattern, text_lower):
                return result
        
        # Check for direct grade mentions (without negation context)
        # Look at the last sentence for the final determination
        sentences = re.split(r'[.!?]+', text_lower)
        for sentence in reversed(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Check for incorrect/wrong (but not in context of "not wrong")
            if re.search(r'\b(incorrect|wrong|false|invalid)\b', sentence):
                return 'Incorrect'
            if re.search(r'\bpartial\b', sentence):
                return 'Partial'
            if re.search(r'\b(correct|right|true|valid)\b', sentence):
                return 'Correct'
        
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
