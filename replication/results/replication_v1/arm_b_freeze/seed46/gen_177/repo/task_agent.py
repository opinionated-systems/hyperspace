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
    grade = grade.strip().lower()
    grade = grade.strip('.,;:!?"\'""''')
    
    # Handle empty or whitespace-only grades
    if not grade:
        return 'None'
    
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
        'full marks', 'complete', 'accepted', 'pass', 'passed',
        'solved', 'solution correct', 'answer correct', 'valid',
        'success', 'successful', 'accurate', 'perfect'
    ]
    incorrect_variations = [
        'incorrect', 'wrong', 'false', 'no', 'none', 'zero',
        'invalid', 'rejected', 'fail', 'failed', 'error', 'mistake',
        'unsolved', 'solution incorrect', 'answer incorrect',
        'unsuccessful', 'inaccurate', 'unacceptable'
    ]
    partial_variations = [
        'partial', 'partial credit', 'half', 'incomplete',
        'partially correct', 'partially right', 'some credit',
        'partially solved', 'partial solution', 'partially accepted',
        'almost', 'nearly', 'mostly correct', 'minor errors'
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
    
    # Handle edge cases with punctuation and common suffixes
    grade_clean = grade.strip('.,;:!?"\'""''()[]{}')
    grade_clean = re.sub(r'\s+', ' ', grade_clean)  # Normalize whitespace
    
    # Check cleaned version
    if grade_clean in ['0', '0.0', '0.5']:
        return 'Incorrect'
    if grade_clean in ['1', '1.0', '1.5', '2', '2.0']:
        return 'Correct'
    
    # Handle grades with common suffixes like "points" or "marks"
    grade_no_suffix = re.sub(r'\s*(?:points?|marks?|score|grade)$', '', grade_clean).strip()
    if grade_no_suffix in ['0', '1', '2']:
        return 'Incorrect' if grade_no_suffix == '0' else 'Correct'
    
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

    def _validate_grade(self, grade: str) -> str:
        """Validate and normalize extracted grade to standard format.
        
        Returns the normalized grade if valid, or the original if no
        standard normalization applies.
        """
        if not grade or grade == "None":
            return "None"
        
        # First check for raw numeric grades (0, 1, 2) - these are valid IMO grades
        stripped = grade.strip()
        if stripped in ["0", "1", "2"]:
            return stripped
        
        # Normalize the grade
        normalized = _normalize_grade(grade)
        
        # Check if it's a valid standard grade
        valid_grades = ["Correct", "Incorrect", "Partial", "0", "1", "2"]
        if normalized in valid_grades:
            return normalized
        
        # Try to extract numeric grades from the normalized value
        numeric_match = re.search(r'\b([0-2])\b', normalized)
        if numeric_match:
            return numeric_match.group(1)
        
        # Return the normalized value even if not in standard list
        # (allows for domain-specific grades)
        return normalized

    def _extract_prediction(self, msg_history: list[dict]) -> str:
        """Extract prediction from message history with multiple fallback strategies.
        
        Enhanced with cross-message search and confidence scoring to handle
        cases where the grade might be in earlier messages or multiple formats.
        Also handles cases where the model outputs reasoning after the grade.
        """
        if not msg_history:
            return "None"
        
        # Collect all assistant messages for cross-message search
        assistant_messages = []
        for msg in reversed(msg_history):
            if msg.get("role") == "assistant":
                content = msg.get("text", "") or msg.get("content", "")
                if content:
                    assistant_messages.append(content)
        
        if not assistant_messages:
            return "None"
        
        # Try each message in reverse chronological order (most recent first)
        for msg_idx, last_message in enumerate(assistant_messages):
            prefix = f"[msg-{msg_idx}] " if msg_idx > 0 else ""
            
            # Strategy 1: Extract from <json> tags (highest confidence)
            extracted = _extract_jsons(last_message)
            if extracted:
                for json_obj in reversed(extracted):  # Try most recent JSON first
                    grade = self._get_grade_from_json(json_obj)
                    validated = self._validate_grade(grade)
                    if validated != "None":
                        self.log_fn(f"{prefix}Extracted grade from <json> tag: {validated}")
                        return validated
            
            # Strategy 2: Extract from markdown code blocks
            extracted = _extract_json_from_markdown(last_message)
            if extracted:
                for json_obj in reversed(extracted):
                    grade = self._get_grade_from_json(json_obj)
                    validated = self._validate_grade(grade)
                    if validated != "None":
                        self.log_fn(f"{prefix}Extracted grade from markdown: {validated}")
                        return validated
            
            # Strategy 2.5: Look for JSON-like structures without proper tags
            # This handles cases where model outputs JSON but forgets the tags
            json_like_pattern = r'\{\s*"reasoning"\s*:\s*"([^"]*)"\s*,\s*"response"\s*:\s*"([^"]*)"\s*\}'
            match = re.search(json_like_pattern, last_message, re.DOTALL)
            if match:
                grade = match.group(2).strip()
                validated = self._validate_grade(grade)
                if validated != "None":
                    self.log_fn(f"{prefix}Extracted grade from JSON-like structure: {validated}")
                    return validated
            
            # Strategy 3: Look for grade patterns in plain text
            grade = self._extract_grade_from_text(last_message)
            validated = self._validate_grade(grade)
            if validated != "None":
                self.log_fn(f"{prefix}Extracted grade from text pattern: {validated}")
                return validated
        
        # Final fallback: return None if nothing found
        self.log_fn(f"Failed to extract grade from {len(assistant_messages)} assistant messages")
        return "None"

    def _get_grade_from_json(self, json_obj: dict) -> str:
        """Extract grade from JSON object with field priority.
        
        Enhanced to handle cases where the response field contains extra text
        or where the grade is embedded in a longer string.
        """
        # Priority order for grade fields
        priority_fields = ["response", "grade", "answer", "result", "score", "evaluation"]
        
        for field in priority_fields:
            if field in json_obj:
                value = json_obj[field]
                if isinstance(value, str):
                    # First, try to extract a standalone numeric grade (0, 1, 2)
                    stripped = value.strip()
                    if stripped in ["0", "1", "2"]:
                        return stripped
                    # Try to find a numeric grade at the start
                    numeric_match = re.match(r'^\s*([0-2])\b', stripped)
                    if numeric_match:
                        return numeric_match.group(1)
                    # Try to find "Grade: X" or "Response: X" pattern
                    grade_prefix_match = re.search(r'(?:grade|response|score)\s*[:=]\s*([0-2])\b', stripped, re.IGNORECASE)
                    if grade_prefix_match:
                        return grade_prefix_match.group(1)
                    return _normalize_grade(value)
                elif isinstance(value, (int, float)):
                    # Convert numeric grades directly
                    if value == 0:
                        return "0"
                    elif value == 1:
                        return "1"
                    elif value == 2:
                        return "2"
                    elif value >= 1:
                        return "Correct"
                    elif value > 0:
                        return "Partial"
                    else:
                        return "Incorrect"
        
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
        Includes multi-line support and handles edge cases better.
        
        This version includes improved pattern organization and additional
        edge case handling for more reliable grade extraction.
        """
        # Organized patterns by category for better maintainability
        
        # Category 1: JSON-like field patterns (highest priority)
        json_patterns = [
            r'"response"\s*:\s*"([^"]+)"',
            r'"grade"\s*:\s*"([^"]+)"',
            r'"response"\s*:\s*(\d+)',
            r'"grade"\s*:\s*(\d+)',
            r'"evaluation"\s*:\s*"([^"]+)"',
            r'"result"\s*:\s*"([^"]+)"',
            r'"answer"\s*:\s*"([^"]+)"',
            r'"score"\s*:\s*(\d+)',
        ]
        
        # Category 2: Explicit grade assignment patterns
        assignment_patterns = [
            r'grade[\s]*[:=][\s]*["\']?([^"\'\n,]+)["\']?',
            r'response[\s]*[:=][\s]*["\']?([^"\'\n,]+)["\']?',
            r'final[\s]+grade[\s]*[:=][\s]*["\']?([^"\'\n,]+)["\']?',
            r'assign[\s]+["\']?([^"\'\n,]+)["\']?',
            r'^[\s]*[Gg]rade[\s]*[:=][\s]*["\']?([^"\'\n,]+)["\']?',
            r'[Tt]he\s+grade\s+(?:is|should\s+be)\s+["\']?([^"\'\n,]+)["\']?',
        ]
        
        # Category 3: IMO-style grade statements
        imo_patterns = [
            r'(?:the\s+)?(?:student\s+(?:receives?|gets?|earns?|deserves?)|I\s+(?:assign|give|award|grant))\s+["\']?(\d+(?:\.\d+)?)["\']?\s*(?:points?|marks?)?',
            r'(?:score|grade|mark|rating)[\s]*[:=\s]+["\']?(\d+(?:\.\d+)?)["\']?',
            r'(?:grade|score|mark|result|evaluation)[\s]+(?:is|of|equals?|will\s+be)[\s]+["\']?([^"\'\n.]+)["\']?',
        ]
        
        # Category 4: Standalone grade patterns
        standalone_patterns = [
            r'\b([0-2](?:\.0|\.5)?)\b',
            r'\b(Correct|Incorrect|Partial|Right|Wrong|Acceptable|Unacceptable)\b',
        ]
        
        # Category 5: Contextual grade patterns
        contextual_patterns = [
            r'(?:the\s+)?(?:answer|solution|response)\s+(?:is|should\s+be|would\s+be)\s+["\']?([^"\'\n,]+)["\']?',
            r'(?:therefore|thus|hence|so)[,\s]+(?:the\s+)?(?:grade|score|result)\s+(?:is|should\s+be)\s+["\']?([^"\'\n.]+)["\']?',
            r'(?:final|overall|total)\s+(?:grade|score|result)[\s]*[:=\s]+["\']?([^"\'\n,]+)["\']?',
        ]
        
        # Combine all patterns in priority order
        all_patterns = json_patterns + assignment_patterns + imo_patterns + standalone_patterns + contextual_patterns
        
        # Try each pattern with both single-line and multi-line modes
        for pattern in all_patterns:
            # Try case-insensitive search first
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return _normalize_grade(match.group(1).strip())
            # Also try with multiline flag for patterns that might span lines
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                return _normalize_grade(match.group(1).strip())
        
        # Fallback 1: Look for numeric grades (0, 1, 2) as standalone tokens
        # This handles cases where the model just outputs a number
        numeric_matches = re.findall(r'\b([0-2])\b', text)
        if numeric_matches:
            # Return the last numeric match (often the final grade)
            return _normalize_grade(numeric_matches[-1])
        
        # Fallback 2: Look for grade keywords in the last sentence/line
        lines = text.strip().split('\n')
        last_lines = lines[-3:] if len(lines) >= 3 else lines  # Check last 3 lines
        for line in reversed(last_lines):
            # Check for standalone grades at the end of lines
            end_match = re.search(r'[:=\s]+([0-2]|Correct|Incorrect|Partial)[\s]*\.?$', line, re.IGNORECASE)
            if end_match:
                return _normalize_grade(end_match.group(1).strip())
        
        # Fallback 3: Check for grade in the very last word of the text
        last_word_match = re.search(r'\b([0-2]|Correct|Incorrect|Partial)\s*$', text.strip(), re.IGNORECASE)
        if last_word_match:
            return _normalize_grade(last_word_match.group(1).strip())
        
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
