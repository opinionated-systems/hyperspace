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
    Supports both <json> and <JSON> tag variants.
    """
    results = []
    search_from = 0
    while True:
        # Try lowercase <json> first, then uppercase <JSON>
        start = text.find("<json>", search_from)
        if start == -1:
            start = text.find("<JSON>", search_from)
            if start == -1:
                break
            # Adjust for uppercase tag length
            tag_end = start + 6
        else:
            tag_end = start + 6
        
        # Find corresponding closing tag (try lowercase first, then uppercase)
        end = text.find("</json>", start)
        if end == -1:
            end = text.find("</JSON>", start)
            if end == -1:
                break
        
        inner = text[tag_end:end].strip()
        search_from = end + 7
        
        # Try to parse the JSON
        try:
            results.append(json.loads(inner))
        except json.JSONDecodeError:
            # Try common fixes: remove trailing commas, fix quotes, handle comments
            try:
                # Remove trailing commas before closing braces/brackets
                fixed = re.sub(r',(\s*[}\]])', r'\1', inner)
                # Remove single-line comments
                fixed = re.sub(r'//[^\n]*', '', fixed)
                # Remove multi-line comments
                fixed = re.sub(r'/\*.*?\*/', '', fixed, flags=re.DOTALL)
                results.append(json.loads(fixed))
            except json.JSONDecodeError:
                # Try extracting just the first valid JSON object
                try:
                    # Find the first { and matching }
                    brace_start = inner.find('{')
                    if brace_start != -1:
                        brace_count = 0
                        for i, char in enumerate(inner[brace_start:]):
                            if char == '{':
                                brace_count += 1
                            elif char == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    results.append(json.loads(inner[brace_start:brace_start+i+1]))
                                    break
                except json.JSONDecodeError:
                    continue
    return results or None


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
        try:
            results.append(json.loads(match))
        except json.JSONDecodeError:
            # Try common fixes
            try:
                # Remove trailing commas
                fixed = re.sub(r',(\s*[}\]])', r'\1', match)
                # Remove single-line comments
                fixed = re.sub(r'//[^\n]*', '', fixed)
                # Remove multi-line comments
                fixed = re.sub(r'/\*.*?\*/', '', fixed, flags=re.DOTALL)
                results.append(json.loads(fixed))
            except json.JSONDecodeError:
                # Try to extract the first valid JSON object
                try:
                    brace_start = match.find('{')
                    if brace_start != -1:
                        brace_count = 0
                        for i, char in enumerate(match[brace_start:]):
                            if char == '{':
                                brace_count += 1
                            elif char == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    results.append(json.loads(match[brace_start:brace_start+i+1]))
                                    break
                except json.JSONDecodeError:
                    continue
    return results or None


def _normalize_grade(grade: str) -> str:
    """Normalize grade to a standard format.

    Handles various grade formats and normalizes them.
    Enhanced to handle numeric grades, boolean values, and more variations.
    """
    if not isinstance(grade, str):
        grade = str(grade)
    
    original = grade.strip()
    grade = original.lower()
    
    # Handle empty or whitespace-only grades
    if not grade:
        return "None"
    
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
    
    # Handle boolean-like values
    if grade in ['true', '1', 't', 'y']:
        return 'Correct'
    if grade in ['false', '0', 'f', 'n']:
        return 'Incorrect'
    
    # Map common variations to standard formats
    correct_variations = [
        'correct', 'right', 'true', 'yes', 'full', 'full credit', 
        'full marks', 'complete', 'valid', 'accepted', 'pass', 'solved',
        'success', 'accurate', 'perfect', 'excellent', 'good', '✓', '✔',
        'correct!', 'correct.', 'correct,', 'correctly', 'correct answer',
        'correct solution', 'correctly solved', 'correctly answered',
        '1', '2', '3', '4', '5', '6', '7',  # IMO-style numeric grades >= 1
    ]
    incorrect_variations = [
        'incorrect', 'wrong', 'false', 'no', 'none', 'zero',
        'invalid', 'rejected', 'fail', 'error', 'mistake', 'unsolved',
        'failure', 'bad', 'unacceptable', 'unsatisfactory', '✗', '✘', '×',
        'incorrect!', 'incorrect.', 'incorrect,', 'wrong!', 'wrong.', 'wrong,',
        'wrong answer', 'wrong solution', 'incorrectly', 'incorrectly solved',
        '0',  # IMO-style zero grade
    ]
    partial_variations = [
        'partial', 'partial credit', 'half', 'incomplete',
        'partially correct', 'partially right', 'some credit',
        'partially solved', 'almost', 'nearly', 'minor errors',
        'partial!', 'partial.', 'partial,', 'partially', 'incomplete!',
        'incomplete.', 'incomplete,', 'half credit', 'half marks',
        'partial solution', 'partial answer', 'partially correct answer',
        '0.5', '0.25', '0.75',  # Common partial numeric grades
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
    grade_words = set(re.findall(r'\b\w+\b', grade))
    
    if any(v in grade_words for v in ['correct', 'right', 'true', 'yes', 'full', 'complete', 'valid', 'accepted', 'pass', 'solved', 'success', 'accurate', 'perfect', 'excellent', 'good']):
        return 'Correct'
    if any(v in grade_words for v in ['incorrect', 'wrong', 'false', 'no', 'none', 'zero', 'invalid', 'rejected', 'fail', 'error', 'mistake', 'unsolved', 'failure', 'bad', 'unacceptable']):
        return 'Incorrect'
    if any(v in grade_words for v in ['partial', 'half', 'incomplete', 'almost', 'nearly']):
        return 'Partial'
    
    # Check for substring matches in the full text
    if any(v in grade for v in correct_variations[:10]):  # Top 10 most common
        return 'Correct'
    elif any(v in grade for v in incorrect_variations[:10]):
        return 'Incorrect'
    elif any(v in grade for v in partial_variations[:10]):
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
        """Extract prediction from message history with multiple fallback strategies.
        
        Enhanced with cross-message search and confidence scoring to handle
        cases where the grade might be in earlier messages or ambiguous.
        """
        if not msg_history:
            return "None"
        
        # Search from the most recent message backwards
        for message in reversed(msg_history):
            last_message = message.get("text", "")
            if not last_message:
                continue
                
            # Strategy 1: Extract from <json> tags
            extracted = _extract_jsons(last_message)
            if extracted:
                grade = self._get_grade_from_json(extracted[-1])
                if grade != "None":
                    return grade
            
            # Strategy 2: Extract from markdown code blocks
            extracted = _extract_json_from_markdown(last_message)
            if extracted:
                grade = self._get_grade_from_json(extracted[-1])
                if grade != "None":
                    return grade
            
            # Strategy 3: Look for grade patterns in plain text
            grade = self._extract_grade_from_text(last_message)
            if grade != "None":
                return grade
        
        # If no grade found in any message, return None
        self.log_fn("No grade found in any message of the history")
        return "None"

    def _get_grade_from_json(self, json_obj: dict) -> str:
        """Extract grade from JSON object with field priority."""
        # Priority order for grade fields
        priority_fields = [
            "response", "grade", "answer", "result", "score", "evaluation",
            "prediction", "output", "verdict", "assessment", "judgment",
            "decision", "conclusion", "final_grade", "final_answer"
        ]
        
        for field in priority_fields:
            if field in json_obj:
                value = json_obj[field]
                if isinstance(value, str):
                    return _normalize_grade(value)
                elif isinstance(value, (int, float)):
                    return str(value)
        
        # If no recognized field, use the first string or numeric value found
        for key, value in json_obj.items():
            if isinstance(value, str):
                return _normalize_grade(value)
            elif isinstance(value, (int, float)):
                return str(value)
        
        return "None"

    def _extract_grade_from_text(self, text: str) -> str:
        """Extract grade from plain text using pattern matching.
        
        Enhanced with additional patterns for edge cases and better handling
        of IMO-style numeric grades.
        """
        # Look for explicit grade statements
        patterns = [
            r'grade[\s]*[:=][\s]*["\']?([^"\'\n]+)["\']?',
            r'response[\s]*[:=][\s]*["\']?([^"\'\n]+)["\']?',
            r'final grade[\s]*[:=][\s]*["\']?([^"\'\n]+)["\']?',
            r'assign[\s]+["\']?([^"\'\n]+)["\']?',
            r'["\']?grade["\']?\s*is\s*["\']?([^"\'\n]+)["\']?',
            r'["\']?score["\']?\s*[:=]\s*["\']?([^"\'\n]+)["\']?',
            r'["\']?result["\']?\s*[:=]\s*["\']?([^"\'\n]+)["\']?',
            r'["\']?evaluation["\']?\s*[:=]\s*["\']?([^"\'\n]+)["\']?',
            r'["\']?answer["\']?\s*[:=]\s*["\']?([^"\'\n]+)["\']?',
            r'["\']?the grade is["\']?\s*["\']?([^"\'\n]+)["\']?',
            r'["\']?i assign["\']?\s*["\']?([^"\'\n]+)["\']?',
            r'["\']?therefore["\']?.*?(?:grade|score|result)\s*[:=]?\s*["\']?([^"\'\n]+)["\']?',
            r'["\']?conclusion["\']?.*?(?:grade|score|result)\s*[:=]?\s*["\']?([^"\'\n]+)["\']?',
            r'["\']?final["\']?.*?(?:grade|score|result)\s*[:=]?\s*["\']?([^"\'\n]+)["\']?',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                grade = _normalize_grade(match.group(1).strip())
                if grade != "None":
                    return grade
        
        # Look for standalone grades at the end of the text (common pattern)
        lines = text.strip().split('\n')
        for line in reversed(lines[-10:]):  # Check last 10 lines (increased from 5)
            line = line.strip()
            # Skip empty lines and common non-grade lines
            if not line or line.startswith('#') or line.startswith('//') or line.startswith('*'):
                continue
                
            # Look for common grade patterns
            grade_patterns = [
                r'^\s*([0-9]+)\s*$',  # Just a number
                r'^\s*["\']?([0-9]+)["\']?\s*$',  # Number in quotes
                r'^\s*["\']?(Correct|Incorrect|Partial)["\']?\s*$',  # Standard grades
                r'^\s*["\']?(correct|incorrect|partial)["\']?\s*$',  # Lowercase
                r'^\s*["\']?(True|False)["\']?\s*$',  # Boolean grades
                r'^\s*["\']?(true|false)["\']?\s*$',  # Lowercase boolean
                r'^\s*["\']?(Yes|No)["\']?\s*$',  # Yes/No grades
                r'^\s*["\']?(yes|no)["\']?\s*$',  # Lowercase yes/no
                r'^\s*["\']?(Pass|Fail)["\']?\s*$',  # Pass/Fail grades
                r'^\s*["\']?(pass|fail)["\']?\s*$',  # Lowercase pass/fail
            ]
            for gp in grade_patterns:
                match = re.search(gp, line, re.IGNORECASE)
                if match:
                    return _normalize_grade(match.group(1).strip())
        
        # Look for grade indicators in the text body (less reliable, check last)
        body_patterns = [
            r'\b(Correct|Incorrect|Partial)\b',
            r'\b(correct|incorrect|partial)\b',
        ]
        for pattern in body_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
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
        # Validate inputs
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
            # Last resort: try to extract any grade-like word from the response
            if response:
                fallback_patterns = [
                    r'\b(Correct|Incorrect|Partial)\b',
                    r'\b(correct|incorrect|partial)\b',
                ]
                for pattern in fallback_patterns:
                    match = re.search(pattern, response, re.IGNORECASE)
                    if match:
                        prediction = _normalize_grade(match.group(1))
                        self.log_fn(f"Fallback extraction succeeded: {prediction}")
                        break

        return str(prediction), msg_history
