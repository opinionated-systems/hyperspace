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
            # Also try lowercase variant <JSON>
            start = text.find("<JSON>", search_from)
            if start == -1:
                break
        end = text.find("</json>", start)
        if end == -1:
            end = text.find("</JSON>", start)
            if end == -1:
                break
        inner = text[start + 6:end].strip()
        search_from = end + 7
        
        # Try to parse the JSON
        try:
            results.append(json.loads(inner))
        except json.JSONDecodeError:
            # Try common fixes: remove trailing commas, fix quotes
            try:
                # Remove trailing commas before closing braces/brackets
                fixed = re.sub(r',(\s*[}\]])', r'\1', inner)
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
    # Match ```json ... ``` or just ``` ... ``` blocks (non-greedy but with proper handling)
    # Use a more robust pattern that handles nested content better
    pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    matches = re.findall(pattern, text, re.DOTALL)
    for match in matches:
        match = match.strip()
        if not match:
            continue
        try:
            results.append(json.loads(match))
        except json.JSONDecodeError:
            # Try common fixes
            try:
                # Remove trailing commas before closing braces/brackets
                fixed = re.sub(r',(\s*[}\]])', r'\1', match)
                results.append(json.loads(fixed))
            except json.JSONDecodeError:
                # Try to extract the first valid JSON object using brace counting
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
                    # Try extracting array if object fails
                    try:
                        bracket_start = match.find('[')
                        if bracket_start != -1:
                            bracket_count = 0
                            for i, char in enumerate(match[bracket_start:]):
                                if char == '[':
                                    bracket_count += 1
                                elif char == ']':
                                    bracket_count -= 1
                                    if bracket_count == 0:
                                        results.append(json.loads(match[bracket_start:bracket_start+i+1]))
                                        break
                    except json.JSONDecodeError:
                        continue
    return results or None


def _normalize_grade(grade: str) -> str:
    """Normalize grade to a standard format.

    Handles various grade formats and normalizes them.
    Enhanced to handle numeric grades (IMO-style 0-7 scale) and more variations.
    """
    if not isinstance(grade, str):
        grade = str(grade)
    
    original = grade.strip()
    grade = original.lower()
    
    # First, check for numeric grades (0, 1, 2, etc.)
    # IMO-style grading: 0-7 scale where 7 is full marks
    try:
        numeric_grade = float(grade)
        # IMO scale: 0 = no progress, 1-6 = partial progress, 7 = full solution
        # For binary classification: 0 = Incorrect, 1-7 = Correct
        # For ternary: 0 = Incorrect, 1-6 = Partial, 7 = Correct
        if numeric_grade == 0:
            return 'Incorrect'
        elif numeric_grade >= 6:  # 6 or 7 on IMO scale = essentially correct
            return 'Correct'
        else:  # 1-5 = partial credit
            return 'Partial'
    except ValueError:
        pass
    
    # Map common variations to standard formats
    correct_variations = [
        'correct', 'right', 'true', 'yes', 'full', 'full credit', 
        'full marks', 'complete', 'valid', 'accepted', 'pass', 'solved',
        'success', 'accurate', 'perfect', 'excellent', 'good', '7', '6'
    ]
    incorrect_variations = [
        'incorrect', 'wrong', 'false', 'no', 'none', 'zero', '0',
        'invalid', 'rejected', 'fail', 'error', 'mistake', 'unsolved',
        'failure', 'bad', 'unacceptable', 'unsatisfactory'
    ]
    partial_variations = [
        'partial', 'partial credit', 'half', 'incomplete',
        'partially correct', 'partially right', 'some credit',
        'partially solved', 'almost', 'nearly', 'minor errors',
        '1', '2', '3', '4', '5'
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

For IMO-style problems with numeric grades (0-7 scale):
- **0**: No progress or completely wrong → Grade as "Incorrect"
- **1-5**: Some progress but incomplete or incorrect final answer → Grade as "Partial"
- **6-7**: Correct or nearly correct solution → Grade as "Correct"

## Response Format (CRITICAL - FOLLOW EXACTLY):

You MUST respond in JSON format wrapped in <json> tags. The response field must contain ONLY one of: "Correct", "Incorrect", or "Partial".

<json>
{{
    "reasoning": "Your detailed step-by-step analysis following the Evaluation Framework above",
    "response": "Correct"
}}
</json>

OR

<json>
{{
    "reasoning": "Your detailed step-by-step analysis following the Evaluation Framework above",
    "response": "Incorrect"
}}
</json>

OR

<json>
{{
    "reasoning": "Your detailed step-by-step analysis following the Evaluation Framework above",
    "response": "Partial"
}}
</json>

IMPORTANT RULES:
1. The "response" field must contain ONLY one of these exact values: "Correct", "Incorrect", or "Partial"
2. Do NOT include explanations, quotes, or additional text in the response field
3. Do NOT use numeric grades (0, 1, 2, etc.) in the response field - use the text labels above
4. The reasoning field should contain your full analysis
5. Always wrap your entire response in <json>...</json> tags"""

    def _extract_prediction(self, msg_history: list[dict]) -> str:
        """Extract prediction from message history with multiple fallback strategies."""
        if not msg_history:
            return "None"
        
        last_message = msg_history[-1].get("text", "")
        if not last_message:
            return "None"
        
        # Strategy 1: Extract from <json> tags (most reliable)
        extracted = _extract_jsons(last_message)
        if extracted:
            result = self._get_grade_from_json(extracted[-1])
            if result != "None":
                return result
        
        # Strategy 2: Extract from markdown code blocks
        extracted = _extract_json_from_markdown(last_message)
        if extracted:
            result = self._get_grade_from_json(extracted[-1])
            if result != "None":
                return result
        
        # Strategy 3: Look for grade patterns in plain text
        result = self._extract_grade_from_text(last_message)
        if result != "None":
            return result
        
        # Strategy 4: Last resort - check if the entire message is just a grade
        stripped = last_message.strip()
        if stripped in ['0', '1', '2', '3', '4', '5', '6', '7']:
            return _normalize_grade(stripped)
        if stripped.lower() in ['correct', 'incorrect', 'partial']:
            return _normalize_grade(stripped)
        
        return "None"

    def _get_grade_from_json(self, json_obj: dict) -> str:
        """Extract grade from JSON object with field priority."""
        # Priority order for grade fields (most specific to least specific)
        priority_fields = [
            "grade",      # Most specific - explicitly a grade
            "response",   # The task agent's expected output field
            "result",     # Generic result field
            "answer",     # Answer field
            "evaluation", # Evaluation result
            "score",      # Numeric score
            "verdict",    # Verdict/assessment
            "assessment", # Assessment field
        ]
        
        for field in priority_fields:
            if field in json_obj:
                value = json_obj[field]
                if isinstance(value, str):
                    normalized = _normalize_grade(value)
                    if normalized != value:  # Successfully normalized
                        return normalized
                elif isinstance(value, (int, float)):
                    # Convert numeric to string and normalize
                    return _normalize_grade(str(value))
        
        # If no recognized field, use the first string or numeric value found
        for key, value in json_obj.items():
            if isinstance(value, str):
                normalized = _normalize_grade(value)
                if normalized != value:  # Successfully normalized
                    return normalized
            elif isinstance(value, (int, float)):
                return _normalize_grade(str(value))
        
        return "None"

    def _extract_grade_from_text(self, text: str) -> str:
        """Extract grade from plain text using pattern matching."""
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
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return _normalize_grade(match.group(1).strip())
        
        # Look for standalone grades at the end of the text (common pattern)
        lines = text.strip().split('\n')
        for line in reversed(lines[-10:]):  # Check last 10 lines (increased from 5)
            line = line.strip()
            # Skip empty lines and common non-grade lines
            if not line or line.startswith('#') or line.startswith('//') or line.startswith('*'):
                continue
            # Look for common grade patterns
            grade_patterns = [
                r'^\s*([0-7])\s*$',  # Single digit 0-7 (IMO scale)
                r'^\s*["\']?([0-7])["\']?\s*$',  # Digit in quotes
                r'^\s*["\']?(Correct|Incorrect|Partial)["\']?\s*$',  # Standard grades
                r'^\s*["\']?(correct|incorrect|partial)["\']?\s*$',  # Lowercase
                r'^\s*Grade:\s*["\']?([^"\'\n]+)["\']?\s*$',  # "Grade: X" format
                r'^\s*The grade is\s*["\']?([^"\'\n]+)["\']?\s*$',  # "The grade is X"
            ]
            for gp in grade_patterns:
                match = re.search(gp, line, re.IGNORECASE)
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
            # Log more details for debugging
            last_msg = msg_history[-1].get("text", "") if msg_history else ""
            self.log_fn(f"Failed to extract prediction. Last message preview: {last_msg[:500] if last_msg else 'empty'}")
            # Try to extract from the raw response as well
            if response and response != last_msg:
                prediction = self._extract_prediction([{"text": response}])
                if prediction != "None":
                    self.log_fn(f"Successfully extracted from raw response: {prediction}")

        return str(prediction), msg_history
