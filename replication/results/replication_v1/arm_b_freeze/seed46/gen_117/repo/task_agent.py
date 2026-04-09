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

from agent.llm_client import get_response_from_llm, get_response_from_llm_with_system, EVAL_MODEL

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
    
    # First, check for exact numeric grades (0, 1, 2) - preserve these for IMO
    if original_grade in ['0', '1', '2']:
        return original_grade
    
    # Check for numeric grades (0, 1, 2, etc.)
    # These are common in IMO-style grading
    try:
        numeric_grade = float(grade)
        if numeric_grade == 0:
            return '0'
        elif numeric_grade == 1:
            return '1'
        elif numeric_grade >= 2:
            return '2'
        elif 0 < numeric_grade < 1:
            return '1'  # Partial credit maps to 1
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
        'partially solved', 'partial solution', 'partially'
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
    
    # Handle edge cases with punctuation
    grade_clean = grade.strip('.,;:!?"\'')
    if grade_clean in ['0', '0.0', '0.5']:
        return '0'
    if grade_clean in ['1', '1.0', '1.5']:
        return '1'
    if grade_clean in ['2', '2.0']:
        return '2'
    
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

## IMO Grading Standards:

In IMO-style grading, the following standards apply:
- **Grade 2 (Correct)**: Complete solution with correct answer and valid reasoning
- **Grade 1 (Partial)**: Significant progress toward solution, or correct answer with minor gaps in reasoning
- **Grade 0 (Incorrect)**: No meaningful progress, fundamentally flawed approach, or completely wrong answer

## Evaluation Framework:

Follow this structured evaluation process carefully:

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

## Response Format (CRITICAL):

You MUST respond in JSON format wrapped in <json> tags. The response field MUST be exactly one of: "0", "1", "2", "Correct", "Incorrect", or "Partial".

<json>
{{
    "reasoning": "Your detailed step-by-step analysis following the Evaluation Framework above",
    "response": "0"
}}
</json>

Important: 
- The "response" field must contain ONLY the grade value: "0", "1", "2", "Correct", "Incorrect", or "Partial"
- Do not include any other text, explanation, or formatting in the response field
- Use "0" for incorrect, "1" for partial credit, "2" for fully correct (IMO standard)
- Or use "Correct", "Incorrect", "Partial" for categorical grading"""

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
        
        # Additional check: look for numeric grades in the original string
        # This handles cases like "Grade: 2" or "Score: 1"
        numeric_match = re.search(r'\b([0-2])\b', stripped)
        if numeric_match:
            return numeric_match.group(1)
        
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
        """
        # Look for explicit grade statements with flexible patterns
        # Ordered by specificity - more specific patterns first
        patterns = [
            # JSON-like field patterns (highest priority)
            r'"response"\s*:\s*"([^"]+)"',
            r'"grade"\s*:\s*"([^"]+)"',
            r'"response"\s*:\s*(\d+)',
            r'"grade"\s*:\s*(\d+)',
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
            # Final grade statement patterns
            r'(?:final|overall|total)\s+(?:grade|score|mark)[\s]*[:=\s]+["\']?([^"\'\n,]+)["\']?',
            r'(?:therefore|thus|hence|conclusion)[,:]?\s+(?:the\s+)?(?:grade|score|mark)[\s]*[:=\s]+["\']?([^"\'\n,]+)["\']?',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return _normalize_grade(match.group(1).strip())
        
        # Fallback: look for numeric grades (0, 1, 2) as standalone tokens
        # This handles cases where the model just outputs a number
        numeric_matches = re.findall(r'\b([0-2])\b', text)
        if numeric_matches:
            # Return the last numeric match (often the final grade)
            return _normalize_grade(numeric_matches[-1])
        
        # Last resort: look for the grade at the very end of the text
        # Sometimes models put the grade at the end after all reasoning
        last_lines = text.strip().split('\n')[-3:]  # Last 3 lines
        for line in reversed(last_lines):
            # Look for standalone numbers 0, 1, 2
            match = re.search(r'^\s*["\']?([0-2])["\']?\s*$', line.strip())
            if match:
                return match.group(1)
            # Look for Correct/Incorrect/Partial at the end
            match = re.search(r'\b(Correct|Incorrect|Partial)\b', line, re.IGNORECASE)
            if match:
                return _normalize_grade(match.group(1))
        
        return "None"

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with enhanced error handling.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        instruction = self._build_grading_prompt(inputs)
        
        # System message to guide the model's behavior
        system_msg = """You are an expert mathematical grader for IMO (International Mathematical Olympiad) problems. 
Your task is to evaluate student solutions and assign grades according to IMO standards.

You must ALWAYS respond in the exact JSON format specified in the prompt, with the grade in the "response" field.
The grade must be exactly one of: "0", "1", "2", "Correct", "Incorrect", or "Partial".

Be thorough in your analysis but concise in your final grade assignment."""

        try:
            response, msg_history, info = get_response_from_llm_with_system(
                msg=instruction,
                system_msg=system_msg,
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
            
            # Retry once with a clearer instruction if first attempt failed
            retry_instruction = instruction + "\n\n## IMPORTANT REMINDER:\nYour previous response did not contain a valid grade in the required format. Please ensure your response includes a JSON block with a 'response' field containing exactly one of: '0', '1', '2', 'Correct', 'Incorrect', or 'Partial'."
            
            try:
                retry_response, retry_msg_history, retry_info = get_response_from_llm_with_system(
                    msg=retry_instruction,
                    system_msg=system_msg,
                    model=self.model,
                    msg_history=[],
                )
                retry_prediction = self._extract_prediction(retry_msg_history)
                if retry_prediction != "None":
                    return str(retry_prediction), retry_msg_history
            except Exception as e2:
                self.log_fn(f"Retry LLM call failed: {e2}")

        return str(prediction), msg_history
