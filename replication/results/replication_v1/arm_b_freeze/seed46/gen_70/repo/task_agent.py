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
    if grade is None:
        return "None"
    
    if not isinstance(grade, str):
        grade = str(grade)
    
    grade = grade.strip().lower()
    
    # Handle empty string
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
    
    # Map common variations to standard formats
    correct_variations = [
        'correct', 'right', 'true', 'yes', 'full', 'full credit', 
        'full marks', 'complete', 'valid', 'accepted', 'pass',
        'solved', 'solution correct', 'answer correct', 'success',
        'accurate', 'perfect', 'excellent', 'good'
    ]
    incorrect_variations = [
        'incorrect', 'wrong', 'false', 'no', 'none', 'zero',
        'invalid', 'rejected', 'fail', 'error', 'mistake',
        'unsolved', 'not correct', 'not valid', 'bad',
        'unacceptable', 'missing', 'empty', 'blank'
    ]
    partial_variations = [
        'partial', 'partial credit', 'half', 'incomplete',
        'partially correct', 'partially right', 'some credit',
        'partially solved', 'in progress', 'almost', 'nearly',
        'minor errors', 'small mistake', 'mostly correct'
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

        # Check for empty or whitespace-only student answer
        student_answer_stripped = student_answer.strip() if student_answer else ""
        is_empty_answer = not student_answer_stripped

        empty_answer_note = ""
        if is_empty_answer:
            empty_answer_note = """
⚠️ IMPORTANT: The student's answer appears to be empty or contains only whitespace. 
In such cases, the grade should typically be '0' (Incorrect) unless the grading 
guidelines specify otherwise for empty submissions.
"""

        return f"""You are an expert mathematical grader specializing in {domain} problems.

Your task is to evaluate a student's answer to a mathematics problem and assign a grade.

## Problem Statement:
{problem}

## Official Solution:
{solution}

## Grading Guidelines:
{grading_guidelines}
{empty_answer_note}

## Student's Answer:
{student_answer if student_answer_stripped else "[EMPTY ANSWER - No response provided by student]"}

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
- Is the answer complete or are there gaps in the reasoning?

### Step 4: Grade Assignment
Based on the grading guidelines, assign the appropriate grade considering:
- Correctness of the final answer
- Validity of the reasoning process
- Completeness of the solution
- Adherence to the expected solution method
- Presence of any errors or misconceptions

## Response Format:

You MUST respond in JSON format wrapped in <json> tags:

<json>
{{
    "reasoning": "Your detailed step-by-step analysis following the Evaluation Framework above",
    "response": "The final grade you assign (e.g., '0', '1', '2', 'Correct', 'Incorrect', 'Partial')"
}}
</json>

Important: 
- The "response" field must contain ONLY the grade value, nothing else.
- For IMO-style problems, use numeric grades (0, 1, 2, 3, etc.) as specified in the grading guidelines.
- For binary grading, use 'Correct' or 'Incorrect'.
- Use 'Partial' only when partial credit is explicitly allowed."""

    def _extract_prediction(self, msg_history: list[dict]) -> str:
        """Extract prediction from message history with multiple fallback strategies.
        
        Enhanced to handle more edge cases and provide better logging for debugging.
        """
        if not msg_history:
            self.log_fn("No message history available")
            return "None"
        
        last_message = msg_history[-1].get("text", "")
        if not last_message:
            self.log_fn("Last message is empty")
            return "None"
        
        # Strategy 1: Extract from <json> tags (preferred format)
        extracted = _extract_jsons(last_message)
        if extracted:
            self.log_fn(f"Extracted JSON from <json> tags: {extracted[-1]}")
            return self._get_grade_from_json(extracted[-1])
        
        # Strategy 2: Extract from markdown code blocks
        extracted = _extract_json_from_markdown(last_message)
        if extracted:
            self.log_fn(f"Extracted JSON from markdown: {extracted[-1]}")
            return self._get_grade_from_json(extracted[-1])
        
        # Strategy 3: Look for grade patterns in plain text
        text_grade = self._extract_grade_from_text(last_message)
        if text_grade != "None":
            self.log_fn(f"Extracted grade from text: {text_grade}")
            return text_grade
        
        # Strategy 4: Last resort - look for any numeric value that could be a grade
        numeric_grade = self._extract_numeric_grade(last_message)
        if numeric_grade != "None":
            self.log_fn(f"Extracted numeric grade: {numeric_grade}")
            return numeric_grade
        
        self.log_fn(f"Failed to extract grade from response: {last_message[:200]}...")
        return "None"

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
        
        Enhanced with multiple extraction strategies and fallback mechanisms
        to handle various response formats from different LLMs.
        """
        # Strategy 1: Look for explicit grade statements with various formats
        patterns = [
            # Standard grade assignment patterns
            r'grade[\s]*[:=][\s]*["\']?([^"\'\n]+)["\']?',
            r'response[\s]*[:=][\s]*["\']?([^"\'\n]+)["\']?',
            r'final grade[\s]*[:=][\s]*["\']?([^"\'\n]+)["\']?',
            r'assign[\s]+["\']?([^"\'\n]+)["\']?',
            # IMO-style numeric grades (0, 1, 2, 3, etc.)
            r'(?:score|points?|mark)[\s]*[:=][\s]*(\d+)',
            r'(?:the\s+)?grade\s+(?:is|of)\s+["\']?(\d+|[^"\'\n]+)["\']?',
            # Evaluation result patterns
            r'evaluation[\s]*[:=][\s]*["\']?([^"\'\n]+)["\']?',
            r'result[\s]*[:=][\s]*["\']?([^"\'\n]+)["\']?',
            # Answer/verdict patterns
            r'(?:the\s+)?(?:answer|verdict|decision)\s+(?:is\s+)?["\']?([^"\'\n]+)["\']?',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return _normalize_grade(match.group(1).strip())
        
        # Strategy 2: Look for standalone numeric grades at end of text
        # Common pattern: "The grade is 2" or just "2" at the end
        end_patterns = [
            r'(?:^|\n)\s*(\d+)\s*$',  # Standalone number at end
            r'grade\s*[:\-]?\s*(\d+)(?:\s|$)',  # Grade followed by number
            r'(?:score|mark)\s*[:\-]?\s*(\d+)(?:\s|$)',
        ]
        for pattern in end_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return _normalize_grade(match.group(1).strip())
        
        # Strategy 3: Look for explicit correctness indicators
        correctness_patterns = [
            (r'\b(correct|right|true|valid|accepted|full\s+credit)\b', 'Correct'),
            (r'\b(incorrect|wrong|false|invalid|rejected|no\s+credit)\b', 'Incorrect'),
            (r'\b(partial|partially\s+correct|partial\s+credit|half\s+credit)\b', 'Partial'),
        ]
        for pattern, grade in correctness_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return grade
        
        return "None"

    def _extract_numeric_grade(self, text: str) -> str:
        """Extract numeric grade as a last resort fallback.
        
        Looks for standalone numbers (0-7) that are likely IMO-style grades.
        Only returns a value if a clear numeric grade is found.
        """
        # Look for numbers 0-7 that appear to be grades
        # Common patterns in grading contexts
        patterns = [
            r'(?:^|\n|\s)([0-7])(?:\s*$|\s*\n)',  # Standalone number at end or on its own line
            r'grade\s*(?:is|:)?\s*([0-7])(?:\s|$|\.)',  # "grade is 2" or "grade: 2"
            r'score\s*(?:is|:)?\s*([0-7])(?:\s|$|\.)',  # "score is 2" or "score: 2"
            r'(?:^|\n)\s*([0-7])\s*points?\s*(?:$|\n)',  # "2 points" or "2 point"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                grade = match.group(1)
                return _normalize_grade(grade)
        
        return "None"

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with enhanced error handling.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Validate required inputs
        required_fields = ["domain", "problem", "solution", "grading_guidelines", "student_answer"]
        missing_fields = [f for f in required_fields if f not in inputs or not inputs.get(f)]
        if missing_fields:
            self.log_fn(f"Warning: Missing input fields: {missing_fields}")
        
        instruction = self._build_grading_prompt(inputs)
        
        # Log the problem being graded (truncated for readability)
        problem_preview = inputs.get("problem", "")[:100]
        self.log_fn(f"Grading problem: {problem_preview}...")

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
        else:
            self.log_fn(f"Successfully extracted prediction: {prediction}")

        return str(prediction), msg_history
