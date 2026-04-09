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
from collections import Counter

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
    
    IMO-style grading uses 0, 1, 2 as raw scores, but we normalize
    to semantic labels for consistency: 'Correct', 'Incorrect', 'Partial'.
    """
    if not isinstance(grade, str):
        grade = str(grade)
    
    original_grade = grade.strip()
    grade = original_grade.lower()
    
    # Handle edge cases with punctuation first
    grade_clean = grade.strip('.,;:!?"\'')
    
    # IMO numeric grades: 0 = Incorrect, 1 = Partial, 2 = Correct
    # Also handle 0.5 as Partial credit
    if grade_clean in ['0', '0.0']:
        return 'Incorrect'
    if grade_clean in ['1', '1.0', '0.5']:
        return 'Partial'
    if grade_clean in ['2', '2.0', '1.5']:
        return 'Correct'
    
    # Check for numeric grades (0, 1, 2, etc.) - IMO-style
    try:
        numeric_grade = float(grade_clean)
        if numeric_grade <= 0:
            return 'Incorrect'
        elif numeric_grade >= 1.5:
            return 'Correct'
        else:
            return 'Partial'
    except ValueError:
        pass
    
    # Map common variations to standard formats
    correct_variations = [
        'correct', 'right', 'true', 'yes', 'full', 'full credit', 
        'full marks', 'complete', 'accepted', 'pass',
        'solved', 'solution correct', 'answer correct', 'valid',
        'perfect', 'excellent', 'good', 'success'
    ]
    incorrect_variations = [
        'incorrect', 'wrong', 'false', 'no', 'none', 'zero',
        'invalid', 'rejected', 'fail', 'error', 'mistake',
        'unsolved', 'solution incorrect', 'answer incorrect',
        'bad', 'poor', 'unacceptable'
    ]
    partial_variations = [
        'partial', 'partial credit', 'half', 'incomplete',
        'partially correct', 'partially right', 'some credit',
        'partially solved', 'partial solution', 'half credit',
        'partially valid', 'incomplete but valid', 'minor errors'
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
Based on the grading guidelines, assign the appropriate grade using these definitions:

- **Correct** (IMO score: 2): The student's answer is fully correct with valid reasoning and complete solution.
- **Partial** (IMO score: 1): The student's answer has some correct elements but is incomplete or has minor errors.
- **Incorrect** (IMO score: 0): The student's answer is wrong, missing, or fundamentally flawed.

Consider:
- Correctness of the final answer
- Validity of the reasoning process
- Completeness of the solution
- Adherence to the expected solution method

## Response Format:

You MUST respond in JSON format wrapped in <json> tags:

<json>
{{
    "reasoning": "Your detailed step-by-step analysis following the Evaluation Framework above",
    "response": "One of: Correct, Partial, or Incorrect"
}}
</json>

Important: 
- The "response" field must contain ONLY one of these three values: "Correct", "Partial", or "Incorrect"
- Do not include any other text, numbers, or explanations in the "response" field
- Use "Partial" for answers that have some merit but are not fully correct"""

    def _validate_grade(self, grade: str) -> str:
        """Validate and normalize extracted grade to standard format.
        
        Returns the normalized grade if valid, or "None" if the grade
        cannot be validated. This ensures only valid grades are returned.
        """
        if not grade or grade == "None":
            return "None"
        
        stripped = grade.strip()
        
        # First check for raw numeric grades (0, 1, 2) - these are valid IMO grades
        # Map them to semantic labels for consistency
        if stripped == "0":
            return "Incorrect"
        if stripped == "1":
            return "Partial"
        if stripped == "2":
            return "Correct"
        
        # Normalize the grade
        normalized = _normalize_grade(grade)
        
        # Check if it's a valid standard grade
        valid_grades = ["Correct", "Incorrect", "Partial"]
        if normalized in valid_grades:
            return normalized
        
        # Try to extract numeric grades from the normalized value
        numeric_match = re.search(r'\b([0-2])\b', normalized.lower())
        if numeric_match:
            val = numeric_match.group(1)
            if val == "0":
                return "Incorrect"
            elif val == "1":
                return "Partial"
            elif val == "2":
                return "Correct"
        
        # Return "None" for unvalidated grades to force re-extraction
        return "None"

    def _extract_prediction(self, msg_history: list[dict]) -> str:
        """Extract prediction from message history with multiple fallback strategies.
        
        Uses a cascading approach: try structured formats first (JSON tags, 
        markdown), then fall back to pattern matching in plain text.
        Enhanced with cross-message validation for improved robustness.
        """
        if not msg_history:
            return "None"
        
        # Collect all assistant messages for cross-validation
        assistant_messages = []
        for msg in msg_history:
            if msg.get("role") == "assistant":
                text = msg.get("text", "")
                if text:
                    assistant_messages.append(text)
        
        if not assistant_messages:
            # Fallback to last message regardless of role
            last_text = msg_history[-1].get("text", "")
            if last_text:
                assistant_messages.append(last_text)
        
        if not assistant_messages:
            return "None"
        
        # Get the last message for primary extraction
        last_message = assistant_messages[-1]
        
        # Strategy 1: Extract from <json> tags (highest priority - structured format)
        extracted = _extract_jsons(last_message)
        if extracted:
            for json_obj in reversed(extracted):  # Try last JSON first
                grade = self._get_grade_from_json(json_obj)
                validated = self._validate_grade(grade)
                if validated != "None":
                    return validated
        
        # Strategy 2: Extract from markdown code blocks
        extracted = _extract_json_from_markdown(last_message)
        if extracted:
            for json_obj in reversed(extracted):  # Try last JSON first
                grade = self._get_grade_from_json(json_obj)
                validated = self._validate_grade(grade)
                if validated != "None":
                    return validated
        
        # Strategy 3: Look for grade patterns in plain text
        grade = self._extract_grade_from_text(last_message)
        validated = self._validate_grade(grade)
        if validated != "None":
            return validated
        
        # Strategy 4: Cross-message validation - check if multiple messages agree
        if len(assistant_messages) > 1:
            grades_found = []
            for msg_text in assistant_messages[-3:]:  # Check last 3 messages
                g = self._extract_grade_from_text(msg_text)
                v = self._validate_grade(g)
                if v != "None":
                    grades_found.append(v)
            
            # If multiple messages agree on a grade, use it
            if grades_found:
                from collections import Counter
                grade_counts = Counter(grades_found)
                most_common = grade_counts.most_common(1)[0]
                if most_common[1] >= 2:  # At least 2 messages agree
                    return most_common[0]
        
        # Strategy 5: Last resort - check if the entire message is just a grade
        clean_msg = last_message.strip().strip('"\'<>[]{}')
        validated = self._validate_grade(clean_msg)
        if validated != "None":
            return validated
        
        # Strategy 6: Check for grade in reasoning/thinking content
        # Sometimes the grade is embedded in the reasoning text
        reasoning_grade = self._extract_grade_from_reasoning(last_message)
        if reasoning_grade != "None":
            return reasoning_grade
        
        return "None"

    def _extract_grade_from_reasoning(self, text: str) -> str:
        """Extract grade from reasoning/thinking content.
        
        Sometimes models embed the grade decision within their reasoning text
        before providing the final JSON. This method extracts such grades.
        """
        if not text:
            return "None"
        
        text_lower = text.lower()
        
        # Look for conclusion patterns in reasoning
        conclusion_patterns = [
            r'(?:therefore|thus|hence|conclusion|conclude|final decision|i conclude)[,:]\s*(?:the\s+)?(?:grade|answer|result)\s*(?:is|should\s+be)\s*["\']?([^"\'\n,;]{3,30})["\']?',
            r'(?:based\s+on|given|considering)\s+(?:this|the\s+above)[,:]\s*(?:the\s+)?(?:grade|answer|result)\s*(?:is|should\s+be)\s*["\']?([^"\'\n,;]{3,30})["\']?',
        ]
        
        for pattern in conclusion_patterns:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                grade = _normalize_grade(match.group(1).strip())
                if grade in ["Correct", "Incorrect", "Partial"]:
                    return grade
        
        # Look for explicit decision statements near the end of the text
        lines = text.strip().split('\n')
        last_lines = lines[-5:] if len(lines) > 5 else lines
        last_text = ' '.join(last_lines).lower()
        
        # Check final lines for grade indicators
        if any(phrase in last_text for phrase in ['grade is correct', 'answer is correct', 'this is correct']):
            return "Correct"
        if any(phrase in last_text for phrase in ['grade is incorrect', 'answer is incorrect', 'this is incorrect', 'is wrong']):
            return "Incorrect"
        if any(phrase in last_text for phrase in ['grade is partial', 'partial credit', 'partially correct', 'some credit']):
            return "Partial"
        
        return "None"

    def _get_grade_from_json(self, json_obj: dict) -> str:
        """Extract grade from JSON object with field priority.
        
        Tries multiple field names in priority order, handling both
        string and numeric grade values.
        """
        if not isinstance(json_obj, dict):
            return "None"
        
        # Priority order for grade fields (most specific first)
        priority_fields = [
            "response",   # Primary field from our prompt
            "grade",      # Common alternative
            "evaluation", # Semantic alternative
            "result",     # Generic result field
            "score",      # Numeric score field
            "answer",     # Generic answer field
        ]
        
        # First pass: check priority fields
        for field in priority_fields:
            if field in json_obj:
                value = json_obj[field]
                if isinstance(value, str):
                    normalized = _normalize_grade(value)
                    if normalized != "None":
                        return normalized
                elif isinstance(value, (int, float)):
                    # Convert numeric to semantic label
                    if value == 0:
                        return "Incorrect"
                    elif value == 1:
                        return "Partial"
                    elif value >= 2:
                        return "Correct"
                    elif 0 < value < 1:
                        return "Partial"
                    else:
                        return "Incorrect"
        
        # Second pass: look for any string or numeric value that could be a grade
        for key, value in json_obj.items():
            if isinstance(value, str):
                normalized = _normalize_grade(value)
                if normalized in ["Correct", "Incorrect", "Partial"]:
                    return normalized
            elif isinstance(value, (int, float)):
                # Only accept small integers (0, 1, 2) as potential grades
                if value in [0, 1, 2]:
                    if value == 0:
                        return "Incorrect"
                    elif value == 1:
                        return "Partial"
                    else:
                        return "Correct"
        
        return "None"

    def _extract_grade_from_text(self, text: str) -> str:
        """Extract grade from plain text using pattern matching.
        
        Enhanced with additional patterns for robust extraction from
        various response formats including IMO-style numeric grades.
        Uses a multi-layered approach with priority ordering.
        """
        if not text or not text.strip():
            return "None"
            
        text_stripped = text.strip()
        text_lower = text_stripped.lower()
        
        # Layer 1: JSON-like field patterns (highest priority)
        json_patterns = [
            r'"response"\s*:\s*"([^"]+)"',
            r'"grade"\s*:\s*"([^"]+)"',
            r'"evaluation"\s*:\s*"([^"]+)"',
            r'"result"\s*:\s*"([^"]+)"',
            r'"response"\s*:\s*(\d+(?:\.\d+)?)',
            r'"grade"\s*:\s*(\d+(?:\.\d+)?)',
            r'"score"\s*:\s*(\d+(?:\.\d+)?)',
        ]
        
        for pattern in json_patterns:
            match = re.search(pattern, text_stripped, re.IGNORECASE)
            if match:
                grade = _normalize_grade(match.group(1).strip())
                if grade != "None":
                    return grade
        
        # Layer 2: Explicit grade assignment statements
        assignment_patterns = [
            r'(?:final\s+)?grade\s*[:=]\s*["\']?([^"\'\n,;]+)["\']?',
            r'(?:final\s+)?score\s*[:=]\s*["\']?([^"\'\n,;]+)["\']?',
            r'response\s*[:=]\s*["\']?([^"\'\n,;]+)["\']?',
            r'(?:I\s+)?(?:assign|give|award)\s*["\']?([^"\'\n,;]{1,20})["\']?',
        ]
        
        for pattern in assignment_patterns:
            match = re.search(pattern, text_stripped, re.IGNORECASE)
            if match:
                grade = _normalize_grade(match.group(1).strip())
                if grade != "None":
                    return grade
        
        # Layer 3: IMO-style numeric grade statements
        imo_patterns = [
            r'(?:the\s+)?(?:student\s+(?:receives?|gets?|earns?)|I\s+(?:assign|give|award))\s+["\']?(\d+(?:\.\d+)?)["\']?\s*(?:points?|marks?)?',
            r'(?:score|grade|mark)\s+of\s+["\']?(\d+(?:\.\d+)?)["\']?',
            r'(?:score|grade|mark|result)\s+is\s+["\']?(\d+(?:\.\d+)?)["\']?',
        ]
        
        for pattern in imo_patterns:
            match = re.search(pattern, text_stripped, re.IGNORECASE)
            if match:
                return _normalize_grade(match.group(1).strip())
        
        # Layer 4: Standalone numeric grades (0, 1, 2, 0.5, 1.5, 2.0)
        # Look for these at the end of the text (often the final conclusion)
        standalone_numeric = re.findall(r'\b([0-2](?:\.0|\.5)?)\b', text_stripped)
        if standalone_numeric:
            # Return the last numeric match (often the final grade)
            return _normalize_grade(standalone_numeric[-1])
        
        # Layer 5: Explicit semantic grade keywords
        semantic_patterns = [
            r'\b(Correct|Incorrect|Partial|Right|Wrong|Pass|Fail)\b',
            r'(?:the\s+)?(?:answer|solution)\s+(?:is|should\s+be)\s+["\']?([^"\'\n,;]{3,20})["\']?',
        ]
        
        for pattern in semantic_patterns:
            match = re.search(pattern, text_stripped, re.IGNORECASE)
            if match:
                grade = _normalize_grade(match.group(1).strip())
                if grade != "None":
                    return grade
        
        # Layer 6: Contextual indicators (lowest priority)
        # Check for partial credit indicators
        partial_indicators = [
            'partial credit', 'partially correct', 'half credit',
            'some credit', 'incomplete but', 'partial solution',
            'partially right', 'partially solved', 'partial marks',
            'partially valid', 'incomplete solution', 'minor errors'
        ]
        for indicator in partial_indicators:
            if indicator in text_lower:
                return "Partial"
        
        # Check for explicit incorrect indicators
        incorrect_indicators = [
            'is incorrect', 'is wrong', 'not correct', 'not right',
            'answer is wrong', 'solution is wrong', 'does not match',
            'failed to', 'did not solve', 'incorrect answer',
            'incorrect solution', 'is not correct', 'is not right'
        ]
        for indicator in incorrect_indicators:
            if indicator in text_lower:
                return "Incorrect"
        
        # Check for explicit correct indicators
        correct_indicators = [
            'is correct', 'is right', 'correct answer', 'correct solution',
            'fully correct', 'completely correct', 'answer is correct',
            'solution is correct', 'perfect solution', 'valid solution'
        ]
        for indicator in correct_indicators:
            if indicator in text_lower:
                return "Correct"
        
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
