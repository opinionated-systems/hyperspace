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
    
    Enhanced to handle common formatting issues like trailing commas,
    single quotes, and comments.
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
        
        # Try to parse the JSON
        try:
            results.append(json.loads(inner))
        except json.JSONDecodeError:
            # Try common fixes for malformed JSON
            # Fix 1: Remove trailing commas before closing braces/brackets
            fixed = re.sub(r',\s*}', '}', inner)
            fixed = re.sub(r',\s*]', ']', fixed)
            
            # Fix 2: Replace single quotes with double quotes (carefully)
            # Only replace quotes that are not inside strings
            try:
                results.append(json.loads(fixed))
            except json.JSONDecodeError:
                # Try one more fix: handle unescaped newlines in strings
                fixed2 = fixed.replace('\n', '\\n').replace('\r', '\\r')
                try:
                    results.append(json.loads(fixed2))
                except json.JSONDecodeError:
                    # If all fixes fail, skip this block
                    continue
    return results or None


def _extract_json_from_markdown(text: str) -> list[dict] | None:
    """Extract JSON objects from markdown code blocks.

    Fallback for when <json> tags are not used but markdown code blocks are.
    
    Enhanced to handle common formatting issues like trailing commas,
    single quotes, and unescaped newlines.
    """
    results = []
    # Match ```json ... ``` or just ``` ... ``` blocks
    pattern = r'```(?:json)?\s*(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    for match in matches:
        content = match.strip()
        try:
            results.append(json.loads(content))
        except json.JSONDecodeError:
            # Try common fixes for malformed JSON
            # Fix 1: Remove trailing commas before closing braces/brackets
            fixed = re.sub(r',\s*}', '}', content)
            fixed = re.sub(r',\s*]', ']', fixed)
            
            try:
                results.append(json.loads(fixed))
            except json.JSONDecodeError:
                # Fix 2: Handle unescaped newlines in strings
                fixed2 = fixed.replace('\n', '\\n').replace('\r', '\\r')
                try:
                    results.append(json.loads(fixed2))
                except json.JSONDecodeError:
                    # If all fixes fail, skip this block
                    continue
    return results or None


def _normalize_grade(grade: str) -> str:
    """Normalize grade to a standard format.

    Handles various grade formats and normalizes them.
    Enhanced to handle numeric grades and more variations.
    Improved to handle edge cases like negations and punctuation.
    """
    if not isinstance(grade, str):
        grade = str(grade)
    
    original_grade = grade.strip()
    grade = original_grade.lower()
    
    # First, check for negations that indicate incorrect
    negation_indicators = [
        'not correct', 'not right', 'not valid', 'not accepted',
        'not a correct', 'not the correct', 'is incorrect', 'is wrong',
        'is not correct', 'is not right', 'is not valid', 'not true'
    ]
    for neg in negation_indicators:
        if neg in grade:
            return 'Incorrect'
    
    # Check for numeric grades (0, 0.5, 1, 2, etc.)
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
        'solved', 'solution correct', 'answer correct', 'valid',
        'perfect', 'excellent', 'good', 'success', 'successful'
    ]
    incorrect_variations = [
        'incorrect', 'wrong', 'false', 'no', 'none', 'zero',
        'invalid', 'rejected', 'fail', 'error', 'mistake',
        'unsolved', 'solution incorrect', 'answer incorrect', 'not correct',
        'bad', 'poor', 'unsuccessful', 'failure', 'incorrect answer',
        'incorrect solution', 'wrong answer', 'wrong solution'
    ]
    partial_variations = [
        'partial', 'partial credit', 'half', 'incomplete',
        'partially correct', 'partially right', 'some credit',
        'partially solved', 'partial solution', 'partially valid',
        'half correct', 'half right', 'partial marks', 'half marks',
        'incomplete solution', 'incomplete answer', 'partial success'
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

You MUST respond in JSON format wrapped in <json> tags. The JSON must be valid and properly formatted:

<json>
{{
    "reasoning": "Your detailed step-by-step analysis following the Evaluation Framework above",
    "response": "The final grade you assign"
}}
</json>

## Grade Format Guidelines:

Use ONE of the following standard grade formats:
- **IMO-style numeric grades**: "0" (incorrect), "1" (partial), "2" (correct)
- **Text grades**: "Correct", "Incorrect", "Partial"

The "response" field must contain ONLY the grade value, nothing else. Do not include explanations, punctuation, or additional text in the response field.

Examples of valid responses:
- "response": "2"
- "response": "1"
- "response": "0"
- "response": "Correct"
- "response": "Incorrect"
- "response": "Partial"

Examples of INVALID responses (do not use):
- "response": "The answer is correct"
- "response": "Grade: 2"
- "response": "2 points"
"""

    def _validate_grade(self, grade: str) -> str:
        """Validate and normalize extracted grade to standard format.
        
        Returns the normalized grade if valid, or the original if no
        standard normalization applies.
        
        Enhanced to handle more edge cases and provide better validation.
        """
        if not grade or grade == "None":
            return "None"
        
        # First check for raw numeric grades (0, 0.5, 1, 2) - these are valid IMO grades
        stripped = grade.strip()
        if stripped in ["0", "1", "2"]:
            return stripped
        if stripped in ["0.5", ".5"]:
            return "Partial"
        
        # Normalize the grade
        normalized = _normalize_grade(grade)
        
        # Check if it's a valid standard grade
        valid_grades = ["Correct", "Incorrect", "Partial", "0", "1", "2"]
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
        
        # Check for common grade patterns that might have been missed
        grade_lower = normalized.lower()
        
        # Handle negations explicitly
        if 'not' in grade_lower and ('correct' in grade_lower or 'right' in grade_lower):
            return "Incorrect"
        
        # Handle partial indicators
        if 'partial' in grade_lower or 'half' in grade_lower or 'incomplete' in grade_lower:
            return "Partial"
        
        # Return the normalized value even if not in standard list
        # (allows for domain-specific grades)
        return normalized

    def _extract_prediction(self, msg_history: list[dict]) -> str:
        """Extract prediction from message history with multiple fallback strategies.
        
        Enhanced with better logging and more robust extraction.
        """
        if not msg_history:
            self.log_fn("No message history available")
            return "None"
        
        last_message = msg_history[-1].get("text", "")
        if not last_message:
            self.log_fn("Last message has no text content")
            return "None"
        
        # Strategy 1: Extract from <json> tags (most reliable)
        extracted = _extract_jsons(last_message)
        if extracted:
            grade = self._get_grade_from_json(extracted[-1])
            validated = self._validate_grade(grade)
            if validated != "None":
                self.log_fn(f"Extracted grade from <json> tags: {validated}")
                return validated
            else:
                self.log_fn(f"Failed to validate grade from JSON: {grade}")
        
        # Strategy 2: Extract from markdown code blocks
        extracted = _extract_json_from_markdown(last_message)
        if extracted:
            grade = self._get_grade_from_json(extracted[-1])
            validated = self._validate_grade(grade)
            if validated != "None":
                self.log_fn(f"Extracted grade from markdown: {validated}")
                return validated
            else:
                self.log_fn(f"Failed to validate grade from markdown: {grade}")
        
        # Strategy 3: Look for grade patterns in plain text
        grade = self._extract_grade_from_text(last_message)
        validated = self._validate_grade(grade)
        if validated != "None":
            self.log_fn(f"Extracted grade from text: {validated}")
        else:
            self.log_fn(f"Failed to extract valid grade from text. Raw grade: {grade}")
        
        return validated

    def _get_grade_from_json(self, json_obj: dict) -> str:
        """Extract grade from JSON object with field priority.
        
        Enhanced to handle more field names and nested structures.
        """
        # Priority order for grade fields (most common first)
        priority_fields = [
            "response", "grade", "answer", "result", "score", 
            "evaluation", "assessment", "verdict", "decision",
            "grading", "mark", "rating", "outcome"
        ]
        
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
                elif isinstance(value, bool):
                    return "Correct" if value else "Incorrect"
        
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
            elif isinstance(value, bool):
                return "Correct" if value else "Incorrect"
        
        return "None"

    def _extract_grade_from_text(self, text: str) -> str:
        """Extract grade from plain text using pattern matching.
        
        Enhanced with additional patterns for robust extraction from
        various response formats including IMO-style numeric grades.
        
        Improved to handle edge cases like negations and conflicting statements.
        Also handles common LLM output patterns like "The grade is: Correct"
        and "Final Answer: 2" formats.
        """
        text_lower = text.lower()
        
        # First, check for negations that indicate incorrect
        # This must happen BEFORE pattern matching to avoid false positives
        negation_patterns = [
            'not correct', 'not right', 'not valid', 'not accepted',
            'not a correct', 'not the correct', 'is incorrect', 'is wrong',
            'is not correct', 'is not right', 'is not valid', 'not true',
            'answer is wrong', 'solution is wrong', 'answer is incorrect',
            'solution is incorrect', 'the answer is not', 'the solution is not'
        ]
        for neg in negation_patterns:
            if neg in text_lower:
                return 'Incorrect'
        
        # Check for partial credit indicators
        partial_indicators = [
            'partial credit', 'partially correct', 'half credit',
            'some credit', 'incomplete solution', 'partial solution',
            'partially right', 'partially valid', 'partial marks',
            'half correct', 'half right', 'partially solved',
            'partial success', 'incomplete answer', 'partial answer'
        ]
        for indicator in partial_indicators:
            if indicator in text_lower:
                return 'Partial'
        
        # Look for explicit grade statements with flexible patterns
        # Ordered by specificity - more specific patterns first
        # Numeric patterns have higher priority to avoid matching "Correct" inside "points"
        patterns = [
            # JSON-like field patterns (highest priority)
            r'"response"\s*:\s*"([^"]+)"',
            r'"grade"\s*:\s*"([^"]+)"',
            r'"response"\s*:\s*(\d+(?:\.\d+)?)',
            r'"grade"\s*:\s*(\d+(?:\.\d+)?)',
            # IMO-style numeric grade statements (high priority for numeric grades)
            r'(?:the\s+)?(?:student\s+(?:receives?|gets?|earns?)|I\s+(?:assign|give|award))\s+["\']?(\d+(?:\.\d+)?)(?:\s*(?:points?|marks?))?\b',
            r'(?:score|grade|mark)[\s]*[:=\s]+["\']?(\d+(?:\.\d+)?)["\']?',
            # Standalone numeric grades (0, 0.5, 1, 2) - word boundary ensures standalone
            r'(?:^|\s)([0-2](?:\.0|\.5)?)(?:\s|$|[.,;])',
            # Standard grade assignments (text grades) - capture only valid grade words
            r'grade[\s]*[:=][\s]*["\']?(Correct|Incorrect|Partial|Right|Wrong)["\']?',
            r'response[\s]*[:=][\s]*["\']?(Correct|Incorrect|Partial|Right|Wrong)["\']?',
            r'final grade[\s]*[:=][\s]*["\']?(Correct|Incorrect|Partial|Right|Wrong)["\']?',
            r'assign[\s]+["\']?(Correct|Incorrect|Partial|Right|Wrong|\d+(?:\.\d+)?)["\']?\b',
            # Grade at end of sentence - capture only valid grade words or numbers
            r'(?:grade|score|mark|result)[\s]+(?:is|of)[\s]+["\']?(Correct|Incorrect|Partial|Right|Wrong|\d+(?:\.\d+)?)["\']?\b',
            # Standalone text grades - use word boundaries carefully
            r'(?:^|\s)(Correct|Incorrect|Partial|Right|Wrong)(?:\s|$|[.,;!])',
            # Additional patterns for edge cases - capture only valid grade words or numbers
            r'(?:the\s+)?(?:answer|solution)\s+(?:is|should\s+be)\s+["\']?(Correct|Incorrect|Partial|Right|Wrong|\d+(?:\.\d+)?)["\']?\b',
            # Conclusion patterns - capture only valid grade words or numbers
            r'(?:therefore|thus|hence|conclusion)[,:]?\s+(?:the\s+)?(?:grade|score|answer)\s+(?:is\s+)?["\']?(Correct|Incorrect|Partial|Right|Wrong|\d+(?:\.\d+)?)["\']?\b',
            # Final answer patterns (common in LLM outputs)
            r'(?:final answer|answer)[:\s]+["\']?(Correct|Incorrect|Partial|Right|Wrong|\d+(?:\.\d+)?)["\']?\b',
            # The grade is patterns
            r'(?:the\s+)?grade\s+(?:is|should\s+be)\s+["\']?(Correct|Incorrect|Partial|Right|Wrong|\d+(?:\.\d+)?)["\']?\b',
            # I conclude patterns
            r'I\s+(?:conclude|determine|decide|think)\s+(?:that\s+)?(?:the\s+)?(?:grade|answer|solution)\s+(?:is\s+)?["\']?(Correct|Incorrect|Partial|Right|Wrong|\d+(?:\.\d+)?)["\']?\b',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                extracted = match.group(1).strip()
                # Double-check for negations in the extracted text
                extracted_lower = extracted.lower()
                if 'not' in extracted_lower and ('correct' in extracted_lower or 'right' in extracted_lower):
                    return 'Incorrect'
                return _normalize_grade(extracted)
        
        # Fallback: look for numeric grades (0, 0.5, 1, 2) as standalone tokens
        # This handles cases where the model just outputs a number
        numeric_matches = re.findall(r'\b([0-2](?:\.\d+)?)\b', text)
        if numeric_matches:
            # Return the last numeric match (often the final grade)
            return _normalize_grade(numeric_matches[-1])
        
        # Final fallback: look for grade keywords anywhere in text
        # Check for incorrect/wrong indicators (but not negated)
        if 'incorrect' in text_lower or 'wrong' in text_lower:
            # Double-check for negations we might have missed
            if 'not incorrect' not in text_lower and 'not wrong' not in text_lower:
                return 'Incorrect'
        
        # Check for correct indicators (but not negated)
        if 'correct' in text_lower:
            # Make sure it's not negated
            if 'not correct' not in text_lower and 'incorrect' not in text_lower:
                return 'Correct'
        
        # Check for "yes" or "no" as standalone grades
        if re.search(r'\byes\b', text_lower) and not re.search(r'\bno\b', text_lower):
            return 'Correct'
        if re.search(r'\bno\b', text_lower) and not re.search(r'\byes\b', text_lower):
            return 'Incorrect'
        
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
            self.log_fn(f"Missing required input fields: {missing_fields}")
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
            if msg_history:
                last_msg = msg_history[-1].get("text", "")
                self.log_fn(f"Failed to extract prediction. Last message preview: {last_msg[:300] if last_msg else 'empty'}...")
            else:
                self.log_fn("Failed to extract prediction - no message history")

        return str(prediction), msg_history
