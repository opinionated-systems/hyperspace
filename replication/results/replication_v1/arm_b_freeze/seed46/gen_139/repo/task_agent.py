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
        'full marks', 'complete', 'valid', 'accepted', 'pass', 'solved',
        'success', 'accurate', 'perfect', 'excellent', 'good'
    ]
    incorrect_variations = [
        'incorrect', 'wrong', 'false', 'no', 'none', 'zero',
        'invalid', 'rejected', 'fail', 'error', 'mistake', 'unsolved',
        'failure', 'bad', 'unsatisfactory', 'not correct', 'not right'
    ]
    partial_variations = [
        'partial', 'partial credit', 'half', 'incomplete',
        'partially correct', 'partially right', 'some credit',
        'partially solved', 'almost', 'nearly', 'mostly'
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

## Grade Categories:

Use these standard grade values:
- **Correct** or **1+**: The answer is fully correct with valid reasoning
- **Incorrect** or **0**: The answer is wrong or has critical flaws
- **Partial**: The answer has some correct elements but is incomplete or has errors

## Response Format:

You MUST respond in JSON format wrapped in <json> tags:

<json>
{{
    "reasoning": "Your detailed step-by-step analysis following the Evaluation Framework above. Be thorough and specific about what the student did right and wrong.",
    "response": "The final grade you assign - use exactly one of: 'Correct', 'Incorrect', or 'Partial' (or numeric equivalents like '0', '1', '2')"
}}
</json>

Important: 
- The "response" field must contain ONLY the grade value
- Be consistent with the grading guidelines provided
- If numeric grades are specified in the guidelines, use those values"""

    def _extract_prediction(self, msg_history: list[dict]) -> str:
        """Extract prediction from message history with multiple fallback strategies."""
        if not msg_history:
            self.log_fn("No message history available for prediction extraction")
            return "None"
        
        last_message = msg_history[-1].get("text", "")
        if not last_message:
            self.log_fn("Last message has no text content")
            return "None"
        
        # Strategy 1: Extract from <json> tags
        extracted = _extract_jsons(last_message)
        if extracted:
            self.log_fn(f"Extracted JSON from <json> tags: {len(extracted)} objects")
            result = self._get_grade_from_json(extracted[-1])
            if result != "None":
                self.log_fn(f"Successfully extracted grade from JSON: {result}")
                return result
        
        # Strategy 2: Extract from markdown code blocks
        extracted = _extract_json_from_markdown(last_message)
        if extracted:
            self.log_fn(f"Extracted JSON from markdown: {len(extracted)} objects")
            result = self._get_grade_from_json(extracted[-1])
            if result != "None":
                self.log_fn(f"Successfully extracted grade from markdown: {result}")
                return result
        
        # Strategy 3: Look for grade patterns in plain text
        self.log_fn("Falling back to plain text extraction")
        result = self._extract_grade_from_text(last_message)
        if result != "None":
            self.log_fn(f"Successfully extracted grade from text: {result}")
            return result
        
        # Strategy 4: Direct keyword search for common grade values
        text_lower = last_message.lower()
        if any(word in text_lower for word in ['correct', 'right', 'true', 'valid', 'accepted', 'solved', 'accurate']):
            self.log_fn("Extracted grade 'Correct' from keyword match")
            return 'Correct'
        elif any(word in text_lower for word in ['incorrect', 'wrong', 'false', 'invalid', 'rejected', 'error', 'unsolved']):
            self.log_fn("Extracted grade 'Incorrect' from keyword match")
            return 'Incorrect'
        elif any(word in text_lower for word in ['partial', 'incomplete', 'half', 'partially']):
            self.log_fn("Extracted grade 'Partial' from keyword match")
            return 'Partial'
        
        # Strategy 5: Check if the entire response is just a grade
        stripped = last_message.strip().lower()
        if stripped in ['correct', 'incorrect', 'partial', '0', '1', '2', '3', '4', '5', '6', '7']:
            self.log_fn(f"Extracted grade '{stripped}' from direct match")
            return _normalize_grade(stripped)
        
        self.log_fn("Failed to extract grade from any source")
        return "None"

    def _get_grade_from_json(self, json_obj: dict) -> str:
        """Extract grade from JSON object with field priority."""
        if not isinstance(json_obj, dict):
            self.log_fn(f"Invalid JSON object type: {type(json_obj)}")
            return "None"
        
        # Priority order for grade fields
        priority_fields = ["response", "grade", "answer", "result", "score", "evaluation", "verdict", "decision"]
        
        for field in priority_fields:
            if field in json_obj:
                value = json_obj[field]
                self.log_fn(f"Found grade field '{field}' with value: {value}")
                if isinstance(value, str):
                    normalized = _normalize_grade(value)
                    if normalized != value.strip().lower():
                        return normalized
                    return value
                elif isinstance(value, (int, float)):
                    return str(value)
                elif isinstance(value, bool):
                    return "Correct" if value else "Incorrect"
        
        # If no recognized field, use the first string or numeric value found
        for key, value in json_obj.items():
            if isinstance(value, str):
                self.log_fn(f"Using first string value from field '{key}': {value}")
                normalized = _normalize_grade(value)
                if normalized != value.strip().lower():
                    return normalized
                return value
            elif isinstance(value, (int, float)):
                self.log_fn(f"Using first numeric value from field '{key}': {value}")
                return str(value)
            elif isinstance(value, bool):
                self.log_fn(f"Using first boolean value from field '{key}': {value}")
                return "Correct" if value else "Incorrect"
        
        self.log_fn(f"No valid grade found in JSON object with keys: {list(json_obj.keys())}")
        return "None"

    def _extract_grade_from_text(self, text: str) -> str:
        """Extract grade from plain text using pattern matching.
        
        Enhanced with multiple extraction strategies and fallback mechanisms
        to handle various response formats from different LLMs.
        """
        # Strategy 1: Look for explicit grade statements with various formats
        patterns = [
            # Standard grade assignment patterns
            (r'grade[\s]*[:=][\s]*["\']?([^"\'\n]+)["\']?', 'grade assignment'),
            (r'response[\s]*[:=][\s]*["\']?([^"\'\n]+)["\']?', 'response assignment'),
            (r'final grade[\s]*[:=][\s]*["\']?([^"\'\n]+)["\']?', 'final grade'),
            (r'assign[\s]+["\']?([^"\'\n]+)["\']?', 'assign statement'),
            # IMO-style numeric grades (0, 1, 2, 3, etc.)
            (r'(?:score|points?|mark)[\s]*[:=][\s]*(\d+)', 'score/mark assignment'),
            (r'(?:the\s+)?grade\s+(?:is|of)\s+["\']?(\d+|[^"\'\n]+)["\']?', 'grade is/of'),
            # Evaluation result patterns
            (r'evaluation[\s]*[:=][\s]*["\']?([^"\'\n]+)["\']?', 'evaluation assignment'),
            (r'result[\s]*[:=][\s]*["\']?([^"\'\n]+)["\']?', 'result assignment'),
            # Answer/verdict patterns
            (r'(?:the\s+)?(?:answer|verdict|decision)\s+(?:is\s+)?["\']?([^"\'\n]+)["\']?', 'answer/verdict'),
            # Additional patterns for common LLM outputs
            (r'(?:therefore|thus|hence|so)[,\s]+(?:the\s+)?grade\s+(?:is|should\s+be)\s+["\']?([^"\'\n]+)["\']?', 'conclusion grade'),
            (r'(?:i\s+)?(?:assign|give|award)[\s]+["\']?([^"\'\n]+)["\']?', 'assign/give/award'),
        ]
        
        for pattern, pattern_name in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                grade = _normalize_grade(match.group(1).strip())
                self.log_fn(f"Extracted grade '{grade}' using pattern: {pattern_name}")
                return grade
        
        # Strategy 2: Look for standalone numeric grades at end of text
        # Common pattern: "The grade is 2" or just "2" at the end
        end_patterns = [
            (r'(?:^|\n)\s*(\d+)\s*$', 'standalone number at end'),
            (r'grade\s*[:\-]?\s*(\d+)(?:\s|$)', 'grade with number'),
            (r'(?:score|mark)\s*[:\-]?\s*(\d+)(?:\s|$)', 'score/mark with number'),
            (r'(?:^|\n)\s*["\']?(\d+)["\']?\s*$', 'quoted number at end'),
        ]
        for pattern, pattern_name in end_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                grade = _normalize_grade(match.group(1).strip())
                self.log_fn(f"Extracted grade '{grade}' using pattern: {pattern_name}")
                return grade
        
        # Strategy 3: Look for explicit correctness indicators
        correctness_patterns = [
            (r'\b(correct|right|true|valid|accepted|full\s+credit|solved)\b', 'Correct', 'correctness indicator'),
            (r'\b(incorrect|wrong|false|invalid|rejected|no\s+credit|unsolved)\b', 'Incorrect', 'incorrectness indicator'),
            (r'\b(partial|partially\s+correct|partial\s+credit|half\s+credit|partially\s+solved)\b', 'Partial', 'partial indicator'),
        ]
        for pattern, grade, pattern_name in correctness_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                self.log_fn(f"Extracted grade '{grade}' using pattern: {pattern_name}")
                return grade
        
        self.log_fn("No grade patterns matched in text")
        return "None"

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with enhanced error handling.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Log input information for debugging
        domain = inputs.get("domain", "unknown")
        self.log_fn(f"Starting grading for domain: {domain}")
        
        instruction = self._build_grading_prompt(inputs)

        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
            self.log_fn(f"LLM call successful, response length: {len(response) if response else 0} chars")
        except Exception as e:
            self.log_fn(f"LLM call failed: {e}")
            return "None", []

        # Extract prediction using multiple strategies
        prediction = self._extract_prediction(msg_history)
        
        if prediction == "None":
            self.log_fn(f"Failed to extract prediction from response: {response[:200] if response else 'empty'}")
        else:
            self.log_fn(f"Final prediction: {prediction}")

        return str(prediction), msg_history
