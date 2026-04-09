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
        'solved', 'solution correct', 'answer correct'
    ]
    incorrect_variations = [
        'incorrect', 'wrong', 'false', 'no', 'none', 'zero',
        'invalid', 'rejected', 'fail', 'error', 'mistake',
        'unsolved', 'not correct', 'not valid'
    ]
    partial_variations = [
        'partial', 'partial credit', 'half', 'incomplete',
        'partially correct', 'partially right', 'some credit',
        'partially solved', 'in progress'
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

Follow this structured evaluation process:

### Step 1: Problem Understanding
- What is the problem asking for?
- What are the key concepts and techniques needed?
- What is the expected answer format?

### Step 2: Solution Analysis
- Compare the student's approach to the official solution
- Identify any correct steps or valid alternative methods
- Note any errors, gaps, or misconceptions

### Step 3: Grading Decision
Based on the grading guidelines, assign one of these grades:
- **Correct**: The answer is fully correct with proper reasoning
- **Partial**: The answer has some correct elements but is incomplete or has minor errors
- **Incorrect**: The answer is wrong or shows fundamental misunderstanding

## Output Format:

Provide your evaluation in this exact format:

<evaluation>
### Analysis:
[Your detailed analysis of the student's work]

### Grade: [Correct/Partial/Incorrect]
</evaluation>

Be thorough in your analysis and strict in your grading according to the provided guidelines."""

    def _extract_from_tags(self, text: str) -> str | None:
        """Extract grade from <evaluation> tags."""
        import re
        match = re.search(r'<evaluation>.*?</evaluation>', text, re.DOTALL | re.IGNORECASE)
        if match:
            content = match.group(0)
            # Look for grade within the evaluation block
            grade_match = re.search(r'###?\s*Grade:\s*(\w+)', content, re.IGNORECASE)
            if grade_match:
                return _normalize_grade(grade_match.group(1))
        return None

    def _extract_prediction(self, msg_history: list[dict]) -> str:
        """Extract prediction from message history with multiple fallback strategies.
        
        Enhanced to handle more edge cases and provide better logging for debugging.
        Also searches through the entire message history if the last message doesn't contain a grade.
        """
        if not msg_history:
            self.log_fn("No message history available")
            return "None"
        
        # Try last message first
        last_message = msg_history[-1].get("text", "")
        if last_message:
            result = self._try_extract_grade(last_message)
            if result != "None":
                return result
        
        # If last message failed, search backwards through history
        self.log_fn("Last message had no grade, searching history...")
        for i, msg in enumerate(reversed(msg_history[:-1]), 1):
            text = msg.get("text", "")
            if text:
                result = self._try_extract_grade(text, silent=True)
                if result != "None":
                    self.log_fn(f"Found grade in message -{i} from end: {result}")
                    return result
        
        self.log_fn(f"Failed to extract grade from any message in history")
        return "None"
    
    def _try_extract_grade(self, text: str, silent: bool = False) -> str:
        """Try to extract grade from text using multiple strategies.
        
        Args:
            text: The text to search for a grade
            silent: If True, don't log extraction details
            
        Returns:
            The extracted grade or "None" if not found
        """
        # Strategy 1: Extract from <evaluation> tags (matches prompt format)
        eval_result = self._extract_from_tags(text)
        if eval_result and eval_result != "None":
            if not silent:
                self.log_fn(f"Extracted grade from <evaluation> tags: {eval_result}")
            return eval_result
        
        # Strategy 2: Extract from <json> tags
        extracted = _extract_jsons(text)
        if extracted:
            if not silent:
                self.log_fn(f"Extracted JSON from <json> tags: {extracted[-1]}")
            return self._get_grade_from_json(extracted[-1])
        
        # Strategy 3: Extract from markdown code blocks
        extracted = _extract_json_from_markdown(text)
        if extracted:
            if not silent:
                self.log_fn(f"Extracted JSON from markdown: {extracted[-1]}")
            return self._get_grade_from_json(extracted[-1])
        
        # Strategy 4: Look for grade patterns in plain text
        text_grade = self._extract_grade_from_text(text)
        if text_grade != "None":
            if not silent:
                self.log_fn(f"Extracted grade from text: {text_grade}")
            return text_grade
        
        # Strategy 5: Last resort - look for any numeric value that could be a grade
        numeric_grade = self._extract_numeric_grade(text)
        if numeric_grade != "None":
            if not silent:
                self.log_fn(f"Extracted numeric grade: {numeric_grade}")
            return numeric_grade
        
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
