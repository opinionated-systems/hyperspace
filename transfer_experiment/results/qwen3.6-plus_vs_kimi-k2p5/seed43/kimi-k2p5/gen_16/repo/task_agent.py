"""
Task agent: solves a given task with a single LLM call.

Reimplemented from facebookresearch/HyperAgents task_agent.py.
Same interface, same JSON output format, same extraction logic.

This is the IMPROVED task agent with better prompting and robust extraction.
The meta agent modifies this file during self-improvement. The evaluation harness 
loads whatever task_agent.py exists at the agent's repo path.
"""

from __future__ import annotations

import json
import logging
import re

from agent.llm_client import get_response_from_llm, EVAL_MODEL

logger = logging.getLogger(__name__)


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks."""
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


def _extract_json_flexible(text: str) -> dict | None:
    """Extract JSON using multiple fallback methods with improved robustness."""
    # Try <json> tags first
    results = _extract_jsons(text)
    if results:
        return results[-1]
    
    # Try ```json code blocks (with or without language specifier)
    json_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    matches = re.findall(json_block_pattern, text, re.DOTALL)
    for match in matches:
        try:
            return json.loads(match.strip())
        except json.JSONDecodeError:
            continue
    
    # Try single backtick code blocks
    single_tick_pattern = r'`([^`]+)`'
    matches = re.findall(single_tick_pattern, text, re.DOTALL)
    for match in matches:
        try:
            return json.loads(match.strip())
        except json.JSONDecodeError:
            continue
    
    # Try finding JSON-like objects with regex pattern for common response formats
    json_like_pattern = r'\{\s*"(?:response|grade|result|evaluation)"\s*:\s*"[^"]+"\s*\}'
    matches = re.findall(json_like_pattern, text, re.IGNORECASE)
    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue
    
    # Try finding raw JSON objects with curly braces (improved version with proper string handling)
    brace_count = 0
    start_idx = -1
    in_string = False
    escape_next = False
    
    for i, char in enumerate(text):
        if escape_next:
            escape_next = False
            continue
        if char == '\\' and in_string:
            escape_next = True
            continue
        if char == '"' and (i == 0 or text[i-1] != '\\'):
            in_string = not in_string
            continue
        if not in_string:
            if char == '{':
                if brace_count == 0:
                    start_idx = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start_idx != -1:
                    try:
                        return json.loads(text[start_idx:i+1])
                    except json.JSONDecodeError:
                        continue
    
    return None


def _extract_grade_from_text(text: str) -> str | None:
    """Extract grade from raw text using pattern matching with improved accuracy."""
    text_lower = text.lower()
    
    # Look for explicit grade patterns (ordered by specificity)
    patterns = [
        # JSON-style patterns (highest priority)
        r'"response"\s*:\s*"(correct|incorrect|partial)"',
        r'"grade"\s*:\s*"(correct|incorrect|partial)"',
        r'"result"\s*:\s*"(correct|incorrect|partial)"',
        r'"evaluation"\s*:\s*"(correct|incorrect|partial)"',
        # Assignment patterns
        r'grade\s*[:=]\s*"(correct|incorrect|partial)"',
        r'grade\s*[:=]\s*(correct|incorrect|partial)\b',
        r'response\s*[:=]\s*"(correct|incorrect|partial)"',
        r'response\s*[:=]\s*(correct|incorrect|partial)\b',
        # Statement patterns
        r'\bthe answer is\s+(correct|incorrect|partial)\b',
        r'\bthe grade is\s+(correct|incorrect|partial)\b',
        r'\bthe response is\s+(correct|incorrect|partial)\b',
        r'\bthe result is\s+(correct|incorrect|partial)\b',
        r'\bevaluation[\s:]+(correct|incorrect|partial)\b',
        # Additional patterns for common LLM response formats
        r'\bgrade[d]?\s+(correct|incorrect|partial)\b',
        r'\bmarked as\s+(correct|incorrect|partial)\b',
        r'\bconsidered\s+(correct|incorrect|partial)\b',
        r'\bdetermined to be\s+(correct|incorrect|partial)\b',
        r'\bassessment[\s:]+(correct|incorrect|partial)\b',
        r'\bverdict[\s:]+(correct|incorrect|partial)\b',
        # Standalone grades (lowest priority, require word boundaries)
        r'\b(correct|incorrect|partial)\b',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            grade = match.group(1).capitalize()
            if grade in ["Correct", "Incorrect", "Partial"]:
                return grade
    
    return None


def _normalize_prediction(prediction: str) -> str:
    """Normalize prediction to one of the valid grades."""
    if not prediction:
        return "None"
    
    valid_grades = ["Correct", "Incorrect", "Partial"]
    pred_clean = str(prediction).strip()
    
    # Remove quotes and extra whitespace
    pred_clean = pred_clean.strip('"\'').strip()
    pred_lower = pred_clean.lower()
    
    # Exact match check (case-insensitive)
    for grade in valid_grades:
        if pred_lower == grade.lower():
            return grade
    
    # Check for substring match - look for the grade words as whole words
    for grade in valid_grades:
        if re.search(r'\b' + grade.lower() + r'\b', pred_lower):
            return grade
    
    # Check if any grade appears as substring (for cases like "mostly correct")
    for grade in ["incorrect", "partial", "correct"]:
        if grade in pred_lower:
            return grade.capitalize()
    
    return "None"


class TaskAgent:
    """Task agent that solves IMO grading problems.
    
    This agent evaluates student answers against official solutions using
    an LLM with robust JSON extraction and fallback text parsing.
    """

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.
        
        Args:
            inputs: Dictionary containing problem, solution, grading_guidelines,
                   student_answer, and optional domain fields.
        
        Returns:
            Tuple of (prediction, message_history) where prediction is one of
            "Correct", "Incorrect", "Partial", or "None".
        """
        # Extract fields with defaults
        domain = inputs.get('domain', 'Mathematics')
        problem = inputs.get('problem', '')
        solution = inputs.get('solution', '')
        grading_guidelines = inputs.get('grading_guidelines', '')
        student_answer = inputs.get('student_answer', '')
        
        # Build improved instruction with clearer formatting and grade definitions
        instruction = f"""You are an expert grader for {domain} problems.

Your task is to evaluate a student's answer and assign exactly one grade: "Correct", "Incorrect", or "Partial".

GRADE DEFINITIONS:
- Correct: The student's answer is fully correct, complete, and matches the official solution. All key steps and reasoning are present and accurate.
- Incorrect: The student's answer is wrong, incomplete in a critical way, or fundamentally misunderstands the problem. Major errors or missing critical components.
- Partial: The student's answer has some correct elements but is incomplete, has minor errors, or misses key aspects. Shows understanding but not full mastery.

PROBLEM:
{problem}

OFFICIAL SOLUTION:
{solution}

GRADING GUIDELINES:
{grading_guidelines}

STUDENT'S ANSWER:
{student_answer}

Evaluate the student's answer carefully. Compare it against the official solution and grading guidelines.

IMPORTANT INSTRUCTIONS:
1. Analyze the student's answer step-by-step against the official solution.
2. Consider the grading guidelines when making your determination.
3. Choose ONE grade: "Correct", "Incorrect", or "Partial".
4. Respond with ONLY a JSON object in this exact format:

<json>
{{
    "response": "Correct"
}}
</json>

Replace "Correct" with "Incorrect" or "Partial" as appropriate. Do not include any other text before or after the JSON."""

        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
        except Exception as e:
            self.log_fn(f"LLM call failed: {e}")
            return "Error", []

        # Extract prediction
        prediction = "None"
        try:
            if msg_history:
                last_message = msg_history[-1]["text"]
                self.log_fn(f"Raw LLM response: {last_message[:200]}...")
                
                # Try JSON extraction first
                extracted = _extract_json_flexible(last_message)
                if extracted:
                    self.log_fn(f"Extracted JSON: {extracted}")
                    # Check for grade fields in priority order
                    for field in ["response", "grade", "result", "evaluation"]:
                        if field in extracted and extracted[field]:
                            prediction = str(extracted[field])
                            self.log_fn(f"Found grade in field '{field}': {prediction}")
                            break
                
                # If JSON extraction failed or didn't have expected fields, try text extraction
                if prediction == "None":
                    text_grade = _extract_grade_from_text(last_message)
                    if text_grade:
                        prediction = text_grade
                        self.log_fn(f"Extracted grade from text: {prediction}")
        except Exception as e:
            self.log_fn(f"Extraction error: {e}")

        # Normalize prediction
        prediction = _normalize_prediction(prediction)
        self.log_fn(f"Final prediction: {prediction}")

        return str(prediction), msg_history
