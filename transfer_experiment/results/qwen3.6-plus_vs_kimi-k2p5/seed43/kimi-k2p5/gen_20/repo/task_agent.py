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
    json_like_patterns = [
        r'\{\s*"response"\s*:\s*"[^"]+"\s*\}',
        r'\{\s*"grade"\s*:\s*"[^"]+"\s*\}',
        r'\{\s*"result"\s*:\s*"[^"]+"\s*\}',
        r'\{\s*"evaluation"\s*:\s*"[^"]+"\s*\}',
    ]
    for pattern in json_like_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
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
                        start_idx = -1
                        continue
    
    return None


def _extract_grade_from_text(text: str) -> str | None:
    """Extract grade from raw text using pattern matching with improved accuracy."""
    text_lower = text.lower()
    
    # Look for explicit grade patterns (ordered by specificity - highest priority first)
    patterns = [
        # JSON-style patterns (highest priority)
        r'"response"\s*:\s*"(correct|incorrect|partial)"',
        r'"grade"\s*:\s*"(correct|incorrect|partial)"',
        r'"result"\s*:\s*"(correct|incorrect|partial)"',
        r'"evaluation"\s*:\s*"(correct|incorrect|partial)"',
        # Assignment patterns with quotes
        r'grade\s*[:=]\s*"(correct|incorrect|partial)"',
        r'response\s*[:=]\s*"(correct|incorrect|partial)"',
        r'result\s*[:=]\s*"(correct|incorrect|partial)"',
        # Assignment patterns without quotes
        r'grade\s*[:=]\s*(correct|incorrect|partial)(?:\s|$|[^a-z])',
        r'response\s*[:=]\s*(correct|incorrect|partial)(?:\s|$|[^a-z])',
        # Statement patterns with "is"
        r'\bthe answer is\s+(correct|incorrect|partial)\b',
        r'\bthe grade is\s+(correct|incorrect|partial)\b',
        r'\bthe response is\s+(correct|incorrect|partial)\b',
        r'\bthe result is\s+(correct|incorrect|partial)\b',
        r'\bthe evaluation is\s+(correct|incorrect|partial)\b',
        # Statement patterns with "should be"
        r'\b(?:should|must|would)\s+be\s+(correct|incorrect|partial)\b',
        # Assessment/verdict patterns
        r'\bevaluation[\s:]+(correct|incorrect|partial)\b',
        r'\bassessment[\s:]+(correct|incorrect|partial)\b',
        r'\bverdict[\s:]+(correct|incorrect|partial)\b',
        # Action patterns
        r'\bgrade[d]?\s+(as\s+)?(correct|incorrect|partial)\b',
        r'\bmarked as\s+(correct|incorrect|partial)\b',
        r'\bconsidered\s+(correct|incorrect|partial)\b',
        r'\bdetermined to be\s+(correct|incorrect|partial)\b',
        r'\bcategorized as\s+(correct|incorrect|partial)\b',
        # Conclusion patterns
        r'\bconclusion[\s:]+(correct|incorrect|partial)\b',
        r'\bfinal (?:grade|assessment)[\s:]+(correct|incorrect|partial)\b',
        # Standalone grades at word boundaries (lowest priority)
        r'\b(correct|incorrect|partial)\b',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            # Get the last group that matched (handles optional groups)
            grade = match.group(match.lastindex).capitalize()
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
    
    # Check for whole word match - highest priority
    for grade in valid_grades:
        if re.search(r'\b' + grade.lower() + r'\b', pred_lower):
            return grade
    
    # Check for grade words as substrings with word boundaries on at least one side
    # This handles cases like "mostly correct" or "partially correct"
    for grade in ["incorrect", "partial", "correct"]:
        # Look for the grade word with word boundary at start or end
        pattern = r'(?:\b' + grade + r'|' + grade + r'\b)'
        if re.search(pattern, pred_lower):
            return grade.capitalize()
    
    # Final fallback: any occurrence of the grade words
    for grade in ["incorrect", "partial", "correct"]:
        if grade in pred_lower:
            return grade.capitalize()
    
    return "None"


def _extract_grade_with_context(text: str) -> str | None:
    """Extract grade using context-aware analysis of the full response."""
    text_lower = text.lower()
    
    # Look for conclusion/summary sections that often contain the final grade
    # These patterns look for explicit conclusion markers followed by grade words
    conclusion_patterns = [
        r'(?:conclusion|summary|final|verdict|assessment|grade)[\s:]*(.{0,150})',
        r'(?:therefore|thus|hence|in conclusion|to summarize)[,\s]*(.{0,150})',
        r'(?:the student|this answer|the response)[\s\w]*(?:is|should be|would be|can be)[\s:]*(.{0,100})',
        r'(?:i (?:conclude|determine|assess|grade)|my (?:conclusion|assessment|verdict))[,\s:]*(.{0,150})',
        r'(?:overall|in total|taking everything into account)[,\s:]*(.{0,150})',
    ]
    
    for pattern in conclusion_patterns:
        matches = re.findall(pattern, text_lower, re.IGNORECASE | re.DOTALL)
        for match in matches:
            # Look for grade words in the matched context
            for grade in ["correct", "incorrect", "partial"]:
                if re.search(r'\b' + grade + r'\b', match):
                    return grade.capitalize()
    
    # Check the last paragraph which often contains the conclusion
    paragraphs = [p.strip() for p in text_lower.split('\n\n') if p.strip()]
    if paragraphs:
        last_para = paragraphs[-1]
        # Look for grade words in the last paragraph
        for grade in ["correct", "incorrect", "partial"]:
            if re.search(r'\b' + grade + r'\b', last_para):
                return grade.capitalize()
    
    # Check last few lines for standalone grades
    lines = [l.strip() for l in text_lower.strip().split('\n') if l.strip()]
    for line in reversed(lines[-5:]):  # Check last 5 non-empty lines
        for grade in ["correct", "incorrect", "partial"]:
            if re.search(r'^[\s]*' + grade + r'[\s]*$', line) or \
               re.search(r'^[\s]*["\']?' + grade + r'["\']?[\s]*$', line):
                return grade.capitalize()
    
    # Check for grade words near the end of the text
    last_200_chars = text_lower[-200:]
    for grade in ["correct", "incorrect", "partial"]:
        if re.search(r'\b' + grade + r'\b', last_200_chars):
            return grade.capitalize()
    
    return None


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
        
        # Build improved instruction with chain-of-thought reasoning and clearer criteria
        instruction = f"""You are an expert grader for {domain} problems, specializing in evaluating mathematical proofs and solutions.

Your task is to evaluate a student's answer and assign exactly one grade: "Correct", "Incorrect", or "Partial".

GRADE DEFINITIONS:
- Correct: The student's answer is fully correct and complete. It contains all key steps, correct reasoning, and matches the official solution's conclusions. Minor presentation issues are acceptable if the mathematics is sound.
- Incorrect: The student's answer contains fundamental errors, wrong conclusions, or critical omissions that invalidate the solution. The student has not demonstrated understanding of the core problem.
- Partial: The student's answer shows understanding of the problem and has some correct elements, but is incomplete, contains non-critical errors, or misses key aspects of the full solution.

PROBLEM:
{problem}

OFFICIAL SOLUTION:
{solution}

GRADING GUIDELINES:
{grading_guidelines}

STUDENT'S ANSWER:
{student_answer}

EVALUATION PROCESS (think step-by-step):
1. First, identify the key claims and conclusions in the official solution.
2. Check if the student's answer addresses the main question correctly.
3. Verify if the student's reasoning is logically sound and mathematically valid.
4. Compare the student's approach with the official solution - different valid approaches are acceptable.
5. Identify any errors, omissions, or gaps in the student's answer.
6. Based on the above analysis, determine the appropriate grade.

IMPORTANT: You must respond with ONLY a JSON object in this exact format:

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

        # Extract prediction - default to "None" if extraction fails
        prediction = "None"
        try:
            if msg_history and len(msg_history) > 0:
                last_message = msg_history[-1].get("text", "")
                if last_message:
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
                    
                    # Try context-aware extraction as final fallback
                    if prediction == "None":
                        context_grade = _extract_grade_with_context(last_message)
                        if context_grade:
                            prediction = context_grade
                            self.log_fn(f"Extracted grade from context: {prediction}")
        except Exception as e:
            self.log_fn(f"Extraction error: {e}")

        # Normalize prediction
        prediction = _normalize_prediction(prediction)
        self.log_fn(f"Final prediction: {prediction}")

        return str(prediction), msg_history
