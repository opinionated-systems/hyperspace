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
    Includes multiple fallback strategies for robust JSON parsing.
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
        
        # Try multiple parsing strategies
        parsed = _try_parse_json(inner)
        if parsed is not None:
            results.append(parsed)
    
    return results or None


def _try_parse_json(text: str) -> dict | None:
    """Try multiple strategies to parse JSON text.
    
    Returns the parsed dict or None if all strategies fail.
    """
    # Strategy 1: Direct parsing
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Strategy 2: Remove trailing commas before closing braces/brackets
    try:
        fixed = re.sub(r',(\s*[}\]])', r'\1', text)
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    
    # Strategy 3: Fix single quotes to double quotes
    try:
        fixed = text.replace("'", '"')
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    
    # Strategy 4: Extract JSON-like content between first { and last }
    try:
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and start < end:
            fixed = text[start:end+1]
            return json.loads(fixed)
    except (json.JSONDecodeError, ValueError):
        pass
    
    # Strategy 5: Handle unquoted keys
    try:
        fixed = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', text)
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    
    return None


def _extract_grade_from_text(text: str) -> str | None:
    """Extract a grade from plain text when JSON extraction fails.
    
    Looks for common grade patterns like numbers 0-7, or phrases like
    'full credit', 'partial credit', 'no credit', etc.
    Uses multiple strategies to find the most likely grade.
    """
    text_lower = text.lower()
    
    # Strategy 1: Look for explicit grade statements (most reliable)
    explicit_patterns = [
        (r'grade\s*(?:is|:)?\s*([0-7])\b', 'grade_statement'),
        (r'score\s*(?:is|:)?\s*([0-7])\b', 'score_statement'),
        (r'(?:assigned?|give|giving)\s+(?:a?\s*)?(?:grade|score)\s*(?:of\s*)?([0-7])\b', 'assignment'),
        (r'(?:the\s+)?(?:grade|score)\s+(?:of\s*)?([0-7])\b', 'grade_of'),
        (r'(?:grade|score)\s*[=:]\s*([0-7])\b', 'equals'),
        (r'(?:award|awarding)\s+(?:a?\s*)?([0-7])\b', 'award'),
    ]
    
    for pattern, _ in explicit_patterns:
        matches = re.findall(pattern, text_lower)
        if matches:
            return matches[-1]  # Return last match (usually the final decision)
    
    # Strategy 2: Look for IMO-style numeric grades with context
    # Avoid matching numbers that are part of other content (like "step 1" or "7 points")
    numeric_patterns = [
        r'(?:^|\n|\.)\s*([0-7])\s*(?:/\s*7)?\s*(?:points?)?\s*(?:$|\n|\.\s)',
        r'\bfinal\s+(?:grade|score)\s*:?\s*([0-7])\b',
        r'\b(?:grade|score)\s*:?\s*([0-7])\b',
        r'\b([0-7])\s*/\s*7\b',
        r'\b([0-7])\s*out\s+of\s+7\b',
    ]
    
    for pattern in numeric_patterns:
        matches = re.findall(pattern, text_lower)
        if matches:
            return matches[-1]
    
    # Strategy 3: Look for standalone numbers 0-7 (more careful)
    # Only match if preceded by grade-related words or at line start
    standalone_pattern = r'(?:grade|score|mark|points?|is|:)\s*([0-7])\b|\n\s*([0-7])\s*(?:\n|$|\.)'
    matches = re.findall(standalone_pattern, text_lower)
    if matches:
        # Extract the non-empty group from each match
        for match in matches:
            grade = match[0] if match[0] else match[1]
            if grade:
                return grade
    
    # Strategy 4: Look for text-based grades with confidence scoring
    text_patterns = [
        (r'\bfull\s+credit\b', '7', 1.0),
        (r'\bcomplete(?:ly)?\s+correct\b', '7', 0.9),
        (r'\bcorrect\s+solution\b', '7', 0.9),
        (r'\bperfect\s+(?:score|solution)\b', '7', 0.95),
        (r'\ball\s+points?\b', '7', 0.9),
        (r'\bno\s+credit\b', '0', 1.0),
        (r'\bno\s+points?\b', '0', 0.9),
        (r'\bzero\s+(?:points?|credit)\b', '0', 0.9),
        (r'\bzero\b', '0', 0.7),
        (r'\bincorrect\s+(?:solution|answer)\b', '0', 0.85),
        (r'\bwrong\s+(?:solution|answer)\b', '0', 0.85),
        (r'\bincorrect\b', '0', 0.6),
        (r'\bwrong\b', '0', 0.5),
        (r'\bpartial\s+credit\b', 'Partial credit', 0.8),
        (r'\bpartial\s+(?:grade|score)\b', 'Partial credit', 0.8),
        (r'\bincomplete\b', 'Partial credit', 0.6),
        (r'\bpartial\b', 'Partial credit', 0.5),
    ]
    
    best_grade = None
    best_confidence = 0.0
    
    for pattern, grade, confidence in text_patterns:
        if re.search(pattern, text_lower):
            if confidence > best_confidence:
                best_confidence = confidence
                best_grade = grade
    
    if best_grade:
        return best_grade
    
    return None


class TaskAgent:
    """Task agent that solves IMO grading problems."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def _validate_inputs(self, inputs: dict) -> tuple[bool, str]:
        """Validate that all required fields are present in inputs.
        
        Returns:
            (is_valid, error_message)
        """
        required_fields = ["domain", "problem", "solution", "grading_guidelines", "student_answer"]
        missing = [f for f in required_fields if f not in inputs]
        if missing:
            return False, f"Missing required fields: {missing}"
        return True, ""

    def _build_grading_prompt(self, inputs: dict) -> str:
        """Build a structured prompt for the IMO grading task."""
        return f"""You are an expert mathematics grader for the International Mathematical Olympiad (IMO).

Your task is to evaluate a student's solution to a mathematical problem and provide a grade.

PROBLEM DOMAIN: {inputs['domain']}

PROBLEM STATEMENT:
{inputs['problem']}

OFFICIAL SOLUTION:
{inputs['solution']}

GRADING GUIDELINES:
{inputs['grading_guidelines']}

STUDENT'S ANSWER TO EVALUATE:
{inputs['student_answer']}

Instructions:
1. Carefully read the problem, official solution, and grading guidelines
2. Analyze the student's answer for correctness and completeness
3. Compare the student's approach with the official solution
4. Check if the student has:
   - Understood the problem correctly
   - Used valid mathematical reasoning
   - Reached a correct conclusion
   - Provided a complete solution
5. Assign an appropriate grade based on the grading guidelines

IMO grades are integers from 0 to 7, where:
- 7 = Full credit (complete and correct solution with proper reasoning)
- 6 = Minor flaw in an otherwise correct solution (e.g., small error in calculation)
- 5 = Significant progress with one major gap or error
- 3-4 = Partial progress (some correct ideas but incomplete)
- 1-2 = Minimal progress (some relevant ideas but mostly incorrect)
- 0 = No credit (no meaningful progress or completely wrong)

Respond ONLY in JSON format with the following schema:
<json>
{{
    "response": "Your grade here (0-7)"
}}
</json>

Important: 
- The response field must contain ONLY a single digit from 0 to 7
- Do not include any explanation, text, or other characters
- Use only numeric grades: 0, 1, 2, 3, 4, 5, 6, or 7
- Wrap your entire JSON response in <json>...</json> tags
- Example correct response: <json>{{"response": "5"}}</json>"""

    def _normalize_grade(self, grade: str | int | float) -> str:
        """Normalize a grade to a valid IMO grade (0-7).
        
        Args:
            grade: The extracted grade value
            
        Returns:
            Normalized grade string (0-7) or "None" if invalid
        """
        if grade is None:
            return "None"
        
        # Convert to string and clean up
        grade_str = str(grade).strip()
        
        # Handle "Partial credit" - map to a reasonable middle value
        if "partial" in grade_str.lower():
            return "4"  # Middle of the partial credit range
        
        # Try to extract numeric value
        try:
            # Remove any non-numeric characters except decimal point
            numeric_str = re.sub(r'[^\d.]', '', grade_str)
            if numeric_str:
                numeric_grade = float(numeric_str)
                # Clamp to valid range
                numeric_grade = max(0, min(7, numeric_grade))
                # Round to nearest integer
                return str(int(round(numeric_grade)))
        except (ValueError, TypeError):
            pass
        
        # Check for text-based grades with confidence scoring
        grade_lower = grade_str.lower()
        
        # High confidence matches
        if any(term in grade_lower for term in ["full credit", "complete solution", "perfect", "all correct"]):
            return "7"
        if any(term in grade_lower for term in ["no credit", "zero", "none", "incorrect", "wrong", "0/7"]):
            return "0"
        
        # Medium confidence matches
        if any(term in grade_lower for term in ["mostly correct", "minor error", "small mistake", "6/7"]):
            return "6"
        if any(term in grade_lower for term in ["significant progress", "major gap", "5/7"]):
            return "5"
        
        # Lower confidence matches (partial credit indicators)
        if any(term in grade_lower for term in ["some progress", "partial solution", "incomplete", "3/7", "4/7"]):
            # Try to determine if it's closer to 3 or 4
            if "mostly" in grade_lower or "significant" in grade_lower:
                return "4"
            return "3"
        if any(term in grade_lower for term in ["minimal progress", "few ideas", "2/7"]):
            return "2"
        if any(term in grade_lower for term in ["very minimal", "slight idea", "1/7"]):
            return "1"
        
        # Fallback: check for standalone digits 0-7
        digit_match = re.search(r'\b([0-7])\b', grade_str)
        if digit_match:
            return digit_match.group(1)
        
        return "None"

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Validate inputs
        is_valid, error_msg = self._validate_inputs(inputs)
        if not is_valid:
            self.log_fn(f"Input validation failed: {error_msg}")
            return f"Error: {error_msg}", []

        # Build structured prompt
        instruction = self._build_grading_prompt(inputs)

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON with improved error handling
        prediction = "None"
        try:
            if msg_history and len(msg_history) > 0:
                response_text = msg_history[-1]["text"]
                extracted = _extract_jsons(response_text)
                if extracted:
                    last_extract = extracted[-1]
                    if isinstance(last_extract, dict) and "response" in last_extract:
                        raw_prediction = last_extract["response"]
                        prediction = self._normalize_grade(raw_prediction)
                        self.log_fn(f"Extracted grade from JSON: {raw_prediction} -> {prediction}")
                    else:
                        self.log_fn(f"No 'response' key found in extracted JSON: {last_extract}")
                        # Try to extract from the raw text as fallback
                        text_grade = _extract_grade_from_text(response_text)
                        if text_grade:
                            prediction = self._normalize_grade(text_grade)
                            self.log_fn(f"Extracted grade from text fallback: {text_grade} -> {prediction}")
                else:
                    self.log_fn("No JSON blocks found in response, trying text extraction")
                    # Try to extract from the raw text as fallback
                    text_grade = _extract_grade_from_text(response_text)
                    if text_grade:
                        prediction = self._normalize_grade(text_grade)
                        self.log_fn(f"Extracted grade from text fallback: {text_grade} -> {prediction}")
            else:
                self.log_fn("Empty message history")
        except json.JSONDecodeError as e:
            self.log_fn(f"JSON decode error: {e}")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
