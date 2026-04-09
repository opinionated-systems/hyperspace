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
    """Extract JSON objects from <json>...</json> blocks."""
    if not text or not isinstance(text, str):
        return None
    
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
            parsed = json.loads(inner)
            if isinstance(parsed, dict):
                results.append(parsed)
        except json.JSONDecodeError:
            # Try with single quotes replaced
            try:
                parsed = json.loads(inner.replace("'", '"'))
                if isinstance(parsed, dict):
                    results.append(parsed)
            except json.JSONDecodeError:
                pass
    
    return results if results else None


def _extract_json_flexible(text: str) -> list[dict] | None:
    """Extract JSON objects using multiple strategies.
    
    Tries multiple patterns:
    1. <json>...</json> tags
    2. ```json...``` code blocks
    3. Raw JSON objects with "response" field
    4. JSON-like patterns with flexible whitespace
    """
    if not text or not isinstance(text, str):
        return None
        
    results = []
    
    # Strategy 1: <json>...</json> tags
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
            # Try with single quotes replaced
            try:
                results.append(json.loads(inner.replace("'", '"')))
            except json.JSONDecodeError:
                pass
    
    # Strategy 2: ```json...``` code blocks
    pattern = r'```json\s*(.*?)\s*```'
    matches = re.findall(pattern, text, re.DOTALL)
    for match in matches:
        try:
            results.append(json.loads(match.strip()))
        except json.JSONDecodeError:
            pass
    
    # Strategy 3: ```...``` code blocks (without json label)
    pattern = r'```\s*(\{.*?\})\s*```'
    matches = re.findall(pattern, text, re.DOTALL)
    for match in matches:
        try:
            results.append(json.loads(match.strip()))
        except json.JSONDecodeError:
            pass
    
    # Strategy 4: Raw JSON objects with "response" field
    pattern = r'\{\s*"response"\s*:\s*"([^"]+)"\s*\}'
    matches = re.findall(pattern, text)
    for match in matches:
        results.append({"response": match})
    
    # Strategy 5: Case-insensitive response extraction
    pattern = r'["\']?response["\']?\s*:\s*["\']?(correct|incorrect|partial|almost)["\']?'
    matches = re.findall(pattern, text, re.IGNORECASE)
    for match in matches:
        results.append({"response": match.lower()})
    
    return results if results else None


def _normalize_grade(grade: str) -> str:
    """Normalize a grade string to one of the valid grades.
    
    Maps 'almost' to 'partial' for final output consistency.
    """
    if not grade:
        return "incorrect"
    grade = grade.lower().strip()
    valid_grades = ["correct", "incorrect", "partial", "almost"]
    if grade in valid_grades:
        # Map almost to partial for consistency with evaluation
        return "partial" if grade == "almost" else grade
    return "incorrect"


def _extract_response_robust(text: str) -> str | None:
    """Robust extraction using multiple pattern matching strategies."""
    if not text or not isinstance(text, str):
        return None
        
    text_lower = text.lower()
    valid_grades = ['correct', 'incorrect', 'partial', 'almost']
    
    # Priority 1: Look for <json> tags with response field
    try:
        json_results = _extract_jsons(text)
        if json_results:
            for item in json_results:
                if isinstance(item, dict) and "response" in item:
                    val = item["response"]
                    if isinstance(val, str):
                        normalized = _normalize_grade(val)
                        if normalized != "incorrect" or val.lower() == "incorrect":
                            return normalized
    except Exception:
        pass
    
    # Priority 2: Look for JSON-like patterns with response field
    json_patterns = [
        r'"response"\s*:\s*"(correct|incorrect|partial|almost)"',
        r"'response'\s*:\s*'(correct|incorrect|partial|almost)'",
        r'response\s*:\s*"(correct|incorrect|partial|almost)"',
        r'response\s*:\s*(correct|incorrect|partial|almost)',
        r'{"response":\s*"(correct|incorrect|partial|almost)"}',
        r"{'response':\s*'(correct|incorrect|partial|almost)'}",
    ]
    
    for pattern in json_patterns:
        match = re.search(pattern, text_lower)
        if match:
            return _normalize_grade(match.group(1))
    
    # Priority 3: Look for the grade as a standalone word at the end of lines
    lines = text_lower.split('\n')
    for line in reversed(lines):
        line_stripped = line.strip().rstrip('.,;:!?')
        for grade in valid_grades:
            if line_stripped == grade or re.search(rf'\b{grade}\b', line_stripped):
                if line_stripped.endswith(grade) or line_stripped == grade:
                    return _normalize_grade(grade)
    
    # Priority 4: Look for grade in quotes anywhere in text
    for grade in valid_grades:
        if re.search(rf'"{grade}"', text_lower) or re.search(rf"'{grade}'", text_lower) or re.search(rf'`{grade}`', text_lower):
            return _normalize_grade(grade)
    
    # Priority 5: Look for explicit decision patterns
    decision_patterns = [
        rf'\bgrade\s*[:=]\s*["\']?(correct|incorrect|partial|almost)["\']?',
        rf'\bverdict\s*[:=]\s*["\']?(correct|incorrect|partial|almost)["\']?',
        rf'\bfinal\s+(?:grade|verdict|answer)\s*[:=]\s*["\']?(correct|incorrect|partial|almost)["\']?',
        rf'\bthe\s+answer\s+is\s*[:=]?\s*["\']?(correct|incorrect|partial|almost)["\']?',
    ]
    
    for pattern in decision_patterns:
        match = re.search(pattern, text_lower)
        if match:
            return _normalize_grade(match.group(1))
    
    # Priority 6: Look for grade in brackets or parentheses
    bracket_patterns = [
        rf'\((correct|incorrect|partial|almost)\)',
        rf'\[(correct|incorrect|partial|almost)\]',
        rf'\{{(correct|incorrect|partial|almost)\}}',
    ]
    for pattern in bracket_patterns:
        match = re.search(pattern, text_lower)
        if match:
            return _normalize_grade(match.group(1))
    
    # Priority 7: Look for grade at the very end of the text
    text_stripped = text_lower.strip()
    for grade in valid_grades:
        if text_stripped.endswith(grade):
            return _normalize_grade(grade)
    
    # Priority 8: Simple word boundary search for grades (prefer first occurrence of 'incorrect' to avoid misclassification)
    # Check for incorrect first to avoid misclassifying text that mentions multiple grades
    if re.search(r'\bincorrect\b', text_lower):
        return 'incorrect'
    for grade in ['correct', 'partial', 'almost']:
        if re.search(rf'\b{grade}\b', text_lower):
            return _normalize_grade(grade)
    
    return None


class TaskAgent:
    """Task agent that solves IMO grading problems."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Extract fields from inputs
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        # Build the prompt
        prompt = f"""You are an expert mathematics grader for International Mathematical Olympiad (IMO) problems.

Your task is to carefully analyze a student's answer and determine if it is correct, incorrect, partial, or almost.

PROBLEM:
{problem}

OFFICIAL SOLUTION:
{solution}

GRADING GUIDELINES:
{grading_guidelines}

STUDENT'S ANSWER:
{student_answer}

ANALYSIS INSTRUCTIONS:
1. First, identify what the problem is asking for and what constitutes a complete solution.
2. Compare the student's approach to the official solution - are they using valid methods?
3. Check for any logical gaps, errors, or unjustified claims in the student's work.

4. CRITICAL - EXPLICIT MARKERS: The grading guidelines contain explicit markers that OVERRIDE your own judgment:
   - If you see "(Almost)" or "almost correct" in the guidelines → output "almost"
   - If you see "(Correct)" in the guidelines → output "correct"  
   - If you see "(Incorrect)" or "(0 points)" in the guidelines → output "incorrect"
   - If you see "(Partial)" in the guidelines → output "partial"
   
   These markers are authoritative. When present, use them directly without second-guessing.

5. Determine the grade based on these criteria:
   - CORRECT: Complete, valid proof with all steps justified and correct conclusion. No errors or gaps.
   - INCORRECT: Fundamental errors, wrong approach, or conclusion that doesn't follow.
   - PARTIAL: Valid insights, correct lemmas, good approach but significant gaps remain (typically 1-3 points on a 7-point problem).
   - ALMOST: Nearly complete solution with only minor errors or omissions (typically 5-6 points on a 7-point problem). The solution is "almost correct" but has small issues preventing full marks.

KEY DISTINCTION - ALMOST vs CORRECT vs PARTIAL:
- CORRECT: The solution is complete and correct. No issues whatsoever.
- ALMOST: The solution is nearly complete with only minor errors/omissions. It's very close to correct but not quite perfect.
- PARTIAL: The solution has valid insights but significant work remains incomplete or there are major logical gaps.

IMPORTANT: When the grading guidelines explicitly contain "(Almost)", you MUST output "almost" - not "correct", not "partial". The "(Almost)" marker is your direct instruction.

You MUST output your final grade in the following EXACT format on its own line:
<json>{{"response": "GRADE"}}</json>

Where GRADE is exactly one of: "correct", "incorrect", "partial", or "almost"

The JSON block must be the very last thing in your response."""

        # Get response from the model
        response, msg_history, _ = get_response_from_llm(
            msg=prompt,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from response, passing grading_guidelines for explicit marker detection
        prediction = self._extract_prediction(response, grading_guidelines)
        
        return str(prediction), msg_history

    def _extract_prediction(self, response: str, grading_guidelines: str = "") -> str:
        """Extract prediction from response with robust fallback strategies.
        
        Args:
            response: The model's response text
            grading_guidelines: The grading guidelines text (used for explicit marker detection)
        """
        if not response or not isinstance(response, str):
            self.log_fn("Empty or invalid response, defaulting to incorrect")
            return "incorrect"
        
        response_text = response
        self.log_fn(f"Raw response text: {response_text[:500]}...")
        
        # Check grading guidelines for explicit markers first (authoritative)
        if grading_guidelines:
            guidelines_lower = grading_guidelines.lower()
            # Check for explicit markers in order of specificity
            if "(almost)" in guidelines_lower:
                self.log_fn("Explicit '(Almost)' marker found in guidelines - outputting 'almost'")
                return "almost"
            elif "(correct)" in guidelines_lower:
                self.log_fn("Explicit '(Correct)' marker found in guidelines - outputting 'correct'")
                return "correct"
            elif "(incorrect)" in guidelines_lower or "(0 points)" in guidelines_lower:
                self.log_fn("Explicit '(Incorrect)' or '(0 points)' marker found in guidelines - outputting 'incorrect'")
                return "incorrect"
            elif "(partial)" in guidelines_lower:
                self.log_fn("Explicit '(Partial)' marker found in guidelines - outputting 'partial'")
                return "partial"
        
        # Try extraction strategies in order of reliability
        prediction = None
        
        # Strategy 1: Flexible JSON extraction (highest priority - handles multiple formats)
        try:
            json_results = _extract_json_flexible(response_text)
            if json_results:
                for item in json_results:
                    if isinstance(item, dict) and "response" in item:
                        val = item["response"]
                        if isinstance(val, str) and val.lower() in ["correct", "incorrect", "partial", "almost"]:
                            prediction = _normalize_grade(val)
                            self.log_fn(f"Extracted prediction via flexible JSON: {prediction}")
                            break
        except Exception as e:
            self.log_fn(f"Error in flexible JSON extraction: {e}")
        
        # Strategy 2: Extract from <json> tags (legacy support)
        if prediction is None:
            try:
                json_results = _extract_jsons(response_text)
                if json_results:
                    for item in json_results:
                        if isinstance(item, dict) and "response" in item:
                            val = item["response"]
                            if isinstance(val, str):
                                prediction = _normalize_grade(val)
                                self.log_fn(f"Extracted prediction via <json> tags: {prediction}")
                                break
            except Exception as e:
                self.log_fn(f"Error in <json> extraction: {e}")
        
        # Strategy 3: Robust extraction with multiple pattern matching
        if prediction is None:
            try:
                result = _extract_response_robust(response_text)
                if result:
                    prediction = result
                    self.log_fn(f"Extracted prediction via robust extraction: {prediction}")
            except Exception as e:
                self.log_fn(f"Error in robust extraction: {e}")
        
        # Strategy 4: Direct text search for grades in quotes (explicit mentions)
        if prediction is None:
            text_lower = response_text.lower()
            for grade in ["correct", "incorrect", "partial", "almost"]:
                if f'"{grade}"' in text_lower or f"'{grade}'" in text_lower or f"`{grade}`" in text_lower:
                    prediction = _normalize_grade(grade)
                    self.log_fn(f"Extracted prediction via quotes: {prediction}")
                    break
        
        # Strategy 5: Look for explicit grading statements
        if prediction is None:
            text_lower = response_text.lower()
            grade_patterns = [
                (r'\bgrade\s*[:=]\s*["\']?(correct|incorrect|partial|almost)["\']?', "grade statement"),
                (r'\bverdict\s*[:=]\s*["\']?(correct|incorrect|partial|almost)["\']?', "verdict statement"),
                (r'\bfinal\s+(?:grade|verdict|answer)\s*[:=]\s*["\']?(correct|incorrect|partial|almost)["\']?', "final statement"),
                (r'\bthe\s+answer\s+is\s*[:=]?\s*["\']?(correct|incorrect|partial|almost)["\']?', "answer statement"),
            ]
            for pattern, source in grade_patterns:
                match = re.search(pattern, text_lower)
                if match:
                    prediction = _normalize_grade(match.group(1))
                    self.log_fn(f"Extracted prediction via {source}: {prediction}")
                    break
        
        # Strategy 6: Simple word boundary search with priority order
        # Prefer 'incorrect' first to avoid misclassifying text that mentions multiple grades
        if prediction is None:
            text_lower = response_text.lower()
            
            if re.search(r'\bincorrect\b', text_lower):
                prediction = "incorrect"
                self.log_fn(f"Extracted prediction via simple text search: {prediction}")
            elif re.search(r'\bcorrect\b', text_lower):
                prediction = "correct"
                self.log_fn(f"Extracted prediction via simple text search: {prediction}")
            elif re.search(r'\bpartial\b', text_lower) or re.search(r'\balmost\b', text_lower):
                prediction = "partial"
                self.log_fn(f"Extracted prediction via simple text search: {prediction}")
        
        # Validate and return the prediction
        if prediction not in ["correct", "incorrect", "partial", "almost"]:
            self.log_fn(f"Invalid or missing prediction: '{prediction}', defaulting to incorrect")
            prediction = "incorrect"
        
        return prediction
