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

VALID_GRADES = ["correct", "incorrect", "partial", "almost"]


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks or raw JSON.
    
    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    Also tries to find raw JSON objects if tags are not present.
    """
    if not text:
        return None
        
    results = []
    search_from = 0

    # First try to find <json>...</json> blocks
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

    # Also try to find ```json code blocks
    search_from = 0
    while True:
        start = text.find("```json", search_from)
        if start == -1:
            break
        end = text.find("```", start + 7)
        if end == -1:
            break
        inner = text[start + 7:end].strip()
        search_from = end + 3
        try:
            results.append(json.loads(inner))
        except json.JSONDecodeError:
            continue

    # If no results, try to find raw JSON objects (looking for balanced braces)
    if not results:
        i = 0
        while i < len(text):
            if text[i] == '{':
                brace_count = 1
                j = i + 1
                while j < len(text) and brace_count > 0:
                    if text[j] == '{':
                        brace_count += 1
                    elif text[j] == '}':
                        brace_count -= 1
                    j += 1

                if brace_count == 0:
                    json_str = text[i:j].strip()
                    try:
                        results.append(json.loads(json_str))
                    except json.JSONDecodeError:
                        pass
                i = j
            else:
                i += 1

    return results or None


def _extract_grade_from_json(text: str) -> str | None:
    """Extract grade from JSON objects in the text.
    
    This is the most reliable extraction method and should be tried first.
    """
    if not text:
        return None
    
    json_results = _extract_jsons(text)
    if not json_results:
        return None
    
    # Check from last to first (most likely to contain final answer)
    for json_obj in reversed(json_results):
        if isinstance(json_obj, dict):
            # Try common grade field names in order of preference
            for field in ["grade", "result", "prediction", "response", "answer", "verdict"]:
                if field in json_obj:
                    val = str(json_obj[field]).lower().strip().strip('"').strip("'")
                    if val in VALID_GRADES:
                        return val
            
            # If no standard field found, try any value in the JSON
            for val in json_obj.values():
                val_str = str(val).lower().strip().strip('"').strip("'")
                if val_str in VALID_GRADES:
                    return val_str
    
    return None


def _extract_grade_from_text(text: str) -> str | None:
    """Extract grade from various formats in the text with improved patterns."""
    if not text:
        return None
        
    text_clean = text.strip()
    text_lower = text_clean.lower()
    
    # Priority 1: Look for JSON grade field patterns (most reliable)
    # Try to find the LAST occurrence (most likely to be the final answer)
    json_patterns = [
        r'"grade"\s*:\s*"([^"]*)"',
        r'"grade"\s*:\s*\'([^\']*)\'',
        r'"grade"\s*:\s*([a-zA-Z]+)',
        r'"response"\s*:\s*"([^"]*)"',
        r'"prediction"\s*:\s*"([^"]*)"',
        r'"result"\s*:\s*"([^"]*)"',
        r'"answer"\s*:\s*"([^"]*)"',
        r'"verdict"\s*:\s*"([^"]*)"',
    ]
    
    for pattern in json_patterns:
        # Find all matches and take the last one (most likely final answer)
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        if matches:
            grade = matches[-1].group(1).strip().lower()
            if grade in VALID_GRADES:
                return grade
    
    # Priority 2: Look for explicit grade assignments in text (last occurrence)
    text_patterns = [
        r'\bgrade\s*[:=]\s*["\']?([a-zA-Z]+)["\']?',
        r'\bfinal\s+grade\s*[:=]\s*["\']?([a-zA-Z]+)["\']?',
        r'\bassigned\s+grade\s*[:=]\s*["\']?([a-zA-Z]+)["\']?',
        r'\bthe\s+grade\s+is\s*["\']?([a-zA-Z]+)["\']?',
        r'\bi\s+assign\s*["\']?([a-zA-Z]+)["\']?',
        r'\bthis\s+is\s+([a-zA-Z]+)\b',
        r'\bverdict\s*[:=]\s*["\']?([a-zA-Z]+)["\']?',
        r'\bconclusion\s*[:=]\s*["\']?([a-zA-Z]+)["\']?',
        r'\bfinal\s+grade\s+selection\s*[:=]\s*["\']?([a-zA-Z]+)["\']?',
        r'\bgrade\s+should\s+be\s*["\']?([a-zA-Z]+)["\']?',
        r'\bgrade\s*:\s*["\']?([a-zA-Z]+)["\']?',
        r'\bfinal\s+answer\s*[:=]\s*["\']?([a-zA-Z]+)["\']?',
        r'\bdecision\s*[:=]\s*["\']?([a-zA-Z]+)["\']?',
    ]
    
    for pattern in text_patterns:
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        if matches:
            grade = matches[-1].group(1).strip().lower()
            if grade in VALID_GRADES:
                return grade
    
    # Priority 3: Look for quoted grades (strong indicator) - last occurrence
    quoted_patterns = [
        (r'["\']almost["\']', 'almost'),
        (r'["\']partial["\']', 'partial'),
        (r'["\']incorrect["\']', 'incorrect'),
        (r'["\']correct["\']', 'correct'),
    ]
    for pattern, grade in quoted_patterns:
        matches = list(re.finditer(pattern, text_lower))
        if matches:
            return grade
    
    # Priority 4: Look for grade mentioned in final verification checklist
    # This is a strong indicator of the final decision
    checklist_pattern = r'final\s*check.*?(almost|partial|incorrect|correct)'
    checklist_match = re.search(checklist_pattern, text_lower)
    if checklist_match:
        grade = checklist_match.group(1).strip().lower()
        if grade in VALID_GRADES:
            return grade
    
    # Priority 5: Look for "grade: X" or "grade should be X" patterns
    grade_decision_pattern = r'(?:grade|verdict|conclusion)(?:\s+(?:is|should\s+be|selection))?[\s:]+["\']?(almost|partial|incorrect|correct)["\']?'
    decision_matches = list(re.finditer(grade_decision_pattern, text_lower))
    if decision_matches:
        grade = decision_matches[-1].group(1).strip().lower()
        if grade in VALID_GRADES:
            return grade
    
    # Priority 6: Look for standalone grade keywords in the last 300 words
    # This is where the conclusion typically appears
    words = text_lower.split()
    last_part = ' '.join(words[-300:]) if len(words) > 300 else text_lower
    
    # Check in order of specificity (more specific first to avoid misclassification)
    # Use word boundaries to avoid matching "partial" inside "almost"
    if re.search(r'\balmost\b', last_part):
        return 'almost'
    elif re.search(r'\bpartial\b', last_part):
        return 'partial'
    elif re.search(r'\bincorrect\b', last_part):
        return 'incorrect'
    elif re.search(r'\bcorrect\b', last_part):
        return 'correct'
    
    return None


def _extract_grade(text: str) -> str:
    """Extract grade from LLM response text using multiple strategies.
    
    This function uses a hierarchical approach:
    1. First try JSON extraction (most reliable)
    2. Then try text pattern extraction
    3. Finally fall back to keyword search in the last section
    """
    if not text:
        return "none"
    
    # Strategy 1: Try JSON extraction first (most reliable)
    json_result = _extract_grade_from_json(text)
    if json_result:
        return json_result
    
    # Strategy 2: Try text pattern extraction
    text_result = _extract_grade_from_text(text)
    if text_result:
        return text_result
    
    # Strategy 3: Look for grade in the last 1000 characters (where conclusion typically is)
    text_lower = text.lower()
    last_section = text_lower[-1000:] if len(text_lower) > 1000 else text_lower
    
    # Check in order of specificity in the last section first
    # "almost" is most specific (contains "partial" as substring)
    for grade in ["almost", "partial", "incorrect", "correct"]:
        if re.search(rf'\b{grade}\b', last_section):
            return grade
    
    # Strategy 4: Look anywhere in text, but prioritize later occurrences
    for grade in ["almost", "partial", "incorrect", "correct"]:
        # Find all occurrences and take the last one
        matches = list(re.finditer(rf'\b{grade}\b', text_lower))
        if matches:
            # Check if this is in the last 25% of the text (likely conclusion)
            last_match = matches[-1]
            position_ratio = last_match.start() / len(text_lower)
            if position_ratio > 0.75:  # In the last quarter of text
                return grade
    
    # Strategy 5: Final fallback - any occurrence anywhere in text
    for grade in ["almost", "partial", "incorrect", "correct"]:
        if re.search(rf'\b{grade}\b', text_lower):
            return grade
    
    return "none"


def _normalize_grade(grade: str) -> str:
    """Normalize grade to one of the expected lowercase values."""
    if not grade:
        return "none"
    
    grade_lower = grade.lower().strip().strip('"').strip("'")
    
    # Map to standard grades
    if grade_lower in VALID_GRADES:
        return grade_lower
    
    # Partial matches as fallback
    if "almost" in grade_lower:
        return "almost"
    elif "partial" in grade_lower:
        return "partial"
    elif "incorrect" in grade_lower:
        return "incorrect"
    elif "correct" in grade_lower:
        return "correct"
    
    return "none"


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
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = f"""You are an expert mathematical olympiad grader. Your task is to evaluate a student's solution and assign exactly one of four grades: "correct", "incorrect", "partial", or "almost".

## Problem Statement:
{problem}

## Official Solution:
{solution}

## Student's Answer:
{student_answer}

## Grading Guidelines (FOLLOW THESE EXACTLY):
{grading_guidelines}

## Grade Definitions

**CORRECT**: Complete, fully correct solution (95-100%). The main result is fully proven with no significant issues. All key steps are valid and complete.

**INCORRECT**: No meaningful progress (0-30%). No valid mathematical reasoning. The student did NOT meet ANY criteria from the "(Partial)" section of the grading guidelines.

**PARTIAL**: Significant progress but incomplete (30-75%). The main result is NOT fully proven. The student achieved at least one criterion from the "(Partial)" section of the grading guidelines.

**ALMOST**: Nearly complete with only minor issues (75-95%). The main result IS proven. Only small computational errors, minor gaps, or slight oversights exist.

## Decision Process (Follow This Exactly)

**Step 1: Check Partial Criteria**
- Review the "(Partial)" section in the grading guidelines
- Did the student achieve ANY criterion from this section? 
- If NO → Grade is "incorrect"
- If YES → Continue to Step 2

**Step 2: Check Main Result**
- Is the main theorem/result FULLY proven?
- If NO → Grade is "partial"
- If YES → Continue to Step 3

**Step 3: Check for Issues**
- Are there only minor issues (small computational errors, minor gaps)?
- If YES → Grade is "almost"
- If NO (no issues at all) → Grade is "correct"

## Response Format (CRITICAL - FOLLOW EXACTLY)

You MUST output your response in this exact JSON format:

<json>
{{"grade": "correct"}}
</json>

OR

<json>
{{"grade": "incorrect"}}
</json>

OR

<json>
{{"grade": "partial"}}
</json>

OR

<json>
{{"grade": "almost"}}
</json>

**IMPORTANT:**
- The grade value MUST be exactly one of: "correct", "incorrect", "partial", or "almost" (all lowercase)
- Use double quotes around the grade value
- Wrap the JSON in <json>...</json> tags
- Do not include any other text after the JSON block

## Your Analysis (Complete Before Grading)

Before giving your final grade, briefly analyze:
1. What Partial criteria did the student achieve? (List each with YES/NO)
2. Is the main result fully proven? (YES/NO)
3. Are there only minor issues? (YES/NO)

Then provide your final grade in the exact JSON format specified above.

Your grade:"""

        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
        except Exception as e:
            self.log_fn(f"Error getting LLM response: {e}")
            return "incorrect", []

        # Extract prediction with improved logic
        prediction = "none"
        
        try:
            # Get the text to search - prioritize msg_history over direct response
            search_text = None
            
            # First try to get the response from the msg_history
            if msg_history and len(msg_history) > 0:
                last_message = msg_history[-1].get("text", "")
                if last_message:
                    search_text = last_message
            
            # Fallback to direct response if msg_history is empty
            if not search_text and response:
                search_text = response
            
            if search_text:
                # Use the comprehensive extraction function
                prediction = _extract_grade(search_text)
                
                # Log if we had to fall back to text extraction
                if prediction == "none":
                    preview = search_text[:300] if len(search_text) > 300 else search_text
                    self.log_fn(f"Warning: Could not extract valid grade from: {preview}...")
            else:
                self.log_fn("Warning: No response text available for extraction")
                            
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            import traceback
            self.log_fn(f"Traceback: {traceback.format_exc()}")

        # Normalize prediction
        prediction = _normalize_grade(prediction)
        
        # Final validation - ensure we always return a valid grade
        if prediction not in VALID_GRADES:
            self.log_fn(f"Warning: Invalid prediction '{prediction}', defaulting to 'incorrect'")
            prediction = "incorrect"

        return str(prediction), msg_history
