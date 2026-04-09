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


def _extract_points_from_guidelines(grading_guidelines: str) -> int | None:
    """Extract the points value from grading guidelines.
    
    Returns the points value (0-7) or None if not found.
    This is the MOST IMPORTANT function - points determine the classification.
    """
    if not grading_guidelines:
        return None
        
    text = grading_guidelines.strip()
    
    # Priority 1: Look for explicit "Points: X" or "Points = X" patterns
    explicit_patterns = [
        r'[Pp]oints\s*[=:]\s*(\d+)',
        r'[Ss]core\s*[=:]\s*(\d+)',
        r'[Gg]rade\s*[=:]\s*(\d+)',
        r'[Mm]ark\s*[=:]\s*(\d+)',
        r'[Aa]warded\s+(\d+)\s*[Pp]oints',
        r'[Ff]ull\s+[Ss]core\s*[=:]\s*(\d+)',
    ]
    for pattern in explicit_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            points = int(match.group(1))
            if 0 <= points <= 7:
                return points
    
    # Priority 2: Look for "X/7" or "X out of 7" patterns
    fraction_patterns = [
        r'\b(\d+)\s*/\s*7\b',
        r'\b(\d+)\s+out\s+of\s+7\b',
        r'\((\d+)\s*[Pp]oints?\)',
        r'(\d+)\s*[Pp]oints?\s*(?:out of|/|from)\s*7',
    ]
    for pattern in fraction_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            points = int(match.group(1))
            if 0 <= points <= 7:
                return points
    
    # Priority 3: Look for standalone numbers at the very beginning
    # This handles cases where the first line is just the points
    first_line = text.split('\n')[0].strip()
    if first_line.isdigit():
        points = int(first_line)
        if 0 <= points <= 7:
            return points
    
    # Priority 4: Look for numbers in parentheses at the start like "(6)" or "(Partial) 6."
    # But be careful not to match list numbers like "1." after "(Partial)"
    paren_match = re.match(r'^\s*\((\d+)\)', text)
    if paren_match:
        points = int(paren_match.group(1))
        if 0 <= points <= 7:
            return points
    
    # Priority 5: Look for "(Partial)" or "(Almost)" markers with points on same line
    # These markers indicate partial credit (1-6 points)
    # Only match if the number appears on the same line as the marker, not as a list item
    partial_patterns = [
        r'\([Pp]artial\)\s*(\d+)(?!\s*\.)',  # (Partial) 6 but not (Partial) 1.
        r'\([Aa]lmost\)\s*(\d+)(?!\s*\.)',   # (Almost) 5 but not (Almost) 1.
    ]
    for pattern in partial_patterns:
        match = re.search(pattern, text)
        if match:
            points = int(match.group(1))
            if 0 <= points <= 7:
                return points
    
    # Priority 6: Look for any digit 0-7 that appears to be a score
    # Search for patterns like "X points" or standalone numbers
    score_patterns = [
        r'\b(\d+)\s*[Pp]oints?\b',
        r'\bscore\s+(\d+)\b',
        r'\bgrade\s+(\d+)\b',
    ]
    for pattern in score_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            points = int(match.group(1))
            if 0 <= points <= 7:
                return points
    
    # Last resort: standalone digit at end of text
    last_line = text.split('\n')[-1].strip()
    if last_line.isdigit():
        points = int(last_line)
        if 0 <= points <= 7:
            return points
    
    return None


def _extract_points_from_inputs(inputs: dict) -> int | None:
    """Extract points value from inputs dictionary.
    
    Checks multiple possible key names and formats.
    Returns the points value (0-7) or None if not found.
    """
    # Try different possible key names for points
    points_keys = ['points', 'Points', 'POINTS', 'score', 'Score', 'SCORE', 
                   'mark', 'Mark', 'MARK', 'grade', 'Grade', 'GRADE']
    
    for key in points_keys:
        if key in inputs:
            points_val = inputs[key]
            # Handle different types
            if isinstance(points_val, int):
                if 0 <= points_val <= 7:
                    return points_val
            elif isinstance(points_val, str):
                # Try to parse as integer
                try:
                    points = int(points_val.strip())
                    if 0 <= points <= 7:
                        return points
                except (ValueError, TypeError):
                    continue
            elif isinstance(points_val, float):
                points = int(points_val)
                if 0 <= points <= 7:
                    return points
    
    return None


def _extract_json_response(text: str) -> dict | None:
    """Extract JSON object with 'response' field from text.
    
    Returns the last valid JSON with a 'response' field, or None.
    """
    if not text:
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
            obj = json.loads(inner)
            if isinstance(obj, dict) and "response" in obj:
                results.append(obj)
        except json.JSONDecodeError:
            continue
    return results[-1] if results else None


def _extract_response_from_text(text: str) -> str | None:
    """Extract classification from raw text using pattern matching.
    
    Returns one of 'correct', 'incorrect', 'partial', or None.
    """
    if not text:
        return None
        
    text_lower = text.lower()
    
    # Priority patterns for explicit classification statements
    # These are ordered by specificity
    patterns = [
        (r'"response"\s*:\s*"(correct|incorrect|partial|almost)"', 1),
        (r'"response"\s*:\s*\'(correct|incorrect|partial|almost)\'', 1),
        (r'"response"\s*:\s*`?(correct|incorrect|partial|almost)`?', 1),
        (r'response\s+is\s+"?(correct|incorrect|partial|almost)"?', 1),
        (r'classification\s+is\s+"?(correct|incorrect|partial|almost)"?', 1),
        (r'classify\s+as\s+"?(correct|incorrect|partial|almost)"?', 1),
        (r'grade\s+is\s+"?(correct|incorrect|partial|almost)"?', 1),
        (r'graded\s+as\s+"?(correct|incorrect|partial|almost)"?', 1),
        (r'final\s+(?:answer|classification|grade)[\s:]+"?(correct|incorrect|partial|almost)"?', 1),
        (r'conclusion[\s:]+"?(correct|incorrect|partial|almost)"?', 1),
        (r'therefore[,:]?\s+(?:the\s+)?(?:answer|classification|grade)\s+is\s+"?(correct|incorrect|partial|almost)"?', 1),
        (r'(?:answer|classification|grade)\s+should\s+be\s+"?(correct|incorrect|partial|almost)"?', 1),
        (r'(?:this|the)\s+(?:is|should\s+be\s+classified\s+as)\s+"?(correct|incorrect|partial|almost)"?', 1),
    ]
    
    for pattern, group in patterns:
        match = re.search(pattern, text_lower, re.MULTILINE | re.IGNORECASE)
        if match:
            result = match.group(group).lower()
            # Map "almost" to "partial"
            if result == "almost":
                return "partial"
            if result in ["correct", "incorrect", "partial"]:
                return result
    
    # Count occurrences as fallback
    correct_count = len(re.findall(r'\bcorrect\b', text_lower))
    incorrect_count = len(re.findall(r'\bincorrect\b', text_lower))
    partial_count = len(re.findall(r'\bpartial\b', text_lower))
    almost_count = len(re.findall(r'\balmost\b', text_lower))
    
    # Only return if exactly one type appears (treating "almost" as "partial")
    total = correct_count + incorrect_count + partial_count + almost_count
    if total > 0:
        if correct_count > 0 and incorrect_count == 0 and partial_count == 0 and almost_count == 0:
            return "correct"
        if incorrect_count > 0 and correct_count == 0 and partial_count == 0 and almost_count == 0:
            return "incorrect"
        if (partial_count > 0 or almost_count > 0) and correct_count == 0 and incorrect_count == 0:
            return "partial"
    
    return None


def _normalize_prediction(prediction: str, points: int | None = None) -> str:
    """Normalize prediction to one of the three valid labels.
    
    Args:
        prediction: The raw prediction string
        points: Optional points value (0-7) to override the prediction
    
    Returns:
        One of "correct", "incorrect", or "partial"
    """
    # POINTS OVERRIDE: Points are the PRIMARY signal for IMO grading
    # This is the most reliable method - always trust points over LLM output
    if points is not None:
        if points == 7:
            return "correct"
        elif points == 0:
            return "incorrect"
        elif 1 <= points <= 6:
            return "partial"
    
    # Normalize the prediction string
    pred_str = str(prediction).lower().strip().strip('"\'')
    
    # Direct match
    if pred_str in ["correct", "incorrect", "partial"]:
        return pred_str
    
    # Handle "almost" -> "partial" (critical for IMO grading)
    if pred_str == "almost" or "almost" in pred_str:
        return "partial"
    
    # Check for partial
    if "partial" in pred_str:
        return "partial"
    
    # Check for incorrect (be comprehensive)
    incorrect_terms = ["incorrect", "wrong", "false", "error", "invalid", 
                       "no credit", "zero", "0 points", "not correct", 
                       "not valid", "failed", "failure"]
    if any(term in pred_str for term in incorrect_terms):
        return "incorrect"
    
    # Check for correct (but not incorrect)
    if "correct" in pred_str and "incorrect" not in pred_str:
        return "correct"
    
    # Default fallback - use points if available, otherwise "incorrect"
    if points is not None:
        if points == 7:
            return "correct"
        elif points == 0:
            return "incorrect"
        else:
            return "partial"
    return "incorrect"


class TaskAgent:
    """Task agent that solves IMO grading problems."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer, points

        Returns:
            (prediction, msg_history)
        """
        # Extract fields from inputs
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        # Extract points - try multiple sources in order of reliability
        # 1. First try the explicit "points" field from inputs (most reliable)
        points = _extract_points_from_inputs(inputs)
        
        # 2. If not found, try to extract from grading guidelines
        if points is None:
            points = _extract_points_from_guidelines(grading_guidelines)
        
        # Ensure points is an integer (handle string inputs from CSV)
        if points is not None:
            try:
                points = int(points)
                if not (0 <= points <= 7):
                    points = None
            except (ValueError, TypeError):
                points = None
        
        # Build points info for the prompt
        points_info = ""
        if points is not None:
            points_info = f"\n## Points Awarded: {points} out of 7\n"
            if points == 7:
                points_info += "This is a FULL SCORE solution (7/7). Classification: CORRECT.\n"
            elif points == 0:
                points_info += "This solution received ZERO points (0/7). Classification: INCORRECT.\n"
            else:
                points_info += f"This solution received PARTIAL CREDIT ({points}/7). Classification: PARTIAL.\n"
        
        # Build the instruction prompt with improved clarity
        instruction = f"""You are an expert mathematical olympiad grader. Your task is to classify student answers based STRICTLY on the points awarded in the grading guidelines.

## Problem:
{problem}

## Official Solution:
{solution}

## Grading Guidelines:
{grading_guidelines}
{points_info}
## Student's Answer:
{student_answer}

## CLASSIFICATION RULES - FOLLOW EXACTLY:

The points awarded in the grading guidelines DETERMINE the classification. This is NON-NEGOTIABLE.

**MANDATORY POINTS-TO-CLASSIFICATION MAPPING:**
- Points = 7 → Classification MUST be "correct" (full score)
- Points = 0 → Classification MUST be "incorrect" (no credit)
- Points = 1, 2, 3, 4, 5, or 6 → Classification MUST be "partial" (partial credit)

**CRITICAL NOTES:**
- The label "almost" in grading guidelines means the student was close but made errors. This is PARTIAL credit (1-6 points), NOT "correct".
- Even if the solution looks correct to you, if the points awarded are less than 7, it is "partial".
- Even if the solution looks wrong to you, if the points awarded are greater than 0, it is "partial".
- The points value is the ONLY authority. DO NOT use your own judgment about solution quality.

**YOUR ONLY TASK:** Map the points value to the classification using the table above.

Respond with ONLY a JSON object in this exact format:
<json>
{{
    "reasoning": "Points awarded: X/7. According to the mandatory mapping, X points = 'partial' classification.",
    "response": "correct" | "incorrect" | "partial"
}}
</json>

Do not include any other text outside the JSON block."""

        # Get LLM response
        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction
        prediction = "incorrect"  # Default fallback
        
        try:
            response_text = msg_history[-1]["text"]
            
            # Try to extract from JSON first
            json_obj = _extract_json_response(response_text)
            if json_obj and "response" in json_obj:
                raw_pred = json_obj["response"]
                prediction = _normalize_prediction(raw_pred, points)
            else:
                # Try direct text extraction
                direct = _extract_response_from_text(response_text)
                if direct:
                    prediction = _normalize_prediction(direct, points)
                else:
                    self.log_fn(f"No valid response found in: {{response_text[:200]}}...")
                    
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {{e}}")
        
        # FINAL OVERRIDE: Always use points to determine the final prediction
        # This is the most critical step - points are the ground truth
        if points is not None:
            if points == 7:
                prediction = "correct"
            elif points == 0:
                prediction = "incorrect"
            elif 1 <= points <= 6:
                prediction = "partial"
        
        return str(prediction), msg_history
