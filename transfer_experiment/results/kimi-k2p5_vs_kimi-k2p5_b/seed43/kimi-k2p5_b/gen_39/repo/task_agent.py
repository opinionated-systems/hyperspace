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
    This is the most reliable way to determine the correct classification.
    """
    if not grading_guidelines:
        return None
        
    text = grading_guidelines.strip()
    text_lower = text.lower()
    
    # First priority: Look for explicit point values like "7 points", "0 points", etc.
    # These are the most reliable indicators
    explicit_patterns = [
        r'\b(\d+)\s*/\s*7\b',  # X/7 format
        r'\b(\d+)\s+out\s+of\s*7\b',  # X out of 7 format
        r'[Aa]warded\s+(\d+)\s*[Pp]oints?\b',  # Awarded X points
        r'[Ss]core[d]?\s*[=:]?\s*(\d+)\s*[Pp]oints?\b',  # Score: X points
        r'\b(\d+)\s*[Pp]oints?\s*(?:out of|/|from)\s*7\b',  # X points out of 7
        r'[Ff]ull\s+[Ss]core[:\s]+(\d+)\b',  # Full score: X
        r'[Gg]rade[d]?\s*[:\s]+(\d+)\b',  # Grade: X
        r'[Mm]ark[s]?\s*[:\s]+(\d+)\b',  # Marks: X
        r'\((\d+)\s*points?\)',  # (X points) format
        r'\bpoints?\s*[:=]\s*(\d+)\b',  # points: X or points = X
        r'\b(\d+)\s*pts?\b',  # X pts or X pt format
    ]
    for pattern in explicit_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            points = int(match.group(1))
            if 0 <= points <= 7:
                return points
    
    # Second priority: Look for standalone numbers 0-7 at the start or end
    standalone_patterns = [
        r'^\s*(\d+)\s*\n',  # Number at start followed by newline
        r'^\s*(\d+)\s*$',    # Standalone number
        r'\n\s*(\d+)\s*$',   # Number at end
    ]
    for pattern in standalone_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            points = int(match.group(1))
            if 0 <= points <= 7:
                return points
    
    # Last resort: if the entire text is just a digit 0-7
    if text.isdigit():
        points = int(text)
        if 0 <= points <= 7:
            return points
    
    # Check for category markers in grading guidelines
    # These indicate the classification level and can be mapped to point ranges
    # Priority: Check for specific point indicators first, then category markers
    
    # Check for (Correct) marker - indicates 7 points
    if re.search(r'\(correct\)', text_lower):
        return 7
    
    # Check for (Almost) marker - indicates 5-6 points (use 6 as representative)
    if re.search(r'\(almost\)', text_lower):
        return 6
    
    # Check for (Partial) marker - indicates 1-4 points (use 2 as representative)
    if re.search(r'\(partial\)', text_lower):
        return 2
    
    # Check for (Incorrect) marker - indicates 0 points
    if re.search(r'\(incorrect\)', text_lower):
        return 0
    
    # Check for point value indicators in text
    # Look for patterns like "X points" where X is 0-7
    point_value_patterns = [
        r'\b([0-7])\s*points?\b',
        r'\bpoints?\s*[:=]\s*([0-7])\b',
    ]
    for pattern in point_value_patterns:
        match = re.search(pattern, text_lower)
        if match:
            return int(match.group(1))
    
    return None


def _points_to_classification(points: int) -> str:
    """Convert points to classification label.
    
    Based on the IMO grading scheme:
    - 7 points = correct (full score)
    - 5-6 points = almost (minor errors)
    - 1-4 points = partial (significant progress but gaps)
    - 0 points = incorrect (no valid progress)
    """
    if points == 7:
        return "correct"
    elif 5 <= points <= 6:
        return "almost"
    elif 1 <= points <= 4:
        return "partial"
    else:
        return "incorrect"


def _is_valid_label(text: str) -> bool:
    """Check if text is a valid classification label (not math/problem content).
    
    This function is STRICT - it only returns True for text that is exactly
    or very close to one of the four valid labels. Any text that contains
    mathematical content, reasoning, or other extraneous text is rejected.
    """
    if not text:
        return False
    
    text_str = str(text).strip()
    text_lower = text_str.lower()
    
    # If it's exactly one of the four labels, it's valid
    if text_lower in ["correct", "incorrect", "partial", "almost"]:
        return True
    
    # Clean up common variations: remove quotes, extra whitespace, trailing punctuation
    cleaned = text_lower.strip('"\'').strip().rstrip('.:,;!?')
    if cleaned in ["correct", "incorrect", "partial", "almost"]:
        return True
    
    # STRICT REJECTION: Check for mathematical expressions or problem content
    # These are strong indicators that the text is NOT a valid label
    math_indicators = [
        r'\$\$.*?\$\$',      # LaTeX display math
        r'\$[^$]+\$',        # Inline LaTeX math
        r'\\[a-zA-Z]+',       # LaTeX commands
        r'<step',             # Step tags
        r'<[a-zA-Z]',          # HTML/XML tags
        r'\^\{',              # Exponents in LaTeX
        r'\\frac',            # Fractions
        r'\\sum',             # Sums
        r'\\int',             # Integrals
        r'\\begin',           # LaTeX environments
        r'\\[\[\]\(\)\{\}]', # LaTeX brackets
        r'\{[^}]+\}',         # Curly braces with content
        r'[_\^]',              # Subscripts/superscripts
        r'\d+\s*[\=\+\-\*/]',  # Numbers with operators
        r'[\=\+\-\*/\^]',       # Math operators
    ]
    
    for pattern in math_indicators:
        if re.search(pattern, text_str):
            return False
    
    # STRICT REJECTION: Check for very long text (labels should be short)
    if len(text_str) > 50:
        return False
    
    # Check for newlines - allow single newline, reject multiple
    if text_str.count('\n') > 1 or text_str.count('\r') > 1:
        return False
    
    # Check for multiple sentences or clauses
    if '.' in text_str and len(text_str) > 50:
        return False
    
    # Check for reasoning phrases that indicate non-label content
    reasoning_indicators = [
        'because', 'since', 'therefore', 'thus', 'hence', 'consequently',
        'proof of', 'lemma', 'theorem', 'analysis shows', 'conclusion is',
        'the solution', 'the answer', 'this problem', 'student solution',
    ]
    
    for indicator in reasoning_indicators:
        if indicator in text_lower:
            return False
    
    # Accept if label appears as a standalone word
    for label in ["correct", "incorrect", "partial", "almost"]:
        # Check if label is at the start or end with minimal punctuation
        if re.match(rf'^{label}\s*[:;,.!?]*\s*$', text_lower):
            return True
        if re.search(rf'\b{label}\s*[:;,.!?]*\s*$', text_lower):
            return True
        # Check for label surrounded by word boundaries
        if re.search(rf'\b{label}\b', text_lower) and len(text_str) < 30:
            return True
    
    # Default: reject as invalid
    return False


def _extract_json_from_text(text: str) -> dict | None:
    """Extract JSON object from text using multiple strategies.
    
    This function tries multiple strategies to find and parse JSON objects
    from text, with a focus on finding the "response" field that contains
    the classification label.
    """
    if not text:
        return None
        
    text = text.strip()
    
    # Strategy 1: <json>...</json> tags
    json_tag_pattern = r'<json>\s*(.*?)\s*</json>'
    matches = re.findall(json_tag_pattern, text, re.DOTALL | re.IGNORECASE)
    for match in matches:
        try:
            data = json.loads(match.strip())
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            fixed = _fix_json(match.strip())
            if fixed:
                try:
                    data = json.loads(fixed)
                    if isinstance(data, dict):
                        return data
                except json.JSONDecodeError:
                    pass
    
    # Strategy 2: ```json...``` code blocks
    code_block_pattern = r'```json\s*(.*?)\s*```'
    matches = re.findall(code_block_pattern, text, re.DOTALL | re.IGNORECASE)
    for match in matches:
        try:
            data = json.loads(match.strip())
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            fixed = _fix_json(match.strip())
            if fixed:
                try:
                    data = json.loads(fixed)
                    if isinstance(data, dict):
                        return data
                except json.JSONDecodeError:
                    pass
    
    # Strategy 3: ```...``` code blocks with JSON content
    code_block_pattern2 = r'```\s*(\{[\s\S]*?\})\s*```'
    matches = re.findall(code_block_pattern2, text, re.DOTALL | re.IGNORECASE)
    for match in matches:
        try:
            data = json.loads(match.strip())
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            fixed = _fix_json(match.strip())
            if fixed:
                try:
                    data = json.loads(fixed)
                    if isinstance(data, dict):
                        return data
                except json.JSONDecodeError:
                    pass
    
    # Strategy 4: Raw JSON objects (non-greedy matching)
    # Try to find JSON objects that contain a "response" field
    json_pattern = r'\{[^{}]*"response"[^{}]*\}'
    matches = re.findall(json_pattern, text, re.DOTALL)
    for match in matches:
        try:
            data = json.loads(match)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            fixed = _fix_json(match)
            if fixed:
                try:
                    data = json.loads(fixed)
                    if isinstance(data, dict):
                        return data
                except json.JSONDecodeError:
                    pass
    
    # Strategy 5: More aggressive JSON object extraction
    # Look for balanced braces with "response" field
    json_pattern2 = r'\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}'
    matches = re.findall(json_pattern2, text, re.DOTALL)
    for match in matches:
        if '"response"' in match or "'response'" in match:
            try:
                data = json.loads(match)
                if isinstance(data, dict):
                    return data
            except json.JSONDecodeError:
                fixed = _fix_json(match)
                if fixed:
                    try:
                        data = json.loads(fixed)
                        if isinstance(data, dict):
                            return data
                    except json.JSONDecodeError:
                        pass
    
    # Strategy 6: Look for any JSON-like object with common fields
    json_pattern3 = r'\{[\s\S]*?"(?:response|classification|grade|result|evaluation|verdict|outcome)"[\s\S]*?\}'
    matches = re.findall(json_pattern3, text, re.DOTALL)
    for match in matches:
        try:
            data = json.loads(match)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            fixed = _fix_json(match)
            if fixed:
                try:
                    data = json.loads(fixed)
                    if isinstance(data, dict):
                        return data
                except json.JSONDecodeError:
                    pass
    
    return None


def _fix_json(text: str) -> str | None:
    """Try to fix common JSON formatting issues."""
    if not text or not text.strip():
        return None
    
    text = text.strip()
    
    # Fix 1: Remove trailing commas before closing braces/brackets
    text = re.sub(r',\s*}', '}', text)
    text = re.sub(r',\s*\]', ']', text)
    
    # Fix 2: Replace single quotes with double quotes (simple cases)
    if "'" in text and '"' not in text:
        text = text.replace("'", '"')
    
    # Fix 3: Handle escaped quotes that might cause issues
    # Don't unescape if they're properly escaped
    
    # Fix 4: Ensure the text starts with { and ends with }
    if not text.startswith('{'):
        start = text.find('{')
        if start != -1:
            text = text[start:]
    
    if not text.endswith('}'):
        end = text.rfind('}')
        if end != -1:
            text = text[:end+1]
    
    # Fix 5: Handle newlines in string values (replace with spaces)
    text = re.sub(r'(?<=")\n(?=")', ' ', text)
    text = re.sub(r'(?<=")\r\n(?=")', ' ', text)
    
    # Fix 6: Handle unescaped quotes within string values
    # This is a simplified fix - replace problematic patterns
    text = re.sub(r'(?<=")\s*"\s*(?=")', ' ', text)
    
    # Fix 7: Handle control characters in strings
    text = re.sub(r'[\x00-\x1F\x7F]', ' ', text)
    
    # Fix 8: Handle multiple consecutive spaces
    text = re.sub(r' +', ' ', text)
    
    return text if text.startswith('{') and text.endswith('}') else None


def _extract_response_direct(text: str) -> str | None:
    """Extract response by looking for keywords directly in text.
    
    This function is designed to extract classification labels from LLM responses.
    It prioritizes JSON field patterns and explicit classification statements.
    """
    if not text:
        return None
        
    text_lower = text.lower()
    
    # First, try to find explicit JSON-like field patterns (most reliable)
    # Look for "response": "label" pattern first
    json_response_pattern = r'"response"\s*:\s*"([^"]+)"'
    match = re.search(json_response_pattern, text_lower)
    if match:
        response_value = match.group(1).strip().lower().rstrip('.:,;!?')
        if response_value in ["correct", "almost", "partial", "incorrect"]:
            return response_value
    
    # Look for other common JSON field patterns
    json_field_patterns = [
        r'"classification"\s*:\s*"([^"]+)"',
        r'"grade"\s*:\s*"([^"]+)"',
        r'"result"\s*:\s*"([^"]+)"',
        r'"evaluation"\s*:\s*"([^"]+)"',
        r'"verdict"\s*:\s*"([^"]+)"',
        r'"outcome"\s*:\s*"([^"]+)"',
    ]
    for pattern in json_field_patterns:
        match = re.search(pattern, text_lower)
        if match:
            value = match.group(1).strip().lower().rstrip('.:,;!?')
            if value in ["correct", "almost", "partial", "incorrect"]:
                return value
    
    # Look for single-quoted JSON values
    json_single_quote_pattern = r"'response'\s*:\s*'([^']+)'"
    match = re.search(json_single_quote_pattern, text_lower)
    if match:
        value = match.group(1).strip().lower().rstrip('.:,;!?')
        if value in ["correct", "almost", "partial", "incorrect"]:
            return value
    
    # Look for explicit classification statements with priority
    # Priority: almost > partial > incorrect > correct (most specific first)
    
    # Check for "almost" first (most specific)
    almost_patterns = [
        r'\bclassif(?:y|ied)\s+as\s+almost\b',
        r'\bthis\s+is\s+almost\b',
        r'\bgrade[d]?\s+as\s+almost\b',
        r'\bis\s+almost\s+correct\b',
        r'\bfinal\s+(?:classification|grade|evaluation)\s*[:\-]?\s*almost\b',
        r'\bprediction\s*[:\-]?\s*almost\b',
        r'\boutput\s*[:\-]?\s*almost\b',
    ]
    for pattern in almost_patterns:
        if re.search(pattern, text_lower):
            return "almost"
    
    # Check for "partial"
    partial_patterns = [
        r'\bclassif(?:y|ied)\s+as\s+partial\b',
        r'\bthis\s+is\s+partial\b',
        r'\bgrade[d]?\s+as\s+partial\b',
        r'\bfinal\s+(?:classification|grade|evaluation)\s*[:\-]?\s*partial\b',
        r'\bprediction\s*[:\-]?\s*partial\b',
        r'\boutput\s*[:\-]?\s*partial\b',
    ]
    for pattern in partial_patterns:
        if re.search(pattern, text_lower):
            return "partial"
    
    # Check for "incorrect"
    incorrect_patterns = [
        r'\bclassif(?:y|ied)\s+as\s+incorrect\b',
        r'\bthis\s+is\s+incorrect\b',
        r'\bgrade[d]?\s+as\s+incorrect\b',
        r'\bfinal\s+(?:classification|grade|evaluation)\s*[:\-]?\s*incorrect\b',
        r'\bprediction\s*[:\-]?\s*incorrect\b',
        r'\boutput\s*[:\-]?\s*incorrect\b',
    ]
    for pattern in incorrect_patterns:
        if re.search(pattern, text_lower):
            return "incorrect"
    
    # Check for "correct"
    correct_patterns = [
        r'\bclassif(?:y|ied)\s+as\s+correct\b',
        r'\bthis\s+is\s+correct\b',
        r'\bgrade[d]?\s+as\s+correct\b',
        r'\bis\s+correct\b',
        r'\bfinal\s+(?:classification|grade|evaluation)\s*[:\-]?\s*correct\b',
        r'\bprediction\s*[:\-]?\s*correct\b',
        r'\boutput\s*[:\-]?\s*correct\b',
    ]
    for pattern in correct_patterns:
        if re.search(pattern, text_lower):
            return "correct"
    
    # Count standalone occurrences as fallback (only in the first 300 chars to avoid math content)
    text_sample = text_lower[:300]
    almost_count = len(re.findall(r'\balmost\b', text_sample))
    partial_count = len(re.findall(r'\bpartial\b', text_sample))
    incorrect_count = len(re.findall(r'\bincorrect\b', text_sample))
    correct_count = len(re.findall(r'\bcorrect\b', text_sample))
    
    # If only one appears, use that
    if almost_count > 0 and partial_count == 0 and incorrect_count == 0 and correct_count == 0:
        return "almost"
    if partial_count > 0 and almost_count == 0 and incorrect_count == 0 and correct_count == 0:
        return "partial"
    if incorrect_count > 0 and almost_count == 0 and partial_count == 0 and correct_count == 0:
        return "incorrect"
    if correct_count > 0 and almost_count == 0 and partial_count == 0 and incorrect_count == 0:
        return "correct"
    
    # Priority-based selection when multiple appear (most specific first)
    # "almost" is most specific, then "partial", then "incorrect", then "correct"
    if almost_count > 0:
        return "almost"
    if partial_count > 0:
        return "partial"
    if incorrect_count > 0:
        return "incorrect"
    if correct_count > 0:
        return "correct"
    
    return None


def _normalize_prediction(prediction: str, points: int | None = None) -> str:
    """Normalize prediction to one of the four valid labels.
    
    Args:
        prediction: The raw prediction string from the LLM
        points: Optional points value from grading guidelines (0-7)
    
    Returns:
        One of "correct", "almost", "partial", or "incorrect"
    """
    if not prediction:
        # If we have points, use them as the ground truth
        if points is not None:
            return _points_to_classification(points)
        return "incorrect"
    
    pred_str = str(prediction).strip()
    pred_lower = pred_str.lower()
    
    # Direct match (exact)
    if pred_lower in ["correct", "incorrect", "partial", "almost"]:
        return pred_lower
    
    # Check for exact match after removing quotes and extra whitespace
    pred_clean = pred_lower.strip('"\'').strip().rstrip('.:,;!?')
    if pred_clean in ["correct", "incorrect", "partial", "almost"]:
        return pred_clean
    
    # Extract from text using the direct extraction function
    extracted = _extract_response_direct(pred_str)
    if extracted:
        return extracted
    
    # Check for label as a word boundary match
    for label in ["correct", "incorrect", "partial", "almost"]:
        if re.search(rf'\b{label}\b', pred_lower):
            return label
    
    # Priority: almost > partial > incorrect > correct (most specific first)
    # This order helps avoid misclassification when multiple words appear
    if "almost" in pred_lower:
        return "almost"
    if "partial" in pred_lower:
        return "partial"
    if "incorrect" in pred_lower:
        return "incorrect"
    if "correct" in pred_lower:
        return "correct"
    
    # If we have points, use them as the ground truth
    if points is not None:
        return _points_to_classification(points)
    
    return "incorrect"


def _check_grading_guidelines_for_almost(grading_guidelines: str) -> bool:
    """Check if grading guidelines indicate 'almost' level.
    
    Returns True if guidelines suggest the solution is nearly complete
    with only minor issues.
    """
    if not grading_guidelines:
        return False
    
    guidelines_lower = grading_guidelines.lower()
    
    # Check for explicit point indicators of "almost" level (5-6 points)
    almost_point_indicators = [
        r'\b5[-\s]?6\s+points\b',
        r'\b6\s+points\b',
        r'\b5\s+points\b',
    ]
    
    for pattern in almost_point_indicators:
        if re.search(pattern, guidelines_lower):
            return True
    
    # Check for (Almost) marker anywhere in guidelines
    if "(almost)" in guidelines_lower:
        return True
    
    # Check for keywords suggesting minor issues only
    minor_issue_keywords = [
        "minor mistakes",
        "minor errors",
        "small errors",
        "technical errors",
        "verification contains",
        "almost complete",
        "nearly complete",
    ]
    
    for keyword in minor_issue_keywords:
        if keyword in guidelines_lower:
            return True
    
    return False


def _check_grading_guidelines_for_partial(grading_guidelines: str) -> bool:
    """Check if grading guidelines strongly indicate 'partial' level.
    
    Returns True if guidelines explicitly indicate partial level
    through point values or clear statements of significant gaps.
    """
    if not grading_guidelines:
        return False
    
    guidelines_lower = grading_guidelines.lower()
    
    # Check for explicit point indicators of "partial" level (1-4 points)
    partial_point_indicators = [
        r'\b1[-\s]?4\s+points\b',
        r'\b2[-\s]?3\s+points\b',
        r'\b3[-\s]?4\s+points\b',
        r'\b4\s+points\b',
        r'\b3\s+points\b',
        r'\b2\s+points\b',
        r'\b1\s+point\b',
    ]
    
    for pattern in partial_point_indicators:
        if re.search(pattern, guidelines_lower):
            return True
    
    # Check for keywords suggesting significant gaps
    significant_gap_keywords = [
        "significant gaps",
        "major gaps",
        "incomplete proof",
        "missing key steps",
        "far from complete",
        "only partial progress",
    ]
    
    for keyword in significant_gap_keywords:
        if keyword in guidelines_lower:
            return True
    
    return False


def _check_grading_guidelines_for_correct(grading_guidelines: str) -> bool:
    """Check if grading guidelines explicitly indicate 'correct' level.
    
    Returns True if guidelines explicitly indicate correct level
    through point values (7 points) or clear statements.
    """
    if not grading_guidelines:
        return False
    
    guidelines_lower = grading_guidelines.lower()
    
    # Check for explicit 7 points indicator
    if re.search(r'\b7\s+points?\b', guidelines_lower):
        return True
    
    # Check for explicit (Correct) marker
    if "(correct)" in guidelines_lower:
        return True
    
    # Check for strong indicators of perfect solution
    perfect_keywords = [
        "complete solution",
        "correct solution",
        "full marks",
        "full credit",
        "perfect solution",
        "7/7",
        "seven points",
    ]
    
    for keyword in perfect_keywords:
        if keyword in guidelines_lower:
            return True
    
    return False


def _check_grading_guidelines_for_incorrect(grading_guidelines: str) -> bool:
    """Check if grading guidelines strongly indicate 'incorrect' level.
    
    Returns True if guidelines explicitly indicate incorrect level
    through point values (0 points) or clear statements of no progress.
    """
    if not grading_guidelines:
        return False
    
    guidelines_lower = grading_guidelines.lower()
    
    # Check for explicit 0 points indicator
    if re.search(r'\b0\s+points?\b', guidelines_lower):
        return True
    
    # Check for explicit (Incorrect) marker
    if "(incorrect)" in guidelines_lower:
        return True
    
    # Check for strong indicators of no valid progress
    no_progress_keywords = [
        "no valid lemmas",
        "no non-trivial observations",
        "fundamentally flawed",
        "no meaningful progress",
        "completely wrong",
        "trivial only",
        "zero points",
        "0/7",
        "no progress",
    ]
    
    for keyword in no_progress_keywords:
        if keyword in guidelines_lower:
            return True
    
    return False


class TaskAgent:
    """Task agent that solves IMO grading problems."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.
        
        The primary approach is to use the Points field from inputs
        and map them directly to classification labels, as this is the
        most reliable method. If Points is not available, fall back to
        extracting from grading guidelines.
        """
        # Extract key fields from inputs
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        # First priority: Use the Points field directly if available (ground truth)
        # Check both "Points" (capital P) and "points" (lowercase)
        points = None
        for key in ["Points", "points", "POINTS"]:
            if key in inputs:
                try:
                    points_val = inputs[key]
                    if isinstance(points_val, (int, float)):
                        points = int(points_val)
                    elif isinstance(points_val, str):
                        points = int(float(points_val.strip()))
                    if points is not None and 0 <= points <= 7:
                        prediction = _points_to_classification(points)
                        self.log_fn(f"Points from input field '{key}': {points}/7 → Classification: {prediction}")
                        return str(prediction), []
                except (ValueError, TypeError):
                    points = None
                    continue
        
        # Second priority: Extract points from grading guidelines
        if points is None:
            points = _extract_points_from_guidelines(grading_guidelines)
        
        # If we successfully extracted points, use them directly
        if points is not None:
            prediction = _points_to_classification(points)
            self.log_fn(f"Points extracted from guidelines: {points}/7 → Classification: {prediction}")
            return str(prediction), []
        
        # Fallback: Use LLM-based classification if points extraction failed
        # This should rarely happen with the improved extraction patterns
        instruction = f"""You are an expert mathematical olympiad grader. Your task is to classify the student's answer into EXACTLY ONE of four categories: "correct", "almost", "partial", or "incorrect".

## Problem Statement:
{problem}

## Official Solution:
{solution}

## Grading Guidelines:
{grading_guidelines}

## Student's Answer:
{student_answer}

## Classification Rules (READ CAREFULLY):

**"correct"** (7 points): PERFECT solution. Complete, valid reasoning throughout, correct final answer, no gaps or errors. The solution could be published as-is. Use this when the solution is truly flawless.

**"almost"** (5-6 points): Nearly complete solution. Main proof structure is sound and complete. Only minor technical errors (typos, small calculation errors, NOT conceptual gaps). The core mathematical argument is complete and correct, just needs minor polishing.

**"partial"** (1-4 points): Meaningful progress (valid lemmas, right approach) but SIGNIFICANT gaps remain. The solution is far from complete. Student made some real progress but major work is missing or there are serious conceptual errors.

**"incorrect"** (0 points): No valid lemmas or non-trivial observations. Fundamentally flawed or trivial only. No meaningful mathematical progress.

## IMPORTANT GUIDANCE:
- Look at the grading guidelines carefully - they indicate the point value
- If guidelines say "(Correct)" or indicate 7 points → "correct"
- If guidelines say "(Almost)" or indicate 5-6 points → "almost"
- If guidelines say "(Partial)" or indicate 1-4 points → "partial"
- If guidelines say "(Incorrect)" or indicate 0 points → "incorrect"

## OUTPUT FORMAT (STRICT JSON - MANDATORY):
You MUST output ONLY a JSON object. The response field MUST contain EXACTLY ONE of these four words: "correct", "almost", "partial", or "incorrect". NO OTHER TEXT is allowed in the response field.

```json
{{
    "reasoning": "Your detailed analysis here.",
    "response": "almost"
}}
```

Replace "almost" with "correct", "partial", or "incorrect" as appropriate."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from response
        prediction = "incorrect"  # Default fallback
        try:
            if msg_history and len(msg_history) > 0:
                response_text = msg_history[-1].get("text", "")
                
                if response_text:
                    # First try: extract from JSON
                    json_data = _extract_json_from_text(response_text)
                    
                    if json_data and isinstance(json_data, dict):
                        # Look for response field first
                        if "response" in json_data:
                            pred_val = json_data["response"]
                            # Validate the prediction value
                            if isinstance(pred_val, str) and _is_valid_label(pred_val):
                                prediction = _normalize_prediction(pred_val, points)
                            else:
                                # Try to extract from the invalid value
                                extracted = _extract_response_direct(str(pred_val))
                                if extracted:
                                    prediction = extracted
                                elif points is not None:
                                    prediction = _points_to_classification(points)
                        else:
                            # Check other common fields
                            for field in ["classification", "grade", "result", "evaluation", "verdict", "outcome"]:
                                if field in json_data:
                                    pred_val = json_data[field]
                                    if isinstance(pred_val, str) and _is_valid_label(pred_val):
                                        prediction = _normalize_prediction(pred_val, points)
                                        break
                                    else:
                                        extracted = _extract_response_direct(str(pred_val))
                                        if extracted:
                                            prediction = extracted
                                            break
                    
                    # If JSON extraction failed or didn't yield a valid prediction, try direct extraction
                    if prediction == "incorrect":
                        direct = _extract_response_direct(response_text)
                        if direct:
                            prediction = direct
                        elif points is not None:
                            # Fall back to points-based classification
                            prediction = _points_to_classification(points)
                            
                    # Log if we still don't have a valid prediction
                    if prediction == "incorrect" and not _is_valid_label(response_text[:200]):
                        self.log_fn(f"No valid response found in: {response_text[:200]}...")
                else:
                    self.log_fn("Empty response text in msg_history")
            else:
                self.log_fn("Empty msg_history")
                
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            import traceback
            self.log_fn(traceback.format_exc())
            # On exception, try to use points if available
            if points is not None:
                prediction = _points_to_classification(points)

        # Final validation - ensure prediction is one of the four valid labels
        if prediction not in ["correct", "incorrect", "partial", "almost"]:
            # Try to fall back to points-based classification
            if points is not None:
                prediction = _points_to_classification(points)
            else:
                prediction = "incorrect"

        return str(prediction), msg_history
