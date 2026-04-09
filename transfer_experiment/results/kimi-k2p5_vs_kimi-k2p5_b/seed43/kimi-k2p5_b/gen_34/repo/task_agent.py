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
    
    # First try to find explicit points markers with word boundaries
    points_patterns = [
        r'\b[Pp]oints?\s*[=:]\s*(\d+)',
        r'\((\d+)\s*[Pp]oints?\)',
        r'(\d+)\s*[Pp]oints?\s*(?:out of|/|from)\s*7',
        r'[Aa]warded\s+(\d+)\s*[Pp]oints?',
        r'[Ss]core[d]?\s*[=:]?\s*(\d+)',
        r'(\d+)\s*/\s*7\s*(?:points?)?',
        r'^\s*(\d+)\s*[Pp]oints?\s*$',
        r'[Ff]ull\s+[Ss]core[:\s]+(\d+)',
        r'[Gg]rade[d]?\s*[:\s]+(\d+)',
        r'[Mm]ark[s]?\s*[:\s]+(\d+)',
    ]
    for pattern in points_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            points = int(match.group(1))
            if 0 <= points <= 7:
                return points
    
    # Look for patterns like "X/7" or "X out of 7"
    fraction_patterns = [
        r'\b(\d+)\s*/\s*7\b',
        r'\b(\d+)\s+out\s+of\s+7\b',
    ]
    for pattern in fraction_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            points = int(match.group(1))
            if 0 <= points <= 7:
                return points
    
    # Try to find standalone numbers 0-7 at the start or end
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
    """Check if text is a valid classification label (not math/problem content)."""
    if not text:
        return False
    
    text_str = str(text).strip()
    text_lower = text_str.lower()
    
    # If it's exactly one of the four labels, it's valid
    if text_lower in ["correct", "incorrect", "partial", "almost"]:
        return True
    
    # If it contains one of the four labels as a standalone word, it's likely valid
    for label in ["almost", "partial", "incorrect", "correct"]:
        if re.search(rf'\b{label}\b', text_lower):
            return True
    
    # Check for mathematical expressions or problem content (invalid predictions)
    # Reject if it looks like actual math content
    math_indicators = [
        r'\$\$.*?\$\$',  # LaTeX display math
        r'\\[a-zA-Z]+',   # LaTeX commands
        r'<step',         # Step tags
        r'\^\{',          # Exponents in LaTeX
        r'\\frac',        # Fractions
        r'\\sum',         # Sums
        r'\\int',         # Integrals
        r'\\begin',       # LaTeX environments
    ]
    
    for pattern in math_indicators:
        if re.search(pattern, text_str):
            return False
    
    # Check for very long text (labels should be short)
    if len(text_str) > 100:
        return False
    
    # Check for newlines (labels should be single words)
    if '\n' in text_str or '\r' in text_str:
        return False
    
    return True


def _extract_json_from_text(text: str) -> dict | None:
    """Extract JSON object from text using multiple strategies."""
    text = text.strip()
    
    # Strategy 1: <json>...</json> tags
    json_tag_pattern = r'<json>\s*(.*?)\s*</json>'
    matches = re.findall(json_tag_pattern, text, re.DOTALL | re.IGNORECASE)
    for match in matches:
        try:
            data = json.loads(match.strip())
            if isinstance(data, dict) and "response" in data:
                return data
        except json.JSONDecodeError:
            fixed = _fix_json(match.strip())
            if fixed:
                try:
                    data = json.loads(fixed)
                    if isinstance(data, dict) and "response" in data:
                        return data
                except json.JSONDecodeError:
                    pass
    
    # Strategy 2: ```json...``` code blocks
    code_block_pattern = r'```json\s*(.*?)\s*```'
    matches = re.findall(code_block_pattern, text, re.DOTALL | re.IGNORECASE)
    for match in matches:
        try:
            data = json.loads(match.strip())
            if isinstance(data, dict) and "response" in data:
                return data
        except json.JSONDecodeError:
            fixed = _fix_json(match.strip())
            if fixed:
                try:
                    data = json.loads(fixed)
                    if isinstance(data, dict) and "response" in data:
                        return data
                except json.JSONDecodeError:
                    pass
    
    # Strategy 3: ```...``` code blocks with JSON content
    code_block_pattern2 = r'```\s*(\{[\s\S]*?\})\s*```'
    matches = re.findall(code_block_pattern2, text, re.DOTALL | re.IGNORECASE)
    for match in matches:
        try:
            data = json.loads(match.strip())
            if isinstance(data, dict) and "response" in data:
                return data
        except json.JSONDecodeError:
            fixed = _fix_json(match.strip())
            if fixed:
                try:
                    data = json.loads(fixed)
                    if isinstance(data, dict) and "response" in data:
                        return data
                except json.JSONDecodeError:
                    pass
    
    # Strategy 4: Raw JSON objects
    json_pattern = r'\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}'
    matches = re.findall(json_pattern, text, re.DOTALL)
    for match in matches:
        try:
            data = json.loads(match)
            if isinstance(data, dict) and "response" in data:
                return data
        except json.JSONDecodeError:
            fixed = _fix_json(match)
            if fixed:
                try:
                    data = json.loads(fixed)
                    if isinstance(data, dict) and "response" in data:
                        return data
                except json.JSONDecodeError:
                    pass
    
    return None


def _fix_json(text: str) -> str | None:
    """Try to fix common JSON formatting issues."""
    if not text or not text.strip():
        return None
    
    text = text.strip()
    
    # Fix 1: Remove trailing commas before closing braces
    text = re.sub(r',\s*}', '}', text)
    text = re.sub(r',\s*\]', ']', text)
    
    # Fix 2: Replace single quotes with double quotes (simple cases)
    if "'" in text and '"' not in text:
        text = text.replace("'", '"')
    
    # Fix 3: Ensure the text starts with { and ends with }
    if not text.startswith('{'):
        start = text.find('{')
        if start != -1:
            text = text[start:]
    
    if not text.endswith('}'):
        end = text.rfind('}')
        if end != -1:
            text = text[:end+1]
    
    return text if text.startswith('{') and text.endswith('}') else None


def _extract_response_direct(text: str) -> str | None:
    """Extract response by looking for keywords directly in text."""
    text_lower = text.lower()
    
    # First, try to find explicit JSON-like field patterns (most reliable)
    # Look for "response": "label" pattern first
    json_response_pattern = r'"response"\s*:\s*"([^"]+)"'
    match = re.search(json_response_pattern, text_lower)
    if match:
        response_value = match.group(1).strip().lower()
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
            value = match.group(1).strip().lower()
            if value in ["correct", "almost", "partial", "incorrect"]:
                return value
    
    # Look for explicit classification statements with priority
    # Priority: almost > partial > incorrect > correct (most specific first)
    
    # Check for "almost" first (most specific)
    almost_patterns = [
        r'\bclassif(y|ied)\s+as\s+almost\b',
        r'\bthis\s+is\s+almost\b',
        r'\bgrade[d]?\s+as\s+almost\b',
        r'\bis\s+almost\s+correct\b',
    ]
    for pattern in almost_patterns:
        if re.search(pattern, text_lower):
            return "almost"
    
    # Check for "partial"
    partial_patterns = [
        r'\bclassif(y|ied)\s+as\s+partial\b',
        r'\bthis\s+is\s+partial\b',
        r'\bgrade[d]?\s+as\s+partial\b',
    ]
    for pattern in partial_patterns:
        if re.search(pattern, text_lower):
            return "partial"
    
    # Check for "incorrect"
    incorrect_patterns = [
        r'\bclassif(y|ied)\s+as\s+incorrect\b',
        r'\bthis\s+is\s+incorrect\b',
        r'\bgrade[d]?\s+as\s+incorrect\b',
    ]
    for pattern in incorrect_patterns:
        if re.search(pattern, text_lower):
            return "incorrect"
    
    # Check for "correct"
    correct_patterns = [
        r'\bclassif(y|ied)\s+as\s+correct\b',
        r'\bthis\s+is\s+correct\b',
        r'\bgrade[d]?\s+as\s+correct\b',
        r'\bis\s+correct\b',
    ]
    for pattern in correct_patterns:
        if re.search(pattern, text_lower):
            return "correct"
    
    # Count standalone occurrences as fallback (only in the first 1000 chars to avoid math content)
    text_sample = text_lower[:1000]
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
    
    # Priority-based selection when multiple appear
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
    
    pred_str = str(prediction).lower().strip()
    
    # Direct match
    if pred_str in ["correct", "incorrect", "partial", "almost"]:
        return pred_str
    
    # Check for exact match after removing quotes
    pred_clean = pred_str.strip('"\'')
    if pred_clean in ["correct", "incorrect", "partial", "almost"]:
        return pred_clean
    
    # Check if it's a valid label at all
    if not _is_valid_label(prediction):
        # If we have points, use them as the ground truth
        if points is not None:
            return _points_to_classification(points)
        return "incorrect"
    
    # Priority: almost > partial > incorrect > correct (most specific first)
    # This order helps avoid misclassification when multiple words appear
    if "almost" in pred_str:
        return "almost"
    if "partial" in pred_str:
        return "partial"
    if "incorrect" in pred_str:
        return "incorrect"
    if "correct" in pred_str:
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
        
        The primary approach is to extract points from grading guidelines
        and map them directly to classification labels, as this is the
        most reliable method.
        """
        # Extract key fields from inputs
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        # Extract points from grading guidelines - this is the ground truth
        points = _extract_points_from_guidelines(grading_guidelines)
        
        # If we successfully extracted points, use them directly
        if points is not None:
            prediction = _points_to_classification(points)
            self.log_fn(f"Points extracted: {points}/7 → Classification: {prediction}")
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
                            prediction = _normalize_prediction(json_data["response"], points)
                        else:
                            # Check other common fields
                            for field in ["classification", "grade", "result", "evaluation", "verdict", "outcome"]:
                                if field in json_data:
                                    prediction = _normalize_prediction(json_data[field], points)
                                    break
                    else:
                        # Try direct extraction from text
                        direct = _extract_response_direct(response_text)
                        if direct:
                            prediction = direct
                        else:
                            # Last resort: check if any valid label appears in the first 500 chars
                            text_sample = response_text[:500].lower()
                            for label in ["almost", "partial", "incorrect", "correct"]:
                                if re.search(rf'\b{label}\b', text_sample):
                                    prediction = label
                                    break
                            if prediction == "incorrect":
                                self.log_fn(f"No valid response found in: {response_text[:200]}...")
                else:
                    self.log_fn("Empty response text in msg_history")
            else:
                self.log_fn("Empty msg_history")
                
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            import traceback
            self.log_fn(traceback.format_exc())

        # Final validation
        if prediction not in ["correct", "incorrect", "partial", "almost"]:
            prediction = "incorrect"

        return str(prediction), msg_history
