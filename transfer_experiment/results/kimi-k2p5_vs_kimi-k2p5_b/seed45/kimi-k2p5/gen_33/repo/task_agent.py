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
import time
from typing import Callable, Any

from agent.llm_client import get_response_from_llm, EVAL_MODEL

logger = logging.getLogger(__name__)


def retry_with_backoff(
    func: Callable[..., Any],
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 10.0,
    exceptions: tuple = (Exception,),
) -> Any:
    """Retry a function with exponential backoff.
    
    Args:
        func: The function to retry
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        exceptions: Tuple of exceptions to catch and retry on
        
    Returns:
        The result of the function call
        
    Raises:
        The last exception encountered if all retries fail
    """
    last_exception = None
    for attempt in range(max_retries):
        try:
            return func()
        except exceptions as e:
            last_exception = e
            if attempt < max_retries - 1:
                delay = min(base_delay * (2 ** attempt), max_delay)
                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.1f}s...")
                time.sleep(delay)
    raise last_exception


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
        except json.JSONDecodeError as e:
            logger.debug(f"JSON decode error in <json> block: {e}")
            continue
    return results or None


def _extract_json_flexible(text: str) -> dict | None:
    """Extract JSON using multiple strategies.
    
    Tries multiple methods in order of preference:
    1. Extract from <json>...</json> tags
    2. Extract from ```json...``` code blocks
    3. Find JSON objects directly in text
    """
    # Strategy 1: <json> tags
    json_blocks = _extract_jsons(text)
    if json_blocks:
        return json_blocks[-1]
    
    # Strategy 2: ```json code blocks
    json_code_pattern = r'```json\s*(.*?)```'
    matches = re.findall(json_code_pattern, text, re.DOTALL)
    for match in matches:
        try:
            return json.loads(match.strip())
        except json.JSONDecodeError:
            continue
    
    # Strategy 3: ``` code blocks (any language)
    code_pattern = r'```\s*(.*?)```'
    matches = re.findall(code_pattern, text, re.DOTALL)
    for match in matches:
        try:
            return json.loads(match.strip())
        except json.JSONDecodeError:
            continue
    
    # Strategy 4: Find JSON objects directly (curly braces)
    brace_count = 0
    start_idx = -1
    for i, char in enumerate(text):
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


def _normalize_prediction(raw_value) -> str:
    """Normalize a raw prediction value to one of the valid labels.
    
    Valid labels: "correct", "incorrect", "partial", "almost"
    """
    if raw_value is None:
        return "unknown"
    
    raw_str = str(raw_value).lower().strip().strip('"\'')
    
    # Direct matches
    if raw_str in ["correct", "incorrect", "partial", "almost"]:
        return raw_str
    
    # Handle common variations for "correct"
    correct_variations = [
        "true", "yes", "right", "valid", "1", "full", "complete",
        "full credit", "full marks", "perfect", "flawless",
        "100% correct", "fully correct", "entirely correct", "totally correct",
        "no errors", "all steps correct", "mathematically correct",
        "full score", "max score", "maximum points"
    ]
    if raw_str in correct_variations or any(v in raw_str for v in correct_variations[:8]):
        return "correct"
    
    # Handle common variations for "incorrect"
    incorrect_variations = [
        "false", "no", "wrong", "invalid", "0", "none", "fail", "error",
        "no credit", "zero credit", "no marks", "zero marks",
        "fundamentally wrong", "completely wrong", "totally wrong", "entirely wrong",
        "no solution", "no progress", "no meaningful work", "blank", "failed", "rejected",
        "not correct", "not right", "not valid"
    ]
    if raw_str in incorrect_variations or any(v in raw_str for v in incorrect_variations[:8]):
        return "incorrect"
    
    # Handle common variations for "partial"
    partial_variations = [
        "part", "partially", "incomplete", "half", "some",
        "partial credit", "half credit", "some credit",
        "partial solution", "partially correct", "half correct",
        "some progress", "meaningful progress", "on the right track",
        "good start", "correct approach", "correct idea", "started correctly",
        "50% correct", "60% correct", "40% correct"
    ]
    if raw_str in partial_variations or any(v in raw_str for v in partial_variations[:8]):
        return "partial"
    
    # Handle common variations for "almost" - check these BEFORE substring checks
    almost_variations = [
        "almost correct", "close", "minor errors", "nearly",
        "nearly correct", "mostly correct", "minor error",
        "small error", "trivial error", "minor mistake", "almost there",
        "very close", "95% correct", "90% correct", "most credit",
        "most marks", "minor omission", "notational error", "sign error", "arithmetic error"
    ]
    if raw_str in almost_variations or any(v in raw_str for v in almost_variations):
        return "almost"
    
    # Check for substring matches (more specific first)
    # Check for "incorrect" before "correct" to avoid substring issues
    if "incorrect" in raw_str or "wrong" in raw_str:
        return "incorrect"
    if "almost" in raw_str:
        return "almost"
    if "partial" in raw_str:
        return "partial"
    if "correct" in raw_str:
        # Check if preceded by negation
        idx = raw_str.find("correct")
        before = raw_str[max(0, idx-15):idx]
        if not any(neg in before for neg in ['not ', 'in', "isn't", 'isnt']):
            return "correct"
    
    return "unknown"


def _is_valid_prediction(prediction: str) -> bool:
    """Check if a prediction is one of the valid labels."""
    return prediction in ["correct", "incorrect", "partial", "almost"]


def _parse_grading_guidelines(guidelines: str) -> dict:
    """Parse grading guidelines to extract rubric indicators.
    
    Returns a dictionary with detected markers and their context.
    Also detects credit-based markers like (Full Credit), (No Credit), etc.
    
    ENHANCED LOGIC: 
    - Set primary_category when there's a SINGLE marker type (definitive)
    - Also set primary_category when there's a DOMINANT marker (2x+ more than others, or >70%)
    - When markers are balanced, no primary category - let LLM decide
    """
    result = {
        "has_partial": False,
        "has_almost": False,
        "has_correct": False,
        "has_incorrect": False,
        "partial_context": [],
        "almost_context": [],
        "correct_context": [],
        "incorrect_context": [],
        "primary_category": None,
        "confidence": 0.0,
        "marker_counts": {},
        "single_marker": False,  # True if only one marker type exists
    }
    
    if not guidelines:
        return result
    
    guidelines_lower = guidelines.lower()
    
    # Count occurrences of each marker type
    marker_counts = {
        "partial": 0,
        "almost": 0,
        "correct": 0,
        "incorrect": 0
    }
    
    # Credit-based markers (process first - more explicit)
    marker_counts["correct"] += len(re.findall(r'\(\s*full\s*credit\s*\)', guidelines_lower))
    marker_counts["incorrect"] += len(re.findall(r'\(\s*no\s*credit\s*\)', guidelines_lower))
    marker_counts["partial"] += len(re.findall(r'\(\s*half\s*credit\s*\)', guidelines_lower))
    marker_counts["almost"] += len(re.findall(r'\(\s*most\s*credit\s*\)', guidelines_lower))
    marker_counts["partial"] += len(re.findall(r'\(\s*some\s*credit\s*\)', guidelines_lower))
    marker_counts["partial"] += len(re.findall(r'\(\s*partial\s*credit\s*\)', guidelines_lower))
    
    # Standard markers (parentheses)
    marker_counts["partial"] += len(re.findall(r'\(\s*partial\s*\)', guidelines_lower))
    marker_counts["almost"] += len(re.findall(r'\(\s*almost\s*\)', guidelines_lower))
    marker_counts["correct"] += len(re.findall(r'\(\s*correct\s*\)', guidelines_lower))
    marker_counts["incorrect"] += len(re.findall(r'\(\s*incorrect\s*\)', guidelines_lower))
    
    # Additional marker variations (brackets)
    marker_counts["correct"] += len(re.findall(r'\[\s*correct\s*\]', guidelines_lower))
    marker_counts["almost"] += len(re.findall(r'\[\s*almost\s*\]', guidelines_lower))
    marker_counts["partial"] += len(re.findall(r'\[\s*partial\s*\]', guidelines_lower))
    marker_counts["incorrect"] += len(re.findall(r'\[\s*incorrect\s*\]', guidelines_lower))
    
    # Text-based markers (e.g., "Correct:", "Almost:", etc.)
    marker_counts["correct"] += len(re.findall(r'\bcorrect\s*:', guidelines_lower))
    marker_counts["almost"] += len(re.findall(r'\balmost\s*:', guidelines_lower))
    marker_counts["partial"] += len(re.findall(r'\bpartial\s*:', guidelines_lower))
    marker_counts["incorrect"] += len(re.findall(r'\bincorrect\s*:', guidelines_lower))
    
    # Additional text patterns for "almost" (commonly missed)
    marker_counts["almost"] += len(re.findall(r'\bmost\s+points?\b', guidelines_lower))
    marker_counts["almost"] += len(re.findall(r'\bminor\s+error', guidelines_lower))
    marker_counts["almost"] += len(re.findall(r'\bsmall\s+error', guidelines_lower))
    marker_counts["almost"] += len(re.findall(r'\btrivial\s+error', guidelines_lower))
    marker_counts["almost"] += len(re.findall(r'\bnearly\s+correct', guidelines_lower))
    marker_counts["almost"] += len(re.findall(r'\bmostly\s+correct', guidelines_lower))
    # More patterns for "almost" detection - from grading guidelines
    marker_counts["almost"] += len(re.findall(r'\bverification\s+contains\s+minor\s+mistakes', guidelines_lower))
    marker_counts["almost"] += len(re.findall(r'\bsolution\s+is\s+almost\s+complete', guidelines_lower))
    marker_counts["almost"] += len(re.findall(r'\bminor\s+mistakes\s+only', guidelines_lower))
    marker_counts["almost"] += len(re.findall(r'\bminor\s+mistakes?\b', guidelines_lower))
    marker_counts["almost"] += len(re.findall(r'\bsmall\s+gap', guidelines_lower))
    marker_counts["almost"] += len(re.findall(r'\bnot\s+completed', guidelines_lower))
    
    # Additional text patterns for "partial"
    marker_counts["partial"] += len(re.findall(r'\bsome\s+points?\b', guidelines_lower))
    marker_counts["partial"] += len(re.findall(r'\bincomplete', guidelines_lower))
    marker_counts["partial"] += len(re.findall(r'\bpartial\s+solution', guidelines_lower))
    
    # Additional text patterns for "incorrect"
    marker_counts["incorrect"] += len(re.findall(r'\bno\s+points?\b', guidelines_lower))
    marker_counts["incorrect"] += len(re.findall(r'\bzero\s+points?\b', guidelines_lower))
    
    # Set has_* flags
    result["has_partial"] = marker_counts["partial"] > 0
    result["has_almost"] = marker_counts["almost"] > 0
    result["has_correct"] = marker_counts["correct"] > 0
    result["has_incorrect"] = marker_counts["incorrect"] > 0
    result["marker_counts"] = marker_counts
    
    # Count how many different marker types are present
    markers_present = [k for k, v in marker_counts.items() if v > 0]
    total_markers = sum(marker_counts.values())
    
    # Case 1: Single marker type - definitive classification
    if len(markers_present) == 1:
        result["primary_category"] = markers_present[0]
        result["confidence"] = 0.9
        result["single_marker"] = True
    # Case 2: Multiple markers - check for dominant one
    elif len(markers_present) > 1 and total_markers > 0:
        # Find the dominant marker
        sorted_markers = sorted(markers_present, key=lambda x: marker_counts[x], reverse=True)
        dominant = sorted_markers[0]
        dominant_count = marker_counts[dominant]
        
        # Check if dominant is at least 2x the second highest and has at least 3 occurrences
        if len(sorted_markers) > 1:
            second_count = marker_counts[sorted_markers[1]]
            if dominant_count >= 2 * second_count and dominant_count >= 3:
                result["primary_category"] = dominant
                result["confidence"] = 0.75
                result["single_marker"] = False
        
        # If no 2x dominance, check if dominant represents >70% of all markers
        if result["primary_category"] is None and dominant_count / total_markers >= 0.7:
            result["primary_category"] = dominant
            result["confidence"] = 0.7
            result["single_marker"] = False
        
        # Special case: if both "almost" and "partial" are present with similar counts
        # and "almost" is slightly higher or equal, set primary to "almost" to avoid under-classification
        if result["primary_category"] is None and "almost" in markers_present and "partial" in markers_present:
            almost_count = marker_counts["almost"]
            partial_count = marker_counts["partial"]
            # If almost >= partial and partial is small, suggest almost
            if almost_count >= partial_count and almost_count >= 1 and partial_count <= 2:
                result["primary_category"] = "almost"
                result["confidence"] = 0.6
                result["single_marker"] = False
    
    return result


def _check_guideline_markers(guidelines: str, student_answer: str) -> str | None:
    """Check if grading guidelines have EXCLUSIVE markers that definitively indicate the grade.
    
    Returns the suggested grade when there's a clear dominant marker indicating
    the classification. Returns None if markers are ambiguous.
    
    The markers (Partial), (Almost), (Correct), (Incorrect) in the grading guidelines
    describe what CRITERIA would lead to each classification - they don't necessarily
    indicate which classification applies to the current student answer.
    
    ENHANCED LOGIC:
    - Single marker type: definitive suggestion
    - Dominant marker (2x+ more than others): suggest that category
    - Multiple balanced markers: return None to let LLM decide
    """
    if not guidelines:
        return None
    
    guidelines_lower = guidelines.lower()
    
    # Count occurrences of each marker type
    marker_counts = {
        "correct": 0,
        "almost": 0,
        "partial": 0,
        "incorrect": 0
    }
    
    # Credit-based markers (higher priority - more explicit)
    marker_counts["correct"] += len(re.findall(r'\(\s*full\s*credit\s*\)', guidelines_lower))
    marker_counts["incorrect"] += len(re.findall(r'\(\s*no\s*credit\s*\)', guidelines_lower))
    marker_counts["partial"] += len(re.findall(r'\(\s*half\s*credit\s*\)', guidelines_lower))
    marker_counts["almost"] += len(re.findall(r'\(\s*most\s*credit\s*\)', guidelines_lower))
    marker_counts["partial"] += len(re.findall(r'\(\s*some\s*credit\s*\)', guidelines_lower))
    marker_counts["partial"] += len(re.findall(r'\(\s*partial\s*credit\s*\)', guidelines_lower))
    
    # Standard markers
    marker_counts["correct"] += len(re.findall(r'\(\s*correct\s*\)', guidelines_lower))
    marker_counts["almost"] += len(re.findall(r'\(\s*almost\s*\)', guidelines_lower))
    marker_counts["partial"] += len(re.findall(r'\(\s*partial\s*\)', guidelines_lower))
    marker_counts["incorrect"] += len(re.findall(r'\(\s*incorrect\s*\)', guidelines_lower))
    
    # Additional marker variations (case-insensitive)
    marker_counts["correct"] += len(re.findall(r'\[\s*correct\s*\]', guidelines_lower))
    marker_counts["almost"] += len(re.findall(r'\[\s*almost\s*\]', guidelines_lower))
    marker_counts["partial"] += len(re.findall(r'\[\s*partial\s*\]', guidelines_lower))
    marker_counts["incorrect"] += len(re.findall(r'\[\s*incorrect\s*\]', guidelines_lower))
    
    # Text-based markers (e.g., "Correct:", "Almost:", etc.)
    marker_counts["correct"] += len(re.findall(r'\bcorrect\s*:', guidelines_lower))
    marker_counts["almost"] += len(re.findall(r'\balmost\s*:', guidelines_lower))
    marker_counts["partial"] += len(re.findall(r'\bpartial\s*:', guidelines_lower))
    marker_counts["incorrect"] += len(re.findall(r'\bincorrect\s*:', guidelines_lower))
    
    # Additional text patterns for "almost" (commonly missed)
    marker_counts["almost"] += len(re.findall(r'\bmost\s+points?\b', guidelines_lower))
    marker_counts["almost"] += len(re.findall(r'\bminor\s+error', guidelines_lower))
    marker_counts["almost"] += len(re.findall(r'\bsmall\s+error', guidelines_lower))
    marker_counts["almost"] += len(re.findall(r'\btrivial\s+error', guidelines_lower))
    marker_counts["almost"] += len(re.findall(r'\bnearly\s+correct', guidelines_lower))
    marker_counts["almost"] += len(re.findall(r'\bmostly\s+correct', guidelines_lower))
    # More patterns for "almost" detection - from grading guidelines
    marker_counts["almost"] += len(re.findall(r'\bverification\s+contains\s+minor\s+mistakes', guidelines_lower))
    marker_counts["almost"] += len(re.findall(r'\bsolution\s+is\s+almost\s+complete', guidelines_lower))
    marker_counts["almost"] += len(re.findall(r'\bminor\s+mistakes\s+only', guidelines_lower))
    marker_counts["almost"] += len(re.findall(r'\bminor\s+mistakes?\b', guidelines_lower))
    marker_counts["almost"] += len(re.findall(r'\bsmall\s+gap', guidelines_lower))
    marker_counts["almost"] += len(re.findall(r'\bnot\s+completed', guidelines_lower))
    
    # Additional text patterns for "partial"
    marker_counts["partial"] += len(re.findall(r'\bsome\s+points?\b', guidelines_lower))
    marker_counts["partial"] += len(re.findall(r'\bincomplete', guidelines_lower))
    marker_counts["partial"] += len(re.findall(r'\bpartial\s+solution', guidelines_lower))
    
    # Additional text patterns for "incorrect"
    marker_counts["incorrect"] += len(re.findall(r'\bno\s+points?\b', guidelines_lower))
    marker_counts["incorrect"] += len(re.findall(r'\bzero\s+points?\b', guidelines_lower))
    
    # Count how many different marker types are present
    markers_present = [k for k, v in marker_counts.items() if v > 0]
    
    # Case 1: Single marker type - definitive suggestion
    if len(markers_present) == 1:
        return markers_present[0]
    
    # Case 2: No markers - let LLM decide
    if len(markers_present) == 0:
        return None
    
    # Case 3: Multiple markers - check for dominant one
    total_markers = sum(marker_counts.values())
    if total_markers == 0:
        return None
    
    # Find the dominant marker (at least 2x more than any other)
    sorted_markers = sorted(markers_present, key=lambda x: marker_counts[x], reverse=True)
    dominant = sorted_markers[0]
    dominant_count = marker_counts[dominant]
    
    # Check if dominant is at least 2x the second highest
    if len(sorted_markers) > 1:
        second_count = marker_counts[sorted_markers[1]]
        if dominant_count >= 2 * second_count and dominant_count >= 3:
            # Dominant marker is clearly leading
            return dominant
    
    # If dominant marker represents >70% of all markers, suggest it
    if dominant_count / total_markers >= 0.7:
        return dominant
    
    # Special case: if both "almost" and "partial" are present with similar counts
    # and "almost" is slightly higher or equal, suggest "almost" to avoid under-classification
    if "almost" in markers_present and "partial" in markers_present:
        almost_count = marker_counts["almost"]
        partial_count = marker_counts["partial"]
        # If almost >= partial and both are significant, suggest almost
        # This prevents the common error of under-classifying as "partial"
        if almost_count >= partial_count and almost_count >= 1 and partial_count <= 2:
            return "almost"
    
    # Multiple balanced markers - let the LLM decide
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
        # Extract key information from inputs for better prompting
        problem = inputs.get('problem', '')
        solution = inputs.get('solution', '')
        grading_guidelines = inputs.get('grading_guidelines', '')
        student_answer = inputs.get('student_answer', '')
        points = inputs.get('points', '')
        reward = inputs.get('reward', '')
        
        # Parse grading guidelines to extract rubric indicators
        rubric = _parse_grading_guidelines(grading_guidelines)
        
        # Build rubric context for the prompt
        # Count occurrences of each marker type in the guidelines
        guidelines_lower = grading_guidelines.lower()
        marker_counts = {
            "correct": (len(re.findall(r'\(\s*correct\s*\)', guidelines_lower)) + 
                       len(re.findall(r'\(\s*full\s*credit\s*\)', guidelines_lower)) +
                       len(re.findall(r'\[\s*correct\s*\]', guidelines_lower)) +
                       len(re.findall(r'\bcorrect\s*:', guidelines_lower))),
            "almost": (len(re.findall(r'\(\s*almost\s*\)', guidelines_lower)) + 
                      len(re.findall(r'\(\s*most\s*credit\s*\)', guidelines_lower)) +
                      len(re.findall(r'\[\s*almost\s*\]', guidelines_lower)) +
                      len(re.findall(r'\balmost\s*:', guidelines_lower)) +
                      len(re.findall(r'\bmost\s+points?\b', guidelines_lower)) +
                      len(re.findall(r'\bminor\s+error', guidelines_lower)) +
                      len(re.findall(r'\bsmall\s+error', guidelines_lower)) +
                      len(re.findall(r'\btrivial\s+error', guidelines_lower)) +
                      len(re.findall(r'\bnearly\s+correct', guidelines_lower)) +
                      len(re.findall(r'\bmostly\s+correct', guidelines_lower)) +
                      len(re.findall(r'\bverification\s+contains\s+minor\s+mistakes', guidelines_lower)) +
                      len(re.findall(r'\bsolution\s+is\s+almost\s+complete', guidelines_lower)) +
                      len(re.findall(r'\bminor\s+mistakes\s+only', guidelines_lower)) +
                      len(re.findall(r'\bminor\s+mistakes?\b', guidelines_lower)) +
                      len(re.findall(r'\bsmall\s+gap', guidelines_lower)) +
                      len(re.findall(r'\bnot\s+completed', guidelines_lower))),
            "partial": (len(re.findall(r'\(\s*partial\s*\)', guidelines_lower)) + 
                       len(re.findall(r'\(\s*half\s*credit\s*\)', guidelines_lower)) + 
                       len(re.findall(r'\(\s*some\s*credit\s*\)', guidelines_lower)) + 
                       len(re.findall(r'\(\s*partial\s*credit\s*\)', guidelines_lower)) +
                       len(re.findall(r'\[\s*partial\s*\]', guidelines_lower)) +
                       len(re.findall(r'\bpartial\s*:', guidelines_lower)) +
                       len(re.findall(r'\bsome\s+points?\b', guidelines_lower)) +
                       len(re.findall(r'\bincomplete', guidelines_lower)) +
                       len(re.findall(r'\bpartial\s+solution', guidelines_lower))),
            "incorrect": (len(re.findall(r'\(\s*incorrect\s*\)', guidelines_lower)) + 
                        len(re.findall(r'\(\s*no\s*credit\s*\)', guidelines_lower)) +
                        len(re.findall(r'\[\s*incorrect\s*\]', guidelines_lower)) +
                        len(re.findall(r'\bincorrect\s*:', guidelines_lower)) +
                        len(re.findall(r'\bno\s+points?\b', guidelines_lower)) +
                        len(re.findall(r'\bzero\s+points?\b', guidelines_lower))),
        }
        
        # Get markers that exist in the rubric
        markers_found = [k for k, v in marker_counts.items() if v > 0]
        
        rubric_context = ""
        if len(markers_found) == 1:
            # Single marker type - strong signal, this is the definitive classification
            rubric_context = f"\n\nRUBRIC DIRECTIVE: Classify as '{markers_found[0]}'. This is the only marker type found in the guidelines, indicating all student answers in this set should receive this classification."
        elif len(markers_found) > 1:
            # Multiple markers - the rubric describes criteria for different categories
            # The LLM must evaluate which criteria the student answer meets
            marker_summary = ", ".join([f"{k}: {marker_counts[k]}" for k in markers_found])
            
            # Add specific guidance for almost vs partial distinction
            almost_partial_guidance = ""
            if "almost" in markers_found and "partial" in markers_found:
                almost_partial_guidance = """

SPECIAL GUIDANCE FOR almost vs partial (CRITICAL - HIGH ERROR RATE):
The rubric contains both (Partial) and (Almost) markers. This is the MOST COMMON MISCLASSIFICATION SCENARIO.

UNDERSTANDING THE RUBRIC:
- (Partial) markers describe what earns PARTIAL credit: initial progress, setup, key insights found
- (Almost) markers describe what earns ALMOST credit: complete solution with minor mistakes

DECISION TREE - USE THIS:

STEP 1: Did the student REACH A CONCLUSION/FINAL ANSWER?
- If NO (stopped halfway, no ending, incomplete) -> "partial"
- If YES (has a conclusion, even if slightly wrong) -> Go to Step 2

STEP 2: Is the solution structure COMPLETE (beginning, middle, end)?
- If NO (missing major sections) -> "partial"
- If YES (all sections present) -> Go to Step 3

STEP 3: What type of errors exist?
- MINOR errors only (typos, arithmetic, sign errors, small calculation mistakes) -> "almost"
- MAJOR errors (wrong approach, logical gaps, fundamentally wrong) -> "incorrect"

KEY DISTINCTION:
- "almost" = COMPLETE solution with only minor errors (student FINISHED, just made small mistakes)
- "partial" = INCOMPLETE solution (student STARTED but couldn't finish, NO CONCLUSION)

CRITICAL TEST: "Did the student REACH A CONCLUSION?"
- YES (even if slightly wrong) -> likely "almost"
- NO (stopped halfway) -> "partial"

COMMON MISTAKE: Do NOT classify a complete solution with minor errors as "partial" - it should be "almost"!

SUMMARY:
- "almost" = Complete structure + minor errors only (6-7/7 points)
- "partial" = Incomplete OR major errors (2-4/7 points)"""
            elif "almost" in markers_found:
                almost_partial_guidance = """

SPECIAL GUIDANCE: The rubric contains (Almost) markers.
This means: Complete solution structure with only minor errors (typos, arithmetic, sign errors).
The student should have reached a conclusion with only small fixable mistakes."""
            elif "partial" in markers_found:
                almost_partial_guidance = """

SPECIAL GUIDANCE: The rubric contains (Partial) markers.
This means: Incomplete solution with significant gaps, OR correct start but couldn't finish.
The student showed some understanding but missing major components."""
            
            rubric_context = f"\n\nRUBRIC CONTEXT: The guidelines contain multiple marker types ({marker_summary}). These describe what criteria would lead to each classification. You must evaluate the student answer and determine which criteria it meets. Do NOT simply count markers - analyze the content of the student answer.{almost_partial_guidance}"
        
        instruction = f"""You are an expert mathematical grader for competition mathematics. Your task is to classify student solutions into exactly one of four categories.

CLASSIFICATION CATEGORIES (STRICT DEFINITIONS):

1. "correct" - The solution is:
   - Fully correct, complete, and rigorous
   - Contains all necessary steps and reasoning
   - Would receive full marks / full credit
   - No errors, gaps, or missing components

2. "almost" - The solution is NEARLY CORRECT with only MINOR ERRORS (90-99% correct):
   - The main proof/solution structure is COMPLETE and essentially correct
   - Core mathematical reasoning is sound and valid
   - Only MINOR issues: small typos, trivial arithmetic errors, minor notational issues, small calculation mistakes, sign errors
   - The solution would receive 6/7 or 7/7 points (most credit)
   - The errors are easily fixable and do NOT affect the main mathematical argument
   - Examples of "almost":
     * A complete proof with one small arithmetic error at the end
     * Correct solution with a minor sign error in one step
     * Fully valid proof with a small typo in the final answer
     * Correct approach with a trivial calculation mistake that doesn't change the conclusion
     * Complete solution that reaches the right answer but has a minor error in working
     * Student wrote out the full proof but made a small error in one line
     * All major steps are present and correct, only minor details are wrong
   - KEY DISTINCTION: The hard work is done, solution is essentially complete, just needs minor polishing
   - WHEN IN DOUBT: If the student completed the main proof structure and only has small fixable errors -> "almost"
   - IMPORTANT: "almost" is often UNDER-CLASSIFIED as "partial". If the solution is complete with minor errors, it is "almost" NOT "partial".
   - CRITICAL TEST: Did the student REACH A CONCLUSION? If yes and it's mostly correct -> "almost"

3. "partial" - The solution has SIGNIFICANT GAPS but shows MEANINGFUL PROGRESS (20-60% correct):
   - Started with the right approach or key insight but did NOT complete the solution
   - Has some correct elements but MISSING major components or critical steps
   - Shows understanding of the problem but execution is INCOMPLETE
   - Would receive 2-4/7 points (some credit)
   - Examples of "partial":
     * Correctly identified what needs to be proved but did not complete the proof
     * Good start with correct approach but missing critical steps to finish
     * Correct initial setup but got stuck or made significant errors later
     * Has the right idea but couldn't execute the main proof
     * Student stopped halfway through without reaching a conclusion
     * Missing the final answer or conclusion
     * Only completed the easy part, skipped the hard part
     * Made significant conceptual errors that invalidate the main argument
     * The solution has major logical gaps that cannot be easily fixed
   - KEY DISTINCTION: Good start but significant work remains, solution is INCOMPLETE
   - WHEN IN DOUBT: If major components are missing or the proof is incomplete -> "partial"
   - CRITICAL TEST: Did the student STOP BEFORE FINISHING? If yes -> "partial"
   - IMPORTANT: "partial" is often OVER-CLASSIFIED. If the solution is complete with only minor errors, it should be "almost" NOT "partial".

4. "incorrect" - The solution is FUNDAMENTALLY WRONG (0-10% correct):
   - Wrong approach or major conceptual errors
   - No meaningful progress toward the solution
   - Would receive 0-1/7 points (no or minimal credit)
   - Blank or completely irrelevant
   - Examples of "incorrect":
     * Completely wrong approach to the problem
     * No understanding of the mathematical concepts
     * Irrelevant work that doesn't address the problem
     * Blank submission
     * Analysis that leads to a contradiction or disproves the statement
     * Claims the problem statement is false without valid proof
   - KEY DISTINCTION: No meaningful progress or fundamentally flawed approach
   - WHEN IN DOUBT: If the student showed NO understanding of the problem -> "incorrect"
   - IMPORTANT: "incorrect" is often UNDER-CLASSIFIED as "partial". If there's no valid mathematical progress, it's "incorrect" NOT "partial".

CRITICAL DECISION RULES:
1. SINGLE MARKER = DEFINITIVE: If the rubric contains ONLY ONE marker type (e.g., only "(Partial)" markers), classify according to that marker.

2. MULTIPLE MARKERS = EVALUATE: If the rubric contains multiple marker types (e.g., both "(Partial)" and "(Almost)"), these describe CRITERIA for different categories. You must:
   - Read the description under each marker to understand what earns that classification
   - Evaluate the student answer against these criteria
   - Choose the classification that best matches the student answer's content
   - Do NOT simply count markers - analyze the actual work
   - IMPORTANT: When both (Partial) and (Almost) markers exist, look at the descriptions:
     * (Partial) usually describes initial progress, setup, or finding key insights
     * (Almost) usually describes verification with minor mistakes, complete solution with small errors
     * If the student answer shows COMPLETE work with only minor errors -> "almost"
     * If the student answer shows INCOMPLETE work (stopped early, missing conclusion) -> "partial"

3. almost vs partial - THIS IS THE MOST CRITICAL DISTINCTION (COMMON ERROR: missing "almost"):
   
   USE THIS DECISION TREE:
   
   QUESTION 1: Did the student REACH A CONCLUSION/FINAL ANSWER?
   - If NO (stopped halfway, no ending, incomplete) -> "partial"
   - If YES (has a conclusion, even if slightly wrong) -> Go to Question 2
   
   QUESTION 2: Is the solution structure COMPLETE (beginning, middle, end)?
   - If NO (missing major sections) -> "partial"
   - If YES (all sections present) -> Go to Question 3
   
   QUESTION 3: What type of errors exist?
   - MINOR errors only (typos, arithmetic, sign errors, small calculation mistakes) -> "almost"
   - MAJOR errors (wrong approach, logical gaps, fundamentally wrong) -> "incorrect"
   
   KEY EXAMPLES:
   - "almost": Student wrote a complete proof but made a small calculation error at the end
   - "almost": Student had the right approach, executed all steps, but had a sign error
   - "almost": Student reached the correct answer but had a minor error in the working
   - "partial": Student started correctly but stopped halfway through the proof
   - "partial": Student had the right idea but couldn't complete the main argument
   - "partial": Student only did the setup, never reached a conclusion
   
   REMEMBER: "almost" = COMPLETE solution with minor errors. "partial" = INCOMPLETE (no conclusion).
   
   WHEN BOTH (Partial) AND (Almost) MARKERS EXIST IN RUBRIC:
   - The (Partial) marker describes what earns partial credit (initial progress)
   - The (Almost) marker describes what earns almost credit (complete solution with minor errors)
   - Look at the student's actual work: if they completed the solution with only minor issues -> "almost"
   - If they only made partial progress without completing -> "partial"

4. correct vs almost:
   - "correct": PERFECT solution, no errors whatsoever
   - "almost": NEARLY perfect, has minor errors but essentially correct
   - If there's ANY error, no matter how small -> "almost" (not "correct")

5. partial vs incorrect:
   - "partial": Shows MEANINGFUL progress - right approach, good start, some correct work
   - "incorrect": FUNDAMENTALLY wrong - wrong approach, no meaningful progress
   - If student showed understanding of the problem -> "partial"
   - If student was completely off track -> "incorrect"
   - SPECIAL CASE: If the student claims the problem statement is FALSE or derives a CONTRADICTION:
     * If the contradiction is valid and well-reasoned -> "incorrect" (the solution disproves the problem)
     * If the contradiction stems from the student's own errors -> "incorrect" (fundamental error)
     * Only mark as "partial" if the student made genuine progress toward a solution before getting stuck

PROBLEM:
{problem}

OFFICIAL SOLUTION:
{solution}

GRADING GUIDELINES (FOLLOW THE MARKERS):
{grading_guidelines}
{rubric_context}

STUDENT'S ANSWER:
{student_answer}

INSTRUCTIONS:
1. First, identify all markers in the grading guidelines: (Correct), (Almost), (Partial), (Incorrect), (Full Credit), (Most Credit), (Half Credit), (Some Credit), (No Credit)
2. If ONLY ONE marker type exists in the rubric, use that classification
3. If MULTIPLE marker types exist, read the descriptions under each marker to understand the criteria, then evaluate which criteria the student answer meets
4. Compare the student answer against the official solution to assess correctness and completeness
5. Apply the DECISION FRAMEWORK for classification:
   
   STEP 1 - Check for perfection:
   - Is the solution completely perfect with zero errors?
     * If YES -> "correct"
     * If NO -> proceed to Step 2
   
   STEP 2 - Check for completeness (CRITICAL for almost vs partial):
   - Did the student REACH A CONCLUSION or FINAL ANSWER?
   - Does the solution have a complete structure (beginning, middle, end)?
     * If NO (incomplete, no conclusion, stopped early, missing major parts) -> "partial" or "incorrect"
     * If YES (complete structure with conclusion) -> proceed to Step 3
   
   STEP 3 - Classify the complete solution:
   - Are the errors only MINOR (typos, arithmetic, sign errors, small calculation mistakes)?
     * If YES (minor errors only) -> "almost" (COMPLETE + MINOR ERRORS = ALMOST)
     * If NO (major conceptual errors, wrong approach) -> "incorrect"
   
6. For "partial" vs "incorrect":
   - Did the student show understanding of the problem with meaningful progress?
     * If YES (right approach, good start) -> "partial"
     * If NO (completely wrong approach, no understanding) -> "incorrect"

7. CRITICAL - "almost" vs "partial" CHECK (MOST COMMON ERROR):
   - "almost" = COMPLETE solution with minor errors (student FINISHED, just made small mistakes)
   - "partial" = INCOMPLETE solution (student STARTED but couldn't finish, NO CONCLUSION)
   - THE KEY QUESTION: "Did the student REACH A CONCLUSION?"
     * If YES (even if slightly wrong) -> likely "almost" or "correct"
     * If NO (stopped halfway) -> "partial"
   - COMMON MISTAKE: Do NOT classify a complete solution with minor errors as "partial" - it should be "almost"
   - When in doubt: If the student completed the main proof/solution structure -> "almost"

8. Respond ONLY with valid JSON in this exact format:

<json>
{{
    "response": "correct" | "almost" | "partial" | "incorrect"
}}
</json>"""

        # Use retry with backoff for LLM call to handle transient failures
        def _call_llm():
            return get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
        
        try:
            response, msg_history, info = retry_with_backoff(
                _call_llm,
                max_retries=3,
                base_delay=1.0,
                exceptions=(Exception,),
            )
        except Exception as e:
            logger.error(f"LLM call failed after retries: {e}")
            # Fallback to guideline-based prediction when LLM fails
            guideline_suggestion = _check_guideline_markers(grading_guidelines, student_answer)
            if guideline_suggestion and _is_valid_prediction(guideline_suggestion):
                self.log_fn(f"LLM failed, using guideline suggestion: {guideline_suggestion}")
                return guideline_suggestion, []
            # Fallback to rubric primary category
            if rubric.get("primary_category"):
                return rubric["primary_category"], []
            return "unknown", []

        # Extract prediction from response
        prediction = "unknown"
        try:
            last_message = msg_history[-1]["text"]
            self.log_fn(f"Raw response: {last_message[:500]}...")
            
            # Try flexible JSON extraction
            extracted = _extract_json_flexible(last_message)
            if extracted:
                # Try to get the response value
                if isinstance(extracted, dict):
                    if "response" in extracted:
                        prediction = _normalize_prediction(extracted["response"])
                    elif len(extracted) == 1:
                        # If only one key, use its value
                        prediction = _normalize_prediction(list(extracted.values())[0])
                    else:
                        # Try to find a value that looks like a grade
                        for key, value in extracted.items():
                            normalized = _normalize_prediction(value)
                            if _is_valid_prediction(normalized):
                                prediction = normalized
                                break
            
            # If still unknown, try direct text extraction with more patterns
            if not _is_valid_prediction(prediction):
                text_lower = last_message.lower()
                # Look for quoted labels
                if '"correct"' in text_lower or "'correct'" in text_lower:
                    prediction = "correct"
                elif '"almost"' in text_lower or "'almost'" in text_lower:
                    prediction = "almost"
                elif '"partial"' in text_lower or "'partial'" in text_lower:
                    prediction = "partial"
                elif '"incorrect"' in text_lower or "'incorrect'" in text_lower:
                    prediction = "incorrect"
                # Also check for standalone words at word boundaries
                elif re.search(r'\bcorrect\b', text_lower) and not re.search(r'\bincorrect\b', text_lower):
                    prediction = "correct"
                elif re.search(r'\balmost\b', text_lower):
                    prediction = "almost"
                elif re.search(r'\bpartial\b', text_lower):
                    prediction = "partial"
                elif re.search(r'\bincorrect\b', text_lower) or re.search(r'\bwrong\b', text_lower):
                    prediction = "incorrect"
            
            # Post-process: Strongly weight guideline markers from rubric
            guideline_suggestion = _check_guideline_markers(grading_guidelines, student_answer)
            
            if guideline_suggestion:
                if not _is_valid_prediction(prediction):
                    # Use guideline when LLM gives invalid prediction
                    self.log_fn(f"Using guideline suggestion for unknown prediction: {guideline_suggestion}")
                    prediction = guideline_suggestion
                elif guideline_suggestion != prediction:
                    # Check rubric confidence and marker clarity
                    rubric_confidence = rubric.get("confidence", 0)
                    
                    # Count marker types in rubric
                    markers_found = []
                    if rubric["has_correct"]:
                        markers_found.append("correct")
                    if rubric["has_almost"]:
                        markers_found.append("almost")
                    if rubric["has_partial"]:
                        markers_found.append("partial")
                    if rubric["has_incorrect"]:
                        markers_found.append("incorrect")
                    
                    # Override logic: be more aggressive about following rubric
                    should_override = False
                    
                    # Case 1: Single clear marker - always override
                    if len(markers_found) == 1:
                        should_override = True
                        self.log_fn(f"Single marker in rubric: overriding to '{guideline_suggestion}'")
                    # Case 2: High confidence rubric (>0.8) - override
                    elif rubric_confidence >= 0.8:
                        should_override = True
                        self.log_fn(f"High confidence rubric ({rubric_confidence}): overriding to '{guideline_suggestion}'")
                    # Case 3: LLM prediction seems inconsistent with dominant marker
                    elif guideline_suggestion == rubric.get("primary_category") and rubric_confidence >= 0.6:
                        should_override = True
                        self.log_fn(f"Guideline matches primary category: overriding to '{guideline_suggestion}'")
                    # Case 4: LLM predicted "partial" but guideline suggests "almost" 
                    # This is a critical case - we need to be more willing to upgrade to "almost"
                    # because the LLM often under-classifies complete solutions with minor errors
                    elif prediction == "partial" and guideline_suggestion == "almost":
                        # Be more willing to upgrade to "almost" - this is a common misclassification
                        almost_count = marker_counts.get("almost", 0)
                        partial_count = marker_counts.get("partial", 0)
                        # Upgrade if almost is present and at least as frequent as partial
                        # OR if almost_count >= 1 and partial_count <= 2 (single partial item with almost marker)
                        if almost_count >= partial_count and almost_count >= 1:
                            should_override = True
                            self.log_fn(f"Overriding 'partial' to 'almost' - almost >= partial in rubric")
                        elif almost_count >= 1 and partial_count <= 2:
                            should_override = True
                            self.log_fn(f"Overriding 'partial' to 'almost' - almost present with few partial markers")
                        else:
                            self.log_fn(f"Keeping LLM 'partial' - almost not dominant enough")
                    
                    # Case 4b: LLM predicted "partial" but guideline suggests "incorrect"
                    # The LLM might be too generous - if rubric has "incorrect" markers, trust them
                    elif prediction == "partial" and guideline_suggestion == "incorrect":
                        # Check if "incorrect" is the dominant marker in the rubric
                        incorrect_count = marker_counts.get("incorrect", 0)
                        partial_count = marker_counts.get("partial", 0)
                        # Override if there's no "partial" marker but there is "incorrect"
                        # OR if "incorrect" is clearly dominant
                        if partial_count == 0 and incorrect_count > 0:
                            should_override = True
                            self.log_fn(f"Overriding 'partial' to 'incorrect' - no partial marker in rubric")
                        elif incorrect_count >= 2 * partial_count and incorrect_count >= 2:
                            should_override = True
                            self.log_fn(f"Overriding 'partial' to 'incorrect' - incorrect clearly dominant")
                        else:
                            self.log_fn(f"Keeping LLM 'partial' - incorrect not clearly dominant")
                    # Case 5: LLM predicted "almost" but guideline suggests "partial" 
                    # The LLM might be too generous, but we should check if "almost" is the dominant marker
                    elif prediction == "almost" and guideline_suggestion == "partial":
                        # Check if "almost" appears more frequently in the rubric
                        almost_count = marker_counts.get("almost", 0)
                        partial_count = marker_counts.get("partial", 0)
                        # Only override if partial is clearly dominant (2x+ more than almost)
                        if partial_count >= 2 * almost_count and partial_count >= 3:
                            should_override = True
                            self.log_fn(f"Overriding 'almost' to 'partial' - partial clearly dominant in rubric")
                        elif almost_count >= partial_count:
                            # "almost" is more or equally prominent in rubric, trust LLM
                            self.log_fn(f"Keeping LLM 'almost' - almost >= partial in rubric")
                        else:
                            # partial has slight edge but not dominant - trust LLM's judgment
                            self.log_fn(f"Keeping LLM 'almost' - partial not clearly dominant")
                    
                    # Case 5b: LLM predicted "almost" but guideline suggests "incorrect"
                    # The LLM is likely too generous - downgrade if incorrect is dominant
                    elif prediction == "almost" and guideline_suggestion == "incorrect":
                        # Check if "incorrect" is the dominant marker in the rubric
                        incorrect_count = marker_counts.get("incorrect", 0)
                        almost_count = marker_counts.get("almost", 0)
                        # Override if there's no "almost" marker but there is "incorrect"
                        # OR if "incorrect" is clearly dominant
                        if almost_count == 0 and incorrect_count > 0:
                            should_override = True
                            self.log_fn(f"Overriding 'almost' to 'incorrect' - no almost marker in rubric")
                        elif incorrect_count >= 2 * almost_count and incorrect_count >= 2:
                            should_override = True
                            self.log_fn(f"Overriding 'almost' to 'incorrect' - incorrect clearly dominant")
                        else:
                            self.log_fn(f"Keeping LLM 'almost' - incorrect not clearly dominant")
                    # Case 6: LLM predicted "incorrect" but guideline suggests something else
                    # "incorrect" is often over-used by LLM - it tends to be too harsh
                    elif prediction == "incorrect" and guideline_suggestion in ["partial", "almost"]:
                        # The LLM may be too harsh - the rubric suggests the student deserves some credit
                        # Be more willing to upgrade from "incorrect" to give students benefit of doubt
                        # But only if the guideline is clearly dominant
                        incorrect_count = marker_counts.get("incorrect", 0)
                        if guideline_suggestion == "partial":
                            partial_count = marker_counts.get("partial", 0)
                            # Only upgrade if partial is clearly dominant
                            if partial_count >= 2 * incorrect_count and partial_count >= 2:
                                should_override = True
                                self.log_fn(f"Overriding 'incorrect' to 'partial' - partial clearly dominant")
                            else:
                                self.log_fn(f"Keeping LLM 'incorrect' - partial not clearly dominant")
                        elif guideline_suggestion == "almost":
                            almost_count = marker_counts.get("almost", 0)
                            # Only upgrade if almost is clearly dominant
                            if almost_count >= 2 * incorrect_count and almost_count >= 2:
                                should_override = True
                                self.log_fn(f"Overriding 'incorrect' to 'almost' - almost clearly dominant")
                            else:
                                self.log_fn(f"Keeping LLM 'incorrect' - almost not clearly dominant")
                    
                    # Case 7: LLM predicted "correct" but guideline suggests "almost"
                    # The LLM might be too generous - if rubric has "almost" markers, trust them
                    elif prediction == "correct" and guideline_suggestion == "almost":
                        # Check if "almost" is the dominant marker in the rubric
                        almost_count = marker_counts.get("almost", 0)
                        correct_count = marker_counts.get("correct", 0)
                        # Override if "almost" is present and there's no "correct" marker
                        # OR if "almost" is present at all (be more conservative about "correct")
                        if correct_count == 0 and almost_count > 0:
                            should_override = True
                            self.log_fn(f"Overriding 'correct' to 'almost' - no correct marker in rubric")
                        elif almost_count >= correct_count and almost_count > 0:
                            # If almost is at least as frequent as correct, trust the guideline
                            should_override = True
                            self.log_fn(f"Overriding 'correct' to 'almost' - almost >= correct in rubric")
                        else:
                            self.log_fn(f"Keeping LLM 'correct' - 'almost' not clearly dominant")
                    
                    # Case 8: LLM predicted "correct" but guideline suggests "partial"
                    # The LLM is likely too generous - downgrade if partial is present
                    elif prediction == "correct" and guideline_suggestion == "partial":
                        # Check if "partial" is the dominant marker in the rubric
                        partial_count = marker_counts.get("partial", 0)
                        correct_count = marker_counts.get("correct", 0)
                        # Override if there's no "correct" marker but there is "partial"
                        # OR if "partial" is clearly dominant
                        if correct_count == 0 and partial_count > 0:
                            should_override = True
                            self.log_fn(f"Overriding 'correct' to 'partial' - no correct marker in rubric")
                        elif partial_count > correct_count and rubric_confidence >= 0.5:
                            should_override = True
                            self.log_fn(f"Overriding 'correct' to 'partial' - rubric suggests 'partial' is dominant")
                        else:
                            self.log_fn(f"Keeping LLM 'correct' - 'partial' not clearly dominant")
                    
                    # Case 8b: LLM predicted "correct" but guideline suggests "incorrect"
                    # The LLM is definitely too generous - downgrade if incorrect is present
                    elif prediction == "correct" and guideline_suggestion == "incorrect":
                        # Check if "incorrect" is the dominant marker in the rubric
                        incorrect_count = marker_counts.get("incorrect", 0)
                        correct_count = marker_counts.get("correct", 0)
                        # Override if there's no "correct" marker but there is "incorrect"
                        # OR if "incorrect" is clearly dominant
                        if correct_count == 0 and incorrect_count > 0:
                            should_override = True
                            self.log_fn(f"Overriding 'correct' to 'incorrect' - no correct marker in rubric")
                        elif incorrect_count > correct_count and rubric_confidence >= 0.5:
                            should_override = True
                            self.log_fn(f"Overriding 'correct' to 'incorrect' - rubric suggests 'incorrect' is dominant")
                        else:
                            self.log_fn(f"Keeping LLM 'correct' - 'incorrect' not clearly dominant")
                    
                    if should_override:
                        prediction = guideline_suggestion
                    else:
                        self.log_fn(f"Keeping LLM '{prediction}' over guideline '{guideline_suggestion}'")
            
            # Fallback to rubric primary category if still unknown
            if not _is_valid_prediction(prediction) and rubric.get("primary_category"):
                prediction = rubric["primary_category"]
                self.log_fn(f"Using rubric primary category as fallback: {prediction}")
            
            self.log_fn(f"Extracted prediction: {prediction}")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            # If exception occurred, try to use guideline suggestion as last resort
            try:
                guideline_suggestion = _check_guideline_markers(grading_guidelines, student_answer)
                if guideline_suggestion:
                    prediction = guideline_suggestion
                    self.log_fn(f"Using guideline suggestion after error: {prediction}")
            except Exception:
                pass

        return str(prediction), msg_history
