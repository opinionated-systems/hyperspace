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
    
    IMPROVED LOGIC: Count occurrences of each marker type to determine primary category.
    The most frequent marker indicates the intended classification.
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
    
    # Standard markers - count at start of lines or standalone
    marker_counts["partial"] += len(re.findall(r'\(\s*partial\s*\)', guidelines_lower))
    marker_counts["almost"] += len(re.findall(r'\(\s*almost\s*\)', guidelines_lower))
    marker_counts["correct"] += len(re.findall(r'\(\s*correct\s*\)', guidelines_lower))
    marker_counts["incorrect"] += len(re.findall(r'\(\s*incorrect\s*\)', guidelines_lower))
    
    # Credit-based markers
    marker_counts["correct"] += len(re.findall(r'\(\s*full\s*credit\s*\)', guidelines_lower))
    marker_counts["incorrect"] += len(re.findall(r'\(\s*no\s*credit\s*\)', guidelines_lower))
    marker_counts["partial"] += len(re.findall(r'\(\s*half\s*credit\s*\)', guidelines_lower))
    marker_counts["almost"] += len(re.findall(r'\(\s*most\s*credit\s*\)', guidelines_lower))
    marker_counts["partial"] += len(re.findall(r'\(\s*some\s*credit\s*\)', guidelines_lower))
    marker_counts["partial"] += len(re.findall(r'\(\s*partial\s*credit\s*\)', guidelines_lower))
    
    # Also check for markers at line starts (common in rubrics)
    lines = guidelines_lower.split('\n')
    for line in lines:
        line = line.strip()
        if line.startswith('(partial)') or line.startswith('(half credit)') or line.startswith('(some credit)'):
            marker_counts["partial"] += 1
        elif line.startswith('(almost)') or line.startswith('(most credit)'):
            marker_counts["almost"] += 1
        elif line.startswith('(correct)') or line.startswith('(full credit)'):
            marker_counts["correct"] += 1
        elif line.startswith('(incorrect)') or line.startswith('(no credit)'):
            marker_counts["incorrect"] += 1
    
    # Set has_* flags
    result["has_partial"] = marker_counts["partial"] > 0
    result["has_almost"] = marker_counts["almost"] > 0
    result["has_correct"] = marker_counts["correct"] > 0
    result["has_incorrect"] = marker_counts["incorrect"] > 0
    result["marker_counts"] = marker_counts
    
    # Determine primary category based on most frequent marker
    max_count = max(marker_counts.values())
    total_markers = sum(marker_counts.values())
    
    if max_count > 0:
        # Get all markers with max count
        top_markers = [k for k, v in marker_counts.items() if v == max_count]
        
        if len(top_markers) == 1:
            # Single dominant marker - high confidence
            result["primary_category"] = top_markers[0]
            # Higher confidence if this marker is a large fraction of all markers
            if total_markers > 0:
                dominance = max_count / total_markers
                result["confidence"] = 0.7 + 0.25 * dominance  # 0.7 to 0.95
            else:
                result["confidence"] = 0.9
        else:
            # Multiple markers with same count - use priority order with lower confidence
            priority = ["correct", "almost", "partial", "incorrect"]
            for p in priority:
                if p in top_markers:
                    result["primary_category"] = p
                    result["confidence"] = 0.6
                    break
    
    return result


def _check_guideline_markers(guidelines: str, student_answer: str) -> str | None:
    """Check if grading guidelines have explicit markers that indicate the grade.
    
    Returns the suggested grade based on guideline markers, or None if unclear.
    The markers (Partial), (Almost), (Correct), (Incorrect) in the grading guidelines
    indicate the INTENDED classification for this solution.
    
    IMPROVED LOGIC:
    - Count occurrences of each marker type
    - The MOST FREQUENT marker indicates the primary classification
    - For ties, use priority order (correct > almost > partial > incorrect)
    - This handles cases where rubrics describe multiple aspects of a solution
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
    
    # Credit-based markers (higher weight as they're more explicit)
    marker_counts["correct"] += 2 * len(re.findall(r'\(\s*full\s*credit\s*\)', guidelines_lower))
    marker_counts["incorrect"] += 2 * len(re.findall(r'\(\s*no\s*credit\s*\)', guidelines_lower))
    marker_counts["partial"] += 2 * len(re.findall(r'\(\s*half\s*credit\s*\)', guidelines_lower))
    marker_counts["almost"] += 2 * len(re.findall(r'\(\s*most\s*credit\s*\)', guidelines_lower))
    marker_counts["partial"] += 2 * len(re.findall(r'\(\s*some\s*credit\s*\)', guidelines_lower))
    marker_counts["partial"] += 2 * len(re.findall(r'\(\s*partial\s*credit\s*\)', guidelines_lower))
    
    # Standard markers
    marker_counts["correct"] += len(re.findall(r'\(\s*correct\s*\)', guidelines_lower))
    marker_counts["almost"] += len(re.findall(r'\(\s*almost\s*\)', guidelines_lower))
    marker_counts["partial"] += len(re.findall(r'\(\s*partial\s*\)', guidelines_lower))
    marker_counts["incorrect"] += len(re.findall(r'\(\s*incorrect\s*\)', guidelines_lower))
    
    # Check for markers at line starts (common in rubrics) - give extra weight
    lines = guidelines_lower.split('\n')
    for line in lines:
        line = line.strip()
        if line.startswith('(partial)') or line.startswith('(half credit)') or line.startswith('(some credit)'):
            marker_counts["partial"] += 2
        elif line.startswith('(almost)') or line.startswith('(most credit)'):
            marker_counts["almost"] += 2
        elif line.startswith('(correct)') or line.startswith('(full credit)'):
            marker_counts["correct"] += 2
        elif line.startswith('(incorrect)') or line.startswith('(no credit)'):
            marker_counts["incorrect"] += 2
    
    # Find the marker with highest count
    max_count = max(marker_counts.values())
    if max_count == 0:
        return None
    
    # Get all markers with max count
    top_markers = [k for k, v in marker_counts.items() if v == max_count]
    
    # If only one top marker, return it
    if len(top_markers) == 1:
        return top_markers[0]
    
    # For ties, use priority order
    priority_order = ["correct", "almost", "partial", "incorrect"]
    for grade in priority_order:
        if grade in top_markers:
            return grade
    
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
        
        # Build rubric context for the prompt using counting-based approach
        # Count occurrences of each marker type in the guidelines
        guidelines_lower = grading_guidelines.lower()
        marker_counts = {
            "correct": len(re.findall(r'\(\s*correct\s*\)', guidelines_lower)) + len(re.findall(r'\(\s*full\s*credit\s*\)', guidelines_lower)),
            "almost": len(re.findall(r'\(\s*almost\s*\)', guidelines_lower)) + len(re.findall(r'\(\s*most\s*credit\s*\)', guidelines_lower)),
            "partial": len(re.findall(r'\(\s*partial\s*\)', guidelines_lower)) + len(re.findall(r'\(\s*half\s*credit\s*\)', guidelines_lower)) + len(re.findall(r'\(\s*some\s*credit\s*\)', guidelines_lower)) + len(re.findall(r'\(\s*partial\s*credit\s*\)', guidelines_lower)),
            "incorrect": len(re.findall(r'\(\s*incorrect\s*\)', guidelines_lower)) + len(re.findall(r'\(\s*no\s*credit\s*\)', guidelines_lower)),
        }
        
        # Get markers that exist in the rubric
        markers_found = [k for k, v in marker_counts.items() if v > 0]
        
        rubric_context = ""
        if len(markers_found) == 1:
            # Single marker type - strong signal
            rubric_context = f"\n\nRUBRIC DIRECTIVE: Classify as '{markers_found[0]}'. This is the only marker type found in the guidelines."
        elif len(markers_found) > 1:
            # Multiple markers - find the most frequent
            max_count = max(marker_counts.values())
            top_markers = [k for k, v in marker_counts.items() if v == max_count]
            
            if len(top_markers) == 1:
                # One marker is clearly dominant
                rubric_context = f"\n\nRUBRIC DIRECTIVE: Classify as '{top_markers[0]}'. This marker appears most frequently ({max_count} times) in the guidelines."
            else:
                # Tie - use priority order
                priority = ["correct", "almost", "partial", "incorrect"]
                for p in priority:
                    if p in top_markers:
                        rubric_context = f"\n\nRUBRIC DIRECTIVE: Consider '{p}' as primary. Multiple markers found with equal frequency ({max_count} each)."
                        break
        
        # Add explicit marker summary
        if markers_found:
            marker_summary = ", ".join([f"{k}: {marker_counts[k]}" for k in markers_found])
            rubric_context += f"\nMarker counts in rubric: {marker_summary}."
        
        instruction = f"""You are an expert mathematical grader for competition mathematics. Your task is to classify student solutions into exactly one of four categories.

CLASSIFICATION CATEGORIES (STRICT DEFINITIONS):

1. "correct" - The solution is:
   - Fully correct, complete, and rigorous
   - Contains all necessary steps and reasoning
   - Would receive full marks / full credit
   - No errors, gaps, or missing components

2. "almost" - The solution is NEARLY CORRECT (90-99% complete):
   - The main proof structure is complete and sound
   - Core mathematical reasoning is correct
   - Only MINOR issues: small typos, trivial arithmetic errors, minor notational issues
   - The solution needs only polishing, not fundamental changes
   - Example: A complete proof with one small calculation error

3. "partial" - The solution has SIGNIFICANT GAPS (30-70% complete):
   - Shows meaningful progress with some correct elements
   - Started with the right approach but incomplete
   - Missing KEY steps, critical insights, or major proof components
   - Has the right idea but far from a complete solution
   - Example: Correctly identified the invariant but didn't complete the proof

4. "incorrect" - The solution is FUNDAMENTALLY WRONG:
   - Wrong approach or major conceptual errors
   - No meaningful progress toward the solution
   - Blank or completely irrelevant

CRITICAL DECISION RULES:
1. RUBRIC MARKERS ARE DEFINITIVE: The grading guidelines contain explicit markers:
   - (Correct) or (Full Credit) → Use "correct"
   - (Almost) or (Most Credit) → Use "almost"  
   - (Partial), (Half Credit), or (Some Credit) → Use "partial"
   - (Incorrect) or (No Credit) → Use "incorrect"
   
2. When rubric markers are present, FOLLOW THEM EXACTLY. Do not override the rubric with your own judgment.

3. "almost" vs "partial" distinction:
   - "almost": Solution is essentially complete, minor fixes needed
   - "partial": Solution is incomplete, significant work remains

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
1. First, identify any markers in the grading guidelines: (Correct), (Almost), (Partial), (Incorrect), (Full Credit), (Most Credit), (Half Credit), (Some Credit), (No Credit)
2. If markers are present, use them to determine the classification
3. Evaluate the student answer against the official solution
4. Respond ONLY with valid JSON in this exact format:

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
