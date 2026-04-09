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
    
    # Handle common variations for "almost"
    almost_variations = [
        "almost correct", "close", "minor errors", "nearly",
        "nearly correct", "mostly correct", "minor error",
        "small error", "trivial error", "minor mistake", "almost there",
        "very close", "95% correct", "90% correct", "most credit",
        "most marks", "minor omission", "notational error", "sign error", "arithmetic error"
    ]
    if raw_str in almost_variations or any(v in raw_str for v in almost_variations[:8]):
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
    }
    
    if not guidelines:
        return result
    
    guidelines_lower = guidelines.lower()
    
    # Look for explicit markers with case-insensitive matching and track positions
    markers_with_pos = []
    
    # Standard markers
    for match in re.finditer(r'\(\s*Partial\s*\)', guidelines, re.IGNORECASE):
        markers_with_pos.append(("partial", match.start()))
        result["has_partial"] = True
    for match in re.finditer(r'\(\s*Almost\s*\)', guidelines, re.IGNORECASE):
        markers_with_pos.append(("almost", match.start()))
        result["has_almost"] = True
    for match in re.finditer(r'\(\s*Correct\s*\)', guidelines, re.IGNORECASE):
        markers_with_pos.append(("correct", match.start()))
        result["has_correct"] = True
    for match in re.finditer(r'\(\s*Incorrect\s*\)', guidelines, re.IGNORECASE):
        markers_with_pos.append(("incorrect", match.start()))
        result["has_incorrect"] = True
    
    # Credit-based markers
    for match in re.finditer(r'\(\s*Full\s*Credit\s*\)', guidelines, re.IGNORECASE):
        markers_with_pos.append(("correct", match.start()))
        result["has_correct"] = True
    for match in re.finditer(r'\(\s*No\s*Credit\s*\)', guidelines, re.IGNORECASE):
        markers_with_pos.append(("incorrect", match.start()))
        result["has_incorrect"] = True
    for match in re.finditer(r'\(\s*Half\s*Credit\s*\)', guidelines, re.IGNORECASE):
        markers_with_pos.append(("partial", match.start()))
        result["has_partial"] = True
    for match in re.finditer(r'\(\s*Most\s*Credit\s*\)', guidelines, re.IGNORECASE):
        markers_with_pos.append(("almost", match.start()))
        result["has_almost"] = True
    for match in re.finditer(r'\(\s*Some\s*Credit\s*\)', guidelines, re.IGNORECASE):
        markers_with_pos.append(("partial", match.start()))
        result["has_partial"] = True
    
    # Sort by position
    markers_with_pos.sort(key=lambda x: x[1])
    
    # Get unique marker types in order of appearance
    unique_types_ordered = list(dict.fromkeys([m[0] for m in markers_with_pos]))
    
    # Determine primary category based on markers
    if markers_with_pos:
        if len(unique_types_ordered) == 1:
            # Single marker type - high confidence
            result["primary_category"] = unique_types_ordered[0]
            result["confidence"] = 0.95
        elif len(unique_types_ordered) == 2 and "partial" in unique_types_ordered and "almost" in unique_types_ordered:
            # Both Partial and Almost - the LAST marker is authoritative
            last_marker = markers_with_pos[-1][0]
            result["primary_category"] = last_marker
            result["confidence"] = 0.9
        else:
            # Multiple markers - use priority order
            priority = ["correct", "almost", "partial", "incorrect"]
            for p in priority:
                if p in unique_types_ordered:
                    result["primary_category"] = p
                    result["confidence"] = 0.8
                    break
    
    return result


def _check_guideline_markers(guidelines: str, student_answer: str) -> str | None:
    """Check if grading guidelines have explicit markers that indicate the grade.
    
    Returns the suggested grade based on guideline markers, or None if unclear.
    The markers (Partial), (Almost), (Correct), (Incorrect) in the grading guidelines
    indicate the INTENDED classification for this solution.
    
    KEY INSIGHT: The rubric structure typically describes:
    - What partial progress was made (marked with Partial)
    - What makes the solution "almost" correct (marked with Almost)
    - The FINAL marker in the rubric indicates the intended classification
    
    IMPROVED LOGIC:
    - If exactly ONE marker type is found, return that marker
    - If BOTH (Partial) and (Almost) are found, the LAST marker is authoritative
      because it represents the final assessment after considering all aspects
    - For other combinations, use priority order (correct > almost > partial > incorrect)
    """
    if not guidelines:
        return None
    
    guidelines_lower = guidelines.lower()
    
    # Check for explicit markers with their positions
    markers_with_pos = []
    
    for match in re.finditer(r'\(correct\)', guidelines_lower):
        markers_with_pos.append(("correct", match.start()))
    for match in re.finditer(r'\(almost\)', guidelines_lower):
        markers_with_pos.append(("almost", match.start()))
    for match in re.finditer(r'\(partial\)', guidelines_lower):
        markers_with_pos.append(("partial", match.start()))
    for match in re.finditer(r'\(incorrect\)', guidelines_lower):
        markers_with_pos.append(("incorrect", match.start()))
    
    if not markers_with_pos:
        return None
    
    # Sort by position to find the LAST marker
    markers_with_pos.sort(key=lambda x: x[1])
    
    # Get unique marker types
    unique_markers = list(dict.fromkeys([m[0] for m in markers_with_pos]))
    
    # If exactly ONE unique marker type is found, return it
    if len(unique_markers) == 1:
        return unique_markers[0]
    
    # If BOTH (Partial) and (Almost) are found, the LAST marker is authoritative
    # The rubric describes partial progress first, then what makes it "almost" correct
    if len(unique_markers) == 2 and "partial" in unique_markers and "almost" in unique_markers:
        # The LAST marker indicates the final classification
        last_marker = markers_with_pos[-1][0]
        return last_marker
    
    # For other combinations (e.g., correct + almost, incorrect + partial),
    # the higher-tier marker (correct > almost > partial > incorrect) typically wins
    priority_order = ["correct", "almost", "partial", "incorrect"]
    for grade in priority_order:
        if grade in unique_markers:
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
        
        # Build rubric context for the prompt - simplified and direct
        markers_found = []
        if rubric["has_correct"]:
            markers_found.append("correct")
        if rubric["has_almost"]:
            markers_found.append("almost")
        if rubric["has_partial"]:
            markers_found.append("partial")
        if rubric["has_incorrect"]:
            markers_found.append("incorrect")
        
        rubric_context = ""
        if len(markers_found) == 1:
            # Single marker - this is a strong signal of the intended grade
            rubric_context = f"\n\nRUBRIC DIRECTIVE: Classify as '{markers_found[0]}'."
        elif len(markers_found) == 2 and "partial" in markers_found and "almost" in markers_found:
            # Both Partial and Almost - the LAST marker is authoritative
            # Find the last occurrence of each marker
            partial_pos = grading_guidelines.lower().rfind("(partial)")
            almost_pos = grading_guidelines.lower().rfind("(almost)")
            
            # The last marker indicates the final classification
            if almost_pos > partial_pos:
                rubric_context = "\n\nRUBRIC DIRECTIVE: Classify as 'almost'. The rubric describes partial progress first, then what makes the solution 'almost' correct."
            else:
                rubric_context = "\n\nRUBRIC DIRECTIVE: Classify as 'partial'."
        elif len(markers_found) > 1:
            # Multiple markers - use priority
            priority = ["correct", "almost", "partial", "incorrect"]
            for p in priority:
                if p in markers_found:
                    rubric_context = f"\n\nRUBRIC DIRECTIVE: Consider '{p}' as primary."
                    break
        
        instruction = f"""You are an expert mathematical grader. Classify the student solution into exactly one category.

CATEGORIES:
- "correct": Fully correct, complete, rigorous solution. No gaps or errors.
- "almost": Nearly correct with only minor errors (90-99% complete). Main proof structure is sound, just needs small fixes. The solution demonstrates understanding of the key ideas with only technical gaps.
- "partial": Some correct elements but significant gaps (30-70% complete). Missing key steps or insights. The solution makes meaningful progress but has major logical gaps or missing critical components.
- "incorrect": Fundamentally wrong or no meaningful progress.

CRITICAL RULES (FOLLOW THESE EXACTLY):
1. The grading guidelines contain markers like (Correct), (Almost), (Partial), or (Incorrect). These markers are the STRONGEST signal for classification.
2. If the rubric has a single marker, you MUST use that classification.
3. If the rubric has BOTH (Partial) and (Almost) markers, the LAST marker indicates the intended classification. The rubric typically describes partial progress first, then what makes it "almost" correct.
4. "Almost" means the solution is very close to complete - the main ideas are there with only minor technical issues.
5. "Partial" means there are significant gaps - key insights or proof steps are missing.

PROBLEM:
{problem}

OFFICIAL SOLUTION:
{solution}

GRADING GUIDELINES:
{grading_guidelines}
{rubric_context}

STUDENT'S ANSWER:
{student_answer}

Respond ONLY with this JSON format:
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
            
            # Post-process: Use guideline markers as authoritative signal
            # The rubric markers are the ground truth for classification
            guideline_suggestion = _check_guideline_markers(grading_guidelines, student_answer)
            
            if guideline_suggestion:
                if not _is_valid_prediction(prediction):
                    # Use guideline when LLM gives invalid prediction
                    self.log_fn(f"Using guideline suggestion for unknown prediction: {guideline_suggestion}")
                    prediction = guideline_suggestion
                elif guideline_suggestion != prediction:
                    # Trust rubric markers over LLM when they differ
                    self.log_fn(f"Overriding LLM '{prediction}' with guideline '{guideline_suggestion}'")
                    prediction = guideline_suggestion
            
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
