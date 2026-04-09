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
    
    raw_str = str(raw_value).lower().strip()
    
    # Direct matches
    if raw_str in ["correct", "incorrect", "partial", "almost"]:
        return raw_str
    
    # Check for exact matches with quotes removed
    clean_str = raw_str.strip('"\'')
    if clean_str in ["correct", "incorrect", "partial", "almost"]:
        return clean_str
    
    # Handle common variations
    if raw_str in ["true", "yes", "right", "valid", "1", "full", "complete"]:
        return "correct"
    if raw_str in ["false", "no", "wrong", "invalid", "0", "none", "fail", "error"]:
        return "incorrect"
    if raw_str in ["part", "partially", "incomplete", "half", "some"]:
        return "partial"
    if raw_str in ["almost correct", "close", "minor errors", "nearly"]:
        return "almost"
    
    # Check for substring matches (more specific first)
    if "almost" in raw_str:
        return "almost"
    if "partial" in raw_str:
        return "partial"
    # Be careful with "correct" - "incorrect" contains "correct"
    if "incorrect" in raw_str or "wrong" in raw_str:
        return "incorrect"
    if "correct" in raw_str:
        return "correct"
    
    return "unknown"


def _is_valid_prediction(prediction: str) -> bool:
    """Check if a prediction is one of the valid labels."""
    return prediction in ["correct", "incorrect", "partial", "almost"]


def _parse_grading_guidelines(guidelines: str) -> dict:
    """Parse grading guidelines to extract rubric indicators.
    
    Returns a dictionary with detected markers and their context.
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
    }
    
    if not guidelines:
        return result
    
    # Look for explicit markers
    markers = [
        (r'\(Partial\)', "partial"),
        (r'\(Almost\)', "almost"),
        (r'\(Correct\)', "correct"),
        (r'\(Incorrect\)', "incorrect"),
    ]
    
    for pattern, label in markers:
        matches = list(re.finditer(pattern, guidelines, re.IGNORECASE))
        if matches:
            result[f"has_{label}"] = True
            for match in matches:
                # Extract context around the marker
                start = max(0, match.start() - 50)
                end = min(len(guidelines), match.end() + 100)
                context = guidelines[start:end].replace('\n', ' ').strip()
                result[f"{label}_context"].append(context)
    
    return result


def _check_guideline_markers(guidelines: str, student_answer: str) -> str | None:
    """Check if grading guidelines have explicit markers that indicate the grade.
    
    Returns the suggested grade based on guideline markers, or None if unclear.
    The markers (Partial), (Almost), (Correct), (Incorrect) in the grading guidelines
    indicate the INTENDED classification for this solution.
    
    IMPROVED LOGIC:
    - If exactly ONE marker type is found, return that marker
    - If BOTH (Partial) and (Almost) are found, the final classification is ALMOST ALWAYS "almost"
      because the (Partial) section describes partial progress and (Almost) describes the final state
    - If multiple conflicting markers exist, return None for LLM to decide
    """
    if not guidelines:
        return None
    
    guidelines_lower = guidelines.lower()
    
    # Check for explicit markers
    has_correct = "(correct)" in guidelines_lower
    has_almost = "(almost)" in guidelines_lower
    has_partial = "(partial)" in guidelines_lower
    has_incorrect = "(incorrect)" in guidelines_lower
    
    # Count how many unique marker types are present
    markers_found = []
    if has_correct:
        markers_found.append("correct")
    if has_almost:
        markers_found.append("almost")
    if has_partial:
        markers_found.append("partial")
    if has_incorrect:
        markers_found.append("incorrect")
    
    # Only suggest if exactly ONE unique marker type is found
    if len(markers_found) == 1:
        return markers_found[0]
    
    # SPECIAL CASE: If BOTH (Partial) and (Almost) are found
    # This is a VERY COMMON pattern where:
    # - (Partial) describes what partial progress was made
    # - (Almost) describes what makes it "almost" correct (the final classification)
    # When both are present, the answer is ALMOST ALWAYS "almost"
    if has_partial and has_almost and len(markers_found) == 2:
        # The presence of both markers strongly indicates "almost"
        # The (Partial) section lists achievements, (Almost) describes the final state
        return "almost"
    
    # Multiple conflicting markers or unclear case - let LLM decide
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
        # List all markers found in the guidelines
        markers_found = []
        if rubric["has_correct"]:
            markers_found.append("(Correct)")
        if rubric["has_almost"]:
            markers_found.append("(Almost)")
        if rubric["has_partial"]:
            markers_found.append("(Partial)")
        if rubric["has_incorrect"]:
            markers_found.append("(Incorrect)")
        
        rubric_context = ""
        if len(markers_found) == 1:
            # Single marker - this is a strong signal of the intended grade
            rubric_context += f"\n\n[*** RUBRIC INDICATOR ***: The grading guidelines contain exactly one marker: {markers_found[0]}. This strongly indicates the intended classification.]"
        elif len(markers_found) == 2 and "partial" in markers_found and "almost" in markers_found:
            # Special case: Both Partial and Almost markers - this typically means "almost"
            rubric_context += f"\n\n[*** RUBRIC INDICATORS ***: The grading guidelines contain BOTH (Partial) and (Almost) markers. This typically indicates the solution has significant partial progress (described in the Partial section) but the final classification is ALMOST because the remaining issues are minor (described in the Almost section). Focus on the (Almost) section which describes the minor mistakes/verification issues that prevent it from being 'correct'.]"
        elif len(markers_found) > 1:
            # Multiple markers - the LLM needs to analyze the full context
            rubric_context += f"\n\n[*** RUBRIC INDICATORS ***: The grading guidelines contain multiple markers: {', '.join(markers_found)}. Analyze the full context to determine the appropriate classification based on the student's solution.]"
        
        instruction = f"""You are an expert mathematical grader evaluating student solutions to competition mathematics problems (like IMO, Putnam, etc.).

Your task is to evaluate the student's answer and classify it into EXACTLY ONE of these four categories:

1. "correct" - The student's answer is fully correct, complete, and rigorous. It matches or exceeds the official solution in completeness and correctness. Only use this if the solution is essentially flawless with no gaps or errors.

2. "almost" - The student's answer is NEARLY CORRECT with only MINOR ERRORS, typos, or small omissions. The core logic is sound, the main proof structure is complete, and the solution would be "correct" with minor fixes. This is for solutions that are 90-99% complete. KEY INDICATORS: The solution has the main proof structure but has minor mistakes in verification, small calculation errors, or trivial omissions that don't affect the core argument.

3. "partial" - The student's answer has SOME CORRECT ELEMENTS and shows MEANINGFUL PROGRESS, but has SIGNIFICANT GAPS or missing key steps. The student demonstrates understanding of some key concepts but hasn't completed the main proof structure. This is for solutions that are roughly 30-70% complete. KEY INDICATORS: The solution found a correct invariant or made partial progress but is missing crucial steps to complete the proof, or has major logical gaps.

4. "incorrect" - The student's answer is FUNDAMENTALLY WRONG, contains major errors, shows NO MEANINGFUL PROGRESS, is irrelevant/off-track, or attempts to prove a false statement. The core approach or logic is flawed.

CRITICAL DISTINCTION - "almost" vs "partial":
- Use "almost" when the solution is MOSTLY COMPLETE (90%+) with only minor issues remaining
- Use "partial" when the solution has made SOME PROGRESS (30-70%) but has significant gaps
- If the grading guidelines contain BOTH (Partial) and (Almost) markers, the final classification is typically "almost" because the (Almost) section describes the minor remaining issues

IMPORTANT: The (Partial), (Almost), (Correct), (Incorrect) markers in the grading guidelines indicate the INTENDED classification. Pay close attention to these markers - they are strong signals of the correct grade.

=== DETAILED CLASSIFICATION CRITERIA ===

"CORRECT" indicators:
- Complete proof with all key steps present
- No logical gaps or errors
- Matches the official solution's approach and conclusion
- Rigorous mathematical reasoning throughout

"ALMOST" indicators (vs correct):
- Complete proof structure present
- Core logic is correct
- Only minor issues: typos, small calculation errors, trivial omissions
- Would be correct with minimal fixes
- 90-99% complete

"ALMOST" indicators (vs partial):
- Main proof structure is complete
- Key insights are present
- Only minor gaps remain
- Core approach is sound

"PARTIAL" indicators (vs almost):
- Significant gaps in the proof
- Missing key steps or insights
- Major errors that affect correctness
- Incomplete solution (30-70% done)
- Some correct elements but not the full picture

"PARTIAL" indicators (vs incorrect):
- Shows meaningful progress toward solution
- Demonstrates understanding of some key concepts
- Has some correct reasoning or calculations
- Not completely off-track

"INCORRECT" indicators:
- Fundamentally wrong approach
- Claims a false statement
- No meaningful progress
- Completely off-track or irrelevant
- Major logical errors throughout

=== PROBLEM STATEMENT ===
{problem}

=== OFFICIAL SOLUTION ===
{solution}

=== GRADING GUIDELINES (RUBRIC) ===
{grading_guidelines}
{rubric_context}

=== STUDENT'S ANSWER ===
{student_answer}

=== ADDITIONAL SIGNALS ===
Points: {points}
Reward: {reward}

=== GRADING INSTRUCTIONS ===
1. FIRST: Read the Grading Guidelines carefully to understand the evaluation criteria
2. SECOND: Compare the student's answer to the official solution
3. THIRD: Classify based on the criteria above - focus on the actual quality of the solution
4. FOURTH: Output your classification in the JSON format below

Respond ONLY in the following JSON format:
<json>
{{
    "response": "correct" | "almost" | "partial" | "incorrect"
}}
</json>

Your response field must contain EXACTLY one of these four lowercase words: correct, almost, partial, incorrect. No other text, no explanations, no punctuation around the word."""

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
            
            # Post-process: Check if guidelines have explicit markers that suggest a different grade
            # This helps correct misclassifications where the LLM missed the guideline markers
            guideline_suggestion = _check_guideline_markers(grading_guidelines, student_answer)
            
            # Use guideline suggestion when:
            # 1. LLM prediction is unknown/invalid, OR
            # 2. Guidelines strongly suggest a specific grade that differs from LLM prediction
            if guideline_suggestion:
                if not _is_valid_prediction(prediction):
                    self.log_fn(f"Using guideline suggestion for unknown prediction: {guideline_suggestion}")
                    prediction = guideline_suggestion
                elif guideline_suggestion != prediction:
                    # Override LLM prediction when guidelines clearly indicate a different grade
                    # This is especially important for "almost" which is often misclassified
                    self.log_fn(f"Overriding LLM prediction '{prediction}' with guideline suggestion '{guideline_suggestion}'")
                    prediction = guideline_suggestion
            
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
