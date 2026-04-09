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
    
    # Look for explicit markers with case-insensitive matching
    markers = [
        (r'\(\s*Partial\s*\)', "partial"),
        (r'\(\s*Almost\s*\)', "almost"),
        (r'\(\s*Correct\s*\)', "correct"),
        (r'\(\s*Incorrect\s*\)', "incorrect"),
        # Credit-based markers
        (r'\(\s*Full\s*Credit\s*\)', "correct"),
        (r'\(\s*No\s*Credit\s*\)', "incorrect"),
        (r'\(\s*Half\s*Credit\s*\)', "partial"),
        (r'\(\s*Most\s*Credit\s*\)', "almost"),
        (r'\(\s*Some\s*Credit\s*\)', "partial"),
    ]
    
    detected_markers = []
    for pattern, label in markers:
        matches = list(re.finditer(pattern, guidelines, re.IGNORECASE))
        if matches:
            result[f"has_{label}"] = True
            detected_markers.append((label, len(matches)))
            for match in matches:
                # Extract context around the marker
                start = max(0, match.start() - 50)
                end = min(len(guidelines), match.end() + 100)
                context = guidelines[start:end].replace('\n', ' ').strip()
                result[f"{label}_context"].append(context)
    
    # Determine primary category based on markers
    if detected_markers:
        # Count unique marker types
        unique_types = set(m[0] for m in detected_markers)
        
        if len(unique_types) == 1:
            # Single marker type - high confidence
            result["primary_category"] = list(unique_types)[0]
            result["confidence"] = 0.95
        else:
            # Multiple markers - use priority order
            priority = ["correct", "almost", "partial", "incorrect"]
            for p in priority:
                if p in unique_types:
                    result["primary_category"] = p
                    result["confidence"] = 0.8
                    break
    
    return result


def _check_guideline_markers(guidelines: str, student_answer: str) -> str | None:
    """Check if grading guidelines have explicit markers that indicate the grade.
    
    Returns the suggested grade based on guideline markers, or None if unclear.
    The markers (Partial), (Almost), (Correct), (Incorrect) in the grading guidelines
    indicate the INTENDED classification for this solution.
    
    IMPROVED LOGIC:
    - If exactly ONE marker type is found, return that marker
    - If BOTH (Partial) and (Almost) are found, the FINAL marker typically indicates 
      the intended classification. The (Partial) section describes partial progress made,
      while the (Almost) section describes what makes it "almost" correct.
    - If multiple conflicting markers exist, return None for LLM to decide
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
    
    # Get unique marker types
    unique_markers = list(dict.fromkeys([m[0] for m in markers_with_pos]))
    
    # If exactly ONE unique marker type is found, return it
    if len(unique_markers) == 1:
        return unique_markers[0]
    
    # If BOTH (Partial) and (Almost) are found, the LAST marker typically indicates
    # the FINAL classification. The rubric structure usually describes progress first,
    # then the final state.
    if len(unique_markers) == 2 and "partial" in unique_markers and "almost" in unique_markers:
        # Sort by position to find which appears last
        markers_with_pos.sort(key=lambda x: x[1])
        last_marker = markers_with_pos[-1][0]
        
        # Count occurrences of each marker
        partial_count = sum(1 for m in markers_with_pos if m[0] == "partial")
        almost_count = sum(1 for m in markers_with_pos if m[0] == "almost")
        
        # If (Almost) appears more times, it's likely "almost"
        # Otherwise, if (Partial) is dominant, it might be "partial"
        if almost_count > partial_count:
            return "almost"
        elif partial_count > almost_count:
            return "partial"
        else:
            # Equal counts - use LAST occurrence (final state)
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
            marker = markers_found[0].lower().strip("()")
            rubric_context += f"\n\n[*** RUBRIC INDICATOR ***: The grading guidelines contain exactly one marker: {markers_found[0]}. This STRONGLY indicates the intended classification is '{marker}'. You MUST classify this solution as '{marker}' unless there is overwhelming evidence to the contrary.]"
        elif len(markers_found) == 2 and "(Partial)" in markers_found and "(Almost)" in markers_found:
            # Special case: Both Partial and Almost markers - the LAST marker indicates final state
            # Count occurrences in the guidelines
            partial_count = len(re.findall(r'\(partial\)', grading_guidelines.lower()))
            almost_count = len(re.findall(r'\(almost\)', grading_guidelines.lower()))
            
            # Find which marker appears last (final state)
            partial_pos = grading_guidelines.lower().rfind("(partial)")
            almost_pos = grading_guidelines.lower().rfind("(almost)")
            
            if almost_count > partial_count:
                rubric_context += f"\n\n[*** RUBRIC INDICATORS ***: The grading guidelines contain BOTH (Partial) and (Almost) markers, but (Almost) appears more frequently ({almost_count} vs {partial_count}). This indicates the FINAL classification should be 'almost'. The (Partial) section describes partial progress made, while the (Almost) section describes the minor remaining issues. Use 'almost'.]"
            elif partial_count > almost_count:
                rubric_context += f"\n\n[*** RUBRIC INDICATORS ***: The grading guidelines contain BOTH (Partial) and (Almost) markers, but (Partial) appears more frequently ({partial_count} vs {almost_count}). This indicates the classification should be 'partial'. The solution has made partial progress but still has significant gaps. Use 'partial'.]"
            else:
                # Equal counts - use LAST occurrence (final state)
                if almost_pos > partial_pos:
                    rubric_context += f"\n\n[*** RUBRIC INDICATORS ***: The grading guidelines contain BOTH (Partial) and (Almost) markers with equal frequency. (Almost) appears LAST, indicating the FINAL classification should be 'almost'. The (Partial) section describes partial progress made, while the (Almost) section describes the minor remaining issues. Use 'almost'.]"
                else:
                    rubric_context += f"\n\n[*** RUBRIC INDICATORS ***: The grading guidelines contain BOTH (Partial) and (Almost) markers with equal frequency. (Partial) appears LAST, indicating the classification should be 'partial'. The solution has made partial progress but still has significant gaps. Use 'partial'.]"
        elif len(markers_found) > 1:
            # Multiple markers - the LLM needs to analyze the full context
            # Priority: correct > almost > partial > incorrect
            priority = ["(Correct)", "(Almost)", "(Partial)", "(Incorrect)"]
            highest_marker = None
            for p in priority:
                if p in markers_found:
                    highest_marker = p
                    break
            if highest_marker:
                rubric_context += f"\n\n[*** RUBRIC INDICATORS ***: The grading guidelines contain multiple markers: {', '.join(markers_found)}. The highest priority marker is {highest_marker}, which typically indicates the intended classification. Analyze the full context, prioritizing the {highest_marker} marker.]"
            else:
                rubric_context += f"\n\n[*** RUBRIC INDICATORS ***: The grading guidelines contain multiple markers: {', '.join(markers_found)}. Analyze the full context to determine the appropriate classification. The marker that appears most frequently or describes the FINAL STATE of the solution is typically the correct classification.]"
        
        instruction = f"""You are an expert mathematical grader evaluating student solutions to competition mathematics problems (like IMO, Putnam, etc.).

Your task is to evaluate the student's answer and classify it into EXACTLY ONE of these four categories:

1. "correct" - The student's answer is fully correct, complete, and rigorous. It matches or exceeds the official solution in completeness and correctness. Only use this if the solution is essentially flawless with no gaps or errors.

2. "almost" - The student's answer is NEARLY CORRECT with only MINOR ERRORS, typos, or small omissions. The core logic is sound, the main proof structure is complete, and the solution would be "correct" with minor fixes. This is for solutions that are 90-99% complete. KEY INDICATORS: The solution has the main proof structure but has minor mistakes in verification, small calculation errors, or trivial omissions that don't affect the core argument.

3. "partial" - The student's answer has SOME CORRECT ELEMENTS and shows MEANINGFUL PROGRESS, but has SIGNIFICANT GAPS or missing key steps. The student demonstrates understanding of some key concepts but hasn't completed the main proof structure. This is for solutions that are roughly 30-70% complete. KEY INDICATORS: The solution found a correct invariant or made partial progress but is missing crucial steps to complete the proof, or has major logical gaps.

4. "incorrect" - The student's answer is FUNDAMENTALLY WRONG, contains major errors, shows NO MEANINGFUL PROGRESS, is irrelevant/off-track, or attempts to prove a false statement. The core approach or logic is flawed.

=== RUBRIC MARKERS - FOLLOW THESE EXPLICITLY ===
The grading guidelines below contain EXPLICIT markers that indicate the intended classification:
- (Correct) → Use "correct"
- (Almost) → Use "almost"  
- (Partial) → Use "partial"
- (Incorrect) → Use "incorrect"

IMPORTANT: These markers are the STRONGEST SIGNAL of the correct grade. The (Partial) section describes what partial progress was made, while the (Almost) section describes what makes it "almost" correct. When both appear, the (Almost) section typically describes the FINAL classification.

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
- Solution is nearly complete, just needs minor fixes

"PARTIAL" indicators (vs almost):
- Significant gaps in the proof
- Missing key steps or insights
- Major errors that affect correctness
- Incomplete solution (30-70% done)
- Some correct elements but not the full picture
- Solution is far from complete

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

=== CRITICAL DECISION RULES ===

When deciding between "almost" and "partial":
1. If the solution has the COMPLETE main proof structure with only minor issues → "almost"
2. If the solution is MISSING significant parts of the proof → "partial"
3. If the grading guidelines have BOTH (Partial) and (Almost) markers:
   - The LAST marker typically indicates the FINAL classification
   - The (Partial) section describes progress made
   - The (Almost) section describes the final state with minor issues
   - When in doubt, prefer "almost" if the solution is mostly complete

When the rubric has a single marker:
- You MUST use that classification unless there is overwhelming evidence to the contrary

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
1. FIRST: Read the Grading Guidelines carefully and identify the rubric markers (Correct/Almost/Partial/Incorrect)
2. SECOND: Compare the student's answer to the official solution
3. THIRD: Classify based on the criteria above AND the rubric markers
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
            
            # Additional check: If we have a single clear marker in the rubric, use it
            # This is a stronger override for cases where LLM ignores the rubric
            if _is_valid_prediction(prediction):
                # Count markers in guidelines
                markers_in_guidelines = []
                g_lower = grading_guidelines.lower()
                if re.search(r'\(correct\)', g_lower):
                    markers_in_guidelines.append("correct")
                if re.search(r'\(almost\)', g_lower):
                    markers_in_guidelines.append("almost")
                if re.search(r'\(partial\)', g_lower):
                    markers_in_guidelines.append("partial")
                if re.search(r'\(incorrect\)', g_lower):
                    markers_in_guidelines.append("incorrect")
                
                # If exactly one marker and it differs from prediction, strongly consider overriding
                if len(markers_in_guidelines) == 1 and markers_in_guidelines[0] != prediction:
                    self.log_fn(f"Single marker '{markers_in_guidelines[0]}' found in guidelines, overriding '{prediction}'")
                    prediction = markers_in_guidelines[0]
            
            # Additional fallback: If prediction is still unknown, use rubric primary category
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
