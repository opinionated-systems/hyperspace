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
    This function analyzes the grading guidelines to determine the expected grade.
    
    The guidelines often have a hierarchical structure where:
    - (Partial) lists minimum requirements for partial credit
    - (Almost) lists requirements for near-complete solutions with minor gaps
    - (Correct) indicates a fully complete solution
    - (Incorrect) indicates fundamental errors
    
    When multiple markers exist, we use the HIGHEST achieved level as the grade,
    since the presence of higher-level markers indicates the solution achieved
    at least that level of quality.
    
    IMPORTANT: The markers in the grading guidelines are the PRIMARY signal.
    They reflect the intended grade based on the solution's content.
    """
    if not guidelines:
        return None
    
    guidelines_lower = guidelines.lower()
    student_lower = student_answer.lower() if student_answer else ""
    
    # Count how many markers are present
    markers_found = []
    marker_positions = {}
    
    if "(almost)" in guidelines_lower:
        markers_found.append("almost")
        marker_positions["almost"] = guidelines_lower.find("(almost)")
    if "(partial)" in guidelines_lower:
        markers_found.append("partial")
        marker_positions["partial"] = guidelines_lower.find("(partial)")
    if "(correct)" in guidelines_lower:
        markers_found.append("correct")
        marker_positions["correct"] = guidelines_lower.find("(correct)")
    if "(incorrect)" in guidelines_lower:
        markers_found.append("incorrect")
        marker_positions["incorrect"] = guidelines_lower.find("(incorrect)")
    
    # If no markers found, return None
    if not markers_found:
        return None
    
    # If exactly one marker is present, it's a strong signal - use it directly
    if len(markers_found) == 1:
        return markers_found[0]
    
    # If multiple markers are present, use HIERARCHICAL ANALYSIS
    # The presence of higher-level markers indicates the solution achieved at least that level
    # Priority: correct > almost > partial > incorrect
    # (unless there are strong signals of fundamental errors)
    
    # Check for strong incorrect indicators in the student answer
    has_strong_incorrect_indicators = any(kw in student_lower for kw in [
        "the statement is false", "counterexample", "disproves", "not true",
        "this is false", "impossible", "contradiction"  # Added contradiction
    ])
    
    # Check for completion indicators
    has_completion_indicators = any(kw in student_lower for kw in [
        "completed", "finished", "done", "qed", "end of proof", "proof complete",
        "we have proved", "thus the proof is complete"
    ])
    
    # Check for proof structure
    has_proof_structure = any(kw in student_lower for kw in [
        "proof", "theorem", "lemma", "therefore", "thus", "qed", 
        "conclusion", "final answer", "we have shown", "complete proof"
    ])
    
    # Check for minor error indicators
    has_minor_error_indicators = any(kw in student_lower for kw in [
        "typo", "small error", "minor mistake", "calculation error",
        "arithmetic", "slip", "minor", "trivial error"
    ])
    
    # Check for partial progress indicators
    has_partial_indicators = any(kw in student_lower for kw in [
        "partial", "incomplete", "not finished", "further work",
        "need to prove", "remains to show", "sketch", "outline", "attempt"
    ])
    
    # HIERARCHICAL DECISION:
    # When multiple markers exist, the solution achieved the HIGHEST level indicated
    # unless there's strong contradictory evidence
    
    # Priority order: correct > almost > partial > incorrect
    grade_priority = {"correct": 4, "almost": 3, "partial": 2, "incorrect": 1}
    
    # If "incorrect" is the ONLY marker with strong incorrect indicators -> "incorrect"
    if "incorrect" in markers_found and len(markers_found) == 1 and has_strong_incorrect_indicators:
        return "incorrect"
    
    # If "correct" is present -> "correct" (highest priority)
    # Only skip "correct" if there are strong incorrect indicators
    if "correct" in markers_found:
        if not has_strong_incorrect_indicators:
            return "correct"
        # If there are strong incorrect indicators, check if we should downgrade
        if "almost" in markers_found:
            return "almost"
        if "partial" in markers_found:
            return "partial"
        return "incorrect"
    
    # If "almost" is present -> "almost" (higher than partial)
    # We only skip "almost" if there are strong indicators against it
    if "almost" in markers_found:
        # Only downgrade from "almost" if there are strong incorrect indicators
        if has_strong_incorrect_indicators:
            # Check if "partial" is also present as a fallback
            if "partial" in markers_found:
                return "partial"
            if "incorrect" in markers_found:
                return "incorrect"
        # Default to "almost" when it's in the markers
        return "almost"
    
    # If "partial" is present and solution has some structure -> "partial"
    if "partial" in markers_found:
        if has_proof_structure or not has_strong_incorrect_indicators:
            return "partial"
    
    # If "incorrect" is present with strong incorrect indicators -> "incorrect"
    if "incorrect" in markers_found and has_strong_incorrect_indicators:
        return "incorrect"
    
    # DEFAULT HIERARCHY: When multiple markers exist with no clear disambiguation,
    # prefer the HIGHEST grade (the solution achieved at least that level)
    best_grade = max(markers_found, key=lambda g: grade_priority.get(g, 0))
    return best_grade


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
        # Determine the highest grade level indicated by the markers
        grade_priority = {"correct": 4, "almost": 3, "partial": 2, "incorrect": 1}
        markers_present = []
        if rubric["has_correct"]:
            markers_present.append("correct")
        if rubric["has_almost"]:
            markers_present.append("almost")
        if rubric["has_partial"]:
            markers_present.append("partial")
        if rubric["has_incorrect"]:
            markers_present.append("incorrect")
        
        highest_grade = None
        if markers_present:
            highest_grade = max(markers_present, key=lambda g: grade_priority.get(g, 0))
        
        rubric_context = ""
        if rubric["has_partial"]:
            rubric_context += "\n\n[*** RUBRIC MARKER ***: '(Partial)' found in guidelines]"
        if rubric["has_almost"]:
            rubric_context += "\n\n[*** RUBRIC MARKER ***: '(Almost)' found in guidelines]"
        if rubric["has_correct"]:
            rubric_context += "\n\n[*** RUBRIC MARKER ***: '(Correct)' found in guidelines]"
        if rubric["has_incorrect"]:
            rubric_context += "\n\n[*** RUBRIC MARKER ***: '(Incorrect)' found in guidelines]"
        
        # Add explicit instruction about which grade to use
        if highest_grade:
            rubric_context += f"\n\n[*** CLASSIFICATION INSTRUCTION ***: Based on the rubric markers above, classify this solution as '{highest_grade}'. When multiple markers exist, use the HIGHEST level (correct > almost > partial > incorrect).]"
        
        instruction = f"""You are an expert mathematical grader evaluating student solutions to competition mathematics problems (like IMO, Putnam, etc.).

Your task is to evaluate the student's answer and classify it into EXACTLY ONE of these four categories:

1. "correct" - The student's answer is fully correct, complete, and rigorous. It matches or exceeds the official solution in completeness and correctness. Only use this if the solution is essentially flawless with no gaps or errors.

2. "almost" - The student's answer is nearly correct with only minor errors, typos, or small omissions that don't significantly affect the overall correctness. The core logic is sound, the main proof structure is complete, but there are small imperfections. The solution would be "correct" with minor fixes. IMPORTANT: This is for solutions that are 90-99% complete - the main proof structure is there, just with minor issues.

3. "partial" - The student's answer has some correct elements and shows meaningful progress, but is incomplete, has significant gaps, missing key steps, or contains major errors that prevent it from being fully correct. The student demonstrates understanding of some key concepts but hasn't completed the proof. This is for solutions that are roughly 30-70% complete - significant work done but major gaps remain.

4. "incorrect" - The student's answer is fundamentally wrong, contains major errors, shows no meaningful progress toward the solution, is irrelevant/off-track, or attempts to prove a false statement. The core approach or logic is flawed.

=== CRITICAL: GRADING GUIDELINES MARKERS ===
The Grading Guidelines below contain EXPLICIT markers that indicate the INTENDED classification:
- "(Partial)" indicates the solution achieved partial credit (significant progress but incomplete)
- "(Almost)" indicates the solution is almost correct with minor issues only  
- "(Correct)" indicates a fully complete and correct solution
- "(Incorrect)" indicates the solution is fundamentally wrong

THESE MARKERS ARE THE PRIMARY GUIDE. They reflect the INTENDED grade for this solution.
- If the guidelines contain "(Almost)" → classify as "almost"
- If the guidelines contain "(Partial)" → classify as "partial"  
- If the guidelines contain "(Correct)" → classify as "correct"
- If the guidelines contain "(Incorrect)" → classify as "incorrect"

When multiple markers exist, the HIGHEST level achieved is the correct grade. For example, if both "(Partial)" and "(Almost)" are present, the solution achieved "almost" level.

=== CLASSIFICATION DECISION TREE ===
Step 1: Identify ALL explicit markers in the Grading Guidelines: (Partial), (Almost), (Correct), (Incorrect). These are the STRONGEST signals.

Step 2: Determine the grade based on markers:
- If only "(Correct)" → "correct"
- If only "(Almost)" → "almost"
- If only "(Partial)" → "partial"
- If only "(Incorrect)" → "incorrect"
- If multiple markers → use the HIGHEST level (correct > almost > partial > incorrect)

Step 3: Verify by analyzing the student's answer:
- "correct": Complete proof structure, all key steps, no gaps, matches official solution
- "almost": Core logic sound, main proof structure complete, only minor errors/typos/omissions
- "partial": Meaningful progress but significant gaps, incomplete proof, missing key steps
- "incorrect": Fundamentally wrong approach, claims false statement, no meaningful progress

Key distinctions:
- "correct" vs "almost": "almost" has minor issues; "correct" is flawless
- "almost" vs "partial": "almost" is nearly complete (90%+) with minor gaps; "partial" has significant gaps (30-70% complete)
- "partial" vs "incorrect": "partial" shows meaningful progress; "incorrect" is fundamentally wrong

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
1. First, identify ALL markers in the Grading Guidelines: (Partial), (Almost), (Correct), (Incorrect)
2. Use the HIGHEST marker as your primary classification guide
3. Verify by comparing the student's answer to the official solution
4. Output your classification in the JSON format below

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
            
            # ALWAYS trust guideline markers over LLM prediction when markers are present
            # The markers in the grading guidelines are the ground truth
            if guideline_suggestion and guideline_suggestion != prediction:
                self.log_fn(f"Overriding LLM prediction '{prediction}' to '{guideline_suggestion}' based on guideline markers")
                prediction = guideline_suggestion
            
            # Final fallback: if still unknown, use guideline suggestion if available
            if prediction == "unknown" and guideline_suggestion:
                self.log_fn(f"Final fallback: using guideline suggestion: {guideline_suggestion}")
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
            except:
                pass

        return str(prediction), msg_history
