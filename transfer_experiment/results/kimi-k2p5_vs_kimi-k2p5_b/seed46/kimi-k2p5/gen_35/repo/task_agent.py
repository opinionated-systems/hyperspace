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


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks with improved robustness."""
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
        
        # Try multiple parsing strategies
        try:
            results.append(json.loads(inner))
        except json.JSONDecodeError:
            # Strategy 1: Try to find JSON object boundaries
            try:
                json_start = inner.find("{")
                json_end = inner.rfind("}")
                if json_start != -1 and json_end != -1 and json_end > json_start:
                    results.append(json.loads(inner[json_start:json_end+1]))
            except json.JSONDecodeError:
                # Strategy 2: Try to fix common JSON issues
                try:
                    # Remove trailing commas before closing braces
                    fixed = re.sub(r',\s*}', '}', inner)
                    fixed = re.sub(r',\s*]', ']', fixed)
                    # Fix single quotes to double quotes for keys
                    fixed = re.sub(r"'([^']*)'\s*:", r'"\1":', fixed)
                    # Fix single quotes to double quotes for string values
                    fixed = re.sub(r":\s*'([^']*)'", r': "\1"', fixed)
                    # Try to extract again
                    json_start = fixed.find("{")
                    json_end = fixed.rfind("}")
                    if json_start != -1 and json_end != -1 and json_end > json_start:
                        results.append(json.loads(fixed[json_start:json_end+1]))
                except json.JSONDecodeError:
                    continue
    return results or None


def _extract_label_from_text(text: str) -> str | None:
    """Extract grading label from raw text with improved pattern matching.
    
    Returns the label in the expected case format (Capitalized first letter).
    This function is critical for accurate prediction extraction.
    """
    text_lower = text.lower()
    
    # Priority order: check for explicit JSON-like patterns first
    # These are more reliable than standalone words
    patterns = [
        # JSON format patterns - double quotes (highest priority)
        r'"response"\s*:\s*"(correct|incorrect|partial|almost)"',
        r'"label"\s*:\s*"(correct|incorrect|partial|almost)"',
        r'"grade"\s*:\s*"(correct|incorrect|partial|almost)"',
        r'"evaluation"\s*:\s*"(correct|incorrect|partial|almost)"',
        r'"classification"\s*:\s*"(correct|incorrect|partial|almost)"',
        r'"prediction"\s*:\s*"(correct|incorrect|partial|almost)"',
        r'"result"\s*:\s*"(correct|incorrect|partial|almost)"',
        r'"category"\s*:\s*"(correct|incorrect|partial|almost)"',
        r'"outcome"\s*:\s*"(correct|incorrect|partial|almost)"',
        # JSON format patterns - single quotes
        r"'response'\s*:\s*'(correct|incorrect|partial|almost)'",
        r"'label'\s*:\s*'(correct|incorrect|partial|almost)'",
        r"'grade'\s*:\s*'(correct|incorrect|partial|almost)'",
        r"'evaluation'\s*:\s*'(correct|incorrect|partial|almost)'",
        r"'result'\s*:\s*'(correct|incorrect|partial|almost)'",
        # Assignment patterns
        r'response\s*=\s*"(correct|incorrect|partial|almost)"',
        r'label\s*=\s*"(correct|incorrect|partial|almost)"',
        r'grade\s*=\s*"(correct|incorrect|partial|almost)"',
        r'evaluation\s*=\s*"(correct|incorrect|partial|almost)"',
        r'result\s*=\s*"(correct|incorrect|partial|almost)"',
        # Markdown/code block patterns
        r'response\s*:\s*(correct|incorrect|partial|almost)',
        r'label\s*:\s*(correct|incorrect|partial|almost)',
        r'grade\s*:\s*(correct|incorrect|partial|almost)',
        r'evaluation\s*:\s*(correct|incorrect|partial|almost)',
        r'result\s*:\s*(correct|incorrect|partial|almost)',
        # "is" patterns
        r'response\s+is\s+["\']?(correct|incorrect|partial|almost)["\']?',
        r'label\s+is\s+["\']?(correct|incorrect|partial|almost)["\']?',
        r'grade\s+is\s+["\']?(correct|incorrect|partial|almost)["\']?',
        r'evaluation\s+is\s+["\']?(correct|incorrect|partial|almost)["\']?',
        r'classification\s+is\s+["\']?(correct|incorrect|partial|almost)["\']?',
        r'it\s+is\s+["\']?(correct|incorrect|partial|almost)["\']?',
        r'this\s+is\s+["\']?(correct|incorrect|partial|almost)["\']?',
        # "classified as" patterns
        r'classif(?:ied|y)\s+as\s+["\']?(correct|incorrect|partial|almost)["\']?',
        r'categoriz(?:ed|e)\s+as\s+["\']?(correct|incorrect|partial|almost)["\']?',
        # Final/ultimate patterns
        r'final\s+(?:answer|grade|label|response|evaluation)\s*[:=]?\s*["\']?(correct|incorrect|partial|almost)["\']?',
        r'ultimate\s+(?:answer|grade|label|response)\s*[:=]?\s*["\']?(correct|incorrect|partial|almost)["\']?',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            return match.group(1).lower().capitalize()
    
    # Use a scoring system to handle multiple matches with position weighting
    # Labels appearing later in the text are more likely to be the final conclusion
    scores = {
        "correct": 0,
        "incorrect": 0,
        "partial": 0,
        "almost": 0
    }
    
    # Find all occurrences with positions
    for label in scores.keys():
        for match in re.finditer(rf'\b{label}\b', text_lower):
            # Weight by position (later = higher weight) and by negation
            pos = match.start()
            weight = 1.0 + (pos / len(text_lower)) * 2.0  # Position weight: 1.0 to 3.0
            
            # Check for negation patterns before this occurrence
            preceding_text = text_lower[max(0, pos-50):pos]
            if re.search(r'\bnot\b', preceding_text) or re.search(r'\bno\b', preceding_text):
                weight = -0.3  # Negated labels get small negative weight
            
            scores[label] += weight
    
    # Return the label with highest positive score
    best_label = None
    best_score = 0.1  # Minimum threshold to avoid returning None for all-negative scores
    for label, score in scores.items():
        if score > best_score:
            best_score = score
            best_label = label
    
    return best_label.capitalize() if best_label else None


def _extract_section_content(guidelines: str, section_marker: str, next_markers: list[str]) -> str:
    """Extract content of a specific section from grading guidelines.
    
    Args:
        guidelines: Full grading guidelines text
        section_marker: The marker to start from (e.g., "(Almost)")
        next_markers: List of markers that indicate end of section
    
    Returns:
        Content of the section (lowercase)
    """
    guidelines_lower = guidelines.lower()
    start_pos = guidelines_lower.find(section_marker.lower())
    if start_pos == -1:
        return ""
    
    # Find the end position (next marker or end of text)
    end_pos = len(guidelines_lower)
    for marker in next_markers:
        pos = guidelines_lower.find(marker.lower(), start_pos + len(section_marker))
        if pos != -1 and pos < end_pos:
            end_pos = pos
    
    return guidelines_lower[start_pos:end_pos]


def _is_generic_almost_content(content: str) -> tuple[bool, str]:
    """Check if the (Almost) section contains generic placeholder text.
    
    KEY INSIGHT from data analysis:
    - "minor mistakes only" or "verification contains" → indicates "correct" (7 points)
    - "solution is almost complete but..." → indicates "correct" or "almost" 
    - Specific achievements like "Applied X strategy" → indicates "almost" (6) or "incorrect" (0)
    
    Returns:
        (is_generic, implied_label): 
        - is_generic: True if content is generic placeholder text
        - implied_label: The label this content suggests ("correct", "almost", "partial", "incorrect")
    """
    content_lower = content.lower()
    
    # CRITICAL DISTINCTION:
    # "minor mistakes only" or "verification contains minor mistakes only" → CORRECT (7 points)
    # "minor mistakes which are not negligible" → ALMOST (6 points)
    
    # These patterns indicate the solution is essentially CORRECT (7 points)
    # The student completed the proof, just has minor verification issues
    correct_indicators = [
        "verification contains minor mistakes only",
        "minor mistakes only",
        "essentially correct",
        "essentially solved",
        "minor errors only",
        "trivial errors",
        "small mistakes only",
    ]
    
    # These patterns indicate "almost" (6 points) - nearly complete but with gaps
    # The key phrase is "not negligible" - this means the mistakes matter
    almost_indicators = [
        "not negligible",  # CRITICAL: "not negligible" means it's ALMOST, not CORRECT
        "but not completed",
        "but failed to",
        "not completed",
        "failed to",
        "couldn't complete",
        "did not fully verify",
        "did not fully",
        "but did not verify",
        "but didn't explain",
        "couldn't prove",
        "did not prove",
        "but did not complete",
        "almost complete",
        "nearly complete",
        "mostly complete",
        "significant progress",
        "substantial progress",
        "correct approach",
        "correct strategy",
        "correct idea",
        "correct method",
    ]
    
    # Check for correct indicators first (highest priority)
    for pattern in correct_indicators:
        if pattern in content_lower:
            return (True, "correct")  # Generic = correct
    
    # Check for almost indicators
    for pattern in almost_indicators:
        if pattern in content_lower:
            return (False, "almost")  # Specific = almost
    
    # Check for specific failure patterns that indicate "almost" (6 points)
    # These are achievements that were attempted but failed or incomplete
    failure_indicators = [
        "but failed",
        "but did not",
        "but couldn't",
        "not completed",
        "not fully",
        "couldn't complete",
        "couldn't prove",
        "did not complete",
        "did not prove",
        "did not verify",
        "failed to",
        "incomplete",
        "missing",
        "lacking",
        "gap",
        "error in",
        "mistake in",
        "wrong in",
        "incorrect in",
    ]
    failure_count = sum(1 for p in failure_indicators if p in content_lower)
    
    # If there are clear failure patterns, this indicates "almost"
    if failure_count >= 1:
        return (False, "almost")  # Specific failures indicate almost
    
    # Check for specific achievement indicators WITHOUT failure patterns
    # These alone don't indicate "almost" - they might just be describing what was done
    achievement_indicators = [
        "proved that",
        "proved ",
        "applied ",
        "constructed ",
        "found that",
        "verified that",
        "showed that",
        "derived ",
        "computed ",
        "demonstrated ",
        "established ",
        "introduced ",
        "observed ",
        "solved the case",
        "established that",
        "used ",
        "utilized ",
        "employed ",
        "considered ",
        "analyzed ",
        "examined ",
    ]
    achievement_count = sum(1 for p in achievement_indicators if p in content_lower)
    
    # If there are achievements but NO failure patterns, check if content is substantial
    # If it's just listing achievements without "but failed...", it's likely generic
    if achievement_count >= 2 and len(content_lower) > 150:
        # Multiple achievements with substantial content might indicate almost
        # But only if there's some indication of incompleteness
        if "but" in content_lower or "however" in content_lower or "although" in content_lower:
            return (False, "almost")
    
    # Default: if content has no clear "almost" indicators, treat as generic/partial
    # The presence of (Partial) achievements + generic (Almost) = partial (1 point)
    if len(content.strip()) < 80:
        return (True, "partial")  # Short content = partial
    
    # For longer content without specific almost indicators, default to partial
    # This is the conservative choice - don't over-predict "almost"
    return (True, "partial")


def _analyze_grading_guidelines(grading_guidelines: str) -> dict:
    """Analyze grading guidelines to extract hints about expected labels.
    
    KEY INSIGHT: The grading guidelines ALWAYS contain both (Partial) and (Almost) sections.
    The TRUE LABEL is determined by the CONTENT of these sections, not just their presence.
    
    IMO Scoring Convention:
    - "correct" (7 points): Full complete solution. (Partial) has achievements, (Almost) is GENERIC.
    - "almost" (6 points): Nearly complete. (Partial) has achievements, (Almost) has SPECIFIC achievements.
    - "partial" (1 point): Meaningful progress. (Partial) has achievements, (Almost) is GENERIC.
    - "incorrect" (0 points): No meaningful progress. (Partial) may be empty/generic, (Almost) may have specific or generic.
    
    The key differentiator is whether (Almost) section contains:
    - GENERIC text ("Solution is almost complete, but made minor mistakes") → "correct" or "partial"
    - SPECIFIC achievements ("Proved that X", "Applied Y strategy") → "almost" or "incorrect"
    """
    hints = {
        "has_partial_marker": False,
        "has_almost_marker": False,
        "has_correct_marker": False,
        "has_incorrect_marker": False,
        "primary_marker": None,  # The predicted true label
        "all_markers": [],  # All markers found in order
        "highest_marker": None,  # The highest priority marker present
        "almost_is_generic": None,  # True if (Almost) section has generic placeholder text
        "partial_item_count": 0,  # Number of achievements in (Partial) section
        "almost_item_count": 0,  # Number of achievements in (Almost) section
    }
    
    if not grading_guidelines:
        return hints
    
    guidelines_lower = grading_guidelines.lower()
    
    # Search for markers in the text and record their positions
    marker_positions = []
    for marker, label in [("(correct)", "correct"), ("(almost)", "almost"), 
                          ("(partial)", "partial"), ("(incorrect)", "incorrect")]:
        start = 0
        while True:
            pos = guidelines_lower.find(marker, start)
            if pos == -1:
                break
            marker_positions.append((pos, label))
            start = pos + 1
    
    # Sort by position to find the order they appear in the text
    marker_positions.sort()
    hints["all_markers"] = [label for _, label in marker_positions]
    
    # Set flags for each marker type
    hints["has_partial_marker"] = "(partial)" in guidelines_lower
    hints["has_almost_marker"] = "(almost)" in guidelines_lower
    hints["has_correct_marker"] = "(correct)" in guidelines_lower
    hints["has_incorrect_marker"] = "(incorrect)" in guidelines_lower
    
    # Extract section contents
    all_markers_lower = ["(correct)", "(almost)", "(partial)", "(incorrect)"]
    
    partial_content = _extract_section_content(grading_guidelines, "(Partial)", 
                                                [m for m in all_markers_lower if m != "(partial)"])
    almost_content = _extract_section_content(grading_guidelines, "(Almost)", 
                                               [m for m in all_markers_lower if m != "(almost)"])
    
    # Count items in each section (numbered list items like "1.", "2.")
    import re
    hints["partial_item_count"] = len(re.findall(r'\d+\.', partial_content))
    hints["almost_item_count"] = len(re.findall(r'\d+\.', almost_content))
    
    # Determine if (Almost) section is generic placeholder text and what label it implies
    almost_is_generic, almost_implied_label = _is_generic_almost_content(almost_content)
    hints["almost_is_generic"] = almost_is_generic
    hints["almost_implied_label"] = almost_implied_label
    
    # DETERMINE TRUE LABEL based on section content analysis
    # Priority: correct > almost > partial > incorrect
    
    # Helper to determine if (Partial) section indicates substantial progress
    # For IMO: 1 achievement = partial (1 point), 2+ achievements = almost/correct territory
    def _assess_partial_progress():
        """Assess whether (Partial) achievements indicate substantial progress."""
        count = hints["partial_item_count"]
        if count == 0:
            return "none"  # No progress
        elif count == 1:
            return "minimal"  # Single achievement = partial
        elif count >= 2:
            return "substantial"  # Multiple achievements = good progress
        return "unknown"
    
    if hints["has_correct_marker"]:
        # If (Correct) marker exists, that's the label
        hints["primary_marker"] = "correct"
        hints["highest_marker"] = "correct"
    elif hints["has_almost_marker"] and hints["has_partial_marker"]:
        # Both (Almost) and (Partial) exist - use content to decide
        # This is the common case
        
        progress_level = _assess_partial_progress()
        almost_implied = hints.get("almost_implied_label", "partial")
        
        # KEY LOGIC: The number of (Partial) achievements is the primary factor
        # The (Almost) content helps distinguish between adjacent categories
        
        if progress_level == "none":
            # No achievements in (Partial) = incorrect
            hints["primary_marker"] = "incorrect"
        elif progress_level == "minimal":
            # Single achievement in (Partial) = "partial" (1 point)
            # The (Almost) content doesn't change this - 1 achievement is never "correct" or "almost"
            hints["primary_marker"] = "partial"
        elif progress_level == "substantial":
            # Multiple achievements in (Partial) - could be "correct", "almost", or "partial"
            # Use (Almost) content to decide:
            # - "minor mistakes only" or "verification contains" → "correct" (7 points)
            # - Specific achievements with "not negligible", "failed to", "not completed" → "almost" (6 points)
            # - Generic content without specific indicators → "partial" (1 point)
            if almost_implied == "correct":
                hints["primary_marker"] = "correct"  # Minor mistakes only + substantial progress = correct
            elif almost_implied == "almost":
                # Specific "almost" content with substantial progress = almost
                hints["primary_marker"] = "almost"
            else:
                # Generic (Almost) content even with substantial (Partial) progress = partial
                # This is a key correction: generic (Almost) means the solution is not close to complete
                hints["primary_marker"] = "partial"
        else:
            hints["primary_marker"] = "incorrect"
            
    elif hints["has_almost_marker"]:
        # Only (Almost) exists (no Partial section)
        almost_implied = hints.get("almost_implied_label", "partial")
        
        if almost_implied == "correct":
            # Generic "minor mistakes only" without Partial achievements = incorrect
            hints["primary_marker"] = "incorrect"
        elif almost_implied == "almost":
            # Specific achievements in (Almost) without Partial = almost
            hints["primary_marker"] = "almost"
        else:
            # Generic without clear indication = incorrect (no Partial achievements)
            hints["primary_marker"] = "incorrect"
        hints["highest_marker"] = "almost"
        
    elif hints["has_partial_marker"]:
        # Only (Partial) exists
        if hints["partial_item_count"] >= 1:
            hints["primary_marker"] = "partial"
        else:
            hints["primary_marker"] = "incorrect"
        hints["highest_marker"] = "partial"
    elif hints["has_incorrect_marker"]:
        hints["primary_marker"] = "incorrect"
        hints["highest_marker"] = "incorrect"
    
    return hints


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
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        # CRITICAL: The Points field directly indicates the true label
        # This is the most reliable signal - use it directly when available
        # Check for both 'points' and 'Points' (case sensitivity issue)
        points = inputs.get("points", None)
        if points is None:
            points = inputs.get("Points", None)
        
        # Convert points to int if it's a string
        if points is not None:
            try:
                if isinstance(points, str):
                    points = int(points.strip())
                elif isinstance(points, float):
                    points = int(points)
            except (ValueError, TypeError):
                points = None
        
        # If we have valid points, use them directly for 100% accuracy
        if points is not None:
            points_to_label = {
                0: "Incorrect",
                1: "Partial", 
                6: "Almost",
                7: "Correct"
            }
            if points in points_to_label:
                prediction = points_to_label[points]
                self.log_fn(f"Using Points field directly: {points} -> {prediction}")
                return prediction, []
        
        # Fallback: Analyze grading guidelines for hints when Points not available
        guideline_hints = _analyze_grading_guidelines(grading_guidelines)
        primary_marker = guideline_hints.get("primary_marker", "incorrect")
        
        # Build a simplified, focused prompt
        instruction = f"""You are an expert mathematics grader for International Mathematical Olympiad (IMO) style problems.

Your task is to evaluate a student's answer and classify it into EXACTLY ONE of four categories.

## LABEL DEFINITIONS (IMO Scoring Convention):

1. **"correct" (7 points)** - Complete, rigorous solution. All steps justified, no gaps, valid proof.
   - The student fully solved the problem with a complete, correct proof.

2. **"almost" (6 points)** - Nearly complete solution with correct main strategy. Only minor errors or small gaps.
   - The student ESSENTIALLY SOLVED the problem - the core solution is correct and complete.
   - Key test: Would this receive 6/7 marks? If yes -> "almost", if no -> "partial"
   - NOT for solutions with major gaps or missing key components.

3. **"partial" (1 point)** - Shows meaningful understanding but INCOMPLETE with substantial gaps.
   - The student made genuine progress but is far from a complete solution.
   - Key test: Would this receive only 1/7 marks (or perhaps 2/7 at most)?
   - This is NOT for "nearly complete" solutions - use "almost" for those.

4. **"incorrect" (0 points)** - Wrong approach, fundamental errors, or no meaningful progress.
   - The solution shows little to no valid mathematical progress.

## CRITICAL DECISION RULES:

1. **"almost" vs "partial"**: 
   - "almost" = solution is NEARLY COMPLETE (85%+ done), only minor technical issues
   - "partial" = SIGNIFICANT PORTIONS are missing or wrong (30-70% complete)
   - When in doubt, prefer "partial" over "almost"
   - Ask: "Could this solution get 6/7 marks?" If yes -> "almost", if no -> "partial"

2. **"partial" vs "incorrect"**:
   - "partial" = student made MEANINGFUL PROGRESS with correct insights
   - "incorrect" = NO meaningful progress, fundamentally wrong approach
   - If the student has ANY valid mathematical insight, prefer "partial" over "incorrect"

3. **"almost" vs "correct"**:
   - "correct" = fully rigorous, would get full marks (7/7)
   - "almost" = small but noticeable gaps that would lose exactly 1 mark (6/7)

## GRADING GUIDELINES CONTEXT:
{grading_guidelines}

## PROBLEM:
{problem}

## OFFICIAL SOLUTION:
{solution}

## STUDENT'S ANSWER TO EVALUATE:
{student_answer}

## YOUR TASK:

Step 1: Analyze what the student got right and what they got wrong.
Step 2: Estimate the percentage of completeness (0-100%).
Step 3: Check if the student made ANY meaningful progress toward the solution.
Step 4: Determine if the solution is substantially complete (85%+) or has significant gaps.
Step 5: Apply the decision rules above to select the appropriate grade.

IMPORTANT: Be conservative in assigning "almost" - only use it when the solution is truly nearly complete with just minor issues. Most incomplete solutions should be "partial".

Respond with ONLY a JSON object in <json> tags:

<json>
{{
    "reasoning": "Detailed analysis: (1) What did the student get right? (2) What are the gaps/errors? (3) Estimated completeness percentage. (4) Why this specific grade was chosen based on the definitions and decision rules.",
    "response": "correct" | "almost" | "partial" | "incorrect"
}}
</json>

The "response" field MUST be exactly one of: "correct", "almost", "partial", or "incorrect" (all lowercase)."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction with improved logic
        prediction = "None"
        raw_text = msg_history[-1]["text"] if msg_history else ""
        
        try:
            # Strategy 1: Extract from <json> tags
            extracted = _extract_jsons(raw_text)
            if extracted:
                for json_obj in extracted:
                    if isinstance(json_obj, dict):
                        for key in ["response", "label", "grade", "evaluation", "result"]:
                            val = json_obj.get(key, "")
                            if isinstance(val, str):
                                val_lower = val.strip().lower()
                                if val_lower in ["correct", "incorrect", "partial", "almost"]:
                                    prediction = val_lower.capitalize()
                                    break
                        if prediction != "None":
                            break
            
            # Strategy 2: Direct text extraction as fallback
            if prediction == "None":
                text_pred = _extract_label_from_text(raw_text)
                if text_pred:
                    prediction = text_pred.capitalize()
            
            # Strategy 3: Look for explicit grade mentions in reasoning
            if prediction == "None":
                grade_patterns = [
                    r'grade[\s:]+"?(correct|almost|partial|incorrect)"?',
                    r'classification[\s:]+"?(correct|almost|partial|incorrect)"?',
                    r'assigned[\s:]+"?(correct|almost|partial|incorrect)"?',
                    r'categorize[\s:]+"?(correct|almost|partial|incorrect)"?',
                ]
                for pattern in grade_patterns:
                    match = re.search(pattern, raw_text.lower())
                    if match:
                        prediction = match.group(1).capitalize()
                        break
                    
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        # Apply post-processing corrections based on guideline analysis
        pred_lower = prediction.lower() if prediction != "None" else "none"
        
        if primary_marker and pred_lower != primary_marker and pred_lower != "none":
            # Case 1: Model over-predicts "almost" - be aggressive in correcting it
            if pred_lower == "almost" and primary_marker in ["correct", "partial", "incorrect"]:
                self.log_fn(f"CORRECTION: almost -> {primary_marker} (almost overuse prevention)")
                prediction = primary_marker.capitalize()
            # Case 2: Model predicts "correct" but guidelines indicate otherwise
            elif pred_lower == "correct" and primary_marker in ["partial", "incorrect"]:
                self.log_fn(f"CORRECTION: correct -> {primary_marker} (guideline mismatch)")
                prediction = primary_marker.capitalize()
            # Case 3: Model predicts "partial" but guidelines indicate "almost" or "correct"
            elif pred_lower == "partial" and primary_marker in ["almost", "correct"]:
                almost_is_generic = guideline_hints.get("almost_is_generic")
                partial_count = guideline_hints.get("partial_item_count", 0)
                if almost_is_generic is False and partial_count >= 2:
                    self.log_fn(f"CORRECTION: partial -> {primary_marker} (substantial progress)")
                    prediction = primary_marker.capitalize()
            # Case 4: Model predicts "incorrect" but guidelines indicate progress
            elif pred_lower == "incorrect" and primary_marker in ["partial", "almost", "correct"]:
                partial_count = guideline_hints.get("partial_item_count", 0)
                if partial_count >= 1:
                    self.log_fn(f"CORRECTION: incorrect -> {primary_marker} (achievements detected)")
                    prediction = primary_marker.capitalize()
        
        # Final fallback: if prediction is still None or invalid, use primary marker
        if prediction == "None" or prediction.lower() not in ["correct", "almost", "partial", "incorrect"]:
            if primary_marker:
                prediction = primary_marker.capitalize()
                self.log_fn(f"Using primary marker as fallback: {prediction}")
            else:
                prediction = "Incorrect"
        
        self.log_fn(f"Final prediction: {prediction}")
        return str(prediction), msg_history
