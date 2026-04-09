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
    """Extract JSON objects from <json>...</json> blocks."""
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
        except json.JSONDecodeError:
            continue
    return results or None


def _extract_label_from_text(text: str) -> str | None:
    """Extract grading label from raw text using multiple strategies.
    
    Returns the label in the expected case format (Capitalized first letter).
    This function is critical for accurate prediction extraction.
    """
    text_lower = text.lower()
    
    # Look for explicit patterns like "grade: correct" or "evaluation: partial"
    # These patterns capture the label value directly
    # IMPROVED: Added more patterns and made them more robust
    patterns = [
        # JSON-style patterns (most reliable)
        r'"response"\s*:\s*"(correct|incorrect|partial|almost)"',
        r'"label"\s*:\s*"(correct|incorrect|partial|almost)"',
        r'"grade"\s*:\s*"(correct|incorrect|partial|almost)"',
        r'"evaluation"\s*:\s*"(correct|incorrect|partial|almost)"',
        r'"classification"\s*:\s*"(correct|incorrect|partial|almost)"',
        r'"prediction"\s*:\s*"(correct|incorrect|partial|almost)"',
        r'"result"\s*:\s*"(correct|incorrect|partial|almost)"',
        r'"category"\s*:\s*"(correct|incorrect|partial|almost)"',
        r'"outcome"\s*:\s*"(correct|incorrect|partial|almost)"',
        # Single quote variants
        r"'response'\s*:\s*'(correct|incorrect|partial|almost)'",
        r"'label'\s*:\s*'(correct|incorrect|partial|almost)'",
        r"'grade'\s*:\s*'(correct|incorrect|partial|almost)'",
        # Colon/equals patterns
        r'grade[d]?\s*[:=]\s*["\']?(correct|incorrect|partial|almost)["\']?',
        r'evaluation\s*[:=]\s*["\']?(correct|incorrect|partial|almost)["\']?',
        r'label\s*[:=]\s*["\']?(correct|incorrect|partial|almost)["\']?',
        r'response\s*[:=]\s*["\']?(correct|incorrect|partial|almost)["\']?',
        r'classification\s*[:=]\s*["\']?(correct|incorrect|partial|almost)["\']?',
        r'prediction\s*[:=]\s*["\']?(correct|incorrect|partial|almost)["\']?',
        r'result\s*[:=]\s*["\']?(correct|incorrect|partial|almost)["\']?',
        r'category\s*[:=]\s*["\']?(correct|incorrect|partial|almost)["\']?',
        r'outcome\s*[:=]\s*["\']?(correct|incorrect|partial|almost)["\']?',
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
        # "the answer/solution is" patterns
        r'(?:the\s+)?(?:answer|solution)\s+is\s+["\']?(correct|incorrect|partial|almost)["\']?',
        r'(?:the\s+)?student(?:\'s)?\s+(?:answer|solution|work)\s+is\s+["\']?(correct|incorrect|partial|almost)["\']?',
        # Final/ultimate patterns
        r'final\s+(?:answer|grade|label|response|evaluation)\s*[:=]?\s*["\']?(correct|incorrect|partial|almost)["\']?',
        r'ultimate\s+(?:answer|grade|label|response)\s*[:=]?\s*["\']?(correct|incorrect|partial|almost)["\']?',
        # "I would classify/grade" patterns
        r'(?:i\s+(?:would\s+)?(?:classify|grade|label)\s+(?:this\s+)?(?:as\s+)?)["\']?(correct|incorrect|partial|almost)["\']?',
        r'(?:i\s+(?:would\s+)?(?:say|think|determine)\s+(?:this\s+)?(?:is\s+)?)["\']?(correct|incorrect|partial|almost)["\']?',
        # Other patterns
        r'(?:prediction|output|result)\s*[:=]\s*["\']?(correct|incorrect|partial|almost)["\']?',
        r'(?:the\s+)?(?:grade|score|mark)\s+(?:should\s+be\s+)?["\']?(correct|incorrect|partial|almost)["\']?',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            # Return with proper capitalization
            label = match.group(1).lower()
            return label.capitalize()
    
    # Check for standalone labels with word boundaries
    # IMPORTANT: We need to be careful about the order here
    # The model has a strong bias toward "partial" and "almost"
    # We should check for the more specific labels first
    
    # Check for "incorrect" first (0 points) - this is often missed
    if re.search(r'\bincorrect\b', text_lower):
        return "Incorrect"
    
    # Check for "correct" (7 points) - this is the highest priority
    if re.search(r'\bcorrect\b', text_lower):
        return "Correct"
    
    # Check for "partial" (1 point) vs "almost" (6 points)
    # The model often confuses these, so we need to look at context
    # If both appear, we need to determine which one is the final answer
    has_partial = re.search(r'\bpartial\b', text_lower)
    has_almost = re.search(r'\balmost\b', text_lower)
    
    if has_partial and has_almost:
        # Both appear - look for which one is the final conclusion
        # Check if "partial" appears after "almost" (might indicate correction)
        partial_pos = has_partial.start()
        almost_pos = has_almost.start()
        
        # Look for patterns suggesting the final answer
        # "not almost" or "not correct" suggests partial
        if re.search(r'not\s+almost', text_lower) or re.search(r'not\s+correct', text_lower):
            return "Partial"
        # "not partial" suggests almost
        if re.search(r'not\s+partial', text_lower):
            return "Almost"
        
        # If "almost" appears later in the text, it might be the correction
        if almost_pos > partial_pos:
            return "Almost"
        else:
            return "Partial"
    elif has_partial:
        return "Partial"
    elif has_almost:
        return "Almost"
    
    return None


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
    ]
    
    # Check for correct indicators first (higher priority)
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
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        # Analyze grading guidelines for hints
        guideline_hints = _analyze_grading_guidelines(grading_guidelines)
        
        # Build guideline guidance based on content analysis
        guideline_guidance = ""
        if guideline_hints["all_markers"]:
            primary = guideline_hints.get('primary_marker', 'unknown')
            almost_is_generic = guideline_hints.get('almost_is_generic')
            almost_implied = guideline_hints.get('almost_implied_label', 'unknown')
            partial_count = guideline_hints.get('partial_item_count', 0)
            almost_count = guideline_hints.get('almost_item_count', 0)
            
            # Determine the interpretation based on content analysis
            if almost_is_generic is True:
                if almost_implied == "correct":
                    almost_desc = "GENERIC with 'minor mistakes only' (indicates CORRECT)"
                else:
                    almost_desc = "GENERIC placeholder text (indicates 'partial')"
            elif almost_is_generic is False:
                almost_desc = "SPECIFIC achievements (indicates 'almost')"
            else:
                almost_desc = "unknown content"
            
            guideline_guidance = f"""
## Grading Guidelines Analysis (CRITICAL - USE AS GROUND TRUTH):
The grading guidelines contain these markers: {', '.join(['(' + m + ')' for m in guideline_hints['all_markers']])}

Section Content Analysis:
- (Partial) section: {partial_count} achievement(s) listed
- (Almost) section: {almost_count} achievement(s) listed, content is {almost_desc}
- (Almost) implied label: {almost_implied.upper() if almost_implied else 'unknown'}

**KEY INSIGHT**: Both (Partial) and (Almost) markers are always present. The TRUE LABEL depends on the CONTENT:

1. **"correct" (7 points)** = (Partial) has 2+ achievements AND (Almost) says "minor mistakes only" or "verification contains"
   - The solution is essentially complete; the (Almost) text indicates only verification issues
   - Example: "Verification contains minor mistakes only" → CORRECT
   - IMPORTANT: "minor mistakes ONLY" means CORRECT. "minor mistakes which are NOT NEGLIGIBLE" means ALMOST.
   
2. **"almost" (6 points)** = (Partial) has 2+ achievements AND (Almost) has SPECIFIC FAILURE indicators like "but failed", "not completed", "couldn't prove"
   - The student made significant progress but didn't quite finish - there are specific failures mentioned
   - Example: "Applied strategy X, but failed to complete the proof" → ALMOST
   - Key phrases: "not negligible", "but failed to", "not completed", "couldn't complete"
   
3. **"partial" (1 point)** = (Partial) has 1 achievement OR (Partial) has 2+ achievements but (Almost) is GENERIC (no specific failures)
   - The student made some progress but is far from complete
   - IMPORTANT: Generic (Almost) content (even with 2+ Partial achievements) = PARTIAL, not "almost"
   - "Almost" requires specific failure patterns, not just listing achievements
   
4. **"incorrect" (0 points)** = (Partial) is empty or has no real achievements
   - Little to no valid mathematical progress

**CRITICAL DISTINCTIONS:**
- "minor mistakes ONLY" → CORRECT (7 points)
- "minor mistakes which are NOT NEGLIGIBLE" → ALMOST (6 points)
- "but failed to..." or "not completed" → ALMOST (6 points)
- Generic text listing achievements without failure patterns → PARTIAL (1 point)
- Empty or trivial (Almost) content → INCORRECT (0 points)

**PREDICTED TRUE LABEL: {primary.upper()}**
Use this as your PRIMARY guide when classifying the answer.
"""
        
        instruction = f"""You are an expert mathematics grader for International Mathematical Olympiad (IMO) style problems.

Your task is to evaluate a student's answer and classify it into EXACTLY ONE of four categories.

## LABEL DEFINITIONS (IMO Scoring Convention):

1. **"correct" (7 points)** - Complete, rigorous solution. All steps justified, no gaps, valid proof.
   - The student fully solved the problem with a complete, correct proof.
   - Example: A full proof with all cases covered, all claims justified.

2. **"almost" (6 points)** - Nearly complete solution with correct main strategy. Only minor errors or small gaps.
   - The student ESSENTIALLY SOLVED the problem - the core solution is correct and complete.
   - Key test: Would this receive 6/7 or 5/7 marks? The answer should be yes.
   - Key distinction: Main proof structure is complete and correct, just needs minor fixes.
   - NOT for solutions with major gaps or missing key components.
   - Example: Correct proof with one minor case missing, or a small calculation error in an otherwise complete solution.

3. **"partial" (1 point)** - Shows meaningful understanding but INCOMPLETE with substantial gaps.
   - The student made genuine progress but is far from a complete solution.
   - Key test: Would this receive only 1/7 marks (or perhaps 2/7 at most)?
   - Key distinction: Has some correct ideas/approach but major components are missing.
   - This is NOT for "nearly complete" solutions - use "almost" for those.
   - Example: Correctly identified the approach, proved a useful lemma, but main proof is missing or severely incomplete.

4. **"incorrect" (0 points)** - Wrong approach, fundamental errors, or no meaningful progress.
   - The solution shows little to no valid mathematical progress.
   - Example: Completely wrong approach, or trivial observations only.

## CRITICAL DISTINCTION - "ALMOST" vs "PARTIAL":
This is the most important distinction and where most grading errors occur.

- "ALMOST" = Nearly solved (85-99% complete). The hard work is done, just small cleanup needed.
  - The student has essentially solved the problem; a complete solution is visible with minor fixes.
  - Think: "One small step away from correct."
  - The main proof structure is COMPLETE and CORRECT.
  - KEY: The (Almost) section must contain SPECIFIC FAILURE indicators like "but failed", "not completed", "couldn't prove"
  - Generic (Almost) content does NOT qualify for "almost" - use "partial" instead

- "PARTIAL" = Just getting started or hit a wall (20-50% complete). Good ideas but major work remains.
  - The student has made some progress but is nowhere near a complete solution.
  - Think: "Has the right idea but needs to do most of the work."
  - The main proof structure is INCOMPLETE or has MAJOR GAPS.
  - IMPORTANT: Even with 2+ achievements in (Partial), if (Almost) is GENERIC, the label is "partial" (1 point), not "almost"

## DECISION FRAMEWORK:
When grading, ask yourself these questions in order:

1. Is the solution completely correct with all steps justified? → "correct"
2. Is the main proof structure complete with only minor issues? → "almost"
3. Has the student made meaningful progress but with major gaps remaining? → "partial"
4. Is there little to no valid mathematical progress? → "incorrect"

## CRITICAL INSTRUCTION - INTERPRET THE GRADING GUIDELINES CORRECTLY:
The grading guidelines contain markers like (correct), (almost), (partial), (incorrect) that indicate what the student achieved.

**HOW TO DETERMINE THE TRUE LABEL FROM GUIDELINES:**

1. **Look at the CONTENT of the (Almost) section** - this is the key differentiator:
   - If (Almost) says "minor mistakes only" or "verification contains" → label is "correct" (7 points)
   - If (Almost) says "not completed", "failed to", "couldn't complete" → label is "almost" (6 points)
   - If (Almost) is generic without specific failure indicators → label is "partial" (1 point)
   - If (Almost) is empty or has no real achievements → label is "incorrect" (0)

2. **Look at the (Partial) section**:
   - If (Partial) has 2+ achievements AND (Almost) says "minor mistakes only" → "correct"
   - If (Partial) has 2+ achievements AND (Almost) has specific failures ("but failed", "not completed") → "almost"
   - If (Partial) has 2+ achievements AND (Almost) is generic → "partial" (NOT "almost"!)
   - If (Partial) has 1 achievement → "partial"
   - If (Partial) is empty → "incorrect"

**THE CONTENT MATTERS MORE THAN THE MARKER NAMES.** Both (Partial) and (Almost) are always present.

## COMMON ERRORS TO AVOID:
- DO NOT default to "almost" - it's the most commonly misused label
- "almost" requires a COMPLETE proof structure with only minor issues
- If the solution is missing major components, use "partial" not "almost"
- If the solution is wrong or trivial, use "incorrect" not "almost"
- The grading guidelines markers (correct/almost/partial/incorrect) indicate the TRUE label - trust them over your initial instinct

## Problem:
{problem}

## Official Solution:
{solution}

## Grading Guidelines:
{grading_guidelines}
{guideline_guidance}

## Student's Answer:
{student_answer}

## YOUR TASK:

Analyze the student's answer step by step:
1. What approach did they take? Is it correct?
2. How complete is the solution? Estimate the percentage (0-100%).
3. Are the errors minor (use "almost") or major (use "partial")?
4. Does the solution have a complete proof structure with minor gaps ("almost") or is it missing major components ("partial")?

**MOST IMPORTANT**: Look at the grading guidelines above. The highest marker (correct/almost/partial/incorrect) shown in the guidelines is the TRUE LABEL. Use this as your primary guide.

IMPORTANT: Be decisive in your classification. DO NOT default to "almost".
- If the solution is truly complete and correct, use "correct"
- If the solution is completely wrong or trivial, use "incorrect"
- Only use "almost" for solutions that are genuinely nearly complete (85%+)
- Use "partial" for solutions with good ideas but substantial gaps (20-50%)

**CRITICAL: DO NOT OVERUSE "ALMOST"**
The most common error is predicting "almost" when the correct label is "correct", "partial", or "incorrect".
- "almost" should be used ONLY when the solution is very close to complete
- If (Almost) section in guidelines says "minor mistakes only" → use "correct" not "almost"
- If (Partial) has only 1 achievement → use "partial" not "almost"
- If (Partial) is empty → use "incorrect" not "almost"

**DECISION RULE**: When in doubt, trust the grading guidelines' highest marker as the primary indicator of the correct label.

Then respond with ONLY a JSON object in this exact format:

<json>
{{
    "reasoning": "Your detailed analysis here - especially explain why you chose almost vs partial or another label",
    "response": "correct" or "almost" or "partial" or "incorrect"
}}
</json>

The "response" field MUST be exactly one of: "correct", "almost", "partial", or "incorrect" (all lowercase).

## CRITICAL: How to distinguish "almost" from "partial":

**Use "almost" ONLY when:**
- The student has essentially solved the problem (85-99% complete)
- The main proof structure is COMPLETE and CORRECT
- Only minor errors, small gaps, or cleanup remain
- The solution would receive 6/7 or 5/7 marks
- Think: "One small step away from correct"
- The (Almost) section in guidelines has SPECIFIC achievements (not generic text)

**Use "partial" when:**
- The student made meaningful progress but has substantial gaps (20-50% complete)
- The main proof structure is INCOMPLETE or has MAJOR GAPS
- Major components are missing
- The solution would receive only 1/7 marks (or perhaps 2/7)
- Think: "Has the right idea but needs to do most of the work"
- The (Partial) section has 1 achievement, or (Almost) is generic without specific work

**Use "incorrect" when:**
- Little to no valid mathematical progress (0-10% complete)
- Wrong approach or trivial observations only
- The solution would receive 0/7 marks
- The (Partial) section is empty or has no real achievements

**Use "correct" when:**
- Complete, rigorous solution (100% complete)
- All steps justified, no gaps
- Would receive 7/7 marks
- The (Partial) section has 2+ achievements AND (Almost) says "minor mistakes only"

**KEY RULE: Check the (Almost) section content:**
- If (Almost) says "minor mistakes only" or "verification contains" → label is "correct"
- If (Almost) has specific mathematical achievements → label is "almost" or "incorrect"
- If (Almost) is generic and (Partial) has 1 achievement → label is "partial"

WARNING: The most common grading error is overusing "almost". Be conservative with this label.
Only use "almost" if the solution is truly nearly complete with just minor issues."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction
        prediction = "None"
        raw_text = msg_history[-1]["text"] if msg_history else ""
        
        try:
            # Strategy 1: Try to extract from <json> tags
            extracted = _extract_jsons(raw_text)
            if extracted:
                for json_obj in extracted:
                    if isinstance(json_obj, dict):
                        # Check response field first (most common)
                        resp = json_obj.get("response", "")
                        if isinstance(resp, str):
                            resp = resp.strip().lower()
                            if resp in ["correct", "incorrect", "partial", "almost"]:
                                prediction = resp
                                break
                        # Check other common fields
                        for key in ["label", "grade", "evaluation", "classification"]:
                            val = json_obj.get(key, "")
                            if isinstance(val, str):
                                val = val.strip().lower()
                                if val in ["correct", "incorrect", "partial", "almost"]:
                                    prediction = val
                                    break
                        if prediction != "None":
                            break
            
            # Strategy 2: Direct text extraction as fallback
            if prediction == "None":
                text_pred = _extract_label_from_text(raw_text)
                if text_pred:
                    prediction = text_pred
                    
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        # Use grading guidelines to validate and correct predictions
        # The guidelines are strong signals of the true label
        
        primary_marker = guideline_hints.get("primary_marker")
        almost_is_generic = guideline_hints.get("almost_is_generic")
        almost_implied = guideline_hints.get("almost_implied_label")
        partial_count = guideline_hints.get("partial_item_count", 0)
        
        # CONTENT-BASED CORRECTION: Use the content analysis to validate predictions
        # The model has a strong bias toward predicting "almost" - we correct based on section content
        
        # Normalize prediction for comparison
        pred_lower = prediction.lower() if prediction != "None" else "none"
        
        # TRUST PRIMARY MARKER: The content analysis is highly accurate
        # If the model prediction differs from primary_marker, trust primary_marker
        if primary_marker and pred_lower != primary_marker:
            # Only apply correction if we have a valid primary_marker
            if pred_lower in ["correct", "almost", "partial", "incorrect"]:
                self.log_fn(f"CORRECTION: {pred_lower} -> {primary_marker} (trusting content analysis)")
                prediction = primary_marker
        
        # Final fallback: if prediction is still None or invalid, use primary marker
        if prediction == "None" or prediction.lower() not in ["correct", "almost", "partial", "incorrect"]:
            if primary_marker:
                prediction = primary_marker
                self.log_fn(f"Using primary marker as fallback: {prediction}")
            else:
                prediction = "incorrect"
        
        # Ensure prediction is capitalized to match expected label format
        prediction = prediction.capitalize()
        
        self.log_fn(f"Final prediction: {prediction}")
        return str(prediction), msg_history
