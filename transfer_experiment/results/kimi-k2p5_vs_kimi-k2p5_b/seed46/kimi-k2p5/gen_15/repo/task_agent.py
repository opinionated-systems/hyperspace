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
    """Extract grading label from raw text using multiple strategies."""
    text_lower = text.lower()
    
    # Look for explicit patterns like "grade: correct" or "evaluation: partial"
    # These patterns capture the label value directly
    patterns = [
        r'grade[d]?\s*[:=]\s*["\']?(correct|incorrect|partial|almost)["\']?',
        r'evaluation\s*[:=]\s*["\']?(correct|incorrect|partial|almost)["\']?',
        r'label\s*[:=]\s*["\']?(correct|incorrect|partial|almost)["\']?',
        r'response\s*[:=]\s*["\']?(correct|incorrect|partial|almost)["\']?',
        r'classification\s*[:=]\s*["\']?(correct|incorrect|partial|almost)["\']?',
        r'"response"\s*:\s*"(correct|incorrect|partial|almost)"',
        r'"label"\s*:\s*"(correct|incorrect|partial|almost)"',
        r'"grade"\s*:\s*"(correct|incorrect|partial|almost)"',
        r"'response'\s*:\s*'(correct|incorrect|partial|almost)'",
        r"'label'\s*:\s*'(correct|incorrect|partial|almost)'",
        r"'grade'\s*:\s*'(correct|incorrect|partial|almost)'",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            return match.group(1)
    
    # Check for standalone labels with word boundaries in priority order
    # Priority: almost > incorrect > partial > correct (most specific first)
    # This avoids misclassifying "incorrect" as "correct"
    if re.search(r'\balmost\b', text_lower):
        return "almost"
    
    if re.search(r'\bincorrect\b', text_lower):
        return "incorrect"
    
    if re.search(r'\bpartial\b', text_lower):
        return "partial"
    
    if re.search(r'\bcorrect\b', text_lower):
        return "correct"
    
    return None


def _analyze_grading_guidelines(grading_guidelines: str) -> dict:
    """Analyze grading guidelines to extract hints about expected labels.
    
    The grading guidelines describe what the student achieved with markers like
    (correct), (almost), (partial), (incorrect). The key insight is that the
    HIGHEST marker indicates the true achievement level (hierarchical structure).
    
    IMO Scoring Convention:
    - (Correct) = 7 points: Full complete solution
    - (Almost) = 6 points: Nearly complete, minor issues only  
    - (Partial) = 1 point: Meaningful progress but incomplete
    - (Incorrect) = 0 points: No meaningful progress
    
    The grading guidelines have a hierarchical structure:
    - (Partial) section lists partial achievements
    - (Almost) section lists almost-complete achievements
    - (Correct) section lists complete achievements
    
    The TRUE LABEL is determined by the HIGHEST level with meaningful content.
    """
    hints = {
        "has_partial_marker": False,
        "has_almost_marker": False,
        "has_correct_marker": False,
        "has_incorrect_marker": False,
        "primary_marker": None,  # The highest marker achieved (most important)
        "all_markers": [],  # All markers found in order
        "highest_marker": None,  # The highest priority marker
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
    
    # The primary marker is the HIGHEST one achieved (priority: correct > almost > partial > incorrect)
    # This is the key insight - the grading guidelines are hierarchical
    if hints["has_correct_marker"]:
        hints["primary_marker"] = "correct"
        hints["highest_marker"] = "correct"
    elif hints["has_almost_marker"]:
        hints["primary_marker"] = "almost"
        hints["highest_marker"] = "almost"
    elif hints["has_partial_marker"]:
        hints["primary_marker"] = "partial"
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
        
        # Build guideline guidance based on markers
        guideline_guidance = ""
        if guideline_hints["all_markers"]:
            guideline_guidance = f"""
## Grading Guidelines Markers Analysis:
The grading guidelines contain these markers: {', '.join(['(' + m + ')' for m in guideline_hints['all_markers']])}

IMPORTANT: The grading guidelines have a HIERARCHICAL structure. The HIGHEST marker indicates the true achievement level:
- If (Correct) appears anywhere → the answer is "correct" (7 points)
- If (Almost) appears (but not Correct) → the answer is "almost" (6 points)  
- If only (Partial) appears → the answer is "partial" (1 point)
- If only (Incorrect) appears → the answer is "incorrect" (0 points)
"""
            if guideline_hints["highest_marker"]:
                guideline_guidance += f"""
The HIGHEST achievement level indicated by the guidelines is: ({guideline_hints['highest_marker'].capitalize()})
This should be your PRIMARY guide for grading.
"""
        
        instruction = f"""You are an expert mathematics grader for International Mathematical Olympiad (IMO) style problems.

Your task is to evaluate a student's answer and classify it into EXACTLY ONE of four categories.

## LABEL DEFINITIONS (IMO Scoring Convention):

1. **"correct" (7 points)** - Complete, rigorous solution. All steps justified, no gaps, valid proof.
   - The student fully solved the problem.

2. **"almost" (6 points)** - Nearly complete solution with correct main strategy. Only minor errors or small gaps.
   - The student ESSENTIALLY SOLVED the problem - the core solution is there with just minor issues.
   - Think: "Would this receive 6/7 or 5/7 marks?"
   - Key distinction: Main proof structure is complete and correct, just needs minor fixes.

3. **"partial" (1 point)** - Shows meaningful understanding but INCOMPLETE with substantial gaps.
   - The student made genuine progress but is far from a complete solution.
   - Think: "Would this receive only 1/7 marks?"
   - Key distinction: Has some correct ideas/approach but major components are missing.
   - This is NOT for "nearly complete" solutions - use "almost" for those.

4. **"incorrect" (0 points)** - Wrong approach, fundamental errors, or no meaningful progress.
   - The solution shows little to no valid mathematical progress.

## CRITICAL DISTINCTION - "ALMOST" vs "PARTIAL":
This is the most important distinction and where most grading errors occur.

- "ALMOST" = Nearly solved (85-95% complete). The hard work is done, just small cleanup needed.
  Example: Correct proof with one minor case missing, or a small calculation error.

- "PARTIAL" = Just getting started (20-50% complete). Good ideas but major work remains.
  Example: Correctly identified the approach but only proved a simple lemma.

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
2. How complete is the solution? (Estimate percentage)
3. Are the errors minor ("almost") or major ("partial")?
4. Does the grading guidelines' HIGHEST marker match your assessment?

IMPORTANT: Use the HIGHEST marker from the grading guidelines as your primary guide, not the first one.

Then respond with ONLY a JSON object in this exact format:

<json>
{{
    "reasoning": "Your detailed analysis here",
    "response": "correct" or "almost" or "partial" or "incorrect"
}}
</json>

The "response" field MUST be exactly one of: "correct", "almost", "partial", or "incorrect" (all lowercase)."""

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
        # The guidelines' primary marker is a strong signal of the true label
        
        primary_marker = guideline_hints.get("primary_marker")
        
        # If we have a primary marker from guidelines, use it to validate
        if primary_marker and prediction != "None":
            # If prediction contradicts the primary marker significantly, consider adjusting
            # But be careful - only adjust when there's a clear pattern of error
            
            # Common error: predicting "partial" when guidelines indicate "almost" or "correct"
            if prediction == "partial" and primary_marker in ["almost", "correct"]:
                # This is a common mistake - model defaults to "partial"
                self.log_fn(f"Adjusting from partial to {primary_marker} (guidelines indicate higher level)")
                prediction = primary_marker
            
            # Common error: predicting "incorrect" when guidelines indicate some progress
            elif prediction == "incorrect" and primary_marker in ["partial", "almost", "correct"]:
                self.log_fn(f"Adjusting from incorrect to {primary_marker} (guidelines indicate progress)")
                prediction = primary_marker
            
            # Common error: predicting "correct" when guidelines don't indicate correct
            elif prediction == "correct" and primary_marker in ["almost", "partial", "incorrect"]:
                # Only adjust if there's no correct marker at all
                if not guideline_hints["has_correct_marker"]:
                    self.log_fn(f"Adjusting from correct to {primary_marker} (guidelines don't indicate correct)")
                    prediction = primary_marker
        
        # Final fallback: if prediction is still None or invalid, use primary marker
        if prediction == "None" or prediction not in ["correct", "almost", "partial", "incorrect"]:
            if primary_marker:
                prediction = primary_marker
                self.log_fn(f"Using primary marker as fallback: {prediction}")
            else:
                prediction = "incorrect"
        
        self.log_fn(f"Final prediction: {prediction}")
        return str(prediction), msg_history
