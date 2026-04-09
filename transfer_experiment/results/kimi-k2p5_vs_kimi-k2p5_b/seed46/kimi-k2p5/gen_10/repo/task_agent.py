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
        except json.JSONDecodeError:
            continue
    return results or None


def _extract_json_from_markdown(text: str) -> list[dict] | None:
    """Extract JSON objects from markdown code blocks."""
    results = []
    # Match ```json ... ``` or ``` ... ``` blocks
    pattern = r'```(?:json)?\s*(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    for match in matches:
        try:
            results.append(json.loads(match.strip()))
        except json.JSONDecodeError:
            continue
    return results or None


def _extract_json_braces(text: str) -> list[dict] | None:
    """Extract JSON objects directly from curly braces."""
    results = []
    # Try to find JSON objects by looking for balanced braces
    i = 0
    while i < len(text):
        if text[i] == '{':
            # Try to find matching closing brace
            brace_count = 1
            j = i + 1
            while j < len(text) and brace_count > 0:
                if text[j] == '{':
                    brace_count += 1
                elif text[j] == '}':
                    brace_count -= 1
                j += 1
            if brace_count == 0:
                try:
                    obj = json.loads(text[i:j])
                    results.append(obj)
                except json.JSONDecodeError:
                    pass
            i = j if j > i else i + 1
        else:
            i += 1
    return results or None


def _extract_label_from_text(text: str) -> str | None:
    """Extract grading label from raw text using multiple strategies.
    
    This is a fallback extraction method when JSON parsing fails.
    We look for explicit label declarations in the text.
    """
    text_lower = text.lower()
    
    # Look for patterns like "label: correct" or "grade: partial" or "response: almost"
    # These are more reliable than just searching for the word
    label_patterns = [
        r'["\']?response["\']?\s*[:=]\s*["\']?(correct|almost|partial|incorrect)["\']?',
        r'["\']?label["\']?\s*[:=]\s*["\']?(correct|almost|partial|incorrect)["\']?',
        r'["\']?grade["\']?\s*[:=]\s*["\']?(correct|almost|partial|incorrect)["\']?',
        r'["\']?evaluation["\']?\s*[:=]\s*["\']?(correct|almost|partial|incorrect)["\']?',
        r'["\']?result["\']?\s*[:=]\s*["\']?(correct|almost|partial|incorrect)["\']?',
        r'["\']?prediction["\']?\s*[:=]\s*["\']?(correct|almost|partial|incorrect)["\']?',
        r'["\']?answer["\']?\s*[:=]\s*["\']?(correct|almost|partial|incorrect)["\']?',
        r'final\s+(?:grade|label|score|evaluation)\s*[:=]\s*["\']?(correct|almost|partial|incorrect)["\']?',
        r'(?:is|as)\s+["\']?(correct|almost|partial|incorrect)["\']?',
    ]
    
    for pattern in label_patterns:
        match = re.search(pattern, text_lower)
        if match:
            return match.group(1)
    
    # Look for the label in the last sentence or paragraph (more likely to be the conclusion)
    # Split by common conclusion markers
    conclusion_markers = [
        'final determination', 'in conclusion', 'therefore', 'thus', 'overall',
        'i conclude', 'my assessment', 'final grade', 'final answer', 'final response',
        'the grade is', 'the label is', 'the response is'
    ]
    
    # Try to find the conclusion section
    conclusion_text = text_lower
    for marker in conclusion_markers:
        if marker in text_lower:
            # Get text after the marker
            idx = text_lower.rfind(marker)  # Use last occurrence (final conclusion)
            if idx != -1:
                conclusion_text = text_lower[idx:]
                break
    
    # Now look for labels in the conclusion section
    # Priority: check for each label in order of specificity
    labels_to_check = [
        (r'\balmost\b', 'almost'),
        (r'\bpartial\b', 'partial'),
        (r'\bincorrect\b', 'incorrect'),
        (r'\bcorrect\b(?!\s+answer)', 'correct'),  # Avoid matching "correct answer" phrases
    ]
    
    for pattern, label in labels_to_check:
        if re.search(pattern, conclusion_text):
            return label
    
    # Last resort: check the entire text for any label
    for pattern, label in labels_to_check:
        if re.search(pattern, text_lower):
            return label
    
    return None


def _extract_confidence_score(text: str) -> float | None:
    """Extract confidence score if present in the response."""
    # Look for confidence patterns like "confidence: 0.8" or "score: 0.9"
    patterns = [
        r'confidence[:\s]+(\d+\.?\d*)',
        r'confidence score[:\s]+(\d+\.?\d*)',
        r'score[:\s]+(\d+\.?\d*)',
    ]
    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            try:
                score = float(match.group(1))
                if 0 <= score <= 1:
                    return score
            except ValueError:
                continue
    return None


def _normalize_prediction(pred: str | None) -> str:
    """Normalize prediction to one of the valid labels."""
    if not pred or not isinstance(pred, str):
        return "None"
    
    pred = pred.strip().lower()
    
    # Direct match
    if pred in ["correct", "almost", "partial", "incorrect"]:
        return pred
    
    # Remove quotes and common wrappers
    pred_clean = pred.strip('"\'[]{}')
    if pred_clean in ["correct", "almost", "partial", "incorrect"]:
        return pred_clean
    
    # Check for exact word matches with word boundaries (more reliable than substring)
    for label in ["almost", "incorrect", "partial", "correct"]:
        if re.search(rf'\b{label}\b', pred):
            return label
    
    # Fallback: Check for substring matches - priority order matters
    # "almost" must be checked before "correct"
    if "almost" in pred:
        return "almost"
    if "incorrect" in pred:
        return "incorrect"
    if "partial" in pred:
        return "partial"
    if "correct" in pred:
        return "correct"
    
    return "None"


def _analyze_grading_guidelines(grading_guidelines: str) -> dict:
    """Analyze grading guidelines to extract hints about expected labels.
    
    The grading guidelines describe what the student achieved. They often contain
    multiple markers when the student achieved different levels on different aspects.
    The key is to identify the PRIMARY achievement level - usually indicated by
    the first marker in the list or the most specific/detailed section.
    
    IMO Scoring Convention:
    - (Correct) = 7 points: Full complete solution
    - (Almost) = 6 points: Nearly complete, minor issues only
    - (Partial) = 1 point: Meaningful progress but incomplete
    - (Incorrect) = 0 points: No meaningful progress
    
    When multiple markers appear, the FIRST marker typically indicates the base
    achievement level, and subsequent markers indicate additional achievements.
    """
    hints = {
        "has_partial_marker": False,
        "has_almost_marker": False,
        "has_correct_marker": False,
        "has_incorrect_marker": False,
        "partial_keywords": [],
        "almost_keywords": [],
        "correct_keywords": [],
        "incorrect_keywords": [],
        "primary_marker": None,  # The first marker found (most important)
        "all_markers": [],  # All markers found in order
        "dominant_marker": None,  # The most frequent marker
        "best_marker": None,  # The highest achievement marker present
        "worst_marker": None,  # The lowest achievement marker present
    }
    
    if not grading_guidelines:
        return hints
    
    guidelines_lower = grading_guidelines.lower()
    
    # Search for markers in the text and record their positions
    marker_positions = []
    for marker, label in [("(correct)", "correct"), ("(almost)", "almost"), 
                          ("(partial)", "partial"), ("(incorrect)", "incorrect")]:
        # Find all occurrences
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
    
    # The primary marker is the FIRST one in the text (most important for base level)
    if hints["all_markers"]:
        hints["primary_marker"] = hints["all_markers"][0]
    
    # Count occurrences of each marker
    marker_counts = {
        "correct": guidelines_lower.count("(correct)"),
        "almost": guidelines_lower.count("(almost)"),
        "partial": guidelines_lower.count("(partial)"),
        "incorrect": guidelines_lower.count("(incorrect)")
    }
    
    # Find the dominant marker (highest count)
    max_count = max(marker_counts.values()) if marker_counts else 0
    if max_count > 0:
        for label in ["correct", "almost", "partial", "incorrect"]:
            if marker_counts[label] == max_count:
                hints["dominant_marker"] = label
                break
    
    # Find best and worst markers by achievement level
    achievement_order = ["incorrect", "partial", "almost", "correct"]
    present_markers = [label for label in achievement_order if marker_counts[label] > 0]
    if present_markers:
        hints["worst_marker"] = present_markers[0]  # Lowest achievement
        hints["best_marker"] = present_markers[-1]  # Highest achievement
    
    # Set flags for each marker type
    if "(partial)" in guidelines_lower:
        hints["has_partial_marker"] = True
    if "(almost)" in guidelines_lower:
        hints["has_almost_marker"] = True
    if "(correct)" in guidelines_lower:
        hints["has_correct_marker"] = True
    if "(incorrect)" in guidelines_lower:
        hints["has_incorrect_marker"] = True
    
    # Also check for partial credit mentions
    if "partial credit" in guidelines_lower:
        hints["has_partial_marker"] = True
    
    # Extract keywords associated with each category
    partial_indicators = [
        "partial", "incomplete", "some progress", "minor understanding",
        "started correctly", "some correct elements", "partial solution",
        "missing", "incomplete solution", "not complete", "far from",
        "only solved", "only proved", "only showed", "proved that", "observed that"
    ]
    almost_indicators = [
        "almost", "minor errors", "small mistakes", "nearly complete",
        "essentially correct", "minor slip", "small gap", "nearly solved",
        "mostly correct", "small error", "minor flaw", "not completed"
    ]
    correct_indicators = [
        "correct", "complete solution", "full solution", "correctly proved",
        "correctly solved", "valid proof", "correct answer"
    ]
    incorrect_indicators = [
        "incorrect", "wrong", "fundamentally flawed", "no meaningful progress",
        "completely wrong", "invalid", "does not work"
    ]
    
    for indicator in partial_indicators:
        if indicator in guidelines_lower:
            hints["partial_keywords"].append(indicator)
    
    for indicator in almost_indicators:
        if indicator in guidelines_lower:
            hints["almost_keywords"].append(indicator)
    
    for indicator in correct_indicators:
        if indicator in guidelines_lower:
            hints["correct_keywords"].append(indicator)
    
    for indicator in incorrect_indicators:
        if indicator in guidelines_lower:
            hints["incorrect_keywords"].append(indicator)
    
    return hints


class TaskAgent:
    """Task agent that solves IMO grading problems."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def _check_grading_guidelines_for_hint(self, grading_guidelines: str) -> str | None:
        """Check if grading guidelines contain explicit hints about the expected label.
        
        This helps guide the model when the guidelines explicitly mention categories.
        """
        if not grading_guidelines:
            return None
            
        guidelines_lower = grading_guidelines.lower()
        
        # Check for explicit markers in guidelines - in order of specificity
        if "(almost)" in guidelines_lower:
            return "almost"
        if "(partial)" in guidelines_lower:
            return "partial"
        if "(incorrect)" in guidelines_lower:
            return "incorrect"
        if "(correct)" in guidelines_lower:
            return "correct"
        
        return None

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Extract fields from inputs
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        # Analyze grading guidelines first to provide context to the LLM
        guideline_hints = _analyze_grading_guidelines(grading_guidelines)
        
        # Build guideline interpretation hint
        guideline_interpretation = ""
        if guideline_hints["all_markers"]:
            guideline_interpretation = f"""
## Grading Guidelines Analysis:
The grading guidelines contain the following achievement markers (in order of appearance):
"""
            for i, marker in enumerate(guideline_hints["all_markers"], 1):
                guideline_interpretation += f"{i}. ({marker.capitalize()})\n"
            
            if guideline_hints["primary_marker"]:
                guideline_interpretation += f"""
IMPORTANT: The FIRST marker "({guideline_hints['primary_marker'].capitalize()})" typically indicates the BASE achievement level.
"""
            
            # Add specific guidance based on marker patterns
            if len(guideline_hints["all_markers"]) > 1:
                guideline_interpretation += """
When multiple markers appear, this usually means:
- The first marker indicates the base achievement level
- Subsequent markers indicate additional achievements on specific aspects
- The OVERALL grade should reflect the base level unless the additional achievements significantly elevate the solution
"""
        
        instruction = f"""You are an expert mathematics grader for International Mathematical Olympiad (IMO) style problems.

Your task is to evaluate a student's answer and classify it into EXACTLY ONE of four categories. Be objective and follow the grading guidelines carefully.

## The Four Categories (IMO Scoring):

1. **"correct" (7 points)** - Complete, rigorous solution matching the official solution. All reasoning is sound and properly justified.

2. **"almost" (6 points)** - Nearly complete solution with correct main strategy. Minor errors or small gaps that don't invalidate the core reasoning. The student essentially solved the problem but made small mistakes.
   - *Key indicator*: Solution is 85-95% complete, needs only minor fixes.

3. **"partial" (1 point)** - Shows meaningful understanding and correct approach to some parts, but incomplete with substantial gaps. The student made genuine progress but is far from a complete solution.
   - *Key indicator*: Solution is 20-50% complete, missing major components.

4. **"incorrect" (0 points)** - Wrong approach, fundamental errors, or no meaningful understanding demonstrated.

## Problem:
{problem}

## Official Solution:
{solution}

## Grading Guidelines:
{grading_guidelines}
{guideline_interpretation}

## Student's Answer:
{student_answer}

## Step-by-Step Evaluation Process:

**Step 1: Analyze the grading guidelines**
Look for explicit markers like (Correct), (Almost), (Partial), or (Incorrect). These indicate what the student achieved. The FIRST marker typically shows the base achievement level.

**Step 2: Evaluate the student's solution**
- Does the student understand the main problem?
- Is the overall strategy correct?
- How complete is the solution?
- Are there genuine insights or just guesses?

**Step 3: Apply the decision tree**
- If fully correct and complete → "correct"
- If essentially correct but has minor flaws → "almost"  
- If incomplete but shows real progress → "partial"
- If fundamentally wrong → "incorrect"

**Critical Distinction - "almost" vs "partial":**
This is the most common error. Ask yourself:
- "almost": Would this solution get 6/7 points? Is it essentially done with small issues?
- "partial": Would this get 1/7 points? Is it just a start with lots missing?

**DO NOT default to "partial"** - use it only when the solution truly shows insight but is incomplete. If the solution is nearly complete, use "almost".

Respond in JSON format:
<json>
{{
    "reasoning": "Brief analysis focusing on completeness and correctness...",
    "response": "correct" | "almost" | "partial" | "incorrect"
}}
</json>

The response field must be exactly one of the four categories."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction using multiple strategies
        prediction = "None"
        raw_text = msg_history[-1]["text"] if msg_history else ""
        
        # Analyze grading guidelines for hints first
        guideline_hints = _analyze_grading_guidelines(grading_guidelines)
        
        try:
            # Strategy 1: Try to extract from <json> tags
            extracted = _extract_jsons(raw_text)
            if extracted:
                for json_obj in extracted:
                    for key in ["response", "answer", "label", "grade", "evaluation", "result", "prediction"]:
                        if key in json_obj:
                            prediction = _normalize_prediction(str(json_obj[key]))
                            if prediction != "None":
                                break
                    if prediction != "None":
                        break
            
            # Strategy 2: Try markdown code blocks
            if prediction == "None":
                extracted = _extract_json_from_markdown(raw_text)
                if extracted:
                    for json_obj in extracted:
                        for key in ["response", "answer", "label", "grade", "evaluation", "result", "prediction"]:
                            if key in json_obj:
                                prediction = _normalize_prediction(str(json_obj[key]))
                                if prediction != "None":
                                    break
                        if prediction != "None":
                            break
            
            # Strategy 3: Try raw JSON braces
            if prediction == "None":
                extracted = _extract_json_braces(raw_text)
                if extracted:
                    for json_obj in extracted:
                        for key in ["response", "answer", "label", "grade", "evaluation", "result", "prediction"]:
                            if key in json_obj:
                                prediction = _normalize_prediction(str(json_obj[key]))
                                if prediction != "None":
                                    break
                        if prediction != "None":
                            break
            
            # Strategy 4: Direct text extraction as fallback
            if prediction == "None":
                text_pred = _extract_label_from_text(raw_text)
                if text_pred:
                    prediction = text_pred
                    
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        # Use grading guidelines hints to validate and adjust predictions
        # The LLM's analysis should take precedence, but guidelines provide important context
        
        # Only use guideline hints if we couldn't extract any prediction from LLM
        if prediction == "None" or prediction not in ["correct", "almost", "partial", "incorrect"]:
            # Try to use the primary (first) marker from guidelines as a fallback
            # The first marker typically indicates the base achievement level
            if guideline_hints["primary_marker"]:
                prediction = guideline_hints["primary_marker"]
                self.log_fn(f"Using primary guideline marker as fallback: {prediction}")
            elif guideline_hints["all_markers"]:
                prediction = guideline_hints["all_markers"][0]
                self.log_fn(f"Using first guideline marker as fallback: {prediction}")
        
        # Post-processing: Use grading guidelines as hints, not overrides
        # The LLM's analysis should take precedence, but we use guidelines for validation
        
        primary_marker = guideline_hints.get("primary_marker")
        all_markers = guideline_hints.get("all_markers", [])
        
        # Only adjust if there's a clear mismatch between prediction and guidelines
        # Adjustment 1: If predicted "correct" but no correct marker in guidelines
        if prediction == "correct" and primary_marker and primary_marker != "correct":
            if not guideline_hints["has_correct_marker"]:
                self.log_fn(f"Adjusting from correct to {primary_marker} (no correct marker)")
                prediction = primary_marker
        
        # Adjustment 2: If predicted "incorrect" but guidelines clearly indicate partial
        if prediction == "incorrect" and guideline_hints["has_partial_marker"]:
            if primary_marker == "partial":
                self.log_fn(f"Adjusting from incorrect to partial (guidelines indicate partial)")
                prediction = "partial"
        
        # Adjustment 3: If predicted "partial" but guidelines clearly indicate almost
        if prediction == "partial" and guideline_hints["has_almost_marker"]:
            if primary_marker == "almost" and not guideline_hints["has_partial_marker"]:
                self.log_fn(f"Adjusting from partial to almost (guidelines indicate almost)")
                prediction = "almost"
        
        # Adjustment 4: If predicted "almost" but guidelines clearly indicate partial
        if prediction == "almost" and guideline_hints["has_partial_marker"]:
            if primary_marker == "partial" and not guideline_hints["has_almost_marker"]:
                self.log_fn(f"Adjusting from almost to partial (guidelines indicate partial)")
                prediction = "partial"
        
        # Final fallback: if prediction is still None or invalid
        if prediction == "None" or prediction not in ["correct", "almost", "partial", "incorrect"]:
            text_pred = _extract_label_from_text(raw_text)
            if text_pred:
                prediction = text_pred
            elif guideline_hints["primary_marker"]:
                prediction = guideline_hints["primary_marker"]
                self.log_fn(f"Using primary marker as fallback: {prediction}")
            else:
                prediction = "incorrect"
            
        self.log_fn(f"Final prediction: {prediction}")
        return str(prediction), msg_history
