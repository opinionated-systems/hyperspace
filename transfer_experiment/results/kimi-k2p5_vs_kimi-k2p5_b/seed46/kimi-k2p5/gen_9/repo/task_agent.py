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

Your task is to evaluate a student's answer to a mathematical problem and classify it into EXACTLY ONE of these four categories, which correspond to IMO scoring:

1. "correct" (7 points) - The student's answer is fully correct, complete, and matches the official solution. All reasoning is sound, all claims are properly justified, and the final answer is correct. This is a perfect or near-perfect solution.

2. "almost" (6 points) - The answer demonstrates a complete or nearly complete understanding of the solution approach, with the main ideas and strategy being correct. However, there are minor verification mistakes, small computational errors, or slight gaps in justification that don't invalidate the core reasoning. The student has essentially solved the problem but made minor mistakes in execution. The solution is substantially complete with only small issues.

3. "partial" (1 point) - The answer has some correct elements and shows understanding of key concepts, but is incomplete, missing significant components, or has substantial gaps in the reasoning. The student has made meaningful progress but hasn't reached a complete or nearly complete solution. This is for answers that show genuine insight but are far from complete.

4. "incorrect" (0 points) - The answer is wrong, fundamentally flawed, or shows no meaningful understanding of the problem. The approach is misguided or the reasoning contains critical errors.

## Problem:
{problem}

## Official Solution:
{solution}

## Grading Guidelines:
{grading_guidelines}
{guideline_interpretation}

## Student's Answer:
{student_answer}

## Instructions:
1. Carefully read the problem, official solution, and grading guidelines
2. Analyze the student's answer step by step
3. Compare the student's reasoning and conclusion with the official solution
4. Make a final determination using EXACTLY one of: "correct", "almost", "partial", or "incorrect"

## CRITICAL DISTINCTIONS - READ CAREFULLY:

**"correct" (7 points) vs "almost" (6 points):**
- "correct" = everything is right, complete, and properly justified
- "almost" = main solution strategy is correct but there are minor execution errors or small gaps
- Key test: Would this solution receive full marks or lose just 1 point?

**"almost" (6 points) vs "partial" (1 point) - THIS IS THE MOST IMPORTANT DISTINCTION:**
- "almost" = The solution is NEARLY COMPLETE. The core argument is essentially correct. The student has solved the main problem but has minor issues. Think: "This is essentially a correct solution with small flaws."
- "partial" = The solution is INCOMPLETE or has SUBSTANTIAL GAPS. The student made progress but significant work remains. Think: "This shows understanding but is far from a complete solution."
- Key test: Is the solution essentially complete (just needs minor fixes) or does it need significant additional work?
- "almost" = 85-95% complete; "partial" = 20-50% complete
- **CRITICAL**: "almost" means the student essentially solved the problem. "partial" means they only made progress toward a solution.
- **DO NOT overgrade incomplete solutions as "almost"** - if the solution is missing major components, it should be "partial"

**"partial" (1 point) vs "incorrect" (0 points):**
- "partial" = shows meaningful understanding of the problem and correct approach to some parts
- "incorrect" = fundamental misunderstanding or completely wrong approach
- Key test: Did the student demonstrate any genuine insight into the problem?

## Decision Framework:
When deciding between categories, ask:
1. Is the main solution strategy correct? (If no → "incorrect"; If yes → continue)
2. Is the solution essentially complete with only minor issues? (If yes → "almost"; If no → continue)
3. Does it show meaningful understanding and correct approach to some parts? (If yes → "partial"; If no → "incorrect")

## Examples:
- "correct": Complete, rigorous proof matching the official solution
- "almost": Student found correct invariant but made minor calculation error in verification; or proved main theorem but missed a small case that doesn't affect core argument
- "partial": Student identified correct approach but only solved 1 of 3 required cases; or has correct initial steps but solution is incomplete with significant gaps remaining
- "incorrect": Student used completely wrong approach or made fundamental logical errors

## Detailed Examples of "almost" vs "partial":
**"almost" examples:**
- Student proved that c >= 3 and applied infinite descent to prove c = 3, but didn't quite complete the final step
- Student found the correct answer and proved most cases, but made a minor error in one case that doesn't affect the main result
- Student's solution is essentially complete but has a small gap in justification that could be easily fixed

**"partial" examples:**
- Student proved that c >= 3 but didn't make progress on proving c = 3
- Student correctly guessed all possible answers and proved bijectivity, but didn't complete the main proof
- Student proved that points lie on a circle and observed a key property, but didn't complete the proof
- Student made meaningful progress but the solution is far from complete

## IMPORTANT NOTES:
- The grading guidelines may contain markers like "(Partial)", "(Almost)", "(Correct)", or "(Incorrect)" - these describe what the student achieved
- **The FIRST marker in the guidelines typically indicates the BASE achievement level** - pay special attention to this
- Your job is to determine the OVERALL grade based on the complete analysis of the student's answer
- "partial" should be used for answers that show genuine insight but are incomplete - do not overuse "almost" for incomplete solutions
- When in doubt between "almost" and "partial", ask: "Is this 90% done (almost) or 30% done (partial)?"
- When in doubt between "partial" and "incorrect", ask: "Did the student prove anything meaningful or make any genuine progress?"
- Be honest and objective - don't inflate grades
- **Pay special attention to the grading guidelines** - they often contain explicit markers that indicate the expected grade

Respond in JSON format with the following schema:
<json>
{{
    "reasoning": "Your detailed analysis of the student's answer...",
    "response": "correct" | "almost" | "partial" | "incorrect"
}}
</json>

The "response" field MUST be exactly one of: "correct", "almost", "partial", or "incorrect" (lowercase, no extra text)."""

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
        
        # Post-processing adjustment based on grading guidelines
        # These adjustments are CONSERVATIVE - we only adjust when there's strong evidence
        
        # Get the primary marker (first one in guidelines) - this is the most important signal
        primary_marker = guideline_hints.get("primary_marker")
        all_markers = guideline_hints.get("all_markers", [])
        
        # Adjustment 1: If LLM predicted "correct" but primary marker is lower, be skeptical
        if prediction == "correct" and primary_marker and primary_marker != "correct":
            # Only allow "correct" if there's strong evidence (correct marker present)
            if not guideline_hints["has_correct_marker"]:
                # No correct marker in guidelines, downgrade to primary marker level
                self.log_fn(f"Adjusting from correct to {primary_marker} (no correct marker in guidelines)")
                prediction = primary_marker
        
        # Adjustment 2: If LLM predicted "almost" but primary marker is "partial", downgrade
        if prediction == "almost" and primary_marker == "partial":
            # The base achievement is partial - only upgrade to almost if there's strong evidence
            # Check if almost appears before partial (indicating it's the primary)
            if len(all_markers) > 0 and all_markers[0] != "almost":
                # Partial is the primary marker, downgrade
                self.log_fn(f"Adjusting from almost to partial (primary marker is partial)")
                prediction = "partial"
        
        # Adjustment 3: If LLM predicted "partial" but primary marker is "incorrect", check carefully
        if prediction == "partial" and primary_marker == "incorrect":
            # Guidelines say incorrect as primary - only keep partial if there's strong evidence
            # of meaningful progress in the student answer itself
            if not guideline_hints["has_partial_marker"] and guideline_hints["has_incorrect_marker"]:
                # No partial marker, has incorrect marker - downgrade
                self.log_fn(f"Adjusting from partial to incorrect (primary marker is incorrect)")
                prediction = "incorrect"
        
        # Adjustment 4: Handle "almost" vs "partial" confusion based on keyword evidence
        if prediction in ["almost", "partial"]:
            partial_count = len(guideline_hints["partial_keywords"])
            almost_count = len(guideline_hints["almost_keywords"])
            
            has_partial_marker = guideline_hints["has_partial_marker"]
            has_almost_marker = guideline_hints["has_almost_marker"]
            
            # Strong adjustment: if guidelines clearly indicate one category with no other markers
            if prediction == "almost" and has_partial_marker and not has_almost_marker:
                # Guidelines only say partial, no almost marker - likely overgrading
                self.log_fn(f"Adjusting from almost to partial (only partial marker in guidelines)")
                prediction = "partial"
            elif prediction == "partial" and has_almost_marker and not has_partial_marker:
                # Guidelines only say almost, no partial marker - likely undergrading
                self.log_fn(f"Adjusting from partial to almost (only almost marker in guidelines)")
                prediction = "almost"
            
            # Moderate adjustment based on keyword evidence (need stronger evidence)
            elif prediction == "almost" and partial_count >= almost_count + 3:
                self.log_fn(f"Adjusting from almost to partial based on keyword evidence ({partial_count} vs {almost_count})")
                prediction = "partial"
            elif prediction == "partial" and almost_count >= partial_count + 3:
                self.log_fn(f"Adjusting from partial to almost based on keyword evidence ({almost_count} vs {partial_count})")
                prediction = "almost"
        
        # Adjustment 5: Handle "partial" vs "incorrect" confusion
        if prediction == "incorrect" and guideline_hints["has_partial_marker"]:
            # Guidelines indicate partial credit - check if we should upgrade
            # Only upgrade if partial is the primary marker or there's strong evidence
            if primary_marker == "partial" or len(guideline_hints["partial_keywords"]) >= 2:
                self.log_fn(f"Adjusting from incorrect to partial (guidelines indicate partial)")
                prediction = "partial"
        
        if prediction == "partial" and guideline_hints["has_incorrect_marker"] and not guideline_hints["has_partial_marker"]:
            # Guidelines indicate incorrect, no partial marker - check if we should downgrade
            if len(guideline_hints["partial_keywords"]) == 0 and len(guideline_hints["incorrect_keywords"]) > 0:
                self.log_fn(f"Adjusting from partial to incorrect (guidelines indicate incorrect)")
                prediction = "incorrect"
        
        # Final fallback: if prediction is still None, try to use text extraction on full response
        if prediction == "None":
            text_pred = _extract_label_from_text(raw_text)
            if text_pred:
                prediction = text_pred
        
        # Ultimate fallback - use primary marker if available, otherwise incorrect
        if prediction == "None" or prediction not in ["correct", "almost", "partial", "incorrect"]:
            if guideline_hints["primary_marker"]:
                prediction = guideline_hints["primary_marker"]
                self.log_fn(f"Using primary marker as ultimate fallback: {prediction}")
            else:
                prediction = "incorrect"  # Conservative default
            
        self.log_fn(f"Final prediction: {prediction}")
        return str(prediction), msg_history
