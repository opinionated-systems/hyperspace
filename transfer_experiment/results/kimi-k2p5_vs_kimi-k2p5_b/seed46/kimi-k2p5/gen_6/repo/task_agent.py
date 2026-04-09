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
    """Extract grading label from raw text using multiple strategies."""
    text_lower = text.lower()
    
    # Look for explicit label mentions with word boundaries
    # Priority order matters - check more specific terms first
    
    # Check for "almost" first (before "correct" to avoid confusion)
    if re.search(r'\balmost\b', text_lower):
        return "almost"
    
    # Check for "incorrect" (more specific than "correct")
    if re.search(r'\bincorrect\b', text_lower):
        return "incorrect"
    
    # Check for "partial"
    if re.search(r'\bpartial\b', text_lower):
        return "partial"
    
    # Check for "correct" (but not "incorrect" - already checked above)
    if re.search(r'\bcorrect\b', text_lower):
        return "correct"
    
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
    
    # Check for substring matches - priority order matters
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
    """Analyze grading guidelines to extract hints about expected labels."""
    hints = {
        "has_partial_marker": False,
        "has_almost_marker": False,
        "has_correct_marker": False,
        "has_incorrect_marker": False,
        "partial_keywords": [],
        "almost_keywords": [],
        "correct_keywords": [],
        "incorrect_keywords": [],
        "suggested_label": None,
    }
    
    if not grading_guidelines:
        return hints
    
    guidelines_lower = grading_guidelines.lower()
    
    # Check for explicit markers - these are strong signals
    if "(partial)" in guidelines_lower:
        hints["has_partial_marker"] = True
        hints["suggested_label"] = "partial"
    if "(almost)" in guidelines_lower:
        hints["has_almost_marker"] = True
        hints["suggested_label"] = "almost"
    if "(correct)" in guidelines_lower:
        hints["has_correct_marker"] = True
        hints["suggested_label"] = "correct"
    if "(incorrect)" in guidelines_lower:
        hints["has_incorrect_marker"] = True
        hints["suggested_label"] = "incorrect"
    
    # Also check for partial credit mentions
    if "partial credit" in guidelines_lower:
        hints["has_partial_marker"] = True
    
    # Extract keywords associated with each category
    partial_indicators = [
        "partial", "incomplete", "some progress", "minor understanding",
        "started correctly", "some correct elements", "partial solution",
        "missing", "incomplete solution", "not complete", "far from",
        "only solved", "only proved", "only showed"
    ]
    almost_indicators = [
        "almost", "minor errors", "small mistakes", "nearly complete",
        "essentially correct", "minor slip", "small gap", "nearly solved",
        "mostly correct", "small error", "minor flaw"
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

## Student's Answer:
{student_answer}

## Instructions:
1. Carefully read the problem, official solution, and grading guidelines
2. Analyze the student's answer step by step
3. Compare the student's reasoning and conclusion with the official solution
4. The grading guidelines often contain explicit markers like "(Partial)", "(Almost)", "(Correct)", or "(Incorrect)" - pay close attention to these
5. Make a final determination using EXACTLY one of: "correct", "almost", "partial", or "incorrect"

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

## IMPORTANT:
- If the grading guidelines explicitly mark something as "(Partial)", "(Almost)", "(Correct)", or "(Incorrect)", strongly consider using that label
- "partial" should be used for answers that show genuine insight but are incomplete - do not overuse "almost" for incomplete solutions
- When in doubt between "almost" and "partial", ask: "Is this 90% done (almost) or 30% done (partial)?"

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

        # Use grading guidelines hints to validate or correct the prediction
        # This is especially important for distinguishing "almost" vs "partial"
        if guideline_hints["suggested_label"]:
            suggested = guideline_hints["suggested_label"]
            
            # If we have a valid prediction but it conflicts with explicit guideline markers
            if prediction in ["correct", "almost", "partial", "incorrect"]:
                if prediction != suggested:
                    # Strong override: if guidelines explicitly say (Partial), (Almost), etc.
                    # and our prediction is different, trust the guidelines
                    if (suggested == "partial" and guideline_hints["has_partial_marker"]) or \
                       (suggested == "almost" and guideline_hints["has_almost_marker"]) or \
                       (suggested == "correct" and guideline_hints["has_correct_marker"]) or \
                       (suggested == "incorrect" and guideline_hints["has_incorrect_marker"]):
                        self.log_fn(f"Adjusting prediction from {prediction} to {suggested} based on explicit guideline marker")
                        prediction = suggested
            
            # If prediction is None or invalid, use the suggested label
            if prediction == "None" or prediction not in ["correct", "almost", "partial", "incorrect"]:
                prediction = suggested
                self.log_fn(f"Using suggested label from grading guidelines: {prediction}")
        
        # Additional keyword-based adjustment for "almost" vs "partial" confusion
        if prediction == "almost" and len(guideline_hints["partial_keywords"]) > len(guideline_hints["almost_keywords"]):
            # More partial indicators than almost indicators - consider switching
            if guideline_hints["has_partial_marker"] or len(guideline_hints["partial_keywords"]) >= 2:
                self.log_fn(f"Adjusting from almost to partial based on keyword analysis")
                prediction = "partial"
        
        elif prediction == "partial" and len(guideline_hints["almost_keywords"]) > len(guideline_hints["partial_keywords"]):
            # More almost indicators than partial indicators - consider switching
            if guideline_hints["has_almost_marker"] or len(guideline_hints["almost_keywords"]) >= 2:
                self.log_fn(f"Adjusting from partial to almost based on keyword analysis")
                prediction = "almost"
        
        # Final fallback: if prediction is still None, try to use text extraction on full response
        if prediction == "None":
            text_pred = _extract_label_from_text(raw_text)
            if text_pred:
                prediction = text_pred
        
        self.log_fn(f"Final prediction: {prediction}")
        return str(prediction), msg_history
