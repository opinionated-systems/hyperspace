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


def _is_valid_label(text: str) -> bool:
    """Check if text is a valid classification label (not math/problem content)."""
    if not text:
        return False
    
    text_str = str(text).strip()
    text_lower = text_str.lower()
    
    # If it's exactly one of the four labels, it's valid
    if text_lower in ["correct", "incorrect", "partial", "almost"]:
        return True
    
    # If it contains one of the four labels as a standalone word, it's likely valid
    for label in ["almost", "partial", "incorrect", "correct"]:
        if re.search(rf'\b{label}\b', text_lower):
            return True
    
    # Check for mathematical expressions or problem content (invalid predictions)
    # Only reject if it looks like actual math content
    math_indicators = [
        r'\$\$.*?\$\$',  # LaTeX display math
        r'\\[a-zA-Z]+',   # LaTeX commands
        r'<step',         # Step tags
    ]
    
    math_count = 0
    for pattern in math_indicators:
        if re.search(pattern, text_str):
            math_count += 1
    
    # If strong math indicators found, it's likely math content
    if math_count >= 2:
        return False
    
    # Check for very long text (labels should be short)
    if len(text_str) > 500:
        return False
    
    return True


def _extract_json_from_text(text: str) -> dict | None:
    """Extract JSON object from text using multiple strategies."""
    text = text.strip()
    
    # Strategy 1: <json>...</json> tags
    json_tag_pattern = r'<json>\s*(.*?)\s*</json>'
    matches = re.findall(json_tag_pattern, text, re.DOTALL | re.IGNORECASE)
    for match in matches:
        try:
            data = json.loads(match.strip())
            if isinstance(data, dict) and "response" in data:
                return data
        except json.JSONDecodeError:
            fixed = _fix_json(match.strip())
            if fixed:
                try:
                    data = json.loads(fixed)
                    if isinstance(data, dict) and "response" in data:
                        return data
                except json.JSONDecodeError:
                    pass
    
    # Strategy 2: ```json...``` code blocks
    code_block_pattern = r'```json\s*(.*?)\s*```'
    matches = re.findall(code_block_pattern, text, re.DOTALL | re.IGNORECASE)
    for match in matches:
        try:
            data = json.loads(match.strip())
            if isinstance(data, dict) and "response" in data:
                return data
        except json.JSONDecodeError:
            fixed = _fix_json(match.strip())
            if fixed:
                try:
                    data = json.loads(fixed)
                    if isinstance(data, dict) and "response" in data:
                        return data
                except json.JSONDecodeError:
                    pass
    
    # Strategy 3: ```...``` code blocks with JSON content
    code_block_pattern2 = r'```\s*(\{[\s\S]*?\})\s*```'
    matches = re.findall(code_block_pattern2, text, re.DOTALL | re.IGNORECASE)
    for match in matches:
        try:
            data = json.loads(match.strip())
            if isinstance(data, dict) and "response" in data:
                return data
        except json.JSONDecodeError:
            fixed = _fix_json(match.strip())
            if fixed:
                try:
                    data = json.loads(fixed)
                    if isinstance(data, dict) and "response" in data:
                        return data
                except json.JSONDecodeError:
                    pass
    
    # Strategy 4: Raw JSON objects
    json_pattern = r'\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}'
    matches = re.findall(json_pattern, text, re.DOTALL)
    for match in matches:
        try:
            data = json.loads(match)
            if isinstance(data, dict) and "response" in data:
                return data
        except json.JSONDecodeError:
            fixed = _fix_json(match)
            if fixed:
                try:
                    data = json.loads(fixed)
                    if isinstance(data, dict) and "response" in data:
                        return data
                except json.JSONDecodeError:
                    pass
    
    return None


def _fix_json(text: str) -> str | None:
    """Try to fix common JSON formatting issues."""
    if not text or not text.strip():
        return None
    
    text = text.strip()
    
    # Fix 1: Remove trailing commas before closing braces
    text = re.sub(r',\s*}', '}', text)
    text = re.sub(r',\s*\]', ']', text)
    
    # Fix 2: Replace single quotes with double quotes (simple cases)
    if "'" in text and '"' not in text:
        text = text.replace("'", '"')
    
    # Fix 3: Ensure the text starts with { and ends with }
    if not text.startswith('{'):
        start = text.find('{')
        if start != -1:
            text = text[start:]
    
    if not text.endswith('}'):
        end = text.rfind('}')
        if end != -1:
            text = text[:end+1]
    
    return text if text.startswith('{') and text.endswith('}') else None


def _extract_response_direct(text: str) -> str | None:
    """Extract response by looking for keywords directly in text."""
    text_lower = text.lower()
    
    # Look for explicit classification statements with priority
    # Priority: almost > partial > incorrect > correct (most specific first)
    
    # Check for "almost" first (most specific)
    almost_patterns = [
        r'"response"\s*:\s*"almost"',
        r'"classification"\s*:\s*"almost"',
        r'"grade"\s*:\s*"almost"',
        r'"result"\s*:\s*"almost"',
        r'"evaluation"\s*:\s*"almost"',
        r'\bclassif(y|ied)\s+as\s+almost\b',
        r'\bthis\s+is\s+almost\b',
    ]
    for pattern in almost_patterns:
        if re.search(pattern, text_lower):
            return "almost"
    
    # Check for "partial"
    partial_patterns = [
        r'"response"\s*:\s*"partial"',
        r'"classification"\s*:\s*"partial"',
        r'"grade"\s*:\s*"partial"',
        r'"result"\s*:\s*"partial"',
        r'"evaluation"\s*:\s*"partial"',
        r'\bclassif(y|ied)\s+as\s+partial\b',
        r'\bthis\s+is\s+partial\b',
    ]
    for pattern in partial_patterns:
        if re.search(pattern, text_lower):
            return "partial"
    
    # Check for "incorrect"
    incorrect_patterns = [
        r'"response"\s*:\s*"incorrect"',
        r'"classification"\s*:\s*"incorrect"',
        r'"grade"\s*:\s*"incorrect"',
        r'"result"\s*:\s*"incorrect"',
        r'"evaluation"\s*:\s*"incorrect"',
        r'\bclassif(y|ied)\s+as\s+incorrect\b',
        r'\bthis\s+is\s+incorrect\b',
    ]
    for pattern in incorrect_patterns:
        if re.search(pattern, text_lower):
            return "incorrect"
    
    # Check for "correct"
    correct_patterns = [
        r'"response"\s*:\s*"correct"',
        r'"classification"\s*:\s*"correct"',
        r'"grade"\s*:\s*"correct"',
        r'"result"\s*:\s*"correct"',
        r'"evaluation"\s*:\s*"correct"',
        r'\bclassif(y|ied)\s+as\s+correct\b',
        r'\bthis\s+is\s+correct\b',
    ]
    for pattern in correct_patterns:
        if re.search(pattern, text_lower):
            return "correct"
    
    # Count standalone occurrences as fallback
    almost_count = len(re.findall(r'\balmost\b', text_lower))
    partial_count = len(re.findall(r'\bpartial\b', text_lower))
    incorrect_count = len(re.findall(r'\bincorrect\b', text_lower))
    correct_count = len(re.findall(r'\bcorrect\b', text_lower))
    
    # If only one appears, use that
    if almost_count > 0 and partial_count == 0 and incorrect_count == 0 and correct_count == 0:
        return "almost"
    if partial_count > 0 and almost_count == 0 and incorrect_count == 0 and correct_count == 0:
        return "partial"
    if incorrect_count > 0 and almost_count == 0 and partial_count == 0 and correct_count == 0:
        return "incorrect"
    if correct_count > 0 and almost_count == 0 and partial_count == 0 and incorrect_count == 0:
        return "correct"
    
    # Priority-based selection when multiple appear
    if almost_count > 0:
        return "almost"
    if partial_count > 0:
        return "partial"
    if incorrect_count > 0:
        return "incorrect"
    if correct_count > 0:
        return "correct"
    
    return None


def _normalize_prediction(prediction: str) -> str:
    """Normalize prediction to one of the four valid labels."""
    pred_str = str(prediction).lower().strip()
    
    # Direct match
    if pred_str in ["correct", "incorrect", "partial", "almost"]:
        return pred_str
    
    # Check if it's a valid label at all
    if not _is_valid_label(prediction):
        return "incorrect"
    
    # Priority: almost > partial > incorrect > correct
    if "almost" in pred_str:
        return "almost"
    if "partial" in pred_str:
        return "partial"
    if "incorrect" in pred_str:
        return "incorrect"
    if "correct" in pred_str:
        return "correct"
    
    return "incorrect"


def _check_grading_guidelines_for_almost(grading_guidelines: str) -> bool:
    """Check if grading guidelines indicate 'almost' level.
    
    Returns True if guidelines suggest the solution is nearly complete
    with only minor issues.
    """
    if not grading_guidelines:
        return False
    
    guidelines_lower = grading_guidelines.lower()
    
    # Check for explicit point indicators of "almost" level (5-6 points)
    almost_point_indicators = [
        r'\b5[-\s]?6\s+points\b',
        r'\b6\s+points\b',
        r'\b5\s+points\b',
    ]
    
    for pattern in almost_point_indicators:
        if re.search(pattern, guidelines_lower):
            return True
    
    # Check for (Almost) marker anywhere in guidelines
    if "(almost)" in guidelines_lower:
        return True
    
    # Check for keywords suggesting minor issues only
    minor_issue_keywords = [
        "minor mistakes",
        "minor errors",
        "small errors",
        "technical errors",
        "verification contains",
        "almost complete",
        "nearly complete",
    ]
    
    for keyword in minor_issue_keywords:
        if keyword in guidelines_lower:
            return True
    
    return False


def _check_grading_guidelines_for_partial(grading_guidelines: str) -> bool:
    """Check if grading guidelines strongly indicate 'partial' level.
    
    Returns True if guidelines explicitly indicate partial level
    through point values or clear statements of significant gaps.
    """
    if not grading_guidelines:
        return False
    
    guidelines_lower = grading_guidelines.lower()
    
    # Check for explicit point indicators of "partial" level (1-4 points)
    partial_point_indicators = [
        r'\b1[-\s]?4\s+points\b',
        r'\b2[-\s]?3\s+points\b',
        r'\b3[-\s]?4\s+points\b',
        r'\b4\s+points\b',
        r'\b3\s+points\b',
        r'\b2\s+points\b',
        r'\b1\s+point\b',
    ]
    
    for pattern in partial_point_indicators:
        if re.search(pattern, guidelines_lower):
            return True
    
    # Check for keywords suggesting significant gaps
    significant_gap_keywords = [
        "significant gaps",
        "major gaps",
        "incomplete proof",
        "missing key steps",
        "far from complete",
        "only partial progress",
    ]
    
    for keyword in significant_gap_keywords:
        if keyword in guidelines_lower:
            return True
    
    return False


def _check_grading_guidelines_for_correct(grading_guidelines: str) -> bool:
    """Check if grading guidelines explicitly indicate 'correct' level.
    
    Returns True if guidelines explicitly indicate correct level
    through point values (7 points) or clear statements.
    """
    if not grading_guidelines:
        return False
    
    guidelines_lower = grading_guidelines.lower()
    
    # Check for explicit 7 points indicator
    if re.search(r'\b7\s+points\b', guidelines_lower):
        return True
    
    # Check for explicit (Correct) marker
    if "(correct)" in guidelines_lower:
        return True
    
    # Check for strong indicators of perfect solution
    perfect_keywords = [
        "complete solution",
        "correct solution",
        "full marks",
        "full credit",
        "perfect solution",
    ]
    
    for keyword in perfect_keywords:
        if keyword in guidelines_lower:
            return True
    
    return False


def _check_grading_guidelines_for_incorrect(grading_guidelines: str) -> bool:
    """Check if grading guidelines strongly indicate 'incorrect' level.
    
    Returns True if guidelines explicitly indicate incorrect level
    through point values (0 points) or clear statements of no progress.
    """
    if not grading_guidelines:
        return False
    
    guidelines_lower = grading_guidelines.lower()
    
    # Check for explicit 0 points indicator
    if re.search(r'\b0\s+points\b', guidelines_lower):
        return True
    
    # Check for explicit (Incorrect) marker
    if "(incorrect)" in guidelines_lower:
        return True
    
    # Check for strong indicators of no valid progress
    no_progress_keywords = [
        "no valid lemmas",
        "no non-trivial observations",
        "fundamentally flawed",
        "no meaningful progress",
        "completely wrong",
        "trivial only",
    ]
    
    for keyword in no_progress_keywords:
        if keyword in guidelines_lower:
            return True
    
    return False


class TaskAgent:
    """Task agent that solves IMO grading problems."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem."""
        # Extract key fields from inputs
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        instruction = f"""You are an expert mathematical olympiad grader. Your task is to classify the student's answer into EXACTLY ONE of four categories: "correct", "almost", "partial", or "incorrect".

## Problem Statement:
{problem}

## Official Solution:
{solution}

## Grading Guidelines:
{grading_guidelines}

## Student's Answer:
{student_answer}

## Classification Rules (READ CAREFULLY):

**"correct"** (7 points): PERFECT solution. Complete, valid reasoning throughout, correct final answer, no gaps or errors. The solution could be published as-is. Use this when the solution is truly flawless.

**"almost"** (5-6 points): Nearly complete solution. Main proof structure is sound and complete. Only minor technical errors (typos, small calculation errors, NOT conceptual gaps). The core mathematical argument is complete and correct, just needs minor polishing.

**"partial"** (1-4 points): Meaningful progress (valid lemmas, right approach) but SIGNIFICANT gaps remain. The solution is far from complete. Student made some real progress but major work is missing or there are serious conceptual errors.

**"incorrect"** (0 points): No valid lemmas or non-trivial observations. Fundamentally flawed or trivial only. No meaningful mathematical progress.

## Decision Process (FOLLOW THIS ORDER STRICTLY):

**STEP 1: Check for "incorrect"**
- Does the solution contain ANY valid mathematical lemma, non-trivial observation, or correct proof step?
- If NO valid progress at all → "incorrect"

**STEP 2: Check for "partial" vs "almost"**
- Are there SIGNIFICANT gaps in the solution? Is the main proof structure incomplete?
- Does the solution have major conceptual errors or missing key steps?
- If significant gaps exist → "partial"
- If the structure is nearly complete with only minor issues → proceed to "almost"

**STEP 3: Check for "almost" vs "correct"**
- Is the solution PERFECT with absolutely no errors or gaps?
- Could this solution be published as-is without any corrections?
- If there are ANY errors, gaps, or incomplete parts → "almost"
- ONLY if truly perfect → "correct"

## CRITICAL DISTINCTIONS:

**"correct" vs "almost":**
- "correct" = TRULY PERFECT, no issues whatsoever
- "almost" = Nearly complete, minor technical errors only
- If the solution has the complete proof structure AND correct reasoning throughout → "correct"
- If there are ANY errors, typos, or minor gaps → "almost"

**"almost" vs "partial":**
- "almost" = Main proof structure is complete and correct, just needs small fixes
- "partial" = Significant gaps, incomplete structure, or major errors
- If the student got most of the way there → "almost"
- If the student only made partial progress → "partial"

**"partial" vs "incorrect":**
- "partial" = At least one valid lemma or meaningful progress
- "incorrect" = Nothing of value, completely wrong approach

## IMPORTANT GUIDANCE:
- Look at the grading guidelines carefully - they indicate the point value
- If guidelines say "(Correct)" or indicate 7 points → "correct"
- If guidelines say "(Almost)" or indicate 5-6 points → "almost"
- If guidelines say "(Partial)" or indicate 1-4 points → "partial"
- If guidelines say "(Incorrect)" or indicate 0 points → "incorrect"
- When in doubt between "almost" and "partial", choose "almost" if the main structure is sound

## OUTPUT FORMAT (STRICT JSON - MANDATORY):
You MUST output ONLY a JSON object. The response field MUST contain EXACTLY ONE of these four words: "correct", "almost", "partial", or "incorrect". NO OTHER TEXT is allowed in the response field.

<json>
{{
    "reasoning": "Your detailed analysis here. Be specific about what the student achieved and what is missing.",
    "response": "almost"
}}
</json>

Replace "almost" with "correct", "partial", or "incorrect" as appropriate."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from response
        prediction = "incorrect"  # Default fallback
        try:
            if msg_history and len(msg_history) > 0:
                response_text = msg_history[-1].get("text", "")
                
                if response_text:
                    # First try: extract from JSON
                    json_data = _extract_json_from_text(response_text)
                    
                    if json_data and isinstance(json_data, dict):
                        # Look for response field first
                        if "response" in json_data:
                            prediction = _normalize_prediction(json_data["response"])
                        else:
                            # Check other common fields
                            for field in ["classification", "grade", "result", "evaluation", "verdict", "outcome"]:
                                if field in json_data:
                                    prediction = _normalize_prediction(json_data[field])
                                    break
                    else:
                        # Try direct extraction from text
                        direct = _extract_response_direct(response_text)
                        if direct:
                            prediction = direct
                        else:
                            self.log_fn(f"No valid response found in: {response_text[:200]}...")
                else:
                    self.log_fn("Empty response text in msg_history")
            else:
                self.log_fn("Empty msg_history")
                
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            import traceback
            self.log_fn(traceback.format_exc())

        # Post-processing: Check grading guidelines for adjustments
        # Be VERY CONSERVATIVE: only adjust when there's EXPLICIT and STRONG evidence
        # The LLM's judgment should be trusted in most cases
        if prediction in ["correct", "almost", "partial", "incorrect"]:
            # Check if guidelines explicitly say "correct" - if so, trust the LLM's prediction
            guidelines_say_correct = _check_grading_guidelines_for_correct(grading_guidelines)
            
            # Only downgrade to "incorrect" if guidelines EXPLICITLY say 0 points or (Incorrect)
            if prediction in ["correct", "almost", "partial"]:
                if _check_grading_guidelines_for_incorrect(grading_guidelines):
                    self.log_fn("Adjusting prediction to 'incorrect' based on explicit grading guidelines")
                    prediction = "incorrect"
            
            # Only downgrade "correct" to "partial" if guidelines explicitly indicate 1-4 points
            # AND guidelines don't explicitly say "correct"
            if prediction == "correct" and not guidelines_say_correct:
                if _check_grading_guidelines_for_partial(grading_guidelines):
                    self.log_fn("Adjusting prediction to 'partial' based on explicit grading guidelines")
                    prediction = "partial"
            
            # Only downgrade "correct" to "almost" if guidelines explicitly indicate 5-6 points
            # AND guidelines don't explicitly say "correct"
            if prediction == "correct" and not guidelines_say_correct:
                if _check_grading_guidelines_for_almost(grading_guidelines):
                    self.log_fn("Adjusting prediction to 'almost' based on explicit grading guidelines")
                    prediction = "almost"

        # Final validation
        if prediction not in ["correct", "incorrect", "partial", "almost"]:
            prediction = "incorrect"

        return str(prediction), msg_history
