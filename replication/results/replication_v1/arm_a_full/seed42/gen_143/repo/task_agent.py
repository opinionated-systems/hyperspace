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
from typing import Optional

from agent.llm_client import get_response_from_llm, EVAL_MODEL

logger = logging.getLogger(__name__)

# Maximum retries for LLM calls
MAX_RETRIES = 3
# Delay between retries in seconds
RETRY_DELAY = 1.0


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


def _get_llm_response_with_retry(
    msg: str,
    model: str,
    max_retries: int = MAX_RETRIES,
    retry_delay: float = RETRY_DELAY
) -> tuple[str, list[dict], dict]:
    """Get LLM response with retry logic for improved reliability.
    
    Args:
        msg: The message to send to the LLM
        model: The model to use
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds
        
    Returns:
        Tuple of (response, msg_history, info)
        
    Raises:
        RuntimeError: If all retry attempts fail
    """
    last_error: Optional[Exception] = None
    
    for attempt in range(max_retries):
        try:
            response, msg_history, info = get_response_from_llm(
                msg=msg,
                model=model,
                msg_history=[],
            )
            logger.info(f"LLM call succeeded on attempt {attempt + 1}")
            return response, msg_history, info
        except Exception as e:
            last_error = e
            logger.warning(f"LLM call failed on attempt {attempt + 1}/{max_retries}: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
    
    raise RuntimeError(f"Failed to get LLM response after {max_retries} attempts: {last_error}")


def _normalize_prediction(prediction: str) -> str:
    """Normalize prediction to one of the four valid labels.
    
    Args:
        prediction: Raw prediction string from LLM
        
    Returns:
        One of: correct, almost, partial, incorrect
    """
    if not prediction:
        return "incorrect"
    
    # Clean up the prediction
    cleaned = prediction.strip().lower()
    
    # Remove any punctuation, quotes, or extra whitespace
    cleaned = cleaned.strip(".!?,:;\"'").strip()
    
    # Map to valid labels
    valid_labels = ["correct", "almost", "partial", "incorrect"]
    
    # Check for exact match first
    if cleaned in valid_labels:
        return cleaned
    
    # Handle common variations and edge cases
    # Check for "partially correct" -> partial
    if "partially" in cleaned and "correct" in cleaned:
        return "partial"
    
    # Check for "mostly correct" or "nearly correct" -> almost
    if ("mostly" in cleaned or "nearly" in cleaned or "almost" in cleaned) and "correct" in cleaned:
        return "almost"
    
    # Check for "completely correct" or "fully correct" -> correct
    if ("completely" in cleaned or "fully" in cleaned or "totally" in cleaned) and "correct" in cleaned:
        return "correct"
    
    # Check for negations that might indicate incorrect
    if "not correct" in cleaned or "isn't correct" in cleaned or "not right" in cleaned:
        return "incorrect"
    
    # Check for partial matches with priority order
    # "incorrect" should be checked before "correct" to avoid false positives
    if "incorrect" in cleaned or "wrong" in cleaned or "error" in cleaned:
        return "incorrect"
    if "almost" in cleaned:
        return "almost"
    if "partial" in cleaned:
        return "partial"
    if "correct" in cleaned or "right" in cleaned:
        return "correct"
    
    # Default to incorrect if no match found
    logger.warning(f"Could not normalize prediction '{prediction}', defaulting to 'incorrect'")
    return "incorrect"


def _extract_label_from_text(text: str) -> str:
    """Extract a valid label from raw text when JSON parsing fails.
    
    Uses multiple strategies to find the label, with priority given to
    explicit JSON-like patterns and quoted labels.
    
    Args:
        text: Raw text from LLM response
        
    Returns:
        One of: correct, almost, partial, incorrect
    """
    if not text:
        return "incorrect"
    
    text_lower = text.lower()
    valid_labels = ["correct", "almost", "partial", "incorrect"]
    
    # Strategy 1: Look for JSON-like patterns with "response" field
    json_pattern = re.search(r'"response"\s*:\s*"([^"]+)"', text, re.IGNORECASE)
    if json_pattern:
        extracted = json_pattern.group(1).strip().lower()
        if extracted in valid_labels:
            return extracted
    
    # Strategy 2: Look for labels in single quotes
    single_quote_pattern = re.search(r"'\s*(correct|almost|partial|incorrect)\s*'", text_lower)
    if single_quote_pattern:
        return single_quote_pattern.group(1)
    
    # Strategy 3: Look for labels after common prefixes
    prefixes = [
        r'label\s*[:=]\s*["\']?\s*(correct|almost|partial|incorrect)',
        r'grade\s*[:=]\s*["\']?\s*(correct|almost|partial|incorrect)',
        r'classification\s*[:=]\s*["\']?\s*(correct|almost|partial|incorrect)',
        r'answer\s*[:=]\s*["\']?\s*(correct|almost|partial|incorrect)',
        r'result\s*[:=]\s*["\']?\s*(correct|almost|partial|incorrect)',
    ]
    for pattern in prefixes:
        match = re.search(pattern, text_lower)
        if match:
            return match.group(1)
    
    # Strategy 4: Look for standalone labels with word boundaries
    # Check in reverse priority order (incorrect first to avoid matching "correct" in "incorrect")
    for label in reversed(valid_labels):
        # Use word boundary to avoid partial matches
        if re.search(rf'\b{label}\b', text_lower):
            return label
    
    # Strategy 5: Check for partial word matches as fallback
    if "incorrect" in text_lower or "wrong" in text_lower:
        return "incorrect"
    if "almost" in text_lower or "nearly" in text_lower:
        return "almost"
    if "partial" in text_lower or "partially" in text_lower:
        return "partial"
    if "correct" in text_lower or "right" in text_lower:
        return "correct"
    
    logger.warning(f"Could not extract label from text, defaulting to 'incorrect': {text[:100]}...")
    return "incorrect"


# Few-shot examples for the prompt - carefully selected to show clear distinctions
FEW_SHOT_EXAMPLES = """
### Example 1 - CORRECT:
Problem: Solve for x: 2x + 4 = 10
Student Answer: 2x = 6, so x = 3
Analysis: The student shows correct algebraic manipulation. They subtracted 4 from both sides to get 2x = 6, then divided by 2 to get x = 3. This is fully correct.
Label: correct

### Example 2 - ALMOST:
Problem: Find the derivative of f(x) = x^3
Student Answer: f'(x) = 3x^2
Analysis: The student correctly applied the power rule. The answer 3x^2 is correct, though they didn't explicitly write "f'(x) =" before the answer. This is a tiny notation issue - the mathematical work is completely correct.
Label: almost

### Example 3 - PARTIAL:
Problem: Solve x^2 = 4
Student Answer: x = 2
Analysis: The student found one solution (x = 2) but missed the other solution (x = -2). They showed correct work for finding the positive root, but the answer is incomplete. This is a significant gap.
Label: partial

### Example 4 - INCORRECT:
Problem: Find the area of a circle with radius 5
Student Answer: 10π
Analysis: The student used the diameter (10) instead of the radius, applying A = d·π instead of A = πr². This shows a fundamental misunderstanding of the formula.
Label: incorrect

### Example 5 - ALMOST (arithmetic slip):
Problem: Compute 15 × 4
Student Answer: 50
Analysis: The correct answer is 60. The student made a small arithmetic error (15 × 4 = 60, not 50). However, they correctly identified that multiplication was needed. This is a tiny computational slip after correct reasoning.
Label: almost

### Example 6 - PARTIAL (wrong formula, right computation):
Problem: Find the volume of a sphere with radius 3
Student Answer: 36π (using V = 4πr² instead of V = (4/3)πr³)
Analysis: The student used the wrong formula (surface area instead of volume) but computed it correctly. They showed valid mathematical work with the formula they used, but applied the wrong concept entirely.
Label: partial
"""


class TaskAgent:
    """Task agent that evaluates student answers with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, temperature: float = 0.0) -> None:
        self.model = model
        self.temperature = temperature
        self.log_fn = logger.info
        self.call_count = 0
        self.error_count = 0

    def forward(self, problem: str, student_answer: str, correct_answer: str, metadata: dict | None = None) -> tuple[str, list[dict]]:
        """Evaluate a student answer and return a label.
        
        Args:
            problem: The problem statement
            student_answer: The student's answer
            correct_answer: The correct answer
            metadata: Optional metadata about the problem
            
        Returns:
            Tuple of (prediction_label, message_history)
        """
        self.call_count += 1
        
        # Build the evaluation prompt with enhanced instructions
        instruction = f"""You are an expert mathematics teacher evaluating student work. Your task is to classify the student's answer into exactly one of four categories.

## Problem:
{problem}

## Correct Answer:
{correct_answer}

## Student's Answer:
{student_answer}

## Label Definitions (7-point grading scale):

**CORRECT** (7/7 points):
- The answer is fully correct and complete
- All steps are correct and properly shown
- Would receive full credit

**ALMOST** (6-7/7 points):
- Nearly correct with only TINY, MINOR issues
- Small arithmetic slip (e.g., 15+30=170 instead of 180)
- Missing notation prefix (answer is right, just no "f'(x)=")
- Sign error at the very end after correct work
- Rounding differences
- The student CLEARLY understands the concept
- KEY: Error is truly tiny - would lose at most 1 point

**PARTIAL** (1-5/7 points):
- Has SOME valid mathematical work but significant gaps
- Missing half the solutions (e.g., only x=2 when x=±2 needed)
- Used wrong formula but computed it correctly
- Incomplete proof with key steps missing
- Shows SOME understanding but would lose 2-6 points
- KEY: There is valid work, but major issues exist

**INCORRECT** (0-1/7 points):
- No valid mathematical reasoning demonstrated
- Completely wrong approach with no redeeming work
- Just guessing or irrelevant text
- Would receive minimal or no credit
- KEY: No valid mathematical thinking shown

## Critical Distinctions - READ CAREFULLY:

**ALMOST vs PARTIAL** - This is the most important distinction:
- ALMOST = Would get 6-7/7 points (tiny issue only)
- PARTIAL = Would get 1-5/7 points (significant gaps)

Ask yourself: "How many points would a teacher give?"

**PARTIAL vs INCORRECT**:
- PARTIAL = Some valid mathematical work exists (even if wrong answer)
- INCORRECT = No valid mathematical reasoning at all

## Step-by-Step Evaluation Process:
1. Read the problem carefully and understand what is being asked
2. Compare the student's answer against the correct solution step by step
3. Identify what the student got right and what they got wrong
4. Determine: On a 7-point scale, how many points would this receive?
   - 7 points → CORRECT
   - 6-7 points → ALMOST (minor issue only)
   - 1-5 points → PARTIAL (significant gaps but some valid work)
   - 0-1 points → INCORRECT (no valid work)
5. Select the label that best matches

## Self-Reflection (MANDATORY - check your answer):
Before finalizing, ask yourself:
- "Did I confuse ALMOST with PARTIAL?" (most common error)
- "Is the error truly tiny (ALMOST=6-7pts) or significant (PARTIAL=1-5pts)?"
- "Is there ANY valid mathematical work?" (if yes, not INCORRECT)
- "Would a teacher give 6-7 points or fewer?"

{FEW_SHOT_EXAMPLES}

## Response Format:
First provide your chain-of-thought analysis, then respond with EXACTLY ONE of these four labels in JSON format:
- "correct" - Fully correct and complete (7/7 points)
- "almost" - Nearly correct, tiny issues only (6-7/7 points)
- "partial" - Some valid work but significant gaps (1-5/7 points)
- "incorrect" - No valid mathematical reasoning (0-1/7 points)

<json>
{{
    "response": "correct|almost|partial|incorrect"
}}
</json>

IMPORTANT: Your response field must contain ONLY one of these four exact words: correct, almost, partial, or incorrect. Do not include any other text."""

        try:
            response, msg_history, info = _get_llm_response_with_retry(
                msg=instruction,
                model=self.model,
            )
        except RuntimeError as e:
            logger.error(f"Failed to get LLM response: {e}")
            self.error_count += 1
            return "Error: Failed to get LLM response", []

        # Extract prediction from JSON with improved error handling
        prediction = "incorrect"  # Default to incorrect if extraction fails
        try:
            if msg_history and len(msg_history) > 0:
                extracted = _extract_jsons(msg_history[-1]["text"])
                if extracted and len(extracted) > 0 and "response" in extracted[-1]:
                    raw_prediction = extracted[-1]["response"]
                    # Normalize and validate the prediction
                    prediction = _normalize_prediction(raw_prediction)
                    logger.info(f"Successfully extracted prediction: {prediction}")
                else:
                    # Try to extract label from raw text if JSON parsing fails
                    text = msg_history[-1].get("text", "")
                    prediction = _extract_label_from_text(text)
                    logger.info(f"Extracted label from text: {prediction}")
            else:
                logger.warning("Empty message history from LLM")
        except Exception as e:
            self.error_count += 1
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history

    def get_stats(self) -> dict:
        """Return agent statistics for monitoring.
        
        Returns:
            Dictionary with call_count and error_count
        """
        return {
            "call_count": self.call_count,
            "error_count": self.error_count,
            "success_rate": (self.call_count - self.error_count) / max(self.call_count, 1)
        }
