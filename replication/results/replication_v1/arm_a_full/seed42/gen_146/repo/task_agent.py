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
    
    # Remove any punctuation or extra whitespace
    cleaned = cleaned.strip(".!?,:;\"'").strip()
    
    # Map to valid labels
    valid_labels = ["correct", "almost", "partial", "incorrect"]
    
    # Check for exact match first
    if cleaned in valid_labels:
        return cleaned
    
    # Check for exact match with whitespace
    cleaned_no_space = cleaned.replace(" ", "").replace("_", "")
    for label in valid_labels:
        if cleaned_no_space == label:
            return label
    
    # Check for partial matches - be careful with "correct" appearing in other words
    # Order matters: check for more specific terms first
    if "almost" in cleaned:
        return "almost"
    if "partial" in cleaned:
        return "partial"
    if "incorrect" in cleaned or "wrong" in cleaned or "error" in cleaned:
        return "incorrect"
    
    # Only return "correct" if it's a standalone word (not part of "partially correct" etc.)
    if cleaned == "correct":
        return "correct"
    
    # Check if "correct" appears as a standalone word
    words = cleaned.split()
    if "correct" in words and "partial" not in cleaned and "incorrect" not in cleaned and "almost" not in cleaned:
        return "correct"
    
    # Default to incorrect if no match found
    logger.warning(f"Could not normalize prediction '{prediction}', defaulting to 'incorrect'")
    return "incorrect"


def _extract_label_from_text(text: str) -> str:
    """Extract a valid label from raw text when JSON parsing fails.
    
    Args:
        text: Raw text from LLM response
        
    Returns:
        One of: correct, almost, partial, incorrect
    """
    if not text:
        return "incorrect"
    
    text_lower = text.lower()
    
    # Look for explicit labels in quotes first (most reliable)
    if '"almost"' in text_lower or "'almost'" in text_lower:
        return "almost"
    if '"partial"' in text_lower or "'partial'" in text_lower:
        return "partial"
    if '"incorrect"' in text_lower or "'incorrect'" in text_lower:
        return "incorrect"
    if '"correct"' in text_lower or "'correct'" in text_lower:
        return "correct"
    
    # Check for keywords in the text - order matters! More specific first
    if "almost" in text_lower:
        return "almost"
    if "partial" in text_lower:
        return "partial"
    if "incorrect" in text_lower or "wrong" in text_lower or "error" in text_lower:
        return "incorrect"
    if "correct" in text_lower:
        # Make sure it's not part of "partially correct" or "almost correct"
        if "partial" not in text_lower and "almost" not in text_lower:
            return "correct"
    
    # Default to incorrect
    logger.warning(f"Could not extract label from text, defaulting to 'incorrect'")
    return "incorrect"


# Few-shot examples showing exact label format with clear distinctions
FEW_SHOT_EXAMPLES = """
Example 1 - CORRECT (complete and accurate):
Problem: Find the sum of 2 + 3.
Solution: The sum of 2 + 3 is 5.
Student Answer: 5
<json>
{"response": "correct"}
</json>

Example 2 - CORRECT (with reasoning):
Problem: Solve 2x + 4 = 10.
Solution: Subtract 4: 2x = 6, then divide by 2: x = 3.
Student Answer: 2x + 4 = 10 → 2x = 6 → x = 3
<json>
{"response": "correct"}
</json>

Example 3 - ALMOST (minor calculation error):
Problem: Solve 2x = 10.
Solution: x = 5.
Student Answer: x = 6 (divided 10 by 2 incorrectly, but method was right)
<json>
{"response": "almost"}
</json>

Example 4 - ALMOST (trivial omission):
Problem: Find area of circle with radius 3.
Solution: A = πr² = 9π ≈ 28.27.
Student Answer: A = π(3)² = 9π = 28.26 (small rounding difference, correct formula and approach)
<json>
{"response": "almost"}
</json>

Example 5 - ALMOST (sign error with correct method):
Problem: Solve x² - 5x + 6 = 0.
Solution: (x - 2)(x - 3) = 0, so x = 2 or x = 3.
Student Answer: (x - 2)(x + 3) = 0, so x = 2 or x = -3. [Factoring method correct, but sign error on second factor]
<json>
{"response": "almost"}
</json>

Example 6 - PARTIAL (incomplete solution, missing key parts):
Problem: Solve x² = 4 for x.
Solution: x = 2 or x = -2.
Student Answer: x = 2 (found one solution but missed the negative solution)
<json>
{"response": "partial"}
</json>

Example 7 - PARTIAL (right direction but incomplete execution):
Problem: Prove sum of triangle angles is 180°.
Solution: Draw parallel line, use alternate interior angles to show angles form straight line.
Student Answer: The angles form a straight line so they sum to 180°. [States conclusion but no explanation of why]
<json>
{"response": "partial"}
</json>

Example 8 - PARTIAL (some valid work but significant gaps):
Problem: Find the derivative of f(x) = 3x² + 2x + 1.
Solution: f'(x) = 6x + 2.
Student Answer: f'(x) = 6x [Correctly derived first term but missed second term]
<json>
{"response": "partial"}
</json>

Example 9 - INCORRECT (fundamentally wrong answer):
Problem: What is the capital of France?
Solution: Paris.
Student Answer: London.
<json>
{"response": "incorrect"}
</json>

Example 10 - INCORRECT (wrong concept applied):
Problem: Find the derivative of x².
Solution: 2x using power rule.
Student Answer: x³/3 (confused derivative with integral).
<json>
{"response": "incorrect"}
</json>

Example 11 - INCORRECT (no valid mathematical reasoning):
Problem: Prove the sum of triangle angles is 180°.
Solution: [Geometric proof with parallel lines and angle relationships].
Student Answer: It's 180 because that's what I learned in school. [No mathematical reasoning provided]
<json>
{"response": "incorrect"}
</json>

Example 12 - ALMOST vs PARTIAL distinction (IMPORTANT):
Problem: Factor x² - 5x + 6.
Solution: (x - 2)(x - 3).
Student Answer A: (x - 2)(x + 3) [sign error, but factoring method correct] → ALMOST
Student Answer B: The roots are x = 2 and x = 3 [found roots but didn't express as factors] → PARTIAL
<json>
{"response": "almost"}
</json>

Example 13 - PARTIAL (started correctly but didn't finish):
Problem: Solve the system: x + y = 5, x - y = 1.
Solution: Add equations: 2x = 6, so x = 3. Substitute: 3 + y = 5, so y = 2.
Student Answer: Adding gives 2x = 6, so x = 3. [Found x but didn't solve for y]
<json>
{"response": "partial"}
</json>

Example 14 - ALMOST (arithmetic slip in correct method):
Problem: Calculate 15 × 4.
Solution: 60.
Student Answer: 15 × 4 = 15 + 15 + 15 + 15 = 30 + 15 + 15 = 45 + 15 = 62. [Correct method but arithmetic error at the end]
<json>
{"response": "almost"}
</json>
"""


class TaskAgent:
    """Task agent that solves IMO grading problems with improved reliability."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.call_count = 0
        self.error_count = 0

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with retry logic.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        self.call_count += 1
        logger.info(f"TaskAgent forward call #{self.call_count}")
        
        # Extract fields from inputs for better structured prompting
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        instruction = f"""You are an expert grading agent for {domain}. Your task is to carefully analyze the student's answer and assign exactly one of four labels.

## Problem Statement:
{problem}

## Correct Solution:
{solution}

## Grading Guidelines:
{grading_guidelines}

## Student's Answer:
{student_answer}

## Decision Framework - Follow These Steps IN ORDER:

**STEP 1: Check for "correct"**
- Does the answer match the solution exactly (or mathematically equivalent form)?
- Is the reasoning complete, sound, and clearly presented?
- Are all required components present?
- If YES → label as "correct"

**STEP 2: If not correct, check for "almost"**
- Is the main method/approach fundamentally correct?
- Is there only ONE minor error (small calculation mistake, trivial omission, rounding difference)?
- Would a single small fix make it fully correct?
- KEY: The error must be minor and the core approach must be right
- If YES → label as "almost"

**STEP 3: If not almost, check for "partial"**
- Does the student show genuine understanding of the problem?
- Is there a valid start or right direction?
- Are there significant gaps, missing steps, or incomplete reasoning?
- KEY: Some valid work exists, but major portions are missing or wrong
- If YES → label as "partial"

**STEP 4: If none of the above, label as "incorrect"**
- The answer is fundamentally wrong, uses wrong concepts, or shows no valid reasoning
- No meaningful progress toward the solution

## Label Definitions with Criteria:

**"correct"** - The answer is fully correct and complete:
- Matches the solution in both method and final result
- Shows clear, valid, complete reasoning
- No errors or omissions of any significance
- Mathematically equivalent answers are acceptable

**"almost"** - Nearly correct with only minor issues:
- Core method/approach is correct
- Only ONE type of minor error: calculation mistake, trivial omission, or small rounding difference
- The error is isolated and easily fixable
- Example: Correct integration but wrong final arithmetic; correct formula but sign error

**"partial"** - Partially correct with significant gaps:
- Shows some genuine understanding of the problem
- Has valid elements but major portions missing or incorrect
- Right idea started but poorly executed or incomplete
- Missing key steps, components, or final answers
- Example: Correct setup but wrong execution; partial solution with gaps

**"incorrect"** - Fundamentally wrong:
- Wrong answer with no valid mathematical reasoning
- Misunderstands the problem or applies wrong concepts
- Answer does not address the question asked
- No meaningful progress or valid mathematical work shown

## Few-Shot Examples:
{FEW_SHOT_EXAMPLES}

## Response Format - CRITICAL:
You must respond with EXACTLY ONE of these four labels in JSON format:
<json>
{{
    "response": "correct|almost|partial|incorrect"
}}
</json>

CRITICAL INSTRUCTIONS:
1. Your response field must contain ONLY one of these four exact words: correct, almost, partial, or incorrect
2. Do not include explanations, reasoning, or any other text in the JSON
3. You MUST use "almost" for answers with minor errors (the model currently underuses this label)
4. You MUST use "partial" for incomplete but partially valid answers (the model currently underuses this label)
5. Be conservative with "correct" - only use when fully confident
6. Follow the decision framework above step by step
7. Double-check your label choice before responding"""

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
