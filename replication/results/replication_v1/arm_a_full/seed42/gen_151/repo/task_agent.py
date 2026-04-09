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
    
    # Check for partial matches - be conservative (prefer lower grades when ambiguous)
    # Check for "almost" first (most specific)
    if "almost" in cleaned:
        return "almost"
    # Check for "partial"
    if "partial" in cleaned:
        return "partial"
    # Check for "incorrect" or "wrong"
    if "incorrect" in cleaned or "wrong" in cleaned:
        return "incorrect"
    # Be VERY conservative with "correct" - only if explicitly stated and no other labels found
    if cleaned == "correct":
        return "correct"
    if "correct" in cleaned:
        # Avoid matching "not correct", "partially correct", "mostly correct", etc.
        if "not correct" in cleaned or "partially" in cleaned or "mostly" in cleaned:
            return "partial"
        # If it contains "correct" but isn't exact, be conservative and use "almost"
        return "almost"
    
    # Default to incorrect if no match found (conservative default)
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
    
    # Look for explicit labels in JSON-like format first (most reliable)
    if "\"almost\"" in text_lower or "'almost'" in text_lower:
        return "almost"
    if "\"partial\"" in text_lower or "'partial'" in text_lower:
        return "partial"
    if "\"incorrect\"" in text_lower or "'incorrect'" in text_lower:
        return "incorrect"
    # Be conservative with "correct" in JSON format - check for exact match
    if '"response": "correct"' in text_lower or "'response': 'correct'" in text_lower:
        return "correct"
    
    # Check for keywords in the text - be conservative (prefer lower grades)
    # Check for "almost" first (most specific)
    if "almost" in text_lower:
        return "almost"
    # Check for "partial" 
    if "partial" in text_lower:
        return "partial"
    # Check for "incorrect" or "wrong"
    if "incorrect" in text_lower or "wrong" in text_lower:
        return "incorrect"
    # Be VERY conservative with "correct" - only if explicitly stated as exact word
    # and no other labels found
    if " correct " in text_lower or text_lower.strip() == "correct":
        # Make sure we're not matching "incorrect" or "partially correct" etc.
        if "not correct" in text_lower or "incorrect" in text_lower or "partially" in text_lower:
            return "partial"
        return "correct"
    
    # Default to incorrect when uncertain
    logger.warning(f"Could not extract label from text, defaulting to 'incorrect'")
    return "incorrect"


# Few-shot examples for better grading accuracy
FEW_SHOT_EXAMPLES = """
Example 1 - CORRECT (Fully correct and complete):
Problem: Find the sum of 2 + 3.
Solution: The sum of 2 + 3 is 5.
Student Answer: The sum of 2 + 3 equals 5 because when we combine 2 items with 3 items, we get 5 items total. This can be verified: 2 + 3 = 5.
<json>
{"response": "correct"}
</json>

Example 2 - ALMOST (Correct answer but minor gaps in justification):
Problem: Find the derivative of x^2.
Solution: The derivative is 2x.
Student Answer: The derivative is 2x. [Correct answer but doesn't show the power rule application]
<json>
{"response": "almost"}
</json>

Example 3 - ALMOST (Correct approach but missing trivial details):
Problem: Prove that for all positive integers n, n^3 - n is divisible by 6.
Solution: Factor as n(n-1)(n+1), product of three consecutive integers, so divisible by both 2 and 3.
Student Answer: n^3 - n = n(n^2-1) = n(n-1)(n+1). Among any three consecutive integers, one is divisible by 2 and one by 3, so the product is divisible by 6. [Correct proof but doesn't explicitly state that 2 and 3 are coprime]
<json>
{"response": "almost"}
</json>

Example 4 - PARTIAL (Partially correct but significant gaps):
Problem: Solve x^2 = 4 for x.
Solution: x = 2 or x = -2.
Student Answer: x = 2
<json>
{"response": "partial"}
</json>

Example 5 - INCORRECT (Wrong or does not address the problem):
Problem: Prove that the sum of angles in a triangle is 180 degrees.
Solution: [Detailed geometric proof]
Student Answer: It's 180 because that's what I learned.
<json>
{"response": "incorrect"}
</json>

Example 6 - INCORRECT (Major conceptual error):
Problem: Find the integral of x dx.
Solution: x^2/2 + C
Student Answer: x^2 + C
<json>
{"response": "incorrect"}
</json>

Example 7 - CORRECT vs ALMOST distinction:
Problem: Prove that sqrt(2) is irrational.
Solution: Assume sqrt(2) = p/q in lowest terms, then 2q^2 = p^2, so p^2 even, so p even, so p=2k, then 2q^2=4k^2, so q^2=2k^2, so q even, contradiction.

Student A (CORRECT - complete proof with all steps):
Assume sqrt(2) is rational, so sqrt(2) = p/q where p,q are coprime integers. Squaring: 2 = p^2/q^2, so 2q^2 = p^2. Thus p^2 is even, so p is even (if p were odd, p^2 would be odd). Write p = 2k. Then 2q^2 = 4k^2, so q^2 = 2k^2. Thus q^2 is even, so q is even. But then p and q are both even, contradicting coprimality. Therefore sqrt(2) is irrational.
<json>
{"response": "correct"}
</json>

Student B (ALMOST - correct but skips some routine justification):
Assume sqrt(2) = p/q in lowest terms. Then 2q^2 = p^2, so p is even. Let p = 2k, then q^2 = 2k^2, so q is even. Contradiction.
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
        
        instruction = f"""You are an expert grading agent for {domain}. Your task is to carefully analyze the student's answer and provide a detailed evaluation.

## Problem Statement:
{problem}

## Correct Solution:
{solution}

## Grading Guidelines:
{grading_guidelines}

## Student's Answer:
{student_answer}

## Grading Criteria - BE STRICT:

**CORRECT**: Use ONLY when the answer is:
- Fully correct and complete with NO gaps or omissions
- Contains ALL required steps, reasoning, and justification
- Matches the correct solution in substance, conclusion, AND completeness
- Has a valid proof/derivation with no logical gaps
- Final answer is correct AND properly justified
- NO missing cases, steps, or components

**ALMOST**: Use when the answer is:
- Has the right approach and correct final answer
- Nearly complete but has MINOR gaps that don't affect the main result
- Missing only INSIGNIFICANT details (e.g., minor algebraic simplifications, obvious edge cases)
- Contains trivial errors (minor notation issues, small typos that don't affect correctness)
- The core proof/argument is sound but could be more polished
- Key insight is present but some routine details are skipped

**PARTIAL**: Use when the answer is:
- Shows some correct work or understanding of the problem
- Has the right idea but incomplete or flawed execution
- Missing KEY steps, cases, or components that affect the solution
- Contains some correct reasoning but also SIGNIFICANT errors or gaps
- Partial progress toward solution but not close to complete

**INCORRECT**: Use when the answer is:
- Wrong or does not address the problem
- Contains major conceptual errors
- Has fundamentally flawed reasoning
- Gives an incorrect final answer with no redeeming correct work
- Merely states a conclusion without justification

## Critical Distinction: CORRECT vs ALMOST
The key difference is COMPLETENESS:
- CORRECT = Complete solution with all steps justified, no gaps
- ALMOST = Correct answer and approach, but minor gaps in justification or missing trivial details

When in doubt between CORRECT and ALMOST, choose ALMOST unless the solution is truly complete.

## Few-Shot Examples:
{FEW_SHOT_EXAMPLES}

## Response Format:
You must respond with EXACTLY ONE of these four labels in JSON format:
<json>
{{
    "response": "correct|almost|partial|incorrect"
}}
</json>

IMPORTANT RULES:
1. Your response field must contain ONLY one of these four exact words: correct, almost, partial, or incorrect
2. Be STRICT - when in doubt between two labels, choose the more conservative (lower) grade
3. "correct" should be RARE - only for TRULY COMPLETE solutions with ALL steps justified
4. "almost" is for answers with correct approach and answer but MINOR gaps in justification
5. "partial" requires some genuine correct work, not just attempting the problem
6. "incorrect" is for answers that are fundamentally wrong or lack valid reasoning
7. KEY DISTINCTION: If the solution skips steps that need justification, use "almost" not "correct"
8. When deciding between CORRECT and ALMOST: If you can identify ANY missing justification, use ALMOST"""

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
