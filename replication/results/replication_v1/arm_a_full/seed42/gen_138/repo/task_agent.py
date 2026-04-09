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
    
    # Check for exact match with quotes
    for label in valid_labels:
        if f'"{label}"' in cleaned or f"'{label}'" in cleaned:
            return label
    
    # Check for negations first (be conservative)
    if "not correct" in cleaned or "not almost" in cleaned or "not partial" in cleaned:
        return "incorrect"
    
    # Check for "partially correct" - should be partial
    if "partially correct" in cleaned:
        return "partial"
    
    # Check for "almost correct" - should be almost
    if "almost correct" in cleaned:
        return "almost"
    
    # Check for "mostly correct" - should be partial (significant but not complete)
    if "mostly correct" in cleaned:
        return "partial"
    
    # Check for partial matches - be conservative (prefer lower grades when ambiguous)
    # Check for "incorrect" first (most conservative)
    if "incorrect" in cleaned:
        return "incorrect"
    # Check for "wrong"
    if "wrong" in cleaned:
        return "incorrect"
    # Check for "almost" (more specific than partial)
    if "almost" in cleaned:
        return "almost"
    # Check for "partial"
    if "partial" in cleaned:
        return "partial"
    # Only return "correct" if explicitly stated and no other labels found
    if "correct" in cleaned:
        return "correct"
    
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
    if '"almost"' in text_lower or "'almost'" in text_lower:
        return "almost"
    if '"partial"' in text_lower or "'partial'" in text_lower:
        return "partial"
    if '"incorrect"' in text_lower or "'incorrect'" in text_lower:
        return "incorrect"
    if '"correct"' in text_lower or "'correct'" in text_lower:
        return "correct"
    
    # Check for negations first (be conservative)
    if "not correct" in text_lower or "not almost" in text_lower:
        return "incorrect"
    
    # Check for compound phrases
    if "partially correct" in text_lower:
        return "partial"
    if "almost correct" in text_lower:
        return "almost"
    if "mostly correct" in text_lower:
        return "partial"
    
    # Check for keywords in the text - be conservative (prefer lower grades)
    # Check for "incorrect" first (most conservative)
    if "incorrect" in text_lower:
        return "incorrect"
    # Check for "wrong"
    if "wrong" in text_lower:
        return "incorrect"
    # Check for "almost" (more specific than partial)
    if "almost" in text_lower:
        return "almost"
    # Check for "partial" 
    if "partial" in text_lower:
        return "partial"
    # Only return "correct" if it's explicitly stated and no other labels found
    if "correct" in text_lower:
        return "correct"
    
    # Default to incorrect when uncertain
    logger.warning(f"Could not extract label from text, defaulting to 'incorrect'")
    return "incorrect"


# Few-shot examples for better grading accuracy
FEW_SHOT_EXAMPLES = """
Example 1 - CORRECT (Fully correct and complete):
Problem: Find the sum of 2 + 3.
Solution: The sum of 2 + 3 is 5.
Student Answer: 5
<json>
{"response": "correct"}
</json>

Example 2 - ALMOST (Nearly correct, minor issues only):
Problem: Find the derivative of x^2.
Solution: The derivative is 2x.
Student Answer: The derivative is 2x, using the power rule where d/dx(x^n) = n*x^(n-1), but the student forgot to mention the constant C in the general solution.
<json>
{"response": "almost"}
</json>

Example 3 - PARTIAL (Partially correct but significant gaps):
Problem: Solve x^2 = 4 for x.
Solution: x = 2 or x = -2.
Student Answer: x = 2
<json>
{"response": "partial"}
</json>

Example 4 - INCORRECT (Wrong or does not address the problem):
Problem: Prove that the sum of angles in a triangle is 180 degrees.
Solution: [Detailed geometric proof]
Student Answer: It's 180 because that's what I learned.
<json>
{"response": "incorrect"}
</json>

Example 5 - INCORRECT (Major conceptual error):
Problem: Find the integral of x dx.
Solution: x^2/2 + C
Student Answer: x^2 + C
<json>
{"response": "incorrect"}
</json>

Example 6 - PARTIAL (Some correct work but incomplete):
Problem: Solve the system: x + y = 5, x - y = 1.
Solution: x = 3, y = 2
Student Answer: From the first equation, x = 5 - y.
<json>
{"response": "partial"}
</json>

Example 7 - ALMOST (Correct answer with minor notation issue):
Problem: Evaluate lim(x→0) sin(x)/x.
Solution: The limit equals 1 by L'Hopital's rule or the standard limit.
Student Answer: The limit is 1. Using L'Hopital: lim(x→0) cos(x)/1 = 1.
<json>
{"response": "almost"}
</json>

Example 8 - INCORRECT (Correct final answer but wrong reasoning):
Problem: Prove that for any positive integer n, n^3 - n is divisible by 6.
Solution: Factor as n(n-1)(n+1), which is product of 3 consecutive integers, hence divisible by 6.
Student Answer: n^3 - n = 6k for some k. This is true for n=1,2,3. By induction, assume true for n, then for n+1: (n+1)^3 - (n+1) = n^3 + 3n^2 + 3n + 1 - n - 1 = n^3 - n + 3n^2 + 3n = 6k + 3n(n+1). Since n(n+1) is even, 3n(n+1) is divisible by 6. So true for n+1.
<json>
{"response": "incorrect"}
</json>

Example 9 - PARTIAL (Correct approach but missing key case):
Problem: Find all real solutions to |x-1| = 2.
Solution: x-1 = 2 or x-1 = -2, so x = 3 or x = -1.
Student Answer: x-1 = 2, so x = 3.
<json>
{"response": "partial"}
</json>

Example 10 - ALMOST (Complete solution with trivial typo):
Problem: Solve 2x + 4 = 10.
Solution: 2x = 6, so x = 3.
Student Answer: 2x = 6, therefore x = 3. (Note: student wrote "therefore" instead of "so")
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

## Grading Criteria - BE STRICT AND CONSERVATIVE:

**CORRECT**: Use ONLY when the answer is:
- Fully correct and complete
- Contains all required steps and reasoning
- Matches the correct solution in substance and conclusion
- No errors, omissions, or significant gaps
- Would receive full marks in a real exam

**ALMOST**: Use when the answer is:
- Nearly correct with only minor issues
- Has the right approach and correct final answer
- Contains only trivial errors (e.g., minor notation issues, small typos that don't affect correctness)
- Missing only insignificant details that don't impact the solution's validity
- Would receive nearly full marks (minor deductions only)

**PARTIAL**: Use when the answer is:
- Partially correct but has significant gaps
- Shows some correct work or understanding
- Missing key steps, cases, or components
- Has the right idea but incomplete execution
- Contains some correct reasoning but also significant errors
- Would receive partial credit (some points but not most)

**INCORRECT**: Use when the answer is:
- Wrong or does not address the problem
- Contains major conceptual errors
- Has fundamentally flawed reasoning
- Gives an incorrect final answer with no redeeming correct work
- Merely states a conclusion without justification
- Would receive little or no credit

## Decision Tree for Grading:
1. Is the final answer wrong AND the reasoning fundamentally flawed? → INCORRECT
2. Is the answer complete with correct final result but has only trivial issues? → ALMOST
3. Does the answer show some correct work but miss key parts? → PARTIAL
4. Is the answer fully complete, correct, and rigorous? → CORRECT

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
3. "correct" should be rare - only for truly complete and accurate answers
4. "almost" is for minor issues only - not for missing significant steps
5. "partial" requires some genuine correct work, not just attempting the problem
6. "incorrect" is for answers that are fundamentally wrong or lack valid reasoning
7. If the answer has correct final answer but flawed reasoning, it's INCORRECT (not ALMOST)
8. If the answer is mostly correct but missing a key case or step, it's PARTIAL (not ALMOST)
9. When uncertain, prefer: incorrect > partial > almost > correct"""

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
