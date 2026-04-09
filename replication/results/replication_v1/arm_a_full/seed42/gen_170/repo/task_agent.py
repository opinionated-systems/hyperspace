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
    
    # Check for "nearly correct" - should be almost
    if "nearly correct" in cleaned:
        return "almost"
    
    # Check for "mostly right" - should be partial
    if "mostly right" in cleaned:
        return "partial"
    
    # Check for "partially right" - should be partial
    if "partially right" in cleaned:
        return "partial"
    
    # Check for "some correct" - should be partial
    if "some correct" in cleaned or "partly correct" in cleaned:
        return "partial"
    
    # Check for "incomplete" - should be partial
    if "incomplete" in cleaned:
        return "partial"
    
    # Check for "has errors" - should be partial or incorrect depending on severity
    if "has errors" in cleaned or "contains errors" in cleaned:
        return "partial"
    
    # Check for "flawed" - should be incorrect
    if "flawed" in cleaned or "fundamentally" in cleaned:
        return "incorrect"
    
    # Check for "vague" or "hand-waving" - should be incorrect
    if "vague" in cleaned or "hand-waving" in cleaned or "no justification" in cleaned:
        return "incorrect"
    
    # Check for "wrong reasoning" - should be incorrect
    if "wrong reasoning" in cleaned or "incorrect reasoning" in cleaned:
        return "incorrect"
    
    # Check for partial matches - be conservative (prefer lower grades when ambiguous)
    # Check for "incorrect" first (most conservative)
    if "incorrect" in cleaned:
        return "incorrect"
    # Check for "wrong"
    if "wrong" in cleaned:
        return "incorrect"
    # Check for "error" (but not "no error")
    if "error" in cleaned and "no error" not in cleaned:
        return "incorrect"
    # Check for "invalid"
    if "invalid" in cleaned:
        return "incorrect"
    # Check for "fail"
    if "fail" in cleaned:
        return "incorrect"
    # Check for "almost" (more specific than partial)
    if "almost" in cleaned:
        return "almost"
    # Check for "partial"
    if "partial" in cleaned:
        return "partial"
    # Check for "part"
    if "part" in cleaned:
        return "partial"
    # Only return "correct" if explicitly stated and no other labels found
    if "correct" in cleaned:
        return "correct"
    # Check for "right"
    if "right" in cleaned:
        return "correct"
    # Check for "complete"
    if "complete" in cleaned and "incomplete" not in cleaned:
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
    if "not correct" in text_lower or "not almost" in text_lower or "not partial" in text_lower:
        return "incorrect"
    
    # Check for compound phrases - use _normalize_prediction logic for consistency
    if "partially correct" in text_lower or "partly correct" in text_lower or "partially right" in text_lower:
        return "partial"
    if "almost correct" in text_lower or "nearly correct" in text_lower:
        return "almost"
    if "mostly correct" in text_lower or "mostly right" in text_lower:
        return "partial"
    if "some correct" in text_lower:
        return "partial"
    
    # Check for indicators of quality
    if "incomplete" in text_lower:
        return "partial"
    if "has errors" in text_lower or "contains errors" in text_lower:
        return "partial"
    if "flawed" in text_lower or "fundamentally" in text_lower:
        return "incorrect"
    if "vague" in text_lower or "hand-waving" in text_lower or "no justification" in text_lower:
        return "incorrect"
    if "wrong reasoning" in text_lower or "incorrect reasoning" in text_lower:
        return "incorrect"
    
    # Check for keywords in the text - be conservative (prefer lower grades)
    # Check for "incorrect" first (most conservative)
    if "incorrect" in text_lower:
        return "incorrect"
    # Check for "wrong"
    if "wrong" in text_lower:
        return "incorrect"
    # Check for "error" (but not "no error")
    if "error" in text_lower and "no error" not in text_lower:
        return "incorrect"
    # Check for "invalid"
    if "invalid" in text_lower:
        return "incorrect"
    # Check for "fail"
    if "fail" in text_lower:
        return "incorrect"
    # Check for "almost" (more specific than partial)
    if "almost" in text_lower:
        return "almost"
    # Check for "partial" 
    if "partial" in text_lower:
        return "partial"
    # Check for "part"
    if "part" in text_lower:
        return "partial"
    # Only return "correct" if it's explicitly stated and no other labels found
    if "correct" in text_lower:
        return "correct"
    # Check for "right"
    if "right" in text_lower:
        return "correct"
    # Check for "complete"
    if "complete" in text_lower and "incomplete" not in text_lower:
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

Example 11 - INCORRECT (No valid reasoning, just stating conclusion):
Problem: Prove that the product of two consecutive integers is even.
Solution: Let n be an integer. Then n(n+1) must be even because one of n or n+1 is even.
Student Answer: The product is even because one number is always even.
<json>
{"response": "incorrect"}
</json>

Example 12 - INCORRECT (Fundamentally flawed approach):
Problem: Find the maximum value of f(x) = -x^2 + 4x.
Solution: f'(x) = -2x + 4 = 0, so x = 2. f(2) = -4 + 8 = 4. Maximum is 4.
Student Answer: The maximum is at x = 0, f(0) = 0.
<json>
{"response": "incorrect"}
</json>

Example 13 - PARTIAL vs INCORRECT boundary (Has some correct intuition but execution is wrong):
Problem: Prove that sqrt(2) is irrational.
Solution: Assume sqrt(2) = p/q in lowest terms. Then 2q^2 = p^2, so p^2 is even, so p is even. Let p = 2k. Then 2q^2 = 4k^2, so q^2 = 2k^2, so q is even. Contradiction since p/q was in lowest terms.
Student Answer: Assume sqrt(2) = p/q. Then square both sides to get 2 = p^2/q^2. This means p^2 = 2q^2.
<json>
{"response": "partial"}
</json>

Example 14 - INCORRECT (No substantive correct work):
Problem: Solve the equation 2^x = 8.
Solution: 2^x = 2^3, so x = 3.
Student Answer: x = 4 because 2*4 = 8.
<json>
{"response": "incorrect"}
</json>

Example 15 - PARTIAL vs ALMOST boundary (Missing significant step, not just trivial issue):
Problem: Find the area of a circle with radius 5.
Solution: Area = πr^2 = π(5)^2 = 25π.
Student Answer: Area = π * 5^2. (Did not compute final value)
<json>
{"response": "partial"}
</json>

Example 16 - INCORRECT (Attempted but completely wrong approach):
Problem: Find the sum of the first 10 positive integers.
Solution: S = n(n+1)/2 = 10*11/2 = 55.
Student Answer: 1+2+3+4+5+6+7+8+9+10 = 100.
<json>
{"response": "incorrect"}
</json>

Example 17 - PARTIAL (Correct setup but calculation error):
Problem: Compute 15 * 12.
Solution: 15 * 12 = 180.
Student Answer: 15 * 12 = 15 * (10 + 2) = 150 + 20 = 170.
<json>
{"response": "partial"}
</json>

Example 18 - INCORRECT (Misunderstood the problem):
Problem: Find the derivative of sin(x) at x = π/2.
Solution: d/dx(sin(x)) = cos(x). At x = π/2, cos(π/2) = 0.
Student Answer: sin(π/2) = 1.
<json>
{"response": "incorrect"}
</json>

Example 19 - PARTIAL vs INCORRECT (Some correct elements but major gaps):
Problem: Prove by induction that 1 + 2 + ... + n = n(n+1)/2.
Solution: [Standard induction proof]
Student Answer: Base case: 1 = 1(2)/2 = 1. For induction, assume true for n, then 1+2+...+n+(n+1) = n(n+1)/2 + (n+1).
<json>
{"response": "partial"}
</json>

Example 20 - INCORRECT (Vague hand-waving without mathematical substance):
Problem: Show that every prime greater than 3 is of the form 6k±1.
Solution: Any integer is of form 6k, 6k+1, 6k+2, 6k+3, 6k+4, or 6k+5. Forms 6k, 6k+2, 6k+4 are even. Form 6k+3 is divisible by 3. So primes > 3 must be 6k+1 or 6k+5 = 6(k+1)-1.
Student Answer: Primes are odd and not divisible by 3, so they must be near multiples of 6.
<json>
{"response": "incorrect"}
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
- This should be RARE - only for truly exceptional answers

**ALMOST**: Use when the answer is:
- Nearly correct with only minor issues
- Has the right approach and correct final answer
- Contains only trivial errors (e.g., minor notation issues, small typos that don't affect correctness)
- Missing only insignificant details that don't impact the solution's validity
- Would receive nearly full marks (minor deductions only)
- The errors must be TRULY trivial - not missing steps, not wrong reasoning

**PARTIAL**: Use when the answer is:
- Partially correct but has significant gaps
- Shows some correct work or understanding
- Missing key steps, cases, or components
- Has the right idea but incomplete execution
- Contains some correct reasoning but also significant errors
- Would receive partial credit (some points but not most)
- Has valid mathematical content but incomplete or with errors

**INCORRECT**: Use when the answer is:
- Wrong or does not address the problem
- Contains major conceptual errors
- Has fundamentally flawed reasoning
- Gives an incorrect final answer with no redeeming correct work
- Merely states a conclusion without justification
- Would receive little or no credit
- Vague hand-waving without mathematical substance
- No valid reasoning provided

## Critical Boundary Guidelines:

**INCORRECT vs PARTIAL boundary:**
- If the answer has NO substantive correct mathematical work → INCORRECT
- If the answer merely states conclusions without reasoning → INCORRECT
- If the answer shows some correct setup or valid reasoning (even if incomplete) → PARTIAL
- "Partial" requires GENUINE correct work, not just attempting the problem

**PARTIAL vs ALMOST boundary:**
- If missing a key case, step, or component → PARTIAL (not ALMOST)
- If the final answer is correct but reasoning has gaps → PARTIAL (not ALMOST)
- If only trivial notation issues or typos → ALMOST
- If missing significant work but has correct approach → PARTIAL

**ALMOST vs CORRECT boundary:**
- If any non-trivial step is missing → ALMOST (not CORRECT)
- If the solution is complete except for truly trivial issues → ALMOST
- Only award CORRECT for truly complete, rigorous solutions

**Correct final answer with wrong reasoning:**
- This is ALWAYS INCORRECT (not ALMOST, not PARTIAL)
- Correct answer by luck with flawed reasoning gets no credit

## Decision Tree for Grading:
1. Does the answer contain NO valid reasoning or correct work? → INCORRECT
2. Is the final answer correct but reasoning is fundamentally flawed? → INCORRECT
3. Does the answer show some correct work but has significant gaps/errors? → PARTIAL
4. Is the answer complete with correct result but has only trivial issues? → ALMOST
5. Is the answer fully complete, correct, and rigorous? → CORRECT

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
2. Be STRICT and CONSERVATIVE - when in doubt, choose the LOWER grade
3. "correct" should be rare - only for truly complete and accurate answers
4. "almost" is for truly trivial issues only - not for missing significant steps
5. "partial" requires genuine correct work with significant gaps - not just attempting
6. "incorrect" is for answers with no valid reasoning or fundamentally flawed work
7. If the answer has correct final answer but flawed reasoning, it's INCORRECT
8. If the answer is mostly correct but missing a key case or step, it's PARTIAL
9. When uncertain, prefer: incorrect > partial > almost > correct
10. Remember: A correct answer with wrong reasoning is INCORRECT
11. Vague statements without mathematical substance are INCORRECT
12. Only award PARTIAL if there is GENUINE correct mathematical content"""

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
