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
    if not prediction or not isinstance(prediction, str):
        return "incorrect"
    
    # Clean up the prediction
    cleaned = prediction.strip().lower()
    
    # Remove any punctuation, quotes, or extra whitespace
    cleaned = cleaned.strip(".!?,:;\"'[]{}()").strip()
    
    # Map to valid labels
    valid_labels = ["correct", "almost", "partial", "incorrect"]
    
    # Check for exact match first
    if cleaned in valid_labels:
        return cleaned
    
    # Check for exact match with quotes (handle various quote styles)
    for label in valid_labels:
        if f'"{label}"' in cleaned or f"'{label}'" in cleaned:
            return label
        # Check if the cleaned string equals the label when quotes are removed
        if cleaned.replace('"', '').replace("'", "") == label:
            return label
    
    # Check for negations first (be conservative)
    negation_patterns = [
        "not correct", "not almost", "not partial", 
        "isn't correct", "isn't almost", "isn't partial",
        "not a correct", "not an almost", "not a partial"
    ]
    if any(pattern in cleaned for pattern in negation_patterns):
        return "incorrect"
    
    # Check for compound phrases that indicate specific labels
    compound_patterns = {
        "partially correct": "partial",
        "almost correct": "almost", 
        "mostly correct": "partial",
        "nearly correct": "almost",
        "partly correct": "partial",
        "somewhat correct": "partial",
        "fully correct": "correct",
        "completely correct": "correct",
        "entirely correct": "correct",
        "totally correct": "correct",
        "absolutely correct": "correct",
        "mostly wrong": "incorrect",
        "completely wrong": "incorrect",
        "totally wrong": "incorrect",
        "entirely wrong": "incorrect",
        "fundamentally wrong": "incorrect",
    }
    
    for pattern, label in compound_patterns.items():
        if pattern in cleaned:
            return label
    
    # Check for partial matches - be conservative (prefer lower grades when ambiguous)
    # Priority order: incorrect > partial > almost > correct
    
    # Check for "incorrect" indicators first (most conservative)
    incorrect_indicators = ["incorrect", "wrong", "error", "mistake", "invalid", "false"]
    for indicator in incorrect_indicators:
        if indicator in cleaned:
            return "incorrect"
    
    # Check for "partial" indicators
    partial_indicators = ["partial", "partly", "incomplete", "missing", "some", "half"]
    for indicator in partial_indicators:
        if indicator in cleaned:
            return "partial"
    
    # Check for "almost" indicators
    almost_indicators = ["almost", "nearly", "close", "minor", "trivial", "small error"]
    for indicator in almost_indicators:
        if indicator in cleaned:
            return "almost"
    
    # Check for "correct" indicators - only if no other indicators found
    correct_indicators = ["correct", "right", "accurate", "true", "valid"]
    for indicator in correct_indicators:
        if indicator in cleaned:
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
    if not text or not isinstance(text, str):
        return "incorrect"
    
    text_lower = text.lower()
    
    # Look for explicit labels in quotes - these are strong signals
    quote_patterns = [
        ('"correct"', "correct"), ("'correct'", "correct"),
        ('"almost"', "almost"), ("'almost'", "almost"),
        ('"partial"', "partial"), ("'partial'", "partial"),
        ('"incorrect"', "incorrect"), ("'incorrect'", "incorrect"),
    ]
    
    for pattern, label in quote_patterns:
        if pattern in text_lower:
            return label
    
    # Look for "response:" or similar patterns followed by a label
    import re
    response_patterns = [
        r'response["\']?\s*[:=]\s*["\']?(correct|almost|partial|incorrect)["\']?',
        r'grade["\']?\s*[:=]\s*["\']?(correct|almost|partial|incorrect)["\']?',
        r'label["\']?\s*[:=]\s*["\']?(correct|almost|partial|incorrect)["\']?',
        r'evaluation["\']?\s*[:=]\s*["\']?(correct|almost|partial|incorrect)["\']?',
    ]
    
    for pattern in response_patterns:
        match = re.search(pattern, text_lower)
        if match:
            return match.group(1)
    
    # Check for compound phrases first (more specific)
    compound_patterns = [
        ("partially correct", "partial"),
        ("almost correct", "almost"),
        ("mostly correct", "partial"),
        ("nearly correct", "almost"),
        ("partly correct", "partial"),
        ("somewhat correct", "partial"),
        ("fully correct", "correct"),
        ("completely correct", "correct"),
        ("entirely correct", "correct"),
        ("totally correct", "correct"),
        ("mostly wrong", "incorrect"),
        ("completely wrong", "incorrect"),
        ("totally wrong", "incorrect"),
        ("entirely wrong", "incorrect"),
        ("fundamentally wrong", "incorrect"),
    ]
    
    for pattern, label in compound_patterns:
        if pattern in text_lower:
            return label
    
    # Check for individual keywords - be conservative (prefer lower grades)
    # Priority: incorrect > partial > almost > correct
    
    incorrect_indicators = ["incorrect", "wrong", "error", "mistake", "invalid", "false", "not correct"]
    for indicator in incorrect_indicators:
        if indicator in text_lower:
            return "incorrect"
    
    partial_indicators = ["partial", "partly", "incomplete", "missing"]
    for indicator in partial_indicators:
        if indicator in text_lower:
            return "partial"
    
    almost_indicators = ["almost", "nearly", "close", "minor"]
    for indicator in almost_indicators:
        if indicator in text_lower:
            return "almost"
    
    correct_indicators = ["correct", "right", "accurate", "true", "valid"]
    for indicator in correct_indicators:
        if indicator in text_lower:
            return "correct"
    
    # Default to incorrect
    logger.warning(f"Could not extract label from text, defaulting to 'incorrect'")
    return "incorrect"


# Few-shot examples for better grading accuracy
FEW_SHOT_EXAMPLES = """
Example 1 - CORRECT (complete and rigorous):
Problem: Find the sum of 2 + 3.
Solution: The sum of 2 + 3 is 5.
Student Answer: 5
<json>
{"response": "correct"}
</json>

Example 2 - PARTIAL (incomplete solution, missing one root):
Problem: Solve x^2 = 4 for x.
Solution: x = 2 or x = -2.
Student Answer: x = 2
<json>
{"response": "partial"}
</json>

Example 3 - INCORRECT (no valid reasoning, just states conclusion):
Problem: Prove that the sum of angles in a triangle is 180 degrees.
Solution: [Detailed geometric proof with parallel lines and alternate angles]
Student Answer: It's 180 because that's what I learned.
<json>
{"response": "incorrect"}
</json>

Example 4 - ALMOST (minor notation error, correct reasoning and answer):
Problem: Find the derivative of f(x) = x^2.
Solution: f'(x) = 2x
Student Answer: f'(x) = 2x (used f' notation instead of df/dx, but correct)
<json>
{"response": "almost"}
</json>

Example 5 - CORRECT (complete proof with all steps justified):
Problem: Prove that for any positive integer n, n^3 - n is divisible by 6.
Solution: Factor as n(n-1)(n+1). Among three consecutive integers, one is divisible by 2 and one by 3, so product is divisible by 6.
Student Answer: We can write n^3 - n = n(n^2 - 1) = n(n-1)(n+1). These are three consecutive integers. In any three consecutive integers, there must be one divisible by 2 and one divisible by 3. Therefore the product is divisible by 6.
<json>
{"response": "correct"}
</json>

Example 6 - PARTIAL (correct approach but calculation error and incomplete):
Problem: Find all real solutions to x^2 - 5x + 6 = 0.
Solution: Factor as (x-2)(x-3) = 0, so x = 2 or x = 3.
Student Answer: Using the quadratic formula, x = (5 ± sqrt(25-24))/2 = (5 ± 1)/2, so x = 3.
<json>
{"response": "partial"}
</json>

Example 7 - INCORRECT (correct final answer but fundamentally wrong reasoning):
Problem: Prove that the product of two odd numbers is odd.
Solution: Let the numbers be 2k+1 and 2m+1. Their product is (2k+1)(2m+1) = 4km + 2k + 2m + 1 = 2(2km + k + m) + 1, which is odd.
Student Answer: 3 × 5 = 15, which is odd. So the product of two odd numbers is odd.
<json>
{"response": "incorrect"}
</json>

Example 8 - ALMOST (correct answer with trivial arithmetic slip):
Problem: Compute 12 × 8.
Solution: 96
Student Answer: 12 × 8 = 90 + 6 = 96
<json>
{"response": "almost"}
</json>

Example 9 - PARTIAL (shows understanding but missing key justification):
Problem: Prove that sqrt(2) is irrational.
Solution: [Standard proof by contradiction assuming sqrt(2) = p/q in lowest terms]
Student Answer: Assume sqrt(2) = p/q. Then 2q^2 = p^2, so p^2 is even, thus p is even. Let p = 2k. Then 2q^2 = 4k^2, so q^2 = 2k^2, making q even too. This contradicts p/q being in lowest terms.
<json>
{"response": "partial"}
</json>

Example 10 - INCORRECT (completely wrong approach):
Problem: Find the area of a circle with radius 3.
Solution: A = πr^2 = 9π
Student Answer: A = 2πr = 6π (used circumference formula instead of area)
<json>
{"response": "incorrect"}
</json>

Example 11 - CORRECT (complete solution with verification):
Problem: Solve the system: x + y = 5, x - y = 1.
Solution: Adding equations: 2x = 6, so x = 3. Then y = 2.
Student Answer: Adding the equations: 2x = 6, so x = 3. Substituting back: 3 + y = 5, so y = 2. Check: 3 + 2 = 5 ✓ and 3 - 2 = 1 ✓
<json>
{"response": "correct"}
</json>

Example 12 - PARTIAL (right idea but significant gaps in reasoning):
Problem: Prove by induction that 1 + 2 + ... + n = n(n+1)/2.
Solution: [Base case n=1, inductive step assuming true for n, proving for n+1]
Student Answer: For n=1: 1 = 1(2)/2 = 1. Assume true for n. Then 1 + 2 + ... + n + (n+1) = n(n+1)/2 + (n+1) = (n+1)(n/2 + 1) = (n+1)(n+2)/2.
<json>
{"response": "partial"}
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
- Fully correct and complete with ALL required steps
- Contains rigorous reasoning that matches the correct solution
- No errors, omissions, or gaps of any kind
- Would receive full marks in a real exam
- Has correct final answer AND correct reasoning throughout

**ALMOST**: Use ONLY when the answer is:
- Nearly perfect with only trivial/minor issues
- Has correct final answer and correct reasoning
- Contains only insignificant errors (e.g., minor notation, trivial typos that don't affect correctness)
- Missing only cosmetic details that don't impact validity
- Would receive nearly full marks (minor deductions only)
- NOT for missing steps or cases - that's PARTIAL

**PARTIAL**: Use when the answer shows:
- Some genuine correct work or understanding
- Right general approach but incomplete execution
- Missing key steps, cases, or components
- Has correct elements mixed with significant errors
- Would receive partial credit (some points but less than half)
- NOT for trivial issues - that's ALMOST

**INCORRECT**: Use when the answer is:
- Fundamentally wrong or does not address the problem
- Contains major conceptual errors or flawed reasoning
- Gives incorrect final answer with no valid work shown
- Merely states a conclusion without any justification
- Has correct final answer but wrong reasoning (this is critical!)
- Would receive little or no credit

## Critical Distinctions:

**CORRECT vs ALMOST**: 
- CORRECT = perfect or near-perfect with no meaningful flaws
- ALMOST = correct answer with only trivial cosmetic issues

**ALMOST vs PARTIAL**:
- ALMOST = correct answer, minor issues only
- PARTIAL = missing key parts or has significant errors, even if some work is right

**PARTIAL vs INCORRECT**:
- PARTIAL = shows some genuine correct understanding
- INCORRECT = fundamentally wrong or no valid reasoning

## Decision Tree - FOLLOW THIS EXACTLY:
1. Does the answer show ANY valid reasoning or correct work? 
   - NO → INCORRECT
   - YES → Continue to 2

2. Is the final answer correct AND is the reasoning fundamentally sound?
   - NO → Continue to 3
   - YES → Continue to 4

3. Does the answer show some correct understanding despite errors?
   - YES → PARTIAL
   - NO → INCORRECT

4. Is the solution complete with ALL steps and no gaps?
   - NO → PARTIAL (missing steps = not almost!)
   - YES → Continue to 5

5. Are there ANY non-trivial errors or missing components?
   - YES → PARTIAL
   - NO → Continue to 6

6. Are there ONLY trivial/cosmetic issues (notation, minor typos)?
   - YES → ALMOST
   - NO (perfect) → CORRECT

## Few-Shot Examples:
{FEW_SHOT_EXAMPLES}

## Response Format:
You must respond with EXACTLY ONE of these four labels in JSON format:
<json>
{{
    "response": "correct|almost|partial|incorrect"
}}
</json>

## CRITICAL RULES - VIOLATING THESE WILL CAUSE ERRORS:
1. Response must be ONLY: correct, almost, partial, or incorrect (no other text)
2. CORRECT is RARE - only for truly perfect or near-perfect answers
3. ALMOST is for TRIVIAL issues only - missing steps means PARTIAL
4. PARTIAL requires genuine correct work - not just attempting the problem
5. INCORRECT for: wrong reasoning, no justification, or fundamentally flawed approach
6. CORRECT final answer with WRONG reasoning = INCORRECT (not almost!)
7. Missing key cases or steps = PARTIAL (not almost!)
8. When uncertain: choose the MORE CONSERVATIVE (lower) grade
9. Prefer: incorrect > partial > almost > correct"""

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
