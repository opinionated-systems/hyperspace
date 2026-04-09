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

from agent.llm_client import get_response_from_llm, EVAL_MODEL, EVAL_TEMPERATURE

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
    
    # Check for negations first - these should lower the grade
    if "not correct" in cleaned or "not almost" in cleaned or "not partial" in cleaned:
        return "incorrect"
    
    # Check for "partially correct" - should be partial, not correct
    if "partially correct" in cleaned:
        return "partial"
    
    # Check for "mostly correct" - could be almost or partial depending on context
    if "mostly correct" in cleaned:
        return "partial"  # Conservative: mostly correct implies some issues
    
    # Check for partial matches - be conservative (prefer lower grades when ambiguous)
    # Check for "incorrect" first (most definitive negative)
    if "incorrect" in cleaned:
        return "incorrect"
    
    # Check for "wrong"
    if "wrong" in cleaned:
        return "incorrect"
    
    # Check for "almost" (specific intermediate grade)
    if "almost" in cleaned:
        return "almost"
    
    # Check for "partial"
    if "partial" in cleaned:
        return "partial"
    
    # Only return "correct" if explicitly stated and no conflicting labels found
    if "correct" in cleaned:
        # Double-check for any negation or qualification
        if any(neg in cleaned for neg in ["not", "hardly", "barely", "mostly", "partially"]):
            return "partial"
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
    
    # Check for negations first - these should lower the grade
    if "not correct" in text_lower or "not almost" in text_lower:
        return "incorrect"
    
    # Check for "partially correct" - should be partial
    if "partially correct" in text_lower:
        return "partial"
    
    # Look for explicit labels in JSON-like format first (most reliable)
    if "\"incorrect\"" in text_lower or "'incorrect'" in text_lower:
        return "incorrect"
    if "\"wrong\"" in text_lower or "'wrong'" in text_lower:
        return "incorrect"
    if "\"almost\"" in text_lower or "'almost'" in text_lower:
        return "almost"
    if "\"partial\"" in text_lower or "'partial'" in text_lower:
        return "partial"
    if "\"correct\"" in text_lower or "'correct'" in text_lower:
        # Check for qualifiers
        if any(neg in text_lower for neg in ["not", "hardly", "barely", "mostly", "partially"]):
            return "partial"
        return "correct"
    
    # Check for keywords in the text - be conservative (prefer lower grades)
    # Check for "incorrect" or "wrong" first (most definitive negative)
    if "incorrect" in text_lower or "wrong" in text_lower:
        return "incorrect"
    
    # Check for "almost" (specific intermediate grade)
    if "almost" in text_lower:
        return "almost"
    
    # Check for "partial" 
    if "partial" in text_lower:
        return "partial"
    
    # Only return "correct" if explicitly stated and no conflicting labels found
    if "correct" in text_lower:
        # Double-check for any negation or qualification
        if any(neg in text_lower for neg in ["not", "hardly", "barely", "mostly", "partially"]):
            return "partial"
        return "correct"
    
    # Default to incorrect when uncertain
    logger.warning(f"Could not extract label from text, defaulting to 'incorrect'")
    return "incorrect"


# Few-shot examples for better grading accuracy - expanded with more edge cases
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

Example 3 - ALMOST (Correct answer with minor notation issue):
Problem: Evaluate the integral of 2x dx from 0 to 1.
Solution: The integral is x^2 evaluated from 0 to 1, which equals 1.
Student Answer: ∫(2x)dx from 0 to 1 = [x²]₀¹ = 1 - 0 = 1. The student used superscript ² instead of ^2, but the math is correct.
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

Example 5 - PARTIAL (Right approach but incomplete execution):
Problem: Find the maximum value of f(x) = -x^2 + 4x.
Solution: f'(x) = -2x + 4 = 0, so x = 2. f(2) = -4 + 8 = 4. Maximum is 4.
Student Answer: Take derivative: f'(x) = -2x + 4. Set to 0: -2x + 4 = 0.
<json>
{"response": "partial"}
</json>

Example 6 - INCORRECT (Wrong or does not address the problem):
Problem: Prove that the sum of angles in a triangle is 180 degrees.
Solution: [Detailed geometric proof]
Student Answer: It's 180 because that's what I learned.
<json>
{"response": "incorrect"}
</json>

Example 7 - INCORRECT (Major conceptual error):
Problem: Find the integral of x dx.
Solution: x^2/2 + C
Student Answer: x^2 + C
<json>
{"response": "incorrect"}
</json>

Example 8 - INCORRECT (Correct final answer but wrong reasoning):
Problem: What is 2 + 2?
Solution: 2 + 2 = 4
Student Answer: 2 + 2 = 5 - 1 = 4
<json>
{"response": "incorrect"}
</json>

Example 9 - INCORRECT (Missing key steps with correct answer):
Problem: Solve the quadratic equation x^2 - 5x + 6 = 0 and show your work.
Solution: Factoring: (x-2)(x-3) = 0, so x = 2 or x = 3.
Student Answer: x = 2, 3
<json>
{"response": "incorrect"}
</json>

Example 10 - ALMOST vs PARTIAL distinction:
Problem: Compute the area of a circle with radius 3.
Solution: A = πr² = π(3)² = 9π ≈ 28.27
Student Answer A (ALMOST): A = π(3)² = 9π. The student forgot to provide the decimal approximation but the exact answer is correct.
Student Answer B (PARTIAL): A = πr² = 3π. The student used the wrong formula (r instead of r²).
<json>
{"response": "almost"}
</json>

Example 11 - PARTIAL vs INCORRECT distinction:
Problem: Find the derivative of sin(x).
Solution: cos(x)
Student Answer A (PARTIAL): The derivative involves cosine. The student shows some understanding but doesn't give the complete answer.
Student Answer B (INCORRECT): The derivative is tan(x). The student gives a completely wrong answer.
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
- Fully correct and complete with all required steps
- Contains proper reasoning and justification
- Matches the correct solution in substance and conclusion
- No errors, omissions, or significant gaps of any kind
- Has both correct final answer AND correct reasoning process

**ALMOST**: Use when the answer is:
- Nearly correct with only truly minor issues
- Has the right approach AND correct final answer
- Contains only trivial errors (e.g., minor notation issues, small typos that don't affect correctness)
- Missing only insignificant details that don't impact the solution
- The core logic and answer are correct

**PARTIAL**: Use when the answer is:
- Shows some genuine correct work or understanding
- Has the right general idea but incomplete execution
- Missing key steps, cases, or components
- Contains some correct reasoning but also significant errors
- Partially correct but has significant gaps
- Has correct approach but wrong final answer, OR incomplete work with correct partial results

**INCORRECT**: Use when the answer is:
- Wrong or does not address the problem
- Contains major conceptual errors
- Has fundamentally flawed reasoning
- Gives an incorrect final answer with no redeeming correct work
- Merely states a conclusion without justification
- Has correct final answer but completely wrong reasoning
- Missing required work/steps even if final answer happens to be correct

## Key Distinctions:

**CORRECT vs ALMOST**: 
- CORRECT requires perfection - all steps, proper reasoning, complete solution
- ALMOST allows minor cosmetic issues only if core is perfect

**ALMOST vs PARTIAL**:
- ALMOST has correct final answer with minor issues
- PARTIAL has significant gaps, missing key steps, or wrong final answer despite good approach

**PARTIAL vs INCORRECT**:
- PARTIAL shows some genuine correct work/understanding
- INCORRECT has no redeeming correct work or fundamentally wrong approach

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
2. Be STRICT and CONSERVATIVE - when in doubt between two labels, choose the more conservative (lower) grade
3. "correct" should be rare - only for truly complete and accurate answers with all steps shown
4. "almost" is for minor issues only - not for missing significant steps or having wrong answers
5. "partial" requires some genuine correct work, not just attempting the problem
6. "incorrect" is for answers that are fundamentally wrong, lack valid reasoning, or miss required work
7. If the answer has the right final answer but wrong or missing reasoning, it is INCORRECT not CORRECT"""

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
