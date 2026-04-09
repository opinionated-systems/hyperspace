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
    # Only return "correct" if explicitly stated and no other labels found
    if "correct" in cleaned:
        # Avoid matching "not correct", "partially correct", etc.
        if "not correct" in cleaned or "partially" in cleaned or "incorrect" in cleaned:
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
    
    # Look for explicit labels in JSON-like format first (most reliable)
    # Check in order of specificity (almost and partial are more specific than correct)
    if "\"almost\"" in text_lower or "'almost'" in text_lower:
        return "almost"
    if "\"partial\"" in text_lower or "'partial'" in text_lower:
        return "partial"
    if "\"incorrect\"" in text_lower or "'incorrect'" in text_lower:
        return "incorrect"
    if "\"correct\"" in text_lower or "'correct'" in text_lower:
        # Make sure we're not matching "incorrect" in the quoted string
        if "\"incorrect\"" not in text_lower and "'incorrect'" not in text_lower:
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
    # Only return "correct" if it's explicitly stated and no other labels found
    if "correct" in text_lower:
        # Make sure we're not matching "incorrect" or "partially correct" etc.
        if "not correct" in text_lower or "partially" in text_lower or "incorrect" in text_lower:
            return "partial"
        return "correct"
    
    # Default to incorrect when uncertain
    logger.warning(f"Could not extract label from text, defaulting to 'incorrect'")
    return "incorrect"


# Few-shot examples for better grading accuracy - expanded to better distinguish almost vs partial
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

Example 3 - ALMOST (Correct answer with trivial notation issue):
Problem: Solve for x: 2x + 4 = 10
Solution: x = 3
Student Answer: x=3 (missing space around equals, but answer is correct)
<json>
{"response": "almost"}
</json>

Example 4 - ALMOST (Correct approach and answer, minor explanation gap):
Problem: Find the area of a circle with radius 5.
Solution: A = πr² = 25π
Student Answer: 25π (correct answer but didn't show the formula A = πr²)
<json>
{"response": "almost"}
</json>

Example 5 - PARTIAL (Partially correct but significant gaps):
Problem: Solve x^2 = 4 for x.
Solution: x = 2 or x = -2.
Student Answer: x = 2
<json>
{"response": "partial"}
</json>

Example 6 - PARTIAL (Some correct work but incomplete reasoning):
Problem: Prove that the sum of angles in a triangle is 180 degrees.
Solution: [Detailed geometric proof with parallel lines and alternate angles]
Student Answer: Draw a line parallel to one side through the opposite vertex. The alternate angles are equal, so the three angles form a straight line.
<json>
{"response": "partial"}
</json>

Example 7 - PARTIAL (Correct method but calculation error):
Problem: Calculate 15 × 12.
Solution: 180
Student Answer: 15 × 12 = 15 × 10 + 15 × 2 = 150 + 20 = 170 (right method, wrong final calculation)
<json>
{"response": "partial"}
</json>

Example 8 - INCORRECT (Wrong or does not address the problem):
Problem: Prove that the sum of angles in a triangle is 180 degrees.
Solution: [Detailed geometric proof]
Student Answer: It's 180 because that's what I learned.
<json>
{"response": "incorrect"}
</json>

Example 9 - INCORRECT (Major conceptual error):
Problem: Find the integral of x dx.
Solution: x^2/2 + C
Student Answer: x^2 + C
<json>
{"response": "incorrect"}
</json>

Example 10 - INCORRECT (Fundamental misunderstanding):
Problem: Solve x^2 = 4 for x.
Solution: x = 2 or x = -2.
Student Answer: x = 16 (squared instead of taking square root)
<json>
{"response": "incorrect"}
</json>

Example 11 - PARTIAL (Some correct work but incomplete):
Problem: Solve the system: x + y = 5, x - y = 1.
Solution: x = 3, y = 2
Student Answer: From the first equation, x = 5 - y.
<json>
{"response": "partial"}
</json>

Example 12 - ALMOST vs PARTIAL distinction:
ALMOST: Has the RIGHT answer with only trivial issues (formatting, minor notation, tiny gaps in explanation)
PARTIAL: Missing key components, has significant errors, or incomplete solution even if partial work is correct
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
- Fully correct and complete
- Contains all required steps and reasoning
- Matches the correct solution in substance and conclusion
- No errors, omissions, or significant gaps

**ALMOST**: Use when the answer is:
- Has the CORRECT final answer
- Uses the right approach/method
- Contains only TRIVIAL errors (e.g., minor notation issues, small typos, formatting)
- Missing only INSIGNIFICANT details that don't affect correctness
- The core solution is sound and complete

**PARTIAL**: Use when the answer is:
- Shows some correct work or understanding BUT has significant gaps
- Missing key steps, cases, or components
- Has the right idea but incomplete execution
- Contains correct reasoning but also significant errors
- Missing the final answer or has an incorrect final answer despite some correct work
- Incomplete solution that doesn't fully address the problem

**INCORRECT**: Use when the answer is:
- Wrong or does not address the problem
- Contains major conceptual errors
- Has fundamentally flawed reasoning
- Gives an incorrect final answer with no redeeming correct work
- Merely states a conclusion without justification

## Key Distinction - ALMOST vs PARTIAL:
- ALMOST = Correct answer + minor cosmetic issues (the solution is essentially right)
- PARTIAL = Significant gaps, missing components, or wrong final answer despite some correct work
- When in doubt: If the final answer is correct and the method is sound → ALMOST; otherwise → PARTIAL

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
4. "almost" requires the CORRECT final answer with only trivial issues
5. "partial" is for incomplete solutions or wrong answers with some correct work
6. "incorrect" is for answers that are fundamentally wrong or lack valid reasoning
7. First check: Is the final answer correct? If yes and only minor issues → ALMOST; If no or major gaps → PARTIAL or lower"""

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
