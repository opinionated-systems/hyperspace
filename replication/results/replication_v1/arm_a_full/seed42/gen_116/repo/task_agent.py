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
    
    # Check for partial matches with priority order
    # "almost" and "partial" should be checked before "correct" to avoid misclassification
    if "almost" in cleaned:
        return "almost"
    if "partial" in cleaned:
        return "partial"
    if "incorrect" in cleaned or "wrong" in cleaned or "error" in cleaned:
        return "incorrect"
    
    # Check for "correct" - but be very careful about context
    # Only return "correct" if it's clearly the intended label
    if cleaned == "correct":
        return "correct"
    
    # For multi-word responses, check if "correct" appears as a standalone word
    # and not in phrases like "partially correct", "almost correct", etc.
    words = cleaned.split()
    if "correct" in words:
        # Check that no conflicting terms are present
        conflicting = ["partial", "almost", "mostly", "somewhat", "nearly", "close"]
        if not any(term in cleaned for term in conflicting):
            return "correct"
    
    # Default to incorrect if no clear match found
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
    if '"correct"' in text_lower or "'correct'" in text_lower:
        return "correct"
    if '"almost"' in text_lower or "'almost'" in text_lower:
        return "almost"
    if '"partial"' in text_lower or "'partial'" in text_lower:
        return "partial"
    if '"incorrect"' in text_lower or "'incorrect'" in text_lower:
        return "incorrect"
    
    # Check for JSON-like patterns with response field
    import re
    json_pattern = r'["\']?response["\']?\s*:\s*["\']?(\w+)["\']?'
    match = re.search(json_pattern, text_lower)
    if match:
        label = match.group(1).lower().strip()
        if label in ["correct", "almost", "partial", "incorrect"]:
            return label
    
    # Check for keywords in the text - order matters!
    # Check for "almost" and "partial" first to avoid misclassification
    if "almost" in text_lower:
        return "almost"
    if "partial" in text_lower:
        return "partial"
    if "incorrect" in text_lower or "wrong" in text_lower or "error" in text_lower:
        return "incorrect"
    
    # For "correct", be careful about context
    if "correct" in text_lower:
        # Check that no conflicting terms are present
        conflicting = ["partial", "almost", "mostly", "somewhat", "nearly", "close"]
        if not any(term in text_lower for term in conflicting):
            return "correct"
    
    # Default to incorrect
    logger.warning(f"Could not extract label from text, defaulting to 'incorrect'")
    return "incorrect"


# Few-shot examples showing exact label format
FEW_SHOT_EXAMPLES = """
Example 1 - CORRECT (fully correct answer):
Problem: Find the sum of 2 + 3.
Solution: The sum of 2 + 3 is 5.
Student Answer: 5
<json>
{"response": "correct"}
</json>

Example 2 - PARTIAL (significant gaps but some correct elements):
Problem: Solve x^2 = 4 for x.
Solution: x = 2 or x = -2.
Student Answer: x = 2
<json>
{"response": "partial"}
</json>

Example 3 - INCORRECT (completely wrong or no valid reasoning):
Problem: Prove that the sum of angles in a triangle is 180 degrees.
Solution: [Detailed geometric proof]
Student Answer: It's 180 because that's what I learned.
<json>
{"response": "incorrect"}
</json>

Example 4 - ALMOST (correct method with minor error):
Problem: Find the derivative of x^2.
Solution: The derivative is 2x.
Student Answer: The derivative is 2x, using the power rule where d/dx(x^n) = n*x^(n-1).
<json>
{"response": "almost"}
</json>

Example 5 - ALMOST (correct approach, small calculation error):
Problem: Compute 15 × 12
Solution: 15 × 12 = 180
Student Answer: 15 × 12 = 150 + 30 = 190 (method correct, arithmetic error)
<json>
{"response": "almost"}
</json>

Example 6 - PARTIAL (correct start but incomplete):
Problem: Solve the system: x + y = 10, x - y = 2
Solution: Add equations: 2x = 12, so x = 6. Substitute: 6 + y = 10, so y = 4.
Student Answer: x = 6 (found first variable but didn't find y)
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
        
        instruction = f"""You are an expert grading agent for {domain}. Your task is to carefully analyze the student's answer and assign exactly one of four labels.

## Problem Statement:
{problem}

## Correct Solution:
{solution}

## Grading Guidelines:
{grading_guidelines}

## Student's Answer:
{student_answer}

## Grading Instructions:
1. Carefully read the problem and understand what is being asked.
2. Compare the student's answer against the correct solution.
3. Check if the student shows their work and reasoning.
4. Identify any errors, omissions, or misconceptions.
5. Consider partial credit where appropriate based on the grading guidelines.

## Label Definitions (READ CAREFULLY):

**"correct"**: The answer is fully correct and complete.
- All steps are correct and logically sound
- Final answer matches the solution exactly
- No errors or omissions

**"almost"**: The answer is nearly correct with only MINOR issues.
- The main approach/method is correct
- Only a small calculation error (e.g., 2+2=5)
- Missing only a trivial step that doesn't affect the main result
- The student clearly understands the concept but made a minor slip

**"partial"**: The answer has SIGNIFICANT gaps or errors in the main argument.
- Some correct elements but missing key parts of the solution
- Started correctly but didn't complete the reasoning
- Has the right idea but made significant errors in execution
- Only solved part of a multi-part problem

**"incorrect"**: The answer is wrong or does not address the problem.
- Completely wrong approach or answer
- No valid mathematical reasoning
- Answer is irrelevant to the question asked
- Fundamental misunderstanding of the problem

## Key Distinctions:
- "almost" vs "partial": "almost" means the student ALMOST got it right (minor slip), while "partial" means they only got PART of it right (significant gaps).
- "partial" vs "incorrect": "partial" has some valid correct elements, "incorrect" has none or is completely wrong.

## Few-Shot Examples:
{FEW_SHOT_EXAMPLES}

## Response Format:
You must respond with EXACTLY ONE of these four labels in JSON format:
<json>
{{
    "response": "correct|almost|partial|incorrect"
}}
</json>

CRITICAL: Your response field must contain ONLY one of these four exact words: correct, almost, partial, or incorrect. Do not include explanations, reasoning, or any other text in the JSON."""

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
