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
    
    # Split by common separators and check each part
    import re
    parts = re.split(r'[\s,;|]+', cleaned)
    for part in parts:
        part = part.strip(".!?,:;\"'")
        if part in valid_labels:
            return part
    
    # Check for partial matches - be careful with "correct" appearing in other words
    # Order matters: check more specific terms first
    if "almost" in cleaned:
        return "almost"
    if "partial" in cleaned:
        return "partial"
    if "incorrect" in cleaned or "wrong" in cleaned:
        return "incorrect"
    
    # Only return "correct" if it's a standalone word (not part of "partially correct" etc.)
    if cleaned == "correct":
        return "correct"
    if "correct" in cleaned and "partial" not in cleaned and "incorrect" not in cleaned and "almost" not in cleaned:
        # Additional check: make sure "correct" is the main label
        words = cleaned.split()
        if "correct" in words:
            return "correct"
    
    # Default to incorrect if no match found
    logger.warning(f"Could not normalize prediction '{prediction}', defaulting to 'incorrect'")
    return "incorrect"


def _extract_label_from_text(text: str) -> str:
    """Extract a valid label from raw text when JSON parsing fails.
    
    Uses a priority-based approach to find the most likely label.
    
    Args:
        text: Raw text from LLM response
        
    Returns:
        One of: correct, almost, partial, incorrect
    """
    if not text:
        return "incorrect"
    
    text_lower = text.lower()
    
    # Priority 1: Look for explicit labels in JSON-like format (most reliable)
    json_patterns = [
        ('"correct"', "correct"),
        ("'correct'", "correct"),
        ('"almost"', "almost"),
        ("'almost'", "almost"),
        ('"partial"', "partial"),
        ("'partial'", "partial"),
        ('"incorrect"', "incorrect"),
        ("'incorrect'", "incorrect"),
    ]
    for pattern, label in json_patterns:
        if pattern in text_lower:
            return label
    
    # Priority 2: Look for labels after common indicators
    indicators = ["label:", "grade:", "rating:", "assessment:", "verdict:", "final answer:", "conclusion:"]
    for indicator in indicators:
        idx = text_lower.find(indicator)
        if idx != -1:
            after = text_lower[idx + len(indicator):].strip()
            for label in ["correct", "almost", "partial", "incorrect"]:
                if after.startswith(label):
                    return label
    
    # Priority 3: Look for standalone labels (word boundaries)
    import re
    for label in ["correct", "almost", "partial", "incorrect"]:
        # Match whole word with word boundaries
        pattern = r'\b' + label + r'\b'
        if re.search(pattern, text_lower):
            return label
    
    # Priority 4: Check for keywords in order of specificity
    # "almost" and "partial" are more specific than "correct" or "incorrect"
    if "almost" in text_lower:
        return "almost"
    if "partial" in text_lower:
        return "partial"
    if "incorrect" in text_lower or "wrong" in text_lower or "error" in text_lower:
        return "incorrect"
    if "correct" in text_lower:
        return "correct"
    
    # Default to incorrect
    logger.warning(f"Could not extract label from text, defaulting to 'incorrect'")
    return "incorrect"


# Few-shot examples showing exact label format with reasoning
FEW_SHOT_EXAMPLES = """
Example 1 - CORRECT:
Problem: Find the sum of 2 + 3.
Solution: The sum of 2 + 3 is 5.
Student Answer: 5
Analysis: The student's answer matches the correct solution exactly. The answer is complete and correct.
<json>
{"response": "correct"}
</json>

Example 2 - PARTIAL:
Problem: Solve x^2 = 4 for x.
Solution: x = 2 or x = -2.
Student Answer: x = 2
Analysis: The student found one correct solution (x = 2) but missed the other solution (x = -2). This is a significant omission in the answer.
<json>
{"response": "partial"}
</json>

Example 3 - INCORRECT:
Problem: Prove that the sum of angles in a triangle is 180 degrees.
Solution: [Detailed geometric proof]
Student Answer: It's 180 because that's what I learned.
Analysis: The student states the correct fact but provides no proof or reasoning. The answer does not demonstrate understanding of the mathematical concept.
<json>
{"response": "incorrect"}
</json>

Example 4 - ALMOST:
Problem: Find the derivative of x^2.
Solution: The derivative is 2x.
Student Answer: The derivative is 2x, using the power rule where d/dx(x^n) = n*x^(n-1).
Analysis: The student's answer is correct and they show their work. However, they made a minor notational error in the formula (should be n*x^(n-1), not n*x^(n-1) - wait that's correct). Actually, looking again, the answer is fully correct with proper reasoning. Let me reconsider - the answer shows correct application of the power rule.
<json>
{"response": "correct"}
</json>

Example 5 - ALMOST (true example):
Problem: Calculate the area of a circle with radius 3.
Solution: Area = πr² = π × 9 = 9π ≈ 28.27
Student Answer: Area = π × 3² = 9π = 28.26
Analysis: The student correctly applied the formula and got 9π, but made a small rounding error (28.26 instead of 28.27). The mathematical reasoning is correct.
<json>
{"response": "almost"}
</json>

Example 6 - PARTIAL:
Problem: Solve the system: x + y = 5, x - y = 1
Solution: Adding equations: 2x = 6, so x = 3. Then y = 2.
Student Answer: x = 3
Analysis: The student correctly found x = 3 but did not find y = 2. The solution is incomplete.
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

## Label Definitions (STRICT INTERPRETATION):
- "correct": The answer is fully correct and complete, matching the solution approach and result. The student demonstrates full understanding with no significant errors or omissions.
- "almost": The answer is nearly correct with only minor issues (e.g., small calculation error, missing trivial step, minor notation issue). The core reasoning is sound.
- "partial": The answer is partially correct but has significant gaps, missing key components, or errors in the main argument. The student shows some understanding but not enough for substantial credit.
- "incorrect": The answer is wrong, does not address the problem, contains fundamental errors, or shows no meaningful understanding of the problem.

## Grading Instructions:
1. Carefully read the problem and understand what is being asked.
2. Compare the student's answer against the correct solution step by step.
3. Check if the student shows their work and reasoning.
4. Identify any errors, omissions, or misconceptions.
5. Determine if errors are minor (almost) or significant (partial/incorrect).
6. When in doubt between two labels, choose the LOWER grade (more conservative).

## Few-Shot Examples:
{FEW_SHOT_EXAMPLES}

## Response Format:
First, provide your analysis of the student's answer. Then, provide your final label in the exact JSON format below.

Your analysis should:
- State what the problem is asking
- Summarize the correct solution approach
- Evaluate what the student got right
- Identify any errors or omissions
- Explain why you chose your final label

After your analysis, you MUST provide the final label in this exact format:

<json>
{{
    "response": "correct|almost|partial|incorrect"
}}
</json>

IMPORTANT: The response field must contain ONLY one of these four exact words: correct, almost, partial, or incorrect."""

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
