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
    
    # Check for quoted labels (most reliable indicator)
    if '"almost"' in cleaned or "'almost'" in cleaned:
        return "almost"
    if '"partial"' in cleaned or "'partial'" in cleaned:
        return "partial"
    if '"incorrect"' in cleaned or "'incorrect'" in cleaned:
        return "incorrect"
    if '"correct"' in cleaned or "'correct'" in cleaned:
        return "correct"
    
    # Check for partial matches with priority order
    # Priority: almost > partial > incorrect > correct
    # This ensures we don't misclassify nuanced labels
    
    # Check for "almost" - must be checked first to avoid being caught by "correct"
    # Be more specific: look for "almost" as a standalone word or clear indicator
    almost_indicators = ["almost", "minor slip", "small error", "tiny mistake", "minor calculation"]
    if any(indicator in cleaned for indicator in almost_indicators):
        return "almost"
    
    # Check for "partial" - check before "correct" and "incorrect"
    partial_indicators = ["partial", "incomplete", "missing steps", "part correct"]
    if any(indicator in cleaned for indicator in partial_indicators):
        return "partial"
    
    # Check for "incorrect"
    incorrect_indicators = ["incorrect", "completely wrong", "fundamental error", "no valid"]
    if any(indicator in cleaned for indicator in incorrect_indicators):
        return "incorrect"
    
    # Check for "correct" - be very careful about context
    # Only return "correct" if it's clearly the intended label
    if cleaned == "correct":
        return "correct"
    
    # For multi-word responses, check if "correct" appears as a standalone word
    # and not in phrases like "partially correct", "almost correct", etc.
    words = cleaned.split()
    if "correct" in words:
        # Check that no conflicting terms are present
        conflicting = ["partial", "almost", "mostly", "somewhat", "nearly", "close", "minor"]
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
    # Check in priority order: almost > partial > incorrect > correct
    if '"almost"' in text_lower or "'almost'" in text_lower:
        return "almost"
    if '"partial"' in text_lower or "'partial'" in text_lower:
        return "partial"
    if '"incorrect"' in text_lower or "'incorrect'" in text_lower:
        return "incorrect"
    if '"correct"' in text_lower or "'correct'" in text_lower:
        return "correct"
    
    # Check for JSON-like patterns with response field
    import re
    json_pattern = r'["\']?response["\']?\s*:\s*["\']?(\w+)["\']?'
    match = re.search(json_pattern, text_lower)
    if match:
        label = match.group(1).lower().strip()
        if label in ["correct", "almost", "partial", "incorrect"]:
            return label
    
    # Check for keywords in the text - order matters!
    # Priority: almost > partial > incorrect > correct
    # This prevents "almost correct" from being classified as "correct"
    
    # Check for "almost" indicators first
    if "almost" in text_lower or "minor" in text_lower or "slip" in text_lower or "small error" in text_lower:
        return "almost"
    
    # Check for "partial" indicators
    if "partial" in text_lower or "incomplete" in text_lower or "part" in text_lower:
        return "partial"
    
    # Check for "incorrect" indicators
    if "incorrect" in text_lower or "wrong" in text_lower or "error" in text_lower or "invalid" in text_lower:
        return "incorrect"
    
    # For "correct", be very careful about context
    if "correct" in text_lower:
        # Check that no conflicting terms are present
        conflicting = ["partial", "almost", "mostly", "somewhat", "nearly", "close", "minor"]
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

Example 2 - ALMOST (correct method with only MINOR error):
Problem: Find the derivative of x^2.
Solution: The derivative is 2x.
Student Answer: The derivative is 2x, using the power rule where d/dx(x^n) = n*x^(n-1).
<json>
{"response": "almost"}
</json>

Example 3 - ALMOST (correct approach, small calculation error):
Problem: Compute 15 × 12
Solution: 15 × 12 = 180
Student Answer: 15 × 12 = 150 + 30 = 190 (method correct, arithmetic error)
<json>
{"response": "almost"}
</json>

Example 4 - ALMOST (correct concept, minor slip):
Problem: Solve for x: 2x + 4 = 10
Solution: 2x = 6, so x = 3
Student Answer: 2x = 6, so x = 4 (correct method, minor arithmetic slip)
<json>
{"response": "almost"}
</json>

Example 5 - ALMOST (correct method, minor calculation slip):
Problem: Compute 100 - 37
Solution: 100 - 37 = 63
Student Answer: 100 - 37 = 64 (small subtraction error, method correct)
<json>
{"response": "almost"}
</json>

Example 6 - ALMOST (correct approach, tiny mistake in final step):
Problem: Find the area of a rectangle with length 5 and width 3.
Solution: Area = length × width = 5 × 3 = 15
Student Answer: Area = 5 × 3 = 16 (correct formula, minor calculation error)
<json>
{"response": "almost"}
</json>

Example 7 - PARTIAL (significant gaps but some correct elements):
Problem: Solve x^2 = 4 for x.
Solution: x = 2 or x = -2.
Student Answer: x = 2 (only found one solution, missing the other)
<json>
{"response": "partial"}
</json>

Example 8 - PARTIAL (correct start but incomplete):
Problem: Solve the system: x + y = 10, x - y = 2
Solution: Add equations: 2x = 12, so x = 6. Substitute: 6 + y = 10, so y = 4.
Student Answer: x = 6 (found first variable but didn't find y)
<json>
{"response": "partial"}
</json>

Example 9 - PARTIAL (right idea but significant execution errors):
Problem: Find the area of a circle with radius 5.
Solution: A = πr^2 = 25π
Student Answer: A = 2πr = 10π (used circumference formula instead of area)
<json>
{"response": "partial"}
</json>

Example 10 - INCORRECT (completely wrong or no valid reasoning):
Problem: Prove that the sum of angles in a triangle is 180 degrees.
Solution: [Detailed geometric proof]
Student Answer: It's 180 because that's what I learned.
<json>
{"response": "incorrect"}
</json>

Example 11 - INCORRECT (fundamental misunderstanding):
Problem: Solve x^2 = 9
Solution: x = 3 or x = -3
Student Answer: x = 81 (completely wrong operation - squared instead of rooted)
<json>
{"response": "incorrect"}
</json>

Example 12 - ALMOST vs PARTIAL distinction:
Problem: Compute 100 - 37
Solution: 100 - 37 = 63
Student Answer A (ALMOST): 100 - 37 = 64 (small subtraction error, method correct)
<json>
{"response": "almost"}
</json>
Student Answer B (PARTIAL): 100 - 30 = 70 (only subtracted part, incomplete method)
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

## Step-by-Step Grading Process:

**STEP 1: Check for "correct"**
- Does the student's answer match the solution exactly?
- Are all steps logically sound and complete?
- Is there ANY error, even a minor one?
- If NO errors at all → label as "correct"
- If ANY error exists → proceed to Step 2

**STEP 2: Check for "almost" (CRITICAL - Most commonly missed!)**

The "almost" label is for answers that are NEARLY CORRECT with only MINOR issues.

Ask yourself:
- Is the main approach/method FULLY CORRECT?
- Is the error just a small calculation slip (e.g., 2+2=5, 15×12=190)?
- Does the student clearly understand the concept?
- Would fixing the minor error make it completely correct?

If YES to all above → label as "almost"

**STEP 3: Distinguish "partial" vs "incorrect"**

If NOT "almost", ask:

A. Does the student understand the main concept/method?
   - YES, but had significant execution errors → "partial"
   - NO, fundamental misunderstanding → "incorrect"

B. How severe are the errors?
   - Used wrong formula but some correct work → "partial"
   - Completely wrong approach or no valid work → "incorrect"

C. How complete is the answer?
   - Partial solution with some correct elements → "partial"
   - No valid elements or irrelevant → "incorrect"

## Label Definitions (READ CAREFULLY):

**"correct"**: The answer is fully correct and complete.
- All steps are correct and logically sound
- Final answer matches the solution exactly
- No errors or omissions whatsoever

**"almost"**: The answer is NEARLY CORRECT with only MINOR issues.
- The main approach/method is FULLY CORRECT
- Only a small calculation error (e.g., 2+2=5, 15×12=190, 100-37=64)
- Missing only a trivial step that doesn't affect the main result
- The student clearly understands the concept but made a minor slip
- The error is MINOR enough that fixing it would make the answer completely correct
- Think: "They ALMOST got it right, just a tiny mistake"
- KEY INDICATOR: Correct method + minor error = "almost"

**"partial"**: The answer has SIGNIFICANT gaps or errors in the main argument.
- Some correct elements but missing key parts of the solution
- Started correctly but didn't complete the reasoning
- Has the right idea but made significant errors in execution
- Only solved part of a multi-part problem
- Used wrong formula but showed some understanding
- Think: "They got PART of it right, but significant gaps remain"

**"incorrect"**: The answer is wrong or does not address the problem.
- Completely wrong approach or answer
- No valid mathematical reasoning
- Answer is irrelevant to the question asked
- Fundamental misunderstanding of the problem
- Think: "This is just wrong, no valid elements"

## Critical Decision Rules:

1. **"almost" vs "partial" (MOST IMPORTANT!)**: 
   - "almost" = correct method + minor error (fixing the error makes it perfect)
   - "partial" = incomplete or significant errors despite some correct elements
   - WHEN IN DOUBT: If the method is correct and error is small → "almost"

2. **"partial" vs "incorrect"**:
   - "partial" = at least some valid mathematical work
   - "incorrect" = no valid mathematical reasoning

3. **When in doubt between two labels**:
   - If the error is minor and method is sound → choose "almost"
   - If there's some valid work but significant issues → choose "partial"
   - If no valid work → choose "incorrect"

## Final Check Before Responding:
- Did you check for "almost"? (Correct method + minor error)
- Is the error truly minor or actually significant?
- Would fixing the error make the answer completely correct?

## Few-Shot Examples:
{FEW_SHOT_EXAMPLES}

## Response Format:
You must respond with EXACTLY ONE of these four labels in JSON format:
<json>
{{
    "response": "correct|almost|partial|incorrect"
}}
</json>

CRITICAL: Your response field must contain ONLY one of these four exact words: correct, almost, partial, or incorrect. Do not include explanations, reasoning, or any other text in the JSON.

## REMEMBER:
- "almost" = Correct method + minor error (most commonly missed!)
- "partial" = Some correct work but significant gaps
- "incorrect" = No valid mathematical reasoning
- "correct" = Perfect answer with no errors
"""

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
