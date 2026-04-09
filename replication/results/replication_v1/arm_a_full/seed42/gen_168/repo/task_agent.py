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
    almost_indicators = ["almost", "minor slip", "small error", "tiny mistake", "minor calculation", 
                        "minor error", "slight error", "tiny error", "small slip"]
    if any(indicator in cleaned for indicator in almost_indicators):
        return "almost"
    
    # Check for "partial" - check before "correct" and "incorrect"
    partial_indicators = ["partial", "incomplete", "missing steps", "part correct", "partly correct",
                         "partially", "some correct", "incomplete solution"]
    if any(indicator in cleaned for indicator in partial_indicators):
        return "partial"
    
    # Check for "incorrect"
    incorrect_indicators = ["incorrect", "completely wrong", "fundamental error", "no valid",
                           "wrong", "invalid", "not correct", "false"]
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
        conflicting = ["partial", "almost", "mostly", "somewhat", "nearly", "close", "minor",
                      "incomplete", "partly", "nearly"]
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
    
    # Check for standalone labels (word boundaries)
    # Priority: almost > partial > incorrect > correct
    for label in ["almost", "partial", "incorrect", "correct"]:
        # Use word boundary to find standalone labels
        pattern = r'\b' + label + r'\b'
        if re.search(pattern, text_lower):
            # For "correct", check no conflicting terms
            if label == "correct":
                conflicting = ["partial", "almost", "mostly", "somewhat", "nearly", "close", "minor",
                              "incomplete", "partly"]
                if any(term in text_lower for term in conflicting):
                    continue
            return label
    
    # Check for keywords in the text - order matters!
    # Priority: almost > partial > incorrect > correct
    # This prevents "almost correct" from being classified as "correct"
    
    # Check for "almost" indicators first
    almost_indicators = ["almost", "minor slip", "small error", "tiny mistake", "minor calculation",
                        "minor error", "slight error", "tiny error"]
    if any(indicator in text_lower for indicator in almost_indicators):
        return "almost"
    
    # Check for "partial" indicators
    partial_indicators = ["partial", "incomplete", "part correct", "partly correct", "partially",
                         "some correct", "incomplete solution"]
    if any(indicator in text_lower for indicator in partial_indicators):
        return "partial"
    
    # Check for "incorrect" indicators
    incorrect_indicators = ["incorrect", "completely wrong", "fundamental error", "no valid",
                           "wrong", "invalid", "not correct"]
    if any(indicator in text_lower for indicator in incorrect_indicators):
        return "incorrect"
    
    # For "correct", be very careful about context
    if "correct" in text_lower:
        # Check that no conflicting terms are present
        conflicting = ["partial", "almost", "mostly", "somewhat", "nearly", "close", "minor",
                      "incomplete", "partly"]
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

Example 13 - ALMOST (correct method, wrong final answer due to arithmetic):
Problem: Find 25% of 80.
Solution: 0.25 × 80 = 20
Student Answer: 0.25 × 80 = 25 (correct method, arithmetic error)
<json>
{"response": "almost"}
</json>

Example 14 - PARTIAL (incomplete solution, missing key steps):
Problem: Find the roots of x^2 - 5x + 6 = 0.
Solution: Factor: (x-2)(x-3) = 0, so x = 2 or x = 3.
Student Answer: (x-2)(x-3) = 0 (factored correctly but didn't state the roots)
<json>
{"response": "partial"}
</json>

Example 15 - ALMOST (correct work, minor sign error):
Problem: Solve: -3x = 9
Solution: x = -3
Student Answer: x = 3 (correct method, sign error only)
<json>
{"response": "almost"}
</json>

Example 16 - PARTIAL (wrong formula but some correct work):
Problem: Find the volume of a sphere with radius 3.
Solution: V = (4/3)πr³ = 36π
Student Answer: V = 4πr² = 36π (used surface area formula, not volume)
<json>
{"response": "partial"}
</json>

Example 17 - ALMOST (correct logic, one number wrong):
Problem: Sum of angles in a pentagon.
Solution: (5-2) × 180 = 540 degrees
Student Answer: (5-2) × 180 = 360 degrees (correct formula, wrong calculation)
<json>
{"response": "almost"}
</json>

Example 18 - INCORRECT (nonsensical answer):
Problem: What is 7 × 8?
Solution: 56
Student Answer: Purple (completely irrelevant)
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
        
        instruction = f"""You are an expert grading agent for {domain}. Your task is to carefully analyze the student's answer and assign exactly one of four labels.

## Problem Statement:
{problem}

## Correct Solution:
{solution}

## Grading Guidelines:
{grading_guidelines}

## Student's Answer:
{student_answer}

## CRITICAL: Use This Decision Tree (Follow Exactly!)

**QUESTION 1: Is the answer PERFECT? (No errors at all)**
- If YES → "correct"
- If NO (any error exists) → Continue to Question 2

**QUESTION 2: Is the main method/approach CORRECT with only MINOR errors?**
- The student used the right method/approach
- Error is just a small calculation mistake (e.g., 2+2=5, arithmetic slip)
- OR missing a tiny detail that doesn't change the main result
- Student clearly understands the concept
- If YES → "almost"
- If NO (major method error or significant gaps) → Continue to Question 3

**QUESTION 3: Does the answer contain ANY valid mathematical work?**
- Some correct steps or partial understanding shown
- Even if incomplete or has significant errors
- If YES → "partial"
- If NO (completely wrong or no valid work) → "incorrect"

## Label Definitions with Examples:

**"correct"** = Perfect answer, zero errors
Example: Problem asks for 2+3, student answers 5 with correct reasoning

**"almost"** = Right method, minor error only
Examples:
- Correct approach but 15×12=190 (should be 180)
- Correct formula but arithmetic slip at the end
- Right method, just forgot to include units
- One tiny typo in an otherwise perfect solution
KEY: Method is RIGHT, error is SMALL

**"partial"** = Some valid work but significant issues
Examples:
- Started correctly but didn't finish
- Used right concept but applied it wrong
- Some correct steps but major gaps
- Understood part of the problem but not all
KEY: Has SOME valid elements but NOT just a minor error

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

## Self-Verification Step (Do This Before Responding!):
Before giving your final answer, verify your classification:

1. If you chose "correct": Confirm there are ZERO errors of any kind
2. If you chose "almost": Confirm the method is RIGHT and error is truly MINOR (just a slip)
3. If you chose "partial": Confirm there is SOME valid work but significant gaps exist
4. If you chose "incorrect": Confirm there is NO valid mathematical reasoning at all

## Response Format:
You must respond with EXACTLY ONE of these four labels in JSON format:
<json>
{{
    "response": "correct|almost|partial|incorrect"
}}
</json>

CRITICAL: Your response field must contain ONLY one of these four exact words: correct, almost, partial, or incorrect. Do not include explanations, reasoning, or any other text in the JSON.

## REMEMBER - The Key Distinctions:
- "correct" = Perfect answer, zero errors
- "almost" = Right method + minor slip only (fixing it makes it perfect)
- "partial" = Some valid work but significant gaps (not just a minor error)
- "incorrect" = No valid mathematical reasoning at all

WHEN IN DOUBT between "almost" and "partial":
- Ask: "Is the method completely correct with just a tiny error?" 
- If YES → "almost"
- If NO (significant issues) → "partial"

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
