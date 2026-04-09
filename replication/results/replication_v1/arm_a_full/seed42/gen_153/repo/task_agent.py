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
    
    # Check for quoted labels (e.g., "correct" or 'almost')
    for label in valid_labels:
        if f'"{label}"' in cleaned or f"'{label}'" in cleaned:
            return label
    
    # Check for common variations and misspellings
    if cleaned in ["correc", "corect", "corrrect", "right", "true", "valid"]:
        return "correct"
    if cleaned in ["almos", "almst", "alomst", "nearly", "close"]:
        return "almost"
    if cleaned in ["part", "prtial", "partal", "some", "incomplete"]:
        return "partial"
    if cleaned in ["incorect", "incorrec", "wrong", "false", "invalid", "error", "none", "fail"]:
        return "incorrect"
    
    # Check for partial matches with priority order
    # "almost" and "partial" should be checked before "correct" to avoid misclassification
    # This handles cases like "partially correct" -> "partial", "almost correct" -> "almost"
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


# Few-shot examples showing exact label format with detailed reasoning
FEW_SHOT_EXAMPLES = """
Example 1 - CORRECT (fully correct answer with complete reasoning):
Problem: Find the sum of 2 + 3.
Solution: The sum of 2 + 3 is 5.
Student Answer: 5
Reasoning: The student provided the exact correct answer with no errors.
<json>
{"response": "correct"}
</json>

Example 2 - PARTIAL (significant gaps but some correct elements):
Problem: Solve x^2 = 4 for x.
Solution: x = 2 or x = -2.
Student Answer: x = 2
Reasoning: The student found one correct solution but missed the other (x = -2). This is a significant omission.
<json>
{"response": "partial"}
</json>

Example 3 - INCORRECT (completely wrong or no valid reasoning):
Problem: Prove that the sum of angles in a triangle is 180 degrees.
Solution: [Detailed geometric proof]
Student Answer: It's 180 because that's what I learned.
Reasoning: The student provided no mathematical reasoning or proof, just a statement without justification.
<json>
{"response": "incorrect"}
</json>

Example 4 - ALMOST (correct method with minor error):
Problem: Find the derivative of x^2.
Solution: The derivative is 2x.
Student Answer: The derivative is 2x, using the power rule where d/dx(x^n) = n*x^(n-1).
Reasoning: The answer is fully correct with proper explanation. This should be "correct" not "almost".
<json>
{"response": "correct"}
</json>

Example 5 - ALMOST (correct approach, small calculation error):
Problem: Compute 15 × 12
Solution: 15 × 12 = 180
Student Answer: 15 × 12 = 150 + 30 = 190 (method correct, arithmetic error: 150+30=180, not 190)
Reasoning: The student used the correct method (15×12 = 15×10 + 15×2 = 150 + 30) but made a small arithmetic error in the final sum.
<json>
{"response": "almost"}
</json>

Example 6 - PARTIAL (correct start but incomplete):
Problem: Solve the system: x + y = 10, x - y = 2
Solution: Add equations: 2x = 12, so x = 6. Substitute: 6 + y = 10, so y = 4.
Student Answer: x = 6 (found first variable but didn't find y)
Reasoning: The student correctly solved for x but failed to complete the solution by finding y.
<json>
{"response": "partial"}
</json>

Example 7 - ALMOST (minor notation error, correct concept):
Problem: Find the integral of 2x.
Solution: ∫2x dx = x^2 + C
Student Answer: x^2 (missing +C, but the integration is correct)
Reasoning: The student correctly integrated but omitted the constant of integration, a minor omission.
<json>
{"response": "almost"}
</json>

Example 8 - PARTIAL (right idea, significant execution errors):
Problem: Factor x^2 - 5x + 6.
Solution: (x - 2)(x - 3)
Student Answer: (x - 1)(x - 6) = x^2 - 7x + 6 (wrong factors, but understood factoring concept)
Reasoning: The student understood they need to factor but chose wrong factors that don't produce the original polynomial.
<json>
{"response": "partial"}
</json>

Example 9 - INCORRECT (fundamental misunderstanding):
Problem: Find the area of a circle with radius 3.
Solution: A = πr² = 9π
Student Answer: A = 2πr = 6π (used circumference formula instead of area)
Reasoning: The student confused area and circumference formulas, showing fundamental misunderstanding.
<json>
{"response": "incorrect"}
</json>

Example 10 - CORRECT (complete proof with all steps):
Problem: Prove that the product of two even numbers is even.
Solution: Let the numbers be 2m and 2n. Their product is 4mn = 2(2mn), which is even.
Student Answer: Let the even numbers be 2m and 2n. Then (2m)(2n) = 4mn = 2(2mn), which is divisible by 2, so even.
Reasoning: The student provided a complete, correct proof with all necessary steps.
<json>
{"response": "correct"}
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

## Step-by-Step Analysis Process:

**Step 1: Understand the Problem**
- What is the problem asking for?
- What is the expected answer format?
- Are there multiple parts or steps required?

**Step 2: Evaluate the Student's Approach**
- Did the student use a valid method to solve the problem?
- Is their reasoning logically sound?
- Did they show their work clearly?

**Step 3: Check for Errors**
- Are there calculation errors? (minor vs major)
- Are there conceptual errors? (fundamental misunderstanding)
- Are there missing steps or incomplete solutions?

**Step 4: Compare Against Solution**
- Does the final answer match the correct solution?
- If different, is the error minor or significant?
- Did they get the main concept right?

**Step 5: Apply Grading Guidelines**
- Consider the specific criteria in the grading guidelines
- Weight the errors appropriately based on their significance

## Label Definitions (READ CAREFULLY):

**"correct"**: The answer is fully correct and complete.
- All steps are correct and logically sound
- Final answer matches the solution exactly
- No errors or omissions
- Clear, valid mathematical reasoning throughout

**"almost"**: The answer is nearly correct with only MINOR issues.
- The main approach/method is correct
- Only a small calculation error (e.g., 2+2=5, arithmetic mistake)
- Missing only a trivial step that doesn't affect the main result
- The student clearly understands the concept but made a minor slip
- Final answer is wrong due to a small error, not misunderstanding

**"partial"**: The answer has SIGNIFICANT gaps or errors in the main argument.
- Some correct elements but missing key parts of the solution
- Started correctly but didn't complete the reasoning
- Has the right idea but made significant errors in execution
- Only solved part of a multi-part problem
- Understood the general concept but applied it incorrectly in important ways

**"incorrect"**: The answer is wrong or does not address the problem.
- Completely wrong approach or answer
- No valid mathematical reasoning
- Answer is irrelevant to the question asked
- Fundamental misunderstanding of the problem or concept
- No meaningful progress toward the solution

## Key Distinctions:

**"almost" vs "partial":**
- "almost" = student ALMOST got it right (minor slip, small error)
- "partial" = student only got PART of it right (significant gaps, incomplete)

**"partial" vs "incorrect":**
- "partial" = has some valid correct elements, made meaningful progress
- "incorrect" = has none or is completely wrong, no valid reasoning

## Common Mistakes to Watch For:
- Don't mark "almost" for answers with significant conceptual errors
- Don't mark "partial" for answers with only trivial omissions
- Don't mark "incorrect" if the student showed valid reasoning, even if incomplete
- Be careful with multi-part problems - missing one part may be "partial" not "almost"

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
