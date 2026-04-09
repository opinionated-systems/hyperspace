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
    
    # Remove any punctuation, quotes, or extra whitespace
    cleaned = cleaned.strip(".!?,:;\"'[]{}()").strip()
    
    # Remove common prefixes/suffixes that LLMs might add
    cleaned = cleaned.replace("label:", "").replace("grade:", "").replace("rating:", "").strip()
    
    # Map to valid labels
    valid_labels = ["correct", "almost", "partial", "incorrect"]
    
    # Check for exact match first
    if cleaned in valid_labels:
        return cleaned
    
    # Check for exact match with quotes removed
    cleaned_no_quotes = cleaned.replace('"', '').replace("'", "").strip()
    if cleaned_no_quotes in valid_labels:
        return cleaned_no_quotes
    
    # Check for exact match with extra whitespace removed
    cleaned_no_space = cleaned.replace(" ", "").strip()
    if cleaned_no_space in valid_labels:
        return cleaned_no_space
    
    # Check for partial matches - be careful about "correct" being too greedy
    # Check for "almost" first (contains "correct" but should be "almost")
    if "almost" in cleaned:
        return "almost"
    if "partial" in cleaned:
        return "partial"
    if "incorrect" in cleaned or "wrong" in cleaned or "error" in cleaned:
        return "incorrect"
    # Only check for "correct" if no other label is present
    if "correct" in cleaned:
        return "correct"
    
    # Default to incorrect if no match found
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
    
    # Look for explicit labels in quotes first (highest priority)
    if '"almost"' in text_lower or "'almost'" in text_lower:
        return "almost"
    if '"partial"' in text_lower or "'partial'" in text_lower:
        return "partial"
    if '"incorrect"' in text_lower or "'incorrect'" in text_lower:
        return "incorrect"
    if '"correct"' in text_lower or "'correct'" in text_lower:
        return "correct"
    
    # Look for labels after common prefixes like "response": or "label":
    import re
    # Pattern to match "response": "label" or similar
    label_pattern = r'["\']?response["\']?\s*:\s*["\']?(correct|almost|partial|incorrect)["\']?'
    match = re.search(label_pattern, text_lower)
    if match:
        return match.group(1)
    
    # Check for keywords in the text - check "almost" and "partial" before "correct"
    # to avoid misclassifying "almost correct" as "correct"
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


# Few-shot examples for better grading accuracy - IMO Grade School Math
FEW_SHOT_EXAMPLES = """
Example 1 - CORRECT (complete and accurate):
Problem: Find the sum of 2 + 3.
Solution: The sum of 2 + 3 is 5.
Student Answer: 5
<json>
{"response": "correct"}
</json>

Example 2 - ALMOST (minor error in correct work):
Problem: Find the derivative of x^2.
Solution: The derivative is 2x.
Student Answer: The derivative is 2x + 1, using the power rule.
Analysis: The student correctly identified the power rule but made a small arithmetic error (+1). The core approach is correct.
<json>
{"response": "almost"}
</json>

Example 3 - PARTIAL (incomplete solution with some correct work):
Problem: Solve x^2 = 4 for x.
Solution: x = 2 or x = -2.
Student Answer: x = 2
Analysis: The student found one correct solution but missed the other. Shows partial understanding but incomplete.
<json>
{"response": "partial"}
</json>

Example 4 - INCORRECT (no valid reasoning):
Problem: Prove that the sum of angles in a triangle is 180 degrees.
Solution: [Detailed geometric proof]
Student Answer: It's 180 because that's what I learned.
Analysis: No mathematical reasoning provided. Just states a fact without proof.
<json>
{"response": "incorrect"}
</json>

Example 5 - CORRECT (complete proof with all steps):
Problem: Prove that for any positive integer n, n^3 - n is divisible by 6.
Solution: Factor as n(n-1)(n+1). Among three consecutive integers, one is divisible by 2 and one by 3, so product is divisible by 6.
Student Answer: We can write n^3 - n = n(n^2 - 1) = n(n-1)(n+1). These are three consecutive integers. In any three consecutive integers, there must be one divisible by 2 and one divisible by 3. Therefore the product is divisible by 6.
Analysis: Complete proof with all necessary steps and correct reasoning.
<json>
{"response": "correct"}
</json>

Example 6 - PARTIAL (correct approach but missing key step):
Problem: Find all real solutions to x^2 - 5x + 6 = 0.
Solution: Factor as (x-2)(x-3) = 0, so x = 2 or x = 3.
Student Answer: Using the quadratic formula, x = (5 ± sqrt(25-24))/2 = (5 ± 1)/2, so x = 3.
Analysis: Used correct method but made an error in final answer (missed x=2). Shows understanding of method but execution error. This is PARTIAL because the method is correct but the execution has a significant error.
<json>
{"response": "partial"}
</json>

Example 7 - ALMOST (correct answer with minor notation issue):
Problem: Evaluate the integral of 2x dx.
Solution: x^2 + C
Student Answer: x^2
Analysis: Correct computation but forgot the constant of integration. Minor omission in otherwise correct work. This is ALMOST because the core answer is correct, just missing a notation convention.
<json>
{"response": "almost"}
</json>

Example 8 - PARTIAL (some correct steps but significant gaps):
Problem: Find the area of a circle with radius 5.
Solution: A = πr^2 = π(5)^2 = 25π
Student Answer: A = πr^2 = π(25)
Analysis: Correct formula and substitution, but didn't simplify to final answer. Incomplete solution. This is PARTIAL because the work is incomplete, not just a minor error.
<json>
{"response": "partial"}
</json>

Example 9 - INCORRECT (wrong approach - no valid reasoning):
Problem: Solve for x: 2x + 4 = 10
Solution: 2x = 6, so x = 3
Student Answer: x = 10 - 4 = 6
Analysis: Student misunderstood the equation structure. No valid algebraic reasoning. This is INCORRECT because there's no valid mathematical reasoning shown.
<json>
{"response": "incorrect"}
</json>

Example 10 - ALMOST (correct reasoning, small calculation error):
Problem: Compute 15 × 12.
Solution: 15 × 12 = 180
Student Answer: 15 × 12 = 15 × 10 + 15 × 2 = 150 + 20 = 170
Analysis: Correct method (distributive property) but arithmetic error (150+20=170 instead of 170). Core understanding is correct. This is ALMOST because the method is perfect, just a small calculation slip.
<json>
{"response": "almost"}
</json>

Example 12 - ALMOST (nearly correct with trivial error):
Problem: Find the derivative of f(x) = 3x^2 + 2x.
Solution: f'(x) = 6x + 2
Student Answer: f'(x) = 6x + 2x = 8x
Analysis: Student correctly applied power rule to 3x^2 getting 6x, but incorrectly simplified 2x + 2x instead of keeping it as 6x + 2. The core calculus understanding is correct, just a simple algebra error at the end.
<json>
{"response": "almost"}
</json>

Example 13 - PARTIAL (some understanding but major gaps):
Problem: Solve the system: x + y = 5, x - y = 1.
Solution: Adding equations: 2x = 6, so x = 3. Then y = 2.
Student Answer: x = 3
Analysis: Student found x correctly but completely omitted finding y. Shows partial understanding of solving systems but incomplete solution. This is PARTIAL, not ALMOST, because a major component is missing.
<json>
{"response": "partial"}
</json>

Example 14 - INCORRECT (fundamental misunderstanding):
Problem: Find the area of a rectangle with length 8 and width 5.
Solution: Area = length × width = 8 × 5 = 40
Student Answer: Area = 8 + 5 = 13
Analysis: Student used addition instead of multiplication, showing a fundamental misunderstanding of area. No valid reasoning for the correct formula. This is INCORRECT.
<json>
{"response": "incorrect"}
</json>

Example 15 - ALMOST (correct work with minor sign error):
Problem: Simplify: (x + 3)(x - 2).
Solution: x^2 - 2x + 3x - 6 = x^2 + x - 6
Student Answer: x^2 - 2x + 3x + 6 = x^2 + x + 6
Analysis: Student correctly applied FOIL method but made a sign error on the last term (-2 × 3 = -6, not +6). The method is correct, just a sign slip. This is ALMOST.
<json>
{"response": "almost"}
</json>

Example 11 - CORRECT (complete with clear reasoning):
Problem: Find the equation of the line passing through (1, 2) and (3, 8).
Solution: Slope m = (8-2)/(3-1) = 3. Using point-slope: y - 2 = 3(x - 1), so y = 3x - 1.
Student Answer: The slope is (8-2)/(3-1) = 6/2 = 3. Using point (1,2): y - 2 = 3(x - 1), which gives y = 3x - 1.
Analysis: Complete solution with correct slope calculation and proper use of point-slope form.
<json>
{"response": "correct"}
</json>

Example 12 - ALMOST (correct work with trivial sign error):
Problem: Solve x^2 - 9 = 0.
Solution: x^2 = 9, so x = ±3.
Student Answer: x^2 = 9, so x = 3.
Analysis: Correct method but missed the negative solution. Minor omission in an otherwise correct solution.
<json>
{"response": "almost"}
</json>

Example 13 - PARTIAL (started correctly but didn't finish):
Problem: Find the derivative of f(x) = x^3 + 2x^2 - 5x + 1.
Solution: f'(x) = 3x^2 + 4x - 5.
Student Answer: Using the power rule: f'(x) = 3x^2 + 4x...
Analysis: Started correctly with power rule but didn't complete the derivative (missing -5). Shows partial understanding.
<json>
{"response": "partial"}
</json>

Example 14 - INCORRECT (completely wrong method):
Problem: Find the area of a rectangle with length 8 and width 5.
Solution: Area = length × width = 8 × 5 = 40.
Student Answer: Area = 8 + 5 = 13.
Analysis: Student used addition instead of multiplication. Wrong method entirely.
<json>
{"response": "incorrect"}
</json>

Example 15 - ALMOST (correct approach, minor algebraic slip):
Problem: Simplify (x^2 - 9)/(x - 3) for x ≠ 3.
Solution: (x^2 - 9)/(x - 3) = (x+3)(x-3)/(x-3) = x + 3.
Student Answer: Factor numerator: (x+3)(x-3)/(x-3) = x + 3 + 1 = x + 4. Wait, no, it's just x + 3.
Analysis: Student made a small slip (+1 error) but self-corrected. Shows correct understanding with minor confusion.
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

## Grading Rubric - Use These Definitions:

**CORRECT**: The answer is fully correct, complete, and demonstrates thorough understanding.
- All required steps are present and correct
- Final answer matches the solution exactly
- Reasoning is sound, logical, and well-explained
- No errors, omissions, or notation issues
- Use this ONLY when the answer is essentially perfect and complete
- Examples: Complete proofs with all steps, correct calculations with proper notation, full solutions

**ALMOST**: The answer is nearly correct with only minor, trivial issues.
- Core approach and reasoning are fully correct
- Minor calculation error (e.g., 2+2=5 in an otherwise correct solution)
- Minor notation issue (e.g., forgot +C in integration, missing dx, sign error in final step)
- Small omission that doesn't affect the core understanding (e.g., missed one of two solutions when both are similar)
- Would receive high partial credit (e.g., 6/7 or 5/7 on a 7-point problem)
- IMPORTANT: The student must demonstrate correct understanding of the method with only trivial mistakes
- KEY DISTINCTION: If the error is minor and the core logic is sound, use ALMOST (not PARTIAL)

**PARTIAL**: The answer is partially correct but has significant gaps, errors, or incompleteness.
- Some correct steps or partial understanding is demonstrated
- Missing key components required for a complete solution
- Has significant errors that affect the core reasoning
- Incomplete solution (e.g., stopped halfway, didn't provide final answer)
- Correct approach but major execution errors (e.g., wrong formula application, significant calculation errors)
- Would receive partial credit (e.g., 2-4 points on a 7-point problem)
- Use this when the student shows some understanding but has substantial gaps
- KEY DISTINCTION: If the error is significant or the solution is incomplete, use PARTIAL (not ALMOST)

**INCORRECT**: The answer is wrong or does not address the problem.
- No valid mathematical reasoning or logic
- Completely wrong approach or method
- No substantive work shown (just a guess or irrelevant answer)
- Fundamental misunderstanding of the problem or concepts
- Would receive minimal or no credit (0-1 points on a 7-point problem)
- Use this when the answer shows no understanding of the problem
- KEY DISTINCTION: If there's no valid reasoning, use INCORRECT (not PARTIAL)

## Decision Framework - Step by Step:
Before selecting a label, carefully analyze:
1. Does the student show ANY valid mathematical reasoning? 
   - If NO -> INCORRECT
   - If YES -> Continue to step 2

2. Is the core approach/method correct?
   - If NO -> INCORRECT or PARTIAL (depending on if any valid steps exist)
   - If YES -> Continue to step 3

3. Is the solution complete with all required steps?
   - If NO (missing key steps or incomplete) -> PARTIAL
   - If YES -> Continue to step 4

4. Is the final answer correct?
   - If NO (major errors) -> PARTIAL
   - If YES or minor errors only -> Continue to step 5

5. Are there any errors? If so, are they minor/trivial or major/significant?
   - Minor/trivial (calculation slip, notation omission) -> ALMOST
   - Major/significant (wrong formula, missing key component) -> PARTIAL
   - No errors -> CORRECT

## Key Distinctions to Remember:
- ALMOST vs PARTIAL: ALMOST is for trivial mistakes in correct work; PARTIAL is for significant gaps or errors
- PARTIAL vs INCORRECT: PARTIAL requires some valid reasoning; INCORRECT means no valid reasoning
- CORRECT vs ALMOST: CORRECT means perfect; ALMOST means nearly perfect with minor issues

## Common Mistakes to Avoid:
- DO NOT label "almost" as "correct" - "almost" means there ARE minor errors
- DO NOT label "partial" as "incorrect" - "partial" means some valid reasoning exists
- DO NOT label "incorrect" as "partial" - "incorrect" means NO valid reasoning
- When in doubt between two labels, choose the one that reflects the student's actual understanding level

## Few-Shot Examples:
{FEW_SHOT_EXAMPLES}

## Response Format:
You must respond with EXACTLY ONE of these four labels in JSON format:
- "correct" - The answer is fully correct and complete
- "almost" - The answer is nearly correct with only minor issues  
- "partial" - The answer is partially correct but has significant gaps
- "incorrect" - The answer is wrong or does not address the problem

<json>
{{
    "response": "correct|almost|partial|incorrect"
}}
</json>

IMPORTANT: Your response field must contain ONLY one of these four exact words: correct, almost, partial, or incorrect. Do not include any other text.

## Final Check Before Responding:
1. Re-read the student answer carefully
2. Compare against the solution
3. Apply the decision framework above
4. Verify your chosen label matches the definitions
5. Ensure your JSON is properly formatted with exactly one of the four valid labels"""

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
