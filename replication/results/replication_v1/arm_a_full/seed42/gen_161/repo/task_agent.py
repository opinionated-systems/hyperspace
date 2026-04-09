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
    
    # Map to valid labels
    valid_labels = ["correct", "almost", "partial", "incorrect"]
    
    # Check for exact match first
    if cleaned in valid_labels:
        return cleaned
    
    # Check for exact match with quotes removed
    cleaned_no_quotes = cleaned.replace('"', '').replace("'", "")
    if cleaned_no_quotes in valid_labels:
        return cleaned_no_quotes
    
    # Check for common variations and misspellings
    # IMPORTANT: Check for "almost" BEFORE "correct" since "almost correct" contains "correct"
    if cleaned in ["almost correct", "nearly correct", "mostly correct", "essentially correct"]:
        return "almost"
    if cleaned in ["partially correct", "part correct", "some correct"]:
        return "partial"
    if cleaned in ["completely incorrect", "totally incorrect", "wrong", "not correct", "false"]:
        return "incorrect"
    if cleaned in ["fully correct", "completely correct", "totally correct", "right"]:
        return "correct"
    
    # Check for partial matches - be careful about order
    # Check for "incorrect" first since it contains a form of "correct"
    if "incorrect" in cleaned or "wrong" in cleaned or "error" in cleaned:
        return "incorrect"
    # Check for "almost" before "correct"
    if "almost" in cleaned or "nearly" in cleaned or "minor" in cleaned:
        return "almost"
    if "partial" in cleaned or "incomplete" in cleaned or "missing" in cleaned:
        return "partial"
    # Only mark as correct if it's explicitly correct and not any other label
    if cleaned == "correct" or ("correct" in cleaned and "not" not in cleaned and "in" not in cleaned and "almost" not in cleaned and "partial" not in cleaned):
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
    
    # Look for explicit labels in JSON-like format first
    if '"incorrect"' in text_lower or "'incorrect'" in text_lower:
        return "incorrect"
    if '"almost"' in text_lower or "'almost'" in text_lower:
        return "almost"
    if '"partial"' in text_lower or "'partial'" in text_lower:
        return "partial"
    if '"correct"' in text_lower or "'correct'" in text_lower:
        return "correct"
    
    # Look for labels in the text - check incorrect first since it contains "correct"
    if "incorrect" in text_lower or "wrong" in text_lower or "error" in text_lower:
        return "incorrect"
    # Check for "almost" before "correct" since "almost correct" contains "correct"
    if "almost" in text_lower or "nearly" in text_lower or "minor" in text_lower:
        return "almost"
    if "partial" in text_lower or "incomplete" in text_lower:
        return "partial"
    # Only mark as correct if explicitly correct and not containing other labels
    if text_lower.strip() == "correct":
        return "correct"
    if "correct" in text_lower and "not" not in text_lower and "in" not in text_lower and "almost" not in text_lower and "partial" not in text_lower:
        return "correct"
    
    # Default to incorrect
    logger.warning(f"Could not extract label from text, defaulting to 'incorrect'")
    return "incorrect"


# Few-shot examples for better grading accuracy - carefully designed to distinguish between labels
# KEY DISTINCTIONS:
# - ALMOST: Core approach correct, only minor/trivial issues (arithmetic slip, small typo, trivial notation)
# - PARTIAL: Significant gaps, missing key components, incomplete solutions, conceptual errors
# - INCORRECT: Wrong approach, no valid reasoning, irrelevant answer
# - CORRECT: Fully complete, all steps correct, thorough understanding demonstrated
FEW_SHOT_EXAMPLES = """
Example 1 - CORRECT (complete and fully correct):
Problem: Find the sum of 2 + 3.
Solution: The sum of 2 + 3 is 5.
Student Answer: 5
Reasoning: The student provided the exact correct answer with no errors.
<json>
{"response": "correct"}
</json>

Example 2 - CORRECT (complete proof with all steps):
Problem: Prove that for any positive integer n, n^3 - n is divisible by 6.
Solution: Factor as n(n-1)(n+1). Among three consecutive integers, one is divisible by 2 and one by 3, so product is divisible by 6.
Student Answer: We can write n^3 - n = n(n^2 - 1) = n(n-1)(n+1). These are three consecutive integers. In any three consecutive integers, there must be one divisible by 2 and one divisible by 3. Therefore the product is divisible by 6.
Reasoning: The student provided a complete, rigorous proof with all necessary steps and correct reasoning.
<json>
{"response": "correct"}
</json>

Example 3 - ALMOST (minor arithmetic error in final step of correct approach):
Problem: Solve for x: 2x + 4 = 10
Solution: Subtract 4 from both sides: 2x = 6, then divide by 2: x = 3.
Student Answer: 2x + 4 = 10, so 2x = 6, therefore x = 4.
Reasoning: The student used the completely correct method (isolate x) but made a trivial arithmetic error at the final step (6/2 = 4 instead of 3). The core approach is perfect; this is just a slip.
<json>
{"response": "almost"}
</json>

Example 4 - ALMOST (correct answer with trivial notation issue):
Problem: Find the derivative of f(x) = x^2
Solution: f'(x) = 2x
Student Answer: f'(x) = 2x (writes "derivative" instead of f')
Reasoning: The answer is mathematically correct. The student used slightly non-standard notation but the mathematics is perfect.
<json>
{"response": "almost"}
</json>

Example 5 - ALMOST (minor memory error on well-known value):
Problem: Find the value of sin(30°).
Solution: sin(30°) = 1/2 = 0.5
Student Answer: sin(30°) = 0.6
Reasoning: The student knew exactly what to evaluate (sin at 30 degrees) but made a minor memory error (0.6 instead of 0.5). The concept is completely correct, just a small recall slip.
<json>
{"response": "almost"}
</json>

Example 6 - PARTIAL (correct approach but incomplete solution - missing one of multiple answers):
Problem: Solve x^2 = 4 for all real x.
Solution: x = 2 or x = -2.
Student Answer: x = 2
Reasoning: The student found one correct solution but missed the other solution (x = -2). The approach was correct but the solution is incomplete - this is a significant gap.
<json>
{"response": "partial"}
</json>

Example 7 - PARTIAL (correct start but missing critical component):
Problem: Find the area of a circle with radius 5.
Solution: A = πr² = π(5)² = 25π ≈ 78.54
Student Answer: The formula is A = πr². With r = 5, we get A = 25.
Reasoning: The student correctly identified the formula and substituted r = 5, but failed to include π in the final answer. This is a significant conceptual gap - the answer is numerically wrong by a factor of π.
<json>
{"response": "partial"}
</json>

Example 8 - PARTIAL (correct method but missing essential element):
Problem: Evaluate the integral of 2x dx.
Solution: ∫2x dx = x² + C
Student Answer: Using the power rule, ∫2x dx = 2 * (x²/2) = x²
Reasoning: The student correctly applied the power rule but forgot the constant of integration (+C), which is essential for indefinite integrals. This is a significant omission.
<json>
{"response": "partial"}
</json>

Example 9 - PARTIAL (partial understanding with significant procedural error):
Problem: Solve the equation 2x + 3 = 7.
Solution: Subtract 3: 2x = 4, then divide by 2: x = 2.
Student Answer: 2x = 7 - 3 = 4, so x = 4.
Reasoning: The student correctly isolated the term with x but failed to divide by 2 at the end. Shows partial understanding but made a significant procedural error.
<json>
{"response": "partial"}
</json>

Example 10 - INCORRECT (no valid mathematical reasoning):
Problem: Prove that the sum of angles in a triangle is 180 degrees.
Solution: [Detailed geometric proof with parallel lines and alternate angles]
Student Answer: It's 180 because that's what I learned in school.
Reasoning: The student provided no mathematical reasoning or proof. This is just a statement without any justification or work shown.
<json>
{"response": "incorrect"}
</json>

Example 11 - INCORRECT (completely wrong approach):
Problem: Solve for x: sin(x) = 0.5
Solution: x = 30° or x = 150° (plus periodic solutions)
Student Answer: x = arcsin(0.5) = 2
Reasoning: The student confused arcsin with a different operation. arcsin(0.5) is not 2. The approach is fundamentally wrong - they don't understand inverse trig functions.
<json>
{"response": "incorrect"}
</json>

Example 12 - INCORRECT (irrelevant answer - answered different question):
Problem: Find the derivative of x³.
Solution: d/dx(x³) = 3x²
Student Answer: The integral of x³ is x⁴/4.
Reasoning: The student answered a completely different question (integration instead of differentiation). The answer is irrelevant to the problem asked.
<json>
{"response": "incorrect"}
</json>

Example 13 - ALMOST (correct with trivial typo that doesn't affect meaning):
Problem: Compute 15 × 4.
Solution: 15 × 4 = 60
Student Answer: 15 × 4 = 60 (writes "fiften" instead of "fifteen" in words)
Reasoning: The answer is mathematically correct. The typo in the word form is trivial and doesn't affect the mathematical correctness.
<json>
{"response": "almost"}
</json>

Example 14 - PARTIAL (missing critical final step):
Problem: Find the roots of x² - 5x + 6 = 0.
Solution: Factor as (x-2)(x-3) = 0, so x = 2 or x = 3.
Student Answer: (x-2)(x-3)
Reasoning: The student correctly factored but didn't state the actual roots. Missing the final answer is a significant gap - the problem asks for roots, not factored form.
<json>
{"response": "partial"}
</json>

Example 15 - ALMOST (correct core with minor formatting issue):
Problem: Prove that the sum of two odd numbers is even.
Solution: Let the odd numbers be 2k+1 and 2m+1. Their sum is 2k+1+2m+1 = 2(k+m+1), which is even.
Student Answer: If we have two odd numbers like 2k+1 and 2m+1, adding them gives 2k+2m+2 = 2(k+m+1). This is even because it has a factor of 2.
Reasoning: The proof is essentially correct and complete. The student could have been slightly more explicit about why 2(k+m+1) is even, but the reasoning is clear and correct.
<json>
{"response": "almost"}
</json>

Example 16 - INCORRECT (fundamental conceptual misunderstanding):
Problem: Find the area of a rectangle with length 5 and width 3.
Solution: Area = length × width = 5 × 3 = 15
Student Answer: Area = 5 + 3 = 8
Reasoning: The student used addition instead of multiplication, showing a fundamental misunderstanding of what area means. This is not a minor slip but a conceptual error.
<json>
{"response": "incorrect"}
</json>

Example 17 - PARTIAL (correct intuition but incomplete formal reasoning):
Problem: Prove that there are infinitely many prime numbers.
Solution: [Classic proof by contradiction assuming finite primes and constructing a new prime]
Student Answer: There must be infinitely many primes because we can always find more by multiplying known primes and adding 1.
Reasoning: The student has the right intuition about constructing new primes, but didn't provide a complete formal proof with all logical steps. The idea is there but the execution is incomplete.
<json>
{"response": "partial"}
</json>

Example 18 - ALMOST (correct solution with minor sign error in final answer):
Problem: Solve x² - 9 = 0
Solution: x² = 9, so x = 3 or x = -3
Student Answer: x² = 9, so x = ±3 (writes "x = 3" as final answer, missing the negative)
Reasoning: The student correctly solved the equation and understood there are two solutions (indicated by ±), but only wrote one in the final answer. This is a minor omission in an otherwise correct solution.
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

## Grading Rubric - Use These Definitions Carefully:

**CORRECT**: The answer is fully correct, complete, and demonstrates thorough understanding.
- All required steps are present and correct
- Final answer matches the solution exactly
- Reasoning is sound, rigorous, and well-explained
- No errors, omissions, or significant gaps
- Would receive full credit (e.g., 7/7 on a 7-point problem)

**ALMOST**: The answer is nearly correct with only minor, superficial issues.
- Core approach and reasoning are completely correct
- Minor arithmetic error (e.g., 6/2 = 4 instead of 3) in final step
- Small notation issue or missing trivial justification
- Understanding is clearly demonstrated
- Would receive high partial credit (e.g., 6/7 or 5/7 on a 7-point problem)
- KEY DISTINCTION: The error is minor and doesn't affect the core reasoning

**PARTIAL**: The answer shows some correct work but has significant gaps or errors.
- Some correct steps or partial understanding shown
- Missing key components (e.g., only one of two solutions, missing constant of integration)
- Has significant conceptual errors or logical gaps
- Incomplete solution or missing final answer
- Would receive partial credit (e.g., 1-5 points on a 7-point problem)
- KEY DISTINCTION: There's a significant gap or error that affects the solution quality

**INCORRECT**: The answer is wrong or does not address the problem.
- No valid mathematical reasoning shown
- Completely wrong approach or method
- No substantive work shown
- Answer is irrelevant to the question asked
- Would receive minimal or no credit (0-1 points on a 7-point problem)

## Critical Decision Guidelines:

When deciding between labels, ask yourself:

1. Is the core approach correct? 
   - If NO → INCORRECT
   - If YES but major gaps → PARTIAL
   - If YES but minor issues → ALMOST
   - If YES and complete → CORRECT

2. Is the solution complete?
   - Missing critical parts → PARTIAL or INCORRECT
   - Complete but with trivial errors → ALMOST
   - Fully complete → CORRECT

3. Does the student demonstrate understanding?
   - No understanding shown → INCORRECT
   - Partial understanding → PARTIAL
   - Good understanding with minor slip → ALMOST
   - Full understanding → CORRECT

## Few-Shot Examples:
""" + FEW_SHOT_EXAMPLES + """

## Grading Instructions:
1. Carefully read the problem and understand what is being asked.
2. Compare the student's answer against the correct solution step by step.
3. Analyze: Is the approach correct? Is it complete? Are there errors?
4. Use the decision guidelines above to select the appropriate label.
5. Be conservative: When in doubt between two labels, choose the lower one.

## Step-by-Step Decision Process:
Follow this exact process to determine the grade:

**Step 1: Check for INCORRECT first**
- Does the student show NO valid mathematical reasoning? → INCORRECT
- Is the approach completely wrong or irrelevant? → INCORRECT
- If YES to either, grade as INCORRECT and stop.

**Step 2: Check for CORRECT**
- Is the answer fully complete with all steps correct? → CORRECT
- Does it match the solution exactly with no errors? → CORRECT
- If YES, grade as CORRECT and stop.

**Step 3: Distinguish ALMOST vs PARTIAL (this is critical)**
- Are the errors MINOR (arithmetic slip, trivial typo, small notation issue)? → ALMOST
- Are the gaps SIGNIFICANT (missing key steps, incomplete solution, major conceptual gaps)? → PARTIAL
- KEY TEST: If the student fixed only their errors, would it be CORRECT with minimal changes? → ALMOST
- KEY TEST: Does the student need to add substantial work to make it correct? → PARTIAL

**Step 4: Final check**
- When truly uncertain between two labels, choose the more conservative (lower) grade.

## Response Format:
First, provide your reasoning for the grade in 2-3 sentences. Then provide your final label in the JSON format below.

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

IMPORTANT: Your response field must contain ONLY one of these four exact words: correct, almost, partial, or incorrect. Do not include any other text in the JSON."""

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
