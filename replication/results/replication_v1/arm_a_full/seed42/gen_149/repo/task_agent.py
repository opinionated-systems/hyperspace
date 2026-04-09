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
    
    # Check for partial matches - be careful about "correct" being too greedy
    # Check for "almost" first (contains "correct" but should be "almost")
    if "almost" in cleaned:
        return "almost"
    if "partial" in cleaned:
        return "partial"
    if "incorrect" in cleaned or "wrong" in cleaned:
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
    
    # Check for keywords in the text - check "almost" and "partial" before "correct"
    # to avoid misclassifying "almost correct" as "correct"
    if "almost" in text_lower:
        return "almost"
    if "partial" in text_lower:
        return "partial"
    if "incorrect" in text_lower or "wrong" in text_lower:
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
Analysis: Used correct method but made an error in final answer (missed x=2). Shows understanding of method but execution error.
<json>
{"response": "partial"}
</json>

Example 7 - ALMOST (correct answer with minor notation issue):
Problem: Evaluate the integral of 2x dx.
Solution: x^2 + C
Student Answer: x^2
Analysis: Correct computation but forgot the constant of integration. Minor omission in otherwise correct work.
<json>
{"response": "almost"}
</json>

Example 8 - PARTIAL (some correct steps but significant gaps):
Problem: Find the area of a circle with radius 5.
Solution: A = πr^2 = π(5)^2 = 25π
Student Answer: A = πr^2 = π(25)
Analysis: Correct formula and substitution, but didn't simplify to final answer. Incomplete solution.
<json>
{"response": "partial"}
</json>

Example 9 - INCORRECT (wrong approach):
Problem: Solve for x: 2x + 4 = 10
Solution: 2x = 6, so x = 3
Student Answer: x = 10 - 4 = 6
Analysis: Student misunderstood the equation structure. No valid algebraic reasoning.
<json>
{"response": "incorrect"}
</json>

Example 10 - ALMOST (correct reasoning, small calculation error):
Problem: Compute 15 × 12.
Solution: 15 × 12 = 180
Student Answer: 15 × 12 = 15 × 10 + 15 × 2 = 150 + 20 = 170
Analysis: Correct method (distributive property) but arithmetic error (150+20=170 instead of 170). Core understanding is correct.
<json>
{"response": "almost"}
</json>

Example 11 - ALMOST (correct proof with minor typo):
Problem: Prove that the sum of the first n positive integers is n(n+1)/2.
Solution: Use induction or pairing argument to show sum = n(n+1)/2.
Student Answer: Pair first with last: 1+n = n+1, 2+(n-1) = n+1, etc. There are n/2 such pairs, so sum = n(n+1)/2.
Analysis: Correct proof method and reasoning. Minor issue: should specify n/2 pairs when n is even, (n-1)/2 pairs plus middle term when n is odd. Core proof is valid.
<json>
{"response": "almost"}
</json>

Example 12 - PARTIAL (correct start but incomplete):
Problem: Find the maximum value of f(x) = -x^2 + 4x on [0, 4].
Solution: f'(x) = -2x + 4 = 0 → x = 2. f(2) = -4 + 8 = 4. Check endpoints: f(0) = 0, f(4) = 0. Maximum is 4.
Student Answer: Take derivative: f'(x) = -2x + 4. Set to 0: -2x + 4 = 0, so x = 2.
Analysis: Correctly found critical point but didn't evaluate function at critical point or endpoints to find maximum. Incomplete solution.
<json>
{"response": "partial"}
</json>

Example 13 - ALMOST (correct approach, sign error):
Problem: Solve for x: 3x - 7 = 8.
Solution: 3x = 15, so x = 5.
Student Answer: 3x - 7 = 8 → 3x = 8 + 7 = 15 → x = 5. Wait, let me check: 3(5) - 7 = 15 - 7 = 8. Correct.
Analysis: Actually correct! But if student wrote x = -5 by mistake, that would be ALMOST (correct method, sign error).
<json>
{"response": "correct"}
</json>

Example 14 - PARTIAL vs ALMOST distinction:
Problem: Compute ∫(2x + 3)dx from 0 to 2.
Solution: [x^2 + 3x] from 0 to 2 = (4 + 6) - 0 = 10.
Student Answer A (ALMOST): ∫(2x + 3)dx = x^2 + 3x + C. At x=2: 4+6=10. At x=0: 0. Answer: 10.
Analysis A: Correct work, just didn't write the evaluation notation formally.
Student Answer B (PARTIAL): ∫(2x + 3)dx = x^2 + 3x. So answer is x^2 + 3x.
Analysis B: Found antiderivative but didn't evaluate at bounds. Significant gap.
<json>
{"response": "partial"}
</json>

Example 15 - INCORRECT vs PARTIAL distinction:
Problem: Find the area of a rectangle with length 5 and width 3.
Solution: Area = length × width = 5 × 3 = 15.
Student Answer A (INCORRECT): Area = 5 + 3 = 8.
Analysis A: Wrong formula used. No understanding of area concept.
Student Answer B (PARTIAL): Area = length × width. I think it's 5 × 3 but I'm not sure.
Analysis B: Knows correct formula but lacks confidence/verification. Some understanding shown.
<json>
{"response": "incorrect"}
</json>
"""


class TaskAgent:
    """Task agent that solves IMO grading problems with self-consistency voting."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.call_count = 0
        self.error_count = 0
        # Number of votes for self-consistency (odd number for tie-breaking)
        self.num_votes = 5  # Increased from 3 to 5 for better consensus

    def _get_single_prediction(self, instruction: str) -> tuple[str, list[dict]]:
        """Get a single prediction from the LLM.
        
        Args:
            instruction: The grading prompt
            
        Returns:
            Tuple of (prediction, msg_history)
        """
        try:
            response, msg_history, info = _get_llm_response_with_retry(
                msg=instruction,
                model=self.model,
            )
        except RuntimeError as e:
            logger.error(f"Failed to get LLM response: {e}")
            return "incorrect", []

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

        return prediction, msg_history

    def _vote_on_predictions(self, predictions: list[str]) -> str:
        """Use majority voting to select the final prediction.
        
        Args:
            predictions: List of predictions from multiple runs
            
        Returns:
            The majority vote prediction
        """
        from collections import Counter
        
        # Count votes
        vote_counts = Counter(predictions)
        
        # Log voting details
        logger.info(f"Self-consistency votes: {dict(vote_counts)}")
        
        # Get the most common prediction
        most_common = vote_counts.most_common(1)[0]
        winner = most_common[0]
        count = most_common[1]
        
        logger.info(f"Majority vote: '{winner}' with {count}/{len(predictions)} votes")
        
        return winner

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with self-consistency voting.

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

## Step-by-Step Analysis Process:

**Step 1: Understand the Problem**
- Read the problem statement carefully
- Identify what is being asked
- Note any constraints or special conditions

**Step 2: Study the Correct Solution**
- Understand the key insights and techniques required
- Identify all necessary steps for a complete solution
- Note the final answer format

**Step 3: Evaluate the Student's Answer**
- Check if the student understood the problem correctly
- Verify if the approach/technique is appropriate
- Check for correct application of mathematical concepts
- Verify calculations and reasoning
- Check if the final answer is correct and complete

**Step 4: Identify Errors (if any)**
- Distinguish between minor errors (calculation, notation) and major errors (wrong approach, missing key insights)
- Check if errors are trivial or significant
- Determine if partial credit is warranted

**Step 5: Match to Rubric**
- CORRECT: Perfect or near-perfect with all key elements
- ALMOST: Correct approach with only trivial mistakes
- PARTIAL: Some correct work but significant gaps or errors
- INCORRECT: No valid reasoning or completely wrong approach

## Grading Rubric - Use These Definitions:

**CORRECT**: The answer is fully correct, complete, and demonstrates thorough understanding.
- All required steps are present and correct
- Final answer matches the solution exactly
- Reasoning is sound, rigorous, and well-explained
- No significant errors or omissions
- Use this ONLY when the answer is essentially perfect

**ALMOST**: The answer is nearly correct with only minor issues.
- Core approach and key insights are correct
- Minor calculation error, arithmetic mistake, or notation issue (e.g., forgot +C in integration, sign error)
- Understanding is clearly demonstrated with only trivial gaps
- Would receive high partial credit (e.g., 6-7/7 on a 7-point problem)
- IMPORTANT: Use this when the student shows correct reasoning with only trivial mistakes
- Examples: Small arithmetic errors in correct work, missing units, minor notation issues

**PARTIAL**: The answer is partially correct but has significant gaps or errors.
- Some correct steps or partial understanding shown
- Missing key components, major calculation errors, or incomplete reasoning
- Has the right idea but execution has significant problems
- Would receive partial credit (e.g., 2-5 points on a 7-point problem)
- Use this when the student shows some understanding but has major gaps
- Examples: Correct approach but wrong final answer, missing critical steps, major conceptual misunderstanding in part of the solution

**INCORRECT**: The answer is wrong or does not address the problem.
- No valid mathematical reasoning or approach
- Completely wrong approach that shows no understanding
- No substantive work shown or answer is gibberish
- Would receive minimal or no credit (0-1 points on a 7-point problem)
- Use this when the answer shows no understanding of the problem
- Examples: Random guessing, completely irrelevant work, fundamental misunderstanding of the problem

## Decision Framework:
Before selecting a label, ask yourself:
1. Does the student show valid mathematical reasoning? (If NO → INCORRECT)
2. Is the core approach correct? (If NO → INCORRECT or PARTIAL)
3. Are all required steps present and correct? (If NO → PARTIAL or ALMOST)
4. Is the final answer correct? (If NO → PARTIAL or ALMOST)
5. Are errors minor/trivial or major/significant? (Minor → ALMOST, Major → PARTIAL)

## Key Distinctions:
- ALMOST vs PARTIAL: ALMOST has only trivial errors; PARTIAL has significant gaps
- PARTIAL vs INCORRECT: PARTIAL shows some valid reasoning; INCORRECT shows none
- CORRECT vs ALMOST: CORRECT is essentially perfect; ALMOST has minor flaws

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

IMPORTANT: Your response field must contain ONLY one of these four exact words: correct, almost, partial, or incorrect. Do not include any other text."""

        # Collect predictions from multiple runs for self-consistency
        predictions = []
        all_histories = []
        
        for vote_idx in range(self.num_votes):
            logger.info(f"Self-consistency vote {vote_idx + 1}/{self.num_votes}")
            prediction, msg_history = self._get_single_prediction(instruction)
            predictions.append(prediction)
            all_histories.extend(msg_history)
        
        # Use majority voting to determine final prediction
        final_prediction = self._vote_on_predictions(predictions)
        
        return str(final_prediction), all_histories

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
