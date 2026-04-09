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
    
    # Check for partial matches with priority order
    # Priority: almost > partial > incorrect > correct
    # This ordering helps address the issue of "correct" being too greedy
    
    # Check for "almost" first (high priority - often misclassified)
    if "almost" in cleaned or "nearly" in cleaned or "minor" in cleaned:
        return "almost"
    
    # Check for "partial" 
    if "partial" in cleaned or "partly" in cleaned or "some" in cleaned:
        return "partial"
    
    # Check for "incorrect" or "wrong"
    if "incorrect" in cleaned or "wrong" in cleaned or "error" in cleaned or "invalid" in cleaned:
        return "incorrect"
    
    # Check for "correct" last (lowest priority to avoid false positives)
    # Only return "correct" if it's a strong match
    if cleaned == "correct" or cleaned == "right" or cleaned == "true":
        return "correct"
    
    # For ambiguous cases, check surrounding context words
    # Words suggesting "almost": minor, small, trivial, slight, little
    if any(word in cleaned for word in ["minor", "small", "trivial", "slight", "little", "tiny"]):
        return "almost"
    
    # Words suggesting "partial": some, half, incomplete, missing
    if any(word in cleaned for word in ["half", "incomplete", "missing", "inadequate"]):
        return "partial"
    
    # Words suggesting "incorrect": wrong, error, mistake, fail
    if any(word in cleaned for word in ["mistake", "fail", "bad", "no understanding"]):
        return "incorrect"
    
    # Default to incorrect for ambiguous cases to be conservative
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

Example 11 - ALMOST vs PARTIAL boundary (correct method, one significant error):
Problem: Solve 3x + 7 = 22 for x.
Solution: 3x = 15, so x = 5
Student Answer: 3x = 22 - 7 = 15, so x = 3
Analysis: Correct method for isolating x, but made arithmetic error at the end (15/3=3 instead of 5). This is ALMOST because the method is correct and only the final step has a minor error.
<json>
{"response": "almost"}
</json>

Example 12 - PARTIAL vs ALMOST boundary (missing key component):
Problem: Find the roots of x^2 - 5x + 6 = 0.
Solution: (x-2)(x-3) = 0, so x = 2 or x = 3
Student Answer: x^2 - 5x + 6 = (x-2)(x-3), so the roots are 2.
Analysis: Correctly factored but only found one of two roots. Missing a key component makes this PARTIAL, not ALMOST.
<json>
{"response": "partial"}
</json>

Example 13 - PARTIAL vs INCORRECT boundary (some valid work but major gaps):
Problem: Prove the Pythagorean theorem for right triangles.
Solution: [Standard geometric proof]
Student Answer: For a right triangle, a² + b² = c². This is true because squares on the sides add up.
Analysis: States the theorem correctly but provides no actual proof. Has some valid mathematical content but no reasoning. This is PARTIAL (not INCORRECT) because the statement itself is correct.
<json>
{"response": "partial"}
</json>

Example 14 - INCORRECT (no valid mathematical content):
Problem: Find the derivative of sin(x).
Solution: cos(x)
Student Answer: sin(x) becomes smaller as x increases, so the derivative is -1.
Analysis: No valid mathematical reasoning. The explanation is conceptually wrong and doesn't use any correct calculus principles.
<json>
{"response": "incorrect"}
</json>

Example 15 - CORRECT (be strict - must be perfect):
Problem: Simplify (x² - 9)/(x - 3) for x ≠ 3.
Solution: (x² - 9)/(x - 3) = (x+3)(x-3)/(x-3) = x + 3
Student Answer: (x² - 9)/(x - 3) = x + 3
Analysis: While the final answer is correct, the student didn't show the factorization step. However, the problem asks to "simplify" and the answer is correct and complete. This is CORRECT because the answer is perfect for what was asked.
<json>
{"response": "correct"}
</json>

Example 16 - ALMOST (correct answer with minor notation issue):
Problem: Find the indefinite integral of 3x² dx.
Solution: x³ + C
Student Answer: x³
Analysis: Correct computation but forgot the constant of integration (+C). This is a minor notation issue in the context of indefinite integrals. ALMOST.
<json>
{"response": "almost"}
</json>

Example 17 - PARTIAL (correct start but incomplete):
Problem: Solve the system: 2x + y = 7, x - y = 1.
Solution: Adding equations: 3x = 8, so x = 8/3, then y = 5/3
Student Answer: 2x + y = 7, x - y = 1. Adding them: 3x = 8.
Analysis: Started correctly and found 3x = 8, but didn't solve for x or y. Incomplete solution with significant work remaining. PARTIAL.
<json>
{"response": "partial"}
</json>

Example 18 - INCORRECT (fundamental misunderstanding):
Problem: Find the area of a rectangle with length 8 and width 5.
Solution: Area = length × width = 8 × 5 = 40
Student Answer: Area = 8 + 5 = 13
Analysis: Used addition instead of multiplication. Shows fundamental misunderstanding of area concept. INCORRECT.
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
        self.num_votes = 3

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
        """Use majority voting to select the final prediction with conservative tie-breaking.
        
        When there's no clear majority or close votes, prefer the more critical (lower) label.
        Label priority (most critical first): incorrect > partial > almost > correct
        
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
        
        # Get sorted vote counts (most common first)
        sorted_votes = vote_counts.most_common()
        
        if len(sorted_votes) == 1:
            # Unanimous decision
            winner = sorted_votes[0][0]
            logger.info(f"Unanimous vote: '{winner}'")
            return winner
        
        # Check for clear majority (>50%)
        total_votes = len(predictions)
        top_vote_count = sorted_votes[0][1]
        
        if top_vote_count > total_votes / 2:
            # Clear majority
            winner = sorted_votes[0][0]
            logger.info(f"Majority vote: '{winner}' with {top_vote_count}/{total_votes} votes")
            return winner
        
        # No clear majority - use conservative tie-breaking
        # Priority order: incorrect (highest) > partial > almost > correct (lowest)
        priority_order = ["incorrect", "partial", "almost", "correct"]
        
        # Find the highest priority label that received any votes
        for label in priority_order:
            if label in vote_counts:
                winner = label
                logger.info(f"Conservative tie-break: '{winner}' (priority-based from {dict(vote_counts)})")
                return winner
        
        # Fallback (shouldn't reach here)
        winner = sorted_votes[0][0]
        logger.info(f"Fallback vote: '{winner}'")
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

## Grading Rubric - Use These Definitions:

**CORRECT**: The answer is fully correct, complete, and demonstrates thorough understanding.
- ALL required steps are present AND correct
- Final answer matches the solution exactly
- Reasoning is sound and well-explained throughout
- NO errors or omissions of any kind
- Use this ONLY when the answer is essentially perfect - be STRICT
- If there's ANY doubt, do NOT use "correct"

**ALMOST**: The answer is nearly correct with only minor issues.
- Core approach is correct and understanding is demonstrated
- Only ONE minor calculation error or notation issue (e.g., forgot +C in integration, small arithmetic error, sign error)
- The error is truly trivial and doesn't affect the overall reasoning
- Would receive high partial credit (e.g., 6-7/7 on a 7-point problem)
- IMPORTANT: Use this when the student shows correct reasoning with only trivial mistakes
- Examples: small arithmetic slip, minor notation omission, sign error with correct method

**PARTIAL**: The answer is partially correct but has significant gaps or errors.
- Some correct steps or partial understanding shown
- Missing key components OR has significant conceptual errors
- Incomplete solution OR missing final answer
- Multiple errors OR one major error that affects the result
- Would receive partial credit (e.g., 2-5 points on a 7-point problem)
- Use this when the student shows some understanding but has major gaps
- Examples: missing steps, wrong formula but some correct work, incomplete solution

**INCORRECT**: The answer is wrong or does not address the problem.
- No valid mathematical reasoning OR completely wrong approach
- No substantive work shown OR work that doesn't address the problem
- Fundamental misunderstanding of the problem or concepts
- Would receive minimal or no credit (0-1 points on a 7-point problem)
- Use this when the answer shows no understanding of the problem
- Examples: random numbers, unrelated work, blank answer, completely wrong method

## Decision Framework - BE STRICT:
Before selecting a label, ask yourself:
1. Does the student show valid mathematical reasoning? (If NO → INCORRECT)
2. Is the core approach correct? (If NO → INCORRECT or PARTIAL)
3. Are ALL required steps present and correct? (If NO → PARTIAL or ALMOST)
4. Is the final answer correct? (If NO → PARTIAL or ALMOST)
5. Are errors minor/trivial or major/significant? (Minor → ALMOST, Major → PARTIAL)

**CRITICAL**: When in doubt between two labels, choose the LOWER one (more critical).
- Doubt between "correct" and "almost"? → Choose "almost"
- Doubt between "almost" and "partial"? → Choose "partial"
- Doubt between "partial" and "incorrect"? → Choose "incorrect"

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
