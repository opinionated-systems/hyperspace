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
    
    # Check for partial matches - order matters!
    # Check more specific terms before general ones to avoid misclassification
    if "almost" in cleaned:
        return "almost"
    if "partial" in cleaned:
        return "partial"
    if "incorrect" in cleaned or "wrong" in cleaned:
        return "incorrect"
    # Only match "correct" if it doesn't contain other labels
    if "correct" in cleaned and "almost" not in cleaned and "partial" not in cleaned and "incorrect" not in cleaned:
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
    
    # Look for explicit labels in quotes first (most reliable)
    # Check in order: almost, partial, incorrect, correct (to avoid "correct" matching too eagerly)
    if "\"almost\"" in text_lower or "'almost'" in text_lower:
        return "almost"
    if "\"partial\"" in text_lower or "'partial'" in text_lower:
        return "partial"
    if "\"incorrect\"" in text_lower or "'incorrect'" in text_lower:
        return "incorrect"
    if "\"correct\"" in text_lower or "'correct'" in text_lower:
        return "correct"
    
    # Check for keywords in the text
    # Order matters: check more specific terms before general ones
    if "almost" in text_lower:
        return "almost"
    if "partial" in text_lower:
        return "partial"
    if "incorrect" in text_lower or "wrong" in text_lower:
        return "incorrect"
    # Only match "correct" if not preceded by "in" (to avoid "incorrect")
    if "correct" in text_lower and "incorrect" not in text_lower:
        return "correct"
    
    # Default to incorrect
    logger.warning(f"Could not extract label from text, defaulting to 'incorrect'")
    return "incorrect"


# Few-shot examples for better grading accuracy
FEW_SHOT_EXAMPLES = """
Example 1 - CORRECT (Complete and perfect):
Problem: Find the sum of 2 + 3.
Solution: The sum of 2 + 3 is 5.
Student Answer: 5
Reasoning: The answer is exactly correct with no errors or omissions.
<json>
{"response": "correct"}
</json>

Example 2 - CORRECT (Complete proof):
Problem: Prove that for any positive integer n, n^3 - n is divisible by 6.
Solution: Factor as n(n-1)(n+1). Among three consecutive integers, one is divisible by 2 and one by 3, so product is divisible by 6.
Student Answer: We can write n^3 - n = n(n^2 - 1) = n(n-1)(n+1). These are three consecutive integers. In any three consecutive integers, there must be one divisible by 2 and one divisible by 3. Therefore the product is divisible by 6.
Reasoning: Complete proof with all steps correct and clearly explained.
<json>
{"response": "correct"}
</json>

Example 3 - ALMOST (Minor calculation error):
Problem: Compute 15 * 12.
Solution: 15 * 12 = 180.
Student Answer: 15 * 12 = 150 + 30 = 190. (Student made a small addition error: 150+30=180, not 190)
Reasoning: The student used the correct method (15*12 = 15*10 + 15*2 = 150 + 30) but made a tiny arithmetic error at the end. The approach is correct, just a minor slip.
<json>
{"response": "almost"}
</json>

Example 4 - ALMOST (Correct but missing trivial final step):
Problem: Solve for x: 2x + 4 = 10.
Solution: 2x = 6, so x = 3.
Student Answer: 2x = 6. (Student stopped here, didn't write x=3 explicitly, but the work is essentially complete)
Reasoning: The student did all the hard work correctly, just didn't state the final answer explicitly. This is a minor omission in an otherwise correct solution.
<json>
{"response": "almost"}
</json>

Example 5 - PARTIAL (Missing one of multiple solutions):
Problem: Solve x^2 = 4 for x.
Solution: x = 2 or x = -2.
Student Answer: x = 2
Reasoning: The student found one correct solution but completely missed the other. This is a significant omission - only half the answer is provided.
<json>
{"response": "partial"}
</json>

Example 6 - PARTIAL (Correct approach but missing key step):
Problem: Find all real solutions to x^2 - 5x + 6 = 0.
Solution: Factor as (x-2)(x-3) = 0, so x = 2 or x = 3.
Student Answer: Using the quadratic formula, x = (5 ± sqrt(25-24))/2 = (5 ± 1)/2, so x = 3.
Reasoning: The student used the correct method but only found one of the two solutions. Significant portion missing.
<json>
{"response": "partial"}
</json>

Example 7 - PARTIAL (Some correct work but incomplete):
Problem: Find the derivative of f(x) = x^3 + 2x + 5.
Solution: f'(x) = 3x^2 + 2
Student Answer: f'(x) = 3x^2
Reasoning: The student correctly differentiated x^3 but completely missed the 2x term. The constant 5 was also ignored (which is correct), but missing the 2x term is a significant error.
<json>
{"response": "partial"}
</json>

Example 8 - INCORRECT (No valid reasoning):
Problem: Prove that the sum of angles in a triangle is 180 degrees.
Solution: [Detailed geometric proof]
Student Answer: It's 180 because that's what I learned.
Reasoning: No mathematical reasoning provided. Just a statement without justification.
<json>
{"response": "incorrect"}
</json>

Example 9 - INCORRECT (Wrong approach):
Problem: Find the area of a circle with radius 3.
Solution: A = πr² = 9π
Student Answer: A = 2πr = 6π
Reasoning: The student confused area formula with circumference formula. Completely wrong method.
<json>
{"response": "incorrect"}
</json>

Example 10 - INCORRECT (Nonsensical answer):
Problem: Solve 2x + 3 = 7.
Solution: 2x = 4, so x = 2.
Student Answer: x = 5 because 2+3=5.
Reasoning: The student completely misunderstood the equation, adding constants instead of solving properly.
<json>
{"response": "incorrect"}
</json>

Example 11 - ALMOST (Sign error in final answer):
Problem: Solve for x: x^2 = 9.
Solution: x = 3 or x = -3.
Student Answer: x = ±3 (student wrote ±3 but meant to show both 3 and -3, or wrote x=3,-3 with a formatting issue)
Reasoning: The student clearly knows both solutions exist and indicated them correctly, just with minor notation formatting. Core understanding is complete.
<json>
{"response": "almost"}
</json>

Example 12 - ALMOST (Correct method, small arithmetic slip):
Problem: Find the area of a rectangle with length 8 and width 5.
Solution: Area = 8 × 5 = 40.
Student Answer: 8 × 5 = 41 (small addition error, but clearly knows to multiply length × width).
Reasoning: The student clearly understands the concept of area and the correct formula. A tiny arithmetic error doesn't change that they fully understand the problem.
<json>
{"response": "almost"}
</json>

Example 13 - PARTIAL (Missing critical verification step):
Problem: Prove that n^2 + n is always even for integer n.
Solution: n^2 + n = n(n+1). Either n is even, or n+1 is even, so product is even.
Student Answer: n^2 + n = n(n+1). The product of two consecutive integers is even.
Reasoning: The student set up the factorization correctly but didn't explain WHY the product of consecutive integers is even. Missing the key reasoning step that one must be even.
<json>
{"response": "partial"}
</json>

Example 14 - ALMOST (Correct answer with minor notation issue):
Problem: Find the derivative of f(x) = 3x^2.
Solution: f'(x) = 6x.
Student Answer: 6x (correct value but forgot to write f'(x) = or dy/dx notation).
Reasoning: The calculation is completely correct. Only missing the formal notation which is a trivial omission.
<json>
{"response": "almost"}
</json>

Example 15 - PARTIAL vs ALMOST distinction:
Problem: Solve the system: x + y = 10, x - y = 2.
Solution: Adding equations: 2x = 12, so x = 6. Then y = 4.
Student Answer: x = 6 (found x correctly but didn't find y).
Reasoning: This is PARTIAL not ALMOST because finding only one variable when the problem asks for the solution to a system is a significant omission. The student needs to find BOTH values.
<json>
{"response": "partial"}
</json>

Example 16 - INCORRECT (No valid work):
Problem: Find the sum of the first 10 positive integers.
Solution: Using the formula n(n+1)/2: 10*11/2 = 55.
Student Answer: 100 (no work shown, just a random guess).
Reasoning: No mathematical reasoning provided. The answer is wrong and there's no indication of how the student arrived at it.
<json>
{"response": "incorrect"}
</json>

Example 17 - PARTIAL (Some valid setup, wrong conclusion):
Problem: Prove that if n is odd, then n² is odd.
Solution: Let n = 2k+1. Then n² = (2k+1)² = 4k² + 4k + 1 = 2(2k² + 2k) + 1, which is odd.
Student Answer: Let n = 2k+1. Then n² = 4k² + 1, which is odd.
Reasoning: The student correctly started with n = 2k+1 but made an error expanding (2k+1)². However, they showed some valid algebraic work and had the right approach.
<json>
{"response": "partial"}
</json>

Example 18 - ALMOST (Correct with trivial typo):
Problem: Find the value of 7 × 8.
Solution: 7 × 8 = 56.
Student Answer: 7 × 8 = 56 (student wrote "7 x 8 = 56" with a lowercase x for multiplication).
Reasoning: The answer is completely correct. Using "x" instead of "×" is a trivial notation issue that doesn't affect correctness.
<json>
{"response": "almost"}
</json>

Example 19 - PARTIAL (Incomplete proof):
Problem: Prove that the product of two consecutive integers is even.
Solution: Let the integers be n and n+1. If n is even, done. If n is odd, then n+1 is even. In either case, the product is even.
Student Answer: The product of two consecutive integers is even because one of them must be even.
Reasoning: The student stated the correct conclusion but didn't provide the case analysis proof. Missing the key reasoning steps makes this PARTIAL, not ALMOST.
<json>
{"response": "partial"}
</json>

Example 20 - INCORRECT (Fundamental misunderstanding):
Problem: Find the derivative of f(x) = x³.
Solution: f'(x) = 3x² using the power rule.
Student Answer: f'(x) = x² because the exponent decreases by 1.
Reasoning: The student misunderstood the power rule - they only decreased the exponent but forgot to multiply by the original exponent. This shows a fundamental misunderstanding of the rule.
<json>
{"response": "incorrect"}
</json>

Example 21 - ALMOST (Correct with minor formatting):
Problem: Solve for x: 3x = 12.
Solution: x = 4.
Student Answer: x=4 (no spaces around the equals sign).
Reasoning: The answer is mathematically correct. Formatting preference (spaces around =) is a trivial issue.
<json>
{"response": "almost"}
</json>

Example 22 - PARTIAL (Right idea, major execution error):
Problem: Compute ∫(2x + 3)dx.
Solution: x² + 3x + C.
Student Answer: x² + 3 + C (forgot to integrate the constant 3 properly).
Reasoning: The student understood integration but made a significant error with the constant term. This is more than a tiny slip - they missed a fundamental part of the integration.
<json>
{"response": "partial"}
</json>

Example 23 - INCORRECT (Completely wrong approach):
Problem: Find the area of a triangle with base 4 and height 3.
Solution: Area = (1/2) × base × height = (1/2) × 4 × 3 = 6.
Student Answer: Area = base + height = 4 + 3 = 7.
Reasoning: The student used completely the wrong formula (perimeter instead of area). No valid area calculation was performed.
<json>
{"response": "incorrect"}
</json>

Example 24 - ALMOST (Correct with insignificant error):
Problem: Simplify (x²)³.
Solution: x⁶.
Student Answer: x^6 (used caret notation instead of superscript).
Reasoning: The mathematical answer is correct. Notation style is a trivial issue.
<json>
{"response": "almost"}
</json>
"""


class TaskAgent:
    """Task agent that solves IMO grading problems with improved reliability."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "", use_ensemble: bool = True, ensemble_size: int = 3) -> None:
        self.model = model
        self.log_fn = logger.info
        self.call_count = 0
        self.error_count = 0
        self.use_ensemble = use_ensemble
        self.ensemble_size = ensemble_size
        
    def _grade_with_confidence(self, inputs: dict) -> tuple[str, float, list[dict]]:
        """Grade a single problem and return prediction with confidence score.
        
        Uses multiple LLM calls with slightly varied prompts to compute
        a confidence score based on agreement between calls.
        
        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer
            
        Returns:
            (prediction, confidence_score, msg_history)
            confidence_score is between 0.0 and 1.0
        """
        votes = {"correct": 0, "almost": 0, "partial": 0, "incorrect": 0}
        all_histories = []
        
        # Run multiple grading attempts with temperature variation for diversity
        for i in range(self.ensemble_size):
            # Vary the prompt slightly for each ensemble member
            variation_prefixes = [
                "",
                "As an expert mathematics grader, ",
                "Carefully evaluating this solution: ",
            ]
            
            # Create modified inputs with variation
            modified_inputs = inputs.copy()
            if i < len(variation_prefixes):
                modified_inputs["_variation_prefix"] = variation_prefixes[i]
            
            prediction, history = self._single_grade(modified_inputs)
            all_histories.extend(history)
            
            if prediction in votes:
                votes[prediction] += 1
            else:
                votes["incorrect"] += 1  # Default for unexpected responses
        
        # Find the label with most votes
        max_votes = max(votes.values())
        winning_label = max(votes, key=votes.get)
        
        # Calculate confidence as proportion of agreement
        confidence = max_votes / self.ensemble_size
        
        # If there's a tie, prefer the more conservative label
        tied_labels = [label for label, count in votes.items() if count == max_votes]
        if len(tied_labels) > 1:
            # Preference order: incorrect > partial > almost > correct (more conservative first)
            preference_order = ["incorrect", "partial", "almost", "correct"]
            for preferred in preference_order:
                if preferred in tied_labels:
                    winning_label = preferred
                    confidence = max_votes / self.ensemble_size  # Lower confidence due to tie
                    break
        
        # If confidence is low (no clear majority), be more conservative
        # This helps avoid over-predicting "correct" or "almost" when uncertain
        if confidence < 0.67:  # Less than 2/3 agreement
            # Check if we should downgrade the prediction
            current_idx = ["correct", "almost", "partial", "incorrect"].index(winning_label)
            # Look for a more conservative label that has significant support
            for label in ["partial", "incorrect"]:
                if votes[label] >= self.ensemble_size // 2:  # At least half support
                    if ["correct", "almost", "partial", "incorrect"].index(label) > current_idx:
                        logger.info(f"Low confidence ({confidence:.2f}), downgrading from {winning_label} to {label}")
                        winning_label = label
                        confidence = votes[label] / self.ensemble_size
                        break
        
        logger.info(f"Ensemble voting results: {votes}, selected: {winning_label} (confidence: {confidence:.2f})")
        
        return winning_label, confidence, all_histories
    
    def _single_grade(self, inputs: dict) -> tuple[str, list[dict]]:
        """Perform a single grading call (internal method)."""
        return self._forward_single(inputs)

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with retry logic.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Use ensemble grading if enabled, otherwise use single grading
        if self.use_ensemble:
            prediction, confidence, msg_history = self._grade_with_confidence(inputs)
            # Store confidence in the last message for potential external use
            if msg_history:
                msg_history[-1]["_confidence"] = confidence
            return prediction, msg_history
        else:
            return self._forward_single(inputs)
    
    def _forward_single(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run a single grading attempt on a problem."""
        self.call_count += 1
        logger.info(f"TaskAgent forward call #{self.call_count}")
        
        # Extract fields from inputs for better structured prompting
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        variation_prefix = inputs.get("_variation_prefix", "")
        
        instruction = f"""{variation_prefix}You are an expert grading agent for {domain}. Your task is to carefully analyze the student's answer and provide a detailed evaluation.

## Problem Statement:
{problem}

## Correct Solution:
{solution}

## Grading Guidelines:
{grading_guidelines}

## Student's Answer:
{student_answer}

## Grading Rubric - Use These Definitions CAREFULLY:

**CORRECT**: The answer is fully correct, complete, and demonstrates thorough understanding.
- All required steps are present and correct
- Final answer matches the solution exactly
- Reasoning is sound and well-explained
- No errors, omissions, or issues of any significance
- Would receive full credit (7/7 on a 7-point problem)

**ALMOST**: The answer is nearly correct with ONLY minor issues that don't affect the core understanding.
- Core approach and all major steps are correct
- Only tiny calculation errors (e.g., 2+2=5) or trivial notation issues
- Missing only the most trivial final step (e.g., stopped at "2x=6" instead of "x=3")
- Understanding is clearly demonstrated
- Would receive high partial credit (6/7 or 7/7 on a 7-point problem)
- KEY DISTINCTION: The error is so minor it barely affects the solution quality

**PARTIAL**: The answer has SIGNIFICANT gaps or errors, but shows some valid work.
- Some correct steps or partial understanding shown
- Missing KEY components (e.g., only found 1 of 2 solutions, missed major terms)
- Has significant errors that affect the result
- Incomplete solution with substantial portions missing
- Would receive partial credit (1-5 points on a 7-point problem)
- KEY DISTINCTION: There's real missing content or significant errors, not just tiny slips

**INCORRECT**: The answer is wrong or shows no valid mathematical reasoning.
- No valid mathematical reasoning or approach
- Completely wrong method or nonsensical answer
- No substantive work shown
- Would receive minimal or no credit (0-1 points on a 7-point problem)

## Critical Decision Guidelines:

**ALMOST vs PARTIAL - This is the most important distinction:**

Use **ALMOST** when ALL of these are true:
- The student clearly demonstrates understanding of the core concept
- The error is truly minor (small arithmetic slip, trivial notation issue)
- The solution is essentially complete with only a tiny gap
- The error doesn't affect the overall correctness of the approach
- Examples: 2+2=5, forgot to write "x=" before the answer, stopped at "2x=6" instead of "x=3"

Use **PARTIAL** when ANY of these are true:
- Missing significant portions of the answer (e.g., only 1 of 2 solutions)
- Missing key reasoning steps or explanations
- Significant conceptual errors mixed with some correct work
- The solution is incomplete in a meaningful way
- Examples: "x^2=4, student says x=2" (missing -2), "found x but not y in a system"

**Quick Test:** If the student fixed their error in 10 seconds with a quick check, it's ALMOST. If they would need to redo significant work, it's PARTIAL.

**More Examples:**
- "x^2=4, student says x=2" → PARTIAL (missing -2 is significant, not trivial)
- "2x+4=10, student gets 2x=6 but doesn't solve for x" → ALMOST (trivial final step)
- "Student uses correct formula but calculates 15×12=190 instead of 180" → ALMOST (small arithmetic error)
- "Student misses half the terms in a derivative" → PARTIAL (significant omission)
- "Student solves system correctly for x=6 but doesn't find y=4" → PARTIAL (missing half the answer)

**INCORRECT vs PARTIAL - Another critical distinction:**

Use **INCORRECT** when:
- No valid mathematical reasoning is shown
- The approach is fundamentally wrong
- Answer is nonsensical or unrelated to the problem
- Would receive 0-1 points on a 7-point problem

Use **PARTIAL** when:
- At least some valid mathematical work is shown
- The student demonstrates partial understanding
- There are significant errors but also some correct steps
- Would receive 1-5 points on a 7-point problem

**Examples:**
- "Student writes random numbers unrelated to the problem" → INCORRECT
- "Student sets up the equation correctly but solves it completely wrong" → PARTIAL (some valid work shown)
- "Student uses a theorem that doesn't apply" → INCORRECT (wrong approach)
- "Student has the right idea but makes major errors in execution" → PARTIAL

## Grading Instructions:
1. Carefully read the problem and understand what is being asked.
2. Compare the student's answer against the correct solution step by step.
3. Check if the student shows valid mathematical work and reasoning.
4. Identify any errors, omissions, or misconceptions.
5. **CRITICAL STEP - Classify the error severity:**
   - Is it a tiny arithmetic slip or trivial omission? → Consider ALMOST
   - Is it missing significant content or has major errors? → Consider PARTIAL
   - Use the "10 second test": Could they fix it with a quick check? → ALMOST
   - Would they need to redo work? → PARTIAL
6. Select the label that best matches the quality of the student's work.

## Decision Flowchart:
```
Is the answer completely correct? → CORRECT
Is the answer completely wrong or nonsense? → INCORRECT
Does it have some valid work? → Continue...
  Is the error tiny (arithmetic slip, trivial omission)? → ALMOST
  Is there significant missing content or major errors? → PARTIAL
```

## Few-Shot Examples:
{FEW_SHOT_EXAMPLES}

## Your Grading Process:
Before selecting a label, explicitly think through:
1. What is the problem asking for?
2. What does the correct solution require?
3. What did the student actually provide?
4. What errors or omissions exist?
5. Are the errors truly minor (ALMOST) or significant (PARTIAL)?

## Response Format:
You must respond with EXACTLY ONE of these four labels in JSON format:
- "correct" - The answer is fully correct and complete
- "almost" - The answer is nearly correct with ONLY truly minor issues  
- "partial" - The answer has SIGNIFICANT gaps or errors (but some valid work)
- "incorrect" - The answer is wrong or shows no valid reasoning

<json>
{{
    "response": "correct|almost|partial|incorrect"
}}
</json>

IMPORTANT: Your response field must contain ONLY one of these four exact words: correct, almost, partial, or incorrect. Do not include any other text."""

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
            Dictionary with call_count, error_count, and ensemble info
        """
        return {
            "call_count": self.call_count,
            "error_count": self.error_count,
            "success_rate": (self.call_count - self.error_count) / max(self.call_count, 1),
            "use_ensemble": self.use_ensemble,
            "ensemble_size": self.ensemble_size if self.use_ensemble else 1,
            "effective_calls": self.call_count * (self.ensemble_size if self.use_ensemble else 1)
        }
    
    def set_ensemble_mode(self, enabled: bool, size: int = 3) -> None:
        """Enable or disable ensemble grading mode.
        
        Args:
            enabled: Whether to use ensemble grading
            size: Number of ensemble members (default 3)
        """
        self.use_ensemble = enabled
        self.ensemble_size = max(2, size) if enabled else 1
        logger.info(f"Ensemble mode {'enabled' if enabled else 'disabled'} with size {self.ensemble_size}")
