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
    
    # Check for exact match with quotes removed
    cleaned_no_quotes = cleaned.strip('"\'').strip()
    if cleaned_no_quotes in valid_labels:
        return cleaned_no_quotes
    
    # Check for partial matches with priority order (more specific first)
    # "almost correct" should map to "almost", not "correct"
    if "almost" in cleaned:
        return "almost"
    if "partial" in cleaned:
        return "partial"
    if "incorrect" in cleaned or "wrong" in cleaned or "error" in cleaned:
        return "incorrect"
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
    
    # Look for explicit labels in quotes (highest priority)
    if '"correct"' in text_lower or "'correct'" in text_lower:
        return "correct"
    if '"almost"' in text_lower or "'almost'" in text_lower:
        return "almost"
    if '"partial"' in text_lower or "'partial'" in text_lower:
        return "partial"
    if '"incorrect"' in text_lower or "'incorrect'" in text_lower:
        return "incorrect"
    
    # Look for labels after common patterns like "response:" or "label:"
    import re
    patterns = [
        r'response[:\s]+["\']?(correct|almost|partial|incorrect)["\']?',
        r'label[:\s]+["\']?(correct|almost|partial|incorrect)["\']?',
        r'grade[:\s]+["\']?(correct|almost|partial|incorrect)["\']?',
        r'evaluation[:\s]+["\']?(correct|almost|partial|incorrect)["\']?',
    ]
    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            return match.group(1)
    
    # Check for keywords with priority (more specific first)
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


# Few-shot examples for better grading accuracy - Enhanced for complex problems
FEW_SHOT_EXAMPLES = """
Example 1 - CORRECT (complete and correct):
Problem: Find the sum of 2 + 3.
Solution: The sum of 2 + 3 is 5.
Student Answer: 5
<json>
{"response": "correct"}
</json>

Example 2 - ALMOST (minor calculation error, otherwise correct):
Problem: Find the area of a circle with radius 5.
Solution: Area = πr² = 25π ≈ 78.54
Student Answer: Area = π × 5² = 3.14159 × 25 = 78.53975 ≈ 78.54
<json>
{"response": "almost"}
</json>

Example 3 - PARTIAL (incomplete - missing one solution):
Problem: Solve x^2 = 4 for x.
Solution: x = 2 or x = -2.
Student Answer: x = 2
<json>
{"response": "partial"}
</json>

Example 4 - INCORRECT (no valid reasoning):
Problem: Prove that the sum of angles in a triangle is 180 degrees.
Solution: [Detailed geometric proof]
Student Answer: It's 180 because that's what I learned.
<json>
{"response": "incorrect"}
</json>

Example 5 - ALMOST (correct work, minor notation issue):
Problem: Find the derivative of x^3.
Solution: The derivative is 3x^2.
Student Answer: d/dx(x^3) = 3x^2
<json>
{"response": "almost"}
</json>

Example 6 - PARTIAL (correct approach but significant error):
Problem: Find all real solutions to x^2 - 5x + 6 = 0.
Solution: Factor as (x-2)(x-3) = 0, so x = 2 or x = 3.
Student Answer: Using the quadratic formula, x = (5 ± sqrt(25-24))/2 = (5 ± 1)/2, so x = 3.
<json>
{"response": "partial"}
</json>

Example 7 - ALMOST vs PARTIAL distinction:
Problem: Solve 2x + 4 = 10.
Solution: x = 3.
Student Answer A (ALMOST): 2x = 6, so x = 3. [Minor: didn't show subtracting 4 from both sides explicitly]
Student Answer B (PARTIAL): 2x = 10, so x = 5. [Major error: forgot to subtract 4]
<json>
{"response": "almost"}
</json>

Example 8 - PARTIAL vs INCORRECT distinction:
Problem: Find the integral of 2x.
Solution: x^2 + C
Student Answer A (PARTIAL): The integral is x^2. [Missing +C, but correct otherwise]
Student Answer B (INCORRECT): The integral is 2. [Completely wrong approach]
<json>
{"response": "partial"}
</json>

Example 9 - CORRECT (complex proof with proper reasoning):
Problem: Prove that for any positive integer n, n^3 - n is divisible by 6.
Solution: Factor as n^3 - n = n(n-1)(n+1). Among three consecutive integers, one is divisible by 2 and one by 3, so product divisible by 6.
Student Answer: We can write n^3 - n = n(n^2-1) = n(n-1)(n+1). These are three consecutive integers. In any three consecutive integers, one is divisible by 2 and one is divisible by 3. Therefore the product is divisible by 6.
<json>
{"response": "correct"}
</json>

Example 10 - ALMOST (complex problem, minor gap in justification):
Problem: Prove that the sum of the first n odd numbers is n^2.
Solution: Use induction or direct formula: 1 + 3 + 5 + ... + (2n-1) = n^2.
Student Answer: The sum is n^2 because each odd number adds the next layer to a square. For n=1: 1=1^2, n=2: 1+3=4=2^2, n=3: 1+3+5=9=3^2. This pattern continues.
<json>
{"response": "almost"}
</json>

Example 11 - PARTIAL (correct start but incomplete):
Problem: Find all prime numbers p such that p+2 and p+6 are also prime.
Solution: Check small primes: p=5 gives 5,7,11 all prime. For p>3, one of p,p+2,p+6 is divisible by 3. So only p=5 works.
Student Answer: Testing p=5: 5, 7, 11 are all prime. Testing p=7: 7, 9, 13 - 9 is not prime. So p=5 works.
<json>
{"response": "partial"}
</json>

Example 12 - INCORRECT (misunderstands problem):
Problem: Prove there are infinitely many primes.
Solution: Assume finite list p1,...,pn. Consider N = p1*p2*...*pn + 1. N is not divisible by any pi, so new prime exists. Contradiction.
Student Answer: There are infinitely many primes because numbers go on forever and some of them must be prime.
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
- Final answer matches the solution
- Reasoning is sound and well-explained
- No significant errors or omissions
- For proofs: valid proof structure with all necessary steps

**ALMOST**: The answer is nearly correct with only minor issues.
- Core approach is correct
- Minor calculation error or notation issue
- Understanding is demonstrated but with small gaps
- Would receive high partial credit (e.g., 7/7 or 6/7 on a 7-point problem)
- For proofs: correct approach with minor gaps in justification

**PARTIAL**: The answer is partially correct but has significant gaps or errors.
- Some correct steps or partial understanding shown
- Missing key components or has significant errors
- Incomplete solution or missing final answer
- Would receive partial credit (e.g., 1-5 points on a 7-point problem)
- For proofs: some valid reasoning but missing critical steps

**INCORRECT**: The answer is wrong or does not address the problem.
- No valid mathematical reasoning
- Completely wrong approach
- No substantive work shown
- Would receive minimal or no credit (0-1 points on a 7-point problem)
- For proofs: no valid proof structure or reasoning

## Critical Distinctions - Read Carefully:

**ALMOST vs PARTIAL**: This is the most important distinction!
- ALMOST = Minor issue only (small calc error, notation slip, one tiny omission). The student clearly understands and would get 6-7/7 points.
- PARTIAL = Significant gaps (missing key steps, major errors, incomplete reasoning). The student shows some understanding but would get 1-5/7 points.

**PARTIAL vs INCORRECT**:
- PARTIAL = Some valid mathematical work shown, even if incomplete or with errors
- INCORRECT = No valid reasoning, completely wrong approach, or no substantive work

## Grading Instructions:
1. Carefully read the problem and understand what is being asked.
2. Compare the student's answer against the correct solution step by step.
3. Check if the student shows valid mathematical work and reasoning.
4. Identify any errors, omissions, or misconceptions.
5. Consider: Would this receive high partial credit (ALMOST) or low partial credit (PARTIAL)?
6. Select the label that best matches the quality of the student's work.

## Chain-of-Thought Analysis (think step by step):
Before giving your final answer, analyze:
- What is the problem asking for?
- What key steps/concepts are required in the solution?
- Which of these does the student's answer contain?
- What errors or omissions exist, and how significant are they?
- Does the student demonstrate understanding of the core concepts?
- If on a 7-point scale, how many points would this receive?
  * 7 points → CORRECT
  * 6-7 points → ALMOST (minor issue only)
  * 1-5 points → PARTIAL (significant gaps but some valid work)
  * 0-1 points → INCORRECT (no valid work)

## Special Considerations for Complex Problems:
- For multi-part problems: check if all parts are addressed
- For proofs: verify logical structure and completeness of argument
- For calculations: check both method and final answer
- Partial credit should be given for correct approach even with errors

## Few-Shot Examples:
{FEW_SHOT_EXAMPLES}

## Response Format:
First provide your chain-of-thought analysis, then respond with EXACTLY ONE of these four labels in JSON format:
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
