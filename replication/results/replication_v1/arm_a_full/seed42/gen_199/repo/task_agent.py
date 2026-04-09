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
    """Extract JSON objects from <json>...</json> blocks or raw JSON.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    Also handles raw JSON objects without tags.
    """
    results = []
    search_from = 0
    
    # First try to find <json>...</json> blocks
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
    
    # If no tagged JSON found, try to find raw JSON objects
    if not results:
        # Look for JSON objects with "response" field
        json_pattern = re.search(r'\{\s*"response"\s*:\s*"([^"]+)"\s*\}', text)
        if json_pattern:
            try:
                results.append(json.loads(json_pattern.group(0)))
            except json.JSONDecodeError:
                pass
    
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
    patterns = [
        r'response[:\s]+["\']?(correct|almost|partial|incorrect)["\']?',
        r'label[:\s]+["\']?(correct|almost|partial|incorrect)["\']?',
        r'grade[:\s]+["\']?(correct|almost|partial|incorrect)["\']?',
        r'evaluation[:\s]+["\']?(correct|almost|partial|incorrect)["\']?',
        r'answer[:\s]+["\']?(correct|almost|partial|incorrect)["\']?',
        r'final[:\s]+["\']?(correct|almost|partial|incorrect)["\']?',
    ]
    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            return match.group(1)
    
    # Look for standalone labels at the end of the text (common pattern)
    # Check last 200 characters for any of the labels
    last_part = text_lower[-200:]
    for label in ["correct", "almost", "partial", "incorrect"]:
        # Check if label appears as a standalone word
        if re.search(r'\b' + label + r'\b', last_part):
            return label
    
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


# Few-shot examples for better grading accuracy
FEW_SHOT_EXAMPLES = """
Example 1 - CORRECT (perfect match):
Problem: Find the sum of 2 + 3.
Solution: 5
Student Answer: 5
<json>
{"response": "correct"}
</json>

Example 2 - ALMOST (tiny arithmetic slip):
Problem: Find the area of a circle with radius 5.
Solution: Area = πr² = 25π ≈ 78.54
Student Answer: Area = π × 5² = 3.14159 × 25 = 78.53975
Label: ALMOST (only 0.00025 difference from correct value, method perfect)
<json>
{"response": "almost"}
</json>

Example 3 - PARTIAL (missing one of two answers):
Problem: Solve x^2 = 4 for x.
Solution: x = 2 or x = -2.
Student Answer: x = 2
Label: PARTIAL (found one valid solution but missed the other)
<json>
{"response": "partial"}
</json>

Example 4 - INCORRECT (no mathematical reasoning):
Problem: Prove that the sum of angles in a triangle is 180 degrees.
Solution: [Detailed geometric proof with parallel lines]
Student Answer: It's 180 because that's what I learned in class.
Label: INCORRECT (no proof, just states a fact without reasoning)
<json>
{"response": "incorrect"}
</json>

Example 5 - ALMOST (correct answer, minor notation):
Problem: Find the derivative of x^3.
Solution: 3x^2
Student Answer: d/dx(x^3) = 3x^2
Label: ALMOST (answer is correct but notation slightly informal)
<json>
{"response": "almost"}
</json>

Example 6 - PARTIAL (major calculation error, some right steps):
Problem: Find all real solutions to x^2 - 5x + 6 = 0.
Solution: x = 2 or x = 3 (factors as (x-2)(x-3)=0).
Student Answer: Using quadratic formula: x = (5 ± √1)/2, so x = 3. [Calculation error: should be x = 2 or x = 3]
Label: PARTIAL (used correct method but made arithmetic error, got one answer right)
<json>
{"response": "partial"}
</json>

Example 7 - CORRECT (complete rigorous proof):
Problem: Prove n^3 - n is divisible by 6 for any positive integer n.
Solution: Factor as n(n-1)(n+1). Three consecutive integers, one divisible by 2 and one by 3.
Student Answer: n^3 - n = n(n-1)(n+1). Three consecutive integers, one divisible by 2 and one by 3. Product divisible by 6.
<json>
{"response": "correct"}
</json>

Example 8 - ALMOST (correct insight, tiny gap in rigor):
Problem: Prove sum of first n odd numbers is n^2.
Solution: 1 + 3 + 5 + ... + (2n-1) = n^2 by induction or direct proof.
Student Answer: Pattern: 1=1^2, 1+3=4=2^2, 1+3+5=9=3^2. This pattern continues for all n.
Label: ALMOST (correct insight, just needs explicit "continues by induction" statement)
<json>
{"response": "almost"}
</json>

Example 9 - PARTIAL (example only, no general proof):
Problem: Find all primes p where p+2 and p+6 are also prime.
Solution: Only p=5 works. For p>3, one of p,p+2,p+6 is divisible by 3.
Student Answer: Testing p=5: 5, 7, 11 are all prime. So p=5 works.
Label: PARTIAL (found correct answer but no proof it's the only one, no general argument)
<json>
{"response": "partial"}
</json>

Example 10 - INCORRECT (circular/wrong reasoning):
Problem: Prove there are infinitely many primes.
Solution: Assume finite list p1,...,pn. Consider N = p1*p2*...*pn + 1. Either N is prime or has a prime factor not in the list. Contradiction.
Student Answer: Numbers go on forever, so some must be prime. Infinity means never ending.
Label: INCORRECT (no valid proof, just restates the claim in different words)
<json>
{"response": "incorrect"}
</json>

Example 11 - ALMOST vs CORRECT distinction:
Problem: Compute 2^10.
Solution: 1024
Student Answer: 1023
Label: ALMOST (one small arithmetic error, method of exponentiation was correct)
<json>
{"response": "almost"}
</json>

Example 12 - PARTIAL vs INCORRECT distinction:
Problem: Solve 2x + 4 = 10.
Solution: x = 3 (subtract 4, divide by 2).
Student Answer: 2x = 10, so x = 5. [Forgot to subtract 4 first]
Label: PARTIAL (attempted valid algebra step but made error, shows some understanding)
<json>
{"response": "partial"}
</json>

Example 13 - INCORRECT (completely wrong approach):
Problem: Find the integral of x^2.
Solution: x^3/3 + C
Student Answer: x^2/2 + C (used derivative rule instead of integral)
Label: INCORRECT (fundamental misunderstanding of calculus operation)
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
        
        instruction = f"""You are an expert grading agent for {domain}. Grade the student's answer by comparing it to the correct solution.

## Problem:
{problem}

## Correct Solution:
{solution}

## Student's Answer:
{student_answer}

## Grading Rubric:

**CORRECT**: Fully correct and complete. All steps present, reasoning sound, final answer matches. (7/7 points)

**ALMOST**: Nearly correct with only minor issues. Core approach correct, minor calculation/notation error. Would get 6-7/7 points.

**PARTIAL**: Partially correct with significant gaps. Some valid work shown but missing key components or has major errors. Would get 1-5/7 points.

**INCORRECT**: Wrong or doesn't address the problem. No valid reasoning, completely wrong approach, or no substantive work. (0-1/7 points)

## Critical Distinctions (READ CAREFULLY):

**ALMOST vs CORRECT**: 
- CORRECT = Perfect or near-perfect with no meaningful errors
- ALMOST = One tiny slip (arithmetic error, minor notation issue) but core understanding is clear

**ALMOST vs PARTIAL**:
- ALMOST = Student clearly knows the method, just made a small mistake
- PARTIAL = Missing significant parts OR has major conceptual gaps OR only got part of a multi-part answer

**PARTIAL vs INCORRECT**:
- PARTIAL = Shows SOME valid mathematical reasoning or correct steps
- INCORRECT = No valid reasoning, completely wrong method, or just states answer without work

## Decision Tree:
1. Is the answer fully correct with all reasoning? → CORRECT
2. Is there just one tiny error but method is perfect? → ALMOST
3. Are there significant gaps but some valid work? → PARTIAL
4. Is there no valid reasoning or completely wrong? → INCORRECT

## Instructions:
1. Compare the student's answer to the correct solution step by step.
2. Check if the student shows valid mathematical work and reasoning.
3. Identify any errors, omissions, or misconceptions.
4. Use the decision tree above to select the label.
5. Be generous with ALMOST for minor slips - don't penalize tiny errors harshly.
6. Be strict with INCORRECT - require actual mathematical reasoning, not just claims.

## Examples:
{FEW_SHOT_EXAMPLES}

## Response Format:
First, briefly explain your reasoning (1-2 sentences). Then respond with EXACTLY ONE of these four labels in JSON format:

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
                # Get the assistant's response text
                assistant_msg = msg_history[-1]
                text = assistant_msg.get("text", "") if isinstance(assistant_msg, dict) else str(assistant_msg)
                
                # Try to extract JSON first
                extracted = _extract_jsons(text)
                if extracted and len(extracted) > 0:
                    last_json = extracted[-1]
                    if isinstance(last_json, dict) and "response" in last_json:
                        raw_prediction = last_json["response"]
                        # Normalize and validate the prediction
                        prediction = _normalize_prediction(raw_prediction)
                        logger.info(f"Successfully extracted prediction from JSON: {prediction}")
                    else:
                        # Try to extract label from raw text if JSON doesn't have response field
                        prediction = _extract_label_from_text(text)
                        logger.info(f"Extracted label from text (no response field): {prediction}")
                else:
                    # Try to extract label from raw text if JSON parsing fails
                    prediction = _extract_label_from_text(text)
                    logger.info(f"Extracted label from text (no JSON): {prediction}")
            else:
                logger.warning("Empty message history from LLM")
        except Exception as e:
            self.error_count += 1
            self.log_fn(f"Error extracting prediction: {e}")
            # Try to extract from response text as fallback
            try:
                if response:
                    prediction = _extract_label_from_text(response)
                    logger.info(f"Extracted label from response fallback: {prediction}")
            except Exception as e2:
                logger.error(f"Fallback extraction also failed: {e2}")

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
