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
    
    # Clean up the prediction - remove all non-alphanumeric except spaces
    cleaned = prediction.strip().lower()
    cleaned = cleaned.strip(".!?,:;\"'()[]{}").strip()
    
    # Remove extra whitespace and normalize
    cleaned = " ".join(cleaned.split())
    
    # Map to valid labels - check for exact match first
    valid_labels = ["correct", "almost", "partial", "incorrect"]
    if cleaned in valid_labels:
        return cleaned
    
    # Check for quoted versions
    for label in valid_labels:
        if f'"{label}"' in cleaned or f"'{label}'" in cleaned:
            return label
    
    # Check for standalone words (avoid matching "correct" inside "incorrect")
    words = cleaned.split()
    for word in words:
        word = word.strip(".!?,:;\"'()[]{}")
        if word in valid_labels:
            return word
    
    # Check for partial matches with priority order
    # Check for "incorrect" first to avoid matching "correct" inside it
    if "incorrect" in cleaned or "wrong" in cleaned:
        return "incorrect"
    if "almost" in cleaned:
        return "almost"
    if "partial" in cleaned:
        return "partial"
    # Check for "correct" last to avoid false positives
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
    for label in ["incorrect", "almost", "partial", "correct"]:
        if f'"{label}"' in text_lower or f"'{label}'" in text_lower:
            return label
    
    # Look for labels after "response": patterns
    import re
    response_pattern = r'["\']?response["\']?\s*:\s*["\']?\s*(correct|almost|partial|incorrect)\s*["\']?'
    match = re.search(response_pattern, text_lower)
    if match:
        return match.group(1)
    
    # Check for standalone words in the text (avoid matching "correct" inside "incorrect")
    # Priority: incorrect > almost > partial > correct
    words = re.findall(r'\b\w+\b', text_lower)
    
    if "incorrect" in words or "wrong" in words:
        return "incorrect"
    if "almost" in words:
        return "almost"
    if "partial" in words:
        return "partial"
    if "correct" in words:
        return "correct"
    
    # Check for substring matches as last resort
    if "incorrect" in text_lower:
        return "incorrect"
    if "almost" in text_lower:
        return "almost"
    if "partial" in text_lower:
        return "partial"
    if "correct" in text_lower:
        return "correct"
    
    # Default to incorrect
    logger.warning(f"Could not extract label from text, defaulting to 'incorrect'")
    return "incorrect"


# Few-shot examples for better grading accuracy
FEW_SHOT_EXAMPLES = """
Example 1 - CORRECT (complete and correct):
Problem: Find the sum of 2 + 3.
Solution: The sum of 2 + 3 is 5.
Student Answer: 5
Analysis: The answer is fully correct and complete.
<json>
{"response": "correct"}
</json>

Example 2 - ALMOST (minor notation issue only):
Problem: Find the derivative of x^3.
Solution: The derivative is 3x^2.
Student Answer: d/dx(x^3) = 3x^2
Analysis: The answer is correct but uses different notation. This is a minor issue only - the student clearly understands the concept and would receive 6-7/7 points.
<json>
{"response": "almost"}
</json>

Example 3 - ALMOST (tiny arithmetic slip, correct approach):
Problem: Find 15 × 4.
Solution: 60
Student Answer: 15 × 4 = 16 × 4 - 4 = 64 - 4 = 60
Analysis: The student used a clever approach and got the right answer, just had a small slip (16×4) that they self-corrected. Minor issue only - 6-7/7 points.
<json>
{"response": "almost"}
</json>

Example 4 - ALMOST (correct answer, minor presentation issue):
Problem: Solve x^2 - 9 = 0.
Solution: x = 3 or x = -3
Student Answer: x = ±3
Analysis: The answer is mathematically correct. Using ± notation is acceptable and shows understanding. Minor presentation difference only - 6-7/7 points.
<json>
{"response": "almost"}
</json>

Example 5 - PARTIAL (incomplete - missing solutions):
Problem: Solve x^2 = 4 for x.
Solution: x = 2 or x = -2.
Student Answer: x = 2
Analysis: The student found one solution but missed the other. This is a significant gap - they would receive partial credit of about 3-4/7 points.
<json>
{"response": "partial"}
</json>

Example 6 - PARTIAL (correct start but major error in execution):
Problem: Find all real solutions to x^2 - 5x + 6 = 0.
Solution: Factor as (x-2)(x-3) = 0, so x = 2 or x = 3.
Student Answer: Using the quadratic formula, x = (5 ± sqrt(25-24))/2 = (5 ± 1)/2, so x = 3.
Analysis: The student used the correct method but made a sign error, getting only one solution instead of two. Significant error - would receive about 3-4/7 points.
<json>
{"response": "partial"}
</json>

Example 7 - PARTIAL (missing key steps):
Problem: Prove the Pythagorean theorem.
Solution: [Complete geometric proof with diagram]
Student Answer: a² + b² = c². This is true for right triangles.
Analysis: The student states the theorem correctly but provides no proof or reasoning. Missing all key steps - would receive about 2-3/7 points.
<json>
{"response": "partial"}
</json>

Example 8 - PARTIAL (incomplete reasoning):
Problem: Find the area of a triangle with base 6 and height 4.
Solution: Area = (1/2) × base × height = (1/2) × 6 × 4 = 12
Student Answer: Area = 6 × 4 = 24
Analysis: The student forgot the 1/2 factor but showed some understanding of the concept. Significant error - would receive about 2-3/7 points.
<json>
{"response": "partial"}
</json>

Example 9 - INCORRECT (no valid mathematical reasoning):
Problem: Prove that the sum of angles in a triangle is 180 degrees.
Solution: [Detailed geometric proof]
Student Answer: It's 180 because that's what I learned in school.
Analysis: No mathematical reasoning provided. Just a statement without any work - 0-1/7 points.
<json>
{"response": "incorrect"}
</json>

Example 10 - INCORRECT (completely wrong approach):
Problem: Find the integral of 2x.
Solution: x^2 + C
Student Answer: The integral is 2 because you divide by the power.
Analysis: The student has no understanding of integration. Completely wrong approach - 0-1/7 points.
<json>
{"response": "incorrect"}
</json>

Example 11 - INCORRECT (irrelevant answer):
Problem: Solve for x: 3x + 7 = 22
Solution: x = 5
Student Answer: The answer is y = 3.
Analysis: The student answered a different question entirely. No valid work toward the actual problem - 0-1/7 points.
<json>
{"response": "incorrect"}
</json>

Example 12 - ALMOST vs PARTIAL - Critical Distinction:
Problem: Solve 2x + 4 = 10 for x.
Solution: x = 3
Student Answer A (ALMOST): 2x = 6, x = 3. [Minor: skipped showing "subtract 4 from both sides" explicitly, but work is clear and correct. Would get 6-7/7 points.]
Student Answer B (PARTIAL): 2x = 10, x = 5. [Major error: forgot to subtract 4. Wrong answer due to conceptual gap. Would get 2-3/7 points.]
Analysis for Answer A: Minor presentation issue only. The answer is correct and the reasoning is clear.
<json>
{"response": "almost"}
</json>

Example 13 - PARTIAL vs INCORRECT - Critical Distinction:
Problem: Find the derivative of f(x) = x^2 + 3x.
Solution: f'(x) = 2x + 3
Student Answer A (PARTIAL): f'(x) = 2x. [Missing the 3x term, but correctly applied power rule to x^2. Some valid work shown. Would get 2-3/7 points.]
Student Answer B (INCORRECT): f'(x) = x^2 + 3. [No understanding of differentiation - just changed the coefficient. No valid mathematical work. Would get 0-1/7 points.]
Analysis for Answer A: Partial understanding demonstrated with significant gaps.
<json>
{"response": "partial"}
</json>

Example 14 - ALMOST (rounding difference in final step):
Problem: Find sin(30°).
Solution: 0.5
Student Answer: sin(30°) = 1/2 ≈ 0.499
Analysis: The student gave the exact answer (1/2) and an approximate decimal. The tiny rounding difference in the approximation is a minor issue. Core understanding is perfect - 6-7/7 points.
<json>
{"response": "almost"}
</json>

Example 15 - PARTIAL (correct method, calculation error):
Problem: Compute 123 + 456.
Solution: 579
Student Answer: 123 + 456 = 123 + 400 + 50 + 6 = 523 + 50 + 6 = 573 + 6 = 579... wait, let me check: 123 + 456 = 579. Actually my first calculation was wrong, it's 579.
Analysis: The student used a valid method but made multiple arithmetic errors before self-correcting. Shows some understanding but with significant execution errors - 3-4/7 points.
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

**CORRECT** (7/7 points): The answer is fully correct, complete, and demonstrates thorough understanding.
- All required steps are present and correct
- Final answer matches the solution exactly
- Reasoning is sound and well-explained
- No errors, omissions, or issues of any kind

**ALMOST** (6-7/7 points): The answer is nearly correct with ONLY minor issues.
- Core approach and final answer are correct
- Minor issues ONLY: small calc error, notation difference, tiny omission, alternative valid notation
- Student clearly demonstrates full understanding
- Would receive 6-7 points on a 7-point scale
- KEY TEST: If the only issues are cosmetic/presentation differences, it's ALMOST

**PARTIAL** (1-5/7 points): The answer has significant gaps or errors but shows some valid work.
- Some correct steps or partial understanding demonstrated
- Missing key components, major errors, or incomplete reasoning
- Wrong answer due to conceptual gaps or execution errors
- Would receive 1-5 points on a 7-point scale
- KEY TEST: If there's a significant conceptual or execution error, it's PARTIAL

**INCORRECT** (0-1/7 points): The answer is wrong with no valid mathematical reasoning.
- No valid mathematical work or reasoning shown
- Completely wrong approach or irrelevant answer
- No substantive understanding demonstrated
- Would receive 0-1 points on a 7-point scale
- KEY TEST: If there's no valid mathematical work at all, it's INCORRECT

## Critical Decision Rules - Apply These in Order:

**Rule 1 - ALMOST vs PARTIAL (Most Important!):**
- Is the final answer correct and is the reasoning fundamentally sound? → ALMOST
- Is there a major error, missing key step, or wrong final answer? → PARTIAL
- When in doubt: If the student would get 6+ points, it's ALMOST. If 5 or fewer, it's PARTIAL.

**Rule 2 - PARTIAL vs INCORRECT:**
- Does the student show ANY valid mathematical reasoning? → PARTIAL
- Is there NO valid mathematical work? → INCORRECT
- Even a single correct step or valid observation makes it PARTIAL, not INCORRECT.

**Rule 3 - CORRECT vs ALMOST:**
- Is the answer 100% perfect with no issues? → CORRECT
- Are there ANY minor issues (notation, presentation, tiny slips)? → ALMOST

## Grading Instructions:
1. Read the problem carefully and identify what is being asked.
2. Study the correct solution to understand all required steps.
3. Analyze the student's answer step by step against the solution.
4. Identify what the student got right and what they got wrong.
5. Apply the Decision Rules above to determine the grade.
6. Be conservative: When uncertain between two grades, use the lower one.

## Chain-of-Thought Analysis (think step by step):
Before giving your final answer, provide detailed analysis:
- What is the problem asking for?
- What are the key steps/concepts in the correct solution?
- Which steps does the student's answer contain correctly?
- What errors or omissions exist?
- How significant are the errors? (minor cosmetic vs significant conceptual)
- On a 7-point scale, how many points would this receive?
- Which Decision Rule applies here?
- Final classification: CORRECT, ALMOST, PARTIAL, or INCORRECT?

## Few-Shot Examples:
{FEW_SHOT_EXAMPLES}

## Response Format:
First provide your detailed chain-of-thought analysis following the format above.

Then, respond with EXACTLY ONE of these four labels in the JSON format below:

<json>
{{
    "response": "correct"
}}
</json>

OR

<json>
{{
    "response": "almost"
}}
</json>

OR

<json>
{{
    "response": "partial"
}}
</json>

OR

<json>
{{
    "response": "incorrect"
}}
</json>

CRITICAL: The "response" field must contain ONLY one of these four exact words: correct, almost, partial, or incorrect. No other text, no explanations, no punctuation around the word."""

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
