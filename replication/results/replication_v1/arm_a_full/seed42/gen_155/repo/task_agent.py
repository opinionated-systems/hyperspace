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
    
    # Check for partial matches - be careful with "correct" appearing in other words
    # Order matters: check for more specific patterns first
    if "almost correct" in cleaned or "nearly correct" in cleaned:
        return "almost"
    if "partially correct" in cleaned:
        return "partial"
    if "almost" in cleaned:
        return "almost"
    if "partial" in cleaned:
        return "partial"
    if "incorrect" in cleaned or "wrong" in cleaned:
        return "incorrect"
    # Only return "correct" if it's a standalone word (not part of "partially correct" etc.)
    if cleaned == "correct" or ("correct" in cleaned and "partial" not in cleaned and "incorrect" not in cleaned and "almost" not in cleaned):
        # Additional check: make sure "correct" is the main label, not just a word in a sentence
        words = cleaned.split()
        if "correct" in words or cleaned == "correct":
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
    if '"correct"' in text_lower or "'correct'" in text_lower:
        return "correct"
    if '"almost"' in text_lower or "'almost'" in text_lower:
        return "almost"
    if '"partial"' in text_lower or "'partial'" in text_lower:
        return "partial"
    if '"incorrect"' in text_lower or "'incorrect'" in text_lower:
        return "incorrect"
    
    # Check for keywords in the text - order matters!
    # Check for compound phrases first to avoid misclassification
    if "almost correct" in text_lower or "nearly correct" in text_lower:
        return "almost"
    if "partially correct" in text_lower:
        return "partial"
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


# Few-shot examples showing exact label format with clear distinctions
FEW_SHOT_EXAMPLES = """
Example 1 - CORRECT (complete and accurate):
Problem: Find the sum of 2 + 3.
Solution: The sum of 2 + 3 is 5.
Student Answer: 5
<json>
{"response": "correct"}
</json>

Example 2 - CORRECT (with reasoning):
Problem: Solve 2x + 4 = 10.
Solution: Subtract 4: 2x = 6, then divide by 2: x = 3.
Student Answer: 2x + 4 = 10 → 2x = 6 → x = 3
<json>
{"response": "correct"}
</json>

Example 3 - ALMOST (minor error, correct method):
Problem: Solve 2x = 10.
Solution: x = 5.
Student Answer: x = 6 (divided 10 by 2 incorrectly)
<json>
{"response": "almost"}
</json>

Example 4 - ALMOST (trivial omission):
Problem: Find area of circle with radius 3.
Solution: A = πr² = 9π ≈ 28.27.
Student Answer: A = π(3)² = 9π = 28.26 (small rounding difference)
<json>
{"response": "almost"}
</json>

Example 5 - PARTIAL (incomplete, missing key parts):
Problem: Solve x² = 4 for x.
Solution: x = 2 or x = -2.
Student Answer: x = 2 (forgot negative solution)
<json>
{"response": "partial"}
</json>

Example 6 - PARTIAL (right direction, significant gaps):
Problem: Prove sum of triangle angles is 180°.
Solution: Draw parallel line, use alternate interior angles to show angles form straight line.
Student Answer: The angles form a straight line so they sum to 180°. [No explanation of how/why]
<json>
{"response": "partial"}
</json>

Example 7 - INCORRECT (fundamentally wrong):
Problem: What is capital of France?
Solution: Paris.
Student Answer: London.
<json>
{"response": "incorrect"}
</json>

Example 8 - INCORRECT (wrong concept):
Problem: Find derivative of x².
Solution: 2x using power rule.
Student Answer: x³/3 (confused derivative with integral).
<json>
{"response": "incorrect"}
</json>

Example 9 - INCORRECT (no valid reasoning):
Problem: Prove sum of triangle angles is 180°.
Solution: [Geometric proof].
Student Answer: It's 180 because that's what I learned. [No mathematical reasoning]
<json>
{"response": "incorrect"}
</json>

Example 10 - ALMOST vs PARTIAL distinction:
Problem: Factor x² - 5x + 6.
Solution: (x - 2)(x - 3).
Student Answer A: (x - 2)(x + 3) [sign error, but factoring method correct] → ALMOST
Student Answer B: The roots are x = 2 and x = 3 [found roots but didn't write as factors] → PARTIAL
<json>
{"response": "almost"}
</json>

Example 11 - ALMOST (minor arithmetic error in complex calculation):
Problem: Calculate 123 × 456.
Solution: 123 × 456 = 56088.
Student Answer: 123 × 456 = 56048 [small arithmetic error, correct method]
<json>
{"response": "almost"}
</json>

Example 12 - ALMOST (correct method, wrong final answer due to calculation):
Problem: Find the area of a circle with radius 5.
Solution: A = πr² = 25π ≈ 78.54.
Student Answer: A = π × 5² = 25π = 75.4 [used π ≈ 3.016 instead of 3.1416, but correct formula]
<json>
{"response": "almost"}
</json>

Example 13 - ALMOST (correct integration, wrong evaluation):
Problem: Evaluate ∫₀² x² dx.
Solution: [x³/3]₀² = 8/3 - 0 = 8/3 ≈ 2.67.
Student Answer: x³/3 from 0 to 2 = 8/3 = 2.666... ≈ 2.7 [correct method, minor rounding]
<json>
{"response": "almost"}
</json>

Example 14 - PARTIAL (incomplete solution with gaps):
Problem: Solve the system: x + y = 10, x - y = 2.
Solution: Add equations: 2x = 12, so x = 6, then y = 4.
Student Answer: x = 6 [found x but didn't find y, incomplete solution]
<json>
{"response": "partial"}
</json>

Example 15 - PARTIAL (right start but major gaps):
Problem: Prove the Pythagorean theorem.
Solution: [Complete geometric or algebraic proof].
Student Answer: Draw a right triangle with sides a, b, c. [Only stated setup, no proof]
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
        
        instruction = f"""You are an expert grading agent for {domain}. Your task is to carefully analyze the student's answer and assign exactly one of four labels.

## Problem Statement:
{problem}

## Correct Solution:
{solution}

## Grading Guidelines:
{grading_guidelines}

## Student's Answer:
{student_answer}

## Decision Framework - Follow These Steps:

**STEP 1: Check for "correct"**
- Does the answer match the solution exactly (or equivalent form)?
- Is the reasoning sound and complete?
- If YES → label as "correct"

**STEP 2: If not correct, check for "almost"**
- Is the main method/approach correct?
- Is there only a minor error (small calculation mistake, trivial omission, rounding difference)?
- Would a small fix make it fully correct?
- If YES → label as "almost"

**STEP 3: If not almost, check for "partial"**
- Does the student show some understanding of the problem?
- Is there a valid start or right direction?
- Are there significant gaps, missing steps, or incomplete reasoning?
- If YES → label as "partial"

**STEP 4: If none of the above, label as "incorrect"**
- The answer is fundamentally wrong, uses wrong concepts, or shows no valid reasoning

## Critical Distinction: "almost" vs "partial"

**Use "almost" when:**
- The student demonstrates FULL understanding of the correct method
- The error is truly minor and isolated (e.g., arithmetic slip, sign error)
- The answer is 90-99% complete and correct
- Example: Correct integration but wrong final arithmetic: ∫x dx = x²/2 + C, student writes x²/2 = 4 instead of x²/2 = 8

**Use "partial" when:**
- The student shows PARTIAL understanding but significant gaps remain
- Missing key conceptual elements or major steps
- The answer is 30-70% complete
- Example: Sets up the problem correctly but cannot complete the solution
- Example: Has the right idea but execution has major flaws

**Key Test:** If fixing the error would take seconds → "almost". If the student needs to add substantial work → "partial".

## Label Definitions with Criteria:

**"correct"** - The answer is fully correct and complete:
- Matches the solution in both method and result
- Shows clear, valid reasoning
- No significant errors or omissions

**"almost"** - Nearly correct with only MINOR issues (90-99% complete):
- Correct overall method/approach - student shows FULL understanding
- Minor calculation error (e.g., arithmetic slip, sign error)
- Trivial omission (e.g., forgot to state final units)
- Small rounding difference
- A quick fix (seconds of work) would make it perfect
- The error is isolated and doesn't affect the overall reasoning

**"partial"** - Partially correct with SIGNIFICANT gaps (30-70% complete):
- Shows some understanding but incomplete
- Missing key steps or components
- Right idea but poorly executed or incomplete
- Started correctly but didn't finish
- Has valid elements but major flaws
- Would need substantial additional work to be correct

**"incorrect"** - Fundamentally wrong:
- Wrong answer with no valid reasoning
- Misunderstands the problem or uses wrong concepts
- Answer does not address the question
- No mathematical/logical validity

## Few-Shot Examples:
{FEW_SHOT_EXAMPLES}

## Response Format:
You must respond with EXACTLY ONE of these four labels in JSON format:
<json>
{{
    "response": "correct|almost|partial|incorrect"
}}
</json>

IMPORTANT: 
- Your response field must contain ONLY one of these four exact words: correct, almost, partial, or incorrect
- Do not include explanations, reasoning, or any other text in the JSON
- Follow the decision framework above step by step
- Be especially careful distinguishing "almost" (minor error, 90-99% correct) from "partial" (significant gaps, 30-70% correct)
- When in doubt between "almost" and "partial", ask: "Would fixing this take seconds (almost) or require substantial work (partial)?"

SELF-VERIFICATION: Before outputting your answer, verify:
1. Did you follow the 4-step decision framework?
2. Is your label one of: correct, almost, partial, incorrect?
3. Is your JSON format exactly: <json>{{"response": "LABEL"}}</json>?
4. If you chose "almost", is the error truly minor and isolated?
5. If you chose "partial", are there significant gaps beyond a quick fix?"""

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
