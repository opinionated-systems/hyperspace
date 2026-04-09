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
    cleaned = cleaned.strip(".!?,:;\"'").strip()
    
    # Map to valid labels
    valid_labels = ["correct", "almost", "partial", "incorrect"]
    
    # Check for exact match first
    if cleaned in valid_labels:
        return cleaned
    
    # Handle common variations and edge cases
    # Check for "partially correct" -> partial
    if "partially" in cleaned and "correct" in cleaned:
        return "partial"
    
    # Check for "mostly correct" or "nearly correct" -> almost
    if ("mostly" in cleaned or "nearly" in cleaned or "almost" in cleaned) and "correct" in cleaned:
        return "almost"
    
    # Check for "completely correct" or "fully correct" -> correct
    if ("completely" in cleaned or "fully" in cleaned or "totally" in cleaned) and "correct" in cleaned:
        return "correct"
    
    # Check for negations that might indicate incorrect
    if "not correct" in cleaned or "isn't correct" in cleaned or "not right" in cleaned:
        return "incorrect"
    
    # Check for partial matches with priority order
    # "incorrect" should be checked before "correct" to avoid false positives
    if "incorrect" in cleaned or "wrong" in cleaned or "error" in cleaned:
        return "incorrect"
    if "almost" in cleaned:
        return "almost"
    if "partial" in cleaned:
        return "partial"
    if "correct" in cleaned or "right" in cleaned:
        return "correct"
    
    # Default to incorrect if no match found
    logger.warning(f"Could not normalize prediction '{prediction}', defaulting to 'incorrect'")
    return "incorrect"


def _extract_label_from_text(text: str) -> str:
    """Extract a valid label from raw text when JSON parsing fails.
    
    Uses multiple strategies to find the label, with priority given to
    explicit JSON-like patterns and quoted labels.
    
    Args:
        text: Raw text from LLM response
        
    Returns:
        One of: correct, almost, partial, incorrect
    """
    if not text:
        return "incorrect"
    
    text_lower = text.lower()
    
    # Strategy 1: Look for explicit labels in quotes (highest priority)
    if '"correct"' in text_lower or "'correct'" in text_lower:
        return "correct"
    if '"almost"' in text_lower or "'almost'" in text_lower:
        return "almost"
    if '"partial"' in text_lower or "'partial'" in text_lower:
        return "partial"
    if '"incorrect"' in text_lower or "'incorrect'" in text_lower:
        return "incorrect"
    
    # Strategy 2: Look for labels after common patterns like "response:", "label:", "grade:"
    import re
    patterns = [
        r'response[:\s]+["\']?(correct|almost|partial|incorrect)["\']?',
        r'label[:\s]+["\']?(correct|almost|partial|incorrect)["\']?',
        r'grade[:\s]+["\']?(correct|almost|partial|incorrect)["\']?',
        r'answer[:\s]+["\']?(correct|almost|partial|incorrect)["\']?',
        r'final answer[:\s]+["\']?(correct|almost|partial|incorrect)["\']?',
        r'the answer is[:\s]+["\']?(correct|almost|partial|incorrect)["\']?',
    ]
    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            return match.group(1)
    
    # Strategy 3: Look for labels at the end of the text (often the conclusion)
    # Check last 200 characters
    last_part = text_lower[-200:] if len(text_lower) > 200 else text_lower
    for label in ["incorrect", "almost", "partial", "correct"]:
        if label in last_part:
            # Verify it's not part of another word by checking boundaries
            idx = last_part.rfind(label)
            if idx >= 0:
                # Check if it's a standalone word
                before = last_part[idx-1] if idx > 0 else ' '
                after = last_part[idx+len(label)] if idx+len(label) < len(last_part) else ' '
                if not (before.isalnum() or after.isalnum()):
                    return label
    
    # Strategy 4: Simple keyword matching with priority
    # Check for negations first
    if "not correct" in text_lower or "isn't correct" in text_lower:
        return "incorrect"
    if "partially correct" in text_lower:
        return "partial"
    if "mostly correct" in text_lower or "nearly correct" in text_lower:
        return "almost"
    
    # Priority order: incorrect > almost > partial > correct
    # (to avoid false positives from "correct" being in other words)
    if "incorrect" in text_lower or "wrong" in text_lower:
        return "incorrect"
    if "almost" in text_lower:
        return "almost"
    if "partial" in text_lower:
        return "partial"
    if "correct" in text_lower or "right" in text_lower:
        return "correct"
    
    # Default to incorrect
    logger.warning(f"Could not extract label from text, defaulting to 'incorrect'")
    return "incorrect"


# Few-shot examples for better grading accuracy
# These examples emphasize the critical distinction between ALMOST and PARTIAL
FEW_SHOT_EXAMPLES = """
Example 1 - CORRECT (complete and fully correct):
Problem: Find the sum of 2 + 3.
Solution: The sum of 2 + 3 is 5.
Student Answer: 5
<json>
{"response": "correct"}
</json>

Example 2 - ALMOST (correct approach, minor arithmetic slip):
Problem: Find 15 × 12.
Solution: 15 × 12 = 180
Student Answer: 15 × 12 = 15 × 10 + 15 × 2 = 150 + 30 = 170
Analysis: The student used the correct distributive property approach but made a small addition error (150+30=170 instead of 180). This is ALMOST because the method is correct and they only need to fix a tiny calculation.
<json>
{"response": "almost"}
</json>

Example 3 - ALMOST (correct work, tiny notation omission):
Problem: Find the derivative of f(x) = x³ + 2x.
Solution: f'(x) = 3x² + 2
Student Answer: 3x² + 2
Analysis: The student computed the correct derivative but didn't write "f'(x) =" or "d/dx". The mathematical work is completely correct - only the notation prefix is missing. This is ALMOST.
<json>
{"response": "almost"}
</json>

Example 4 - PARTIAL (missing critical solutions):
Problem: Solve x² = 16 for all real values of x.
Solution: x = 4 or x = -4
Student Answer: x = 4
Analysis: The student found one correct solution but completely missed the negative solution. This is PARTIAL because a significant part of the answer is missing, not just a minor issue.
<json>
{"response": "partial"}
</json>

Example 5 - PARTIAL (correct start but major conceptual error):
Problem: Find the area of a rectangle with length 8 and width 5.
Solution: Area = length × width = 8 × 5 = 40
Student Answer: Area = 8 + 5 = 13
Analysis: The student identified this as an area problem but used addition instead of multiplication. This is a significant conceptual error about how area works. This is PARTIAL because they understood it was an area problem but applied the wrong operation.
<json>
{"response": "partial"}
</json>

Example 6 - INCORRECT (no valid mathematical reasoning):
Problem: Prove the Pythagorean theorem.
Solution: [Geometric proof showing a² + b² = c²]
Student Answer: The Pythagorean theorem is true because Pythagoras was a famous mathematician.
Analysis: No mathematical reasoning or proof is provided. This is INCORRECT.
<json>
{"response": "incorrect"}
</json>

Example 7 - ALMOST vs PARTIAL - Critical Distinction:
Problem: Solve 3x + 7 = 22 for x.
Solution: x = 5

Student Answer A (ALMOST): 
3x = 22 - 7 = 15
x = 3
Analysis: Correct method, subtracted correctly (22-7=15), but divided wrong (15/3=5, not 3). This is a minor calculation slip. ALMOST.

Student Answer B (PARTIAL):
3x + 7 = 22
x = 22 - 7 = 15
Analysis: The student doesn't understand that they need to subtract 7 from BOTH sides first. They jumped to the wrong operation. This shows a significant gap in understanding equation solving. PARTIAL.
<json>
{"response": "almost"}
</json>

Example 8 - PARTIAL vs INCORRECT:
Problem: Find ∫(2x + 3)dx
Solution: x² + 3x + C

Student Answer A (PARTIAL):
∫(2x + 3)dx = x² + 3x
Analysis: The integration is correct but the constant of integration (+C) is missing. This is a significant omission in calculus. PARTIAL.

Student Answer B (INCORRECT):
∫(2x + 3)dx = 2x²/2 + 3x = x² + 3x + C
Wait, that's actually correct... Let me give a better example:

Student Answer B (INCORRECT):
∫(2x + 3)dx = 2x + 3 + C
Analysis: The student didn't apply the power rule for integration at all. They just added +C to the original expression. This shows no understanding of integration. INCORRECT.
<json>
{"response": "partial"}
</json>

Example 9 - ALMOST (rounding or precision issue):
Problem: Find sin(30°)
Solution: 1/2 or 0.5
Student Answer: sin(30°) = 0.5000001
Analysis: The student has the right concept but a tiny precision error. This is ALMOST.
<json>
{"response": "almost"}
</json>

Example 10 - PARTIAL (incomplete reasoning):
Problem: Show that the sum of angles in a triangle is 180°.
Solution: [Complete proof using parallel lines and alternate angles]
Student Answer: Draw a line parallel to one side through the opposite vertex. The alternate angles are equal, so the three angles form a straight line which is 180°.
Analysis: The student has the right idea and key insight, but the explanation is incomplete - they didn't clearly show which angles are alternate/corresponding. A teacher might give 4-5/7 points. PARTIAL.
<json>
{"response": "partial"}
</json>

Example 11 - ALMOST (sign error at the end):
Problem: Solve |x - 3| = 5
Solution: x = 8 or x = -2
Student Answer: x - 3 = 5 or x - 3 = -5, so x = 8 or x = -2
Wait, that's correct... Let me fix:
Student Answer: x - 3 = 5 or x - 3 = -5, so x = 8 or x = 2
Analysis: The student set up both cases correctly but made a sign error in the second solution (-5+3=-2, not 2). This is a minor calculation slip at the very end. ALMOST.
<json>
{"response": "almost"}
</json>

Example 12 - PARTIAL (wrong formula but some correct work):
Problem: Find the volume of a sphere with radius 3.
Solution: V = (4/3)πr³ = (4/3)π(27) = 36π
Student Answer: V = πr²h = π(3)²(3) = 27π
Analysis: The student used the cylinder formula instead of sphere formula, but they did compute their (wrong) formula correctly. This shows some understanding (knowing volume formulas exist) but wrong formula choice. PARTIAL.
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

**CORRECT**: The answer is fully correct, complete, and demonstrates thorough understanding.
- All required steps are present and correct
- Final answer matches the solution
- Reasoning is sound and well-explained
- No significant errors or omissions
- Would receive full credit (7/7 points)

**ALMOST**: The answer is nearly correct with ONLY minor issues.
- Core approach and method are completely correct
- Minor calculation error (e.g., 2+2=5, arithmetic slip)
- Tiny notation omission (e.g., missing "f(x)=" prefix but answer is right)
- Small rounding/precision issue
- Understanding is clearly demonstrated - student knows what they're doing
- Would receive high partial credit (6-7/7 points)
- KEY: The error is TINY and doesn't affect the overall correctness of the approach

**PARTIAL**: The answer has significant gaps or errors but shows some understanding.
- Some correct steps or partial understanding shown
- Missing key solutions (e.g., only found x=2 when answer is x=±2)
- Major conceptual error (e.g., using wrong formula but some work shown)
- Incomplete solution with significant gaps
- Would receive partial credit (1-5/7 points)
- KEY: There's real mathematical work here, but significant problems exist

**INCORRECT**: The answer is wrong or shows no valid reasoning.
- No valid mathematical reasoning provided
- Completely wrong approach with no redeeming work
- No substantive work shown (just guessing or irrelevant text)
- Would receive minimal or no credit (0-1/7 points)
- KEY: No valid mathematical thinking is demonstrated

## Critical Distinctions - Read Carefully:

**ALMOST vs PARTIAL** - THIS IS THE HARDEST DISTINCTION:
Ask yourself: "Is the error tiny/minor, or is it significant?"

ALMOST = TINY issues only:
- Small arithmetic slip (15+30=170 instead of 180)
- Missing notation prefix (answer is right, just no "f'(x)=")
- Sign error at the very end after correct work
- Rounding to 3.14 instead of 3.14159
- The student CLEARLY understands and would get 6-7/7 points

PARTIAL = SIGNIFICANT issues:
- Missing half the solutions (only x=2 when x=±2 needed)
- Used wrong formula but computed it correctly
- Incomplete proof with key steps missing
- The student shows SOME understanding but would get 1-5/7 points

**PARTIAL vs INCORRECT**:
PARTIAL = Some valid mathematical work exists (even if wrong)
INCORRECT = No valid mathematical reasoning at all

## Grading Instructions:
1. Carefully read the problem and understand what is being asked.
2. Compare the student's answer against the correct solution step by step.
3. Check if the student shows valid mathematical work and reasoning.
4. Identify any errors, omissions, or misconceptions.
5. ASK: "On a 7-point scale, how many points would this receive?"
   - 7 points → CORRECT
   - 6-7 points → ALMOST (minor issue only)
   - 1-5 points → PARTIAL (significant gaps but some valid work)
   - 0-1 points → INCORRECT (no valid work)
6. Select the label that best matches the quality of the student's work.

## Chain-of-Thought Analysis (think step by step):
Before giving your final answer, analyze:
- What is the problem asking for?
- What key steps/concepts are required in the solution?
- Which of these does the student's answer contain?
- What errors or omissions exist, and how significant are they?
- On a 7-point scale, how many points would this receive?

## Self-Reflection (check your answer):
After deciding on a label, ask yourself:
- "Did I confuse ALMOST with PARTIAL?" (most common error)
- "Is the error truly tiny (ALMOST) or significant (PARTIAL)?"
- "Would a teacher give 6-7 points (ALMOST) or 1-5 points (PARTIAL)?"
- "Is there ANY valid mathematical work?" (if yes, not INCORRECT)

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
