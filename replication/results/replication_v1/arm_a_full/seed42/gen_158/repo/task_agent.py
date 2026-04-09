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
    
    # Split by common separators and check each part
    import re
    parts = re.split(r'[\s,;|]+', cleaned)
    for part in parts:
        part = part.strip(".!?,:;\"'")
        if part in valid_labels:
            return part
    
    # Check for partial matches - be careful with "correct" appearing in other words
    # Order matters: check more specific terms first (most specific to least specific)
    if "almost" in cleaned:
        return "almost"
    if "partial" in cleaned:
        return "partial"
    if "incorrect" in cleaned or "wrong" in cleaned or "error" in cleaned:
        return "incorrect"
    
    # Only return "correct" if it's clearly the intended label
    # Be very conservative - "correct" should not be inferred from ambiguous text
    if cleaned == "correct":
        return "correct"
    
    # Check for "correct" as a standalone word with word boundaries
    # This prevents matching "correct" inside words like "incorrect" or "partially correct"
    if re.search(r'\bcorrect\b', cleaned):
        # Make sure it's not part of a phrase like "partially correct" or "mostly correct"
        # which should map to "partial" or "almost"
        if "partial" not in cleaned and "mostly" not in cleaned and "nearly" not in cleaned:
            return "correct"
    
    # Default to incorrect if no clear match found
    # This is the conservative choice - when in doubt, mark as incorrect
    logger.warning(f"Could not normalize prediction '{prediction}', defaulting to 'incorrect'")
    return "incorrect"


def _extract_label_from_text(text: str) -> str:
    """Extract a valid label from raw text when JSON parsing fails.
    
    Uses a priority-based approach to find the most likely label.
    BE CONSERVATIVE: When in doubt, return "incorrect".
    
    Args:
        text: Raw text from LLM response
        
    Returns:
        One of: correct, almost, partial, incorrect
    """
    if not text:
        return "incorrect"
    
    text_lower = text.lower()
    import re
    
    # Priority 1: Look for explicit labels in JSON-like format (most reliable)
    # These are the most reliable indicators of intent
    json_patterns = [
        ('"almost"', "almost"),
        ("'almost'", "almost"),
        ('"partial"', "partial"),
        ("'partial'", "partial"),
        ('"incorrect"', "incorrect"),
        ("'incorrect'", "incorrect"),
        ('"correct"', "correct"),
        ("'correct'", "correct"),
    ]
    for pattern, label in json_patterns:
        if pattern in text_lower:
            return label
    
    # Priority 2: Look for labels after common indicators
    indicators = ["label:", "grade:", "rating:", "assessment:", "verdict:", "final answer:", "conclusion:", "decision:"]
    for indicator in indicators:
        idx = text_lower.find(indicator)
        if idx != -1:
            after = text_lower[idx + len(indicator):].strip()
            # Check for exact matches first
            for label in ["almost", "partial", "incorrect", "correct"]:
                if after.startswith(label):
                    return label
            # Then check for word boundaries
            for label in ["almost", "partial", "incorrect", "correct"]:
                pattern = r'^' + label + r'\b'
                if re.search(pattern, after):
                    return label
    
    # Priority 3: Look for standalone labels with word boundaries
    # Check in order of specificity (most specific first)
    for label in ["almost", "partial", "incorrect"]:
        pattern = r'\b' + label + r'\b'
        if re.search(pattern, text_lower):
            return label
    
    # Priority 4: For "correct", be extra careful
    # Only match if it's clearly a standalone label and not part of other phrases
    correct_pattern = r'\bcorrect\b'
    if re.search(correct_pattern, text_lower):
        # Make sure it's not part of phrases like "partially correct", "mostly correct", etc.
        # that should map to other labels
        if not any(phrase in text_lower for phrase in ["partially correct", "mostly correct", "nearly correct", "almost correct"]):
            return "correct"
    
    # Priority 5: Check for keywords in order of specificity
    # "almost" and "partial" are more specific than "correct" or "incorrect"
    if "almost" in text_lower:
        return "almost"
    if "partial" in text_lower:
        return "partial"
    if "incorrect" in text_lower or "wrong" in text_lower or "error" in text_lower:
        return "incorrect"
    
    # For "correct" without word boundaries, be very conservative
    # Only return correct if it's clearly the intended meaning
    if "correct" in text_lower:
        # Check if it's preceded by "is" or similar confirming words
        confirming_patterns = [
            r'is\s+correct',
            r'answer\s+is\s+correct',
            r'solution\s+is\s+correct',
        ]
        for pattern in confirming_patterns:
            if re.search(pattern, text_lower):
                return "correct"
    
    # Default to incorrect - this is the conservative choice
    logger.warning(f"Could not extract label from text, defaulting to 'incorrect'")
    return "incorrect"


# Few-shot examples showing exact label format with reasoning
FEW_SHOT_EXAMPLES = """
Example 1 - CORRECT (Perfect answer):
Problem: Find the sum of 2 + 3.
Solution: The sum of 2 + 3 is 5.
Student Answer: 5
Analysis: The student's answer matches the correct solution exactly. The answer is complete, correct, and has no errors or omissions.
<json>
{"response": "correct"}
</json>

Example 2 - ALMOST (Tiny error only):
Problem: Calculate the area of a circle with radius 3.
Solution: Area = πr² = π × 9 = 9π ≈ 28.27
Student Answer: Area = π × 3² = 9π = 28.26
Analysis: The student correctly applied the formula and got 9π, but made a small rounding error (28.26 instead of 28.27). The mathematical reasoning is completely correct - only a minor calculation error at the very end.
<json>
{"response": "almost"}
</json>

Example 3 - PARTIAL (Significant gaps):
Problem: Solve x^2 = 4 for x.
Solution: x = 2 or x = -2.
Student Answer: x = 2
Analysis: The student found one correct solution (x = 2) but completely missed the other solution (x = -2). This is a significant omission - the problem asks for all solutions and the student only found half of them.
<json>
{"response": "partial"}
</json>

Example 4 - INCORRECT (No meaningful understanding):
Problem: Prove that the sum of angles in a triangle is 180 degrees.
Solution: [Detailed geometric proof with parallel lines and alternate angles]
Student Answer: It's 180 because that's what I learned in school.
Analysis: The student states the correct fact but provides no proof, no reasoning, and no mathematical work. The answer shows no understanding of how to prove the theorem.
<json>
{"response": "incorrect"}
</json>

Example 5 - ALMOST (Minor notation issue):
Problem: Find the derivative of f(x) = x^3.
Solution: f'(x) = 3x^2
Student Answer: f'(x) = 3x² (used superscript 2 instead of ^2)
Analysis: The student correctly applied the power rule and got the right answer. The only issue is a minor notational difference (superscript vs caret) which doesn't affect the mathematical correctness.
<json>
{"response": "almost"}
</json>

Example 6 - PARTIAL (Incomplete solution):
Problem: Solve the system: x + y = 5, x - y = 1
Solution: Adding equations: 2x = 6, so x = 3. Then y = 2.
Student Answer: x = 3
Analysis: The student correctly found x = 3 but did not find y = 2. The solution is incomplete - they only solved for one variable when the problem requires both.
<json>
{"response": "partial"}
</json>

Example 7 - INCORRECT (Fundamental error):
Problem: Find the integral of 2x.
Solution: ∫2x dx = x^2 + C
Student Answer: ∫2x dx = 2
Analysis: The student fundamentally misunderstood integration, treating it as evaluation at a point rather than finding the antiderivative. This shows no understanding of calculus concepts.
<json>
{"response": "incorrect"}
</json>

Example 8 - PARTIAL (Some understanding, major errors):
Problem: Factor x^2 - 5x + 6.
Solution: (x - 2)(x - 3)
Student Answer: (x - 1)(x - 6)
Analysis: The student understands the concept of factoring quadratics but made errors in finding the correct factors. They got the form right but the numbers wrong, showing partial understanding.
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

## Label Definitions (STRICT INTERPRETATION - BE CONSERVATIVE):
- "correct": The answer is FULLY correct and COMPLETE. The student demonstrates complete understanding with NO errors, NO omissions, and proper reasoning. Only use this when you are completely confident.
- "almost": The answer is NEARLY correct with ONLY minor issues (e.g., tiny calculation error, trivial notation issue). The core reasoning is sound and complete. Use this when the answer would be "correct" except for a very small mistake.
- "partial": The answer has SOME correct elements but has SIGNIFICANT gaps, missing key components, or errors in the main argument. The student shows partial understanding but not enough for full credit. Use this when the student got something right but missed important parts.
- "incorrect": The answer is WRONG, does NOT address the problem, contains FUNDAMENTAL errors, or shows NO meaningful understanding. Use this when the answer fails to demonstrate adequate understanding of the problem.

## Critical Grading Rules:
1. BE CONSERVATIVE: When in doubt, choose the LOWER grade.
2. "correct" should be rare - only for truly perfect answers.
3. "almost" is for answers that are 90-99% correct with only tiny flaws.
4. "partial" is for answers that are 30-70% correct with significant issues.
5. "incorrect" is for answers that are 0-30% correct or fundamentally wrong.
6. If the student misses ANY key component, it's NOT "correct".
7. If the reasoning has ANY significant flaw, it's NOT "almost".

## Few-Shot Examples:
{FEW_SHOT_EXAMPLES}

## Response Format:
First, provide your detailed analysis. Then, provide your final label in the exact JSON format.

Your analysis MUST address:
1. What the problem is asking for (key requirements)
2. What the correct solution requires (all components)
3. What the student actually provided (step by step)
4. What errors or omissions exist (be specific)
5. Why you chose your label based on the definitions above

## Decision Framework:
Ask yourself these questions in order:
1. Does the answer have ANY errors, omissions, or issues? If yes, NOT "correct".
2. Are the errors tiny/minor (rounding, notation)? If yes, "almost".
3. Are there significant gaps or errors but some understanding shown? If yes, "partial".
4. Is the answer fundamentally wrong or showing no understanding? If yes, "incorrect".

After your analysis, you MUST provide the final label in this exact format:

<json>
{{
    "response": "correct|almost|partial|incorrect"
}}
</json>

IMPORTANT: The response field must contain ONLY one of these four exact words: correct, almost, partial, or incorrect.

## Final Check:
Before submitting, verify your label matches the definitions above. When uncertain, choose the more conservative (lower) label."""

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
