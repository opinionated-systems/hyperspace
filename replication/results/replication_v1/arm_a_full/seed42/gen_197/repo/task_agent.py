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
    Also handles markdown code blocks and plain JSON objects.
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
    
    # Also try markdown code blocks with json
    search_from = 0
    while True:
        start = text.find("```json", search_from)
        if start == -1:
            break
        end = text.find("```", start + 7)
        if end == -1:
            break
        inner = text[start + 7:end].strip()
        search_from = end + 3
        try:
            results.append(json.loads(inner))
        except json.JSONDecodeError:
            continue
    
    # Try to find plain JSON objects with "response" field
    if not results:
        try:
            # Look for JSON-like patterns with response field
            import re
            pattern = r'\{\s*["\']response["\']\s*:\s*["\']([^"\']+)["\']\s*\}'
            matches = re.findall(pattern, text)
            for match in matches:
                results.append({"response": match})
        except Exception:
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
    
    Uses a multi-stage approach to handle various input formats and edge cases.
    
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
    
    # Stage 1: Check for exact match
    if cleaned in valid_labels:
        return cleaned
    
    # Stage 2: Check for exact match after removing common prefixes/suffixes
    # Handle cases like "The answer is correct" or "Label: partial"
    prefixes = ["the answer is ", "label: ", "grade: ", "classification: ", 
                "result: ", "category: ", "this is ", "i would say ", 
                "the response is ", "the grade is ", "final answer: ",
                "prediction: ", "output: ", "answer: ", "it is "]
    for prefix in prefixes:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):].strip()
            if cleaned in valid_labels:
                return cleaned
    
    # Stage 3: Check for exact match after removing suffixes
    suffixes = [" answer", " grade", " label", " classification", 
                " result", " category", " response"]
    for suffix in suffixes:
        if cleaned.endswith(suffix):
            cleaned = cleaned[:-len(suffix)].strip()
            if cleaned in valid_labels:
                return cleaned
    
    # Stage 4: Check for partial matches with priority ordering
    # Priority: almost > partial > incorrect > correct (to avoid "correct" matching "incorrect")
    if "almost" in cleaned:
        return "almost"
    if "partial" in cleaned:
        return "partial"
    if "incorrect" in cleaned:
        return "incorrect"
    if "wrong" in cleaned:
        return "incorrect"
    # Only match "correct" if it's not part of another word and not preceded by negation
    if re.search(r'\bcorrect\b', cleaned):
        # Check for negation patterns
        if not re.search(r'\b(not|isn\'t|aint|never)\s+\w*\s*correct\b', cleaned):
            return "correct"
    
    # Stage 5: Check for common synonyms or related terms
    if any(word in cleaned for word in ["right", "accurate", "perfect", "exact", "valid"]):
        return "correct"
    if any(word in cleaned for word in ["error", "mistake", "fail", "invalid", "bad"]):
        return "incorrect"
    if any(word in cleaned for word in ["incomplete", "missing", "half", "some"]):
        return "partial"
    if any(word in cleaned for word in ["minor", "tiny", "small", "slight", "nearly", "close"]):
        return "almost"
    
    # Default to incorrect if no match found
    logger.warning(f"Could not normalize prediction '{prediction}', defaulting to 'incorrect'")
    return "incorrect"


def _extract_label_from_text(text: str) -> str:
    """Extract a valid label from raw text when JSON parsing fails.
    
    Uses a multi-stage extraction strategy with priority ordering:
    1. JSON response field (most reliable)
    2. Explicit label declarations
    3. Contextual analysis of the text
    
    Args:
        text: Raw text from LLM response
        
    Returns:
        One of: correct, almost, partial, incorrect
    """
    if not text:
        return "incorrect"
    
    text_lower = text.lower()
    
    # Stage 1: Look for explicit labels in JSON-like format (most reliable)
    # Pattern to match "response": "label" or 'response': 'label'
    json_pattern = r'["\']?response["\']?\s*:\s*["\']([^"\']+)["\']'
    match = re.search(json_pattern, text_lower)
    if match:
        label = match.group(1).strip().lower()
        if label in ["correct", "almost", "partial", "incorrect"]:
            return label
    
    # Stage 2: Look for standalone quoted labels
    for label in ["correct", "almost", "partial", "incorrect"]:
        # Match the label as a complete word in quotes
        pattern = rf'["\']\s*{label}\s*["\']'
        if re.search(pattern, text_lower):
            return label
    
    # Stage 3: Look for explicit label declarations like "Label: correct" or "Grade: partial"
    declaration_patterns = [
        rf'\b(?:label|grade|classification|category|result)\s*[:=]\s*({"|".join(["correct", "almost", "partial", "incorrect"])})\b',
        rf'\bthe\s+(?:answer|response|grade)\s+is\s+({"|".join(["correct", "almost", "partial", "incorrect"])})\b',
        rf'\bi\s+(?:would|will)\s+(?:grade|classify|label)\s+this\s+as\s+({"|".join(["correct", "almost", "partial", "incorrect"])})\b',
    ]
    for pattern in declaration_patterns:
        match = re.search(pattern, text_lower)
        if match:
            label = match.group(1).strip().lower()
            if label in ["correct", "almost", "partial", "incorrect"]:
                return label
    
    # Stage 4: Check for keywords as whole words with position weighting
    # Words appearing later in the text (conclusion) get higher weight
    words = re.findall(r'\b\w+\b', text_lower)
    
    # Count occurrences with position weighting (later mentions count more)
    label_scores = {"correct": 0, "almost": 0, "partial": 0, "incorrect": 0}
    for i, word in enumerate(words):
        if word in label_scores:
            # Weight by position (later = more important) and frequency
            weight = 1 + (i / max(len(words), 1))  # Position weight 1.0 to 2.0
            label_scores[word] += weight
    
    # Also check for negation patterns that might flip the meaning
    negation_patterns = [r'\bnot\s+correct\b', r'\bnot\s+right\b', r'\bincorrect\b', r'\bwrong\b']
    for pattern in negation_patterns:
        if re.search(pattern, text_lower):
            label_scores["incorrect"] += 2.0  # Boost incorrect for negation patterns
    
    # Return the highest scoring label
    best_label = max(label_scores, key=label_scores.get)
    if label_scores[best_label] > 0:
        return best_label
    
    # Stage 5: Contextual analysis - look for descriptive words
    positive_words = ['perfect', 'excellent', 'complete', 'accurate', 'right', 'valid']
    negative_words = ['wrong', 'error', 'mistake', 'fail', 'invalid', 'missing', 'incomplete']
    
    positive_count = sum(1 for w in positive_words if w in text_lower)
    negative_count = sum(1 for w in negative_words if w in text_lower)
    
    if positive_count > negative_count:
        return "correct"
    elif negative_count > positive_count:
        return "incorrect"
    
    # Default to incorrect if no clear signal
    logger.warning(f"Could not extract clear label from text, defaulting to 'incorrect'. Text preview: {text[:100]}...")
    return "incorrect"


# Few-shot examples for better grading accuracy
FEW_SHOT_EXAMPLES = """
Example 1 - CORRECT (fully correct and complete):
Problem: Find the sum of 2 + 3.
Solution: The sum of 2 + 3 is 5.
Student Answer: 5
<json>
{"response": "correct"}
</json>

Example 2 - CORRECT (fully correct with reasoning):
Problem: Find the area of a rectangle with length 5 and width 3.
Solution: Area = length × width = 5 × 3 = 15.
Student Answer: The area is 15 square units. I calculated this by multiplying length (5) by width (3).
<json>
{"response": "correct"}
</json>

Example 3 - ALMOST (minor arithmetic error, perfect understanding):
Problem: Solve for x: 2x + 4 = 10.
Solution: Subtract 4 from both sides: 2x = 6, then divide by 2: x = 3.
Student Answer: 2x = 6, so x = 6/2 = 4. (Student made arithmetic error: 6/2 = 3, not 4)
<json>
{"response": "almost"}
</json>

Example 4 - ALMOST (correct approach, minor notation issue):
Problem: Find the derivative of f(x) = x^3.
Solution: f'(x) = 3x^2 using the power rule.
Student Answer: 3x^2 (correct answer but missing f'(x) = notation)
<json>
{"response": "almost"}
</json>

Example 5 - ALMOST (correct method, small calculation slip):
Problem: Find the perimeter of a square with side length 8.
Solution: Perimeter = 4 × 8 = 32.
Student Answer: 4 × 8 = 30. (Correct method, wrong multiplication)
<json>
{"response": "almost"}
</json>

Example 6 - PARTIAL (missing critical solution - only one of two answers):
Problem: Solve x^2 = 4 for x.
Solution: x = 2 or x = -2.
Student Answer: x = 2 (missing the negative solution - this is a significant gap)
<json>
{"response": "partial"}
</json>

Example 7 - PARTIAL (conceptual error in factoring approach):
Problem: Factor x^2 - 5x + 6.
Solution: (x - 2)(x - 3).
Student Answer: (x - 1)(x - 6) = x^2 - 7x + 6. I think this is wrong but I'm not sure why.
Reasoning: Student tried to factor but chose wrong numbers. Shows attempt but wrong approach.
<json>
{"response": "partial"}
</json>

Example 8 - PARTIAL (started correctly but incomplete):
Problem: Prove by induction that 1 + 2 + ... + n = n(n+1)/2.
Solution: [Base case n=1, inductive step assuming true for n, proving for n+1]
Student Answer: For n=1: 1 = 1(2)/2 = 1 ✓. Assume true for n, then for n+1... [stops here]
Reasoning: Student understood base case and setup but didn't complete the inductive step.
<json>
{"response": "partial"}
</json>

Example 9 - INCORRECT (completely wrong approach, no meaningful work):
Problem: Prove that the sum of angles in a triangle is 180 degrees.
Solution: [Detailed geometric proof with parallel lines and alternate angles]
Student Answer: It's 180 because that's what I learned in school.
<json>
{"response": "incorrect"}
</json>

Example 10 - INCORRECT (nonsensical or unrelated answer):
Problem: Calculate the integral of 2x dx.
Solution: x^2 + C.
Student Answer: The answer is 7 because integration adds the numbers.
<json>
{"response": "incorrect"}
</json>

Example 11 - ALMOST vs PARTIAL comparison:
Problem: Find all real solutions to x^2 - 5x + 6 = 0.
Solution: x = 2 or x = 3.

Student A (ALMOST): "(x-2)(x-3) = 0, so x = 2 or x = 4" 
→ Correct factoring, one arithmetic slip (3 vs 4). Would be perfect if fixed.
<json>
{"response": "almost"}
</json>

Student B (PARTIAL): "(x-2)(x-3) = 0, so x = 2"
→ Correct factoring but missing one solution. Significant gap in completeness.
<json>
{"response": "partial"}
</json>

Example 12 - ALMOST (proof with minor gap):
Problem: Prove that the sum of two even numbers is even.
Solution: Let a = 2m, b = 2n, then a + b = 2(m + n) which is even.
Student Answer: If a and b are even, then a = 2m and b = 2n for integers m, n. So a + b = 2m + 2n = 2(m+n). Since m+n is an integer, a+b is even. [Missing explicit statement that 2(m+n) is even by definition]
<json>
{"response": "almost"}
</json>

Example 13 - PARTIAL (proof with major logical gap):
Problem: Prove that the sum of two even numbers is even.
Solution: Let a = 2m, b = 2n, then a + b = 2(m + n) which is even.
Student Answer: Even numbers are divisible by 2. So if you add two even numbers, the result is also divisible by 2. [No algebraic demonstration, just assertion]
<json>
{"response": "partial"}
</json>

Example 14 - CORRECT (complete proof with all steps):
Problem: Prove that for any integer n, n^2 + n is always even.
Solution: n^2 + n = n(n+1). Either n is even or n+1 is even, so their product is even.
Student Answer: n^2 + n = n(n+1). Since n and n+1 are consecutive integers, one must be even. The product of an even number with any integer is even. Therefore n^2 + n is always even.
<json>
{"response": "correct"}
</json>

Example 15 - ALMOST (correct logic, tiny error in final statement):
Problem: Find the maximum value of f(x) = -x^2 + 4x.
Solution: f'(x) = -2x + 4 = 0 → x = 2. f(2) = -4 + 8 = 4. Maximum is 4.
Student Answer: Taking derivative: f'(x) = -2x + 4. Setting to 0: -2x + 4 = 0, so x = 2. Plugging back: f(2) = -(2)^2 + 4(2) = -4 + 8 = 4. The maximum occurs at x = 2. [Missing explicit "maximum value is 4" but calculation is correct]
<json>
{"response": "almost"}
</json>

Example 16 - PARTIAL (correct method but incomplete analysis):
Problem: Find the maximum value of f(x) = -x^2 + 4x.
Solution: f'(x) = -2x + 4 = 0 → x = 2. f(2) = -4 + 8 = 4. Maximum is 4.
Student Answer: f'(x) = -2x + 4 = 0, so x = 2. [Stops here without finding the maximum value]
<json>
{"response": "partial"}
</json>

Example 17 - INCORRECT (fundamental misunderstanding):
Problem: Find the maximum value of f(x) = -x^2 + 4x.
Solution: f'(x) = -2x + 4 = 0 → x = 2. f(2) = -4 + 8 = 4. Maximum is 4.
Student Answer: The maximum is infinity because as x gets larger, 4x grows without bound. [Missed that -x^2 dominates for large x]
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
        
        instruction = f"""You are an expert grader for {domain}. Grade the student's answer using exactly one of these four labels.

## Problem:
{problem}

## Correct Solution:
{solution}

## Student's Answer:
{student_answer}

## Label Definitions:

**CORRECT**: Fully correct and complete. All steps present, reasoning sound, final answer matches.

**ALMOST**: Nearly correct with ONLY minor issues:
- Small arithmetic errors (e.g., 6/2=4, 5+3=7)
- Minor notation omissions (e.g., missing f'(x)=)
- Tiny computational slips that don't affect understanding
- Student clearly knows the method; would get 90%+ with error fixed
- The error is a "slip", not a "gap"

**PARTIAL**: Some correct work but SIGNIFICANT gaps:
- Missing critical parts (e.g., only one of two solutions)
- Major conceptual errors mixed with correct work
- Incomplete reasoning or missing key steps
- Student shows partial understanding but cannot complete correctly
- Would get 50-70% credit

**INCORRECT**: Fundamentally wrong or no meaningful work:
- No correct work shown
- Completely wrong approach
- Nonsensical or unrelated answer

## Simple Decision Rule:
- Would fixing ONE small error make it perfect? → ALMOST
- Is there a significant missing piece or conceptual issue? → PARTIAL
- Is the answer fully correct? → CORRECT
- Is there no meaningful understanding? → INCORRECT

## Examples:
{FEW_SHOT_EXAMPLES}

## Response Format (JSON only):
<json>
{{
    "response": "correct|almost|partial|incorrect"
}}
</json>"""

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
