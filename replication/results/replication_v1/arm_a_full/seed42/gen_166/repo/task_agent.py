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
    
    # Check for exact match with whitespace normalization
    cleaned_normalized = " ".join(cleaned.split())
    if cleaned_normalized in valid_labels:
        return cleaned_normalized
    
    # Check for labels surrounded by quotes or brackets
    for label in valid_labels:
        if f'"{label}"' in cleaned or f"'{label}'" in cleaned:
            return label
    
    # Check for partial matches with priority order (more specific first)
    # Check for "almost" first (before "correct")
    if "almost" in cleaned:
        return "almost"
    if "partial" in cleaned:
        return "partial"
    if "incorrect" in cleaned or "wrong" in cleaned:
        return "incorrect"
    # Check for "correct" last to avoid matching "incorrect" or "almost correct"
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
    
    # Look for explicit labels in JSON-like format first (most reliable)
    import re
    
    # Pattern to match "response": "label" or 'response': 'label' with various formats
    json_patterns = [
        r'["\']?response["\']?\s*:\s*["\']([^"\']+)["\']',  # "response": "label"
        r'["\']?response["\']?\s*:\s*(\w+)',  # "response": label (unquoted)
        r'response\s*[=:]\s*["\']?([^"\'\n,}]+)["\']?',  # response = label or response: label
    ]
    
    for pattern in json_patterns:
        match = re.search(pattern, text_lower)
        if match:
            label = match.group(1).strip().lower()
            # Clean up the label
            label = label.strip('"\'').strip()
            if label in ["correct", "almost", "partial", "incorrect"]:
                return label
    
    # Look for standalone quoted labels with word boundaries
    for label in ["correct", "almost", "partial", "incorrect"]:
        # Match the label as a complete word in quotes
        pattern = rf'["\']\s*{label}\s*["\']'
        if re.search(pattern, text_lower):
            return label
    
    # Look for labels after common prefixes
    prefix_patterns = [
        rf'\bthe answer is\s*:?\s*["\']?\s*{label}\s*["\']?',
        rf'\bgrade\s*:?\s*["\']?\s*{label}\s*["\']?',
        rf'\blabel\s*:?\s*["\']?\s*{label}\s*["\']?',
        rf'\bclassification\s*:?\s*["\']?\s*{label}\s*["\']?',
    ]
    
    for label in ["correct", "almost", "partial", "incorrect"]:
        for pattern in prefix_patterns:
            if re.search(pattern, text_lower):
                return label
    
    # Check for keywords as whole words (avoid substring matches like "correct" in "incorrect")
    # Order matters: check for more specific patterns first
    words = re.findall(r'\b\w+\b', text_lower)
    
    # Count occurrences of each label with priority weighting
    # "almost" and "partial" get higher weight since they're more specific
    label_counts = {
        "almost": words.count("almost") * 2,  # Weight more heavily
        "partial": words.count("partial") * 2,  # Weight more heavily
        "incorrect": words.count("incorrect"),
        "correct": words.count("correct")
    }
    
    # Return the most frequent valid label found
    max_count = 0
    best_label = "incorrect"  # default
    for label, count in label_counts.items():
        if count > max_count:
            max_count = count
            best_label = label
    
    if max_count > 0:
        return best_label
    
    # Check for "wrong" as a fallback for incorrect
    if "wrong" in text_lower:
        return "incorrect"
    
    # Default to incorrect
    logger.warning(f"Could not extract label from text, defaulting to 'incorrect'")
    return "incorrect"


# Few-shot examples for better grading accuracy - expanded with clearer distinctions
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

Example 3 - ALMOST (minor arithmetic error):
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
Problem: Calculate 15% of 80.
Solution: 0.15 × 80 = 12.
Student Answer: 0.15 × 80 = 11. (Correct method, wrong final calculation)
<json>
{"response": "almost"}
</json>

Example 6 - PARTIAL (significant gaps, incomplete):
Problem: Solve x^2 = 4 for x.
Solution: x = 2 or x = -2.
Student Answer: x = 2 (missing the negative solution)
<json>
{"response": "partial"}
</json>

Example 7 - PARTIAL (some correct work but major errors):
Problem: Factor x^2 - 5x + 6.
Solution: (x - 2)(x - 3).
Student Answer: (x - 1)(x - 6) = x^2 - 7x + 6. I think this is wrong but I'm not sure why.
<json>
{"response": "partial"}
</json>

Example 8 - PARTIAL (correct start but wrong conclusion):
Problem: Find the roots of x^2 - 5x + 6 = 0.
Solution: (x-2)(x-3)=0, so x=2 or x=3.
Student Answer: x^2 - 5x + 6 = (x-2)(x+3), so x=2 or x=-3. (Correct factoring attempt but sign error leading to wrong root)
<json>
{"response": "partial"}
</json>

Example 9 - INCORRECT (completely wrong approach):
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

Example 11 - INCORRECT (blank or no meaningful work):
Problem: Solve the system: 2x + y = 5, x - y = 1.
Solution: x = 2, y = 1.
Student Answer: [blank] or "I don't know" or just random numbers without work.
<json>
{"response": "incorrect"}
</json>

Example 12 - ALMOST vs PARTIAL distinction:
Problem: Solve 3x + 7 = 22.
Solution: 3x = 15, so x = 5.

ALMOST case - Student Answer: 3x = 15, so x = 6. (Correct method, one arithmetic slip)
<json>
{"response": "almost"}
</json>

PARTIAL case - Student Answer: x = 22 - 7 = 15. (Wrong method - didn't divide by 3, but got partial result)
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

## Label Definitions (READ CAREFULLY):

**CORRECT**: The answer is fully correct, complete, and demonstrates proper understanding.
- All required steps are present and correct
- Final answer matches the solution
- Reasoning is sound and well-explained
- No significant errors or omissions

**ALMOST**: The answer demonstrates strong understanding with only MINOR issues.
- The approach and reasoning are fundamentally correct
- Small arithmetic errors (e.g., 6/2 = 4 instead of 3)
- Minor notation omissions (e.g., missing f'(x) = before the answer)
- Trivial computational mistakes that don't indicate conceptual misunderstanding
- The student clearly knows how to solve the problem
- KEY: The student would get full marks with just a tiny correction

**PARTIAL**: The answer shows SOME correct understanding but has SIGNIFICANT gaps or errors.
- Missing critical parts of the solution (e.g., only one of multiple solutions)
- Major conceptual errors mixed with some correct work
- Incomplete reasoning or missing key steps
- The student demonstrates partial understanding but cannot complete the problem correctly
- Wrong approach but some correct intermediate steps
- KEY: The student shows some knowledge but has substantial gaps

**INCORRECT**: The answer is fundamentally wrong or does not address the problem.
- No meaningful correct work shown
- Completely wrong approach or reasoning
- Answer is nonsensical or unrelated to the problem
- No demonstration of understanding the problem

## CRITICAL DECISION RULES:

**ALMOST vs PARTIAL - This is the most important distinction:**
- Use ALMOST when: The student uses the RIGHT METHOD but makes a SMALL SLIP (arithmetic, sign error, copy error)
- Use PARTIAL when: The student uses a WRONG METHOD or misses MAJOR COMPONENTS, even if some work is correct

**Quick Test for ALMOST:**
Ask: "If I told the student their specific error, would they immediately fix it and get full marks?"
If YES → ALMOST
If NO (they'd need to rethink their approach) → PARTIAL

**Examples of ALMOST:**
- Correct integration but wrong final arithmetic: ∫2x dx = x² + C = 5 + 3 = 7 (should be 8)
- Correct derivative calculation but wrong sign: d/dx(x²) = -2x (should be 2x)
- Correct method but copied number wrong from problem

**Examples of PARTIAL:**
- Only found one solution when there should be two
- Used wrong formula but did calculations correctly with that wrong formula
- Started correctly but then went completely off track
- Some correct steps but major conceptual misunderstanding evident

## Grading Instructions:
1. Carefully read the problem and understand what is being asked.
2. Compare the student's answer against the correct solution.
3. Check if the student shows their work and reasoning.
4. Identify any errors, omissions, or misconceptions.
5. Apply the CRITICAL DECISION RULES above.
6. Classify based on the definitions - be precise about "almost" vs "partial".

## Few-Shot Examples:
{FEW_SHOT_EXAMPLES}

## Response Format:
You must respond with EXACTLY ONE of these four labels in JSON format:
- "correct" - Fully correct and complete
- "almost" - Nearly correct, only minor issues (small arithmetic/notation errors)
- "partial" - Partially correct with significant gaps or major errors
- "incorrect" - Wrong or does not address the problem

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
