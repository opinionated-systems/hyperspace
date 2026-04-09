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
    Also handles markdown code blocks with json.
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
            # Try to fix common JSON issues
            try:
                # Try extracting just the response field if it's a simple object
                if '"response"' in inner or "'response'" in inner:
                    # Look for the response value
                    for quote in ['"', "'"]:
                        pattern = f'"response"{quote}:\s*{quote}'
                        if pattern.replace('"', quote) in inner:
                            # Extract the value after response:
                            parts = inner.split(f'response{quote}:')
                            if len(parts) > 1:
                                value_part = parts[1].strip()
                                # Extract the value
                                if value_part.startswith('"') or value_part.startswith("'"):
                                    end_quote = value_part[1:].find(value_part[0])
                                    if end_quote != -1:
                                        value = value_part[1:end_quote+1]
                                        results.append({"response": value})
                                        break
            except Exception:
                pass
            continue
    
    # Also try markdown code blocks
    if not results:
        search_from = 0
        while True:
            start = text.find("```json", search_from)
            if start == -1:
                # Try without language specifier
                start = text.find("```", search_from)
                if start == -1:
                    break
                end_marker = "```"
                inner_start = start + 3
            else:
                end_marker = "```"
                inner_start = start + 7
            
            end = text.find(end_marker, inner_start)
            if end == -1:
                break
            inner = text[inner_start:end].strip()
            search_from = end + len(end_marker)
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
    parts = re.split(r'[\s,;|]+', cleaned)
    for part in parts:
        part = part.strip(".!?,:;\"'")
        if part in valid_labels:
            return part
    
    # Check for JSON-like format with quotes
    json_patterns = [
        ('"correct"', "correct"),
        ("'correct'", "correct"),
        ('"almost"', "almost"),
        ("'almost'", "almost"),
        ('"partial"', "partial"),
        ("'partial'", "partial"),
        ('"incorrect"', "incorrect"),
        ("'incorrect'", "incorrect"),
    ]
    for pattern, label in json_patterns:
        if pattern in cleaned:
            return label
    
    # Check for partial matches with word boundaries - order matters!
    # Check more specific terms first to avoid misclassification
    if re.search(r'\balmost\b', cleaned):
        return "almost"
    if re.search(r'\bpartial\b', cleaned):
        return "partial"
    if re.search(r'\bincorrect\b', cleaned) or re.search(r'\bwrong\b', cleaned):
        return "incorrect"
    
    # For "correct", be extra careful - it can appear in phrases like "partially correct"
    # Only match if it's a standalone word and not preceded by words that modify it
    if re.search(r'\bcorrect\b', cleaned):
        # Check if it's modified by "partial" or "almost" or "incorrect"
        if not re.search(r'\b(partial|almost|incorrect)\w*\s+\w*\bcorrect\b', cleaned) and \
           not re.search(r'\bcorrect\b\s+\w*\s*\b(partial|almost|incorrect)\b', cleaned):
            return "correct"
    
    # Default to incorrect if no match found
    logger.warning(f"Could not normalize prediction '{prediction}', defaulting to 'incorrect'")
    return "incorrect"


def _extract_label_from_text(text: str) -> str:
    """Extract a valid label from raw text when JSON parsing fails.
    
    Uses a priority-based approach to find the most likely label.
    
    Args:
        text: Raw text from LLM response
        
    Returns:
        One of: correct, almost, partial, incorrect
    """
    if not text:
        return "incorrect"
    
    text_lower = text.lower()
    
    # Priority 1: Look for explicit labels in JSON-like format (most reliable)
    json_patterns = [
        ('"correct"', "correct"),
        ("'correct'", "correct"),
        ('"almost"', "almost"),
        ("'almost'", "almost"),
        ('"partial"', "partial"),
        ("'partial'", "partial"),
        ('"incorrect"', "incorrect"),
        ("'incorrect'", "incorrect"),
    ]
    for pattern, label in json_patterns:
        if pattern in text_lower:
            return label
    
    # Priority 2: Look for labels after common indicators
    indicators = ["label:", "grade:", "rating:", "assessment:", "verdict:", "final answer:", "conclusion:", 
                  "decision:", "evaluation:", "result:", "classification:", "category:"]
    for indicator in indicators:
        idx = text_lower.find(indicator)
        if idx != -1:
            after = text_lower[idx + len(indicator):].strip()
            # Check for exact match at start
            for label in ["correct", "almost", "partial", "incorrect"]:
                if after.startswith(label):
                    return label
            # Check for quoted labels
            for quote in ['"', "'"]:
                for label in ["correct", "almost", "partial", "incorrect"]:
                    if after.startswith(quote + label + quote):
                        return label
    
    # Priority 3: Look for labels in the last sentence/line (often where conclusion is)
    lines = text_lower.split('\n')
    for line in reversed(lines):
        line = line.strip()
        if line:
            # Check for labels at end of line
            for label in ["correct", "almost", "partial", "incorrect"]:
                if line.endswith(label) or line.endswith(label + ".") or line.endswith(label + "!"):
                    return label
    
    # Priority 4: Look for standalone labels with word boundaries (specific order)
    # Check more specific terms first to avoid misclassification
    if re.search(r'\balmost\b', text_lower):
        return "almost"
    if re.search(r'\bpartial\b', text_lower):
        return "partial"
    if re.search(r'\bincorrect\b', text_lower) or re.search(r'\bwrong\b', text_lower):
        return "incorrect"
    
    # For "correct", be careful about phrases like "partially correct" or "almost correct"
    if re.search(r'\bcorrect\b', text_lower):
        # Check if it's modified by other terms
        if not re.search(r'\b(partial|almost|incorrect)\w*\s+\w*\bcorrect\b', text_lower) and \
           not re.search(r'\bcorrect\b\s+\w*\s*\b(partial|almost|incorrect)\b', text_lower):
            return "correct"
    
    # Priority 5: Check for keywords in order of specificity as fallback
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


# Few-shot examples showing exact label format with reasoning
FEW_SHOT_EXAMPLES = """
Example 1 - CORRECT:
Problem: Find the sum of 2 + 3.
Solution: The sum of 2 + 3 is 5.
Student Answer: 5
Analysis: The student's answer matches the correct solution exactly. The answer is complete and correct.
<json>
{"response": "correct"}
</json>

Example 2 - PARTIAL:
Problem: Solve x^2 = 4 for x.
Solution: x = 2 or x = -2.
Student Answer: x = 2
Analysis: The student found one correct solution (x = 2) but missed the other solution (x = -2). This is a significant omission in the answer.
<json>
{"response": "partial"}
</json>

Example 3 - INCORRECT:
Problem: Prove that the sum of angles in a triangle is 180 degrees.
Solution: [Detailed geometric proof]
Student Answer: It's 180 because that's what I learned.
Analysis: The student states the correct fact but provides no proof or reasoning. The answer does not demonstrate understanding of the mathematical concept.
<json>
{"response": "incorrect"}
</json>

Example 4 - ALMOST:
Problem: Calculate the area of a circle with radius 3.
Solution: Area = πr² = π × 9 = 9π ≈ 28.27
Student Answer: Area = π × 3² = 9π = 28.26
Analysis: The student correctly applied the formula and got 9π, but made a small rounding error (28.26 instead of 28.27). The mathematical reasoning is correct.
<json>
{"response": "almost"}
</json>

Example 5 - PARTIAL:
Problem: Solve the system: x + y = 5, x - y = 1
Solution: Adding equations: 2x = 6, so x = 3. Then y = 2.
Student Answer: x = 3
Analysis: The student correctly found x = 3 but did not find y = 2. The solution is incomplete.
<json>
{"response": "partial"}
</json>

Example 6 - ALMOST:
Problem: Find the integral of 2x.
Solution: ∫2x dx = x² + C
Student Answer: x²
Analysis: The student correctly integrated 2x to get x² but forgot the constant of integration (+C). This is a minor omission that doesn't affect the core correctness.
<json>
{"response": "almost"}
</json>

Example 7 - CORRECT:
Problem: Factor x² - 9.
Solution: (x + 3)(x - 3)
Student Answer: (x - 3)(x + 3)
Analysis: The student correctly factored the expression. The order of factors doesn't matter.
<json>
{"response": "correct"}
</json>

Example 8 - INCORRECT:
Problem: Find the derivative of sin(x).
Solution: cos(x)
Student Answer: -cos(x)
Analysis: The student confused the derivative of sin(x) with the derivative of cos(x). This is a fundamental error.
<json>
{"response": "incorrect"}
</json>

Example 9 - PARTIAL:
Problem: Prove by induction that 1 + 2 + ... + n = n(n+1)/2.
Solution: [Base case n=1, inductive step]
Student Answer: For n=1: 1 = 1(2)/2 = 1 ✓. Assume true for n=k, then for n=k+1: 1+2+...+k+(k+1) = k(k+1)/2 + (k+1) = (k+1)(k+2)/2.
Analysis: The student correctly proved the base case and set up the inductive step correctly, but didn't fully complete the algebraic simplification to show it equals (k+1)(k+2)/2.
<json>
{"response": "partial"}
</json>

Example 10 - ALMOST:
Problem: Solve 2x + 4 = 10.
Solution: 2x = 6, so x = 3.
Student Answer: 2x = 6, x = 3
Analysis: The student correctly solved the equation but didn't show the division step explicitly (2x/2 = 6/2). The answer is correct but the work shown is slightly abbreviated.
<json>
{"response": "almost"}
</json>

Example 11 - INCORRECT (wrong final answer):
Problem: Find the value of 15 × 4.
Solution: 15 × 4 = 60
Student Answer: 15 × 4 = 50. I multiplied 10×4=40 and 5×4=20, then 40+20=50.
Analysis: The student showed work and used a valid method (distributive property), but made an arithmetic error (40+20=60, not 50). The final answer is wrong, so this is INCORRECT despite correct method.
<json>
{"response": "incorrect"}
</json>

Example 12 - ALMOST (correct answer, minor error):
Problem: Find the area of a rectangle with length 8 and width 5.
Solution: Area = length × width = 8 × 5 = 40
Student Answer: Area = 8 × 5 = 40 square units
Analysis: The student correctly calculated the area as 40. Adding "square units" is actually more complete than the solution. The answer is correct with no errors.
<json>
{"response": "correct"}
</json>

Example 13 - PARTIAL (incomplete with some understanding):
Problem: Find all prime numbers between 10 and 20.
Solution: The prime numbers are 11, 13, 17, 19.
Student Answer: 11, 13, 17
Analysis: The student correctly identified three of the four prime numbers but missed 19. This is a significant omission since one prime number was missed.
<json>
{"response": "partial"}
</json>

Example 14 - INCORRECT (fundamental misunderstanding):
Problem: If a triangle has angles of 30° and 60°, what is the third angle?
Solution: The sum of angles in a triangle is 180°, so the third angle is 180° - 30° - 60° = 90°.
Student Answer: The third angle is 30° + 60° = 90° because angles add up.
Analysis: The student got the right answer (90°) but used completely wrong reasoning. They added the given angles instead of subtracting from 180°. This shows fundamental misunderstanding of triangle angle sums.
<json>
{"response": "incorrect"}
</json>

Example 15 - ALMOST (notation issue):
Problem: Write the equation of a line with slope 2 passing through (1, 3).
Solution: y - 3 = 2(x - 1) or y = 2x + 1
Student Answer: y = 2x + 1
Analysis: The student correctly found the equation of the line. The answer is mathematically correct and complete.
<json>
{"response": "correct"}
</json>

Example 16 - PARTIAL (missing reasoning):
Problem: Prove that the sum of two even numbers is even.
Solution: Let the numbers be 2m and 2n. Their sum is 2m + 2n = 2(m+n), which is divisible by 2, hence even.
Student Answer: The sum of two even numbers is even because 2+4=6, 4+6=10, 6+8=14.
Analysis: The student provided examples but no general proof. Examples alone don't constitute a proof. The student shows some understanding but not the required proof structure.
<json>
{"response": "partial"}
</json>

Example 17 - INCORRECT (completely wrong method):
Problem: Solve for x: 3x = 12.
Solution: x = 12 ÷ 3 = 4
Student Answer: x = 12 - 3 = 9
Analysis: The student used subtraction instead of division. This is a fundamental error in understanding how to solve linear equations.
<json>
{"response": "incorrect"}
</json>

Example 18 - ALMOST (rounding with correct method):
Problem: Calculate √2 to two decimal places.
Solution: √2 ≈ 1.414213... ≈ 1.41 (to two decimal places)
Student Answer: √2 = 1.414 which is about 1.41
Analysis: The student correctly identified that √2 ≈ 1.41. They provided extra precision (1.414) but correctly rounded to two decimal places as requested.
<json>
{"response": "correct"}
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

## Label Definitions (STRICT INTERPRETATION):
- "correct": The answer is fully correct and complete, matching the solution approach and result. The student demonstrates full understanding with no significant errors or omissions. The answer may use equivalent forms or alternative valid approaches.
- "almost": The answer is nearly correct with only minor issues (e.g., small calculation error, missing trivial step, minor notation issue, forgotten constant of integration). The core reasoning is sound and the student clearly understands the main concepts.
- "partial": The answer is partially correct but has significant gaps, missing key components, or errors in the main argument. The student shows some understanding but not enough for substantial credit. Common cases: missing one of multiple solutions, incomplete proof, major calculation error with some correct work shown.
- "incorrect": The answer is wrong, does not address the problem, contains fundamental errors, or shows no meaningful understanding of the problem. The student may have guessed, used completely wrong methods, or provided irrelevant information.

## Detailed Grading Instructions:
1. Carefully read the problem and understand what is being asked.
2. Compare the student's answer against the correct solution step by step.
3. Check if the student shows their work and reasoning.
4. Identify any errors, omissions, or misconceptions.
5. Determine if errors are minor (almost) or significant (partial/incorrect).
6. When in doubt between two labels, choose the LOWER grade (more conservative).

## Key Decision Rules (Apply in order):
- CORRECT if: Final answer matches exactly, all steps are valid, no errors of any kind
- ALMOST if: Final answer is correct BUT has minor issues (rounding, notation, trivial omissions, one small calculation error)
- ALMOST if: The core method is correct and the answer is essentially right, with only trivial differences
- PARTIAL if: Significant portion correct but missing key elements OR has major errors OR incomplete solution
- PARTIAL if: Multiple solutions required and student found some but not all
- PARTIAL if: Correct final answer but no work shown (for problems requiring reasoning)
- INCORRECT if: Fundamental misunderstanding, wrong method, or no valid work shown
- INCORRECT if: Answer is just a guess with no reasoning
- INCORRECT if: Final answer is wrong, regardless of whether some steps were correct

## Critical Distinctions:
- ALMOST vs CORRECT: ALMOST has minor flaws; CORRECT is flawless
- ALMOST vs PARTIAL: ALMOST has trivial issues; PARTIAL has significant gaps or errors
- PARTIAL vs INCORRECT: PARTIAL shows meaningful understanding; INCORRECT shows fundamental misunderstanding

## Common Edge Cases:
- Equivalent forms: (x-3)(x+3) vs (x+3)(x-3) are both CORRECT
- Rounding errors: 28.26 vs 28.27 is ALMOST (minor calculation difference)
- Missing +C in integration: ALMOST (minor omission, core concept understood)
- Missing one of two solutions: PARTIAL (significant omission, not trivial)
- Correct answer with no work shown: PARTIAL (significant omission of reasoning)
- Partial proof with gaps: PARTIAL (shows some understanding but incomplete)
- Wrong final answer with some correct steps: INCORRECT (final answer matters)
- Right method, wrong execution leading to wrong answer: PARTIAL or INCORRECT (depends on severity)

## Few-Shot Examples:
{FEW_SHOT_EXAMPLES}

## Response Format:
First, provide your analysis of the student's answer. Then, provide your final label in the exact JSON format below.

Your analysis should:
- State what the problem is asking
- Summarize the correct solution approach
- Evaluate what the student got right
- Identify any errors or omissions
- Explain why you chose your final label based on the decision rules above

After your analysis, you MUST provide the final label in this exact format:

<json>
{{
    "response": "correct|almost|partial|incorrect"
}}
</json>

IMPORTANT: The response field must contain ONLY one of these four exact words: correct, almost, partial, or incorrect.

## Self-Verification Step (Perform before finalizing):
Before outputting your final label, ask yourself:
1. Did I check if the final answer matches the solution exactly? (If no → not CORRECT)
2. Are the errors truly minor/trivial, or are they significant? (Significant → PARTIAL or INCORRECT)
3. Does the student show meaningful understanding of the core concept? (No → INCORRECT)
4. Is the final answer wrong even if some steps were right? (Wrong answer → likely INCORRECT)
5. Am I being too generous? When in doubt, choose the more conservative (lower) grade."""

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
                response_text = msg_history[-1].get("text", "")
                
                # Try JSON extraction first
                extracted = _extract_jsons(response_text)
                if extracted and len(extracted) > 0:
                    # Find the last JSON object with a "response" field
                    for json_obj in reversed(extracted):
                        if "response" in json_obj:
                            raw_prediction = json_obj["response"]
                            prediction = _normalize_prediction(raw_prediction)
                            logger.info(f"Successfully extracted prediction from JSON: {prediction}")
                            break
                    else:
                        # No response field found, try text extraction
                        prediction = _extract_label_from_text(response_text)
                        logger.info(f"Extracted label from text (no response field): {prediction}")
                else:
                    # Try to extract label from raw text if JSON parsing fails
                    prediction = _extract_label_from_text(response_text)
                    logger.info(f"Extracted label from text (no JSON): {prediction}")
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
