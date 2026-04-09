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
    Also handles markdown code blocks and plain JSON.
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
            # Try to extract just the response field if full JSON fails
            try:
                # Look for "response": "value" pattern
                match = re.search(r'"response"\s*:\s*"([^"]+)"', inner)
                if match:
                    results.append({"response": match.group(1)})
            except Exception:
                continue
    
    # If no <json> blocks found, try markdown code blocks
    if not results:
        # Try ```json ... ``` blocks
        json_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
        for match in re.finditer(json_block_pattern, text, re.DOTALL):
            try:
                inner = match.group(1).strip()
                results.append(json.loads(inner))
            except json.JSONDecodeError:
                # Try to extract just the response field
                try:
                    resp_match = re.search(r'"response"\s*:\s*"([^"]+)"', inner)
                    if resp_match:
                        results.append({"response": resp_match.group(1)})
                except Exception:
                    continue
    
    # If still no results, try to find any JSON object with a "response" field
    if not results:
        # Look for {"response": "..."} pattern anywhere in text
        resp_pattern = r'\{\s*"response"\s*:\s*"([^"]+)"\s*\}'
        for match in re.finditer(resp_pattern, text):
            try:
                results.append(json.loads(match.group(0)))
            except json.JSONDecodeError:
                results.append({"response": match.group(1)})
    
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
    
    # Check for exact match with quotes
    for label in valid_labels:
        if f'"{label}"' in cleaned or f"'{label}'" in cleaned:
            return label
    
    # Check for negations first (be conservative)
    if "not correct" in cleaned or "not almost" in cleaned or "not partial" in cleaned:
        return "incorrect"
    
    # Check for compound phrases - be careful about ordering
    # "partially correct" should be partial
    if "partially correct" in cleaned:
        return "partial"
    # "almost correct" should be almost
    if "almost correct" in cleaned:
        return "almost"
    # "mostly correct" should be partial (significant but not complete)
    if "mostly correct" in cleaned:
        return "partial"
    # "mostly wrong" or "mostly incorrect" should be incorrect
    if "mostly wrong" in cleaned or "mostly incorrect" in cleaned:
        return "incorrect"
    
    # Check for partial matches - be conservative (prefer lower grades when ambiguous)
    # Check for "incorrect" first (most conservative)
    if "incorrect" in cleaned:
        return "incorrect"
    # Check for "wrong"
    if "wrong" in cleaned:
        return "incorrect"
    # Check for "error" or "mistake"
    if "error" in cleaned or "mistake" in cleaned:
        return "incorrect"
    # Check for "almost" (more specific than partial)
    if "almost" in cleaned:
        return "almost"
    # Check for "partial"
    if "partial" in cleaned:
        return "partial"
    # Check for "incomplete" - treat as partial
    if "incomplete" in cleaned:
        return "partial"
    # Only return "correct" if explicitly stated and no other labels found
    if "correct" in cleaned:
        return "correct"
    # Check for "right" or "valid" as synonyms for correct
    if cleaned in ["right", "valid", "true", "yes"]:
        return "correct"
    
    # Default to incorrect if no match found (conservative default)
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
    if '"almost"' in text_lower or "'almost'" in text_lower:
        return "almost"
    if '"partial"' in text_lower or "'partial'" in text_lower:
        return "partial"
    if '"incorrect"' in text_lower or "'incorrect'" in text_lower:
        return "incorrect"
    if '"correct"' in text_lower or "'correct'" in text_lower:
        return "correct"
    
    # Check for negations first (be conservative)
    if "not correct" in text_lower or "not almost" in text_lower or "not partial" in text_lower:
        return "incorrect"
    
    # Check for compound phrases - order matters
    if "partially correct" in text_lower:
        return "partial"
    if "almost correct" in text_lower:
        return "almost"
    if "mostly correct" in text_lower:
        return "partial"
    if "mostly wrong" in text_lower or "mostly incorrect" in text_lower:
        return "incorrect"
    
    # Check for keywords in the text - be conservative (prefer lower grades)
    # Check for "incorrect" first (most conservative)
    if "incorrect" in text_lower:
        return "incorrect"
    # Check for "wrong"
    if "wrong" in text_lower:
        return "incorrect"
    # Check for "error" or "mistake"
    if "error" in text_lower or "mistake" in text_lower:
        return "incorrect"
    # Check for "almost" (more specific than partial)
    if "almost" in text_lower:
        return "almost"
    # Check for "partial" 
    if "partial" in text_lower:
        return "partial"
    # Check for "incomplete"
    if "incomplete" in text_lower:
        return "partial"
    # Only return "correct" if it's explicitly stated and no other labels found
    if "correct" in text_lower:
        return "correct"
    
    # Default to incorrect when uncertain
    logger.warning(f"Could not extract label from text, defaulting to 'incorrect'")
    return "incorrect"


# Few-shot examples for better grading accuracy - concise version
FEW_SHOT_EXAMPLES = """
Example 1 - CORRECT:
Problem: Find the sum of 2 + 3.
Solution: The sum of 2 + 3 is 5.
Student Answer: 5
<json>{"response": "correct"}</json>

Example 2 - ALMOST:
Problem: Find the derivative of x^2.
Solution: The derivative is 2x.
Student Answer: The derivative is 2x using the power rule, but forgot to mention constant C.
<json>{"response": "almost"}</json>

Example 3 - PARTIAL:
Problem: Solve x^2 = 4 for x.
Solution: x = 2 or x = -2.
Student Answer: x = 2
<json>{"response": "partial"}</json>

Example 4 - INCORRECT (No valid reasoning):
Problem: Prove sum of angles in triangle is 180 degrees.
Solution: [Detailed geometric proof]
Student Answer: It's 180 because that's what I learned.
<json>{"response": "incorrect"}</json>

Example 5 - INCORRECT (Major error):
Problem: Find integral of x dx.
Solution: x^2/2 + C
Student Answer: x^2 + C
<json>{"response": "incorrect"}</json>

Example 6 - PARTIAL (Incomplete):
Problem: Solve: x + y = 5, x - y = 1.
Solution: x = 3, y = 2
Student Answer: From first equation, x = 5 - y.
<json>{"response": "partial"}</json>

Example 7 - ALMOST (Minor notation issue):
Problem: Evaluate lim(x→0) sin(x)/x.
Solution: The limit equals 1.
Student Answer: The limit is 1. Using L'Hopital: lim(x→0) cos(x)/1 = 1.
<json>{"response": "almost"}</json>

Example 8 - INCORRECT (Wrong reasoning despite correct answer):
Problem: Prove n^3 - n divisible by 6 for all positive integers n.
Solution: Factor as n(n-1)(n+1), product of 3 consecutive integers.
Student Answer: [Incorrect induction attempt with flawed logic]
<json>{"response": "incorrect"}</json>

Example 9 - PARTIAL (Missing key case):
Problem: Find all real solutions to |x-1| = 2.
Solution: x = 3 or x = -1.
Student Answer: x-1 = 2, so x = 3.
<json>{"response": "partial"}</json>

Example 10 - ALMOST (Trivial typo):
Problem: Solve 2x + 4 = 10.
Solution: 2x = 6, so x = 3.
Student Answer: 2x = 6, therefore x = 3.
<json>{"response": "almost"}</json>
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
        
        instruction = f"""You are an expert IMO grading agent for {domain}. Grade the student's answer against the correct solution.

## Problem:
{problem}

## Correct Solution:
{solution}

## Grading Guidelines:
{grading_guidelines}

## Student's Answer:
{student_answer}

## Grading Rubric (BE STRICT):
- **CORRECT**: Fully complete, all steps correct, no gaps. Full marks.
- **ALMOST**: Correct approach and answer, only trivial errors (typos, minor notation). Nearly full marks.
- **PARTIAL**: Some correct work but significant gaps, missing key steps/cases, or major errors mixed with correct parts. Partial credit.
- **INCORRECT**: Wrong answer, major conceptual errors, flawed reasoning, or no valid justification. Little/no credit.

## Key Rules:
1. Correct final answer with FLAWED reasoning = INCORRECT
2. Missing key cases/steps = PARTIAL (not ALMOST)
3. When uncertain, choose the LOWER grade
4. "correct" is rare - only for truly complete answers

## Examples:
{FEW_SHOT_EXAMPLES}

## Response:
<json>{{"response": "correct|almost|partial|incorrect"}}</json>"""

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
