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
    """Extract JSON objects from <json>...</json> blocks or standalone JSON.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    Also handles cases where JSON might appear without tags.
    """
    results = []
    
    # First try to find JSON in <json> tags
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
    
    # If no results from tags, try to find standalone JSON objects
    if not results:
        # Look for JSON objects with "response" field
        import re
        # Pattern to match JSON objects
        json_pattern = r'\{\s*"response"\s*:\s*"([^"]+)"\s*\}'
        matches = re.findall(json_pattern, text, re.IGNORECASE)
        for match in matches:
            try:
                # Reconstruct the JSON object
                json_str = f'{{"response": "{match}"}}'
                results.append(json.loads(json_str))
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
    
    # Check for partial matches
    if "correct" in cleaned and "almost" not in cleaned and "partial" not in cleaned and "incorrect" not in cleaned:
        return "correct"
    if "almost" in cleaned:
        return "almost"
    if "partial" in cleaned:
        return "partial"
    if "incorrect" in cleaned or "wrong" in cleaned:
        return "incorrect"
    
    # Default to incorrect if no match found
    logger.warning(f"Could not normalize prediction '{prediction}', defaulting to 'incorrect'")
    return "incorrect"


def _extract_label_from_text(text: str) -> str:
    """Extract a valid label from raw text when JSON parsing fails.
    
    Uses a scoring system based on keyword analysis to determine the most
    likely label from verbose LLM responses.
    
    Args:
        text: Raw text from LLM response
        
    Returns:
        One of: correct, almost, partial, incorrect
    """
    if not text:
        return "incorrect"
    
    text_lower = text.lower()
    
    # Look for explicit JSON labels first (highest priority)
    if '"correct"' in text_lower or "'correct'" in text_lower:
        # Make sure it's not "almost correct" or "partially correct"
        if '"almost"' not in text_lower and '"partial"' not in text_lower:
            return "correct"
    if '"almost"' in text_lower or "'almost'" in text_lower:
        return "almost"
    if '"partial"' in text_lower or "'partial'" in text_lower:
        return "partial"
    if '"incorrect"' in text_lower or "'incorrect'" in text_lower:
        return "incorrect"
    
    # Score-based keyword analysis for verbose responses
    scores = {"correct": 0, "almost": 0, "partial": 0, "incorrect": 0}
    
    # Strong indicators for "correct"
    correct_phrases = [
        "full credit", "full marks", "complete solution", "fully correct",
        "deserves full", "no errors", "no mistakes", "fully meets",
        "meets the problem requirements", "logically sound", "correct and complete"
    ]
    for phrase in correct_phrases:
        if phrase in text_lower:
            scores["correct"] += 3
    
    # Strong indicators for "almost"
    almost_phrases = [
        "almost complete", "minor issues", "minor mistakes", "small gaps",
        "nearly correct", "almost correct", "minor errors", "small error",
        "almost there", "just needs", "slight modification", "nearly there"
    ]
    for phrase in almost_phrases:
        if phrase in text_lower:
            scores["almost"] += 3
    
    # Strong indicators for "partial"
    partial_phrases = [
        "partial credit", "partial marks", "some progress", "useful observations",
        "partial solution", "incomplete proof", "missing steps", "not complete",
        "significant gaps", "fails to prove", "does not establish", "not fully"
    ]
    for phrase in partial_phrases:
        if phrase in text_lower:
            scores["partial"] += 3
    
    # Strong indicators for "incorrect"
    incorrect_phrases = [
        "zero points", "no credit", "does not address", "fundamentally wrong",
        "incorrect conclusion", "logical error", "contradiction", "vacuous truth",
        "earns zero", "none of the", "not demonstrated", "fails to"
    ]
    for phrase in incorrect_phrases:
        if phrase in text_lower:
            scores["incorrect"] += 3
    
    # Check for score mentions
    if "full credit" in text_lower or "full marks" in text_lower:
        scores["correct"] += 2
    if "partial credit" in text_lower or "partial marks" in text_lower:
        scores["partial"] += 2
    if "zero" in text_lower and ("points" in text_lower or "marks" in text_lower or "credit" in text_lower):
        scores["incorrect"] += 2
    
    # Check for grade indicators
    if "score: 0" in text_lower or "score:0" in text_lower:
        scores["incorrect"] += 3
    if "score: 1" in text_lower or "score:1" in text_lower:
        scores["partial"] += 2
    if "score: 2" in text_lower or "score:2" in text_lower:
        scores["partial"] += 1
        scores["almost"] += 1
    if "score: 3" in text_lower or "score:3" in text_lower:
        scores["almost"] += 2
    if "score: 4" in text_lower or "score:4" in text_lower:
        scores["correct"] += 1
    if "score: 5" in text_lower or "score:5" in text_lower:
        scores["correct"] += 2
    if "score: 6" in text_lower or "score:6" in text_lower:
        scores["correct"] += 3
    
    # Keyword presence (lower weight)
    if "correct" in text_lower:
        scores["correct"] += 1
    if "almost" in text_lower:
        scores["almost"] += 1
    if "partial" in text_lower:
        scores["partial"] += 1
    if "incorrect" in text_lower or "wrong" in text_lower:
        scores["incorrect"] += 1
    
    # Find the label with highest score
    max_score = max(scores.values())
    if max_score > 0:
        # Return the label with highest score
        for label in ["correct", "almost", "partial", "incorrect"]:
            if scores[label] == max_score:
                logger.info(f"Extracted label '{label}' from text (scores: {scores})")
                return label
    
    # Default to incorrect if no clear signal
    logger.warning(f"Could not extract clear label from text (scores: {scores}), defaulting to 'incorrect'")
    return "incorrect"


# Few-shot examples for better grading accuracy
FEW_SHOT_EXAMPLES = """
Example 1:
Problem: Find the sum of 2 + 3.
Solution: The sum of 2 + 3 is 5.
Student Answer: 5
<json>
{"response": "correct"}
</json>

Example 2:
Problem: Solve x^2 = 4 for x.
Solution: x = 2 or x = -2.
Student Answer: x = 2
<json>
{"response": "partial"}
</json>

Example 3:
Problem: Prove that the sum of angles in a triangle is 180 degrees.
Solution: [Detailed geometric proof]
Student Answer: It's 180 because that's what I learned.
<json>
{"response": "incorrect"}
</json>

Example 4:
Problem: Find the derivative of x^2.
Solution: The derivative is 2x.
Student Answer: The derivative is 2x, using the power rule where d/dx(x^n) = n*x^(n-1).
<json>
{"response": "almost"}
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
        
        instruction = f"""You are an expert grading agent for {domain}. Your task is to evaluate a student's answer and classify it into exactly one of four categories.

## Problem Statement:
{problem}

## Correct Solution:
{solution}

## Grading Guidelines:
{grading_guidelines}

## Student's Answer:
{student_answer}

## Classification Categories:
- "correct" - The answer is fully correct and complete, deserves full credit
- "almost" - The answer is nearly correct with only minor issues, minor gaps in rigor
- "partial" - The answer is partially correct but has significant gaps, useful observations but incomplete proof
- "incorrect" - The answer is wrong or does not address the problem, earns zero or minimal credit

## Few-Shot Examples:
{FEW_SHOT_EXAMPLES}

## CRITICAL INSTRUCTION:
You MUST respond with ONLY a JSON object in the exact format below. Do NOT include any explanation, analysis, or text before or after the JSON.

<json>
{{
    "response": "correct|almost|partial|incorrect"
}}
</json>

The "response" field must contain EXACTLY ONE of these four words: correct, almost, partial, or incorrect.
No other text is allowed in your response."""

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
