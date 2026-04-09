"""
Task agent: solves a given task with a single LLM call.

Reimplemented from facebookresearch/HyperAgents task_agent.py.
Same interface, same JSON output format, same extraction logic.

This is the INITIAL task agent. The meta agent modifies this file
during self-improvement. The evaluation harness loads whatever
task_agent.py exists at the agent's repo path.

MODIFIED: Added enhanced error handling, detailed logging, input validation,
prediction formatting utilities, and confidence scoring for evaluations.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from agent.llm_client import get_response_from_llm, EVAL_MODEL

logger = logging.getLogger(__name__)


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.

    Args:
        text: The text containing <json>...</json> blocks.

    Returns:
        A list of parsed JSON objects, or None if no valid JSON found.
    """
    results = []
    search_from = 0
    json_count = 0
    error_count = 0

    while True:
        start = text.find("<json>", search_from)
        if start == -1:
            break
        end = text.find("</json>", start)
        if end == -1:
            logger.warning("Found unclosed <json> tag at position %d", start)
            break

        inner = text[start + 6:end].strip()
        search_from = end + 7
        json_count += 1

        try:
            parsed = json.loads(inner)
            results.append(parsed)
            logger.debug("Successfully parsed JSON block %d", json_count)
        except json.JSONDecodeError as e:
            error_count += 1
            logger.warning("Failed to parse JSON block %d: %s", json_count, e)
            continue

    if error_count > 0:
        logger.info("JSON extraction: %d found, %d parsed, %d errors", json_count, len(results), error_count)

    return results or None


class TaskAgent:
    """Task agent that solves IMO grading problems.

    This agent uses an LLM to evaluate student answers against grading guidelines.
    It expects inputs containing domain, problem, solution, grading_guidelines, and student_answer.
    """

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.call_count = 0

    def _validate_inputs(self, inputs: dict) -> None:
        """Validate that required input fields are present.

        Args:
            inputs: The input dictionary to validate.

        Raises:
            ValueError: If required fields are missing.
        """
        required_fields = ["domain", "problem", "solution", "grading_guidelines", "student_answer"]
        missing = [f for f in required_fields if f not in inputs]
        if missing:
            raise ValueError(f"Missing required input fields: {missing}")

    def _format_prediction(self, prediction: str, max_length: int = 500) -> str:
        """Format prediction for display, truncating if necessary.

        Args:
            prediction: The prediction string to format.
            max_length: Maximum length before truncation.

        Returns:
            Formatted prediction string.
        """
        if len(prediction) > max_length:
            return prediction[:max_length] + "... [truncated]"
        return prediction

    def _extract_confidence(self, prediction: str) -> tuple[str, float]:
        """Extract confidence score from prediction text.

        Looks for explicit confidence markers like "confidence: 0.85" or
        "I'm 90% confident" and normalizes them to a 0-1 scale.

        Args:
            prediction: The prediction/evaluation text.

        Returns:
            Tuple of (cleaned_prediction, confidence_score).
            Confidence score defaults to 0.5 if not explicitly stated.
        """
        confidence = 0.5  # Default neutral confidence

        # Pattern 1: "confidence: X" or "confidence is X" (0-1 or percentage)
        pattern1 = re.search(
            r'confidence\s*(?:is|:)?\s*(\d+\.?\d*)\s*(%)?',
            prediction,
            re.IGNORECASE
        )
        if pattern1:
            value = float(pattern1.group(1))
            if pattern1.group(2) or value > 1:  # Has % sign or value > 1
                confidence = min(value / 100, 1.0)
            else:
                confidence = min(value, 1.0)

        # Pattern 2: "I'm X% confident" or "I am X% confident"
        pattern2 = re.search(
            r'(?:I\'m|I am)\s+(\d+\.?\d*)%?\s*confident',
            prediction,
            re.IGNORECASE
        )
        if pattern2:
            value = float(pattern2.group(1))
            confidence = min(value / 100 if value > 1 else value, 1.0)

        # Pattern 3: Qualitative confidence markers
        qualitative_patterns = [
            (r'\b(very\s+high|extremely\s+high)\s+confidence\b', 0.95),
            (r'\b(high|strong)\s+confidence\b', 0.85),
            (r'\b(moderate|medium)\s+confidence\b', 0.65),
            (r'\b(low|weak)\s+confidence\b', 0.35),
            (r'\b(very\s+low|extremely\s+low)\s+confidence\b', 0.15),
            (r'\b(certain|definitely|absolutely)\b', 0.95),
            (r'\b(uncertain|unsure|possibly|maybe)\b', 0.40),
        ]

        for pattern, score in qualitative_patterns:
            if re.search(pattern, prediction, re.IGNORECASE):
                confidence = score
                break

        # Clean up the prediction by removing confidence markers
        cleaned = re.sub(
            r'\s*\(?confidence\s*(?:is|:)?\s*\d+\.?\d*%?\s*\)?\.?\s*',
            ' ',
            prediction,
            flags=re.IGNORECASE
        ).strip()

        return cleaned, round(confidence, 2)

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history) where prediction is the extracted response
            from the LLM's JSON output, and msg_history contains the full
            conversation history.
        """
        self.call_count += 1
        self.log_fn(f"TaskAgent call #{self.call_count} starting")

        # Validate inputs
        try:
            self._validate_inputs(inputs)
        except ValueError as e:
            self.log_fn(f"Input validation failed: {e}")
            raise

        instruction = f"""You are an expert grader for mathematics problems.

Your task is to evaluate a student's answer against the provided solution and grading guidelines.
Include a confidence score (0-1) indicating how certain you are about your evaluation.

Task input:
```
{inputs}
```

Respond in JSON format with the following schema:
<json>
{{
    "response": "Your evaluation/grade here",
    "confidence": 0.85
}}
</json>

The confidence field should be a number between 0 and 1, where 1 means complete certainty."""

        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
        except Exception as e:
            self.log_fn(f"LLM call failed: {e}")
            return "Error: LLM call failed", []

        # Extract prediction from JSON
        prediction = "None"
        confidence = 0.5
        try:
            if msg_history:
                extracted = _extract_jsons(msg_history[-1]["text"])
                if extracted and "response" in extracted[-1]:
                    prediction = extracted[-1]["response"]
                    # Extract confidence from JSON if provided
                    confidence = extracted[-1].get("confidence", 0.5)
                    # Also run text-based extraction as fallback/enhancement
                    prediction, extracted_conf = self._extract_confidence(prediction)
                    # Use JSON confidence if valid, otherwise use extracted
                    if not (0 <= confidence <= 1):
                        confidence = extracted_conf
                    formatted = self._format_prediction(prediction, max_length=100)
                    self.log_fn(f"Successfully extracted prediction: {formatted}")
                    self.log_fn(f"Confidence score: {confidence}")
                else:
                    self.log_fn("No valid 'response' field found in extracted JSON")
            else:
                self.log_fn("Empty message history received from LLM")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        self.log_fn(f"TaskAgent call #{self.call_count} completed")
        # Return prediction with confidence metadata
        result = f"{prediction} [confidence: {confidence}]"
        return result, msg_history
