"""
Task agent: solves a given task with a single LLM call.

Reimplemented from facebookresearch/HyperAgents task_agent.py.
Same interface, same JSON output format, same extraction logic.

This is the INITIAL task agent. The meta agent modifies this file
during self-improvement. The evaluation harness loads whatever
task_agent.py exists at the agent's repo path.

MODIFIED: Added enhanced error handling, detailed logging, input validation,
and structured result formatting with confidence scoring.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any
from dataclasses import dataclass
from enum import Enum

from agent.llm_client import get_response_from_llm, EVAL_MODEL

# Compile regex patterns for JSON extraction fallback
_MD_JSON_PATTERN = re.compile(r'```(?:json)?\s*\n?(.*?)\n?```', re.DOTALL)
_RAW_JSON_PATTERN = re.compile(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', re.DOTALL)

logger = logging.getLogger(__name__)


class ConfidenceLevel(Enum):
    """Confidence levels for extraction results."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"


@dataclass
class ExtractionResult:
    """Structured result from JSON extraction with metadata."""
    data: dict | None
    confidence: ConfidenceLevel
    method_used: str
    raw_text: str | None = None
    error_message: str | None = None
    
    def is_valid(self) -> bool:
        """Check if extraction produced valid data."""
        return self.data is not None and self.confidence != ConfidenceLevel.NONE


def _format_grading_result(
    prediction: str,
    confidence: ConfidenceLevel,
    extraction_method: str,
    call_count: int
) -> dict[str, Any]:
    """Format grading result with structured metadata.
    
    Args:
        prediction: The extracted prediction/grade.
        confidence: Confidence level of the extraction.
        extraction_method: Method used to extract the result.
        call_count: Current agent call count.
        
    Returns:
        Structured dictionary with result and metadata.
    """
    return {
        "result": prediction,
        "metadata": {
            "confidence": confidence.value,
            "extraction_method": extraction_method,
            "agent_call_count": call_count,
            "timestamp": logging.Formatter().formatTime(logging.LogRecord(
                "task_agent", logging.INFO, "", 0, "", (), None
            )),
        },
        "status": "success" if confidence != ConfidenceLevel.NONE else "failed"
    }


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


def _extract_jsons_with_fallback(text: str) -> ExtractionResult:
    """Extract JSON objects with multiple fallback strategies and confidence scoring.

    First tries standard <json>...</json> extraction, then falls back to:
    1. Extracting JSON from markdown code blocks (```json...```)
    2. Extracting raw JSON objects from text
    3. Attempting to repair common JSON syntax errors

    Args:
        text: The text containing JSON data.

    Returns:
        ExtractionResult with data, confidence level, and method used.
    """
    # First try standard extraction (highest confidence)
    results = _extract_jsons(text)
    if results:
        return ExtractionResult(
            data=results[-1] if results else None,
            confidence=ConfidenceLevel.HIGH,
            method_used="standard_json_tags",
            raw_text=text[:500] if text else None
        )

    # Fallback 1: Try markdown code blocks (medium confidence)
    extracted_data = None
    for match in _MD_JSON_PATTERN.finditer(text):
        try:
            inner = match.group(1).strip()
            parsed = json.loads(inner)
            extracted_data = parsed
            logger.debug("Extracted JSON from markdown block")
            break
        except json.JSONDecodeError:
            continue

    if extracted_data:
        logger.info("Used markdown fallback for JSON extraction")
        return ExtractionResult(
            data=extracted_data,
            confidence=ConfidenceLevel.MEDIUM,
            method_used="markdown_code_block",
            raw_text=text[:500] if text else None
        )

    # Fallback 2: Try to find raw JSON objects (low confidence)
    for match in _RAW_JSON_PATTERN.finditer(text):
        try:
            parsed = json.loads(match.group(0))
            extracted_data = parsed
            logger.debug("Extracted raw JSON object")
            break
        except json.JSONDecodeError:
            continue

    if extracted_data:
        logger.info("Used raw JSON fallback for extraction")
        return ExtractionResult(
            data=extracted_data,
            confidence=ConfidenceLevel.LOW,
            method_used="raw_json_pattern",
            raw_text=text[:500] if text else None
        )

    # No extraction succeeded
    return ExtractionResult(
        data=None,
        confidence=ConfidenceLevel.NONE,
        method_used="none",
        raw_text=text[:500] if text else None,
        error_message="Failed to extract valid JSON from response"
    )


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

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
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

Task input:
```
{inputs}
```

Respond in JSON format with the following schema:
<json>
{{
    "response": "Your evaluation/grade here"
}}
</json>"""

        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
        except Exception as e:
            self.log_fn(f"LLM call failed: {e}")
            return "Error: LLM call failed", []

        # Extract prediction from JSON using fallback extraction with confidence scoring
        prediction = "None"
        extraction_result = None
        try:
            if msg_history:
                extraction_result = _extract_jsons_with_fallback(msg_history[-1]["text"])
                if extraction_result.is_valid() and extraction_result.data:
                    if "response" in extraction_result.data:
                        prediction = extraction_result.data["response"]
                        self.log_fn(
                            f"Successfully extracted prediction (confidence: {extraction_result.confidence.value}, "
                            f"method: {extraction_result.method_used}): {prediction[:100]}..."
                        )
                    else:
                        self.log_fn("No valid 'response' field found in extracted JSON")
                else:
                    self.log_fn(
                        f"JSON extraction failed: {extraction_result.error_message or 'Unknown error'}"
                    )
            else:
                self.log_fn("Empty message history received from LLM")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        # Format structured result with metadata
        confidence = extraction_result.confidence if extraction_result else ConfidenceLevel.NONE
        method = extraction_result.method_used if extraction_result else "error"
        structured_result = _format_grading_result(
            prediction=str(prediction),
            confidence=confidence,
            extraction_method=method,
            call_count=self.call_count
        )
        
        self.log_fn(f"TaskAgent call #{self.call_count} completed with status: {structured_result['status']}")
        return str(prediction), msg_history
