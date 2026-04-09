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
from dataclasses import dataclass, field
from typing import Any

from agent.llm_client import get_response_from_llm, EVAL_MODEL

logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    """Result of JSON extraction with metadata."""
    value: Any
    method: str
    confidence: float = 1.0
    raw_text: str = ""


@dataclass
class ValidationResult:
    """Result of response validation."""
    is_valid: bool
    prediction: str
    errors: list[str] = field(default_factory=list)


def _extract_jsons(text: str) -> list[ExtractionResult] | None:
    """Extract JSON objects from <json>...</json> blocks with enhanced robustness.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    Also handles nested JSON objects within the tags.
    
    Enhanced with:
    - Better handling of markdown code blocks inside <json> tags
    - Support for multiple JSON objects in a single block
    - Improved error recovery for malformed JSON
    - Returns ExtractionResult with metadata for better tracking
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
        
        # Remove markdown code block markers if present
        inner = inner.removeprefix("```json").removeprefix("```")
        inner = inner.removesuffix("```").strip()
        
        # Try to parse the inner content as JSON
        try:
            parsed = json.loads(inner)
            results.append(ExtractionResult(
                value=parsed,
                method="primary_direct",
                confidence=1.0,
                raw_text=inner[:200]
            ))
        except json.JSONDecodeError:
            # Try to extract JSON objects using brace matching for nested structures
            try:
                # Find all JSON objects by matching braces
                brace_count = 0
                json_start = -1
                json_objects = []
                for i, char in enumerate(inner):
                    if char == '{':
                        if brace_count == 0:
                            json_start = i
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0 and json_start != -1:
                            json_str = inner[json_start:i+1]
                            json_objects.append(json_str)
                            json_start = -1
                
                # Try to parse each found JSON object
                for json_str in json_objects:
                    try:
                        parsed = json.loads(json_str)
                        results.append(ExtractionResult(
                            value=parsed,
                            method="primary_brace_matching",
                            confidence=0.9,
                            raw_text=json_str[:200]
                        ))
                    except json.JSONDecodeError:
                        continue
                        
            except Exception:
                continue
    return results or None


def _extract_json_fallback(text: str) -> list[ExtractionResult] | None:
    """Fallback JSON extraction using regex for unwrapped JSON objects.

    This handles cases where the model outputs valid JSON without <json> tags.
    Searches for JSON objects with a "response" key using multiple strategies.
    Returns ExtractionResult with metadata for better tracking.
    """
    results = []
    
    # Strategy 1: Pattern to match JSON objects (handles nested braces)
    pattern = r'\{[^{}]*"response"[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.finditer(pattern, text, re.DOTALL)

    for match in matches:
        try:
            obj = json.loads(match.group())
            if "response" in obj:
                results.append(ExtractionResult(
                    value=obj,
                    method="fallback_regex",
                    confidence=0.7,
                    raw_text=match.group()[:200]
                ))
        except json.JSONDecodeError:
            continue

    # Strategy 2: If regex fails, try to find any JSON-like structure
    if not results:
        try:
            # Try to parse the entire text as JSON
            obj = json.loads(text.strip())
            if isinstance(obj, dict) and "response" in obj:
                results.append(ExtractionResult(
                    value=obj,
                    method="fallback_full_parse",
                    confidence=0.8,
                    raw_text=text[:200]
                ))
        except json.JSONDecodeError:
            pass

    # Strategy 3: Try to extract JSON using brace matching for complex nested structures
    if not results:
        try:
            brace_count = 0
            json_start = -1
            for i, char in enumerate(text):
                if char == '{':
                    if brace_count == 0:
                        json_start = i
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0 and json_start != -1:
                        json_str = text[json_start:i+1]
                        try:
                            obj = json.loads(json_str)
                            if isinstance(obj, dict) and "response" in obj:
                                results.append(ExtractionResult(
                                    value=obj,
                                    method="fallback_brace_matching",
                                    confidence=0.6,
                                    raw_text=json_str[:200]
                                ))
                                break  # Take the first valid one
                        except json.JSONDecodeError:
                            continue
        except Exception:
            pass

    return results or None


class TaskAgent:
    """Task agent that solves IMO grading problems with robust JSON extraction.
    
    This agent processes grading tasks by calling an LLM and extracting
    structured JSON responses. It includes fallback mechanisms for robust
    extraction even when the model output format varies.
    
    Attributes:
        model: The LLM model identifier to use for task solving.
        log_fn: Logging function for agent operations.
    """

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self._extraction_stats = {"primary": 0, "fallback": 0, "failed": 0}

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        instruction = f"""You are an agent.

Task input:
```
{inputs}
```

Respond in JSON format with the following schema:
<json>
{{
    "response": ...
}}
</json>"""

        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
        except Exception as e:
            self.log_fn(f"Error calling LLM: {e}")
            return "Error: LLM call failed", [{"role": "system", "text": f"Error: {e}"}]

        # Extract prediction from JSON using primary method
        prediction = "None"
        extraction_method = "failed"
        extraction_confidence = 0.0
        try:
            extracted = _extract_jsons(msg_history[-1]["text"])
            if extracted and "response" in extracted[-1].value:
                result = extracted[-1]
                prediction = result.value["response"]
                extraction_method = result.method
                extraction_confidence = result.confidence
                self._extraction_stats["primary"] += 1
            else:
                # Try fallback extraction
                extracted = _extract_json_fallback(msg_history[-1]["text"])
                if extracted and "response" in extracted[-1].value:
                    result = extracted[-1]
                    prediction = result.value["response"]
                    extraction_method = result.method
                    extraction_confidence = result.confidence
                    self._extraction_stats["fallback"] += 1
                else:
                    self._extraction_stats["failed"] += 1
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            # Try fallback on exception
            try:
                extracted = _extract_json_fallback(msg_history[-1]["text"])
                if extracted and "response" in extracted[-1].value:
                    result = extracted[-1]
                    prediction = result.value["response"]
                    extraction_method = result.method
                    extraction_confidence = result.confidence
                    self._extraction_stats["fallback"] += 1
                else:
                    self._extraction_stats["failed"] += 1
            except Exception as fallback_e:
                self.log_fn(f"Fallback extraction also failed: {fallback_e}")
                self._extraction_stats["failed"] += 1

        self.log_fn(f"Extraction method used: {extraction_method} (confidence: {extraction_confidence})")
        self.log_fn(f"Extraction stats: {self._extraction_stats}")
        return str(prediction), msg_history

    def get_extraction_stats(self) -> dict:
        """Return the current extraction statistics.
        
        Returns:
            Dictionary with counts of primary, fallback, and failed extractions.
        """
        return dict(self._extraction_stats)

    def reset_extraction_stats(self) -> None:
        """Reset extraction statistics to zero."""
        self._extraction_stats = {"primary": 0, "fallback": 0, "failed": 0}

    def _validate_prediction(self, prediction: str, inputs: dict) -> ValidationResult:
        """Validate the extracted prediction against task requirements.
        
        Args:
            prediction: The extracted prediction string
            inputs: The original task inputs for context
            
        Returns:
            ValidationResult with validation status and any errors
        """
        errors = []
        
        # Check for empty or None predictions
        if prediction is None or prediction == "None" or prediction == "":
            errors.append("Prediction is empty or None")
            return ValidationResult(is_valid=False, prediction=prediction, errors=errors)
        
        # Check for error indicators in prediction
        if prediction.startswith("Error:"):
            errors.append(f"Prediction contains error: {prediction}")
            return ValidationResult(is_valid=False, prediction=prediction, errors=errors)
        
        # For grading tasks, check if prediction is a valid grade format
        if "grading_guidelines" in inputs:
            valid_grades = ["A", "B", "C", "D", "F", "Pass", "Fail", "Correct", "Incorrect"]
            # Allow numeric grades too
            prediction_upper = str(prediction).strip().upper()
            if not any(grade.upper() in prediction_upper for grade in valid_grades):
                # Check if it's a numeric score
                try:
                    float(prediction)
                except ValueError:
                    errors.append(f"Prediction '{prediction}' doesn't match expected grade format")
        
        is_valid = len(errors) == 0
        return ValidationResult(is_valid=is_valid, prediction=prediction, errors=errors)
