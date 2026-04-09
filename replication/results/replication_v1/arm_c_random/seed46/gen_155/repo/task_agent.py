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

from agent.llm_client import get_response_from_llm, EVAL_MODEL

logger = logging.getLogger(__name__)


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    """
    results = []
    search_from = 0
    block_count = 0
    while True:
        start = text.find("<json>", search_from)
        if start == -1:
            break
        end = text.find("</json>", start)
        if end == -1:
            logger.warning(f"Found unclosed <json> tag at position {start}")
            break
        block_count += 1
        inner = text[start + 6:end].strip()
        search_from = end + 7
        try:
            parsed = json.loads(inner)
            if isinstance(parsed, dict):
                results.append(parsed)
            else:
                logger.debug(f"JSON in block {block_count} is not a dict: {type(parsed)}")
        except json.JSONDecodeError as e:
            logger.debug(f"JSON decode error in block {block_count}: {e}")
            # Try to extract partial content for debugging
            preview = inner[:100].replace('\n', ' ')
            logger.debug(f"Block {block_count} content preview: {preview}...")
            continue
    
    if block_count > 0 and not results:
        logger.warning(f"Found {block_count} <json> blocks but none parsed successfully")
    
    return results or None


def _extract_json_fallback(text: str) -> list[dict] | None:
    """Fallback JSON extraction using regex for unwrapped JSON objects.

    This handles cases where the model outputs valid JSON without <json> tags.
    Searches for JSON objects with a "response" key.
    """
    results = []
    # Pattern to match JSON objects (handles nested braces)
    pattern = r'\{[^{}]*"response"[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.finditer(pattern, text, re.DOTALL)

    for match in matches:
        try:
            obj = json.loads(match.group())
            if "response" in obj:
                results.append(obj)
        except json.JSONDecodeError:
            continue

    # If regex fails, try to find any JSON-like structure
    if not results:
        try:
            # Try to parse the entire text as JSON
            obj = json.loads(text.strip())
            if isinstance(obj, dict) and "response" in obj:
                results.append(obj)
        except json.JSONDecodeError:
            pass

    return results or None


def _validate_prediction(prediction: str, inputs: dict) -> tuple[bool, str]:
    """Validate that the prediction is appropriate for the task.
    
    Returns:
        (is_valid, reason) tuple where is_valid is True if prediction looks reasonable
    """
    if prediction is None or prediction == "None":
        return False, "Prediction is None"
    
    if prediction == "Error: LLM call failed":
        return False, "LLM call failed"
    
    # Check for empty or whitespace-only predictions
    if not prediction.strip():
        return False, "Prediction is empty or whitespace-only"
    
    # Check if prediction is unreasonably long (might indicate an error)
    if len(prediction) > 10000:
        return False, "Prediction is unreasonably long (>10000 chars)"
    
    return True, "Prediction looks valid"


class TaskAgent:
    """Task agent that solves IMO grading problems with robust JSON extraction."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

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
            import traceback
            error_details = f"Error calling LLM: {e}\n{traceback.format_exc()}"
            self.log_fn(error_details)
            return "Error: LLM call failed", [{"role": "system", "text": error_details}]

        # Extract prediction from JSON using primary method
        prediction = "None"
        extraction_method = "primary"
        try:
            extracted = _extract_jsons(msg_history[-1]["text"])
            if extracted and "response" in extracted[-1]:
                prediction = extracted[-1]["response"]
            else:
                # Try fallback extraction
                extracted = _extract_json_fallback(msg_history[-1]["text"])
                if extracted and "response" in extracted[-1]:
                    prediction = extracted[-1]["response"]
                    extraction_method = "fallback"
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            # Try fallback on exception
            try:
                extracted = _extract_json_fallback(msg_history[-1]["text"])
                if extracted and "response" in extracted[-1]:
                    prediction = extracted[-1]["response"]
                    extraction_method = "fallback"
            except Exception as fallback_e:
                self.log_fn(f"Fallback extraction also failed: {fallback_e}")

        # Validate the prediction
        is_valid, validation_reason = _validate_prediction(prediction, inputs)
        if not is_valid:
            self.log_fn(f"Prediction validation failed: {validation_reason}")
        else:
            self.log_fn(f"Prediction validation passed: {validation_reason}")

        self.log_fn(f"Extraction method used: {extraction_method}")
        return str(prediction), msg_history
