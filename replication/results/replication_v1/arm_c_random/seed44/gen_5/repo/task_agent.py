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
from typing import Any

from agent.llm_client import get_response_from_llm, EVAL_MODEL

logger = logging.getLogger(__name__)


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


def _extract_json_with_retry(text: str, max_retries: int = 2) -> list[dict] | None:
    """Extract JSON with multiple fallback strategies.
    
    First tries standard extraction, then attempts to fix common
    JSON formatting issues like trailing commas.
    """
    # First attempt: standard extraction
    result = _extract_jsons(text)
    if result:
        return result
    
    # Retry with common fixes
    for attempt in range(max_retries):
        try:
            # Try to find and fix JSON with trailing commas
            fixed_text = re.sub(r',(\s*[}\]])', r'\1', text)
            result = _extract_jsons(fixed_text)
            if result:
                logger.debug(f"JSON extraction succeeded after fix attempt {attempt + 1}")
                return result
        except Exception:
            pass
    
    return None


def _validate_response_schema(data: dict, required_keys: list[str]) -> tuple[bool, str]:
    """Validate that the response contains required keys.
    
    Args:
        data: The parsed JSON data
        required_keys: List of keys that must be present
        
    Returns:
        (is_valid, error_message)
    """
    missing_keys = [key for key in required_keys if key not in data]
    if missing_keys:
        return False, f"Missing required keys: {missing_keys}"
    return True, ""


def _safe_extract_prediction(extracted: list[dict] | None, 
                              key: str = "response", 
                              default: Any = "None") -> Any:
    """Safely extract a prediction from extracted JSON data.
    
    Args:
        extracted: List of extracted JSON objects
        key: The key to extract from the last object
        default: Default value if extraction fails
        
    Returns:
        The extracted value or default
    """
    if not extracted or not isinstance(extracted, list):
        return default
    
    if len(extracted) == 0:
        return default
    
    last_item = extracted[-1]
    if not isinstance(last_item, dict):
        return default
    
    return last_item.get(key, default)


class TaskAgent:
    """Task agent that solves IMO grading problems with enhanced validation."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.required_response_keys = ["response"]

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

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON with retry mechanism and validation
        prediction = "None"
        try:
            extracted = _extract_json_with_retry(msg_history[-1]["text"])
            
            if extracted:
                # Validate schema
                is_valid, error_msg = _validate_response_schema(
                    extracted[-1], 
                    self.required_response_keys
                )
                
                if is_valid:
                    prediction = _safe_extract_prediction(extracted, "response", "None")
                    self.log_fn(f"Successfully extracted prediction: {prediction}")
                else:
                    self.log_fn(f"Schema validation failed: {error_msg}")
                    # Try to extract anyway as a fallback
                    prediction = _safe_extract_prediction(extracted, "response", "None")
            else:
                self.log_fn("No JSON data extracted from response")
                
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
