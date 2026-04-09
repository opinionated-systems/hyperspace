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


def _extract_response_heuristic(text: str) -> str | None:
    """Last-resort extraction: look for 'response' key and extract its value.
    
    This handles cases where JSON is malformed but the response value is present.
    """
    # Look for "response": "..." or "response": ...
    pattern = r'"response"\s*:\s*(?:"([^"]*)"|([^,\}\n]+))'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        # Return the first non-None group
        return match.group(1) or match.group(2).strip()
    return None


class TaskAgent:
    """Task agent that solves IMO grading problems with robust JSON extraction."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.max_retries = 3
        self.retry_delay = 1.0  # seconds

    def _call_llm_with_retry(self, instruction: str) -> tuple[str, list[dict], dict]:
        """Call LLM with retry logic for transient failures."""
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=[],
                )
                return response, msg_history, info
            except Exception as e:
                last_error = e
                self.log_fn(f"LLM call failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))  # Exponential backoff
        
        # All retries exhausted
        raise last_error

    def _extract_prediction(self, text: str) -> tuple[str, str]:
        """Extract prediction from response text using multiple methods.
        
        Returns:
            (prediction, extraction_method)
        """
        # Try primary extraction first
        try:
            extracted = _extract_jsons(text)
            if extracted and "response" in extracted[-1]:
                return str(extracted[-1]["response"]), "primary"
        except Exception as e:
            self.log_fn(f"Primary extraction failed: {e}")
        
        # Try fallback extraction
        try:
            extracted = _extract_json_fallback(text)
            if extracted and "response" in extracted[-1]:
                return str(extracted[-1]["response"]), "fallback"
        except Exception as e:
            self.log_fn(f"Fallback extraction failed: {e}")
        
        # Try heuristic extraction as last resort
        try:
            heuristic_result = _extract_response_heuristic(text)
            if heuristic_result is not None:
                return heuristic_result, "heuristic"
        except Exception as e:
            self.log_fn(f"Heuristic extraction failed: {e}")
        
        return "None", "failed"

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
            response, msg_history, info = self._call_llm_with_retry(instruction)
        except Exception as e:
            import traceback
            error_details = f"Error calling LLM after {self.max_retries} retries: {e}\n{traceback.format_exc()}"
            self.log_fn(error_details)
            return "Error: LLM call failed", [{"role": "system", "text": error_details}]

        # Extract prediction from response
        prediction, extraction_method = self._extract_prediction(msg_history[-1]["text"])
        
        self.log_fn(f"Extraction method used: {extraction_method}")
        
        # Log warning if extraction failed
        if extraction_method == "failed":
            self.log_fn(f"Warning: Could not extract prediction from response: {response[:500]}...")
        
        return prediction, msg_history
