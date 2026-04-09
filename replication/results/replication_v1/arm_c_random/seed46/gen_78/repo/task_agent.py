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
import os
import re

from agent.llm_client import get_response_from_llm, EVAL_MODEL

logger = logging.getLogger(__name__)


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    Also handles nested JSON objects within the tags.
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
        
        # Try to parse the inner content as JSON
        try:
            results.append(json.loads(inner))
        except json.JSONDecodeError:
            # Try to extract JSON objects using brace matching for nested structures
            try:
                # Find the outermost JSON object by matching braces
                brace_count = 0
                json_start = -1
                for i, char in enumerate(inner):
                    if char == '{':
                        if brace_count == 0:
                            json_start = i
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0 and json_start != -1:
                            json_str = inner[json_start:i+1]
                            try:
                                results.append(json.loads(json_str))
                                break
                            except json.JSONDecodeError:
                                continue
            except Exception:
                continue
    return results or None


def _extract_json_fallback(text: str) -> list[dict] | None:
    """Fallback JSON extraction using regex for unwrapped JSON objects.

    This handles cases where the model outputs valid JSON without <json> tags.
    Searches for JSON objects with a "response" key using multiple strategies.
    """
    results = []
    
    # Strategy 1: Pattern to match JSON objects (handles nested braces)
    pattern = r'\{[^{}]*"response"[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.finditer(pattern, text, re.DOTALL)

    for match in matches:
        try:
            obj = json.loads(match.group())
            if "response" in obj:
                results.append(obj)
        except json.JSONDecodeError:
            continue

    # Strategy 2: If regex fails, try to find any JSON-like structure
    if not results:
        try:
            # Try to parse the entire text as JSON
            obj = json.loads(text.strip())
            if isinstance(obj, dict) and "response" in obj:
                results.append(obj)
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
                                results.append(obj)
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
        self._debug_mode = os.environ.get("TASK_AGENT_DEBUG", "false").lower() == "true"
        self._call_count = 0

    def _log_debug(self, message: str, data: dict | None = None) -> None:
        """Log debug information when debug mode is enabled.
        
        Args:
            message: The debug message to log.
            data: Optional dictionary with additional debug data.
        """
        if self._debug_mode:
            debug_msg = f"[DEBUG] {message}"
            if data:
                debug_msg += f" | Data: {json.dumps(data, default=str)}"
            self.log_fn(debug_msg)

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        self._call_count += 1
        self._log_debug(f"Starting forward call #{self._call_count}", {
            "model": self.model,
            "input_keys": list(inputs.keys())
        })
        
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
            self._log_debug("LLM call successful", {
                "response_length": len(response),
                "history_length": len(msg_history)
            })
        except Exception as e:
            self.log_fn(f"Error calling LLM: {e}")
            self._log_debug("LLM call failed", {"error": str(e)})
            return "Error: LLM call failed", [{"role": "system", "text": f"Error: {e}"}]

        # Extract prediction from JSON using primary method
        prediction = "None"
        extraction_method = "failed"
        raw_response = msg_history[-1]["text"]
        
        self._log_debug("Starting JSON extraction", {
            "response_preview": raw_response[:200] if raw_response else "empty"
        })
        
        try:
            extracted = _extract_jsons(raw_response)
            if extracted and "response" in extracted[-1]:
                prediction = extracted[-1]["response"]
                extraction_method = "primary"
                self._extraction_stats["primary"] += 1
                self._log_debug("Primary extraction succeeded", {
                    "prediction": str(prediction)[:100]
                })
            else:
                # Try fallback extraction
                self._log_debug("Primary extraction failed, trying fallback")
                extracted = _extract_json_fallback(raw_response)
                if extracted and "response" in extracted[-1]:
                    prediction = extracted[-1]["response"]
                    extraction_method = "fallback"
                    self._extraction_stats["fallback"] += 1
                    self._log_debug("Fallback extraction succeeded", {
                        "prediction": str(prediction)[:100]
                    })
                else:
                    self._extraction_stats["failed"] += 1
                    self._log_debug("Fallback extraction also failed")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            self._log_debug("Primary extraction raised exception", {"error": str(e)})
            # Try fallback on exception
            try:
                extracted = _extract_json_fallback(raw_response)
                if extracted and "response" in extracted[-1]:
                    prediction = extracted[-1]["response"]
                    extraction_method = "fallback"
                    self._extraction_stats["fallback"] += 1
                    self._log_debug("Fallback extraction succeeded after exception")
                else:
                    self._extraction_stats["failed"] += 1
                    self._log_debug("Fallback extraction failed after exception")
            except Exception as fallback_e:
                self.log_fn(f"Fallback extraction also failed: {fallback_e}")
                self._extraction_stats["failed"] += 1
                self._log_debug("Fallback extraction raised exception", {"error": str(fallback_e)})

        self.log_fn(f"Extraction method used: {extraction_method}")
        self.log_fn(f"Extraction stats: {self._extraction_stats}")
        self._log_debug(f"Forward call #{self._call_count} complete", {
            "extraction_method": extraction_method,
            "prediction_preview": str(prediction)[:100]
        })
        return str(prediction), msg_history

    def get_extraction_stats(self) -> dict:
        """Return the current extraction statistics.
        
        Returns:
            Dictionary with counts of primary, fallback, and failed extractions.
        """
        return dict(self._extraction_stats)

    def reset_extraction_stats(self) -> None:
        """Reset extraction statistics and call count to zero."""
        self._extraction_stats = {"primary": 0, "fallback": 0, "failed": 0}
        self._call_count = 0
        self._log_debug("Extraction stats and call count reset")

    def get_call_count(self) -> int:
        """Return the total number of forward calls made.
        
        Returns:
            The number of times forward() has been called.
        """
        return self._call_count
