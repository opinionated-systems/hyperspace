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
            json_obj = _extract_json_with_brace_matching(inner)
            if json_obj is not None:
                results.append(json_obj)
    return results or None


def _extract_json_with_brace_matching(text: str) -> dict | None:
    """Extract a JSON object from text using brace matching.
    
    Finds the outermost JSON object by matching braces and returns it.
    Returns None if no valid JSON object is found.
    """
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
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    continue
    return None


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
        obj = _extract_json_with_brace_matching(text)
        if obj is not None and isinstance(obj, dict) and "response" in obj:
            results.append(obj)

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
        raw_response = msg_history[-1]["text"] if msg_history else ""
        
        # Log the raw response for debugging (truncated for readability)
        max_log_len = 500
        logged_response = raw_response[:max_log_len] + "..." if len(raw_response) > max_log_len else raw_response
        self.log_fn(f"Raw LLM response (truncated): {logged_response}")
        
        try:
            extracted = _extract_jsons(raw_response)
            if extracted and len(extracted) > 0:
                last_extracted = extracted[-1]
                if isinstance(last_extracted, dict) and "response" in last_extracted:
                    prediction = last_extracted["response"]
                    extraction_method = "primary"
                    self._extraction_stats["primary"] += 1
                    self.log_fn(f"Primary extraction succeeded. Response type: {type(prediction).__name__}")
                else:
                    self.log_fn(f"Primary extraction: 'response' key not found. Keys: {list(last_extracted.keys()) if isinstance(last_extracted, dict) else 'N/A'}")
                    # Try fallback extraction
                    extracted = _extract_json_fallback(raw_response)
                    if extracted and "response" in extracted[-1]:
                        prediction = extracted[-1]["response"]
                        extraction_method = "fallback"
                        self._extraction_stats["fallback"] += 1
                        self.log_fn("Fallback extraction succeeded after primary missing key")
                    else:
                        self._extraction_stats["failed"] += 1
                        self.log_fn("Fallback extraction failed: no valid response found")
            else:
                self.log_fn("Primary extraction: no JSON objects found")
                # Try fallback extraction
                extracted = _extract_json_fallback(raw_response)
                if extracted and extracted and "response" in extracted[-1]:
                    prediction = extracted[-1]["response"]
                    extraction_method = "fallback"
                    self._extraction_stats["fallback"] += 1
                    self.log_fn("Fallback extraction succeeded")
                else:
                    self._extraction_stats["failed"] += 1
                    self.log_fn("Fallback extraction failed: no valid response found")
        except Exception as e:
            self.log_fn(f"Error during primary extraction: {e}")
            # Try fallback on exception
            try:
                extracted = _extract_json_fallback(raw_response)
                if extracted and "response" in extracted[-1]:
                    prediction = extracted[-1]["response"]
                    extraction_method = "fallback"
                    self._extraction_stats["fallback"] += 1
                    self.log_fn("Fallback extraction succeeded after primary exception")
                else:
                    self._extraction_stats["failed"] += 1
                    self.log_fn("Fallback extraction failed after primary exception")
            except Exception as fallback_e:
                self.log_fn(f"Fallback extraction also failed: {fallback_e}")
                self._extraction_stats["failed"] += 1

        self.log_fn(f"Extraction method used: {extraction_method}")
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
