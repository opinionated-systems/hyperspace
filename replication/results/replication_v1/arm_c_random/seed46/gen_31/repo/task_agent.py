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
    Uses a robust brace-matching approach to handle nested structures.
    """
    results = []

    # Find all potential JSON object starting positions
    i = 0
    while i < len(text):
        # Look for opening brace followed by "response"
        idx = text.find('"response"', i)
        if idx == -1:
            break

        # Find the opening brace before "response"
        brace_start = text.rfind('{', 0, idx)
        if brace_start == -1:
            i = idx + 1
            continue

        # Find the matching closing brace using brace counting
        brace_count = 0
        brace_end = -1
        in_string = False
        escape_next = False

        for j in range(brace_start, len(text)):
            char = text[j]

            if escape_next:
                escape_next = False
                continue

            if char == '\\' and in_string:
                escape_next = True
                continue

            if char == '"' and not in_string:
                in_string = True
            elif char == '"' and in_string:
                in_string = False

            if not in_string:
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        brace_end = j + 1
                        break

        if brace_end > brace_start:
            json_str = text[brace_start:brace_end]
            try:
                obj = json.loads(json_str)
                if isinstance(obj, dict) and "response" in obj:
                    results.append(obj)
            except json.JSONDecodeError:
                pass
            i = brace_end
        else:
            i = idx + 1

    # If brace matching fails, try to parse the entire text as JSON
    if not results:
        try:
            obj = json.loads(text.strip())
            if isinstance(obj, dict) and "response" in obj:
                results.append(obj)
        except json.JSONDecodeError:
            pass

    return results or None


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
        raw_response = msg_history[-1]["text"]
        
        # Log the raw response for debugging (truncated if very long)
        log_response = raw_response[:500] + "..." if len(raw_response) > 500 else raw_response
        self.log_fn(f"Raw LLM response: {log_response}")
        
        try:
            extracted = _extract_jsons(raw_response)
            if extracted and len(extracted) > 0:
                last_json = extracted[-1]
                if isinstance(last_json, dict) and "response" in last_json:
                    prediction = last_json["response"]
                    self.log_fn(f"Primary extraction successful. Found {len(extracted)} JSON block(s)")
                else:
                    self.log_fn(f"Primary extraction: last JSON missing 'response' key. Keys: {list(last_json.keys()) if isinstance(last_json, dict) else 'N/A'}")
                    # Try fallback extraction
                    extracted = _extract_json_fallback(raw_response)
                    if extracted and "response" in extracted[-1]:
                        prediction = extracted[-1]["response"]
                        extraction_method = "fallback"
            else:
                self.log_fn("Primary extraction: no JSON blocks found with <json> tags")
                # Try fallback extraction
                extracted = _extract_json_fallback(raw_response)
                if extracted and "response" in extracted[-1]:
                    prediction = extracted[-1]["response"]
                    extraction_method = "fallback"
                    self.log_fn("Fallback extraction successful")
                else:
                    self.log_fn("Fallback extraction: no valid JSON with 'response' key found")
        except Exception as e:
            self.log_fn(f"Error in primary extraction: {type(e).__name__}: {e}")
            # Try fallback on exception
            try:
                extracted = _extract_json_fallback(raw_response)
                if extracted and "response" in extracted[-1]:
                    prediction = extracted[-1]["response"]
                    extraction_method = "fallback"
                    self.log_fn("Fallback extraction successful after primary failure")
                else:
                    self.log_fn("Fallback extraction: no valid JSON with 'response' key found")
            except Exception as fallback_e:
                self.log_fn(f"Fallback extraction also failed: {type(fallback_e).__name__}: {fallback_e}")

        self.log_fn(f"Extraction method used: {extraction_method}, prediction type: {type(prediction).__name__}")
        return str(prediction), msg_history
