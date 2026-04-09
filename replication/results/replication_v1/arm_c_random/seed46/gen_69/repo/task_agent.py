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
    Also handles nested JSON objects within the response field.
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
        
        # Try standard parsing first
        try:
            results.append(json.loads(inner))
            continue
        except json.JSONDecodeError:
            pass
        
        # Try cleaning common issues
        try:
            cleaned = _clean_json_string(inner)
            results.append(json.loads(cleaned))
            continue
        except json.JSONDecodeError:
            pass
        
        # Try extracting just the response field if it's a complex nested object
        try:
            # Look for "response": followed by a value (string, number, object, array)
            response_match = re.search(r'"response"\s*:\s*(.+?)(?:,\s*"|$)', inner, re.DOTALL)
            if response_match:
                response_value = response_match.group(1).strip()
                # Try to parse the response value
                try:
                    parsed_value = json.loads(response_value)
                    results.append({"response": parsed_value})
                    continue
                except json.JSONDecodeError:
                    # If it's not valid JSON, treat as string
                    results.append({"response": response_value})
                    continue
        except Exception:
            pass
        
        logger.debug(f"Failed to parse JSON from <json> block")
        continue
    
    return results or None


def _clean_json_string(text: str) -> str:
    """Clean common JSON formatting issues.
    
    Removes trailing commas, comments, and normalizes whitespace.
    """
    import re
    # Remove single-line comments
    text = re.sub(r'//.*$', '', text, flags=re.MULTILINE)
    # Remove multi-line comments
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    # Remove trailing commas before closing braces/brackets
    text = re.sub(r',(\s*[}\]])', r'\1', text)
    return text.strip()


def _extract_json_fallback(text: str) -> list[dict] | None:
    """Fallback JSON extraction using regex for unwrapped JSON objects.

    This handles cases where the model outputs valid JSON without <json> tags.
    Searches for JSON objects with a "response" key.
    """
    results = []
    
    # First try: pattern to match JSON objects with response key (handles nested braces)
    # Use a more robust pattern that can handle nested structures
    pattern = r'\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*"response"(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}'
    matches = re.finditer(pattern, text, re.DOTALL)

    for match in matches:
        try:
            obj = json.loads(match.group())
            if "response" in obj:
                results.append(obj)
        except json.JSONDecodeError:
            # Try with cleaned string
            try:
                cleaned = _clean_json_string(match.group())
                obj = json.loads(cleaned)
                if "response" in obj:
                    results.append(obj)
            except json.JSONDecodeError:
                continue

    # Second try: look for any JSON object in the text
    if not results:
        # Find all potential JSON objects (content between { and })
        brace_pattern = r'\{[\s\S]*?\}'
        for match in re.finditer(brace_pattern, text):
            try:
                obj = json.loads(match.group())
                if isinstance(obj, dict) and "response" in obj:
                    results.append(obj)
            except json.JSONDecodeError:
                try:
                    cleaned = _clean_json_string(match.group())
                    obj = json.loads(cleaned)
                    if isinstance(obj, dict) and "response" in obj:
                        results.append(obj)
                except json.JSONDecodeError:
                    continue

    # Third try: try to parse the entire text as JSON
    if not results:
        try:
            obj = json.loads(text.strip())
            if isinstance(obj, dict) and "response" in obj:
                results.append(obj)
        except json.JSONDecodeError:
            try:
                cleaned = _clean_json_string(text.strip())
                obj = json.loads(cleaned)
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
        self._extraction_stats = {
            "primary_success": 0,
            "fallback_success": 0,
            "total_calls": 0,
            "failed": 0,
        }

    def get_extraction_stats(self) -> dict:
        """Return statistics about JSON extraction methods used."""
        return dict(self._extraction_stats)

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        instruction = f"""You are an expert grading agent. Your task is to evaluate student answers accurately.

Task input:
```
{inputs}
```

Respond in JSON format with the following schema:
<json>
{{
    "response": <your evaluation result>
}}
</json>

Important: Ensure your response is valid JSON with proper escaping."""

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
        self._extraction_stats["total_calls"] += 1
        
        # Get the assistant's response text
        assistant_text = msg_history[-1]["text"] if msg_history else ""
        
        try:
            extracted = _extract_jsons(assistant_text)
            if extracted and "response" in extracted[-1]:
                prediction = extracted[-1]["response"]
                extraction_method = "primary"
                self._extraction_stats["primary_success"] += 1
            else:
                # Try fallback extraction
                extracted = _extract_json_fallback(assistant_text)
                if extracted and "response" in extracted[-1]:
                    prediction = extracted[-1]["response"]
                    extraction_method = "fallback"
                    self._extraction_stats["fallback_success"] += 1
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            # Try fallback on exception
            try:
                extracted = _extract_json_fallback(assistant_text)
                if extracted and "response" in extracted[-1]:
                    prediction = extracted[-1]["response"]
                    extraction_method = "fallback"
                    self._extraction_stats["fallback_success"] += 1
            except Exception as fallback_e:
                self.log_fn(f"Fallback extraction also failed: {fallback_e}")
                self._extraction_stats["failed"] += 1

        self.log_fn(f"Extraction method used: {extraction_method}")
        
        # Log extraction stats periodically for monitoring
        if self._extraction_stats["total_calls"] % 10 == 0:
            total = self._extraction_stats["total_calls"]
            primary_rate = self._extraction_stats["primary_success"] / total * 100
            fallback_rate = self._extraction_stats["fallback_success"] / total * 100
            fail_rate = self._extraction_stats["failed"] / total * 100
            self.log_fn(
                f"Extraction stats (n={total}): "
                f"primary={primary_rate:.1f}%, "
                f"fallback={fallback_rate:.1f}%, "
                f"failed={fail_rate:.1f}%"
            )
        
        return str(prediction), msg_history
