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
    
    Improved to handle nested JSON structures and malformed tags more robustly.
    """
    results = []
    search_from = 0
    text_len = len(text)
    
    while search_from < text_len:
        start = text.find("<json>", search_from)
        if start == -1:
            break
        
        # Find the matching </json> tag, handling potential nesting issues
        end = text.find("</json>", start + 6)
        if end == -1:
            # Malformed: no closing tag found, try to extract anyway
            break
        
        inner = text[start + 6:end].strip()
        search_from = end + 7
        
        # Skip empty content
        if not inner:
            continue
            
        try:
            parsed = json.loads(inner)
            if isinstance(parsed, dict):
                results.append(parsed)
        except json.JSONDecodeError:
            # Try to fix common JSON issues before giving up
            try:
                # Handle trailing commas
                fixed = re.sub(r',\s*}', '}', inner)
                fixed = re.sub(r',\s*]', ']', fixed)
                parsed = json.loads(fixed)
                if isinstance(parsed, dict):
                    results.append(parsed)
            except json.JSONDecodeError:
                continue
    
    return results if results else None


def _extract_json_fallback(text: str) -> list[dict] | None:
    """Fallback JSON extraction using brace depth parsing for unwrapped JSON objects.

    This handles cases where the model outputs valid JSON without <json> tags.
    Searches for JSON objects with a "response" key.
    Improved to handle nested braces more robustly and fix common JSON issues.
    """
    results = []
    
    # First, try to find JSON objects by parsing brace depth
    # This handles nested structures better than regex
    def find_json_objects(s: str) -> list[str]:
        """Find potential JSON objects by tracking brace depth."""
        objects = []
        i = 0
        n = len(s)
        while i < n:
            if s[i] == '{':
                start = i
                depth = 1
                i += 1
                in_string = False
                escape_next = False
                while i < n and depth > 0:
                    c = s[i]
                    if escape_next:
                        escape_next = False
                    elif c == '\\':
                        escape_next = True
                    elif c == '"' and not in_string:
                        in_string = True
                    elif c == '"' and in_string:
                        in_string = False
                    elif not in_string:
                        if c == '{':
                            depth += 1
                        elif c == '}':
                            depth -= 1
                    i += 1
                if depth == 0:
                    objects.append(s[start:i])
            else:
                i += 1
        return objects
    
    def try_parse_json(obj_str: str) -> dict | None:
        """Try to parse JSON string with common fix attempts."""
        try:
            return json.loads(obj_str)
        except json.JSONDecodeError:
            # Try to fix common JSON issues
            try:
                # Handle trailing commas
                fixed = re.sub(r',\s*}', '}', obj_str)
                fixed = re.sub(r',\s*]', ']', fixed)
                # Handle single quotes (convert to double)
                fixed = re.sub(r"'([^']*?)':", r'"\1":', fixed)
                fixed = re.sub(r":\s*'([^']*?)'", r': "\1"', fixed)
                return json.loads(fixed)
            except json.JSONDecodeError:
                return None
    
    # Try to parse each potential JSON object
    for obj_str in find_json_objects(text):
        obj = try_parse_json(obj_str)
        if obj is not None and isinstance(obj, dict) and "response" in obj:
            results.append(obj)
    
    # If still no results, try to parse the entire text as JSON
    if not results:
        obj = try_parse_json(text.strip())
        if obj is not None and isinstance(obj, dict) and "response" in obj:
            results.append(obj)
    
    # Final fallback: try to extract any dict-like structure with response key
    if not results:
        # Look for patterns like "response": "..." or 'response': '...'
        response_pattern = r'["\']response["\']\s*:\s*["\']([^"\']+)["\']'
        match = re.search(response_pattern, text)
        if match:
            results.append({"response": match.group(1)})
        else:
            # Try looser pattern for response extraction
            loose_pattern = r'response["\']?\s*:\s*["\']?([^"\'\n,}]+)'
            match = re.search(loose_pattern, text, re.IGNORECASE)
            if match:
                results.append({"response": match.group(1).strip()})

    return results if results else None


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
        instruction = f"""You are an expert grading agent. Your task is to evaluate student answers based on the provided problem, solution, and grading guidelines.

Task input:
```
{inputs}
```

Respond ONLY in JSON format with the following schema:
<json>
{{
    "response": "Your evaluation/grade here"
}}
</json>

Important: The response field must contain your complete evaluation or grade."""

        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
        except Exception as e:
            self.log_fn(f"Error calling LLM: {e}")
            return "Error: LLM call failed", [{"role": "system", "text": f"Error: {e}"}]

        # Check if we got a valid response
        if not msg_history or len(msg_history) < 2:
            self.log_fn("Warning: Empty or incomplete message history from LLM")
            return "Error: No response from LLM", msg_history if msg_history else [{"role": "system", "text": "No response"}]

        # Extract prediction from JSON using primary method
        prediction = "None"
        extraction_method = "none"
        raw_response = msg_history[-1].get("text", "")
        
        if not raw_response or not raw_response.strip():
            self.log_fn("Warning: Empty response text from LLM")
            return "Error: Empty response from LLM", msg_history
        
        # Log the raw response for debugging
        self.log_fn(f"Raw response length: {len(raw_response)} chars")
        
        try:
            extracted = _extract_jsons(raw_response)
            if extracted and len(extracted) > 0 and "response" in extracted[-1]:
                prediction = extracted[-1]["response"]
                extraction_method = "primary"
                self.log_fn(f"Primary extraction succeeded, response type: {type(prediction).__name__}")
            else:
                # Try fallback extraction
                self.log_fn("Primary extraction failed, trying fallback...")
                extracted = _extract_json_fallback(raw_response)
                if extracted and len(extracted) > 0 and "response" in extracted[-1]:
                    prediction = extracted[-1]["response"]
                    extraction_method = "fallback"
                    self.log_fn(f"Fallback extraction succeeded, response type: {type(prediction).__name__}")
                else:
                    self.log_fn("Warning: No valid JSON with 'response' key found in LLM output")
                    # Last resort: use raw response if it looks reasonable
                    if len(raw_response) < 1000 and "{" not in raw_response:
                        prediction = raw_response.strip()
                        extraction_method = "raw"
                        self.log_fn("Using raw response as prediction")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            # Try fallback on exception
            try:
                extracted = _extract_json_fallback(raw_response)
                if extracted and len(extracted) > 0 and "response" in extracted[-1]:
                    prediction = extracted[-1]["response"]
                    extraction_method = "fallback"
                    self.log_fn(f"Fallback extraction succeeded after exception, response type: {type(prediction).__name__}")
            except Exception as fallback_e:
                self.log_fn(f"Fallback extraction also failed: {fallback_e}")

        self.log_fn(f"Extraction method used: {extraction_method}")
        
        # Convert prediction to string, handling various types
        if prediction is None:
            prediction = "None"
        elif isinstance(prediction, (list, dict)):
            prediction = json.dumps(prediction)
        else:
            prediction = str(prediction)
            
        return prediction, msg_history
