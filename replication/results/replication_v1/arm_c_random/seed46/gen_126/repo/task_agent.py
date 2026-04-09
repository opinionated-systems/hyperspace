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

    Uses a robust stack-based approach to handle nested braces and tags.
    This ensures proper matching of opening/closing tags and braces.
    """
    results = []
    i = 0
    n = len(text)
    
    while i < n:
        # Find the next <json> tag
        start = text.find("<json>", i)
        if start == -1:
            break
        
        # Find the matching </json> using a stack-based approach
        # that properly accounts for nested braces within the JSON
        tag_end = start + 6
        content_start = tag_end
        
        # Track brace depth to find the actual end of JSON content
        brace_depth = 0
        in_string = False
        escape_next = False
        content_idx = content_start
        
        while content_idx < n:
            char = text[content_idx]
            
            if escape_next:
                escape_next = False
                content_idx += 1
                continue
            
            if char == '\\' and in_string:
                escape_next = True
                content_idx += 1
                continue
            
            if char == '"' and not in_string:
                in_string = True
            elif char == '"' and in_string:
                in_string = False
            
            if not in_string:
                if char == '{':
                    brace_depth += 1
                elif char == '}':
                    brace_depth -= 1
                    if brace_depth == 0:
                        # Found the end of the JSON object
                        # Now look for </json> after this
                        json_end = content_idx + 1
                        closing = text.find("</json>", json_end)
                        if closing != -1:
                            inner = text[content_start:json_end].strip()
                            # Try to parse the JSON
                            try:
                                results.append(json.loads(inner))
                            except json.JSONDecodeError as e:
                                # Try to fix common JSON issues
                                try:
                                    fixed = _fix_common_json_issues(inner)
                                    results.append(json.loads(fixed))
                                except json.JSONDecodeError:
                                    logger.debug(f"Failed to parse JSON block: {str(e)[:100]}")
                            i = closing + 7
                            break
                        else:
                            i = json_end
                            break
            
            content_idx += 1
        else:
            # No valid JSON found, move past this <json> tag
            i = tag_end
    
    return results or None


def _fix_common_json_issues(text: str) -> str:
    """Fix common JSON formatting issues.
    
    Handles:
    - Trailing commas before closing braces/brackets
    - Single quotes to double quotes (simple cases)
    - Unescaped newlines in strings
    """
    # Remove trailing commas before closing braces/brackets
    fixed = re.sub(r',(\s*[}\]])', r'\1', text)
    # Fix single quotes to double quotes (simple cases only, not within strings)
    fixed = re.sub(r"(?<!\\)'", '"', fixed)
    # Replace unescaped newlines in strings with escaped newlines
    fixed = re.sub(r'(?<=")\n(?=[^"]*")', r'\\n', fixed)
    return fixed


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
