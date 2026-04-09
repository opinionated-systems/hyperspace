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
        except json.JSONDecodeError as e:
            logger.debug(f"JSON decode error: {e}, content: {inner[:100]}...")
            continue
    return results or None


def _extract_json_fallback(text: str) -> dict | None:
    """Fallback JSON extraction using regex patterns for common formats.
    
    Attempts to extract JSON from code blocks or raw JSON objects.
    """
    # Try to extract from markdown code blocks
    code_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    matches = re.findall(code_block_pattern, text, re.DOTALL)
    for match in matches:
        try:
            return json.loads(match.strip())
        except json.JSONDecodeError:
            continue
    
    # Try to find raw JSON objects with balanced braces
    # This pattern handles nested braces better
    def find_json_objects(s: str) -> list[str]:
        """Find potential JSON objects by tracking brace balance."""
        results = []
        i = 0
        while i < len(s):
            if s[i] == '{':
                start = i
                balance = 1
                i += 1
                in_string = False
                escape_next = False
                
                while i < len(s) and balance > 0:
                    char = s[i]
                    if escape_next:
                        escape_next = False
                    elif char == '\\':
                        escape_next = True
                    elif char == '"' and not in_string:
                        in_string = True
                    elif char == '"' and in_string:
                        in_string = False
                    elif not in_string:
                        if char == '{':
                            balance += 1
                        elif char == '}':
                            balance -= 1
                    i += 1
                
                if balance == 0:
                    results.append(s[start:i])
            else:
                i += 1
        return results
    
    json_objects = find_json_objects(text)
    for obj_str in json_objects:
        try:
            parsed = json.loads(obj_str)
            if "response" in parsed:
                return parsed
        except json.JSONDecodeError:
            continue
    
    return None


def _extract_response_heuristic(text: str) -> str | None:
    """Last-resort heuristic extraction for response field.
    
    Looks for patterns like "response": "..." or response: ...
    """
    # Look for response field in various formats
    patterns = [
        r'"response"\s*:\s*"([^"]+)"',
        r'"response"\s*:\s*"((?:[^"\\]|\\.)*)"',
        r'response\s*:\s*([^\n,}]+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
    
    return None


class TaskAgent:
    """Task agent that solves IMO grading problems with enhanced extraction."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.stats = {
            "total_calls": 0, 
            "json_extracted": 0, 
            "fallback_used": 0, 
            "heuristic_used": 0,
            "raw_used": 0,
            "failed": 0,
            "empty_response": 0,
        }

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        self.stats["total_calls"] += 1
        
        instruction = f"""You are an expert grading agent. Analyze the student answer carefully.

Task input:
```
{json.dumps(inputs, indent=2)}
```

Respond ONLY in JSON format with the following schema:
<json>
{{
    "response": "Your detailed evaluation here"
}}
</json>"""

        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
        except Exception as e:
            self.log_fn(f"LLM call failed: {e}")
            self.stats["failed"] += 1
            return "Error: LLM call failed", []

        # Extract prediction from JSON
        prediction = "None"
        extraction_method = "none"
        
        try:
            if msg_history and len(msg_history) > 0:
                last_message = msg_history[-1].get("text", "")
                
                # Skip if empty response
                if not last_message or not last_message.strip():
                    self.stats["empty_response"] += 1
                    prediction = "Error: Empty response from LLM"
                    extraction_method = "empty"
                else:
                    # Primary extraction method: <json> tags
                    extracted = _extract_jsons(last_message)
                    if extracted and len(extracted) > 0:
                        last_json = extracted[-1]
                        if isinstance(last_json, dict) and "response" in last_json:
                            prediction = last_json["response"]
                            extraction_method = "primary"
                            self.stats["json_extracted"] += 1
                        elif isinstance(last_json, dict):
                            # JSON found but no response field - use first string value
                            for key, value in last_json.items():
                                if isinstance(value, str):
                                    prediction = value
                                    extraction_method = "primary_alt"
                                    self.stats["json_extracted"] += 1
                                    self.log_fn(f"Using alternative field '{key}' from JSON")
                                    break
                            else:
                                prediction = json.dumps(last_json)
                                extraction_method = "primary_json"
                                self.stats["json_extracted"] += 1
                    
                    # Fallback extraction: code blocks and raw JSON
                    if extraction_method == "none":
                        fallback = _extract_json_fallback(last_message)
                        if fallback and isinstance(fallback, dict):
                            if "response" in fallback:
                                prediction = fallback["response"]
                                extraction_method = "fallback"
                                self.stats["fallback_used"] += 1
                                self.log_fn(f"Used fallback extraction for response")
                            else:
                                # Use first string value from fallback JSON
                                for key, value in fallback.items():
                                    if isinstance(value, str):
                                        prediction = value
                                        extraction_method = "fallback_alt"
                                        self.stats["fallback_used"] += 1
                                        self.log_fn(f"Using alternative field '{key}' from fallback JSON")
                                        break
                    
                    # Heuristic extraction: pattern matching
                    if extraction_method == "none":
                        heuristic = _extract_response_heuristic(last_message)
                        if heuristic:
                            prediction = heuristic
                            extraction_method = "heuristic"
                            self.stats["heuristic_used"] += 1
                            self.log_fn(f"Used heuristic extraction")
                    
                    # Last resort: use raw text (truncated)
                    if extraction_method == "none":
                        prediction = last_message[:1000]  # Increased limit for better context
                        extraction_method = "raw"
                        self.stats["raw_used"] += 1
                        self.log_fn(f"Using raw text extraction (limited to 1000 chars)")
                        
                self.log_fn(f"Extraction method: {extraction_method}, prediction length: {len(str(prediction))}")
                
                # Validate prediction isn't empty
                if not prediction or not str(prediction).strip():
                    prediction = "Error: Empty prediction extracted"
                    self.stats["empty_response"] += 1
                    
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            self.stats["failed"] += 1
            prediction = f"Error: Extraction failed - {e}"

        return str(prediction), msg_history
    
    def get_stats(self) -> dict[str, Any]:
        """Return extraction statistics."""
        return self.stats.copy()
    
    def reset_stats(self) -> None:
        """Reset all statistics counters."""
        self.stats = {
            "total_calls": 0, 
            "json_extracted": 0, 
            "fallback_used": 0, 
            "heuristic_used": 0,
            "raw_used": 0,
            "failed": 0,
            "empty_response": 0,
        }
