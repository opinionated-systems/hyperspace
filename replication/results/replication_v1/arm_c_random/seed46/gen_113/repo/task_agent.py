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
    """Extract JSON objects from <json>...</json> blocks with enhanced robustness.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    Also handles nested JSON objects within the tags.
    
    Enhanced with:
    - Better handling of markdown code blocks inside <json> tags
    - Support for multiple JSON objects in a single block
    - Improved error recovery for malformed JSON
    - Optimized brace matching algorithm
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
        
        # Remove markdown code block markers if present
        inner = inner.removeprefix("```json").removeprefix("```")
        inner = inner.removesuffix("```").strip()
        
        # Try to parse the inner content as JSON directly first
        try:
            results.append(json.loads(inner))
            continue
        except json.JSONDecodeError:
            pass
        
        # Fallback: extract JSON objects using brace matching
        json_objects = _extract_json_objects_by_braces(inner)
        for obj in json_objects:
            try:
                results.append(json.loads(obj))
            except json.JSONDecodeError:
                continue
                
    return results or None


def _extract_json_objects_by_braces(text: str) -> list[str]:
    """Extract JSON object strings from text using brace matching.
    
    This handles nested JSON objects by tracking brace depth.
    Returns a list of JSON object strings that can be parsed.
    """
    json_objects = []
    brace_count = 0
    json_start = -1
    in_string = False
    escape_next = False
    
    for i, char in enumerate(text):
        if escape_next:
            escape_next = False
            continue
            
        if char == '\\':
            escape_next = True
            continue
            
        if char == '"' and not in_string:
            in_string = True
        elif char == '"' and in_string:
            in_string = False
        elif not in_string:
            if char == '{':
                if brace_count == 0:
                    json_start = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and json_start != -1:
                    json_objects.append(text[json_start:i+1])
                    json_start = -1
                    
    return json_objects


def _extract_json_fallback(text: str) -> list[dict] | None:
    """Fallback JSON extraction for unwrapped JSON objects.

    This handles cases where the model outputs valid JSON without <json> tags.
    Searches for JSON objects with a "response" key using multiple strategies.
    
    Strategies:
    1. Try to parse the entire text as JSON
    2. Use brace matching to find JSON objects with "response" key
    3. Use regex pattern matching as last resort
    """
    results = []
    
    # Strategy 1: Try to parse the entire text as JSON
    try:
        obj = json.loads(text.strip())
        if isinstance(obj, dict) and "response" in obj:
            results.append(obj)
            return results
    except json.JSONDecodeError:
        pass

    # Strategy 2: Use brace matching to find JSON objects
    json_objects = _extract_json_objects_by_braces(text)
    for json_str in json_objects:
        try:
            obj = json.loads(json_str)
            if isinstance(obj, dict) and "response" in obj:
                results.append(obj)
        except json.JSONDecodeError:
            continue
    
    if results:
        return results

    # Strategy 3: Regex pattern matching for simple cases
    pattern = r'\{[^{}]*"response"[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.finditer(pattern, text, re.DOTALL)

    for match in matches:
        try:
            obj = json.loads(match.group())
            if "response" in obj:
                results.append(obj)
        except json.JSONDecodeError:
            continue

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
        self._total_calls = 0

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        self._total_calls += 1
        
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
        prediction, extraction_method = self._extract_prediction(msg_history)
        
        self.log_fn(f"Extraction method used: {extraction_method}")
        self.log_fn(f"Extraction stats: {self._extraction_stats} (total calls: {self._total_calls})")
        return str(prediction), msg_history

    def _extract_prediction(self, msg_history: list[dict]) -> tuple[str, str]:
        """Extract prediction from the last message in history.
        
        Args:
            msg_history: List of message dictionaries
            
        Returns:
            Tuple of (prediction, extraction_method)
        """
        if not msg_history:
            self._extraction_stats["failed"] += 1
            return "None", "failed"
            
        last_message = msg_history[-1]
        text = last_message.get("text", "")
        
        # Try primary extraction first
        try:
            extracted = _extract_jsons(text)
            if extracted and "response" in extracted[-1]:
                self._extraction_stats["primary"] += 1
                return extracted[-1]["response"], "primary"
        except Exception as e:
            self.log_fn(f"Primary extraction error: {e}")
        
        # Try fallback extraction
        try:
            extracted = _extract_json_fallback(text)
            if extracted and "response" in extracted[-1]:
                self._extraction_stats["fallback"] += 1
                return extracted[-1]["response"], "fallback"
        except Exception as e:
            self.log_fn(f"Fallback extraction error: {e}")
        
        self._extraction_stats["failed"] += 1
        return "None", "failed"

    def get_extraction_stats(self) -> dict:
        """Return the current extraction statistics.
        
        Returns:
            Dictionary with counts of primary, fallback, and failed extractions.
        """
        stats = dict(self._extraction_stats)
        stats["total_calls"] = self._total_calls
        if self._total_calls > 0:
            stats["success_rate"] = (self._total_calls - stats["failed"]) / self._total_calls
        return stats

    def reset_extraction_stats(self) -> None:
        """Reset extraction statistics to zero."""
        self._extraction_stats = {"primary": 0, "fallback": 0, "failed": 0}
        self._total_calls = 0
