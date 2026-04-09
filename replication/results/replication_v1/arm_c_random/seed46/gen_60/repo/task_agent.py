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
    """
    results = []
    
    # Use a stack-based approach to find balanced braces with "response" key
    i = 0
    while i < len(text):
        # Find the start of a potential JSON object
        brace_start = text.find('{', i)
        if brace_start == -1:
            break
            
        # Check if this object contains "response" key nearby
        search_window = text[brace_start:brace_start + 200]
        if '"response"' not in search_window:
            i = brace_start + 1
            continue
        
        # Try to find the matching closing brace using a stack
        stack = []
        j = brace_start
        in_string = False
        escape_next = False
        
        while j < len(text):
            char = text[j]
            
            if escape_next:
                escape_next = False
                j += 1
                continue
                
            if char == '\\' and in_string:
                escape_next = True
                j += 1
                continue
                
            if char == '"' and (j == 0 or text[j-1] != '\\'):
                in_string = not in_string
                j += 1
                continue
                
            if not in_string:
                if char == '{':
                    stack.append('{')
                elif char == '}':
                    if stack:
                        stack.pop()
                    if not stack:
                        # Found a complete JSON object
                        json_str = text[brace_start:j+1]
                        try:
                            obj = json.loads(json_str)
                            if isinstance(obj, dict) and "response" in obj:
                                results.append(obj)
                        except json.JSONDecodeError:
                            pass
                        i = j + 1
                        break
            j += 1
        else:
            # No matching brace found
            i = brace_start + 1

    # If stack-based approach fails, try to parse the entire text as JSON
    if not results:
        try:
            obj = json.loads(text.strip())
            if isinstance(obj, dict) and "response" in obj:
                results.append(obj)
        except json.JSONDecodeError:
            pass

    return results or None


def _extract_json_heuristic(text: str) -> list[dict] | None:
    """Heuristic extraction for malformed JSON responses.
    
    Handles common LLM output issues like:
    - Trailing commas before closing braces
    - Single quotes instead of double quotes
    - Unquoted keys
    - Comments in JSON
    """
    results = []
    
    # Try to find JSON-like blocks with response key
    # Look for patterns like {"response": ...} or { "response" : ... }
    pattern = r'\{\s*"response"\s*:[\s\S]*?\}'
    matches = re.finditer(pattern, text, re.DOTALL)
    
    for match in matches:
        json_str = match.group()
        
        # Try original first
        try:
            obj = json.loads(json_str)
            if "response" in obj:
                results.append(obj)
                continue
        except json.JSONDecodeError:
            pass
        
        # Try fixing common issues
        fixed = json_str
        
        # Remove trailing commas before } or ]
        fixed = re.sub(r',(\s*[}\]])', r'\1', fixed)
        
        # Replace single quotes with double quotes (carefully)
        # Only replace quotes that appear to be delimiting strings
        fixed = re.sub(r"(?<!\\)'([^']*?)'(?=\s*[:}\],])", r'"\1"', fixed)
        
        # Remove comments (both // and /* */)
        fixed = re.sub(r'//[^\n]*', '', fixed)
        fixed = re.sub(r'/\*[\s\S]*?\*/', '', fixed)
        
        try:
            obj = json.loads(fixed)
            if "response" in obj:
                results.append(obj)
                logger.debug(f"Heuristic extraction succeeded after fixes")
        except json.JSONDecodeError:
            continue
    
    return results or None


class TaskAgent:
    """Task agent that solves IMO grading problems with robust JSON extraction."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.max_retries = 2  # Number of retries for JSON extraction failures

    def _build_prompt(self, inputs: dict, retry_count: int = 0) -> str:
        """Build the prompt for the LLM.
        
        On retries, adds stronger emphasis on proper JSON formatting.
        """
        base_instruction = f"""You are an agent.

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

        if retry_count > 0:
            # Add stronger guidance on retry
            base_instruction += f"""

IMPORTANT: Your previous response could not be parsed as valid JSON. 
Please ensure your response:
1. Is wrapped in <json>...</json> tags
2. Uses double quotes for strings and keys
3. Has no trailing commas
4. Contains valid JSON only within the tags

Example of a valid response:
<json>
{{
    "response": "your answer here"
}}
</json>"""
        
        return base_instruction

    def _extract_prediction(self, text: str, attempt: int = 0) -> tuple[str, str]:
        """Extract prediction from response text using multiple methods.
        
        Returns:
            (prediction, extraction_method)
        """
        # Try primary extraction first
        extracted = _extract_jsons(text)
        if extracted and "response" in extracted[-1]:
            return extracted[-1]["response"], "primary"
        
        # Try fallback extraction
        extracted = _extract_json_fallback(text)
        if extracted and "response" in extracted[-1]:
            return extracted[-1]["response"], "fallback"
        
        # Try heuristic extraction as last resort
        extracted = _extract_json_heuristic(text)
        if extracted and "response" in extracted[-1]:
            return extracted[-1]["response"], "heuristic"
        
        return "None", "failed"

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        msg_history = []
        prediction = "None"
        extraction_method = "failed"
        
        for attempt in range(self.max_retries + 1):
            instruction = self._build_prompt(inputs, retry_count=attempt)
            
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=msg_history if attempt > 0 else [],
                )
            except Exception as e:
                self.log_fn(f"Error calling LLM on attempt {attempt}: {e}")
                if attempt == self.max_retries:
                    return "Error: LLM call failed", [{"role": "system", "text": f"Error: {e}"}]
                continue

            # Extract prediction from JSON
            last_text = msg_history[-1]["text"] if msg_history else ""
            prediction, extraction_method = self._extract_prediction(last_text, attempt)
            
            if extraction_method != "failed":
                break
            
            self.log_fn(f"JSON extraction failed on attempt {attempt}, retrying...")
        
        self.log_fn(f"Extraction method used: {extraction_method} after {attempt + 1} attempt(s)")
        return str(prediction), msg_history
