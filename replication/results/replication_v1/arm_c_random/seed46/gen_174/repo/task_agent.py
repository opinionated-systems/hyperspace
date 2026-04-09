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
    """Fallback JSON extraction for unwrapped JSON objects.

    Uses a robust stack-based parser to find balanced JSON objects
    containing a "response" key. Handles nested structures and escaped quotes.
    """
    results = []
    i = 0
    text_len = len(text)
    
    while i < text_len:
        # Find potential JSON object start
        brace_start = text.find('{', i)
        if brace_start == -1:
            break
            
        # Quick check: look for "response" key within reasonable distance
        search_end = min(brace_start + 300, text_len)
        if '"response"' not in text[brace_start:search_end]:
            i = brace_start + 1
            continue
        
        # Parse JSON object with proper state machine
        depth = 0
        j = brace_start
        in_string = False
        escape_next = False
        
        while j < text_len:
            char = text[j]
            
            if escape_next:
                escape_next = False
                j += 1
                continue
                
            if char == '\\':
                escape_next = True
                j += 1
                continue
                
            if char == '"':
                in_string = not in_string
                j += 1
                continue
                
            if not in_string:
                if char == '{':
                    depth += 1
                elif char == '}':
                    depth -= 1
                    if depth == 0:
                        # Complete JSON object found
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
            # No closing brace found
            i = brace_start + 1

    # Final fallback: try parsing entire text as JSON
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
    - Newlines in string values
    """
    results = []
    
    # Find JSON-like blocks with response key using improved pattern
    # Matches both {"response":...} and { "response" : ... } with nested content
    pattern = r'\{\s*"response"\s*:[\s\S]*?\}'
    matches = list(re.finditer(pattern, text, re.DOTALL))
    
    # Sort by length (longest first) to prefer complete objects
    matches.sort(key=lambda m: len(m.group()), reverse=True)
    
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
        
        # Apply progressive fixes
        fixed = json_str
        
        # Fix 1: Remove trailing commas before } or ]
        fixed = re.sub(r',(\s*[}\]])', r'\1', fixed)
        
        # Fix 2: Replace single quotes with double quotes (for keys and string values)
        fixed = re.sub(r"(?<!\\)'([^']*?)'(?=\s*[:}\],])", r'"\1"', fixed)
        
        # Fix 3: Remove comments
        fixed = re.sub(r'//[^\n]*', '', fixed)
        fixed = re.sub(r'/\*[\s\S]*?\*/', '', fixed)
        
        # Fix 4: Escape unescaped newlines in string values
        # This is a simplified approach - replace newlines not preceded by backslash
        fixed = re.sub(r'(?<!\\)\n', r'\\n', fixed)
        fixed = re.sub(r'(?<!\\)\r', r'\\r', fixed)
        
        # Fix 5: Remove control characters
        fixed = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', fixed)
        
        try:
            obj = json.loads(fixed)
            if "response" in obj:
                results.append(obj)
                logger.debug("Heuristic extraction succeeded after fixes")
                continue
        except json.JSONDecodeError:
            pass
        
        # Final attempt: extract just the response value if structure is broken
        response_match = re.search(r'"response"\s*:\s*"([^"]*)"', fixed)
        if response_match:
            response_value = response_match.group(1)
            results.append({"response": response_value})
            logger.debug("Heuristic extraction succeeded with value extraction")
    
    return results or None


class TaskAgent:
    """Task agent that solves IMO grading problems with robust JSON extraction."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.max_retries = 3  # Increased retries for better recovery
        self._extraction_stats = {
            "primary": 0,
            "fallback": 0,
            "heuristic": 0,
            "failed": 0,
        }

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
        # Try primary extraction first (most reliable)
        extracted = _extract_jsons(text)
        if extracted:
            for obj in extracted:
                if isinstance(obj, dict) and "response" in obj:
                    self._extraction_stats["primary"] += 1
                    return obj["response"], "primary"
        
        # Try fallback extraction for unwrapped JSON
        extracted = _extract_json_fallback(text)
        if extracted:
            for obj in extracted:
                if isinstance(obj, dict) and "response" in obj:
                    self._extraction_stats["fallback"] += 1
                    return obj["response"], "fallback"
        
        # Try heuristic extraction for malformed JSON
        extracted = _extract_json_heuristic(text)
        if extracted:
            for obj in extracted:
                if isinstance(obj, dict) and "response" in obj:
                    self._extraction_stats["heuristic"] += 1
                    return obj["response"], "heuristic"
        
        self._extraction_stats["failed"] += 1
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
            
            if attempt < self.max_retries:
                self.log_fn(f"JSON extraction failed on attempt {attempt + 1}, retrying with stronger guidance...")
        
        self.log_fn(
            f"Extraction: method={extraction_method}, "
            f"attempts={attempt + 1}, "
            f"stats={self._extraction_stats}"
        )
        return str(prediction), msg_history
