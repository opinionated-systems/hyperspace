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
        try:
            results.append(json.loads(inner))
        except json.JSONDecodeError as e:
            # Try to extract with more lenient parsing for common issues
            try:
                # Handle cases where JSON might have trailing commas or comments
                cleaned = _clean_json_string(inner)
                results.append(json.loads(cleaned))
            except json.JSONDecodeError:
                logger.debug(f"Failed to parse JSON: {e}")
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
    
    Uses a multi-layered approach:
    1. Balanced brace matching for nested JSON objects
    2. Line-by-line JSON object detection
    3. Full text parsing with aggressive cleaning
    """
    results = []
    
    # First try: balanced brace matching for nested structures
    # This approach properly handles nested braces by counting open/close braces
    def find_json_objects_balanced(text: str) -> list[str]:
        """Find JSON objects using balanced brace counting."""
        objects = []
        i = 0
        while i < len(text):
            if text[i] == '{':
                start = i
                brace_count = 1
                i += 1
                in_string = False
                escape_next = False
                
                while i < len(text) and brace_count > 0:
                    char = text[i]
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
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                    i += 1
                
                if brace_count == 0:
                    objects.append(text[start:i])
            else:
                i += 1
        return objects
    
    # Try balanced brace extraction first
    balanced_objects = find_json_objects_balanced(text)
    for obj_str in balanced_objects:
        if '"response"' in obj_str:
            try:
                obj = json.loads(obj_str)
                if isinstance(obj, dict) and "response" in obj:
                    results.append(obj)
            except json.JSONDecodeError:
                # Try with progressively more aggressive cleaning
                for cleaner in [_clean_json_string, _aggressive_json_clean]:
                    try:
                        cleaned = cleaner(obj_str)
                        obj = json.loads(cleaned)
                        if isinstance(obj, dict) and "response" in obj:
                            results.append(obj)
                            break
                    except json.JSONDecodeError:
                        continue
    
    # Second try: look for JSON-like patterns with response key
    if not results:
        # Pattern to find objects containing "response" key
        response_pattern = r'\{[^}]*"response"[^}]*\}'
        for match in re.finditer(response_pattern, text, re.DOTALL):
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
        for cleaner in [lambda x: x.strip(), _clean_json_string, _aggressive_json_clean]:
            try:
                obj = json.loads(cleaner(text))
                if isinstance(obj, dict) and "response" in obj:
                    results.append(obj)
                    break
            except json.JSONDecodeError:
                continue
    
    # Fourth try: extract response value using regex if all else fails
    if not results:
        response_value = _extract_response_value_regex(text)
        if response_value is not None:
            results.append({"response": response_value})

    return results or None


def _aggressive_json_clean(text: str) -> str:
    """Aggressively clean JSON string for maximum parsing compatibility.
    
    Handles common LLM output issues like:
    - Smart quotes and unicode issues
    - Unescaped newlines in strings
    - Missing quotes around keys
    - Trailing/leading garbage
    """
    import re
    # Replace smart quotes with regular quotes
    text = text.replace('"', '"').replace('"', '"').replace("'", "'")
    # Remove control characters except newlines and tabs
    text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t')
    # Remove trailing commas before closing braces/brackets
    text = re.sub(r',(\s*[}\]])', r'\1', text)
    # Remove single-line comments
    text = re.sub(r'//.*$', '', text, flags=re.MULTILINE)
    # Remove multi-line comments
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    return text.strip()


def _extract_response_value_regex(text: str) -> str | None:
    """Extract response value using regex as last resort.
    
    Looks for patterns like:
    - "response": "value"
    - "response": "multi-line value"
    - response: value (without quotes)
    """
    import re
    
    # Pattern 1: Standard quoted value
    pattern1 = r'"response"\s*:\s*"([^"]*)"'
    match = re.search(pattern1, text)
    if match:
        return match.group(1)
    
    # Pattern 2: Multi-line quoted value
    pattern2 = r'"response"\s*:\s*"((?:[^"\\]|\\.)*)"'
    match = re.search(pattern2, text, re.DOTALL)
    if match:
        return match.group(1).replace('\\n', '\n').replace('\\t', '\t')
    
    # Pattern 3: Unquoted value (alphanumeric/underscore)
    pattern3 = r'"response"\s*:\s*([a-zA-Z0-9_]+)'
    match = re.search(pattern3, text)
    if match:
        return match.group(1)
    
    return None


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
        # Build a more structured prompt for better LLM understanding
        domain = inputs.get("domain", "unknown")
        problem = inputs.get("problem", "")
        
        instruction = f"""You are an expert grading agent for {domain} problems.

Your task is to evaluate a student's answer and provide a grade or assessment.

Problem Statement:
```
{problem}
```

Reference Solution:
```
{inputs.get("solution", "N/A")}
```

Grading Guidelines:
```
{inputs.get("grading_guidelines", "N/A")}
```

Student's Answer:
```
{inputs.get("student_answer", "N/A")}
```

Respond in JSON format with the following schema:
<json>
{{
    "response": "Your grade/assessment here (e.g., 'Correct', 'Partial', 'Incorrect', or a numeric score)"
}}
</json>

Provide only the JSON response, no additional text."""

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
        
        # Get the last assistant response for extraction
        last_response_text = ""
        for msg in reversed(msg_history):
            if msg.get("role") == "assistant":
                last_response_text = msg.get("text", "")
                break
        
        if not last_response_text:
            self.log_fn("Warning: No assistant response found in message history")
            self._extraction_stats["failed"] += 1
            return str(prediction), msg_history
        
        try:
            extracted = _extract_jsons(last_response_text)
            if extracted and len(extracted) > 0 and "response" in extracted[-1]:
                prediction = extracted[-1]["response"]
                extraction_method = "primary"
                self._extraction_stats["primary_success"] += 1
                self.log_fn(f"Primary extraction successful. Response preview: {str(prediction)[:100]}")
            else:
                # Try fallback extraction
                self.log_fn("Primary extraction failed or no 'response' key found, trying fallback...")
                extracted = _extract_json_fallback(last_response_text)
                if extracted and len(extracted) > 0 and "response" in extracted[-1]:
                    prediction = extracted[-1]["response"]
                    extraction_method = "fallback"
                    self._extraction_stats["fallback_success"] += 1
                    self.log_fn(f"Fallback extraction successful. Response preview: {str(prediction)[:100]}")
                else:
                    self.log_fn(f"Fallback extraction also failed. Raw response preview: {last_response_text[:200]}")
                    self._extraction_stats["failed"] += 1
        except Exception as e:
            self.log_fn(f"Error during primary extraction: {e}")
            # Try fallback on exception
            try:
                extracted = _extract_json_fallback(last_response_text)
                if extracted and len(extracted) > 0 and "response" in extracted[-1]:
                    prediction = extracted[-1]["response"]
                    extraction_method = "fallback"
                    self._extraction_stats["fallback_success"] += 1
                    self.log_fn(f"Fallback extraction successful after primary exception. Response preview: {str(prediction)[:100]}")
                else:
                    self._extraction_stats["failed"] += 1
            except Exception as fallback_e:
                self.log_fn(f"Fallback extraction also failed: {fallback_e}")
                self._extraction_stats["failed"] += 1

        self.log_fn(f"Extraction method used: {extraction_method}")
        return str(prediction), msg_history
