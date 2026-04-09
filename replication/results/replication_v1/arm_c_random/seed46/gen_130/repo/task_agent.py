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
    
    Enhanced to handle:
    - Nested braces at any depth
    - Escaped quotes within strings
    - Multiple JSON blocks in sequence
    - Malformed JSON with recovery attempts
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
        
        # Extract content between tags
        inner = text[start + 6:end].strip()
        search_from = end + 7
        
        # Skip empty blocks
        if not inner:
            continue
        
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
            # Use a more robust pattern that handles nested structures
            response_match = re.search(r'"response"\s*:\s*(.+?)(?:,\s*"[^"]*"\s*:|$)', inner, re.DOTALL)
            if response_match:
                response_value = response_match.group(1).strip()
                # Try to parse the response value
                try:
                    parsed_value = json.loads(response_value)
                    results.append({"response": parsed_value})
                    continue
                except json.JSONDecodeError:
                    # If it's not valid JSON, treat as string (remove surrounding quotes if present)
                    if response_value.startswith('"') and response_value.endswith('"'):
                        response_value = response_value[1:-1]
                    results.append({"response": response_value})
                    continue
        except Exception:
            pass
        
        # Last resort: try to find any valid JSON object in the block
        try:
            # Find the first '{' and last '}' to extract potential JSON
            json_start = inner.find('{')
            json_end = inner.rfind('}')
            if json_start != -1 and json_end != -1 and json_end > json_start:
                potential_json = inner[json_start:json_end + 1]
                try:
                    results.append(json.loads(potential_json))
                    continue
                except json.JSONDecodeError:
                    try:
                        cleaned = _clean_json_string(potential_json)
                        results.append(json.loads(cleaned))
                        continue
                    except json.JSONDecodeError:
                        pass
        except Exception:
            pass
        
        logger.debug(f"Failed to parse JSON from <json> block: {inner[:100]}...")
        continue
    
    return results or None


def _clean_json_string(text: str) -> str:
    """Clean common JSON formatting issues.
    
    Removes trailing commas, comments, normalizes whitespace,
    and fixes common JSON syntax errors from LLM outputs.
    """
    import re
    
    # Remove single-line comments
    text = re.sub(r'//.*$', '', text, flags=re.MULTILINE)
    # Remove multi-line comments
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    # Remove trailing commas before closing braces/brackets
    text = re.sub(r',(\s*[}\]])', r'\1', text)
    # Fix single quotes to double quotes (common LLM mistake)
    text = re.sub(r"(?<!\\)'", '"', text)
    # Fix unquoted keys (common LLM mistake: {key: value} -> {"key": value})
    text = re.sub(r'(\{|,\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', text)
    # Remove control characters
    text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f]', '', text)
    # Normalize line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    return text.strip()


def _extract_json_fallback(text: str) -> list[dict] | None:
    """Fallback JSON extraction using regex for unwrapped JSON objects.

    This handles cases where the model outputs valid JSON without <json> tags.
    Searches for JSON objects with a "response" key.
    
    Enhanced with multiple fallback strategies and better error recovery.
    """
    results = []
    
    # Strategy 1: Pattern to match JSON objects with response key (handles nested braces)
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

    # Strategy 2: Look for any JSON object in the text using balanced brace matching
    if not results:
        # Find all potential JSON objects with balanced braces
        start_idx = 0
        while True:
            brace_start = text.find('{', start_idx)
            if brace_start == -1:
                break
            
            # Find matching closing brace
            brace_count = 0
            brace_end = -1
            for i, char in enumerate(text[brace_start:], start=brace_start):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        brace_end = i
                        break
            
            if brace_end == -1:
                start_idx = brace_start + 1
                continue
            
            potential_json = text[brace_start:brace_end + 1]
            start_idx = brace_end + 1
            
            try:
                obj = json.loads(potential_json)
                if isinstance(obj, dict) and "response" in obj:
                    results.append(obj)
            except json.JSONDecodeError:
                try:
                    cleaned = _clean_json_string(potential_json)
                    obj = json.loads(cleaned)
                    if isinstance(obj, dict) and "response" in obj:
                        results.append(obj)
                except json.JSONDecodeError:
                    continue

    # Strategy 3: Try to parse the entire text as JSON
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

    # Strategy 4: Look for response field with regex and construct JSON
    if not results:
        response_pattern = r'"response"\s*:\s*([^,\n}]+|"[^"]*"|\{[^}]*\})'
        match = re.search(response_pattern, text, re.DOTALL)
        if match:
            response_value = match.group(1).strip()
            # Try to parse as JSON value
            try:
                parsed = json.loads(response_value)
                results.append({"response": parsed})
            except json.JSONDecodeError:
                # Treat as string, removing quotes if present
                if response_value.startswith('"') and response_value.endswith('"'):
                    response_value = response_value[1:-1]
                results.append({"response": response_value})

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
        
        # Log the raw response for debugging (truncated)
        debug_text = assistant_text[:500] + "..." if len(assistant_text) > 500 else assistant_text
        self.log_fn(f"Raw LLM response (truncated): {debug_text}")
        
        try:
            extracted = _extract_jsons(assistant_text)
            if extracted and len(extracted) > 0:
                last_extracted = extracted[-1]
                if isinstance(last_extracted, dict) and "response" in last_extracted:
                    prediction = last_extracted["response"]
                    extraction_method = "primary"
                    self._extraction_stats["primary_success"] += 1
                    self.log_fn(f"Primary extraction succeeded. Response type: {type(prediction).__name__}")
                else:
                    self.log_fn(f"Primary extraction: JSON found but no 'response' key. Keys: {list(last_extracted.keys()) if isinstance(last_extracted, dict) else 'N/A'}")
                    # Try fallback extraction
                    extracted = _extract_json_fallback(assistant_text)
                    if extracted and len(extracted) > 0:
                        last_fallback = extracted[-1]
                        if isinstance(last_fallback, dict) and "response" in last_fallback:
                            prediction = last_fallback["response"]
                            extraction_method = "fallback"
                            self._extraction_stats["fallback_success"] += 1
                            self.log_fn(f"Fallback extraction succeeded. Response type: {type(prediction).__name__}")
            else:
                self.log_fn("Primary extraction: No JSON objects found in response")
                # Try fallback extraction
                extracted = _extract_json_fallback(assistant_text)
                if extracted and len(extracted) > 0:
                    last_fallback = extracted[-1]
                    if isinstance(last_fallback, dict) and "response" in last_fallback:
                        prediction = last_fallback["response"]
                        extraction_method = "fallback"
                        self._extraction_stats["fallback_success"] += 1
                        self.log_fn(f"Fallback extraction succeeded. Response type: {type(prediction).__name__}")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            # Try fallback on exception
            try:
                extracted = _extract_json_fallback(assistant_text)
                if extracted and len(extracted) > 0:
                    last_fallback = extracted[-1]
                    if isinstance(last_fallback, dict) and "response" in last_fallback:
                        prediction = last_fallback["response"]
                        extraction_method = "fallback"
                        self._extraction_stats["fallback_success"] += 1
                        self.log_fn(f"Fallback extraction succeeded after primary exception. Response type: {type(prediction).__name__}")
            except Exception as fallback_e:
                self.log_fn(f"Fallback extraction also failed: {fallback_e}")
                self._extraction_stats["failed"] += 1

        self.log_fn(f"Extraction method used: {extraction_method}, prediction: {str(prediction)[:100]}")
        
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
