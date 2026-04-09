"""
Task agent: solves a given task with a single LLM call.

Reimplemented from facebookresearch/HyperAgents task_agent.py.
Same interface, same JSON output format, same extraction logic.

This is the INITIAL task agent. The meta agent modifies this file
during self-improvement. The evaluation harness loads whatever
task_agent.py exists at the agent's repo path.

Enhancements:
- Improved JSON extraction with multiple fallback strategies
- Better error handling and logging
- Response validation with schema checking
- Configurable extraction parameters
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
        except json.JSONDecodeError:
            continue
    return results or None


def _extract_json_from_markdown(text: str) -> list[dict] | None:
    """Extract JSON from markdown code blocks (```json ... ```).
    
    This provides an additional fallback when <json> tags are not used.
    """
    results = []
    # Match ```json ... ``` or ``` ... ``` blocks
    pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    matches = re.findall(pattern, text, re.DOTALL)
    
    for match in matches:
        try:
            # Try to parse the content as JSON
            data = json.loads(match.strip())
            if isinstance(data, dict):
                results.append(data)
        except json.JSONDecodeError:
            # Try to find JSON objects within the block
            try:
                # Look for JSON-like structures
                json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
                json_matches = re.findall(json_pattern, match, re.DOTALL)
                for json_str in json_matches:
                    data = json.loads(json_str)
                    if isinstance(data, dict):
                        results.append(data)
            except json.JSONDecodeError:
                continue
    
    return results or None


def _extract_json_with_retry(text: str, max_retries: int = 2) -> list[dict] | None:
    """Extract JSON with multiple fallback strategies.
    
    First tries standard extraction from <json> tags, then attempts to fix 
    common JSON formatting issues like trailing commas, then tries
    markdown code block extraction, and finally attempts brute-force
    JSON object detection as a last resort.
    """
    # First attempt: standard extraction from <json> tags
    result = _extract_jsons(text)
    if result:
        logger.debug(f"JSON extraction succeeded from <json> tags: {len(result)} objects")
        return result
    
    # Second attempt: markdown code blocks
    result = _extract_json_from_markdown(text)
    if result:
        logger.debug(f"JSON extraction succeeded from markdown blocks: {len(result)} objects")
        return result
    
    # Retry with common fixes for malformed JSON
    for attempt in range(max_retries):
        try:
            # Try to find and fix JSON with trailing commas
            fixed_text = re.sub(r',(\s*[}\]])', r'\1', text)
            # Also fix single quotes to double quotes
            fixed_text = re.sub(r"'([^']*)'", r'"\1"', fixed_text)
            result = _extract_jsons(fixed_text)
            if result:
                logger.debug(f"JSON extraction succeeded after fix attempt {attempt + 1}")
                return result
            # Try markdown extraction on fixed text too
            result = _extract_json_from_markdown(fixed_text)
            if result:
                logger.debug(f"Markdown JSON extraction succeeded after fix attempt {attempt + 1}")
                return result
        except Exception as e:
            logger.debug(f"Fix attempt {attempt + 1} failed: {e}")
    
    # Final fallback: brute-force extraction of any JSON-like structures
    try:
        result = _extract_json_brute_force(text)
        if result:
            logger.debug(f"JSON extraction succeeded via brute-force: {len(result)} objects")
            return result
    except Exception as e:
        logger.debug(f"Brute-force extraction failed: {e}")
    
    logger.warning("All JSON extraction strategies failed")
    return None


def _extract_json_brute_force(text: str) -> list[dict] | None:
    """Brute-force extraction: find any {...} structures and try to parse them.
    
    This is a last-resort strategy that scans the text for curly-brace-delimited
    structures and attempts to parse each one as JSON.
    """
    results = []
    
    # Find all potential JSON object starts
    i = 0
    while i < len(text):
        # Find next opening brace
        brace_start = text.find('{', i)
        if brace_start == -1:
            break
        
        # Try to find a matching closing brace by counting braces
        brace_count = 0
        in_string = False
        escape_next = False
        
        for j in range(brace_start, len(text)):
            char = text[j]
            
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
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        # Found a complete JSON-like structure
                        json_str = text[brace_start:j+1]
                        try:
                            data = json.loads(json_str)
                            if isinstance(data, dict):
                                results.append(data)
                        except json.JSONDecodeError:
                            # Try common fixes
                            try:
                                # Remove trailing commas
                                fixed = re.sub(r',(\s*[}\]])', r'\1', json_str)
                                data = json.loads(fixed)
                                if isinstance(data, dict):
                                    results.append(data)
                            except json.JSONDecodeError:
                                pass
                        i = j + 1
                        break
        else:
            # No matching closing brace found
            i = brace_start + 1
    
    return results or None


def _validate_response_schema(data: dict, required_keys: list[str]) -> tuple[bool, str]:
    """Validate that the response contains required keys.
    
    Args:
        data: The parsed JSON data
        required_keys: List of keys that must be present
        
    Returns:
        (is_valid, error_message)
    """
    missing_keys = [key for key in required_keys if key not in data]
    if missing_keys:
        return False, f"Missing required keys: {missing_keys}"
    return True, ""


def _safe_extract_prediction(extracted: list[dict] | None, 
                              key: str = "response", 
                              default: Any = "None") -> Any:
    """Safely extract a prediction from extracted JSON data.
    
    Args:
        extracted: List of extracted JSON objects
        key: The key to extract from the last object
        default: Default value if extraction fails
        
    Returns:
        The extracted value or default
    """
    if not extracted or not isinstance(extracted, list):
        return default
    
    if len(extracted) == 0:
        return default
    
    last_item = extracted[-1]
    if not isinstance(last_item, dict):
        return default
    
    return last_item.get(key, default)


class TaskAgent:
    """Task agent that solves IMO grading problems with enhanced validation.
    
    This agent uses an LLM to process grading tasks and extract structured
    responses. It includes robust JSON extraction with multiple fallback
    strategies and comprehensive error handling.
    """

    def __init__(
        self, 
        model: str = EVAL_MODEL, 
        log_file: str = "",
        required_keys: list[str] | None = None,
        response_key: str = "response",
        default_response: Any = "None"
    ) -> None:
        """Initialize the TaskAgent.
        
        Args:
            model: The LLM model to use for inference
            log_file: Path to log file (currently unused, for compatibility)
            required_keys: List of required keys in the JSON response
            response_key: The key to extract from the response
            default_response: Default value if extraction fails
        """
        self.model = model
        self.log_fn = logger.info
        self.required_response_keys = required_keys or ["response"]
        self.response_key = response_key
        self.default_response = default_response

    def _build_instruction(self, inputs: dict) -> str:
        """Build the instruction prompt for the LLM.
        
        Args:
            inputs: Task input dictionary
            
        Returns:
            Formatted instruction string
        """
        return f"""You are an expert grading agent. Your task is to evaluate student answers based on the provided problem, solution, and grading guidelines.

Task input:
```
{json.dumps(inputs, indent=2, default=str)}
```

Analyze the problem carefully and provide your evaluation in the following JSON format:
<json>
{{
    "response": "Your detailed evaluation here"
}}
</json>

Ensure your response is valid JSON and follows the specified schema exactly."""

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        instruction = self._build_instruction(inputs)

        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
        except Exception as e:
            self.log_fn(f"LLM call failed: {e}")
            return str(self.default_response), []

        # Extract prediction from JSON with retry mechanism and validation
        prediction = self.default_response
        try:
            if not msg_history or len(msg_history) < 2:
                self.log_fn("Invalid message history from LLM")
                return str(prediction), msg_history
                
            last_message = msg_history[-1].get("text", "")
            if not last_message:
                self.log_fn("Empty response from LLM")
                return str(prediction), msg_history
            
            extracted = _extract_json_with_retry(last_message)
            
            if extracted:
                # Validate schema
                is_valid, error_msg = _validate_response_schema(
                    extracted[-1], 
                    self.required_response_keys
                )
                
                if is_valid:
                    prediction = _safe_extract_prediction(
                        extracted, 
                        self.response_key, 
                        self.default_response
                    )
                    self.log_fn(f"Successfully extracted prediction: {prediction}")
                else:
                    self.log_fn(f"Schema validation failed: {error_msg}")
                    # Try to extract anyway as a fallback
                    prediction = _safe_extract_prediction(
                        extracted, 
                        self.response_key, 
                        self.default_response
                    )
            else:
                self.log_fn("No JSON data extracted from response")
                # Try to extract any meaningful text as fallback
                if last_message.strip():
                    # Remove common formatting artifacts
                    cleaned = re.sub(r'<json>|</json>|```json|```', '', last_message)
                    cleaned = cleaned.strip()
                    if cleaned:
                        prediction = cleaned[:500]  # Limit length
                        
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
