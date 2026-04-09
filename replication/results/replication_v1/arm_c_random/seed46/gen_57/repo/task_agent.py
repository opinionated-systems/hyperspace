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
    
    Enhanced to handle nested JSON objects and partial extractions.
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
        
        # Try to parse the inner content
        try:
            results.append(json.loads(inner))
        except json.JSONDecodeError:
            # Try to extract JSON from within the content if it's wrapped
            # in markdown code blocks or has extra whitespace
            cleaned = inner.strip()
            
            # Remove markdown code block markers if present
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            elif cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            
            cleaned = cleaned.strip()
            
            try:
                results.append(json.loads(cleaned))
            except json.JSONDecodeError:
                # Try to find JSON object boundaries by brace counting
                # This handles cases where the model outputs multiple JSON objects
                brace_start = cleaned.find("{")
                if brace_start != -1:
                    depth = 0
                    for i, char in enumerate(cleaned[brace_start:]):
                        if char == "{":
                            depth += 1
                        elif char == "}":
                            depth -= 1
                            if depth == 0:
                                try:
                                    obj = json.loads(cleaned[brace_start:brace_start+i+1])
                                    results.append(obj)
                                    break
                                except json.JSONDecodeError:
                                    continue
                continue
    return results or None


def _extract_json_fallback(text: str) -> list[dict] | None:
    """Fallback JSON extraction using regex for unwrapped JSON objects.

    This handles cases where the model outputs valid JSON without <json> tags.
    Searches for JSON objects with a "response" key.
    Improved to handle nested braces more robustly.
    """
    results = []
    
    # First, try to find JSON objects by parsing brace depth
    # This handles nested structures better than regex
    def find_json_objects(s: str) -> list[str]:
        """Find potential JSON objects by tracking brace depth."""
        objects = []
        i = 0
        while i < len(s):
            if s[i] == '{':
                start = i
                depth = 1
                i += 1
                while i < len(s) and depth > 0:
                    if s[i] == '{':
                        depth += 1
                    elif s[i] == '}':
                        depth -= 1
                    i += 1
                if depth == 0:
                    objects.append(s[start:i])
            else:
                i += 1
        return objects
    
    # Try to parse each potential JSON object
    for obj_str in find_json_objects(text):
        try:
            obj = json.loads(obj_str)
            if isinstance(obj, dict) and "response" in obj:
                results.append(obj)
        except json.JSONDecodeError:
            continue
    
    # If no results, try regex pattern for simpler cases
    if not results:
        # Pattern to match JSON objects with response key (simpler cases)
        pattern = r'\{[^{}]*"response"[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.finditer(pattern, text, re.DOTALL)
        for match in matches:
            try:
                obj = json.loads(match.group())
                if "response" in obj:
                    results.append(obj)
            except json.JSONDecodeError:
                continue

    # If still no results, try to parse the entire text as JSON
    if not results:
        try:
            obj = json.loads(text.strip())
            if isinstance(obj, dict) and "response" in obj:
                results.append(obj)
        except json.JSONDecodeError:
            pass
    
    # Final fallback: try to extract any dict-like structure with response key
    if not results:
        # Look for patterns like "response": "..." or 'response': '...'
        response_pattern = r'["\']response["\']\s*:\s*["\']([^"\']+)["\']'
        match = re.search(response_pattern, text)
        if match:
            results.append({"response": match.group(1)})

    return results or None


class TaskAgent:
    """Task agent that solves IMO grading problems with robust JSON extraction."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def _format_instruction(self, inputs: dict) -> str:
        """Format the instruction prompt with structured inputs for better LLM understanding.
        
        Args:
            inputs: dict with task inputs
            
        Returns:
            Formatted instruction string
        """
        # Build a more structured prompt that helps the LLM understand the task
        parts = ["You are an expert grading agent. Your task is to evaluate a student's answer.\n"]
        
        # Add each input field with clear labeling
        for key, value in inputs.items():
            if value is not None and str(value).strip():
                parts.append(f"\n{key.replace('_', ' ').title()}:")
                parts.append(f"```")
                parts.append(str(value))
                parts.append("```")
        
        parts.append("\n\nRespond in JSON format with the following schema:")
        parts.append("<json>")
        parts.append('{"response": "Your evaluation/grade here"}')
        parts.append("</json>")
        parts.append("\nEnsure your response is valid JSON with a 'response' key.")
        
        return "\n".join(parts)

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        instruction = self._format_instruction(inputs)

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
        extraction_method = "primary"
        raw_response = msg_history[-1].get("text", "")
        
        if not raw_response or not raw_response.strip():
            self.log_fn("Warning: Empty response text from LLM")
            return "Error: Empty response from LLM", msg_history
        
        try:
            extracted = _extract_jsons(raw_response)
            if extracted and len(extracted) > 0 and "response" in extracted[-1]:
                prediction = extracted[-1]["response"]
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
        
        # Validate prediction is not empty or just whitespace
        if prediction is None or (isinstance(prediction, str) and not prediction.strip()):
            self.log_fn("Warning: Empty prediction extracted, using fallback")
            prediction = "Error: Empty prediction"
        
        # Convert prediction to string, handling various types
        if isinstance(prediction, (list, dict)):
            prediction = json.dumps(prediction)
        else:
            prediction = str(prediction)
        
        # Final validation: ensure we have a meaningful response
        if prediction.strip() in ["None", "null", "", "Error: Empty prediction"]:
            self.log_fn("Warning: Final prediction appears invalid, checking raw response")
            # Try one more time to extract from raw response
            if raw_response and len(raw_response) > 0:
                # Look for any quoted string that might be the answer
                import re
                quoted = re.findall(r'"response"\s*:\s*"([^"]+)"', raw_response)
                if quoted:
                    prediction = quoted[-1]
                    self.log_fn(f"Final extraction from raw response succeeded")
        
        # Log final prediction length for debugging
        self.log_fn(f"Final prediction length: {len(prediction)} chars")
            
        return prediction, msg_history
