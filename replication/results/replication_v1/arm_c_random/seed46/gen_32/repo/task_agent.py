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
        except json.JSONDecodeError:
            continue
    return results or None


def _extract_json_fallback(text: str) -> list[dict] | None:
    """Fallback JSON extraction using regex for unwrapped JSON objects.

    This handles cases where the model outputs valid JSON without <json> tags.
    Searches for JSON objects with a "response" key.
    """
    results = []
    # Pattern to match JSON objects (handles nested braces)
    pattern = r'\{[^{}]*"response"[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.finditer(pattern, text, re.DOTALL)

    for match in matches:
        try:
            obj = json.loads(match.group())
            if "response" in obj:
                results.append(obj)
        except json.JSONDecodeError:
            continue

    # If regex fails, try to find any JSON-like structure
    if not results:
        try:
            # Try to parse the entire text as JSON
            obj = json.loads(text.strip())
            if isinstance(obj, dict) and "response" in obj:
                results.append(obj)
        except json.JSONDecodeError:
            pass

    return results or None


def _sanitize_prediction(prediction: Any) -> str:
    """Sanitize and validate the prediction value.
    
    Ensures the prediction is a valid string representation.
    Handles None, numbers, booleans, and nested structures.
    """
    if prediction is None:
        return "None"
    
    if isinstance(prediction, (str, int, float, bool)):
        return str(prediction)
    
    if isinstance(prediction, (list, dict)):
        # Convert complex structures to JSON string
        try:
            return json.dumps(prediction)
        except (TypeError, ValueError):
            return str(prediction)
    
    return str(prediction)


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
        instruction = f"""You are an expert grading agent for mathematical problems.

Your task is to evaluate a student's answer based on the provided problem, solution, and grading guidelines.

Task input:
```
{inputs}
```

Respond in JSON format with the following schema:
<json>
{{
    "response": "Your evaluation/grade here"
}}
</json>

Important: The "response" field should contain your final evaluation or grade for the student's answer."""

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
        prediction = None
        extraction_method = "none"
        
        try:
            # Get the last assistant message
            last_msg = None
            for msg in reversed(msg_history):
                if msg.get("role") == "assistant":
                    last_msg = msg.get("text", "")
                    break
            
            if not last_msg:
                self.log_fn("No assistant message found in history")
                prediction = "Error: No response from model"
            else:
                # Try primary extraction first
                extracted = _extract_jsons(last_msg)
                if extracted and len(extracted) > 0:
                    last_extracted = extracted[-1]
                    if isinstance(last_extracted, dict) and "response" in last_extracted:
                        prediction = last_extracted["response"]
                        extraction_method = "primary"
                    else:
                        # Try to find any dict with response key
                        for item in extracted:
                            if isinstance(item, dict) and "response" in item:
                                prediction = item["response"]
                                extraction_method = "primary"
                                break
                
                # If primary failed, try fallback
                if prediction is None:
                    extracted = _extract_json_fallback(last_msg)
                    if extracted and len(extracted) > 0:
                        last_extracted = extracted[-1]
                        if isinstance(last_extracted, dict) and "response" in last_extracted:
                            prediction = last_extracted["response"]
                            extraction_method = "fallback"
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            # Try fallback on exception
            try:
                last_msg = None
                for msg in reversed(msg_history):
                    if msg.get("role") == "assistant":
                        last_msg = msg.get("text", "")
                        break
                
                if last_msg:
                    extracted = _extract_json_fallback(last_msg)
                    if extracted and len(extracted) > 0:
                        last_extracted = extracted[-1]
                        if isinstance(last_extracted, dict) and "response" in last_extracted:
                            prediction = last_extracted["response"]
                            extraction_method = "fallback_recovery"
            except Exception as fallback_e:
                self.log_fn(f"Fallback extraction also failed: {fallback_e}")

        # Sanitize and return the prediction
        final_prediction = _sanitize_prediction(prediction)
        self.log_fn(f"Extraction method used: {extraction_method}, prediction type: {type(prediction).__name__}")
        
        return final_prediction, msg_history
