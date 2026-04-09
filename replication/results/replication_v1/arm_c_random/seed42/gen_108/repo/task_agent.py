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
    Also attempts to extract JSON from markdown code blocks as fallback.
    """
    results = []
    search_from = 0
    
    # Primary: Extract from <json>...</json> tags
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
            logger.debug(f"Failed to parse JSON from <json> block: {e}")
            continue
    
    # Fallback: Extract from markdown code blocks ```json ... ```
    if not results:
        pattern = r'```(?:json)?\s*\n(.*?)\n```'
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                results.append(json.loads(match.strip()))
            except json.JSONDecodeError as e:
                logger.debug(f"Failed to parse JSON from markdown block: {e}")
                continue
    
    # Last resort: Try to find any JSON object in the text
    if not results:
        # Look for content between first { and last }
        try:
            start_idx = text.find('{')
            end_idx = text.rfind('}')
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                potential_json = text[start_idx:end_idx + 1]
                results.append(json.loads(potential_json))
        except json.JSONDecodeError:
            pass
    
    return results or None


def _safe_extract_prediction(extracted: list[dict] | None) -> Any:
    """Safely extract prediction from extracted JSON objects.
    
    Tries multiple common keys for the response field.
    """
    if not extracted:
        return None
    
    # Try the last extracted object first (most recent)
    for obj in reversed(extracted):
        if not isinstance(obj, dict):
            continue
        # Try common response keys
        for key in ["response", "answer", "result", "output", "prediction"]:
            if key in obj:
                return obj[key]
    
    return None


class TaskAgent:
    """Task agent that solves IMO grading problems with robust JSON extraction."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.log_file = log_file

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Build a more structured prompt with clearer instructions
        instruction = f"""You are an expert grading agent. Your task is to evaluate a student's answer.

Task input:
```json
{json.dumps(inputs, indent=2, default=str)}
```

Analyze the problem, solution, grading guidelines, and student answer carefully.

Respond in JSON format with the following schema:
<json>
{{
    "response": "Your evaluation result here"
}}
</json>

Ensure your response is valid JSON."""

        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
        except Exception as e:
            self.log_fn(f"LLM call failed: {e}")
            return "Error: LLM call failed", [{"role": "system", "text": f"Error: {e}"}]

        # Extract prediction from JSON with improved error handling
        prediction = "None"
        try:
            if msg_history and len(msg_history) > 0:
                last_message = msg_history[-1]
                text_content = last_message.get("text", "")
                
                extracted = _extract_jsons(text_content)
                pred_value = _safe_extract_prediction(extracted)
                
                if pred_value is not None:
                    prediction = pred_value
                    self.log_fn(f"Successfully extracted prediction: {repr(str(prediction)[:100])}")
                else:
                    self.log_fn(f"No valid prediction found in response. Raw text: {repr(text_content[:200])}")
            else:
                self.log_fn("Empty message history received")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {type(e).__name__}: {e}")
            # Log the raw response for debugging
            if msg_history:
                self.log_fn(f"Raw last message: {repr(msg_history[-1].get('text', 'N/A')[:200])}")

        return str(prediction), msg_history
