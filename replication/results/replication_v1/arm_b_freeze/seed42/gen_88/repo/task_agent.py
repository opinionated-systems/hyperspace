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


def _safe_json_loads(text: str) -> dict | None:
    """Safely parse JSON with multiple fallback strategies.
    
    Args:
        text: Raw text that may contain JSON
        
    Returns:
        Parsed dict or None if parsing fails
    """
    if not text or not isinstance(text, str):
        return None
    
    text = text.strip()
    if not text:
        return None
    
    # Strategy 1: Direct parsing
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Strategy 2: Remove trailing commas before } or ]
    try:
        fixed = re.sub(r',(\s*[}\]])', r'\1', text)
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    
    # Strategy 3: Extract first JSON object using brace counting
    try:
        brace_count = 0
        in_string = False
        escape_next = False
        json_start = -1
        
        for i, char in enumerate(text):
            if escape_next:
                escape_next = False
                continue
            if char == '\\':
                escape_next = True
                continue
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
            if not in_string:
                if char == '{':
                    if brace_count == 0:
                        json_start = i
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0 and json_start >= 0:
                        try:
                            return json.loads(text[json_start:i+1])
                        except json.JSONDecodeError:
                            pass
    except Exception:
        pass
    
    return None


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    Also handles nested JSON objects and common formatting issues.
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

        # Use the robust JSON parser
        parsed = _safe_json_loads(inner)
        if parsed is not None:
            results.append(parsed)
        else:
            logger.debug(f"Failed to parse JSON in <json> block: {inner[:200]}...")
            
    return results or None


class TaskAgent:
    """Task agent that solves IMO grading problems."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def forward(self, inputs: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Validate required fields
        required_fields = ["domain", "problem", "solution", "grading_guidelines", "student_answer"]
        missing = [f for f in required_fields if f not in inputs]
        if missing:
            error_msg = f"Error: Missing required fields: {missing}"
            self.log_fn(f"TaskAgent: {error_msg}")
            return error_msg, []

        # Format inputs nicely for the LLM
        try:
            formatted_inputs = json.dumps(inputs, indent=2, default=str, ensure_ascii=False)
        except (TypeError, ValueError) as e:
            error_msg = f"Error: Failed to serialize inputs - {e}"
            self.log_fn(f"TaskAgent: {error_msg}")
            return error_msg, []

        instruction = f"""You are an expert grading agent. Your task is to evaluate a student's answer to a problem.

Task input:
```json
{formatted_inputs}
```

Instructions:
1. Carefully read the problem, solution, and grading guidelines
2. Evaluate the student's answer against the correct solution
3. Provide your assessment in the exact JSON format below

Respond in JSON format with the following schema:
<json>
{{
    "response": "Your detailed evaluation here. Include: (1) what the student got right/wrong, (2) score if applicable, (3) specific feedback"
}}
</json>

Important: Your response MUST be valid JSON inside <json> tags."""

        self.log_fn("TaskAgent: Sending request to LLM...")

        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
            self.log_fn("TaskAgent: Received response from LLM")
        except Exception as e:
            self.log_fn(f"TaskAgent: LLM call failed: {e}")
            return f"Error: LLM call failed - {e}", []

        # Extract prediction from JSON
        prediction = self._extract_prediction(msg_history)

        return str(prediction), msg_history

    def _extract_prediction(self, msg_history: list[dict[str, Any]]) -> str:
        """Extract prediction from message history.
        
        Args:
            msg_history: List of message dicts from LLM conversation
            
        Returns:
            Extracted prediction string or fallback text
        """
        if not msg_history:
            self.log_fn("TaskAgent: Empty message history")
            return "None"

        try:
            last_msg = msg_history[-1]
            text = last_msg.get("text", "") if isinstance(last_msg, dict) else str(last_msg)
            
            if not text:
                self.log_fn("TaskAgent: Empty text in last message")
                return "None"

            extracted = _extract_jsons(text)
            if extracted:
                last_json = extracted[-1]
                if isinstance(last_json, dict) and "response" in last_json:
                    self.log_fn("TaskAgent: Successfully extracted prediction")
                    return str(last_json["response"])
                else:
                    available_keys = list(last_json.keys()) if isinstance(last_json, dict) else 'N/A'
                    self.log_fn(f"TaskAgent: JSON missing 'response' key. Available keys: {available_keys}")
                    # Return the whole JSON as string if response key missing
                    return json.dumps(last_json, default=str)
            else:
                self.log_fn("TaskAgent: No JSON found in response, using raw text fallback")
                # Fallback: use the raw text if no JSON found
                return text[:2000] if len(text) > 2000 else text
        except Exception as e:
            self.log_fn(f"TaskAgent: Error extracting prediction: {e}")
            # Try to get any text from the last message as fallback
            try:
                if msg_history:
                    last_msg = msg_history[-1]
                    text = last_msg.get("text", "") if isinstance(last_msg, dict) else str(last_msg)
                    return text[:2000] if len(text) > 2000 else text
            except Exception:
                pass
            return "None"
