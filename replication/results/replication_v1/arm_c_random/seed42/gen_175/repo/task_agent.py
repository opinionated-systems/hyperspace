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


def sanitize_json_string(json_str: str) -> str:
    """Sanitize a JSON string by fixing common formatting issues.
    
    Args:
        json_str: Raw JSON string that may have formatting issues
        
    Returns:
        Cleaned JSON string ready for parsing
    """
    # Remove trailing commas before closing braces/brackets
    cleaned = re.sub(r',(\s*[}\]])', r'\1', json_str)
    # Remove any control characters
    cleaned = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', cleaned)
    # Normalize whitespace
    cleaned = cleaned.strip()
    return cleaned


def safe_json_loads(json_str: str, max_retries: int = 3) -> dict[str, Any] | None:
    """Safely parse a JSON string with multiple fallback strategies.
    
    Args:
        json_str: JSON string to parse
        max_retries: Number of cleanup attempts before giving up
        
    Returns:
        Parsed JSON dict or None if parsing fails
    """
    if not json_str or not isinstance(json_str, str):
        return None
    
    for attempt in range(max_retries):
        try:
            if attempt == 0:
                return json.loads(json_str)
            else:
                # Apply increasingly aggressive cleanup
                cleaned = sanitize_json_string(json_str)
                if attempt > 1:
                    # Try to extract just the JSON object if wrapped in other text
                    match = re.search(r'\{[^{}]*\}', cleaned)
                    if match:
                        cleaned = match.group(0)
                return json.loads(cleaned)
        except json.JSONDecodeError:
            if attempt == max_retries - 1:
                logger.warning(f"Failed to parse JSON after {max_retries} attempts")
                return None
            continue
    return None


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    
    Also attempts to extract JSON from markdown code blocks as fallback.
    Uses safe_json_loads for robust parsing with automatic cleanup.
    """
    results = []
    search_from = 0
    
    # First try to find <json>...</json> blocks
    while True:
        start = text.find("<json>", search_from)
        if start == -1:
            break
        end = text.find("</json>", start)
        if end == -1:
            break
        inner = text[start + 6:end].strip()
        search_from = end + 7
        
        parsed = safe_json_loads(inner)
        if parsed is not None:
            results.append(parsed)
    
    # Fallback: try to find JSON in markdown code blocks
    if not results:
        json_code_blocks = re.findall(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
        for block in json_code_blocks:
            parsed = safe_json_loads(block.strip())
            if parsed is not None:
                results.append(parsed)
    
    return results or None


class TaskAgent:
    """Task agent that solves IMO grading problems."""

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
        # Validate inputs
        if not isinstance(inputs, dict):
            raise ValueError(f"Expected dict inputs, got {type(inputs).__name__}")
        
        # Log input keys for debugging
        self.log_fn(f"TaskAgent.forward called with input keys: {list(inputs.keys())}")
        
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

Be thorough and accurate in your evaluation."""

        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
        except Exception as e:
            self.log_fn(f"LLM call failed: {e}")
            return "Error: LLM call failed", [{"role": "error", "text": str(e)}]

        # Extract prediction from JSON
        prediction = "None"
        try:
            if msg_history and len(msg_history) > 0:
                last_msg = msg_history[-1]
                text_content = last_msg.get("text", "")
                extracted = _extract_jsons(text_content)
                if extracted:
                    last_extracted = extracted[-1]
                    if "response" in last_extracted:
                        prediction = last_extracted["response"]
                        self.log_fn(f"Successfully extracted prediction: {repr(str(prediction)[:100])}")
                    else:
                        self.log_fn(f"JSON extracted but 'response' key missing. Keys: {list(last_extracted.keys())}")
                else:
                    self.log_fn("No JSON blocks found in response")
            else:
                self.log_fn("Message history is empty")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
