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


def _sanitize_json_string(text: str) -> str:
    """Sanitize a JSON string by fixing common formatting issues.
    
    Args:
        text: Raw text that may contain JSON
        
    Returns:
        Sanitized text with common JSON issues fixed
    """
    # Remove trailing commas before closing braces/brackets
    text = re.sub(r',(\s*[}\]])', r'\1', text)
    # Fix single quotes to double quotes (common LLM mistake)
    text = re.sub(r"(?<!\\)'", '"', text)
    # Remove control characters
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    return text


def _safe_json_loads(text: str, max_retries: int = 3) -> dict | None:
    """Safely parse JSON with multiple fallback strategies.
    
    Args:
        text: Text to parse as JSON
        max_retries: Number of fallback strategies to try
        
    Returns:
        Parsed JSON dict or None if all strategies fail
    """
    strategies = [
        lambda t: json.loads(t),  # Try raw first
        lambda t: json.loads(_sanitize_json_string(t)),  # Try sanitized
        lambda t: json.loads(t.replace('\n', ' ').replace('\t', ' ')),  # Try flattened
        lambda t: json.loads(re.sub(r'\s+', ' ', t)),  # Try fully normalized
    ]
    
    for i, strategy in enumerate(strategies[:max_retries]):
        try:
            return strategy(text)
        except json.JSONDecodeError:
            logger.debug(f"JSON parse strategy {i+1} failed")
            continue
    
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

        # Try to parse the JSON using safe loader
        parsed = _safe_json_loads(inner, max_retries=4)
        if parsed is not None:
            results.append(parsed)
            continue
            
        # Fallback: try extracting just the first valid JSON object by brace matching
        try:
            brace_count = 0
            in_string = False
            escape_next = False
            json_start = -1

            for i, char in enumerate(inner):
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
                            parsed = _safe_json_loads(inner[json_start:i+1], max_retries=3)
                            if parsed is not None:
                                results.append(parsed)
                                break
        except Exception:
            pass
                
    return results or None


class TaskAgent:
    """Task agent that solves IMO grading problems."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.stats = {"calls": 0, "successes": 0, "json_extracted": 0, "fallbacks": 0}

    def _get_last_message_text(self, msg_history: list[dict]) -> str:
        """Safely extract text from the last message in history.
        
        Args:
            msg_history: List of message dicts
            
        Returns:
            Text content or empty string
        """
        if not msg_history:
            return ""
        last_msg = msg_history[-1]
        if isinstance(last_msg, dict):
            return last_msg.get("text", "") or last_msg.get("content", "")
        return str(last_msg)

    def _extract_prediction(self, msg_history: list[dict]) -> tuple[str, bool]:
        """Extract prediction from message history.
        
        Args:
            msg_history: List of message dicts from LLM
            
        Returns:
            Tuple of (prediction_text, was_json_extracted)
        """
        text = self._get_last_message_text(msg_history)
        if not text:
            return "None", False
            
        extracted = _extract_jsons(text)
        if extracted:
            last_json = extracted[-1]
            if isinstance(last_json, dict) and "response" in last_json:
                return str(last_json["response"]), True
            self.log_fn(f"TaskAgent: JSON missing 'response' key: {list(last_json.keys()) if isinstance(last_json, dict) else 'N/A'}")
        
        # Fallback: truncate raw text
        return (text[:1000] + "...") if len(text) > 1000 else text, False

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        self.stats["calls"] += 1
        
        # Format inputs nicely for the LLM
        formatted_inputs = json.dumps(inputs, indent=2, default=str)

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
            self.stats["successes"] += 1
            self.log_fn("TaskAgent: Received response from LLM")
        except Exception as e:
            self.log_fn(f"TaskAgent: LLM call failed: {e}")
            return f"Error: LLM call failed - {e}", []

        # Extract prediction from JSON
        prediction, was_json = self._extract_prediction(msg_history)
        if was_json:
            self.stats["json_extracted"] += 1
            self.log_fn("TaskAgent: Successfully extracted prediction from JSON")
        else:
            self.stats["fallbacks"] += 1
            self.log_fn("TaskAgent: Using fallback text extraction (no valid JSON found)")

        return prediction, msg_history

    def get_stats(self) -> dict[str, int]:
        """Return agent statistics.
        
        Returns:
            Dict with call statistics
        """
        return dict(self.stats)
