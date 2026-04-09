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
    
    # Fallback: Extract from markdown code blocks (```json ... ```)
    if not results:
        pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                results.append(json.loads(match.strip()))
            except json.JSONDecodeError as e:
                logger.debug(f"Failed to parse JSON from markdown block: {e}")
                continue
    
    # Last resort: Try to find any JSON object in the text
    if not results:
        try:
            # Look for content between first { and last }
            start_idx = text.find('{')
            end_idx = text.rfind('}')
            if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
                potential_json = text[start_idx:end_idx + 1]
                results.append(json.loads(potential_json))
        except (json.JSONDecodeError, ValueError) as e:
            logger.debug(f"Failed to parse JSON from raw text: {e}")
    
    return results or None


class TaskAgent:
    """Task agent that solves IMO grading problems.
    
    This agent processes task inputs and generates responses using an LLM.
    It includes robust JSON extraction with multiple fallback strategies.
    """

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.call_count = 0
        self.log_file = log_file

    def reset_call_count(self) -> None:
        """Reset the call counter. Useful for testing."""
        self.call_count = 0
        self.log_fn("TaskAgent call counter reset")

    def get_stats(self) -> dict:
        """Get agent statistics."""
        return {
            "call_count": self.call_count,
            "model": self.model,
        }

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        self.call_count += 1
        self.log_fn(f"TaskAgent call #{self.call_count}: Processing new task")
        
        # Validate inputs
        if not isinstance(inputs, dict):
            error_msg = f"Invalid inputs type: expected dict, got {type(inputs).__name__}"
            self.log_fn(f"ERROR: {error_msg}")
            return f"Error: {error_msg}", []
        
        instruction = f"""You are an expert agent tasked with evaluating student answers.

Task input:
```
{inputs}
```

Analyze the input carefully and provide your response in the following JSON format:
<json>
{{
    "response": "Your detailed evaluation or answer here"
}}
</json>

Ensure your response is valid JSON and properly enclosed in <json> tags."""

        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
        except Exception as e:
            error_msg = f"LLM call failed: {e}"
            self.log_fn(f"ERROR: {error_msg}")
            return f"Error: {error_msg}", []

        # Extract prediction from JSON
        prediction = "None"
        try:
            if not msg_history:
                self.log_fn("WARNING: Empty message history from LLM")
                return "Error: No response from LLM", msg_history
            
            last_message = msg_history[-1]
            if "text" not in last_message:
                self.log_fn(f"WARNING: Last message missing 'text' key: {last_message.keys()}")
                return "Error: Invalid message format", msg_history
            
            extracted = _extract_jsons(last_message["text"])
            if extracted:
                if "response" in extracted[-1]:
                    prediction = extracted[-1]["response"]
                    self.log_fn(f"Successfully extracted prediction: {str(prediction)[:100]}...")
                else:
                    self.log_fn(f"WARNING: Extracted JSON missing 'response' key: {list(extracted[-1].keys())}")
                    # Try to use the first available key as fallback
                    if extracted[-1]:
                        first_key = list(extracted[-1].keys())[0]
                        prediction = extracted[-1][first_key]
                        self.log_fn(f"Using fallback key '{first_key}' for prediction")
            else:
                self.log_fn(f"WARNING: No JSON extracted from response. Raw response: {last_message['text'][:200]}...")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            import traceback
            self.log_fn(f"Traceback: {traceback.format_exc()}")

        return str(prediction), msg_history
