"""
Task agent: solves a given task with a single LLM call.

Reimplemented from facebookresearch/HyperAgents task_agent.py.
Same interface, same JSON output format, same extraction logic.

This is the INITIAL task agent. The meta agent modifies this file
during self-improvement. The evaluation harness loads whatever
task_agent.py exists at the agent's repo path.

Enhancements:
- Improved error handling with detailed logging
- Better JSON extraction with validation
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
    
    Enhanced with detailed error logging for debugging.
    """
    results = []
    search_from = 0
    block_count = 0
    while True:
        start = text.find("<json>", search_from)
        if start == -1:
            break
        end = text.find("</json>", start)
        if end == -1:
            logger.warning(f"Unclosed <json> tag found at position {start}")
            break
        inner = text[start + 6:end].strip()
        block_count += 1
        search_from = end + 7
        try:
            parsed = json.loads(inner)
            if isinstance(parsed, dict):
                results.append(parsed)
            else:
                logger.warning(f"JSON block {block_count} is not a dict: {type(parsed)}")
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON block {block_count}: {e}")
            continue
    if block_count > 0 and not results:
        logger.warning(f"Found {block_count} JSON blocks but none parsed successfully")
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
        instruction = f"""You are an agent.

Task input:
```
{inputs}
```

Respond in JSON format with the following schema:
<json>
{{
    "response": ...
}}
</json>"""

        self.log_fn(f"TaskAgent.forward called with model={self.model}")
        
        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        self.log_fn(f"LLM response received, msg_history length={len(msg_history)}")

        # Extract prediction from JSON
        prediction = "None"
        try:
            if not msg_history:
                self.log_fn("Warning: Empty message history")
                return str(prediction), msg_history
                
            last_msg = msg_history[-1]
            if "text" not in last_msg:
                self.log_fn(f"Warning: Last message has no 'text' key: {last_msg.keys()}")
                return str(prediction), msg_history
                
            extracted = _extract_jsons(last_msg["text"])
            if extracted:
                if "response" in extracted[-1]:
                    prediction = extracted[-1]["response"]
                    self.log_fn(f"Successfully extracted prediction: {prediction}")
                else:
                    self.log_fn(f"Warning: Extracted JSON missing 'response' key. Keys: {extracted[-1].keys()}")
            else:
                self.log_fn("Warning: No JSON blocks found in response")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {type(e).__name__}: {e}")

        return str(prediction), msg_history
