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
    
    Args:
        text: The text containing <json>...</json> blocks.
        
    Returns:
        A list of parsed JSON objects, or None if no valid JSON found.
    """
    results = []
    search_from = 0
    json_count = 0
    
    while True:
        start = text.find("<json>", search_from)
        if start == -1:
            break
        end = text.find("</json>", start)
        if end == -1:
            logger.warning(f"Found unclosed <json> tag at position {start}")
            break
        
        inner = text[start + 6:end].strip()
        json_count += 1
        search_from = end + 7
        
        try:
            parsed = json.loads(inner)
            results.append(parsed)
            logger.debug(f"Successfully parsed JSON block #{json_count}")
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON block #{json_count}: {e}")
            continue
    
    if json_count > 0 and not results:
        logger.error(f"Found {json_count} JSON blocks but none were valid")
    
    return results or None


class TaskAgent:
    """Task agent that solves IMO grading problems.
    
    This agent uses an LLM to process grading tasks and extracts
    structured responses from JSON-formatted outputs.
    """

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        """Initialize the TaskAgent.
        
        Args:
            model: The LLM model to use for inference.
            log_file: Optional path to a log file (currently unused).
        """
        self.model = model
        self.log_fn = logger.info
        logger.info(f"TaskAgent initialized with model: {model}")

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            A tuple of (prediction, msg_history) where prediction is the
            extracted response string and msg_history is the conversation history.
        """
        if not inputs:
            logger.warning("Empty inputs provided to TaskAgent.forward()")
            return "None", []
        
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

        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return "None", []

        # Extract prediction from JSON
        prediction = "None"
        try:
            if msg_history and len(msg_history) > 0:
                last_message = msg_history[-1].get("text", "")
                extracted = _extract_jsons(last_message)
                if extracted and "response" in extracted[-1]:
                    prediction = extracted[-1]["response"]
                    logger.info(f"Successfully extracted prediction: {prediction}")
                else:
                    logger.warning("No valid 'response' field found in extracted JSON")
            else:
                logger.warning("Empty message history from LLM")
        except Exception as e:
            logger.error(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
