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
        text: The input text containing <json>...</json> blocks.
        
    Returns:
        A list of parsed JSON objects, or None if no valid JSON found.
    """
    results = []
    search_from = 0
    json_open_tag = "<json>"
    json_close_tag = "</json>"
    
    while True:
        start = text.find(json_open_tag, search_from)
        if start == -1:
            break
        end = text.find(json_close_tag, start)
        if end == -1:
            logger.warning("Found opening <json> tag but no closing </json> tag")
            break
        # Extract content between tags (accounting for tag length)
        inner = text[start + len(json_open_tag):end].strip()
        search_from = end + len(json_close_tag)
        
        if not inner:
            logger.debug("Empty JSON block found at position %d, skipping", start)
            continue
        try:
            parsed = json.loads(inner)
            if isinstance(parsed, dict):
                results.append(parsed)
            else:
                logger.debug("Parsed JSON is not a dict, skipping: %s", type(parsed))
        except json.JSONDecodeError as e:
            logger.debug("Failed to parse JSON block at position %d: %s", start, e)
            continue
    
    if not results:
        logger.debug("No valid JSON objects found in text")
    else:
        logger.debug("Successfully extracted %d JSON object(s)", len(results))
    
    return results or None


class TaskAgent:
    """Task agent that solves IMO grading problems.
    
    This agent takes a problem description and student answer as input,
    queries an LLM for evaluation, and returns a structured prediction.
    """

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        """Initialize the TaskAgent.
        
        Args:
            model: The LLM model to use for evaluation.
            log_file: Optional path to a log file (currently unused).
        """
        self.model = model
        self.log_fn = logger.info

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with keys like domain, problem, solution, 
                   grading_guidelines, student_answer

        Returns:
            A tuple of (prediction, msg_history) where prediction is the
            extracted response string and msg_history is the conversation
            history with the LLM.
        """
        instruction = f"""You are an expert grading agent evaluating student answers.

Please analyze the following task input carefully and provide your evaluation.

Task input:
```
{inputs}
```

Respond in JSON format with the following schema:
<json>
{{
    "response": "Your evaluation/grade here"
}}
</json>"""

        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
        except Exception as e:
            logger.error("LLM call failed: %s", e)
            return "Error: LLM call failed", []

        # Extract prediction from JSON
        prediction = "None"
        try:
            if msg_history and len(msg_history) > 0:
                last_message = msg_history[-1]
                text_content = last_message.get("text", "")
                extracted = _extract_jsons(text_content)
                if extracted and "response" in extracted[-1]:
                    prediction = extracted[-1]["response"]
                    logger.debug("Successfully extracted prediction from JSON response")
                else:
                    logger.warning("No 'response' key found in extracted JSON")
            else:
                logger.warning("Empty message history from LLM")
        except (KeyError, IndexError, TypeError) as e:
            self.log_fn("Error extracting prediction: %s", e)
        except Exception as e:
            self.log_fn("Unexpected error extracting prediction: %s", e)

        return str(prediction), msg_history
