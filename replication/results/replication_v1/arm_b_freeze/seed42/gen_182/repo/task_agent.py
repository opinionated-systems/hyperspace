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
    
    Args:
        text: The text to search for JSON blocks.
        
    Returns:
        A list of parsed JSON objects, or None if no valid JSON found.
    """
    results = []
    search_from = 0
    while True:
        start = text.find("<json>", search_from)
        if start == -1:
            break
        end = text.find("</json>", start)
        if end == -1:
            logger.warning("Found opening <json> tag but no closing </json> tag")
            break
        inner = text[start + 6:end].strip()
        search_from = end + 7
        try:
            parsed = json.loads(inner)
            if isinstance(parsed, dict):
                results.append(parsed)
            else:
                logger.warning(f"Parsed JSON is not a dict: {type(parsed)}")
        except json.JSONDecodeError as e:
            logger.debug(f"Failed to parse JSON block: {e}")
            continue
    return results or None


def _extract_json_fuzzy(text: str) -> list[dict] | None:
    """Fallback JSON extraction that looks for JSON-like structures.
    
    This is used when the standard <json> tags are not found.
    It attempts to find JSON objects by looking for curly braces.
    
    Args:
        text: The text to search for JSON objects.
        
    Returns:
        A list of parsed JSON objects, or None if no valid JSON found.
    """
    results = []
    # Look for JSON objects between curly braces
    # This is a simple heuristic that may not work for all cases
    pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.finditer(pattern, text, re.DOTALL)
    
    for match in matches:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, dict):
                results.append(parsed)
        except json.JSONDecodeError:
            continue
    
    return results or None


class TaskAgent:
    """Task agent that solves IMO grading problems.
    
    This agent uses an LLM to process task inputs and extract structured
    responses from JSON-formatted outputs.
    """

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        """Initialize the TaskAgent.
        
        Args:
            model: The LLM model to use for inference.
            log_file: Path to a log file (currently unused, for interface compatibility).
        """
        self.model = model
        self.log_fn = logger.info

    def _build_prompt(self, inputs: dict) -> str:
        """Build the instruction prompt for the LLM.
        
        Args:
            inputs: Dictionary containing task inputs.
            
        Returns:
            The formatted instruction string.
        """
        return f"""You are an agent.

Task input:
```
{json.dumps(inputs, indent=2, default=str)}
```

Respond in JSON format with the following schema:
<json>
{{
    "response": "your answer here"
}}
</json>"""

    def _extract_prediction(self, text: str) -> str:
        """Extract the prediction from LLM response text.
        
        First tries to extract from <json> tags, then falls back to
        fuzzy JSON extraction if that fails.
        
        Args:
            text: The response text from the LLM.
            
        Returns:
            The extracted prediction string, or "None" if extraction fails.
        """
        # Try standard JSON extraction first
        extracted = _extract_jsons(text)
        
        # Fallback to fuzzy extraction if standard fails
        if extracted is None:
            self.log_fn("Standard JSON extraction failed, trying fuzzy extraction")
            extracted = _extract_json_fuzzy(text)
        
        if extracted and "response" in extracted[-1]:
            response = extracted[-1]["response"]
            self.log_fn(f"Successfully extracted prediction: {response}")
            return str(response)
        
        self.log_fn("Could not extract prediction from response")
        return "None"

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        instruction = self._build_prompt(inputs)
        
        self.log_fn(f"Processing task with model: {self.model}")

        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
            
            self.log_fn(f"LLM call completed. Usage: {info.get('usage', {})}")
        except Exception as e:
            self.log_fn(f"LLM call failed: {e}")
            raise

        # Extract prediction from JSON
        prediction = "None"
        try:
            if msg_history:
                prediction = self._extract_prediction(msg_history[-1]["text"])
            else:
                self.log_fn("No message history available")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
