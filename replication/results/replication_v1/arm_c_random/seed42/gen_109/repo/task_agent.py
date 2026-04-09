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
    Includes additional heuristics for common JSON formatting issues.
    """
    results = []
    search_from = 0
    
    # Primary: Extract from <json> tags
    while True:
        start = text.find("<json>", search_from)
        if start == -1:
            break
        end = text.find("</json>", start)
        if end == -1:
            logger.warning("Found opening <json> tag without closing </json> tag")
            break
        inner = text[start + 6:end].strip()
        search_from = end + 7
        try:
            results.append(json.loads(inner))
        except json.JSONDecodeError as e:
            # Try to fix common JSON issues
            fixed = _attempt_json_fix(inner)
            if fixed is not None:
                results.append(fixed)
            else:
                logger.debug(f"Failed to parse JSON from <json> block: {e}")
            continue
    
    # Fallback 1: Extract from markdown code blocks
    if not results:
        pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                results.append(json.loads(match.strip()))
            except json.JSONDecodeError:
                fixed = _attempt_json_fix(match.strip())
                if fixed is not None:
                    results.append(fixed)
                continue
    
    # Fallback 2: Try to find JSON-like structures in the text
    if not results:
        # Look for content between curly braces
        pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                results.append(json.loads(match.strip()))
            except json.JSONDecodeError:
                fixed = _attempt_json_fix(match.strip())
                if fixed is not None:
                    results.append(fixed)
                continue
    
    return results or None


def _attempt_json_fix(text: str) -> dict | None:
    """Attempt to fix common JSON formatting issues.
    
    Fixes:
    - Trailing commas in objects/arrays
    - Single quotes instead of double quotes
    - Unquoted keys
    """
    try:
        # Remove trailing commas before closing braces/brackets
        fixed = re.sub(r',(\s*[}\]])', r'\1', text)
        # Replace single quotes with double quotes (simple cases)
        fixed = re.sub(r"'([^']*?)':", r'"\1":', fixed)
        fixed = re.sub(r":\s*'([^']*?)'", r': "\1"', fixed)
        return json.loads(fixed)
    except json.JSONDecodeError:
        return None


class TaskAgent:
    """Task agent that solves IMO grading problems with robust JSON extraction."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.log_file = log_file
        self._call_count = 0

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        self._call_count += 1
        self.log_fn(f"TaskAgent.forward call #{self._call_count} with model={self.model}")
        
        # Validate inputs
        if not isinstance(inputs, dict):
            self.log_fn(f"Error: inputs must be a dict, got {type(inputs)}")
            return "Error: Invalid inputs", []
        
        # Build instruction with structured formatting
        instruction = self._build_instruction(inputs)

        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
            self.log_fn(f"LLM call successful, response length: {len(response) if response else 0}")
        except Exception as e:
            self.log_fn(f"Error getting response from LLM: {e}")
            return "Error: LLM call failed", []

        # Extract prediction from JSON
        prediction = self._extract_prediction(msg_history)
        
        return str(prediction), msg_history
    
    def _build_instruction(self, inputs: dict) -> str:
        """Build the instruction prompt for the LLM."""
        # Format inputs nicely for better LLM comprehension
        formatted_inputs = json.dumps(inputs, indent=2, ensure_ascii=False)
        
        return f"""You are an expert grading agent. Your task is to evaluate a student's answer based on the provided problem, solution, and grading guidelines.

Task input:
```json
{formatted_inputs}
```

Analyze the student's answer carefully and provide your evaluation.

Respond in JSON format with the following schema:
<json>
{{
    "response": "Your evaluation/grade here"
}}
</json>"""
    
    def _extract_prediction(self, msg_history: list[dict]) -> str:
        """Extract the prediction from the message history."""
        if not msg_history:
            self.log_fn("Empty message history")
            return "None"
        
        last_message = msg_history[-1]
        if not isinstance(last_message, dict):
            self.log_fn(f"Unexpected message format: {type(last_message)}")
            return "None"
        
        text = last_message.get("text", "")
        if not text:
            self.log_fn("Last message has no text content")
            return "None"
        
        # Try to extract JSON
        extracted = _extract_jsons(text)
        if not extracted:
            self.log_fn("No JSON found in response")
            # Fallback: return first 500 chars of response as raw prediction
            return text[:500].strip()
        
        # Get the last extracted JSON object
        last_extracted = extracted[-1]
        if not isinstance(last_extracted, dict):
            self.log_fn(f"Extracted JSON is not a dict: {type(last_extracted)}")
            return str(last_extracted)[:500]
        
        if "response" not in last_extracted:
            self.log_fn(f"Extracted JSON missing 'response' key: {list(last_extracted.keys())}")
            # Return the first value as fallback
            if last_extracted:
                return str(list(last_extracted.values())[0])[:500]
            return "None"
        
        prediction = last_extracted["response"]
        self.log_fn(f"Successfully extracted prediction: {str(prediction)[:200]}")
        return prediction
