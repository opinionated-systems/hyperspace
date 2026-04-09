"""
Task agent: solves a given task with a single LLM call.

Reimplemented from facebookresearch/HyperAgents task_agent.py.
Same interface, same JSON output format, same extraction logic.

This is the INITIAL task agent. The meta agent modifies this file
during self-improvement. The evaluation harness loads whatever
task_agent.py exists at the agent's repo path.

Enhanced with better JSON extraction and error handling.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from agent.llm_client import get_response_from_llm, EVAL_MODEL
from agent.utils import retry, truncate_string

logger = logging.getLogger(__name__)


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
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
        try:
            results.append(json.loads(inner))
        except json.JSONDecodeError:
            continue
    return results or None


def _extract_json_flexible(text: str) -> dict | None:
    """Extract JSON with more flexible parsing.
    
    Tries multiple strategies in order of preference:
    1. Look for <json>...</json> blocks (primary format)
    2. Look for ```json code blocks (markdown format)
    3. Look for JSON objects directly in the text (fallback)
    
    Returns the last valid JSON object found, or None if no valid JSON is found.
    """
    # Strategy 1: <json> tags
    results = _extract_jsons(text)
    if results:
        return results[-1]
    
    # Strategy 2: ```json code blocks
    json_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    for match in re.finditer(json_block_pattern, text, re.DOTALL):
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            continue
    
    # Strategy 3: Look for JSON objects directly
    # Match content between { and } (balanced)
    brace_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    for match in re.finditer(brace_pattern, text, re.DOTALL):
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
    
    return None


class TaskAgent:
    """Task agent that solves IMO grading problems."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self._call_count = 0
        self._success_count = 0

    def _build_prompt(self, inputs: dict) -> str:
        """Build the prompt for the task."""
        # Truncate long inputs
        truncated_inputs = {}
        for key, value in inputs.items():
            if isinstance(value, str) and len(value) > 5000:
                truncated_inputs[key] = truncate_string(value, 5000)
            else:
                truncated_inputs[key] = value
        
        return f"""You are an expert grading agent for mathematical problems.

Your task is to evaluate a student's answer based on the provided problem, solution, and grading guidelines.

Task input:
```json
{json.dumps(truncated_inputs, indent=2)}
```

Instructions:
1. Carefully read the problem, solution, and grading guidelines
2. Evaluate the student's answer against the criteria
3. Provide your assessment in the exact JSON format below

Respond in JSON format with the following schema:
<json>
{{
    "response": "Your detailed evaluation and grade here"
}}
</json>

Important: Your response MUST be valid JSON wrapped in <json> tags."""

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        self._call_count += 1
        
        instruction = self._build_prompt(inputs)

        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return f"Error: LLM call failed - {e}", []

        # Extract prediction from JSON
        prediction = "None"
        raw_response = ""
        try:
            raw_response = msg_history[-1]["text"]
        except (IndexError, KeyError) as e:
            logger.error(f"Failed to get raw response from msg_history: {e}")
            return "Error: Invalid message history", msg_history
        
        try:
            # Try flexible extraction first
            extracted = _extract_json_flexible(raw_response)
            if extracted and "response" in extracted:
                prediction = extracted["response"]
                self._success_count += 1
                logger.debug(f"Successfully extracted JSON response (length: {len(prediction)})")
            else:
                # Log the issue for debugging
                if extracted:
                    logger.warning(f"JSON extracted but missing 'response' key. Keys found: {list(extracted.keys())}")
                else:
                    logger.warning(f"No JSON found in response. Response preview: {raw_response[:200]}...")
                
                # Fallback: try to extract any meaningful content
                text = raw_response
                # Remove common wrappers
                for wrapper in ["<json>", "</json>", "```json", "```"]:
                    text = text.replace(wrapper, "")
                prediction = text.strip()[:1000]  # Limit length
                
        except Exception as e:
            logger.error(f"Error extracting prediction: {e}")
            # Use raw response as fallback
            prediction = raw_response[:1000] if raw_response else "Error: Could not extract prediction"

        return str(prediction), msg_history

    def get_stats(self) -> dict[str, Any]:
        """Get agent statistics."""
        return {
            "total_calls": self._call_count,
            "successful_extractions": self._success_count,
            "extraction_rate": self._success_count / self._call_count if self._call_count > 0 else 0,
        }

    def reset_stats(self) -> None:
        """Reset agent statistics."""
        self._call_count = 0
        self._success_count = 0
