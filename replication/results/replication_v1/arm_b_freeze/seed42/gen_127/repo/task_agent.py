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
import time
from typing import Any

from agent.llm_client import get_response_from_llm, EVAL_MODEL

logger = logging.getLogger(__name__)

# Maximum retries for LLM calls
MAX_RETRIES = 3
# Delay between retries (exponential backoff)
RETRY_DELAY = 2


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
        except json.JSONDecodeError:
            logger.debug(f"Failed to parse JSON from <json> block: {inner[:100]}...")
            continue
    
    # Fallback: Extract from markdown code blocks ```json ... ```
    if not results:
        pattern = r'```json\s*(.*?)\s*```'
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                results.append(json.loads(match.strip()))
            except json.JSONDecodeError:
                logger.debug(f"Failed to parse JSON from markdown block: {match[:100]}...")
                continue
    
    # Last resort: Try to find any JSON object in the text
    if not results:
        # Look for content between curly braces
        pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                parsed = json.loads(match)
                if isinstance(parsed, dict):
                    results.append(parsed)
            except json.JSONDecodeError:
                continue
    
    return results or None


class TaskAgent:
    """Task agent that solves IMO grading problems."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.call_count = 0

    def _build_prompt(self, inputs: dict) -> str:
        """Build the prompt for the LLM."""
        return f"""You are an expert grading agent. Your task is to evaluate student answers accurately.

Task input:
```
{json.dumps(inputs, indent=2, default=str)}
```

Respond in JSON format with the following schema:
<json>
{{
    "response": "Your evaluation result here"
}}
</json>

Important: Ensure your response is valid JSON and follows the schema exactly."""

    def _call_llm_with_retry(self, instruction: str) -> tuple[str, list[dict], dict]:
        """Call LLM with retry logic for resilience."""
        last_error = None
        
        for attempt in range(MAX_RETRIES):
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=[],
                )
                return response, msg_history, info
            except Exception as e:
                last_error = e
                self.log_fn(f"LLM call attempt {attempt + 1}/{MAX_RETRIES} failed: {e}")
                if attempt < MAX_RETRIES - 1:
                    wait_time = RETRY_DELAY * (2 ** attempt)
                    self.log_fn(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)
        
        raise last_error if last_error else RuntimeError("All LLM call attempts failed")

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
        
        instruction = self._build_prompt(inputs)

        try:
            response, msg_history, info = self._call_llm_with_retry(instruction)
        except Exception as e:
            error_msg = f"LLM call failed after {MAX_RETRIES} attempts: {str(e)}"
            self.log_fn(f"ERROR: {error_msg}")
            return f"Error: {error_msg}", []

        # Extract prediction from JSON
        prediction = "None"
        try:
            if msg_history and len(msg_history) > 0:
                last_message = msg_history[-1]
                text_content = last_message.get("text", "")
                extracted = _extract_jsons(text_content)
                if extracted:
                    last_extracted = extracted[-1]
                    if "response" in last_extracted:
                        prediction = last_extracted["response"]
                        self.log_fn(f"Successfully extracted prediction: {str(prediction)[:100]}...")
                    else:
                        self.log_fn(f"Warning: Extracted JSON missing 'response' key. Keys found: {list(last_extracted.keys())}")
                else:
                    self.log_fn(f"Warning: No JSON found in response. Raw text: {text_content[:200]}...")
            else:
                self.log_fn("Warning: Empty message history from LLM")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
