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
    """Extract JSON with fallback strategies for malformed responses.
    
    First tries <json> tags, then falls back to finding JSON-like structures
    in the text for more robust extraction.
    """
    # Try standard extraction first
    results = _extract_jsons(text)
    if results:
        return results[-1]
    
    # Fallback: try to find JSON-like structures with braces
    try:
        # Find content between outermost braces
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            candidate = text[start:end+1]
            # Try to parse it
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
    except json.JSONDecodeError:
        pass
    
    return None


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
        instruction = f"""You are an expert grading agent for mathematical problems.

Your task is to evaluate a student's answer based on the provided problem, solution, and grading guidelines.

Task input:
```
{inputs}
```

You must respond in JSON format with the following schema:
<json>
{{
    "response": <your evaluation result>
}}
</json>

Important: Ensure your response is valid JSON inside the <json> tags."""

        max_retries = 3
        prediction = "None"
        msg_history = []
        
        for attempt in range(max_retries):
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=msg_history if attempt > 0 else [],
            )

            # Extract prediction from JSON with flexible fallback
            try:
                extracted = _extract_jsons(msg_history[-1]["text"])
                if extracted and "response" in extracted[-1]:
                    prediction = extracted[-1]["response"]
                    break
                else:
                    # Try flexible extraction
                    flexible = _extract_json_flexible(msg_history[-1]["text"])
                    if flexible and "response" in flexible:
                        prediction = flexible["response"]
                        break
                    elif attempt < max_retries - 1:
                        # Add feedback for retry
                        instruction = "Your previous response did not contain valid JSON with a 'response' field. Please try again with proper JSON formatting inside <json> tags."
                        self.log_fn(f"Retry {attempt + 1}: JSON extraction failed, attempting retry")
            except Exception as e:
                self.log_fn(f"Error extracting prediction (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    instruction = f"Error parsing your response: {e}. Please respond with valid JSON inside <json> tags."

        return str(prediction), msg_history
