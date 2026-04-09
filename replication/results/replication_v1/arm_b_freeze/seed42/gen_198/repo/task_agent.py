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
        except json.JSONDecodeError as e:
            logger.debug(f"JSON decode error at position {start}: {e}")
            continue
    return results or None


def _extract_json_fuzzy(text: str) -> list[dict] | None:
    """Fuzzy JSON extraction as fallback when strict parsing fails.
    
    Attempts to find JSON-like structures even without proper tags.
    """
    results = []
    # Try to find JSON objects between curly braces
    brace_count = 0
    start_idx = None
    for i, char in enumerate(text):
        if char == '{':
            if brace_count == 0:
                start_idx = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start_idx is not None:
                try:
                    candidate = text[start_idx:i+1]
                    results.append(json.loads(candidate))
                except json.JSONDecodeError:
                    pass
                start_idx = None
    return results or None


class TaskAgent:
    """Task agent that solves IMO grading problems."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.max_retries = 2

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        instruction = f"""You are an expert grading agent for mathematical olympiad problems.

Task input:
```
{inputs}
```

Analyze the student's answer carefully against the solution and grading guidelines.
Respond in JSON format with the following schema:
<json>
{{
    "response": <your grading decision>
}}
</json>"""

        msg_history: list[dict] = []
        prediction = "None"
        
        for attempt in range(self.max_retries + 1):
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=msg_history if attempt > 0 else [],
            )

            # Extract prediction from JSON
            try:
                last_text = msg_history[-1]["text"]
                extracted = _extract_jsons(last_text)
                
                # Fallback to fuzzy extraction if strict parsing fails
                if extracted is None:
                    extracted = _extract_json_fuzzy(last_text)
                    if extracted:
                        self.log_fn(f"Used fuzzy JSON extraction on attempt {attempt + 1}")
                
                if extracted and "response" in extracted[-1]:
                    prediction = extracted[-1]["response"]
                    break
                else:
                    self.log_fn(f"No valid 'response' field found on attempt {attempt + 1}")
                    
            except Exception as e:
                self.log_fn(f"Error extracting prediction on attempt {attempt + 1}: {e}")
                
            # Add retry instruction for next attempt
            if attempt < self.max_retries:
                instruction = "Please respond with valid JSON in the format: <json>{\"response\": ...}</json>"

        return str(prediction), msg_history
