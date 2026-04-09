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


def _extract_json_fuzzy(text: str) -> list[dict] | None:
    """Fuzzy JSON extraction as fallback when tags are missing.
    
    Attempts to find JSON objects by looking for curly brace pairs.
    """
    results = []
    # Try to find JSON objects without tags
    brace_pattern = re.compile(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}')
    
    for match in brace_pattern.finditer(text):
        try:
            obj = json.loads(match.group())
            if isinstance(obj, dict):
                results.append(obj)
        except json.JSONDecodeError:
            continue
    
    return results or None


class TaskAgent:
    """Task agent that solves IMO grading problems."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.max_retries = 2  # Number of retries for malformed JSON

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

        msg_history = []
        prediction = "None"
        
        for attempt in range(self.max_retries + 1):
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=msg_history if attempt > 0 else [],
            )

            # Extract prediction from JSON
            try:
                last_msg = msg_history[-1]["text"]
                extracted = _extract_jsons(last_msg)
                
                # Fallback to fuzzy extraction if no tagged JSON found
                if not extracted:
                    extracted = _extract_json_fuzzy(last_msg)
                    if extracted:
                        self.log_fn(f"Used fuzzy JSON extraction on attempt {attempt + 1}")
                
                if extracted and "response" in extracted[-1]:
                    prediction = extracted[-1]["response"]
                    break
                elif attempt < self.max_retries:
                    # Add feedback to retry
                    retry_msg = "Your previous response did not contain valid JSON with a 'response' field. Please respond with the correct JSON format."
                    msg_history.append({"role": "user", "text": retry_msg})
                    self.log_fn(f"JSON extraction failed on attempt {attempt + 1}, retrying...")
                
            except Exception as e:
                self.log_fn(f"Error extracting prediction on attempt {attempt + 1}: {e}")
                if attempt < self.max_retries:
                    retry_msg = f"Error parsing your response: {str(e)}. Please respond with valid JSON format."
                    msg_history.append({"role": "user", "text": retry_msg})

        return str(prediction), msg_history
