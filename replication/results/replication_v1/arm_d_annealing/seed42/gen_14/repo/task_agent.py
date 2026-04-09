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
    """Fallback JSON extraction for malformed responses.
    
    Attempts to find JSON-like structures even without proper tags.
    """
    results = []
    # Try to find JSON objects between curly braces
    brace_count = 0
    start_idx = -1
    for i, char in enumerate(text):
        if char == '{':
            if brace_count == 0:
                start_idx = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start_idx != -1:
                try:
                    obj = json.loads(text[start_idx:i+1])
                    results.append(obj)
                except json.JSONDecodeError:
                    pass
                start_idx = -1
    return results or None


class TaskAgent:
    """Task agent that solves IMO grading problems."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.stats = {"calls": 0, "json_extracted": 0, "fuzzy_extracted": 0, "failed": 0}

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        self.stats["calls"] += 1
        
        instruction = f"""You are an expert grading agent. Analyze the student answer carefully and provide your assessment.

Task input:
```
{inputs}
```

Respond in JSON format with the following schema:
<json>
{{
    "response": "Your detailed assessment here"
}}
</json>"""

        self.log_fn(f"TaskAgent call #{self.stats['calls']}: Sending request to {self.model}")
        
        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON
        prediction = "None"
        raw_text = msg_history[-1]["text"] if msg_history else ""
        
        try:
            # First try strict extraction with <json> tags
            extracted = _extract_jsons(raw_text)
            if extracted and "response" in extracted[-1]:
                prediction = extracted[-1]["response"]
                self.stats["json_extracted"] += 1
                self.log_fn(f"Successfully extracted JSON response (call #{self.stats['calls']})")
            else:
                # Fallback to fuzzy extraction
                self.log_fn(f"Strict JSON extraction failed, trying fuzzy extraction (call #{self.stats['calls']})")
                extracted = _extract_json_fuzzy(raw_text)
                if extracted and "response" in extracted[-1]:
                    prediction = extracted[-1]["response"]
                    self.stats["fuzzy_extracted"] += 1
                    self.log_fn(f"Fuzzy extraction succeeded (call #{self.stats['calls']})")
                else:
                    self.stats["failed"] += 1
                    self.log_fn(f"All extraction methods failed (call #{self.stats['calls']})")
                    # Try to use raw text as fallback
                    if raw_text.strip():
                        prediction = raw_text.strip()[:500]  # Limit length
        except Exception as e:
            self.stats["failed"] += 1
            self.log_fn(f"Error extracting prediction (call #{self.stats['calls']}): {e}")

        return str(prediction), msg_history
    
    def get_stats(self) -> dict:
        """Return extraction statistics."""
        return self.stats.copy()
