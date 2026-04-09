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
    Also attempts to extract JSON from markdown code blocks as fallback.
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
    
    # Fallback: try to extract JSON from markdown code blocks
    if not results:
        json_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
        matches = re.findall(json_pattern, text, re.DOTALL)
        for match in matches:
            try:
                results.append(json.loads(match.strip()))
            except json.JSONDecodeError:
                continue
    
    # Second fallback: try to find any JSON-like structure with "response" key
    if not results:
        try:
            # Look for patterns like {"response": ...}
            response_pattern = r'"response"\s*:\s*([^}]+)'
            match = re.search(response_pattern, text)
            if match:
                response_val = match.group(1).strip().strip('"').strip("'")
                results.append({"response": response_val})
        except Exception:
            pass
    
    return results or None


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

Respond in JSON format with the following schema:
<json>
{{
    "response": <your evaluation result>
}}
</json>

Provide a clear, concise evaluation in the "response" field."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON
        prediction = "None"
        try:
            extracted = _extract_jsons(msg_history[-1]["text"])
            if extracted and "response" in extracted[-1]:
                prediction = extracted[-1]["response"]
                self.log_fn(f"Successfully extracted prediction: {prediction}")
            else:
                self.log_fn(f"No valid response field found in extracted JSON: {extracted}")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            # Last resort: use the raw response text
            if msg_history and len(msg_history) > 0:
                raw_text = msg_history[-1].get("text", "")
                if raw_text:
                    prediction = raw_text[:500]  # Truncate for safety
                    self.log_fn(f"Using raw text as fallback prediction")

        return str(prediction), msg_history
