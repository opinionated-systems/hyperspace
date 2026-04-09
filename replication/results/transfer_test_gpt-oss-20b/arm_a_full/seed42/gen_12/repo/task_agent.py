"""
# Updated task_agent with improved JSON extraction and fallback parsing

Task agent: solves a given task with a single LLM call.

Reimplemented from facebookresearch/HyperAgents task_agent.py.
Same interface, same JSON output format, same extraction logic.

This is the INITIAL task agent. The meta agent modifies this file
during self-improvement. The evaluation harness loads whatever
# Improved JSON extraction logic: handles <json> tags and fallback to full message parsing

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

Instructions:
1. Carefully read the problem, the correct solution, and the grading guidelines
2. Analyze the student's answer thoroughly
3. Provide your evaluation in the exact JSON format below

Respond in JSON format:
<json>
{{
    "response": "Your detailed evaluation and grade here"
}}
</json>

Important: 
- The "response" field should contain your complete evaluation
- Be thorough and reference specific parts of the student's answer
- Follow the grading guidelines precisely"""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )


        prediction = "None"
        try:
            extracted = _extract_jsons(msg_history[-1]["text"])
            if extracted and "response" in extracted[-1]:
                prediction = extracted[-1]["response"]
            else:
                # Fallback to direct extraction from raw text
                prediction = _extract_response_from_text(msg_history[-1]["text"]) or "None"
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        try:
            extracted = _extract_jsons(msg_history[-1]["text"])
            if extracted and "response" in extracted[-1]:
                prediction = extracted[-1]["response"]
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")


# Helper function to extract a response string from a JSON object or plain text

def _extract_response_from_text(text: str) -> str | None:
    """Attempt to extract a 'response' value from the given text.

    The function first tries to parse the entire text as JSON. If that
    succeeds and contains a ``response`` key, the value is returned.
    If parsing fails, a simple regex is used to find a ``"response"`` key
    in a JSON-like structure.  If that also fails, ``None`` is returned.
    """
    if not text:
        return None
    # Try full JSON parse
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "response" in data:
            return data["response"]
    except Exception:
        pass
    # Fallback regex search for "response": "..."
    match = re.search(r'"response"\s*:\s*"([^"]*)"', text)
    if match:
        return match.group(1)
    return None

