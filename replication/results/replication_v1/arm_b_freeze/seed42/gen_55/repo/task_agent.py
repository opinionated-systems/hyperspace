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
        # Format inputs nicely for the prompt
        formatted_inputs = json.dumps(inputs, indent=2, default=str)
        
        instruction = f"""You are an expert task-solving agent. Your goal is to analyze the given task input and provide a well-reasoned response.

Task input:
```json
{formatted_inputs}
```

Instructions:
1. Carefully read and understand the task input
2. Analyze the problem, solution, and student answer
3. Provide your response in the exact JSON format below

Respond in JSON format with the following schema:
<json>
{{
    "response": "Your detailed answer here"
}}
</json>

Important: Ensure your response is valid JSON and properly escaped."""

        self.log_fn(f"TaskAgent processing input with keys: {list(inputs.keys())}")
        
        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON
        prediction = "None"
        try:
            if msg_history:
                last_msg = msg_history[-1]
                text = last_msg.get("text", "")
                extracted = _extract_jsons(text)
                if extracted:
                    last_extracted = extracted[-1]
                    if isinstance(last_extracted, dict) and "response" in last_extracted:
                        prediction = last_extracted["response"]
                        self.log_fn(f"Successfully extracted prediction: {str(prediction)[:100]}...")
                    else:
                        self.log_fn(f"Extracted JSON missing 'response' key: {last_extracted}")
                else:
                    self.log_fn("No JSON blocks found in response")
            else:
                self.log_fn("Empty message history")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
