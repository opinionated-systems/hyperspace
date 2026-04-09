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

    Falls back to parsing the entire text as JSON if no tags are found.
    Also handles markdown code blocks (```json ... ```) as a secondary fallback.
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

    # Fallback 1: if no tagged JSON found, try parsing the whole text
    if not results:
        try:
            parsed = json.loads(text.strip())
            if isinstance(parsed, dict):
                results.append(parsed)
            elif isinstance(parsed, list):
                results.extend(item for item in parsed if isinstance(item, dict))
        except json.JSONDecodeError:
            pass

    # Fallback 2: try extracting from markdown code blocks
    if not results:
        code_blocks = re.findall(r'```(?:json)?\s*\n(.*?)\n```', text, re.DOTALL)
        for block in code_blocks:
            try:
                parsed = json.loads(block.strip())
                if isinstance(parsed, dict):
                    results.append(parsed)
                elif isinstance(parsed, list):
                    results.extend(item for item in parsed if isinstance(item, dict))
            except json.JSONDecodeError:
                continue

    return results or None


class TaskAgent:
    """Task agent that solves IMO grading problems."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            )
            logger.addHandler(file_handler)
            self.log_fn = logger.info

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        instruction = f"""You are an expert IMO (International Mathematical Olympiad) grading agent. Your task is to evaluate student answers against the provided solution and grading guidelines.

Task input:
```
{inputs}
```

Follow these steps carefully:
1. Read and understand the problem statement and the official solution.
2. Review the grading guidelines to understand the scoring rubric.
3. Analyze the student's answer step by step, checking for correctness, completeness, and mathematical rigor.
4. Compare the student's reasoning with the official solution.
5. Apply the grading guidelines to determine the appropriate score.

Your response must be a single JSON object with the following schema:
<json>
{{
    "score": "numerical score (integer) assigned to the student's answer based on the grading guidelines",
    "label": "one of: correct, incorrect, partial",
    "response": "Your detailed grading analysis explaining the score assignment"
}}
</json>

Important:
- The "score" field must be an integer reflecting the points earned.
- The "label" field must be exactly one of: "correct", "incorrect", or "partial".
- The "response" field must contain your complete grading analysis.
- Be thorough in comparing the student's work against the official solution.
- Explicitly state which parts of the grading criteria the student meets or fails.
- Provide a clear final assessment."""

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
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
