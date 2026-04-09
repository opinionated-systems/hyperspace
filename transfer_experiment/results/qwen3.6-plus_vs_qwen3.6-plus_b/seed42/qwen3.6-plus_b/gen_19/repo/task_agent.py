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

__version__ = "1.5.0"

from agent.llm_client import get_response_from_llm, EVAL_MODEL

logger = logging.getLogger(__name__)


def get_version() -> str:
    """Return the current version of the task agent."""
    return __version__


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    Also falls back to a broader regex search if no <json> tags are found.
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

    # Fallback: if no tagged JSON found, try to extract bare JSON objects
    if not results:
        try:
            # Try parsing the entire text as JSON first
            parsed = json.loads(text.strip())
            if isinstance(parsed, dict):
                results.append(parsed)
        except json.JSONDecodeError:
            # Use a brace-counting approach to find JSON objects at any depth
            parsed = _extract_json_by_brace_counting(text)
            if parsed is not None:
                results.append(parsed)

    return results or None


def _extract_json_by_brace_counting(text: str) -> dict | None:
    """Find the first valid JSON object in text using brace counting.

    This handles arbitrarily deep nesting, unlike the regex-based approach.
    """
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape_next = False

    for i in range(start, len(text)):
        char = text[i]

        if escape_next:
            escape_next = False
            continue

        if char == "\\":
            if in_string:
                escape_next = True
            continue

        if char == '"':
            in_string = not in_string
            continue

        if in_string:
            continue

        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start:i + 1]
                try:
                    parsed = json.loads(candidate)
                    if isinstance(parsed, dict):
                        return parsed
                except json.JSONDecodeError:
                    pass
                # If this candidate failed, keep searching
                next_start = text.find("{", start + 1)
                if next_start == -1:
                    return None
                start = next_start
                depth = 0

    return None


class TaskAgent:
    """Task agent that solves IMO grading problems."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "", max_retries: int = 1) -> None:
        self.model = model
        self.max_retries = max_retries
        self.log_fn = logger.info
        self.log_fn(f"TaskAgent initialized with model: {self.model}, max_retries: {self.max_retries}")

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        instruction = f"""You are an expert IMO grading assistant. Grade the student's answer as "correct", "incorrect", or "partial".

## Problem
{inputs.get('problem', 'N/A')}

## Official Solution
{inputs.get('solution', 'N/A')}

## Grading Guidelines
{inputs.get('grading_guidelines', 'N/A')}

## Student's Answer
{inputs.get('student_answer', 'N/A')}

## Instructions
1. Compare the student's answer against the official solution and grading guidelines.
2. Identify which key steps the student got right, wrong, or partially correct.
3. Check for logical gaps, unjustified claims, missing lemmas, or computational errors.
4. Apply the grading guidelines strictly.

## Grading Criteria
- "correct": Fully correct, complete, and rigorous. All key steps present and justified.
- "incorrect": Fundamentally wrong, missing critical steps, or major logical errors.
- "partial": Meaningful progress with some correct key steps, but incomplete or with minor errors.

Respond in JSON format:
<json>
{{
    "reasoning": "Step-by-step analysis referencing the official solution and grading guidelines.",
    "response": "correct" | "incorrect" | "partial"
}}
</json>"""

        for attempt in range(1, self.max_retries + 1):
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
                    if prediction in ("correct", "incorrect", "partial"):
                        break  # Valid prediction found, exit retry loop
            except Exception as e:
                self.log_fn(f"Error extracting prediction (attempt {attempt}/{self.max_retries}): {e}")

            if attempt < self.max_retries:
                self.log_fn(f"Invalid or missing prediction on attempt {attempt}/{self.max_retries}, retrying...")

        return str(prediction), msg_history
