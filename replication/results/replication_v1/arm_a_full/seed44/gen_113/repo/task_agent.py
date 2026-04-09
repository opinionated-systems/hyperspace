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


def _extract_json_response(text: str) -> dict | None:
    """Extract JSON response from <json>...</json> blocks.
    
    Uses simple string search for tag pairs, then json.loads for parsing.
    Handles nested braces correctly via Python's JSON parser.
    """
    # Find all <json>...</json> blocks
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
    
    # Return the last valid JSON object with a "response" key
    for obj in reversed(results):
        if isinstance(obj, dict) and "response" in obj:
            return obj
    return None


def _extract_response_fallback(text: str) -> str | None:
    """Fallback extraction using regex patterns for malformed outputs."""
    # Pattern to match "response": followed by a value
    patterns = [
        # Number (integer, float, negative, scientific notation)
        (r'"response"\s*:\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)', 'number'),
        # String in double quotes
        (r'"response"\s*:\s*"([^"]*)"', 'string'),
        # Boolean or null
        (r'"response"\s*:\s*(true|false|null)', 'literal'),
    ]
    
    for pattern, ptype in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = match.group(1)
            if ptype == 'number':
                # Return as string, let caller handle type conversion if needed
                return value
            return value.strip()
    
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
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = f"""Grade this mathematics problem solution.

PROBLEM DOMAIN: {domain}

PROBLEM:
{problem}

OFFICIAL SOLUTION:
{solution}

GRADING GUIDELINES:
{grading_guidelines}

STUDENT'S ANSWER:
{student_answer}

Provide your grade in this exact format:
<json>{{"response": <grade>}}</json>

The grade should match the format specified in the grading guidelines (number, string, or boolean)."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from response
        prediction = "None"
        raw_text = msg_history[-1]["text"]
        
        # Try primary extraction from <json> tags
        json_obj = _extract_json_response(raw_text)
        if json_obj:
            prediction = json_obj["response"]
            self.log_fn(f"Extraction success: {prediction}")
        else:
            # Fallback to regex extraction
            fallback = _extract_response_fallback(raw_text)
            if fallback:
                prediction = fallback
                self.log_fn(f"Fallback extraction success: {prediction}")
            else:
                self.log_fn(f"Failed to extract prediction from: {raw_text[:200]}")

        return str(prediction), msg_history
