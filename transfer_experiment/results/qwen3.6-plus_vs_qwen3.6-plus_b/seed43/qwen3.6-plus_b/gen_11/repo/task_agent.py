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

VERSION = "1.8.0"  # Robust label extraction: case-insensitive search, word-boundary matching, priority ordering

VALID_LABELS = {"Correct", "Incorrect", "Partial", "Almost"}
# Ordered by specificity to avoid false matches (e.g. "Almost" matching inside other words)
VALID_LABELS_ORDERED = ["Incorrect", "Correct", "Partial", "Almost"]


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


def _extract_json_regex(text: str) -> list[dict] | None:
    """Fallback: use regex to find JSON-like objects in text."""
    results = []
    # Try to find JSON objects with "response" key
    pattern = r'\{\s*"response"\s*:\s*"([^"]*)"\s*\}'
    matches = re.findall(pattern, text)
    for match in matches:
        results.append({"response": match})
    return results or None


def _extract_json_fallback(text: str) -> dict | None:
    """Fallback: try to parse the entire text or find a JSON object in it.

    Handles cases where the LLM returns raw JSON without <json> tags.
    """
    # Try parsing the whole text first
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # Try to find a JSON object by locating the first '{' and last '}'
    start = text.find("{")
    if start == -1:
        return None
    end = text.rfind("}")
    if end == -1 or end <= start:
        return None
    candidate = text[start:end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def _extract_label_from_text(text: str) -> str | None:
    """Last resort: try to find a valid label anywhere in the text.
    
    Uses word-boundary regex matching to avoid false positives
    (e.g. 'correct' matching inside 'incorrect').
    Searches in priority order: Incorrect > Correct > Partial > Almost.
    """
    for label in VALID_LABELS_ORDERED:
        # Use word boundary matching to avoid partial matches
        pattern = r'\b' + re.escape(label) + r'\b'
        if re.search(pattern, text, re.IGNORECASE):
            return label
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
        domain = inputs.get("domain", "")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        system_prompt = (
            f"You are an expert grader for {domain} problems. "
            "You must respond with a valid JSON object enclosed in <json> tags. "
            'The JSON must have exactly one key "response" with a value that is '
            'one of: "Correct", "Incorrect", "Partial", or "Almost". '
            "Do not include any other text outside the <json> tags."
        )

        instruction = f"""You are an expert grader for {domain} problems. Your task is to evaluate a student's answer against a reference solution and grading guidelines.

## Problem
{problem}

## Reference Solution
{solution}

## Grading Guidelines
{grading_guidelines}

## Student Answer
{student_answer}

Carefully evaluate the student's answer against the reference solution and grading guidelines.
Consider correctness, completeness, and reasoning quality.

Respond ONLY with a JSON object enclosed in <json> tags. The JSON must have this exact format:
<json>
{{"response": "Correct"}}
</json>

The "response" value must be exactly one of: "Correct", "Incorrect", "Partial", or "Almost".
Do not include any other text, explanation, or reasoning outside the <json> tags."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
            system_prompt=system_prompt,
        )

        # Extract prediction from JSON - try multiple strategies
        prediction = "None"
        try:
            response_text = msg_history[-1]["text"]

            # Strategy 1: Extract from <json> tags
            extracted = _extract_jsons(response_text)
            if extracted and "response" in extracted[-1]:
                prediction = extracted[-1]["response"]

            # Strategy 2: Regex fallback for simple JSON
            if prediction == "None":
                regex_results = _extract_json_regex(response_text)
                if regex_results and "response" in regex_results[-1]:
                    prediction = regex_results[-1]["response"]

            # Strategy 3: Full text / brace-matching fallback
            if prediction == "None":
                fallback = _extract_json_fallback(response_text)
                if fallback and "response" in fallback:
                    prediction = fallback["response"]

            # Strategy 4: Look for "Grade: <label>" pattern in text
            if prediction == "None":
                grade_match = re.search(r'Grade:\s*(Correct|Incorrect|Partial|Almost)', response_text, re.IGNORECASE)
                if grade_match:
                    prediction = grade_match.group(1)

            # Strategy 5: Last resort - scan text for valid labels with word boundaries
            if prediction == "None":
                label = _extract_label_from_text(response_text)
                if label:
                    prediction = label

        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
