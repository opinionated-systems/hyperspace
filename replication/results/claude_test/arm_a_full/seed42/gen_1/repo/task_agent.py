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


def _fallback_extract_score(text: str) -> str | None:
    """Last-resort extraction: scan for a bare integer score in the response.

    Looks for patterns like "score: 7", "awarded 3 points", "gives 0 points",
    or a lone digit on its own line.  Returns the matched integer as a string,
    or None if nothing is found.
    """
    # Explicit label patterns (case-insensitive)
    for pattern in [
        r"(?:score|points?|awarded?)[:\s]+(\d+)",
        r"(\d+)\s+(?:out of|/)\s*\d+",
        r"^\s*(\d+)\s*$",
    ]:
        m = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1)
    return None


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an expert mathematical olympiad grader. Your task is to evaluate a
student's answer to an International Mathematical Olympiad (IMO) problem and
assign a score according to the official grading guidelines.

Follow these steps carefully:
1. Read the problem statement and the official solution.
2. Study the grading guidelines to understand what partial credit is available.
3. Analyse the student's answer step by step, identifying which parts are
   correct, which are incomplete, and which are wrong.
4. Decide on a score that is consistent with the grading guidelines.
5. Output your final answer in the exact JSON format shown below.

Be rigorous but fair. Award partial credit where the guidelines allow it.
Do not penalise a student for using a different (but valid) approach.
"""

_USER_TEMPLATE = """\
## Problem

Domain: {domain}

{problem}

## Official Solution

{solution}

## Grading Guidelines

{grading_guidelines}

## Student Answer

{student_answer}

---

Think step by step, then provide your final score in the following JSON block:

<json>
{{
    "reasoning": "<brief explanation of your scoring decision>",
    "response": <integer score>
}}
</json>
"""


class TaskAgent:
    """Task agent that solves IMO grading problems."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines,
                    student_answer

        Returns:
            (prediction, msg_history)
        """
        # Build a structured, domain-aware prompt.
        try:
            instruction = _USER_TEMPLATE.format(
                domain=inputs.get("domain", "Mathematics"),
                problem=inputs.get("problem", ""),
                solution=inputs.get("solution", ""),
                grading_guidelines=inputs.get("grading_guidelines", ""),
                student_answer=inputs.get("student_answer", ""),
            )
        except (KeyError, AttributeError):
            # Fallback: dump the raw inputs dict if the expected keys are absent.
            instruction = (
                f"{_SYSTEM_PROMPT}\n\nTask input:\n```\n{inputs}\n```\n\n"
                "Respond with your score in a <json>{\"response\": <score>}</json> block."
            )

        # Prepend the system prompt as the first user turn so that models
        # without a dedicated system-message slot still receive it.
        system_turn = _SYSTEM_PROMPT.strip()
        msg_history_init: list[dict] = [
            {"role": "user", "text": system_turn},
            {"role": "assistant", "text": "Understood. I will grade the student's answer carefully and output my score in the requested JSON format."},
        ]

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=msg_history_init,
        )

        # --- Extract prediction from JSON block ---
        prediction = "None"
        try:
            extracted = _extract_jsons(msg_history[-1]["text"])
            if extracted:
                last = extracted[-1]
                if "response" in last:
                    prediction = last["response"]
        except Exception as e:
            self.log_fn(f"Error extracting prediction from JSON block: {e}")

        # --- Fallback: scan raw text for a bare score ---
        if prediction == "None":
            try:
                raw_text = msg_history[-1]["text"]
                fallback = _fallback_extract_score(raw_text)
                if fallback is not None:
                    self.log_fn(
                        f"JSON extraction failed; using fallback score: {fallback}"
                    )
                    prediction = fallback
            except Exception as e:
                self.log_fn(f"Fallback extraction error: {e}")

        return str(prediction), msg_history
