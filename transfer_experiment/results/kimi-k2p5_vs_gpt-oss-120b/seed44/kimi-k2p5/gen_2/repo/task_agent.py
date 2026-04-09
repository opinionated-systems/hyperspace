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


def _analyze_grading_guidelines(grading_guidelines: str) -> dict:
    """Analyze the grading guidelines to understand the expected evaluation criteria.
    
    The grading guidelines have a specific structure:
    - (Partial) section: lists achievements that earn partial credit
    - (Almost) section: describes what was almost achieved but had issues
    
    Key patterns:
    - Correct: "Verification contains minor mistakes only" in Almost section
    - Partial: "minor mistakes which are not negligible" or "failed to prove" or "did not verify"
    - Incorrect: Missing key insights or significant gaps
    """
    guidelines = grading_guidelines.lower()
    
    # Check for Almost section indicators
    has_almost = "(almost)" in guidelines
    
    # Key phrases that indicate different levels
    minor_mistakes_only = "minor mistakes only" in guidelines
    not_negligible = "not negligible" in guidelines
    failed_to = "failed to" in guidelines or "did not" in guidelines
    omitted = "omitted" in guidelines
    
    # Count items in Partial section (achievements)
    partial_items = guidelines.count("(partial)")
    
    return {
        "has_almost": has_almost,
        "minor_mistakes_only": minor_mistakes_only,
        "not_negligible": not_negligible,
        "failed_to": failed_to,
        "omitted": omitted,
        "partial_items": partial_items,
    }


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
        # Extract key fields for better prompting
        domain = inputs.get("domain", "")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        # Analyze grading guidelines structure
        guideline_analysis = _analyze_grading_guidelines(grading_guidelines)
        
        instruction = f"""You are an expert mathematical grader. Your task is to evaluate a student's answer to a mathematics problem.

DOMAIN: {domain}

PROBLEM:
{problem}

OFFICIAL SOLUTION:
{solution}

GRADING GUIDELINES:
{grading_guidelines}

STUDENT'S ANSWER TO EVALUATE:
{student_answer}

---

GRADING INSTRUCTIONS:

The grading guidelines have a specific structure with two key sections:

1. (Partial) section: Lists achievements that earn partial credit. These are positive accomplishments by the student.

2. (Almost) section: Describes what was "almost" achieved but had issues. This section is CRITICAL for determining the final grade:
   - If it says "Verification contains minor mistakes only" → The answer is CORRECT
   - If it says "minor mistakes which are not negligible" or mentions "failed to", "did not verify", "omitted" → The answer is PARTIAL
   - If the (Partial) section is missing key items or the (Almost) section describes major gaps → The answer is INCORRECT

Based on this analysis, determine if the student's answer is:
- "correct" - Fully correct solution (minor mistakes only in verification)
- "incorrect" - Wrong or significantly flawed, missing key insights
- "partial" - Has correct elements but incomplete or has non-negligible errors

You must respond with ONLY a JSON object in the following format (no other text):
<json>
{{
    "response": "correct" | "incorrect" | "partial"
}}
</json>"""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON with multiple fallback strategies
        prediction = "None"
        try:
            # Try to extract from JSON tags
            extracted = _extract_jsons(msg_history[-1]["text"])
            if extracted and "response" in extracted[-1]:
                prediction = extracted[-1]["response"]
            else:
                # Fallback: look for direct answer in text
                text = msg_history[-1]["text"].lower()
                if "correct" in text and "incorrect" not in text and "partial" not in text:
                    prediction = "correct"
                elif "partial" in text:
                    prediction = "partial"
                elif "incorrect" in text:
                    prediction = "incorrect"
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            # Final fallback: simple keyword matching
            text = msg_history[-1]["text"].lower()
            if "partial" in text:
                prediction = "partial"
            elif "incorrect" in text:
                prediction = "incorrect"
            elif "correct" in text:
                prediction = "correct"

        return str(prediction), msg_history
