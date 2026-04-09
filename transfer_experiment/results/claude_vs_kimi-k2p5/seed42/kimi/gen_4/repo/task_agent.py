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

# Valid grade labels
_VALID_GRADES = {"correct", "partial", "incorrect"}


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


def _extract_grade_fallback(text: str) -> str | None:
    """Fallback: scan the text for a bare grade keyword.

    Looks for lines like 'Grade: correct' or a standalone grade word
    near the end of the response, in case the model forgot the JSON tags.
    Prioritizes grades found near the end of the response.
    """
    # Try labelled patterns first: "grade: correct", "verdict: partial", etc.
    labelled = re.search(
        r'(?:grade|verdict|assessment|result|prediction|decision)\s*[:\-]\s*(correct|partial|incorrect)',
        text,
        re.IGNORECASE,
    )
    if labelled:
        return labelled.group(1).lower()

    # Try a bare JSON-like fragment without tags: {"response": "correct"}
    bare_json = re.search(
        r'\{[^{}]*"response"\s*:\s*"(correct|partial|incorrect)"[^{}]*\}',
        text,
        re.IGNORECASE,
    )
    if bare_json:
        return bare_json.group(1).lower()

    # Try to find grade in quotes near the end of the text
    # This handles cases like: "The grade is 'correct'."
    quoted = re.search(
        r'["\'](correct|partial|incorrect)["\']',
        text,
        re.IGNORECASE,
    )
    if quoted:
        return quoted.group(1).lower()

    # Last resort: find the last occurrence of a valid grade word as a
    # standalone token (surrounded by non-word characters or string boundaries).
    # We search from the end of the text to prioritize later mentions.
    matches = list(re.finditer(r'\b(correct|partial|incorrect)\b', text, re.IGNORECASE))
    if matches:
        # Return the last match (most likely to be the final decision)
        return matches[-1].group(1).lower()

    return None


class TaskAgent:
    """Task agent that solves IMO grading problems."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_file = log_file
        self.log_fn = logger.info

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines,
                    student_answer

        Returns:
            (prediction, msg_history)
        """
        problem = inputs.get("problem", "")
        official_solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = f"""You are an expert mathematics competition grader.
Your task is to evaluate a student's answer to a competition problem and assign
one of three grades: "correct", "partial", or "incorrect".

Grade Definitions
-----------------
- "correct"   : The student's answer is fully correct and complete.
  All key ideas, claims, and conclusions match the official solution
  (minor presentation differences are acceptable). The student has solved
  the problem completely with proper justification.
- "partial"   : The student's answer contains meaningful progress toward
  the correct answer but is incomplete or contains non-trivial errors.
  Examples include: a correct approach with a wrong final answer, a correct
  answer without adequate justification, a solution that handles only some
  cases, or significant progress toward the solution with key insights correct.
- "incorrect" : The student's answer is wrong, contains fundamental errors,
  or makes no meaningful progress toward the solution.

Problem
-------
{problem}

Official Solution
-----------------
{official_solution}

Grading Guidelines
------------------
{grading_guidelines}

Student's Answer
----------------
{student_answer}

Grading Instructions
--------------------
Step 1. Carefully read the problem, the official solution, and the grading
        guidelines. Understand what constitutes a complete solution.
Step 2. Evaluate the student's answer systematically:
        - Check if the final answer matches the official solution
        - Verify if the reasoning and proof steps are correct
        - Identify any errors, gaps, or missing components
        - Assess whether partial credit is warranted for correct insights
Step 3. Decide on a grade based on the definitions above:
        - "correct" only if fully correct and complete
        - "partial" if there is meaningful progress but incomplete
        - "incorrect" if wrong or no meaningful progress
Step 4. Write your final answer as a JSON object enclosed in <json> tags.
        The "response" field MUST be exactly one of: "correct", "partial", "incorrect".

Example output format:
<json>
{{
    "reasoning": "The student correctly identified the key invariant and provided a complete proof for both directions.",
    "response": "correct"
}}
</json>

Now provide your grading decision below."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # ------------------------------------------------------------------ #
        # Extract prediction from the assistant's response.                   #
        # Strategy:                                                            #
        #   1. Look for <json>...</json> blocks in the last assistant message. #
        #   2. If that fails, try a regex fallback on the full response text.  #
        #   3. Default to "incorrect" for safety if no valid grade found.      #
        # ------------------------------------------------------------------ #
        prediction = "incorrect"  # Default to incorrect for safety
        
        # Get assistant text from message history or response
        assistant_text = ""
        try:
            if msg_history:
                last_msg = msg_history[-1]
                # Handle both "content" and "text" keys
                assistant_text = last_msg.get("content", "") or last_msg.get("text", "") or ""
            if not assistant_text and response:
                assistant_text = response
        except Exception:
            assistant_text = response or ""

        # Try to extract from JSON blocks first
        if assistant_text:
            try:
                extracted = _extract_jsons(assistant_text)
                if extracted:
                    # Prefer the last JSON block that contains a valid "response" key.
                    for obj in reversed(extracted):
                        grade = str(obj.get("response", "")).strip().lower()
                        if grade in _VALID_GRADES:
                            prediction = grade
                            break
                    else:
                        # No valid grade found in any block; take the last block's value if present
                        last_response = str(extracted[-1].get("response", "")).strip().lower()
                        if last_response in _VALID_GRADES:
                            prediction = last_response
            except Exception as e:
                self.log_fn(f"Error extracting prediction from JSON blocks: {e}")

        # Fallback: scan the raw text for a grade keyword
        if prediction not in _VALID_GRADES:
            try:
                fallback = _extract_grade_fallback(assistant_text)
                if fallback and fallback in _VALID_GRADES:
                    prediction = fallback
                    self.log_fn(f"Used fallback grade extraction: {prediction}")
            except Exception as e:
                self.log_fn(f"Error in fallback grade extraction: {e}")

        # Ensure we always return a valid grade
        if prediction not in _VALID_GRADES:
            prediction = "incorrect"

        return str(prediction), msg_history
