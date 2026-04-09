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

__version__ = "1.1.0"


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


def _extract_label_from_text(text: str) -> str | None:
    """Extract label directly from text using pattern matching as fallback."""
    text_lower = text.lower()

    # Look for explicit label mentions in JSON-like patterns first
    # Check for patterns like "response": "partial" or "label": "correct"
    json_pattern = re.search(r'["\'](?:response|label|classification|answer|grade|result)["\']\s*:\s*["\'](\w+)["\']', text_lower)
    if json_pattern:
        label = json_pattern.group(1)
        if label in ("correct", "incorrect", "partial"):
            return label

    # Look for explicit label mentions
    # Check for "partial" first since it contains "correct"
    if re.search(r'\bpartial\b', text_lower):
        return "partial"
    if re.search(r'\bincorrect\b|\bwrong\b', text_lower):
        return "incorrect"
    if re.search(r'\bcorrect\b', text_lower):
        return "correct"

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
        # Extract fields from inputs for better prompting
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = f"""You are an expert mathematical grader for International Mathematical Olympiad (IMO) problems.

Your task is to evaluate a student's answer and classify it as EXACTLY ONE of:
- "correct": The answer is fully correct and complete
- "partial": The answer has some valid progress but is incomplete or has significant gaps
- "incorrect": The answer is wrong or fundamentally flawed

PROBLEM:
{problem}

OFFICIAL SOLUTION:
{solution}

GRADING GUIDELINES:
{grading_guidelines}

STUDENT'S ANSWER TO EVALUATE:
{student_answer}

Analyze the student's answer carefully:
1. Check if the student correctly identified the key insights and approach
2. Verify if the proof/solution is logically sound and complete
3. Look for any gaps, errors, or missing steps
4. Compare against the official solution and grading guidelines

IMPORTANT: Your response must contain ONLY a JSON object inside <json> tags. Do not include any other text, explanation, or reasoning outside the JSON block.

<json>
{{"response": "correct"}}
</json>

OR

<json>
{{"response": "partial"}}
</json>

OR

<json>
{{"response": "incorrect"}}
</json>"""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON
        prediction = None
        try:
            # Get the last assistant message
            last_message = msg_history[-1]["text"] if msg_history else ""

            # Try JSON extraction first
            extracted = _extract_jsons(last_message)
            if extracted:
                for obj in extracted:
                    if "response" in obj:
                        prediction = str(obj["response"]).lower().strip()
                        break

            # If JSON extraction failed, try text extraction
            if prediction is None:
                prediction = _extract_label_from_text(last_message)

            # Normalize the prediction
            if prediction:
                # Clean up the prediction
                prediction = prediction.strip('"\'')

                # Map to valid labels
                if prediction == "partial":
                    prediction = "partial"
                elif prediction in ["incorrect", "wrong", "false", "error"]:
                    prediction = "incorrect"
                elif prediction == "correct":
                    prediction = "correct"
                else:
                    # Try to infer from content
                    if "partial" in prediction:
                        prediction = "partial"
                    elif "incorrect" in prediction or "wrong" in prediction:
                        prediction = "incorrect"
                    elif "correct" in prediction:
                        prediction = "correct"
                    else:
                        prediction = None

        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            prediction = None

        # Default to "incorrect" if we couldn't extract a valid prediction
        if prediction not in ["correct", "incorrect", "partial"]:
            prediction = "incorrect"

        return prediction, msg_history
