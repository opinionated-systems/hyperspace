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
        # Extract fields from inputs for clearer prompting
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        instruction = f"""You are an expert mathematics grader for International Mathematical Olympiad (IMO) style problems.

Your task is to evaluate a student's answer and classify it into one of three categories:
- "correct": The student's answer is fully correct, complete, and matches the expected solution.
- "incorrect": The student's answer is wrong, incomplete in a critical way, or demonstrates fundamental misunderstanding.
- "partial": The student's answer shows significant progress, correct insights, or partial correctness but is not fully complete or has minor errors.

PROBLEM:
```
{problem}
```

OFFICIAL SOLUTION:
```
{solution}
```

GRADING GUIDELINES:
```
{grading_guidelines}
```

STUDENT'S ANSWER TO EVALUATE:
```
{student_answer}
```

Carefully analyze the student's answer against the official solution and grading guidelines. Consider:
1. Does the student understand the core concepts?
2. Are the key steps correct?
3. Is the final answer correct?
4. How does this compare to the grading guidelines?

Respond with EXACTLY ONE of these three labels: "correct", "incorrect", or "partial".

Respond in JSON format with the following schema:
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

        # Extract prediction from JSON
        prediction = "None"
        try:
            extracted = _extract_jsons(msg_history[-1]["text"])
            if extracted and "response" in extracted[-1]:
                prediction = extracted[-1]["response"]
                # Normalize to one of the three valid labels
                prediction_str = str(prediction).strip().lower()
                if prediction_str in ["correct", "incorrect", "partial"]:
                    prediction = prediction_str
                elif "correct" in prediction_str and "partial" not in prediction_str and "incorrect" not in prediction_str:
                    prediction = "correct"
                elif "partial" in prediction_str:
                    prediction = "partial"
                elif "incorrect" in prediction_str or "wrong" in prediction_str:
                    prediction = "incorrect"
                else:
                    prediction = "incorrect"  # Default fallback
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
