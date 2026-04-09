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


def _extract_prediction_from_text(text: str) -> str | None:
    """Extract prediction from raw text when JSON extraction fails.
    
    Looks for the three valid prediction values in the text.
    Returns the first match found, or None if no valid prediction.
    """
    text_lower = text.lower()
    
    # Look for exact matches of valid predictions
    # Check in order of specificity (partial before correct to avoid partial match issues)
    for prediction in ["incorrect", "partial", "correct"]:
        # Use word boundary check to avoid matching "correct" inside "incorrect"
        import re
        pattern = r'\b' + prediction + r'\b'
        if re.search(pattern, text_lower):
            return prediction
    
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
        # Extract fields from inputs for clearer prompting
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        instruction = f"""You are an expert mathematical grader for International Mathematical Olympiad (IMO) problems.

Your task is to evaluate a student's answer to a mathematics problem and classify it into exactly one of these three categories:
- "correct": The student's answer is fully correct and complete
- "incorrect": The student's answer is wrong or fundamentally flawed  
- "partial": The student's answer has some correct elements but is incomplete or has significant gaps

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

Carefully compare the student's answer against the official solution and grading guidelines. Consider:
1. Does the student understand the core problem?
2. Are their key claims and reasoning correct?
3. Did they identify the right approach?
4. Are there gaps or errors in their solution?

IMPORTANT: You must respond ONLY with a JSON object in the following format. Do not include any other text before or after the JSON.

<json>
{{
    "response": "correct" | "incorrect" | "partial"
}}
</json>

The value of "response" must be exactly one of: "correct", "incorrect", or "partial" (without quotes in the JSON value)."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON
        prediction = "incorrect"  # Default to incorrect
        try:
            response_text = msg_history[-1]["text"]
            extracted = _extract_jsons(response_text)
            if extracted and "response" in extracted[-1]:
                prediction = extracted[-1]["response"]
                # Normalize the prediction
                prediction = str(prediction).strip().lower()
                if prediction not in ["correct", "incorrect", "partial"]:
                    self.log_fn(f"Invalid prediction value: {prediction}, trying text extraction")
                    # Try to extract from raw text as fallback
                    text_prediction = _extract_prediction_from_text(response_text)
                    if text_prediction:
                        prediction = text_prediction
                    else:
                        prediction = "incorrect"
            else:
                # No JSON found, try text extraction
                text_prediction = _extract_prediction_from_text(response_text)
                if text_prediction:
                    prediction = text_prediction
                    self.log_fn(f"Extracted prediction from text: {prediction}")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            prediction = "incorrect"

        return str(prediction), msg_history
