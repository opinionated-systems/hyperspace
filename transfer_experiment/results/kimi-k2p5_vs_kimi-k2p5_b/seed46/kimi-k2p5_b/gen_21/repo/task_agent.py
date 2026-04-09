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

# Valid grading labels for IMO evaluation - 3 class system
VALID_LABELS = ["correct", "partial", "incorrect"]


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks."""
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
    """Extract grading label from raw text using multiple strategies."""
    text_lower = text.lower()
    
    # Look for explicit label mentions in order of specificity
    # Check for "incorrect" first (most specific)
    if re.search(r'\bincorrect\b', text_lower):
        return "incorrect"
    
    # Check for "correct" (but not "incorrect" - already checked above)
    if re.search(r'\bcorrect\b', text_lower):
        return "correct"
    
    # Check for "partial"
    if re.search(r'\bpartial\b', text_lower):
        return "partial"
    
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
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        instruction = f"""You are an expert mathematics grader for International Mathematical Olympiad (IMO) problems. Your task is to evaluate student answers and assign exactly one of three grades.

## GRADE DEFINITIONS:

**"correct"**: The answer is fully correct and complete. It contains a valid proof or solution with all necessary steps. Minor typos are acceptable, but there must be no logical gaps or significant errors.

**"partial"**: The answer has some correct elements and shows meaningful progress toward the solution, but is incomplete or has significant gaps. The student understood part of the problem but not the full solution.

**"incorrect"**: The answer is wrong or fundamentally flawed. No valid mathematical progress toward the solution. The approach is completely wrong or trivial.

## GRADING GUIDELINES CONTEXT:
{grading_guidelines}

## PROBLEM:
{problem}

## OFFICIAL SOLUTION:
{solution}

## STUDENT'S ANSWER TO EVALUATE:
{student_answer}

## YOUR TASK:

Analyze the student's answer carefully. Compare it to the official solution and grading guidelines. Determine which of the three grades best describes the student's answer.

Respond with ONLY a JSON object in <json> tags:

<json>
{{
    "reasoning": "Brief explanation of your evaluation",
    "response": "correct" | "partial" | "incorrect"
}}
</json>

The "response" field MUST be exactly one of: "correct", "partial", or "incorrect" (all lowercase)."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction using multiple strategies
        prediction = "None"
        try:
            raw_text = msg_history[-1]["text"] if msg_history else ""
            
            # Strategy 1: Try to extract from JSON tags
            extracted = _extract_jsons(raw_text)
            if extracted:
                for key in ["response", "label", "grade", "evaluation"]:
                    val = extracted[-1].get(key, "")
                    if isinstance(val, str):
                        val_clean = val.strip().lower()
                        # Check for valid labels
                        if val_clean in VALID_LABELS:
                            prediction = val_clean.capitalize()
                            break
                        # Handle cases where the value might contain extra text
                        for label in VALID_LABELS:
                            if label in val_clean:
                                prediction = label.capitalize()
                                break
                        if prediction != "None":
                            break
            
            # Strategy 2: If JSON extraction failed, try text extraction
            if prediction == "None":
                text_pred = _extract_label_from_text(raw_text)
                if text_pred:
                    prediction = text_pred.capitalize()
                    
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
