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


def _extract_label_from_text(text: str) -> str | None:
    """Extract grading label from raw text using multiple strategies."""
    text_lower = text.lower()
    
    # Look for explicit label mentions
    # Check for "correct" (but not "incorrect")
    if re.search(r'\bcorrect\b', text_lower) and not re.search(r'\bincorrect\b', text_lower):
        return "correct"
    
    # Check for "incorrect"
    if re.search(r'\bincorrect\b', text_lower):
        return "incorrect"
    
    # Check for "partial"
    if re.search(r'\bpartial\b', text_lower):
        return "partial"
    
    # Check JSON-like structures without tags
    try:
        # Try to find JSON objects directly
        json_pattern = re.search(r'\{[^}]*"(?:response|label|grade|evaluation)"[^}]*\}', text, re.DOTALL)
        if json_pattern:
            data = json.loads(json_pattern.group())
            for key in ["response", "label", "grade", "evaluation"]:
                if key in data:
                    val = str(data[key]).lower()
                    if "incorrect" in val:
                        return "incorrect"
                    elif "correct" in val:
                        return "correct"
                    elif "partial" in val:
                        return "partial"
                    return val
    except Exception:
        pass
    
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
        # Extract key fields for better prompting
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        instruction = f"""You are an expert mathematics grader for IMO (International Mathematical Olympiad) problems.

Your task is to evaluate a student's answer against the official solution and grading guidelines.

PROBLEM:
{problem}

OFFICIAL SOLUTION:
{solution}

GRADING GUIDELINES:
{grading_guidelines}

STUDENT'S ANSWER:
{student_answer}

Evaluate the student's answer and classify it into one of three categories:
- "correct": The answer is fully correct and complete
- "incorrect": The answer is wrong or fundamentally flawed  
- "partial": The answer has some correct elements but is incomplete or has minor errors

Respond in JSON format with the following schema:
<json>
{{
    "response": "correct" | "incorrect" | "partial",
    "reasoning": "Brief explanation of your evaluation"
}}
</json>"""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction using multiple strategies
        prediction = "None"
        try:
            # Strategy 1: Try to extract from JSON tags
            extracted = _extract_jsons(msg_history[-1]["text"])
            if extracted:
                for key in ["response", "label", "grade", "evaluation"]:
                    if key in extracted[-1]:
                        val = str(extracted[-1][key]).lower().strip()
                        # Normalize the value
                        if "incorrect" in val:
                            prediction = "incorrect"
                        elif "correct" in val:
                            prediction = "correct"
                        elif "partial" in val:
                            prediction = "partial"
                        else:
                            prediction = val
                        break
            
            # Strategy 2: If JSON extraction failed, try text extraction
            if prediction == "None":
                text_pred = _extract_label_from_text(msg_history[-1]["text"])
                if text_pred:
                    prediction = text_pred
                    
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
