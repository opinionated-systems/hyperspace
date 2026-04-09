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
    
    # Look for explicit patterns like "grade: correct" or "evaluation: partial"
    patterns = [
        r'grade[d]?\s*[:=]\s*(correct|incorrect|partial|almost)',
        r'evaluation\s*[:=]\s*(correct|incorrect|partial|almost)',
        r'label\s*[:=]\s*(correct|incorrect|partial|almost)',
        r'response\s*[:=]\s*(correct|incorrect|partial|almost)',
        r'classification\s*[:=]\s*(correct|incorrect|partial|almost)',
        r'"response"\s*:\s*"(correct|incorrect|partial|almost)"',
        r'"label"\s*:\s*"(correct|incorrect|partial|almost)"',
        r'"grade"\s*:\s*"(correct|incorrect|partial|almost)"',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            return match.group(1)
    
    # Check for standalone labels with word boundaries in priority order
    # Priority: almost > incorrect > partial > correct (most specific first)
    if re.search(r'\balmost\b', text_lower):
        return "almost"
    
    if re.search(r'\bincorrect\b', text_lower):
        return "incorrect"
    
    if re.search(r'\bpartial\b', text_lower):
        return "partial"
    
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
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        instruction = f"""You are an expert mathematics grader for International Mathematical Olympiad (IMO) style problems.

Your task is to evaluate a student's answer and classify it into exactly one of these four categories: "correct", "almost", "partial", or "incorrect".

LABEL DEFINITIONS:
- "correct": Fully correct and complete solution with rigorous proof
- "almost": Nearly correct (70-90% complete), right approach, only minor issues
- "partial": Some correct insights but significant gaps, less than 50% complete
- "incorrect": Fundamentally wrong approach, no meaningful progress

Problem:
{problem}

Official Solution:
{solution}

Grading Guidelines:
{grading_guidelines}

Student's Answer:
{student_answer}

Analyze the student's answer step by step. Then respond with ONLY a JSON object in this exact format:

<json>
{{
    "reasoning": "Your detailed analysis here",
    "response": "correct" or "almost" or "partial" or "incorrect"
}}
</json>

The "response" field MUST be exactly one of: "correct", "almost", "partial", or "incorrect" (all lowercase, no quotes around the value in the field)."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction
        prediction = "None"
        raw_text = msg_history[-1]["text"] if msg_history else ""
        
        try:
            # Strategy 1: Try to extract from <json> tags
            extracted = _extract_jsons(raw_text)
            if extracted:
                for json_obj in extracted:
                    if isinstance(json_obj, dict):
                        # Check response field first
                        resp = json_obj.get("response", "")
                        if isinstance(resp, str):
                            resp = resp.strip().lower()
                            if resp in ["correct", "incorrect", "partial", "almost"]:
                                prediction = resp
                                break
                        # Check other common fields
                        for key in ["label", "grade", "evaluation", "classification"]:
                            val = json_obj.get(key, "")
                            if isinstance(val, str):
                                val = val.strip().lower()
                                if val in ["correct", "incorrect", "partial", "almost"]:
                                    prediction = val
                                    break
                        if prediction != "None":
                            break
            
            # Strategy 2: Direct text extraction as fallback
            if prediction == "None":
                text_pred = _extract_label_from_text(raw_text)
                if text_pred:
                    prediction = text_pred
                    
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        self.log_fn(f"Final prediction: {prediction}")
        return str(prediction), msg_history
