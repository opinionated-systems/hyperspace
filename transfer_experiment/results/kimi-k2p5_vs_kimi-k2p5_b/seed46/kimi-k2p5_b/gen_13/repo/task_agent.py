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
            # Try to extract JSON object boundaries
            try:
                json_start = inner.find("{")
                json_end = inner.rfind("}")
                if json_start != -1 and json_end != -1 and json_end > json_start:
                    results.append(json.loads(inner[json_start:json_end+1]))
            except json.JSONDecodeError:
                continue
    return results or None


def _extract_label_from_text(text: str) -> str | None:
    """Extract grading label from raw text."""
    text_lower = text.lower()
    
    # Priority order: check for more specific patterns first
    # Pattern: key: "value" or key: 'value' or key: value
    patterns = [
        r'["\']?response["\']?\s*[:=]\s*["\']?(correct|incorrect|partial|almost)["\']?',
        r'["\']?label["\']?\s*[:=]\s*["\']?(correct|incorrect|partial|almost)["\']?',
        r'["\']?grade["\']?\s*[:=]\s*["\']?(correct|incorrect|partial|almost)["\']?',
        r'["\']?evaluation["\']?\s*[:=]\s*["\']?(correct|incorrect|partial|almost)["\']?',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            return match.group(1)
    
    # Check for standalone labels with word boundaries
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
        
        instruction = f"""You are an expert mathematics grader for International Mathematical Olympiad (IMO) problems. Evaluate the student's answer and assign one of four grades.

GRADE DEFINITIONS:
- "correct": Fully correct and complete solution (90-100%). Valid proof with all necessary steps.
- "almost": Nearly correct with minor gaps only (70-89%). Right approach, small technical issues.
- "partial": Some correct insights but significant gaps (30-69%). Meaningful progress but incomplete.
- "incorrect": Fundamentally wrong or no valid progress (0-29%). Invalid reasoning.

GRADING NOTES:
- Be conservative with "correct" - only for essentially complete proofs.
- "Almost" = 6-7/7 marks, "Partial" = 2-5/7 marks, "Incorrect" = 0-1/7 marks.
- When in doubt between "almost" and "partial", choose "partial" if there are significant gaps.

Problem:
{problem}

Official Solution:
{solution}

Grading Guidelines:
{grading_guidelines}

Student's Answer:
{student_answer}

Analyze the student's answer and respond with ONLY a JSON object in <json> tags:

<json>
{{
    "reasoning": "Your detailed analysis of strengths and weaknesses",
    "response": "correct"
}}
</json>

The "response" field MUST be exactly one of: "correct", "almost", "partial", or "incorrect" (all lowercase)."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction
        prediction = "None"
        raw_text = msg_history[-1]["text"] if msg_history else ""
        
        try:
            # Strategy 1: Extract from <json> tags
            extracted = _extract_jsons(raw_text)
            if extracted:
                for json_obj in extracted:
                    if isinstance(json_obj, dict):
                        for key in ["response", "label", "grade", "evaluation"]:
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
