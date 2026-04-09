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
    """Extract JSON objects from <json>...</json> blocks or raw JSON.
    
    Uses multiple strategies to find JSON content.
    """
    results = []
    
    # Strategy 1: Look for <json>...</json> tags
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
    
    # Strategy 2: Look for JSON objects in code blocks
    json_block_pattern = re.findall(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    for block in json_block_pattern:
        try:
            results.append(json.loads(block.strip()))
        except json.JSONDecodeError:
            continue
    
    # Strategy 3: Look for raw JSON objects with grade/response/label fields
    raw_json_pattern = re.findall(r'\{[^{}]*"(?:grade|response|label|evaluation)"[^{}]*\}', text, re.DOTALL)
    for match in raw_json_pattern:
        try:
            results.append(json.loads(match))
        except json.JSONDecodeError:
            # Try to fix common JSON issues
            try:
                # Replace single quotes with double quotes
                fixed = match.replace("'", '"')
                results.append(json.loads(fixed))
            except json.JSONDecodeError:
                continue
    
    return results or None


def _extract_label_from_text(text: str) -> str | None:
    """Extract grading label from raw text using multiple strategies."""
    text_lower = text.lower()
    
    # Look for explicit label mentions with word boundaries
    # Check for "incorrect" first (most specific - contains "correct")
    if re.search(r'\bin\s*correct\b|\bincorrect\b', text_lower):
        return "incorrect"
    
    # Check for "partial"
    if re.search(r'\bpartial\b', text_lower):
        return "partial"
    
    # Check for "correct" last (least specific)
    if re.search(r'\bcorrect\b', text_lower):
        return "correct"
    
    # Look for grade assignments like "grade: correct" or "grade = incorrect"
    grade_assignment = re.search(r'grade["\']?\s*[:=]\s*["\']?(\w+)', text_lower)
    if grade_assignment:
        val = grade_assignment.group(1)
        if val in VALID_LABELS:
            return val
    
    return None


def _normalize_grade(value: str) -> str | None:
    """Normalize a grade value to one of the valid labels."""
    if not isinstance(value, str):
        value = str(value)
    value = value.lower().strip()
    
    # Direct match
    if value in VALID_LABELS:
        return value
    
    # Check for partial matches
    if "incorrect" in value:
        return "incorrect"
    if "partial" in value:
        return "partial"
    if "correct" in value:
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

IMPORTANT: You must respond with ONLY a JSON object in <json> tags. Do NOT include any other text before or after the JSON.

<json>
{{
    "reasoning": "Brief explanation of your evaluation",
    "grade": "correct" | "partial" | "incorrect"
}}
</json>

The "grade" field MUST be exactly one of: "correct", "partial", or "incorrect" (all lowercase)."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction using multiple strategies
        prediction = "None"
        try:
            raw_text = msg_history[-1]["text"] if msg_history else ""
            
            # Strategy 1: Try to extract from JSON tags and other JSON patterns
            extracted = _extract_jsons(raw_text)
            if extracted:
                for json_obj in extracted:
                    for key in ["grade", "response", "label", "evaluation"]:
                        if key in json_obj:
                            val = json_obj[key]
                            normalized = _normalize_grade(val)
                            if normalized:
                                prediction = normalized
                                break
                    if prediction != "None":
                        break
            
            # Strategy 2: If JSON extraction failed, try text extraction
            if prediction == "None":
                text_pred = _extract_label_from_text(raw_text)
                if text_pred:
                    prediction = text_pred
            
            # Strategy 3: Look for any of the valid labels in the text
            if prediction == "None":
                text_lower = raw_text.lower()
                for label in VALID_LABELS:
                    if label in text_lower:
                        prediction = label
                        break
                    
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
