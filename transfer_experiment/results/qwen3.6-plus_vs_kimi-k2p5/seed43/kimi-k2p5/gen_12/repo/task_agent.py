"""
Task agent: solves a given task with a single LLM call.

Reimplemented from facebookresearch/HyperAgents task_agent.py.
Same interface, same JSON output format, same extraction logic.

This is the OPTIMIZED task agent with simplified prompting and robust extraction.
The meta agent modifies this file during self-improvement. The evaluation harness 
loads whatever task_agent.py exists at the agent's repo path.
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


def _extract_json_flexible(text: str) -> dict | None:
    """Extract JSON using multiple fallback methods."""
    # Try <json> tags first
    results = _extract_jsons(text)
    if results:
        return results[-1]
    
    # Try ```json code blocks
    json_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    matches = re.findall(json_block_pattern, text, re.DOTALL)
    for match in matches:
        try:
            return json.loads(match.strip())
        except json.JSONDecodeError:
            continue
    
    # Try finding raw JSON objects with curly braces
    brace_count = 0
    start_idx = -1
    for i, char in enumerate(text):
        if char == '{':
            if brace_count == 0:
                start_idx = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start_idx != -1:
                try:
                    return json.loads(text[start_idx:i+1])
                except json.JSONDecodeError:
                    continue
    
    return None


def _extract_grade_from_text(text: str) -> str | None:
    """Extract grade from raw text using pattern matching."""
    text_lower = text.lower()
    
    # Look for explicit grade patterns
    patterns = [
        r'"response"\s*:\s*"?(correct|incorrect|partial)"?',
        r'grade\s*[:=]\s*"?(correct|incorrect|partial)"?',
        r'\bthe answer is\s+(correct|incorrect|partial)\b',
        r'\bthe grade is\s+(correct|incorrect|partial)\b',
        r'\b(correct|incorrect|partial)\b',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            grade = match.group(1).capitalize()
            if grade in ["Correct", "Incorrect", "Partial"]:
                return grade
    
    return None


class TaskAgent:
    """Task agent that solves IMO grading problems."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem."""
        # Extract fields with defaults
        domain = inputs.get('domain', 'Mathematics')
        problem = inputs.get('problem', '')
        solution = inputs.get('solution', '')
        grading_guidelines = inputs.get('grading_guidelines', '')
        student_answer = inputs.get('student_answer', '')
        
        # Build simplified, focused instruction
        instruction = f"""You are an expert grader for {domain} problems.

Evaluate the student's answer and assign exactly one grade: "Correct", "Incorrect", or "Partial".

PROBLEM:
{problem}

OFFICIAL SOLUTION:
{solution}

GRADING GUIDELINES:
{grading_guidelines}

STUDENT'S ANSWER:
{student_answer}

Respond with ONLY this JSON format:
<json>
{{
    "response": "Correct" or "Incorrect" or "Partial"
}}
</json>"""

        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
        except Exception as e:
            self.log_fn(f"LLM call failed: {e}")
            return "Error", []

        # Extract prediction
        prediction = "None"
        try:
            if msg_history:
                last_message = msg_history[-1]["text"]
                extracted = _extract_json_flexible(last_message)
                
                if extracted and "response" in extracted:
                    prediction = extracted["response"]
                else:
                    # Fallback to text extraction
                    text_grade = _extract_grade_from_text(last_message)
                    if text_grade:
                        prediction = text_grade
        except Exception as e:
            self.log_fn(f"Extraction error: {e}")

        # Normalize prediction
        valid_grades = ["Correct", "Incorrect", "Partial"]
        pred_clean = str(prediction).strip().lower().replace('"', '').replace("'", "")
        
        for grade in valid_grades:
            if pred_clean == grade.lower():
                prediction = grade
                break
        else:
            # Check for substring match
            for grade in ["incorrect", "partial", "correct"]:
                if grade in pred_clean:
                    prediction = grade.capitalize()
                    break
            else:
                prediction = "None"

        return str(prediction), msg_history
