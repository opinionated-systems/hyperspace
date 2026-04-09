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


def _extract_json_from_markdown(text: str) -> list[dict] | None:
    """Extract JSON objects from markdown code blocks."""
    results = []
    # Match ```json ... ``` or just ``` ... ``` blocks
    pattern = r'```(?:json)?\s*(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    for match in matches:
        try:
            results.append(json.loads(match.strip()))
        except json.JSONDecodeError:
            continue
    return results or None


def _extract_any_json(text: str) -> list[dict] | None:
    """Extract JSON objects using multiple strategies."""
    # Try <json> tags first
    results = _extract_jsons(text)
    if results:
        return results
    
    # Try markdown code blocks
    results = _extract_json_from_markdown(text)
    if results:
        return results
    
    # Try to find JSON objects directly
    results = []
    # Look for patterns that look like JSON objects
    pattern = r'\{[^{}]*"[^"]+"[^{}]*\}'
    matches = re.findall(pattern, text)
    for match in matches:
        try:
            results.append(json.loads(match))
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
        # Build a more detailed prompt that encourages structured output
        problem = inputs.get('problem', '')
        solution = inputs.get('solution', '')
        grading_guidelines = inputs.get('grading_guidelines', '')
        student_answer = inputs.get('student_answer', '')
        
        instruction = f"""You are an expert mathematical grader for IMO-style problems. Your task is to evaluate a student's answer and assign exactly one of four grades: Correct, Incorrect, Partial, or Almost.

PROBLEM:
{problem}

OFFICIAL SOLUTION:
{solution}

GRADING GUIDELINES:
{grading_guidelines}

STUDENT'S ANSWER:
{student_answer}

GRADE DEFINITIONS:
- Correct: The student's answer is fully correct, complete, and matches the official solution. All key steps and reasoning are present and valid.
- Incorrect: The student's answer is fundamentally wrong, missing critical insights, or based on incorrect reasoning. No significant progress toward the solution.
- Partial: The student made meaningful progress (e.g., found a key invariant, set up the right framework, proved a lemma) but the solution is incomplete or has significant gaps. The student did NOT complete the full proof or verification.
- Almost: The student's solution is nearly complete with only minor, negligible mistakes (e.g., small calculation errors, typos, or slight oversights that don't affect the main argument). The core proof structure and reasoning are correct.

IMPORTANT DISTINCTIONS:
- Partial vs Almost: "Partial" means significant work is missing (e.g., only found an invariant but didn't prove sufficiency). "Almost" means the solution is essentially complete with only trivial errors.
- Partial vs Correct: "Partial" is missing key components of the full solution. "Correct" must be complete and fully verified.
- Almost vs Incorrect: "Almost" has the right approach and nearly complete proof. "Incorrect" has fundamental flaws or wrong approach.

Analyze the student's answer step by step:
1. What key elements from the official solution are present?
2. What is missing or incorrect?
3. Based on the grading guidelines, which criteria are satisfied?
4. Which grade best matches based on the definitions above?

Respond in JSON format:
<json>
{{
    "response": "Correct" or "Incorrect" or "Partial" or "Almost"
}}
</json>

You must choose exactly one of these four labels. Be conservative: only assign "Correct" if fully complete, "Almost" if nearly complete with minor issues, "Partial" if significant progress but incomplete, and "Incorrect" if fundamentally wrong.
"""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON using multiple strategies
        prediction = "None"
        try:
            extracted = _extract_any_json(response)
            if extracted:
                # Try to find a response field
                for item in extracted:
                    if isinstance(item, dict):
                        if "response" in item:
                            prediction = item["response"]
                            break
                        # Also check for common alternative field names
                        for key in ["grade", "label", "result", "evaluation", "answer"]:
                            if key in item:
                                prediction = item[key]
                                break
                        if prediction != "None":
                            break
            
            # If no JSON found, try to extract a simple label from the text
            if prediction == "None":
                # Look for common grade labels in the response
                text_lower = response.lower()
                # Check for exact matches first
                if "grade: correct" in text_lower or '"correct"' in text_lower:
                    prediction = "correct"
                elif "grade: incorrect" in text_lower or '"incorrect"' in text_lower:
                    prediction = "incorrect"
                elif "grade: partial" in text_lower or '"partial"' in text_lower:
                    prediction = "partial"
                elif "grade: almost" in text_lower or '"almost"' in text_lower:
                    prediction = "almost"
                elif "correct" in text_lower and "incorrect" not in text_lower:
                    prediction = "correct"
                elif "incorrect" in text_lower or "wrong" in text_lower:
                    prediction = "incorrect"
                elif "partial" in text_lower:
                    prediction = "partial"
                elif "almost" in text_lower:
                    prediction = "almost"
                else:
                    # Use the first line of the response as prediction
                    first_line = response.strip().split('\n')[0][:100]
                    if first_line:
                        prediction = first_line
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
