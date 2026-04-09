"""
Task agent: solves a given task with self-consistency and improved prompting.

Reimplemented from facebookresearch/HyperAgents task_agent.py.
Enhanced with few-shot examples and self-consistency voting for better accuracy.

This is the INITIAL task agent. The meta agent modifies this file
during self-improvement. The evaluation harness loads whatever
task_agent.py exists at the agent's repo path.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter

from agent.llm_client import get_response_from_llm, EVAL_MODEL

logger = logging.getLogger(__name__)

# Few-shot examples for better grading accuracy
FEW_SHOT_EXAMPLES = """
## Example 1:
Problem: Find the sum of 2 + 3.
Official Solution: The sum is 5.
Grading Guidelines: Award 1 point for correct answer (5), 0 otherwise.
Student's Answer: 5
Grade: correct

## Example 2:
Problem: Prove that the sum of angles in a triangle is 180 degrees.
Official Solution: Using parallel lines and alternate interior angles, we can show the three angles form a straight line, thus sum to 180°.
Grading Guidelines: Award 1 point for complete proof with clear reasoning. Award 0.5 for partial proof with gaps. Award 0 for incorrect or no attempt.
Student's Answer: The angles add up to 180 because that's a property of triangles.
Grade: partial

## Example 3:
Problem: Solve x^2 - 4 = 0.
Official Solution: x^2 = 4, so x = ±2.
Grading Guidelines: Award 1 point for both solutions (2 and -2). Award 0.5 for one correct solution. Award 0 for incorrect answer.
Student's Answer: x = 2
Grade: partial

## Example 4:
Problem: What is the derivative of x^2?
Official Solution: The derivative is 2x.
Grading Guidelines: Award 1 point for correct answer (2x), 0 otherwise.
Student's Answer: 3x
Grade: incorrect
"""


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    Also handles markdown code blocks and plain JSON.
    """
    results = []
    search_from = 0
    
    # First try to find <json>...</json> blocks
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
    
    # Also try markdown code blocks ```json ... ```
    search_from = 0
    while True:
        start = text.find("```json", search_from)
        if start == -1:
            break
        end = text.find("```", start + 7)
        if end == -1:
            break
        inner = text[start + 7:end].strip()
        search_from = end + 3
        try:
            results.append(json.loads(inner))
        except json.JSONDecodeError:
            continue
    
    # Try to find plain JSON objects (for robustness)
    if not results:
        try:
            # Look for JSON-like structures with braces
            brace_start = text.find("{")
            while brace_start != -1:
                # Try to find matching closing brace
                brace_count = 0
                for i, char in enumerate(text[brace_start:]):
                    if char == "{":
                        brace_count += 1
                    elif char == "}":
                        brace_count -= 1
                        if brace_count == 0:
                            try:
                                json_obj = json.loads(text[brace_start:brace_start + i + 1])
                                results.append(json_obj)
                            except json.JSONDecodeError:
                                pass
                            break
                brace_start = text.find("{", brace_start + 1)
        except Exception:
            pass
    
    return results or None


def _normalize_grade(grade: str) -> str:
    """Normalize grade to standard format."""
    if not grade:
        return "incorrect"
    
    grade_lower = str(grade).lower().strip()
    
    # Map various formats to standard labels
    if any(x in grade_lower for x in ["correct", "right", "1", "full", "true", "yes"]):
        if "partial" not in grade_lower and "almost" not in grade_lower:
            return "correct"
    
    if any(x in grade_lower for x in ["partial", "almost", "0.5", "half", "incomplete"]):
        return "partial"
    
    if any(x in grade_lower for x in ["incorrect", "wrong", "0", "false", "no", "none", "error"]):
        return "incorrect"
    
    # Try to extract numeric score
    try:
        num = float(grade)
        if num >= 0.9:
            return "correct"
        elif num >= 0.4:
            return "partial"
        else:
            return "incorrect"
    except (ValueError, TypeError):
        pass
    
    return "incorrect"


def _extract_prediction_from_json(json_obj: dict) -> str:
    """Extract prediction from JSON object."""
    # Try common keys in order of preference
    for key in ["response", "grade", "answer", "assessment", "evaluation", "result", "prediction"]:
        if key in json_obj:
            return str(json_obj[key])
    
    # If no known key found, return first string value
    for key, value in json_obj.items():
        if isinstance(value, str):
            return value
    
    return "None"


class TaskAgent:
    """Task agent that solves IMO grading problems with self-consistency."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.num_samples = 3  # Number of samples for self-consistency

    def _single_call(self, instruction: str) -> tuple[str, list[dict]]:
        """Make a single LLM call and extract prediction."""
        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON
        prediction = "None"
        try:
            extracted = _extract_jsons(msg_history[-1]["text"])
            if extracted:
                # Try to get response from the last valid JSON
                last_json = extracted[-1]
                prediction = _extract_prediction_from_json(last_json)
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with self-consistency.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Extract fields with defaults for robustness
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        instruction = f"""You are an expert grader for {domain} problems. Your task is to evaluate a student's answer based on the problem, official solution, and grading guidelines.

{FEW_SHOT_EXAMPLES}

## Now grade this problem:

## Problem:
{problem}

## Official Solution:
{solution}

## Grading Guidelines:
{grading_guidelines}

## Student's Answer:
{student_answer}

## Your Task:
1. Carefully analyze the student's answer step by step
2. Compare it against the official solution and grading guidelines
3. Determine the appropriate grade: "correct" (full credit), "partial" (partial credit), or "incorrect" (no credit)
4. Provide detailed reasoning for your assessment

Respond in JSON format with the following schema:
<json>
{{
    "reasoning": "Your step-by-step analysis here, explaining what the student got right and wrong",
    "response": "Your final grade here - must be exactly one of: 'correct', 'partial', or 'incorrect'"
}}
</json>"""

        # Self-consistency: make multiple calls and vote
        predictions = []
        all_histories = []
        
        for i in range(self.num_samples):
            try:
                prediction, msg_history = self._single_call(instruction)
                normalized = _normalize_grade(prediction)
                predictions.append(normalized)
                all_histories.extend(msg_history)
                self.log_fn(f"Sample {i+1}: raw='{prediction}' -> normalized='{normalized}'")
            except Exception as e:
                self.log_fn(f"Sample {i+1} failed: {e}")
                predictions.append("incorrect")  # Default to incorrect on error
        
        # Majority voting
        if predictions:
            vote_counts = Counter(predictions)
            final_prediction = vote_counts.most_common(1)[0][0]
            self.log_fn(f"Self-consistency votes: {dict(vote_counts)} -> final: '{final_prediction}'")
        else:
            final_prediction = "incorrect"
        
        return str(final_prediction), all_histories
