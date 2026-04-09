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
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def _build_grading_prompt(self, inputs: dict) -> str:
        """Build a structured prompt for IMO grading with chain-of-thought reasoning."""
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        return f"""You are an expert {domain} grader for the International Mathematical Olympiad (IMO).

Your task is to evaluate a student's solution to a problem and assign a score from 0 to 7 points (IMO scoring).

## Problem Statement:
{problem}

## Official Solution:
{solution}

## Grading Guidelines:
{grading_guidelines}

## Student's Answer:
{student_answer}

## Evaluation Instructions:

1. **Analyze the Problem**: Understand what the problem is asking and what constitutes a complete solution.

2. **Review the Official Solution**: Note the key steps, techniques, and insights required.

3. **Evaluate the Student's Answer**: 
   - Check if the student understood the problem correctly
   - Identify which key steps the student completed
   - Note any errors, gaps, or incorrect reasoning
   - Check for partial progress that deserves partial credit

4. **Assign Score (0-7)**:
   - 7: Complete, correct solution
   - 6: Minor flaw in an otherwise correct solution
   - 5-3: Partial progress with varying degrees of completeness
   - 2-1: Significant progress but major gaps
   - 0: No meaningful progress or completely wrong

5. **Provide Reasoning**: Explain your scoring decision with specific references to the student's work.

Respond in JSON format with the following schema:
<json>
{{
    "thinking": "Your detailed chain-of-thought analysis here...",
    "score": 7,
    "reasoning": "Explanation of why this score was assigned, referencing specific parts of the student's solution"
}}
</json>

The "score" field must be an integer from 0 to 7."""

    def _extract_score(self, data: dict) -> str:
        """Extract the score from the JSON response, with validation."""
        # Try to get score directly
        if "score" in data:
            score = data["score"]
            # Validate it's an integer 0-7
            try:
                score_int = int(score)
                if 0 <= score_int <= 7:
                    return str(score_int)
            except (ValueError, TypeError):
                pass
        
        # Fallback: try to extract from response field for backward compatibility
        if "response" in data:
            return str(data["response"])
        
        # Try to find any numeric field that might be the score
        for key in ["points", "grade", "mark", "rating"]:
            if key in data:
                return str(data[key])
        
        return "None"

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        instruction = self._build_grading_prompt(inputs)

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON
        prediction = "None"
        try:
            # Try to extract from the last assistant message
            last_msg = msg_history[-1]["text"] if msg_history else ""
            extracted = _extract_jsons(last_msg)
            
            if extracted:
                prediction = self._extract_score(extracted[-1])
                self.log_fn(f"Extracted score: {prediction}")
            else:
                # Fallback: try to find any number 0-7 in the response
                numbers = re.findall(r'\b([0-7])\b', last_msg)
                if numbers:
                    prediction = numbers[-1]  # Take the last number found
                    self.log_fn(f"Fallback extraction: {prediction}")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
