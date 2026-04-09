"""
Task agent: solves a given task with chain-of-thought reasoning.

Reimplemented from facebookresearch/HyperAgents task_agent.py.
Enhanced with structured reasoning, few-shot examples, and self-correction.

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


# Few-shot examples for IMO grading
FEW_SHOT_EXAMPLES = """
Example 1:
Problem: Find the sum of 2 + 3.
Solution: The sum is 5.
Student answer: 5
Analysis: The student's answer matches the correct solution exactly.
Grade: {"correct": true, "explanation": "Correct answer"}

Example 2:
Problem: Solve x^2 = 4.
Solution: x = 2 or x = -2.
Student answer: x = 2
Analysis: The student found one correct solution but missed the negative root.
Grade: {"correct": false, "explanation": "Incomplete answer - missing x = -2"}
"""


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.max_retries = 2

    def _build_prompt(self, inputs: dict) -> str:
        """Build a structured prompt with chain-of-thought instructions."""
        domain = inputs.get("domain", "mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        return f"""You are an expert grader for {domain} problems.

Your task is to evaluate a student's answer by comparing it to the correct solution.

{FEW_SHOT_EXAMPLES}

Now evaluate this problem:

PROBLEM:
{problem}

CORRECT SOLUTION:
{solution}

GRADING GUIDELINES:
{grading_guidelines}

STUDENT'S ANSWER:
{student_answer}

Follow this reasoning process:
1. Understand what the problem is asking
2. Review the correct solution step by step
3. Analyze the student's answer against the solution
4. Check for partial credit according to guidelines
5. Make a final determination

Respond in JSON format with the following schema:
<json>
{{
    "reasoning": "Your step-by-step analysis here",
    "correct": true/false,
    "partial_credit": 0.0 to 1.0,
    "response": "Your final grade/explanation here"
}}
</json>"""

    def _extract_prediction(self, msg_history: list[dict]) -> str:
        """Extract prediction from message history with error handling."""
        if not msg_history:
            return "None"
        
        last_msg = msg_history[-1]
        text = last_msg.get("text", "")
        
        if not text:
            return "None"

        try:
            extracted = _extract_jsons(text)
            if not extracted:
                return "None"
            
            result = extracted[-1]
            
            # Prefer the structured response field
            if "response" in result:
                return str(result["response"])
            
            # Fallback: construct response from other fields
            if "correct" in result:
                correctness = "correct" if result["correct"] else "incorrect"
                explanation = result.get("explanation", result.get("reasoning", ""))
                return f"{correctness}: {explanation}"
            
            return str(result)
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            return "None"

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with self-correction.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        instruction = self._build_prompt(inputs)
        msg_history = []
        
        # Primary attempt
        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=msg_history,
        )

        prediction = self._extract_prediction(msg_history)
        
        # Self-correction if extraction failed
        retry_count = 0
        while prediction == "None" and retry_count < self.max_retries:
            self.log_fn(f"Retry {retry_count + 1}: Failed to extract valid JSON, attempting correction")
            
            correction_msg = """Your previous response did not contain valid JSON or was missing required fields.

Please respond again using the exact JSON format specified:
<json>
{
    "reasoning": "Your step-by-step analysis here",
    "correct": true/false,
    "partial_credit": 0.0 to 1.0,
    "response": "Your final grade/explanation here"
}
</json>"""
            
            response, msg_history, info = get_response_from_llm(
                msg=correction_msg,
                model=self.model,
                msg_history=msg_history,
            )
            
            prediction = self._extract_prediction(msg_history)
            retry_count += 1

        return str(prediction), msg_history
