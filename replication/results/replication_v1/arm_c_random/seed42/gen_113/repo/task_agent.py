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
        # Extract fields from inputs for better prompt construction
        domain = inputs.get("domain", "")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        instruction = f"""You are an expert mathematical grader evaluating student solutions to competition mathematics problems (IMO, USAMO, Putnam, etc.).

Your task is to analyze the student's answer and determine the correct grade based on the provided solution and grading guidelines.

## Problem Domain
{domain}

## Problem Statement
{problem}

## Official Solution
{solution}

## Grading Guidelines
{grading_guidelines}

## Student's Answer
{student_answer}

## Instructions

1. **Read the problem carefully** - Understand what is being asked and what constitutes a complete solution. Identify key requirements and constraints.

2. **Study the official solution** - Note the intended approach, key insights, critical steps, and the level of rigor expected.

3. **Review the grading guidelines** - These specify how points are allocated and what constitutes each grade level:
   - "Correct" (7/7): Complete, rigorous solution with all key steps properly justified and no significant errors
   - "Almost" (6/7 or partial): Solution is nearly complete with the right approach, but has minor, non-negligible mistakes (e.g., small calculation errors, missing edge case, slight lack of rigor in one step)
   - "Partial" (1-5/7): Contains some correct observations or partial progress toward the solution but is incomplete or has significant gaps
   - "Incorrect" (0/7): No meaningful progress toward the solution, or fundamentally flawed approach

4. **Analyze the student's answer systematically**:
   - **Approach**: Did the student identify the correct strategy/method?
   - **Key insights**: Did the student identify the critical insights needed?
   - **Execution**: Are the calculations and logical steps correct?
   - **Rigor**: Are claims properly justified with appropriate reasoning?
   - **Completeness**: Does the solution address all parts of the problem?
   - **Common errors**: Check for common mistakes like off-by-one errors, missing cases, circular reasoning, etc.

5. **Compare against the rubric**:
   - Award points for each correct insight or step completed
   - Deduct points for unjustified claims or logical gaps
   - Consider partial credit for meaningful progress even if incomplete

6. **Provide your reasoning** - Explain your analysis step by step, citing specific evidence from the student's answer. Be explicit about what they got right and what was wrong.

7. **Output the final grade** in the JSON format below. Be conservative - only award "Correct" for truly complete solutions.

Respond in JSON format with the following schema:
<json>
{{
    "reasoning": "Your detailed analysis of the student's answer, comparing to the solution and explaining the grade. Cite specific evidence.",
    "response": "One of: Correct, Almost, Partial, or Incorrect"
}}
</json>"""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON with improved error handling
        prediction = "None"
        try:
            extracted = _extract_jsons(msg_history[-1]["text"])
            if extracted:
                # Try to find a valid response in any of the extracted JSONs
                for item in reversed(extracted):
                    if "response" in item:
                        prediction = item["response"]
                        break
                    elif "grade" in item:
                        prediction = item["grade"]
                        break
                # Validate the prediction is one of the expected values
                valid_grades = ["Correct", "Almost", "Partial", "Incorrect"]
                if prediction not in valid_grades:
                    self.log_fn(f"Warning: Unexpected grade value '{prediction}', defaulting to Partial")
                    prediction = "Partial"
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            prediction = "Partial"  # Default to Partial on error rather than None

        return str(prediction), msg_history
