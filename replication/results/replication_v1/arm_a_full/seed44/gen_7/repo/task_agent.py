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
    """Task agent that solves IMO grading problems with structured reasoning."""

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

        prompt = f"""You are an expert {domain} grader for the International Mathematical Olympiad (IMO).

Your task is to evaluate a student's answer to a competition problem.

## Problem Statement
{problem}

## Official Solution
{solution}

## Grading Guidelines
{grading_guidelines}

## Student's Answer
{student_answer}

## Evaluation Instructions

Please evaluate the student's answer following these steps:

1. **Understand the Problem**: Identify what the problem is asking and the key mathematical concepts involved.

2. **Analyze the Official Solution**: Note the key steps, techniques, and the final answer.

3. **Review Grading Guidelines**: Understand the scoring rubric and partial credit rules.

4. **Evaluate Student's Answer**: 
   - Check if the final answer is correct
   - Assess the reasoning and proof techniques used
   - Identify any gaps or errors in the logic
   - Determine if partial credit is warranted

5. **Assign a Score**: Based on the grading guidelines, assign an appropriate score.

## Response Format

Respond in JSON format with the following schema:

<json>
{{
    "reasoning": "Your detailed step-by-step analysis and reasoning process",
    "evaluation": {{
        "answer_correct": true/false,
        "proof_complete": true/false,
        "key_strengths": ["strength 1", "strength 2"],
        "key_weaknesses": ["weakness 1", "weakness 2"]
    }},
    "score": "The numerical score (e.g., 7, 3, 0, etc.)",
    "response": "The final score as a number or string"
}}
</json>

The "response" field should contain the final score that will be used for evaluation."""

        return prompt

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        instruction = self._build_grading_prompt(inputs)

        self.log_fn(f"TaskAgent: Processing problem with domain: {inputs.get('domain', 'Unknown')}")

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON
        prediction = "None"
        reasoning = ""
        evaluation_details = {}

        try:
            extracted = _extract_jsons(msg_history[-1]["text"])
            if extracted:
                data = extracted[-1]
                if "response" in data:
                    prediction = data["response"]
                elif "score" in data:
                    prediction = data["score"]
                
                # Extract additional metadata if available
                reasoning = data.get("reasoning", "")
                evaluation_details = data.get("evaluation", {})
                
                self.log_fn(f"TaskAgent: Extracted prediction: {prediction}")
                if reasoning:
                    self.log_fn(f"TaskAgent: Reasoning length: {len(reasoning)} chars")
        except json.JSONDecodeError as e:
            self.log_fn(f"TaskAgent: JSON decode error: {e}")
        except Exception as e:
            self.log_fn(f"TaskAgent: Error extracting prediction: {e}")

        return str(prediction), msg_history
