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


def _extract_json_with_regex(text: str) -> list[dict] | None:
    """Fallback JSON extraction using regex for malformed outputs."""
    results = []
    # Try to find JSON objects in code blocks
    code_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    matches = re.findall(code_block_pattern, text, re.DOTALL)
    for match in matches:
        try:
            results.append(json.loads(match.strip()))
        except json.JSONDecodeError:
            continue
    
    # Try to find JSON objects with curly braces
    if not results:
        json_pattern = r'\{[^{}]*"response"[^{}]*\}'
        matches = re.findall(json_pattern, text)
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
        # Extract key fields for better prompt construction
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = f"""You are an expert mathematical grader evaluating student solutions to competition mathematics problems.

Your task is to grade a student's answer to an IMO-level mathematics problem with precision and consistency. You must carefully analyze all provided materials before making any grading decision.

PROBLEM DOMAIN: {domain}

PROBLEM STATEMENT:
```
{problem}
```

OFFICIAL SOLUTION:
```
{solution}
```

GRADING GUIDELINES (RUBRIC):
```
{grading_guidelines}
```

STUDENT'S ANSWER:
```
{student_answer}
```

GRADING PROTOCOL - Follow these steps in order:

STEP 1 - PROBLEM ANALYSIS:
- Identify the key mathematical concepts and techniques required
- Note the critical steps that must be present for a complete solution
- Understand what constitutes a correct proof/answer

STEP 2 - SOLUTION MAPPING:
- Break down the official solution into logical steps
- Identify which steps are essential vs. optional
- Note alternative valid approaches that could also solve the problem

STEP 3 - RUBRIC INTERPRETATION:
- Parse the grading guidelines carefully
- Identify point allocations for each component
- Note partial credit conditions explicitly

STEP 4 - STUDENT WORK ANALYSIS:
- Read the student's answer completely before judging
- Map their solution steps against the official solution
- Check for:
  * Correct problem interpretation
  * Valid approach selection
  * Mathematical accuracy in calculations
  * Logical flow and rigor in proofs
  * Completeness of the solution
- Identify any errors: conceptual, computational, or logical
- Note any creative or alternative valid approaches

STEP 5 - SCORE DETERMINATION:
- Start from zero and award points for correct elements
- OR start from full credit and deduct for errors (per rubric)
- Ensure the score aligns precisely with the grading guidelines
- Double-check: would another expert grader reach the same conclusion?

STEP 6 - FINAL VERIFICATION:
- Verify your score matches the rubric's specifications exactly
- Confirm you haven't missed any partial credit opportunities
- Ensure numerical scores are formatted as numbers, not strings

Respond ONLY in the following JSON format:
<json>
{{
    "response": <your numerical score or evaluation>
}}
</json>

The "response" field must contain your final grading decision (typically a number or specific evaluation string as specified in the grading guidelines). Be precise and objective in your assessment."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON with multiple fallback strategies
        prediction = "None"
        extraction_attempts = [
            lambda: _extract_jsons(msg_history[-1]["text"]),
            lambda: _extract_json_with_regex(msg_history[-1]["text"]),
        ]
        
        for attempt_fn in extraction_attempts:
            try:
                extracted = attempt_fn()
                if extracted and "response" in extracted[-1]:
                    prediction = extracted[-1]["response"]
                    break
            except Exception as e:
                self.log_fn(f"Extraction attempt failed: {e}")
                continue
        
        if prediction == "None":
            self.log_fn(f"Failed to extract prediction from response. Raw text: {msg_history[-1]['text'][:500]}")

        return str(prediction), msg_history
