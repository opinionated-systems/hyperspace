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
    Also handles markdown code blocks as fallback.
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
    
    # Fallback: try to find JSON in markdown code blocks
    if not results:
        json_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
        matches = re.findall(json_pattern, text, re.DOTALL)
        for match in matches:
            try:
                results.append(json.loads(match.strip()))
            except json.JSONDecodeError:
                continue
    
    # Second fallback: try to find any JSON object in the text
    if not results:
        # Look for patterns like {"key": "value"}
        brace_pattern = r'\{[^{}]*"[^"]*"[^{}]*\}'
        matches = re.findall(brace_pattern, text)
        for match in matches:
            try:
                results.append(json.loads(match))
            except json.JSONDecodeError:
                continue
    
    return results or None


def _normalize_score(score: str) -> str:
    """Normalize score to a standard format.
    
    Handles various score formats and converts them to a clean string.
    """
    if score is None:
        return "None"
    
    score_str = str(score).strip()
    
    # Remove any explanatory text after the score
    # e.g., "7 (full marks)" -> "7"
    score_str = re.split(r'\s+\(|\s+-|\s*:', score_str)[0].strip()
    
    # Extract just the numeric part if there's extra text
    numeric_match = re.search(r'^\d+', score_str)
    if numeric_match:
        return numeric_match.group()
    
    return score_str if score_str else "None"


class TaskAgent:
    """Task agent that solves IMO grading problems with structured reasoning."""

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
        # Extract fields for better structured prompting
        domain = inputs.get("domain", "")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = f"""You are an expert mathematical grader for the International Mathematical Olympiad (IMO).

Your task is to evaluate a student's answer to a mathematical problem and assign an appropriate score based on the official solution and grading guidelines.

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

## Your Task

1. **Analyze the student's answer**: Compare it against the official solution and grading guidelines.
2. **Identify key steps**: Check which steps from the official solution are present in the student's answer.
3. **Evaluate correctness**: Determine if the student's reasoning is valid and complete.
4. **Assign a score**: Based on the grading guidelines, assign the appropriate score.

## Scoring Guidelines

- Read the grading guidelines carefully - they specify exact point allocations
- Award points only for correct mathematical reasoning that leads toward the solution
- Partial credit is given for significant progress even if the final answer is incorrect
- No credit is given for:
  * Incorrect statements or calculations
  * Vague or incomplete arguments
  * Answers that don't address the problem
  * Plagiarized or irrelevant content

Respond in JSON format with the following schema:
<json>
{{
    "reasoning": "Your detailed analysis of the student's answer, comparing it to the official solution and explaining your evaluation. Be specific about what the student got right and wrong.",
    "response": "The final score as a single number (e.g., '0', '1', '2', '7')"
}}
</json>

IMPORTANT: The "response" field must contain ONLY the numeric score, with no additional text or explanation.

Be thorough in your reasoning and precise in your scoring."""

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
                # Try to get response field first, fallback to other common fields
                last_extract = extracted[-1]
                if "response" in last_extract:
                    prediction = last_extract["response"]
                elif "score" in last_extract:
                    prediction = last_extract["score"]
                elif "answer" in last_extract:
                    prediction = last_extract["answer"]
                else:
                    # If no recognized field, use the first value found
                    prediction = list(last_extract.values())[0] if last_extract else "None"
                
                # Normalize the score
                prediction = _normalize_score(prediction)
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
