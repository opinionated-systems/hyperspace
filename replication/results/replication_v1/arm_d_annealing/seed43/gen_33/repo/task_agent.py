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


def _extract_score_from_text(text: str) -> int | None:
    """Extract a numerical score from text when JSON parsing fails.
    
    Looks for patterns like:
    - "score: 7" or "score is 7"
    - "give 7 points" or "award 7 points"
    - "the answer is 7"
    - standalone numbers that look like scores (0-7)
    """
    # Pattern: score-related keywords followed by number
    patterns = [
        r'(?:score|grade|mark|points?)[\s:=]+(\d+)',
        r'(?:give|award|assign)[\s]+(\d+)[\s]+(?:points?|marks?)',
        r'(?:the\s+)?(?:answer|result|final\s+score)[\s:=]+(\d+)',
        r'response[\s":]+(\d+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            score = int(match.group(1))
            if 0 <= score <= 7:  # IMO scores are typically 0-7
                return score
    
    # Fallback: look for standalone numbers 0-7 in the last line
    lines = text.strip().split('\n')
    for line in reversed(lines[-3:]):  # Check last 3 lines
        numbers = re.findall(r'\b([0-7])\b', line)
        if numbers:
            return int(numbers[-1])
    
    return None


class TaskAgent:
    """Task agent that solves IMO grading problems with enhanced reasoning."""

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
        # Extract fields for structured prompting
        domain = inputs.get("domain", "")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = f"""You are an expert mathematical grader for the International Mathematical Olympiad (IMO).

Your task is to evaluate a student's answer to a mathematical problem and assign a score based on the official grading guidelines.

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

Follow this systematic grading process:

### Step 1: Understand the Problem
- Identify the key mathematical concepts and techniques required
- Note the critical steps in the official solution

### Step 2: Analyze the Student's Answer
- Identify what the student did correctly and completely
- Identify any errors, gaps, or incorrect reasoning
- Check if the student proved all necessary claims
- Note any creative or alternative valid approaches

### Step 3: Compare Against Grading Guidelines
- Check which partial credit criteria are met
- Verify if the student's work satisfies each rubric item
- Consider if incomplete proofs warrant partial credit

### Step 4: Determine the Score
- The score should be an integer (typically 0, 1, 2, ..., 7)
- Follow the grading guidelines strictly
- Award partial credit when appropriate based on the guidelines
- When in doubt, favor the student (IMO tradition)

### Step 5: Provide Your Final Answer
Respond in JSON format with the following schema:
<json>
{{
    "response": <numerical_score>,
    "reasoning": "<brief explanation of your grading decision>"
}}
</json>

The response field should contain ONLY the numerical score (e.g., 0, 1, 2, 3, etc.)."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON with enhanced error handling
        prediction = "None"
        try:
            extracted = _extract_jsons(msg_history[-1]["text"])
            if extracted:
                last_json = extracted[-1]
                if "response" in last_json:
                    prediction = last_json["response"]
                    # Validate that the prediction is a number
                    if isinstance(prediction, (int, float)):
                        prediction = str(int(prediction))
                    else:
                        # Try to extract a number from string
                        match = re.search(r'\d+', str(prediction))
                        if match:
                            prediction = match.group(0)
                        else:
                            prediction = "0"
                else:
                    # Try to extract score from reasoning field or full text
                    score = _extract_score_from_text(msg_history[-1]["text"])
                    if score is not None:
                        prediction = str(score)
            else:
                # No JSON found, try to extract from text
                score = _extract_score_from_text(msg_history[-1]["text"])
                if score is not None:
                    prediction = str(score)
                    self.log_fn(f"Extracted score {score} from text (no JSON)")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            # Last resort: try to extract from text
            try:
                score = _extract_score_from_text(msg_history[-1]["text"])
                if score is not None:
                    prediction = str(score)
            except Exception:
                pass

        return str(prediction), msg_history
