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
    
    Looks for patterns like "score: 2", "grade: 1", "awarded 3 points", etc.
    Returns the first valid integer found in context of scoring, or None.
    """
    # Pattern: look for score/grade/points followed by a number
    patterns = [
        r'(?:score|grade|points?|mark|awarded)[\s:]*(\d+)',
        r'(\d+)\s*(?:points?|marks?)',
        r'(?:final|total|assigned)[\s\w]*[:\s]*(\d+)',
        r'(?:^|\n)\s*(\d+)\s*(?:$|\n)',
    ]
    
    text_lower = text.lower()
    for pattern in patterns:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            try:
                score = int(match)
                if 0 <= score <= 10:  # Reasonable score range for IMO
                    return score
            except ValueError:
                continue
    
    # Fallback: find any standalone digit that could be a score
    standalone = re.findall(r'(?:^|\s)([0-9])(?:\s|$|\.|,)', text)
    for s in standalone:
        try:
            score = int(s)
            if 0 <= score <= 7:  # Most IMO problems are 0-7 points
                return score
        except ValueError:
            continue
    
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

Follow this structured grading process:

### Step 1: Initial Assessment
Read the student's answer carefully. Identify the key mathematical concepts and techniques required by the problem.

### Step 2: Compare with Official Solution
- Check if the student used correct mathematical approach
- Verify if key theorems/lemmas were applied correctly
- Note any alternative valid approaches

### Step 3: Identify Errors and Gaps
- Mark any computational errors
- Identify logical gaps in reasoning
- Check if all conditions of the problem were addressed
- Note any missing steps in the proof

### Step 4: Apply Grading Guidelines
- Review the specific point allocation in the guidelines
- Award partial credit for correct partial progress
- Deduct points for errors according to the rubric

### Step 5: Final Score Determination
- Sum up the points earned
- Verify the score aligns with the guidelines
- Ensure consistency with similar solutions

## Output Format

Respond in JSON format with the following schema:
<json>
{{
    "reasoning": "Brief explanation of your grading decision (1-2 sentences)",
    "response": <numerical_score>
}}
</json>

The "response" field must contain ONLY the numerical score (e.g., 0, 1, 2, 3, etc.).
The "reasoning" field should briefly justify the score."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON with enhanced fallback logic
        prediction = "0"  # Default to 0 instead of "None"
        try:
            last_message = msg_history[-1]["text"]
            extracted = _extract_jsons(last_message)
            
            if extracted:
                last_json = extracted[-1]
                
                # Try to get response field
                if "response" in last_json:
                    raw_prediction = last_json["response"]
                    
                    # Validate that the prediction is a number
                    if isinstance(raw_prediction, (int, float)):
                        score = int(raw_prediction)
                        if 0 <= score <= 10:  # Valid IMO score range
                            prediction = str(score)
                        else:
                            prediction = str(max(0, min(score, 10)))  # Clamp to valid range
                    else:
                        # Try to extract a number from string
                        match = re.search(r'\d+', str(raw_prediction))
                        if match:
                            prediction = match.group(0)
                        else:
                            # Try text extraction as fallback
                            text_score = _extract_score_from_text(last_message)
                            if text_score is not None:
                                prediction = str(text_score)
                else:
                    # No response field, try to find any number in the JSON
                    text_score = _extract_score_from_text(last_message)
                    if text_score is not None:
                        prediction = str(text_score)
            else:
                # No JSON found, try text extraction
                text_score = _extract_score_from_text(last_message)
                if text_score is not None:
                    prediction = str(text_score)
                    
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            # Try one more time with just the raw text
            try:
                if msg_history:
                    text_score = _extract_score_from_text(msg_history[-1]["text"])
                    if text_score is not None:
                        prediction = str(text_score)
            except Exception:
                pass

        return str(prediction), msg_history
