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


def _extract_score_from_text(text: str) -> str | None:
    """Extract a numeric score from raw text as fallback."""
    # Look for patterns like "score: 7", "score is 7", "final score: 7", etc.
    patterns = [
        r'(?:score|grade|mark|points?)\s*(?:is|:|=)\s*["\']?(\d+)["\']?',
        r'(?:final|total|assigned)\s+(?:score|grade|mark)\s*(?:is|:|=)\s*["\']?(\d+)["\']?',
        r'["\']?(\d+)["\']?\s*(?:points?|marks?)',
        r'\b(\d+)\b',  # Last resort: any standalone number
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def _validate_score(score: str, grading_guidelines: str) -> str:
    """Validate and normalize the extracted score."""
    if not score or score == "None":
        return "None"
    
    # Clean the score string
    score_str = str(score).strip().strip('"\'')
    
    # Try to extract just the numeric part
    numeric_match = re.search(r'\d+', score_str)
    if numeric_match:
        score_str = numeric_match.group(0)
    
    # Check if it's a valid non-negative integer
    try:
        score_int = int(score_str)
        if score_int < 0:
            return "0"
        return str(score_int)
    except ValueError:
        return "None"


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

Follow this systematic approach to grade the student's answer:

1. **Understand the Problem**: Carefully read the problem statement and understand what is being asked.

2. **Study the Official Solution**: Identify all key steps, theorems, and logical deductions required for a complete solution.

3. **Analyze the Grading Guidelines**: Note the partial credit rules and how points are allocated for different solution approaches.

4. **Evaluate the Student's Answer**:
   - Check if the student correctly identified the problem type and approach
   - Verify each mathematical claim and calculation
   - Identify which key steps from the official solution are present
   - Note any missing steps, errors, or logical gaps
   - Consider alternative valid approaches mentioned in the guidelines

5. **Assign a Score**:
   - Award full points only for complete, correct solutions
   - Award partial credit according to the guidelines for incomplete but correct partial solutions
   - Give 0 points for completely incorrect or irrelevant answers
   - Be precise: the score must be a non-negative integer

## Response Format

You MUST respond in JSON format with the following schema:
<json>
{{
    "reasoning": "Your detailed step-by-step analysis. First, summarize the official solution's key steps. Then analyze what the student did correctly and incorrectly. Finally, explain how you arrived at the score based on the grading guidelines.",
    "response": "The final score as a single non-negative integer (e.g., 0, 1, 2, 7)"
}}
</json>

Important:
- The "response" field must contain ONLY a numeric score (no text, no explanations)
- Be thorough in your reasoning but precise in your scoring
- Follow the grading guidelines exactly when assigning partial credit"""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON with multiple fallback strategies
        prediction = "None"
        try:
            # Strategy 1: Extract from <json> blocks
            extracted = _extract_jsons(msg_history[-1]["text"])
            if extracted:
                last_extract = extracted[-1]
                # Try known field names in order of preference
                for field in ["response", "score", "answer", "grade", "mark"]:
                    if field in last_extract:
                        prediction = last_extract[field]
                        break
                else:
                    # If no recognized field, use the first string/number value found
                    for val in last_extract.values():
                        if isinstance(val, (str, int, float)):
                            prediction = str(val)
                            break
            else:
                # Strategy 2: Try to extract score from raw text
                text_score = _extract_score_from_text(msg_history[-1]["text"])
                if text_score:
                    prediction = text_score
                    self.log_fn(f"Extracted score from raw text: {prediction}")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            # Strategy 3: Last resort - try to find any number in the response
            try:
                text_score = _extract_score_from_text(msg_history[-1]["text"])
                if text_score:
                    prediction = text_score
            except:
                pass

        # Validate and normalize the score
        prediction = _validate_score(prediction, grading_guidelines)

        return str(prediction), msg_history
