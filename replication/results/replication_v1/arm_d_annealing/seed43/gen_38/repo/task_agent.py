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
    """Extract score from text using regex patterns as fallback."""
    # Look for patterns like "score: 7", "score is 7", "score of 7", etc.
    patterns = [
        r'["\']?score["\']?\s*[:=]\s*["\']?(\d+)["\']?',
        r'["\']?score["\']?\s+is\s+["\']?(\d+)["\']?',
        r'["\']?score["\']?\s+of\s+["\']?(\d+)["\']?',
        r'(?:score|grade|points?)\s*[:=]\s*(\d+)',
        r'(?:score|grade|points?)\s+(?:is|of)\s+(\d+)',
        r'(?:^|\s)(\d+)\s*(?:points?|/\s*7)\s*(?:$|\s)',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def _validate_score(score: str, max_score: int = 7) -> str | None:
    """Validate that the score is a valid number within expected range."""
    try:
        # Try to extract just the numeric part
        numeric_match = re.search(r'\d+', str(score))
        if numeric_match:
            numeric_score = int(numeric_match.group())
            if 0 <= numeric_score <= max_score:
                return str(numeric_score)
    except (ValueError, TypeError):
        pass
    return None


class TaskAgent:
    """Task agent that solves IMO grading problems with improved robustness."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.max_retries = 2  # Number of retries for failed JSON extraction

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with retry logic.

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

        instruction = f"""You are an expert mathematical grader for International Mathematical Olympiad (IMO) problems.

Your task is to evaluate a student's answer to a mathematical problem and assign a score.

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
1. Carefully read the problem, official solution, and grading guidelines.
2. Analyze the student's answer step by step, checking each claim and calculation.
3. Compare the student's approach with the official solution.
4. Identify any errors, omissions, or creative valid approaches.
5. Assign a score based on the grading guidelines (typically 0-7 points for IMO problems).

## Grading Criteria
- 7 points: Complete, correct solution with clear reasoning
- 6 points: Minor flaw in an otherwise correct solution
- 5 points: Significant progress with one or more errors
- 3-4 points: Partial progress with some correct ideas
- 1-2 points: Some relevant ideas but minimal progress
- 0 points: No meaningful progress or completely wrong

Respond ONLY in JSON format with the following schema:
<json>
{{
    "reasoning": "Your detailed step-by-step analysis of the student's answer",
    "score": "The numerical score (0-7)",
    "response": "The final score as a number (0-7)"
}}
</json>

IMPORTANT: Both "score" and "response" must contain the same numeric value from 0 to 7."""

        msg_history = []
        prediction = "None"
        
        # Try with retries for better robustness
        for attempt in range(self.max_retries + 1):
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=msg_history if attempt > 0 else [],
                )

                # Extract prediction from JSON with multiple fallback strategies
                last_text = msg_history[-1]["text"] if msg_history else ""
                
                # Strategy 1: Extract from <json> blocks
                extracted = _extract_jsons(last_text)
                if extracted:
                    last_json = extracted[-1]
                    # Try response field first, then score field
                    raw_score = None
                    if "response" in last_json:
                        raw_score = last_json["response"]
                    elif "score" in last_json:
                        raw_score = last_json["score"]
                    
                    if raw_score is not None:
                        validated = _validate_score(str(raw_score))
                        if validated:
                            prediction = validated
                            break
                
                # Strategy 2: Extract from text patterns
                text_score = _extract_score_from_text(last_text)
                if text_score:
                    validated = _validate_score(text_score)
                    if validated:
                        prediction = validated
                        break
                
                # If we get here and it's not the last attempt, add a retry prompt
                if attempt < self.max_retries:
                    retry_msg = (
                        "Your previous response did not contain a valid JSON score. "
                        "Please respond with a valid JSON object containing the score. "
                        "Format: <json>{\"reasoning\": \"...\", \"score\": \"X\", \"response\": \"X\"}</json>"
                    )
                    msg_history.append({"role": "user", "text": retry_msg})
                    
            except Exception as e:
                self.log_fn(f"Error in attempt {attempt + 1}: {e}")
                if attempt == self.max_retries:
                    break

        return str(prediction), msg_history
