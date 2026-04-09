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
    """Extract numerical score from text when JSON parsing fails.
    
    Looks for patterns like "score: 5", "score of 5", "5/7", etc.
    Returns the first valid score found, or None if no score detected.
    """
    # Pattern: score followed by number (0-7 for IMO)
    patterns = [
        r'score[\s]*[:=][\s]*(\d)',
        r'score[\s]+of[\s]+(\d)',
        r'(?:^|\s)(\d)[\s]*[/\-][\s]*7',
        r'(?:assign|give|award)[\s]+(?:a[\s]+)?score[\s]+(?:of[\s]+)?(\d)',
        r'grade[\s]*[:=][\s]*(\d)',
        r'(?:^|\s)([0-7])(?:\s|$|[^\d])',
    ]
    
    text_lower = text.lower()
    for pattern in patterns:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            try:
                score = int(match)
                if 0 <= score <= 7:  # Valid IMO score range
                    return score
            except (ValueError, TypeError):
                continue
    return None


class TaskAgent:
    """Task agent that solves IMO grading problems with enhanced evaluation."""

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
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = f"""You are an expert IMO (International Mathematical Olympiad) grader. Your task is to evaluate a student's solution to a mathematics problem.

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
Carefully evaluate the student's answer against the official solution and grading guidelines. Consider:
1. Mathematical correctness - are the calculations and reasoning valid?
2. Completeness - does the answer address all parts of the problem?
3. Clarity - is the reasoning clear and well-structured?
4. Score - assign a score based on the grading guidelines (typically 0-7 for IMO problems)

Provide your evaluation in JSON format with the following schema:
<json>
{{
    "score": <integer 0-7>,
    "evaluation": "Your detailed evaluation including: (1) brief justification, (2) specific errors found if any, (3) strengths of the solution",
    "reasoning": "Step-by-step reasoning for the score assigned"
}}
</json>

Be thorough and fair in your grading. The score must be an integer between 0 and 7."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON with fallback mechanisms
        prediction = "None"
        try:
            # Try to extract from JSON blocks first
            extracted = _extract_jsons(msg_history[-1]["text"])
            if extracted:
                last_json = extracted[-1]
                
                # Build comprehensive evaluation string
                parts = []
                
                # Extract score
                score = None
                if "score" in last_json:
                    try:
                        score = int(last_json["score"])
                        if 0 <= score <= 7:
                            parts.append(f"Score: {score}/7")
                    except (ValueError, TypeError):
                        pass
                
                # If no valid score in JSON, try to extract from text
                if score is None:
                    score = _extract_score_from_text(msg_history[-1]["text"])
                    if score is not None:
                        parts.append(f"Score: {score}/7")
                
                # Add evaluation text
                if "evaluation" in last_json:
                    parts.append(f"Evaluation: {last_json['evaluation']}")
                elif "response" in last_json:
                    parts.append(f"Evaluation: {last_json['response']}")
                
                # Add reasoning if available
                if "reasoning" in last_json:
                    parts.append(f"Reasoning: {last_json['reasoning']}")
                
                if parts:
                    prediction = " | ".join(parts)
                else:
                    # Fallback: use the entire JSON content
                    prediction = json.dumps(last_json)
            else:
                # No JSON found, try to extract score from raw text
                score = _extract_score_from_text(msg_history[-1]["text"])
                if score is not None:
                    prediction = f"Score: {score}/7 | Raw text evaluation (no JSON found)"
                else:
                    prediction = "Error: No valid JSON or score found in response"
                    
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            # Try fallback extraction
            try:
                score = _extract_score_from_text(msg_history[-1]["text"])
                if score is not None:
                    prediction = f"Score: {score}/7 | Extracted via fallback (error: {e})"
            except Exception:
                pass

        return str(prediction), msg_history
