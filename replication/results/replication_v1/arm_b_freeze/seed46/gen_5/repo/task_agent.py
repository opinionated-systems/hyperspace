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
    """Fallback extraction: try to find a numerical score in the text.
    
    Looks for patterns like "score: 7", "Score: 7", "assigned score of 7",
    or standalone numbers that could be scores (0-7 for IMO problems).
    """
    # Look for explicit score mentions
    patterns = [
        r'["\']score["\']\s*[:=]\s*["\']?(\d+)["\']?',
        r'["\']response["\']\s*[:=]\s*["\']?(\d+)["\']?',
        r'(?:score|assigned|grade|points?)\s*(?:of|is|:|=)\s*(\d+)',
        r'(?:final|total)\s*(?:score|grade|points?)\s*(?:is|:|=)\s*(\d+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            score = match.group(1)
            if score.isdigit() and 0 <= int(score) <= 7:
                return score
    
    # Last resort: look for standalone digits 0-7 in the last 200 chars
    # (likely the conclusion)
    last_part = text[-200:] if len(text) > 200 else text
    digits = re.findall(r'\b([0-7])\b', last_part)
    if digits:
        return digits[-1]  # Return the last one found
    
    return None


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with structured reasoning.

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

        instruction = f"""You are an expert IMO (International Mathematical Olympiad) grader. Your task is to evaluate a student's solution to a mathematical problem with precision and consistency.

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

Please evaluate the student's answer following this structured approach:

### Step 1: Problem Analysis
- Summarize what the problem is asking
- Identify key mathematical concepts and techniques required
- Note the expected difficulty level

### Step 2: Official Solution Breakdown
- Identify the critical proof steps and key insights
- Note the final answer format and value
- Understand the logical flow from assumptions to conclusion

### Step 3: Grading Criteria Mapping
- Map each grading guideline to specific solution components
- Identify partial credit opportunities
- Note common error patterns that affect scoring

### Step 4: Student Answer Evaluation
- **Correctness**: Does the final answer match the official solution?
- **Completeness**: Are all required proof steps present?
- **Logical Structure**: Is the reasoning sound and well-organized?
- **Mathematical Rigor**: Are definitions clear, notation consistent, and logic valid?
- **Partial Credit**: Identify any correct sub-results or valid alternative approaches
- **Errors**: Note any logical gaps, computational mistakes, or missing justifications

### Step 5: Score Determination
Based on your analysis, assign a score from 0-7 that accurately reflects:
- Full correctness and rigor (7 points)
- Minor gaps in justification (5-6 points)
- Significant progress with key insights (3-4 points)
- Some relevant work but major gaps (1-2 points)
- No meaningful progress or completely incorrect (0 points)

Respond ONLY in JSON format with the following schema:
<json>
{{
    "reasoning": "Your detailed chain-of-thought analysis covering all steps above. Be specific about what the student did correctly and incorrectly.",
    "score": "The numerical score (0-7) assigned to the student's answer",
    "response": "The final score (same as score field, for compatibility)"
}}
</json>

Important: Ensure your JSON is valid and the score is a single integer between 0 and 7."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON with improved error handling
        prediction = "None"
        last_message_text = msg_history[-1].get("text", "") if msg_history else ""
        
        try:
            extracted = _extract_jsons(last_message_text)
            if extracted:
                result = extracted[-1]
                # Prefer "response" field, fallback to "score" field
                if "response" in result:
                    prediction = result["response"]
                elif "score" in result:
                    prediction = result["score"]
                else:
                    # Try fallback extraction if JSON exists but fields are missing
                    prediction = _extract_score_from_text(last_message_text) or "None"
            else:
                # No JSON found, try text-based extraction
                prediction = _extract_score_from_text(last_message_text) or "None"
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            # Try fallback extraction on error
            try:
                prediction = _extract_score_from_text(last_message_text) or "None"
            except Exception:
                pass

        return str(prediction), msg_history
