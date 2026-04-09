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


def _normalize_prediction(prediction: str) -> str:
    """Normalize prediction to one of the valid labels."""
    if not prediction or not isinstance(prediction, str):
        return "incorrect"
    
    prediction = prediction.lower().strip()
    valid_labels = ["correct", "partial", "almost", "incorrect"]
    
    # Direct match
    if prediction in valid_labels:
        return prediction
    
    # Check for partial matches
    for label in valid_labels:
        if label in prediction:
            return label
    
    # Check for common synonyms
    if any(word in prediction for word in ["right", "valid", "complete", "full", "true"]):
        return "correct"
    if any(word in prediction for word in ["partially", "some", "half", "part"]):
        return "partial"
    if any(word in prediction for word in ["close", "nearly", "minor", "small error"]):
        return "almost"
    if any(word in prediction for word in ["wrong", "false", "invalid", "none", "no", "fail"]):
        return "incorrect"
    
    return "incorrect"


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
        # Extract fields from inputs
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        instruction = f"""You are an expert mathematics grader for the International Mathematical Olympiad (IMO). Your task is to evaluate a student's solution and classify it into exactly one of four categories.

## Problem Statement:
{problem}

## Official Solution:
{solution}

## Grading Guidelines:
{grading_guidelines}

## Student's Answer:
{student_answer}

## Classification Criteria:

**"correct"** - The solution is complete and correct:
- All key steps of the proof are present and logically sound
- The solution follows the intended approach or a valid alternative
- No significant gaps or errors in reasoning
- Would receive full marks in competition

**"partial"** - The solution has significant progress but is incomplete:
- Contains valid and useful observations toward the solution
- Has made substantial progress but missing key components
- Has minor gaps that could likely be filled
- Would receive partial credit (roughly 30-70%)

**"almost"** - The solution is nearly complete but has minor mistakes:
- The main structure of the proof is correct
- Contains non-negligible errors that affect the conclusion
- Significant progress made but fundamental flaw exists
- Would receive some credit but not full marks (roughly 10-30%)

**"incorrect"** - The solution is fundamentally wrong:
- Makes no significant progress toward the solution
- Contains major conceptual errors
- Approach is completely off-track
- Would receive minimal or no credit

## Few-Shot Examples:

Example 1 - Correct:
Problem: Prove that for any positive integer n, n^3 - n is divisible by 6.
Student: We can factor n^3 - n = n(n-1)(n+1). These are three consecutive integers, so one is divisible by 2 and one by 3. Thus the product is divisible by 6.
Classification: <json>{{"response": "correct"}}</json>

Example 2 - Partial:
Problem: Prove that for any positive integer n, n^3 - n is divisible by 6.
Student: We can factor n^3 - n = n(n-1)(n+1). These are three consecutive integers.
Classification: <json>{{"response": "partial"}}</json>

Example 3 - Almost:
Problem: Prove that for any positive integer n, n^3 - n is divisible by 6.
Student: We can factor n^3 - n = n(n-1)(n+1). These are three consecutive integers, so one is divisible by 2. Therefore the product is divisible by 2.
Classification: <json>{{"response": "almost"}}</json>

Example 4 - Incorrect:
Problem: Prove that for any positive integer n, n^3 - n is divisible by 6.
Student: Let's test n=2: 8-2=6 which is divisible by 6. n=3: 27-3=24 divisible by 6. So it's true.
Classification: <json>{{"response": "incorrect"}}</json>

## Your Task:
Analyze the student's answer carefully. Follow these steps:
1. Identify the key claims and steps in the official solution
2. Check which claims/steps appear in the student's answer
3. Verify the logical validity of each step in the student's answer
4. Determine if gaps in reasoning are minor (could be filled) or major (fundamental flaw)
5. Classify based on the criteria above

Provide your classification in the exact JSON format below.

IMPORTANT: Your response MUST be ONLY the JSON object in <json> tags. Do not include any other text, explanation, or analysis outside the JSON.

<json>
{{
    "response": "correct" | "partial" | "almost" | "incorrect"
}}
</json>"""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON with robust error handling
        prediction = "incorrect"  # Default fallback
        try:
            if msg_history and len(msg_history) > 0:
                last_msg = msg_history[-1]
                text = last_msg.get("text", "") if isinstance(last_msg, dict) else str(last_msg)
                extracted = _extract_jsons(text)
                if extracted and len(extracted) > 0:
                    last_extracted = extracted[-1]
                    if isinstance(last_extracted, dict) and "response" in last_extracted:
                        raw_prediction = last_extracted["response"]
                        prediction = _normalize_prediction(raw_prediction)
                        self.log_fn(f"Successfully extracted prediction: {raw_prediction} -> {prediction}")
                    else:
                        self.log_fn(f"Extracted JSON missing 'response' field: {last_extracted}")
                else:
                    # Try to extract any of the valid labels directly from text
                    text_lower = text.lower()
                    for label in ["correct", "partial", "almost", "incorrect"]:
                        if f'"{label}"' in text_lower or label in text_lower[:200]:
                            prediction = label
                            self.log_fn(f"Fallback extraction found: {prediction}")
                            break
                    else:
                        self.log_fn("No JSON found in response, using default")
            else:
                self.log_fn("Empty message history, using default")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {type(e).__name__}: {e}")
            prediction = "incorrect"  # Default fallback on error

        return str(prediction), msg_history
