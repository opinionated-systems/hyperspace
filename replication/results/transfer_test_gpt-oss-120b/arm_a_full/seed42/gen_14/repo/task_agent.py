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


# Few-shot examples for better grading accuracy
FEW_SHOT_EXAMPLES = """
Example 1:
Problem: Find all positive integers n such that n^2 + 3n + 2 is divisible by 4.
Solution: We factor n^2 + 3n + 2 = (n+1)(n+2). Since n+1 and n+2 are consecutive integers, one is even. For divisibility by 4, we need one of them divisible by 4. This happens when n ≡ 3 (mod 4) or n ≡ 2 (mod 4). So n ≡ 2 or 3 (mod 4).
Student Answer: "n must be even"
Analysis: The student only identified that one factor is even, but missed that we need a factor of 4. The answer is incomplete - it should specify n ≡ 2 or 3 (mod 4).
Grade: 0 (incorrect/incomplete)

Example 2:
Problem: Prove that the sum of angles in a triangle is 180 degrees.
Solution: Draw a line through one vertex parallel to the opposite side. Using alternate interior angles, we show the three angles form a straight line.
Student Answer: "Draw parallel line, alternate angles equal, so angles sum to 180"
Analysis: The student captured the key insight and proof structure correctly, though briefly. The reasoning is valid.
Grade: 1 (correct)

Example 3:
Problem: Solve x^2 - 5x + 6 = 0
Solution: Factoring gives (x-2)(x-3) = 0, so x = 2 or x = 3.
Student Answer: "x = 2, x = 3"
Analysis: The student found both correct solutions. Even without showing work, the answer is correct.
Grade: 1 (correct)

Example 4:
Problem: Prove that for any prime p > 3, p^2 ≡ 1 (mod 24).
Solution: Any prime p > 3 is of form 6k±1. Then p^2 = 36k^2 ± 12k + 1 = 12k(3k ± 1) + 1. Since k(3k ± 1) is always even, p^2 ≡ 1 (mod 24).
Student Answer: "All primes > 3 are odd, so p^2 is odd. Also p^2 - 1 = (p-1)(p+1) is divisible by 8 since p-1 and p+1 are consecutive even numbers, and by 3 since p is not divisible by 3. So p^2 ≡ 1 (mod 24)."
Analysis: The student provided a complete and correct proof using a different but valid approach. They correctly identified that (p-1)(p+1) is divisible by 8 (product of two consecutive even numbers, one divisible by 4) and by 3 (since p not divisible by 3, one of p-1 or p+1 is). The reasoning is sound.
Grade: 1 (correct)

Example 5:
Problem: Find the maximum value of f(x) = x(1-x) on [0,1].
Solution: f(x) = x - x^2. Taking derivative: f'(x) = 1 - 2x = 0 gives x = 1/2. f(1/2) = 1/4. Checking endpoints: f(0) = f(1) = 0. Maximum is 1/4.
Student Answer: "The maximum is 1/4 at x = 0.5"
Analysis: The student correctly identified both the maximum value and where it occurs. The answer is correct and complete.
Grade: 1 (correct)
"""


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

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
        # Extract fields with defaults for robustness
        domain = inputs.get("domain", "mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = f"""You are an expert grader for {domain} problems. Your task is to evaluate whether a student's answer is correct based on the official solution and grading guidelines.

Follow this process:
1. Read the problem carefully
2. Study the official solution to understand what constitutes a correct answer
3. Review the grading guidelines for specific criteria
4. Analyze the student's answer - identify what they got right and wrong
5. Make a final determination: 1 if correct, 0 if incorrect

{FEW_SHOT_EXAMPLES}

Now grade this submission:

PROBLEM:
{problem}

OFFICIAL SOLUTION:
{solution}

GRADING GUIDELINES:
{grading_guidelines}

STUDENT ANSWER:
{student_answer}

Provide your analysis and final grade in JSON format:
<json>
{{
    "analysis": "Your step-by-step reasoning about the student's answer",
    "response": 1 or 0
}}
</json>"""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON with improved error handling
        prediction = "0"  # Default to incorrect if extraction fails
        raw_text = ""
        try:
            raw_text = msg_history[-1]["text"]
            extracted = _extract_jsons(raw_text)
            if extracted:
                last_json = extracted[-1]
                if "response" in last_json:
                    pred_val = last_json["response"]
                    # Normalize to 0 or 1
                    if isinstance(pred_val, bool):
                        prediction = "1" if pred_val else "0"
                    elif isinstance(pred_val, (int, float)):
                        prediction = "1" if pred_val >= 0.5 else "0"
                    elif isinstance(pred_val, str):
                        pred_lower = pred_val.lower().strip()
                        if pred_lower in ("1", "true", "correct", "yes", "right"):
                            prediction = "1"
                        elif pred_lower in ("0", "false", "incorrect", "no", "wrong"):
                            prediction = "0"
                        else:
                            # Try to parse as number
                            try:
                                prediction = "1" if float(pred_val) >= 0.5 else "0"
                            except ValueError:
                                prediction = "0"
                    else:
                        prediction = str(pred_val)
                else:
                    # Try to infer from analysis field if present
                    if "analysis" in last_json:
                        analysis_lower = last_json["analysis"].lower()
                        # Check for explicit correctness indicators with negation handling
                        correct_indicators = ["correct", "right", "valid", "true", "accurate"]
                        incorrect_indicators = ["incorrect", "wrong", "invalid", "false", "not correct", "not right", "inaccurate"]
                        
                        has_correct = any(word in analysis_lower for word in correct_indicators)
                        has_incorrect = any(word in analysis_lower for word in incorrect_indicators)
                        
                        if has_correct and not has_incorrect:
                            prediction = "1"
                        elif has_incorrect:
                            prediction = "0"
            self.log_fn(f"[TaskAgent] Extracted prediction: {prediction}")
        except Exception as e:
            self.log_fn(f"[TaskAgent] Error extracting prediction: {e}")

        # Fallback: if JSON extraction failed, try pattern matching on raw text
        if prediction == "0" and raw_text:
            raw_lower = raw_text.lower()
            # Look for explicit grade/response patterns
            if '"response": 1' in raw_text or '"response":1' in raw_text:
                prediction = "1"
            elif '"response": 0' in raw_text or '"response":0' in raw_text:
                prediction = "0"
            elif '"grade": 1' in raw_text or '"grade":1' in raw_text:
                prediction = "1"
            elif '"grade": 0' in raw_text or '"grade":0' in raw_text:
                prediction = "0"
            # Look for explicit statements at end of response
            elif "grade: 1" in raw_lower[-200:] or "grade:1" in raw_lower[-200:]:
                prediction = "1"
            elif "grade: 0" in raw_lower[-200:] or "grade:0" in raw_lower[-200:]:
                prediction = "0"
            # Final check: if analysis clearly states correctness without negation
            elif "the answer is correct" in raw_lower and "not correct" not in raw_lower:
                prediction = "1"
            elif "the answer is incorrect" in raw_lower or "the answer is wrong" in raw_lower:
                prediction = "0"

        return str(prediction), msg_history
