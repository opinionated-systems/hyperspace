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


# Few-shot examples for IMO grading
_FEW_SHOT_EXAMPLES = """
Example 1:
Problem: Find all positive integers n such that n^2 + 3n + 2 is divisible by 4.
Solution: We can factor n^2 + 3n + 2 = (n+1)(n+2). For this to be divisible by 4, either (n+1) or (n+2) must be even, and at least one must be divisible by 2. Since n+1 and n+2 are consecutive integers, one is even. For the product to be divisible by 4, we need the even number to be divisible by 4, or both to be even (impossible for consecutive integers). So we need n+1 ≡ 0 (mod 4) or n+2 ≡ 0 (mod 4), meaning n ≡ 3 (mod 4) or n ≡ 2 (mod 4).
Grading Guidelines: Award 7 points for correct answer (n ≡ 2 or 3 mod 4). Award partial credit: 2 points for correct factorization, 2 points for recognizing need for divisibility by 4, 3 points for correct modular analysis.
Student Answer: n^2 + 3n + 2 = (n+1)(n+2). Since these are consecutive, one is even. For divisibility by 4, we need n ≡ 2 or 3 (mod 4).
Score: 7
Reasoning: The student correctly factored the expression, recognized the consecutive integer property, and arrived at the correct answer with proper modular arithmetic.

Example 2:
Problem: Prove that for any positive integer n, n^3 + 2n is divisible by 3.
Solution: We use induction. Base case: n=1, 1^3 + 2(1) = 3, divisible by 3. Inductive step: assume k^3 + 2k divisible by 3. Then (k+1)^3 + 2(k+1) = k^3 + 3k^2 + 3k + 1 + 2k + 2 = k^3 + 2k + 3(k^2 + k + 1). By induction hypothesis, k^3 + 2k divisible by 3, and 3(k^2 + k + 1) clearly divisible by 3. Thus divisible by 3.
Grading Guidelines: Award 7 points for complete proof. Deduct 2 points for missing base case, 3 points for errors in inductive step, 2 points for lack of clarity.
Student Answer: By induction. Base case n=1 works. Assume true for k, then (k+1)^3 + 2(k+1) = k^3 + 3k^2 + 3k + 1 + 2k + 2 = (k^3 + 2k) + 3(k^2 + k + 1), both terms divisible by 3.
Score: 7
Reasoning: The student provided a complete and correct inductive proof with proper base case and inductive step.

Example 3 (Partial Credit):
Problem: Prove that the sum of the first n positive integers is n(n+1)/2.
Solution: Use induction. Base case: n=1, sum is 1 = 1(2)/2. Inductive step: assume true for k, then sum to k+1 is k(k+1)/2 + (k+1) = (k+1)(k/2 + 1) = (k+1)(k+2)/2.
Grading Guidelines: Award 7 points for complete proof. Award 4 points for correct base case and setup but algebraic error in inductive step. Award 2 points for only correct base case.
Student Answer: Base case: n=1, sum is 1 = 1(2)/2. For induction, assume true for k. Then sum to k+1 is k(k+1)/2 + k+1.
Score: 4
Reasoning: The student correctly established the base case and set up the inductive step properly, but did not complete the algebraic simplification to show the formula holds for k+1.
"""


class TaskAgent:
    """Task agent that solves IMO grading problems."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def _build_prompt(self, inputs: dict) -> str:
        """Build a structured prompt for the grading task."""
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        prompt = f"""You are an expert mathematics grader for International Mathematical Olympiad (IMO) problems. Your task is to evaluate student answers based on the official solution and grading guidelines.

Your responsibilities:
1. Carefully read the problem, official solution, and grading guidelines
2. Evaluate the student's answer against the official solution
3. Identify what parts are correct, partially correct, or incorrect
4. Consider the grading guidelines for partial credit
5. Assign an appropriate score (typically 0-7 points for IMO problems)
6. Provide brief reasoning explaining your grading decision

{_FEW_SHOT_EXAMPLES}

Now evaluate the following:

DOMAIN: {domain}

PROBLEM:
{problem}

OFFICIAL SOLUTION:
{solution}

GRADING GUIDELINES:
{grading_guidelines}

STUDENT ANSWER TO EVALUATE:
{student_answer}

Instructions:
- Compare the student answer to the official solution
- Identify what parts are correct, partially correct, or incorrect
- Consider the grading guidelines for partial credit
- Assign an appropriate score
- Provide concise reasoning for your decision

Respond in JSON format with the following schema:
<json>
{{
    "response": "<numerical_score>",
    "reasoning": "<brief explanation of grading decision>"
}}
</json>

The response field should contain only the numerical score (e.g., "7", "5", "0", etc.).
The reasoning field should briefly explain why you assigned that score."""

        return prompt

    def _validate_score(self, prediction: str) -> str:
        """Validate and normalize the score prediction."""
        if prediction is None:
            return "0"
        
        # Convert to string and strip whitespace
        prediction = str(prediction).strip()
        
        # Try to extract a number from the prediction
        # Handle cases like "Score: 7", "7 points", "7/7", etc.
        # First try to find a standalone number (word boundary)
        number_match = re.search(r'\b([0-7])\b', prediction)
        if number_match:
            return number_match.group(1)
        
        # If no standalone number found, try any digit sequence
        number_match = re.search(r'\d+', prediction)
        if number_match:
            score = number_match.group()
            # Validate it's a reasonable IMO score (0-7)
            try:
                score_int = int(score)
                if 0 <= score_int <= 7:
                    return str(score_int)
                # Clamp to valid range if outside
                if score_int > 7:
                    return "7"
                if score_int < 0:
                    return "0"
            except ValueError:
                pass
        
        # If we can't extract a valid number, return "0"
        return "0"

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        instruction = self._build_prompt(inputs)

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON
        prediction = "0"
        reasoning = ""
        try:
            extracted = _extract_jsons(msg_history[-1]["text"])
            if extracted:
                last_json = extracted[-1]
                if "response" in last_json:
                    prediction = last_json["response"]
                elif isinstance(last_json, dict):
                    # Try to find any numeric-looking value
                    for key, value in last_json.items():
                        if isinstance(value, (int, float)) or (isinstance(value, str) and value.isdigit()):
                            prediction = str(value)
                            break
                
                # Extract reasoning if available
                if "reasoning" in last_json:
                    reasoning = last_json["reasoning"]
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            # Try to extract any number from the response as fallback
            try:
                last_text = msg_history[-1]["text"]
                # Try to find a standalone single digit (IMO score)
                number_match = re.search(r'\b([0-7])\b', last_text)
                if number_match:
                    prediction = number_match.group(1)
                else:
                    # Fallback to any digit
                    number_match = re.search(r'\d+', last_text)
                    if number_match:
                        prediction = number_match.group()
            except Exception:
                pass

        # Validate and normalize the score
        prediction = self._validate_score(prediction)
        
        # Log reasoning if available
        if reasoning:
            self.log_fn(f"Grading reasoning: {reasoning[:200]}")

        return str(prediction), msg_history
