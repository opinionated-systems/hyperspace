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

Example 2:
Problem: Prove that for any positive integer n, n^3 + 2n is divisible by 3.
Solution: We use induction. Base case: n=1, 1^3 + 2(1) = 3, divisible by 3. Inductive step: assume k^3 + 2k divisible by 3. Then (k+1)^3 + 2(k+1) = k^3 + 3k^2 + 3k + 1 + 2k + 2 = k^3 + 2k + 3(k^2 + k + 1). By induction hypothesis, k^3 + 2k divisible by 3, and 3(k^2 + k + 1) clearly divisible by 3. Thus divisible by 3.
Grading Guidelines: Award 7 points for complete proof. Deduct 2 points for missing base case, 3 points for errors in inductive step, 2 points for lack of clarity.
Student Answer: By induction. Base case n=1 works. Assume true for k, then (k+1)^3 + 2(k+1) = k^3 + 3k^2 + 3k + 1 + 2k + 2 = (k^3 + 2k) + 3(k^2 + k + 1), both terms divisible by 3.
Score: 7
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
3. Assign a score based on the grading guidelines (typically 0-7 points for IMO problems)
4. Provide your evaluation in the specified JSON format

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

Respond in JSON format with the following schema:
<json>
{{
    "response": "<numerical_score>"
}}
</json>

The response field should contain only the numerical score (e.g., "7", "5", "0", etc.)."""

        return prompt

    def _validate_score(self, prediction: str) -> str:
        """Validate and normalize the score prediction.
        
        Handles various formats including:
        - Plain numbers: "7", "5"
        - Score prefixes: "Score: 7", "Points: 5"
        - Fractions: "7/7", "5/7"
        - Text with numbers: "7 points", "score of 5"
        - Decimal scores: "6.5", "3.0" (rounded to nearest integer)
        - Range indicators: "5-6", "6 or 7" (takes the first value)
        - Edge cases: negative numbers, out-of-range values, non-numeric strings
        """
        if prediction is None:
            return "0"
        
        # Convert to string and strip whitespace
        prediction = str(prediction).strip()
        
        if not prediction:
            return "0"
        
        # Try to extract a number from the prediction
        # First, look for patterns like "X/Y" (fraction format - numerator is the score)
        fraction_match = re.search(r'(\d+(?:\.\d+)?)\s*/\s*\d+', prediction)
        if fraction_match:
            score_str = fraction_match.group(1)
            try:
                score_float = float(score_str)
                score_int = int(round(score_float))
                if 0 <= score_int <= 7:
                    return str(score_int)
            except ValueError:
                pass
        
        # Look for score keywords with numbers (handles decimals too)
        score_keywords = r'(?:score|points?|grade|mark|value)\s*[:=]?\s*(-?\d+(?:\.\d+)?)'
        keyword_match = re.search(score_keywords, prediction, re.IGNORECASE)
        if keyword_match:
            score_str = keyword_match.group(1)
            try:
                score_float = float(score_str)
                score_int = int(round(score_float))
                # Clamp to valid IMO score range (0-7)
                score_int = max(0, min(7, score_int))
                return str(score_int)
            except ValueError:
                pass
        
        # Look for range patterns like "5-6" or "6 or 7" and extract first number
        range_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:-|to|or|~|–)\s*\d+', prediction)
        if range_match:
            score_str = range_match.group(1)
            try:
                score_float = float(score_str)
                score_int = int(round(score_float))
                if 0 <= score_int <= 7:
                    return str(score_int)
            except ValueError:
                pass
        
        # Look for explicit score declarations with quotes
        quoted_match = re.search(r'["\']?(\d+(?:\.\d+)?)["\']?\s*(?:points?)?', prediction, re.IGNORECASE)
        if quoted_match:
            score_str = quoted_match.group(1)
            try:
                score_float = float(score_str)
                score_int = int(round(score_float))
                if 0 <= score_int <= 7:
                    return str(score_int)
            except ValueError:
                pass
        
        # Fallback: extract any number (including decimals) from the prediction
        # Prioritize numbers at the start or end of the string (common for scores)
        # Try start of string first
        start_match = re.search(r'^\s*(-?\d+(?:\.\d+)?)', prediction)
        if start_match:
            score_str = start_match.group(1)
            try:
                score_float = float(score_str)
                score_int = int(round(score_float))
                # Clamp to valid IMO score range (0-7)
                score_int = max(0, min(7, score_int))
                return str(score_int)
            except ValueError:
                pass
        
        # Try end of string
        end_match = re.search(r'(-?\d+(?:\.\d+)?)\s*$', prediction)
        if end_match:
            score_str = end_match.group(1)
            try:
                score_float = float(score_str)
                score_int = int(round(score_float))
                # Clamp to valid IMO score range (0-7)
                score_int = max(0, min(7, score_int))
                return str(score_int)
            except ValueError:
                pass
        
        # Final fallback: any number in the string
        number_match = re.search(r'(-?\d+(?:\.\d+)?)', prediction)
        if number_match:
            score_str = number_match.group(1)
            try:
                score_float = float(score_str)
                score_int = int(round(score_float))
                # Clamp to valid IMO score range (0-7)
                score_int = max(0, min(7, score_int))
                return str(score_int)
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
        try:
            extracted = _extract_jsons(msg_history[-1]["text"])
            if extracted and "response" in extracted[-1]:
                prediction = extracted[-1]["response"]
            elif extracted:
                # If no "response" key, try to use the last JSON object
                prediction = str(extracted[-1])
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            # Try to extract any number from the response as fallback
            try:
                last_text = msg_history[-1]["text"]
                number_match = re.search(r'\d+', last_text)
                if number_match:
                    prediction = number_match.group()
            except Exception:
                pass

        # Validate and normalize the score
        prediction = self._validate_score(prediction)

        return str(prediction), msg_history
