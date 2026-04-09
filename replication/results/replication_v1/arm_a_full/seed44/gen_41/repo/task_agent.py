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
    Also handles markdown code blocks and plain JSON.
    """
    results = []
    search_from = 0
    
    # First try to find <json>...</json> blocks
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
    
    # If no <json> blocks found, try markdown code blocks
    if not results:
        search_from = 0
        while True:
            start = text.find("```json", search_from)
            if start == -1:
                break
            start += 7
            end = text.find("```", start)
            if end == -1:
                break
            inner = text[start:end].strip()
            search_from = end + 3
            try:
                results.append(json.loads(inner))
            except json.JSONDecodeError:
                continue
    
    # If still no results, try to find any JSON object in the text
    if not results:
        # Look for patterns like {"response": ..., "reasoning": ...}
        json_pattern = re.search(r'\{[^{}]*"response"[^{}]*\}', text, re.DOTALL)
        if json_pattern:
            try:
                results.append(json.loads(json_pattern.group()))
            except json.JSONDecodeError:
                pass
    
    return results or None


# Few-shot examples for IMO grading
_FEW_SHOT_EXAMPLES = """
Example 1 (Full Credit):
Problem: Find all positive integers n such that n^2 + 3n + 2 is divisible by 4.
Solution: We can factor n^2 + 3n + 2 = (n+1)(n+2). For this to be divisible by 4, either (n+1) or (n+2) must be even, and at least one must be divisible by 2. Since n+1 and n+2 are consecutive integers, one is even. For the product to be divisible by 4, we need the even number to be divisible by 4, or both to be even (impossible for consecutive integers). So we need n+1 ≡ 0 (mod 4) or n+2 ≡ 0 (mod 4), meaning n ≡ 3 (mod 4) or n ≡ 2 (mod 4).
Grading Guidelines: Award 7 points for correct answer (n ≡ 2 or 3 mod 4). Award partial credit: 2 points for correct factorization, 2 points for recognizing need for divisibility by 4, 3 points for correct modular analysis.
Student Answer: n^2 + 3n + 2 = (n+1)(n+2). Since these are consecutive, one is even. For divisibility by 4, we need n ≡ 2 or 3 (mod 4).
Score: 7
Reasoning: The student correctly factored the expression, recognized the consecutive integer property, and arrived at the correct answer with proper modular arithmetic.

Example 2 (Full Credit):
Problem: Prove that for any positive integer n, n^3 + 2n is divisible by 3.
Solution: We use induction. Base case: n=1, 1^3 + 2(1) = 3, divisible by 3. Inductive step: assume k^3 + 2k divisible by 3. Then (k+1)^3 + 2(k+1) = k^3 + 3k^2 + 3k + 1 + 2k + 2 = k^3 + 2k + 3(k^2 + k + 1). By induction hypothesis, k^3 + 2k divisible by 3, and 3(k^2 + k + 1) clearly divisible by 3. Thus divisible by 3.
Grading Guidelines: Award 7 points for complete proof. Deduct 2 points for missing base case, 3 points for errors in inductive step, 2 points for lack of clarity.
Student Answer: By induction. Base case n=1 works. Assume true for k, then (k+1)^3 + 2(k+1) = k^3 + 3k^2 + 3k + 1 + 2k + 2 = (k^3 + 2k) + 3(k^2 + k + 1), both terms divisible by 3.
Score: 7
Reasoning: The student provided a complete and correct inductive proof with proper base case and inductive step.

Example 3 (Partial Credit - Incomplete):
Problem: Prove that the sum of the first n positive integers is n(n+1)/2.
Solution: Use induction. Base case: n=1, sum is 1 = 1(2)/2. Inductive step: assume true for k, then sum to k+1 is k(k+1)/2 + (k+1) = (k+1)(k/2 + 1) = (k+1)(k+2)/2.
Grading Guidelines: Award 7 points for complete proof. Award 4 points for correct base case and setup but algebraic error in inductive step. Award 2 points for only correct base case.
Student Answer: Base case: n=1, sum is 1 = 1(2)/2. For induction, assume true for k. Then sum to k+1 is k(k+1)/2 + k+1.
Score: 4
Reasoning: The student correctly established the base case and set up the inductive step properly, but did not complete the algebraic simplification to show the formula holds for k+1.

Example 4 (Partial Credit - Minor Error):
Problem: Find the remainder when 2^100 is divided by 3.
Solution: Note that 2 ≡ -1 (mod 3), so 2^100 ≡ (-1)^100 ≡ 1 (mod 3). The remainder is 1.
Grading Guidelines: Award 7 points for correct answer with valid reasoning. Award 5 points for correct answer with minor computational error in reasoning. Award 2 points for correct approach but wrong answer.
Student Answer: Since 2 ≡ -1 (mod 3), we have 2^100 ≡ (-1)^100 = -1 ≡ 2 (mod 3). The remainder is 2.
Score: 5
Reasoning: The student correctly identified the modular equivalence and applied the exponent, but made a sign error (-1^100 = 1, not -1). The approach was correct but the final answer was wrong due to this error.

Example 5 (Zero Credit - Completely Wrong):
Problem: Prove that √2 is irrational.
Solution: Assume √2 = p/q where p,q are coprime integers. Then 2 = p²/q², so p² = 2q². Thus p² is even, so p is even. Write p = 2k. Then 4k² = 2q², so 2k² = q². Thus q² is even, so q is even. But then p and q are both even, contradicting coprimality. Therefore √2 is irrational.
Grading Guidelines: Award 7 points for complete proof. Award partial credit for correct setup (2 points), correct parity argument (3 points), reaching contradiction (2 points).
Student Answer: √2 is irrational because it cannot be written as a fraction. If we try to write it as p/q, we get a contradiction because the decimal never ends.
Score: 0
Reasoning: The student gave a vague, hand-wavy explanation without any mathematical rigor. No use of the standard proof by contradiction method, no mention of parity or coprimality.

Example 6 (Partial Credit - Good Start):
Problem: Show that for any triangle with sides a, b, c, we have a + b > c.
Solution: This is the triangle inequality. In any triangle, the sum of any two sides must be greater than the third side. This follows from the fact that the shortest path between two points is a straight line.
Grading Guidelines: Award 7 points for complete proof. Award 4 points for stating the triangle inequality correctly but without proof. Award 2 points for mentioning the concept without clear statement.
Student Answer: In a triangle, if you add two sides together, they must be longer than the third side. This is because going directly from one vertex to another is shorter than going through the third vertex.
Score: 4
Reasoning: The student correctly stated the triangle inequality and gave an intuitive geometric explanation, but did not provide a rigorous proof using the straight-line distance property.
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

GRADING PRINCIPLES:
1. Be precise and objective - base your score strictly on the grading guidelines
2. Award partial credit generously when the student shows understanding, even if incomplete
3. Distinguish between conceptual errors (major deductions) and computational slips (minor deductions)
4. Consider the student's reasoning process, not just the final answer
5. IMO problems are scored 0-7 points; 7 is perfect, 0 is completely wrong or blank

YOUR RESPONSIBILITIES:
1. Carefully read the problem, official solution, and grading guidelines
2. Evaluate the student's answer against the official solution
3. Identify what parts are correct, partially correct, or incorrect
4. Consider the grading guidelines for partial credit
5. Assign an appropriate score (0-7 points for IMO problems)
6. Provide brief reasoning explaining your grading decision

SCORING GUIDE:
- 7 points: Complete, correct solution with clear reasoning
- 5-6 points: Minor errors or gaps, but essentially correct approach
- 3-4 points: Significant progress but incomplete or with notable errors
- 1-2 points: Some relevant ideas but largely incorrect or incomplete
- 0 points: No meaningful progress or completely wrong approach

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

INSTRUCTIONS:
1. First, identify the key steps in the official solution
2. Check which steps the student completed correctly
3. Note any errors, omissions, or alternative valid approaches
4. Apply the grading guidelines to determine partial credit
5. Assign a final score between 0 and 7
6. Provide concise reasoning explaining your grading decision

Respond ONLY in JSON format with the following schema:
<json>
{{
    "response": "<numerical_score>",
    "reasoning": "<brief explanation of grading decision>"
}}
</json>

IMPORTANT: 
- The "response" field must contain ONLY a single digit from 0-7 (e.g., "7", "5", "0")
- The "reasoning" field should briefly explain why you assigned that score
- Do not include any text outside the JSON block"""

        return prompt

    def _validate_score(self, prediction: str) -> str:
        """Validate and normalize the score prediction.
        
        Returns a valid IMO score (0-7) as a string.
        """
        if prediction is None:
            return "0"
        
        # Convert to string and strip whitespace
        prediction = str(prediction).strip()
        
        # Handle common patterns like "Score: 7", "7 points", "7/7", "score of 5"
        # First try to find a standalone single digit 0-7 (word boundary)
        number_match = re.search(r'\b([0-7])\b', prediction)
        if number_match:
            return number_match.group(1)
        
        # Try to find patterns like "score: 5" or "points: 3"
        score_pattern = re.search(r'(?:score|points?)[:\s]+(\d+)', prediction, re.IGNORECASE)
        if score_pattern:
            score = score_pattern.group(1)
            try:
                score_int = int(score)
                if 0 <= score_int <= 7:
                    return str(score_int)
                if score_int > 7:
                    return "7"
                if score_int < 0:
                    return "0"
            except ValueError:
                pass
        
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
        last_text = ""
        
        try:
            if msg_history and len(msg_history) > 0:
                last_text = msg_history[-1].get("text", "")
                extracted = _extract_jsons(last_text)
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
            self.log_fn(f"Error extracting prediction from JSON: {e}")
        
        # If no valid prediction from JSON, try to extract from raw text
        if prediction == "0" and last_text:
            # Try to find a standalone single digit (IMO score)
            number_match = re.search(r'\b([0-7])\b', last_text)
            if number_match:
                prediction = number_match.group(1)
            else:
                # Fallback to any digit
                number_match = re.search(r'\d+', last_text)
                if number_match:
                    prediction = number_match.group()

        # Validate and normalize the score
        prediction = self._validate_score(prediction)
        
        # Log reasoning if available
        if reasoning:
            self.log_fn(f"Grading reasoning: {reasoning[:200]}")

        return str(prediction), msg_history
