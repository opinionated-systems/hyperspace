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

Example 3:
Problem: Let ABC be a triangle with AB = AC. Let D be the midpoint of BC. Prove that AD is perpendicular to BC.
Solution: Since AB = AC, triangle ABC is isosceles with apex A. In an isosceles triangle, the median from the apex to the base is also the altitude. Therefore, AD ⊥ BC. Alternatively, using coordinates: place D at origin, B at (-a, 0), C at (a, 0). Then A is at (0, h) for some h > 0. Vector AD = (0, -h), vector BC = (2a, 0). Their dot product is 0, so AD ⊥ BC.
Grading Guidelines: Award 7 points for complete proof. Award 4-5 points for correct approach with minor gaps. Award 2-3 points for stating the property without proof. Award 0-1 point for irrelevant or incorrect work.
Student Answer: Since AB = AC, the triangle is isosceles. The median from A to BC is also the altitude, so they are perpendicular.
Score: 5

Example 4:
Problem: Find the sum of all positive integers n ≤ 100 such that n is divisible by 3 or 5.
Solution: Use inclusion-exclusion. Sum of multiples of 3: 3 + 6 + ... + 99 = 3(1 + 2 + ... + 33) = 3 × 33 × 34/2 = 1683. Sum of multiples of 5: 5 + 10 + ... + 100 = 5(1 + 2 + ... + 20) = 5 × 20 × 21/2 = 1050. Sum of multiples of 15: 15 + 30 + ... + 90 = 15(1 + 2 + ... + 6) = 15 × 6 × 7/2 = 315. Total = 1683 + 1050 - 315 = 2418.
Grading Guidelines: Award 7 points for correct answer (2418). Award 5-6 points for correct method with arithmetic error. Award 3-4 points for correct inclusion-exclusion setup but incorrect execution. Award 1-2 points for partial progress.
Student Answer: Multiples of 3: 3+6+...+99 = 1683. Multiples of 5: 5+10+...+100 = 1050. Total = 1683 + 1050 = 2733.
Score: 4
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

Grading Principles:
- Be fair and consistent with the grading guidelines
- Award partial credit for correct reasoning even if the final answer is wrong
- Deduct points for missing steps, logical errors, or incomplete proofs
- Consider the difficulty level of the problem
- A score of 7 means a complete, correct solution
- A score of 0 means no meaningful progress or completely wrong approach
- Partial credit (1-6) should reflect the proportion of the solution completed correctly

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
- Compare the student answer to the official solution step by step
- Identify what parts are correct, partially correct, or incorrect
- Consider the grading guidelines for partial credit
- Look for key insights, correct methods, and valid reasoning
- Be generous with partial credit for good mathematical thinking
- Assign an appropriate score based on the IMO 0-7 scale

Respond in JSON format with the following schema:
<json>
{{
    "response": "<numerical_score>"
}}
</json>

The response field should contain only the numerical score (e.g., "7", "5", "0", etc.)."""

        return prompt

    def _validate_score(self, prediction: str) -> str:
        """Validate and normalize the score prediction."""
        if prediction is None:
            return "0"
        
        # Convert to string and strip whitespace
        prediction = str(prediction).strip()
        
        # Try to extract a number from the prediction
        # Handle cases like "Score: 7", "7 points", "7/7", etc.
        number_match = re.search(r'\d+', prediction)
        if number_match:
            score = number_match.group()
            # Validate it's a reasonable IMO score (0-7)
            try:
                score_int = int(score)
                if 0 <= score_int <= 7:
                    return str(score_int)
                elif score_int > 7:
                    # Cap at 7 for IMO problems
                    return "7"
                elif score_int < 0:
                    # Floor at 0
                    return "0"
            except ValueError:
                pass
        
        # If we can't extract a valid number, return the original
        return prediction if prediction else "0"

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
