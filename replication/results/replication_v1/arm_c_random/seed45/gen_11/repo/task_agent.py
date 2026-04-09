"""
Task agent: solves a given task with chain-of-thought reasoning and self-reflection.

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
FEW_SHOT_EXAMPLES = """
Example 1:
Problem: Find all positive integers n such that n^2 + 3n + 2 is divisible by 4.
Solution: n^2 + 3n + 2 = (n+1)(n+2). For divisibility by 4, either n+1 or n+2 must be even, and one of them must be divisible by 4. This happens when n ≡ 0 or 3 (mod 4).
Grading Guidelines: Award 1 point for factoring, 1 point for analyzing cases, 1 point for correct answer.
Student Answer: "I factored it as (n+1)(n+2). Since these are consecutive integers, one is even. For divisibility by 4, we need one factor divisible by 4. This happens when n=3,7,11,... or n=0,4,8,... So n ≡ 0 or 3 (mod 4)."
Grade: {"score": 3, "max_score": 3, "rationale": "Complete solution with correct factoring, case analysis, and answer."}

Example 2:
Problem: Prove that the sum of two odd numbers is even.
Solution: Let the odd numbers be 2k+1 and 2m+1. Their sum is 2k+1+2m+1 = 2(k+m+1), which is even.
Grading Guidelines: Award 1 point for setting up odd number representation, 1 point for algebraic manipulation, 1 point for conclusion.
Student Answer: "Odd numbers end in 1,3,5,7,9. Adding two odd numbers gives an even number. For example, 3+5=8 which is even."
Grade: {"score": 1, "max_score": 3, "rationale": "Student only provided examples without general proof. Missing algebraic representation and general reasoning."}

Example 3:
Problem: Prove that for any prime p > 3, p^2 - 1 is divisible by 24.
Solution: p^2 - 1 = (p-1)(p+1). Since p is odd and not divisible by 3, one of p-1, p, p+1 is divisible by 3. Since p is prime > 3, neither p-1 nor p+1 is divisible by p. Among three consecutive integers, one is divisible by 3. Since p is odd, both p-1 and p+1 are even, and one of them is divisible by 4. Thus (p-1)(p+1) is divisible by 8×3 = 24.
Grading Guidelines: Award 1 point for factoring, 1 point for divisibility by 8 argument, 1 point for divisibility by 3 argument, 1 point for combining to get 24.
Student Answer: "p^2 - 1 = (p-1)(p+1). Since p is odd, p-1 and p+1 are consecutive even integers, so one is divisible by 4 and the other by 2, giving divisibility by 8. Also, among p-1, p, p+1, one must be divisible by 3, and since p is prime > 3, it's not divisible by 3, so either p-1 or p+1 is. Therefore p^2 - 1 is divisible by 8×3 = 24."
Grade: {"score": 4, "max_score": 4, "rationale": "Excellent proof covering all required elements: factoring, divisibility by 8, divisibility by 3, and final combination."}

Example 4:
Problem: Find the number of ways to arrange 6 people around a circular table where rotations are considered the same.
Solution: For circular arrangements, we fix one person's position to account for rotational equivalence. The remaining 5 people can be arranged in 5! = 120 ways.
Grading Guidelines: Award 1 point for recognizing circular arrangement formula, 1 point for correct calculation, 1 point for final answer.
Student Answer: "There are 6! = 720 ways to arrange 6 people. But since it's a circle, we divide by 6 to get 120 ways."
Grade: {"score": 3, "max_score": 3, "rationale": "Correct answer with valid reasoning. Student correctly applied circular permutation formula (n-1)! or equivalently divided by n to account for rotational symmetry."}

Example 5:
Problem: Prove that √2 is irrational.
Solution: Assume √2 = p/q in lowest terms. Then 2q^2 = p^2, so p^2 is even, thus p is even. Let p = 2k. Then 2q^2 = 4k^2, so q^2 = 2k^2, making q even. Contradiction: p and q both even, not in lowest terms.
Grading Guidelines: Award 1 point for setting up proof by contradiction, 1 point for showing p is even, 1 point for showing q is even, 1 point for reaching contradiction.
Student Answer: "Suppose √2 is rational. Then it can be written as a fraction p/q. Squaring gives 2 = p^2/q^2, so p^2 = 2q^2. This means p^2 is even, so p is even. But then q must also be even, which contradicts p/q being in lowest terms. Therefore √2 is irrational."
Grade: {"score": 4, "max_score": 4, "rationale": "Complete proof by contradiction with all logical steps clearly presented."}
"""


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with reasoning and reflection.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Step 1: Initial grading with chain-of-thought
        instruction = f"""You are an expert mathematical grader for the International Mathematical Olympiad (IMO).

Your task is to grade a student's answer to a mathematical problem.

{FEW_SHOT_EXAMPLES}

Now grade the following problem:

Domain: {inputs.get('domain', 'Mathematics')}

Problem:
{inputs.get('problem', '')}

Official Solution:
{inputs.get('solution', '')}

Grading Guidelines:
{inputs.get('grading_guidelines', '')}

Student Answer:
{inputs.get('student_answer', '')}

Think step by step:
1. Analyze what the student did correctly according to the official solution
2. Identify any errors, gaps, or missing steps
3. Compare against the grading guidelines
4. Determine the score and provide detailed rationale

Respond in JSON format with the following schema:
<json>
{{
    "thinking": "Your detailed step-by-step analysis here",
    "score": <numerical score>,
    "max_score": <maximum possible score>,
    "rationale": "Detailed explanation of why this score was awarded",
    "response": "<score>/<max_score> - <brief summary>"
}}
</json>"""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON
        prediction = "None"
        try:
            extracted = _extract_jsons(msg_history[-1]["text"])
            if extracted:
                result = extracted[-1]
                if "response" in result:
                    prediction = result["response"]
                elif "score" in result and "max_score" in result:
                    prediction = f"{result['score']}/{result['max_score']}"
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        # Step 2: Self-reflection to verify the grade
        if prediction != "None" and len(msg_history) >= 2:
            reflection_msg = f"""Review your grading above carefully. Perform a critical self-assessment:

1. STRICTNESS CHECK: Did you award points too generously? Re-examine the student's work against the official solution point by point.
2. ERROR DETECTION: Did you overlook any mistakes, gaps, or logical flaws in the student's reasoning?
3. GUIDELINE ALIGNMENT: Does your score precisely match the grading guidelines? Count the specific points awarded vs. the guidelines.
4. CONSISTENCY: Would an expert IMO grader give the same score? Be honest about any leniency.
5. PARTIAL CREDIT: Did you award partial credit appropriately? Too much or too little?

IMPORTANT: If your initial grade was too lenient or strict, you MUST revise it. Do not simply confirm your first assessment.

Respond in JSON format:
<json>
{{
    "reflection": "Detailed self-review addressing each check above",
    "grade_revised": true/false,
    "revised_score": <score>,
    "revised_max_score": <max_score>,
    "revision_reason": "Explanation if grade changed, or 'No change - initial grade accurate'",
    "final_response": "<score>/<max_score> - <brief summary>"
}}
</json>"""
            
            reflection_response, msg_history, _ = get_response_from_llm(
                msg=reflection_msg,
                model=self.model,
                msg_history=msg_history,
            )
            
            # Try to extract revised prediction
            try:
                extracted = _extract_jsons(msg_history[-1]["text"])
                if extracted:
                    result = extracted[-1]
                    if "final_response" in result:
                        prediction = result["final_response"]
                    elif "revised_score" in result and "revised_max_score" in result:
                        prediction = f"{result['revised_score']}/{result['revised_max_score']}"
            except Exception as e:
                self.log_fn(f"Error extracting revised prediction: {e}")

        return str(prediction), msg_history
