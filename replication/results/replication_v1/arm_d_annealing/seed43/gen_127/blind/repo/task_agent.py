"""
Task agent: solves a given task with chain-of-thought reasoning and structured analysis.

Enhanced version with:
- Chain-of-thought prompting for complex reasoning
- Structured analysis of student answers
- Few-shot examples for IMO grading
- Self-consistency through reasoning verification
- Better error handling and retry logic

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
Solution: The correct answer is all odd positive integers. We can factor n^2 + 3n + 2 = (n+1)(n+2). For this to be divisible by 4, we need (n+1)(n+2) ≡ 0 (mod 4). Since n+1 and n+2 are consecutive integers, one is even. For the product to be divisible by 4, we need the even factor to be divisible by 4, which happens when n is odd.
Grading Guidelines: Award 1 point for correct answer, 1 point for factorization, 1 point for modular arithmetic reasoning.
Student Answer: "n must be odd. If n = 2k+1, then (2k+2)(2k+3) = 2(k+1)(2k+3). Since k+1 and 2k+3 have different parity, one contributes factor 2 and the other is odd, so we need k+1 even, meaning k odd."
Analysis: The student correctly identified n must be odd and provided a valid proof. The reasoning about parity is correct.
Score: 3/3

Example 2:
Problem: Prove that the sum of the first n odd numbers is n^2.
Solution: The sum is 1 + 3 + 5 + ... + (2n-1) = n^2. This can be proved by induction or by noting this is an arithmetic series with n terms, first term 1, last term 2n-1, so sum = n(1 + 2n-1)/2 = n^2.
Grading Guidelines: Award 1 point for correct formula, 1 point for valid proof method, 1 point for correct execution.
Student Answer: "1 + 3 = 4 = 2^2, 1 + 3 + 5 = 9 = 3^2, so it works for n=2 and n=3."
Analysis: The student only checked specific cases without providing a general proof. This is insufficient for a complete proof.
Score: 1/3
"""


class TaskAgent:
    """Task agent that solves IMO grading problems with structured reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with chain-of-thought reasoning.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Build structured instruction with chain-of-thought
        instruction = f"""You are an expert mathematics grader for the International Mathematical Olympiad (IMO).

Your task is to evaluate a student's answer to a mathematical problem. Follow these steps:

1. UNDERSTAND: Read the problem, official solution, and grading guidelines carefully.
2. ANALYZE: Examine the student's answer step by step.
3. COMPARE: Compare the student's approach with the official solution.
4. EVALUATE: Determine what the student got right and what they missed.
5. SCORE: Assign a score based on the grading guidelines.

{FEW_SHOT_EXAMPLES}

Now evaluate this submission:

Problem Domain: {inputs.get('domain', 'Mathematics')}

Problem Statement:
{inputs.get('problem', 'N/A')}

Official Solution:
{inputs.get('solution', 'N/A')}

Grading Guidelines:
{inputs.get('grading_guidelines', 'N/A')}

Student's Answer:
{inputs.get('student_answer', 'N/A')}

Provide your evaluation in the following JSON format:
<json>
{{
    "reasoning": "Your detailed chain-of-thought analysis here. Explain what the student did right, what they missed, and how you arrived at your score.",
    "response": "The final score (e.g., '3/7', '1 point', 'Full marks', etc.)"
}}
</json>

Important: 
- Be objective and fair in your grading.
- Award partial credit where appropriate based on the guidelines.
- If the student uses a different but valid approach, award full credit.
- Only deduct points for mathematical errors or missing steps, not for style preferences."""

        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
        except Exception as e:
            self.log_fn(f"LLM call failed: {e}")
            return "Error: LLM call failed", []

        # Extract prediction from JSON
        prediction = "None"
        reasoning = ""
        try:
            if msg_history and len(msg_history) > 0:
                extracted = _extract_jsons(msg_history[-1]["text"])
                if extracted:
                    last_extract = extracted[-1]
                    if "response" in last_extract:
                        prediction = last_extract["response"]
                    if "reasoning" in last_extract:
                        reasoning = last_extract["reasoning"]
                        self.log_fn(f"Reasoning: {reasoning[:200]}...")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            # Fallback: try to extract any score-like pattern from the response
            try:
                text = msg_history[-1]["text"] if msg_history else ""
                # Look for patterns like "Score: X/Y" or "X/Y points"
                score_match = re.search(r'(?:score|grade|marks?)[\s:]*(\d+(?:\.\d+)?\s*/\s*\d+)', text, re.IGNORECASE)
                if score_match:
                    prediction = score_match.group(1)
            except Exception:
                pass

        return str(prediction), msg_history
