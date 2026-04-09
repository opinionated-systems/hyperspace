"""
Task agent: solves a given task with chain-of-thought reasoning and robust error handling.

Enhanced version with:
- Chain-of-thought reasoning for better IMO grading accuracy
- Few-shot examples for grading tasks
- Retry logic with fallback extraction
- Support for multiple answer formats

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


def _extract_any_answer(text: str) -> str | None:
    """Extract answer from text using multiple fallback strategies."""
    # Try JSON extraction first
    jsons = _extract_jsons(text)
    if jsons:
        for j in jsons:
            if "response" in j:
                return str(j["response"])
            if "answer" in j:
                return str(j["answer"])
            if "grade" in j:
                return str(j["grade"])
    
    # Try to find explicit answer markers
    patterns = [
        r'(?:final answer|answer|grade|score|result)[:\s]+([\w\s\-\+\.]+)',
        r'(?:^|\n)([\w\s\-\+\.]+)(?:\n|$)',
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            # Return the last match (usually the conclusion)
            return matches[-1].strip()
    
    return None


# Few-shot examples for IMO grading
GRADING_EXAMPLES = """
Example 1:
Problem: Prove that the sum of angles in a triangle is 180 degrees.
Solution: Draw a line parallel to one side through the opposite vertex...
Student Answer: The angles add up to 180 because of parallel lines.
Grade: Partial (2/7) - The student has the right idea but lacks rigor.

Example 2:
Problem: Find all positive integers n such that n^2 + n + 1 is prime.
Solution: Testing small values: n=1 gives 3 (prime), n=2 gives 7 (prime), n=3 gives 13 (prime), n=4 gives 21 (not prime). For n>4, n^2 + n + 1 > n^2 > n+1, and we can show it's composite.
Student Answer: n = 1, 2, 3
Grade: Correct (7/7) - The student correctly identified all solutions.
"""


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.max_retries = 3

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with chain-of-thought reasoning.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Build structured prompt with chain-of-thought
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = f"""You are an expert {domain} grader for the International Mathematical Olympiad (IMO).

Your task is to grade a student's answer to a problem.

{GRADING_EXAMPLES}

Now grade this problem:

PROBLEM:
{problem}

OFFICIAL SOLUTION:
{solution}

GRADING GUIDELINES:
{guidelines}

STUDENT'S ANSWER:
{student_answer}

Think step by step:
1. Analyze what the student did correctly
2. Identify any errors or missing steps
3. Compare against the official solution and grading guidelines
4. Determine the appropriate grade

Respond in JSON format:
<json>
{{
    "reasoning": "Your detailed analysis here",
    "response": "The final grade (e.g., 'Correct', 'Partial (4/7)', 'Incorrect')"
}}
</json>"""

        # Try with retries
        for attempt in range(self.max_retries):
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=[],
                )
                
                # Extract prediction from JSON
                prediction = None
                extracted = _extract_jsons(msg_history[-1]["text"])
                if extracted:
                    last = extracted[-1]
                    prediction = last.get("response") or last.get("answer") or last.get("grade")
                
                # Fallback extraction if JSON parsing fails
                if prediction is None:
                    prediction = _extract_any_answer(msg_history[-1]["text"])
                
                if prediction is not None:
                    return str(prediction), msg_history
                
                # If no prediction found, retry with clearer instruction
                if attempt < self.max_retries - 1:
                    instruction += "\n\nIMPORTANT: You must provide a 'response' field in your JSON."
                    
            except Exception as e:
                self.log_fn(f"Attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    break
        
        # Final fallback: return the raw assistant text
        try:
            return str(msg_history[-1]["text"]), msg_history
        except:
            return "Error: Could not extract prediction", msg_history
