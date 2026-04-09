"""
Task agent: solves a given task with chain-of-thought reasoning.

Reimplemented from facebookresearch/HyperAgents task_agent.py.
Enhanced with structured reasoning and few-shot examples for better performance.

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
    Also attempts to extract JSON from markdown code blocks as fallback.
    """
    results = []
    search_from = 0
    
    # Primary: Extract from <json>...</json> tags
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
            # Try to clean up common JSON issues
            try:
                # Remove trailing commas before closing braces/brackets
                cleaned = re.sub(r',(\s*[}\]])', r'\1', inner)
                results.append(json.loads(cleaned))
            except json.JSONDecodeError:
                continue
    
    # Fallback: Extract from markdown code blocks if no results yet
    if not results:
        code_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
        for match in re.finditer(code_block_pattern, text, re.DOTALL):
            try:
                inner = match.group(1).strip()
                results.append(json.loads(inner))
            except json.JSONDecodeError:
                # Try to clean up common JSON issues
                try:
                    cleaned = re.sub(r',(\s*[}\]])', r'\1', inner)
                    results.append(json.loads(cleaned))
                except json.JSONDecodeError:
                    continue
    
    return results or None


# Few-shot examples for IMO grading
FEW_SHOT_EXAMPLES = """
Example 1:
Problem: Find the sum of 2 + 3.
Solution: The sum is 5.
Grading Guidelines: Award 1 point for correct answer.
Student Answer: 5
Analysis: The student's answer matches the correct solution exactly.
Score: 1/1

Example 2:
Problem: Solve x^2 = 4.
Solution: x = 2 or x = -2.
Grading Guidelines: Award 1 point for each correct solution (max 2 points).
Student Answer: x = 2
Analysis: The student found one correct solution but missed the negative solution.
Score: 1/2

Example 3:
Problem: Prove that the sum of angles in a triangle is 180 degrees.
Solution: [Detailed geometric proof]
Grading Guidelines: Award 1 point for stating the theorem, 2 points for correct proof steps.
Student Answer: The angles add up to 180 because it's a triangle.
Analysis: The student stated the result but provided no valid proof or reasoning.
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
        # Extract fields for structured prompting
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = f"""You are an expert grader for {domain} problems. Your task is to evaluate a student's answer against the correct solution using the provided grading guidelines.

{FEW_SHOT_EXAMPLES}

Now evaluate the following:

Problem:
```
{problem}
```

Correct Solution:
```
{solution}
```

Grading Guidelines:
```
{grading_guidelines}
```

Student's Answer:
```
{student_answer}
```

Think step by step:
1. Analyze what the problem is asking
2. Review the correct solution approach
3. Compare the student's answer to the correct solution
4. Check against each criterion in the grading guidelines
5. Determine the score

Respond in JSON format with the following schema:
<json>
{{
    "analysis": "Your detailed analysis of the student's answer",
    "score_breakdown": "How points were awarded/deducted based on guidelines",
    "response": "The final score (e.g., '3/5' or '2 points')"
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
                last_json = extracted[-1]
                if "response" in last_json:
                    prediction = last_json["response"]
                elif "score" in last_json:
                    prediction = last_json["score"]
                # Log analysis if available
                if "analysis" in last_json:
                    self.log_fn(f"Analysis: {last_json['analysis'][:200]}...")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
