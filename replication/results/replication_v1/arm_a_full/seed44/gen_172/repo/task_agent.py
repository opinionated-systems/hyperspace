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
from typing import Any

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


def _extract_json_from_markdown(text: str) -> list[dict] | None:
    """Extract JSON objects from markdown code blocks.
    
    Handles ```json ... ``` and ``` ... ``` blocks.
    """
    results = []
    # Match ```json ... ``` or ``` ... ``` blocks
    pattern = r'```(?:json)?\s*(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    for match in matches:
        try:
            results.append(json.loads(match.strip()))
        except json.JSONDecodeError:
            continue
    return results or None


def _extract_any_json(text: str) -> list[dict] | None:
    """Extract JSON objects using multiple strategies.
    
    Tries <json> tags first, then markdown code blocks, then raw JSON objects.
    """
    # Try <json> tags first
    results = _extract_jsons(text)
    if results:
        return results
    
    # Try markdown code blocks
    results = _extract_json_from_markdown(text)
    if results:
        return results
    
    # Try to find raw JSON objects (objects wrapped in {})
    # This is a last resort and may be less reliable
    results = []
    try:
        # Find all potential JSON objects using brace matching
        depth = 0
        start = -1
        for i, char in enumerate(text):
            if char == '{':
                if depth == 0:
                    start = i
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0 and start != -1:
                    try:
                        obj = json.loads(text[start:i+1])
                        results.append(obj)
                    except json.JSONDecodeError:
                        pass
                    start = -1
    except Exception:
        pass
    
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
Solution: Since AB = AC, triangle ABC is isosceles with apex A. In an isosceles triangle, the median from the apex to the base is also the altitude. Therefore AD ⊥ BC.
Grading Guidelines: Award 7 points for complete proof. Award partial credit: 3 points for recognizing isosceles property, 4 points for knowing median=altitude property or proving it.
Student Answer: Triangle ABC is isosceles so AD is perpendicular to BC.
Score: 4
"""


class TaskAgent:
    """Task agent that solves IMO grading problems with enhanced robustness."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.max_score = 7  # IMO problems are typically scored 0-7

    def _build_prompt(self, inputs: dict) -> str:
        """Build a structured prompt for the grading task with chain-of-thought reasoning."""
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
- First, analyze the student answer step by step (chain-of-thought)
- Compare the student answer to the official solution
- Identify what parts are correct, partially correct, or incorrect
- Consider the grading guidelines for partial credit
- Assign an appropriate score between 0 and 7 (inclusive)
- Be precise: full credit (7) only for complete, correct solutions
- Award partial credit generously for correct reasoning with gaps
- Double-check your score against the grading guidelines before finalizing

Respond in JSON format with the following schema:
<json>
{{
    "response": "<numerical_score>",
    "reasoning": "<brief explanation of the score>"
}}
</json>

The response field should contain only the numerical score (e.g., "7", "5", "0", etc.).
The reasoning field should contain a brief explanation of your grading decision."""

        return prompt

    def _validate_score(self, prediction: str | int | None, max_score: int = 7) -> str:
        """Validate and normalize the score prediction.
        
        Args:
            prediction: The predicted score (string, int, or None)
            max_score: Maximum allowed score (default 7 for IMO)
            
        Returns:
            Validated score as a string
        """
        if prediction is None:
            return "0"
        
        # Convert to string and strip whitespace
        prediction_str = str(prediction).strip()
        
        # Try to extract a number from the prediction
        # Handle cases like "Score: 7", "7 points", "7/7", "[7]", etc.
        number_match = re.search(r'\d+', prediction_str)
        if number_match:
            score = number_match.group()
            # Validate it's a reasonable IMO score (0-max_score)
            try:
                score_int = int(score)
                if 0 <= score_int <= max_score:
                    return str(score_int)
                elif score_int > max_score:
                    # Cap at max_score if exceeded
                    return str(max_score)
                else:
                    return "0"
            except ValueError:
                pass
        
        # If we can't extract a valid number, return "0"
        return "0"

    def _extract_score_from_response(self, msg_history: list[dict]) -> tuple[str, str]:
        """Extract score and reasoning from the LLM response.
        
        Args:
            msg_history: Message history from LLM call
            
        Returns:
            Tuple of (score, reasoning)
        """
        if not msg_history:
            return "0", "No response received"
        
        last_text = msg_history[-1].get("text", "")
        
        # Try to extract JSON
        extracted = _extract_any_json(last_text)
        
        if extracted:
            last_json = extracted[-1]
            
            # Try to get score from "response" field
            if isinstance(last_json, dict):
                score = last_json.get("response")
                reasoning = last_json.get("reasoning", "")
                
                if score is not None:
                    return self._validate_score(score), reasoning
                
                # Try other common field names
                for key in ["score", "grade", "points", "value", "result"]:
                    if key in last_json:
                        return self._validate_score(last_json[key]), reasoning
        
        # Fallback: try to extract any number from the response
        # Look for patterns like "Score: 7", "The score is 5", etc.
        patterns = [
            r'[Ss]core[:\s]+(\d+)',
            r'[Gg]rade[:\s]+(\d+)',
            r'[Pp]oints[:\s]+(\d+)',
            r'[Aa]ward[:\s]+(\d+)',
            r'[Ff]inal score[:\s]+(\d+)',
            r'[Tt]otal[:\s]+(\d+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, last_text)
            if match:
                return self._validate_score(match.group(1)), "Extracted from text pattern"
        
        # Last resort: find any standalone number that could be a score
        # Look for numbers 0-7 that appear to be scores
        numbers = re.findall(r'\b([0-7])\b', last_text)
        if numbers:
            # Return the last number found (usually the final score)
            return self._validate_score(numbers[-1]), "Extracted from standalone number"
        
        return "0", "Could not extract valid score"

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with retry logic.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        instruction = self._build_prompt(inputs)
        
        max_retries = 3
        for attempt in range(max_retries):
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )

            # Extract prediction and reasoning
            prediction, reasoning = self._extract_score_from_response(msg_history)
            
            # If we got a valid score, return it
            if prediction != "0" or attempt == max_retries - 1:
                self.log_fn(f"Graded score: {prediction}, Reasoning: {reasoning[:100]}...")
                return str(prediction), msg_history
            
            # If score is 0, might be a parsing issue - retry with a reminder
            self.log_fn(f"Attempt {attempt + 1} failed to extract valid score, retrying...")
            instruction = self._build_prompt(inputs) + "\n\nIMPORTANT: Please ensure your response follows the exact JSON format specified above with a valid numerical score in the 'response' field."
        
        # Should never reach here, but just in case
        return "0", msg_history
