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
import time

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


# Few-shot examples for IMO grading with chain-of-thought reasoning
_FEW_SHOT_EXAMPLES = """
Example 1 (Complete Solution - 7 points):
Problem: Find all positive integers n such that n^2 + 3n + 2 is divisible by 4.
Solution: We can factor n^2 + 3n + 2 = (n+1)(n+2). For this to be divisible by 4, either (n+1) or (n+2) must be even, and at least one must be divisible by 2. Since n+1 and n+2 are consecutive integers, one is even. For the product to be divisible by 4, we need the even number to be divisible by 4, or both to be even (impossible for consecutive integers). So we need n+1 ≡ 0 (mod 4) or n+2 ≡ 0 (mod 4), meaning n ≡ 3 (mod 4) or n ≡ 2 (mod 4).
Grading Guidelines: Award 7 points for correct answer (n ≡ 2 or 3 mod 4). Award partial credit: 2 points for correct factorization, 2 points for recognizing need for divisibility by 4, 3 points for correct modular analysis.
Student Answer: n^2 + 3n + 2 = (n+1)(n+2). Since these are consecutive, one is even. For divisibility by 4, we need n ≡ 2 or 3 (mod 4).

Reasoning: The student correctly factored the expression (2 points), recognized that consecutive integers means one is even (2 points), and correctly determined the modular conditions for divisibility by 4 (3 points). Total: 7 points.
<json>
{
    "response": "7"
}
</json>

Example 2 (Complete Proof - 7 points):
Problem: Prove that for any positive integer n, n^3 + 2n is divisible by 3.
Solution: We use induction. Base case: n=1, 1^3 + 2(1) = 3, divisible by 3. Inductive step: assume k^3 + 2k divisible by 3. Then (k+1)^3 + 2(k+1) = k^3 + 3k^2 + 3k + 1 + 2k + 2 = k^3 + 2k + 3(k^2 + k + 1). By induction hypothesis, k^3 + 2k divisible by 3, and 3(k^2 + k + 1) clearly divisible by 3. Thus divisible by 3.
Grading Guidelines: Award 7 points for complete proof. Deduct 2 points for missing base case, 3 points for errors in inductive step, 2 points for lack of clarity.
Student Answer: By induction. Base case n=1 works. Assume true for k, then (k+1)^3 + 2(k+1) = k^3 + 3k^2 + 3k + 1 + 2k + 2 = (k^3 + 2k) + 3(k^2 + k + 1), both terms divisible by 3.

Reasoning: The student provided a complete induction proof with correct base case (no deduction), correct inductive hypothesis, and correct algebraic manipulation showing divisibility (no errors). The proof is clear and complete. Total: 7 points.
<json>
{
    "response": "7"
}
</json>

Example 3 (Partial Credit - 4 points):
Problem: Prove that the sum of the first n positive integers is n(n+1)/2.
Solution: Use mathematical induction. Base case: n=1, sum is 1 = 1(2)/2. Inductive step: assume true for k, so 1+2+...+k = k(k+1)/2. Then 1+2+...+k+(k+1) = k(k+1)/2 + (k+1) = (k+1)(k/2 + 1) = (k+1)(k+2)/2, which is the formula for n=k+1.
Grading Guidelines: Award 7 points for complete proof. Award 3 points for correct base case only. Award 4 points for correct inductive step setup but errors in algebra. Award 6 points for minor algebraic error in inductive step.
Student Answer: Base case: n=1, 1 = 1(2)/2 = 1. Inductive step: Assume 1+2+...+k = k(k+1)/2. Then adding (k+1): sum = k(k+1)/2 + (k+1) = (k+1)(k+2)/2.

Reasoning: The student correctly stated the base case (3 points). For the inductive step, they set up correctly but made an algebraic error: the final simplification should show (k+1)(k+2)/2, but they didn't show the intermediate step clearly. The algebra is actually correct in this case, but the presentation lacks clarity in connecting the steps. However, looking more carefully, the algebra is correct. Let me reconsider: the student has correct base case (3 points) and correct inductive step (4 points), but lacks some clarity in presentation. Total: 7 points.
<json>
{
    "response": "7"
}
</json>

Example 4 (Incorrect Answer - 0 points):
Problem: Find the remainder when 2^100 is divided by 3.
Solution: Note that 2 ≡ -1 (mod 3), so 2^100 ≡ (-1)^100 ≡ 1 (mod 3). The remainder is 1.
Grading Guidelines: Award 7 points for correct answer with valid reasoning. Award 2 points for recognizing pattern in powers of 2 mod 3. Award 0 points for incorrect answer.
Student Answer: 2^100 is even, so remainder is 2 when divided by 3.

Reasoning: The student's reasoning is fundamentally flawed. While 2^100 is indeed even, this has no bearing on the remainder when divided by 3. The correct answer is 1 (since 2 ≡ -1 mod 3, and (-1)^100 = 1). The student shows no understanding of modular arithmetic. Total: 0 points.
<json>
{
    "response": "0"
}
</json>

Example 5 (Partial Credit - 3 points):
Problem: Prove that for any prime p > 3, p^2 - 1 is divisible by 24.
Solution: For prime p > 3, p is odd and not divisible by 3. Since p is odd, p^2 ≡ 1 (mod 8). Since p is not divisible by 3, p ≡ ±1 (mod 3), so p^2 ≡ 1 (mod 3). Thus p^2 - 1 is divisible by both 8 and 3, hence by 24.
Grading Guidelines: Award 7 points for complete proof. Award 2 points for recognizing p is odd. Award 2 points for showing p^2 ≡ 1 (mod 8). Award 3 points for showing divisibility by 3. Award 0-1 points for minimal progress.
Student Answer: Since p is prime and greater than 3, it must be odd. So p^2 is also odd, and p^2 - 1 is even.

Reasoning: The student correctly recognized that p > 3 is odd (2 points). However, they only concluded that p^2 - 1 is even, which is insufficient. They did not show that p^2 ≡ 1 (mod 8) or address divisibility by 3. The reasoning is incomplete. Total: 2 points for the observation about odd primes.
<json>
{
    "response": "2"
}
</json>
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
4. Provide your reasoning and evaluation in the specified format

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
1. First, think step-by-step about the student's answer
2. Compare it carefully to the official solution
3. Identify what parts are correct, partially correct, or incorrect
4. Consider the grading guidelines for partial credit
5. Provide brief reasoning explaining your evaluation (1-3 sentences)
6. Assign an appropriate integer score between 0 and 7 (inclusive)
7. Respond with your reasoning followed by the JSON format

IMPORTANT: Your response must follow this exact format:

Reasoning: <your brief reasoning for the score>
<json>
{{
    "response": "<numerical_score>"
}}
</json>

The "response" field must contain ONLY a single integer between 0 and 7, as a string (e.g., "7", "5", "0", "3"). The reasoning should come before the JSON block and help justify your score."""

        return prompt

    def _validate_score(self, prediction: str) -> str:
        """Validate and normalize the score prediction.
        
        Handles various formats like "Score: 7", "7 points", "7/7", "7.0", etc.
        Ensures the score is within valid IMO range (0-7).
        Also handles negative scores and out-of-range values gracefully.
        """
        if prediction is None:
            return "0"
        
        # Convert to string and strip whitespace
        prediction = str(prediction).strip()
        
        # Handle empty string
        if not prediction:
            return "0"
        
        # Handle fraction format like "7/7" - extract numerator
        if "/" in prediction:
            parts = prediction.split("/")
            if parts[0].strip().isdigit():
                score = parts[0].strip()
                try:
                    score_int = int(score)
                    if 0 <= score_int <= 7:
                        return str(score_int)
                except ValueError:
                    pass
        
        # Try to extract a number from the prediction
        # Handle cases like "Score: 7", "7 points", "7.0", etc.
        # Look for decimal numbers first, then integers
        decimal_match = re.search(r'-?\d+\.\d+', prediction)
        if decimal_match:
            try:
                score_float = float(decimal_match.group())
                score_int = int(round(score_float))
                # Clamp to valid range
                score_int = max(0, min(7, score_int))
                return str(score_int)
            except ValueError:
                pass
        
        # Try integer match (including negative numbers)
        number_match = re.search(r'-?\d+', prediction)
        if number_match:
            score = number_match.group()
            try:
                score_int = int(score)
                # Clamp to valid range [0, 7]
                score_int = max(0, min(7, score_int))
                return str(score_int)
            except ValueError:
                pass
        
        # If we can't extract a valid number, return "0"
        logger.warning(f"Could not extract valid score from '{prediction}', defaulting to 0")
        return "0"

    def _extract_prediction(self, msg_history: list[dict]) -> str:
        """Extract prediction from message history.
        
        Tries multiple strategies to extract a valid score:
        1. Extract from JSON response field (preferred)
        2. Extract from any numeric field in JSON
        3. Look for "Score: X" or similar patterns in reasoning section
        4. Extract from text using score-related patterns
        5. Extract any number as last resort
        
        Args:
            msg_history: List of message dicts with 'role' and 'text' keys
            
        Returns:
            Extracted prediction string (may need validation)
        """
        try:
            # Get the last assistant message
            last_msg = None
            for msg in reversed(msg_history):
                if msg.get("role") == "assistant":
                    last_msg = msg
                    break
            
            if last_msg is None:
                logger.warning("No assistant message found in history")
                return "0"
            
            last_text = last_msg.get("text", "")
            
            if not last_text:
                logger.warning("Empty assistant message")
                return "0"
            
            # Try to extract JSON first (most reliable)
            extracted = _extract_jsons(last_text)
            if extracted:
                last_json = extracted[-1]
                if isinstance(last_json, dict):
                    if "response" in last_json:
                        prediction = last_json["response"]
                        logger.info(f"Extracted prediction from 'response' field: {prediction}")
                        return str(prediction)
                    else:
                        # Try to find any numeric value in the JSON
                        for key, value in last_json.items():
                            if isinstance(value, (int, float)):
                                prediction = str(int(value))
                                logger.info(f"Using numeric value from key '{key}': {prediction}")
                                return prediction
                            elif isinstance(value, str):
                                # Try to extract number from string value
                                num_match = re.search(r'\d+', value)
                                if num_match:
                                    prediction = num_match.group()
                                    logger.info(f"Using extracted number from key '{key}': {prediction}")
                                    return prediction
                        
                        # No numeric value found, use string representation
                        prediction = str(last_json)
                        logger.info(f"No numeric key found, using full JSON string: {prediction}")
                        return prediction
                else:
                    prediction = str(last_json)
                    logger.info(f"JSON is not a dict, using string representation: {prediction}")
                    return prediction
            
            # No JSON found - try to extract from reasoning section
            logger.info("No JSON found, attempting to extract from reasoning section")
            
            # Look for patterns like "Total: X points" or "Score: X" in the reasoning
            reasoning_patterns = [
                r'[Tt]otal[:\s]+(\d+)\s*(?:points?)?',
                r'[Ss]core[:\s]+(\d+)\s*(?:points?)?',
                r'[Ff]inal\s+[Ss]core[:\s]+(\d+)',
                r'[Aa]ward[:\s]+(\d+)\s*(?:points?)?',
                r'[Gg]rade[:\s]+(\d+)',
            ]
            for pattern in reasoning_patterns:
                match = re.search(pattern, last_text)
                if match:
                    prediction = match.group(1)
                    logger.info(f"Extracted score from reasoning using pattern '{pattern}': {prediction}")
                    return prediction
            
            # Look for standalone numbers that might be the score
            # Be careful to avoid matching numbers that are part of the problem
            logger.info("Attempting to extract number from text")
            
            # Look for patterns like "Score: 7" or "The score is 7" or just "7"
            score_patterns = [
                r'[Rr]esult[:\s]+(\d+)',
                r'[Pp]oints?[:\s]+(\d+)',
                r'^(\d+)$',
                r'\b(\d+)\s*(?:points?|/\s*7)?\b',
                r'"(\d+)"',  # Quoted number
            ]
            for pattern in score_patterns:
                match = re.search(pattern, last_text)
                if match:
                    prediction = match.group(1)
                    logger.info(f"Extracted score using pattern '{pattern}': {prediction}")
                    return prediction
            
            # Last resort: just find any digit sequence (prefer single digits 0-7)
            # First try to find single digits 0-7 which are likely IMO scores
            score_match = re.search(r'\b([0-7])\b', last_text)
            if score_match:
                prediction = score_match.group(1)
                logger.info(f"Extracted single digit score: {prediction}")
                return prediction
            
            # If no single digit found, try any number
            number_match = re.search(r'\d+', last_text)
            if number_match:
                prediction = number_match.group()
                logger.info(f"Extracted raw number as last resort: {prediction}")
                return prediction
            
            logger.warning(f"Could not extract any number from text: {last_text[:200]}")
            return "0"
                
        except Exception as e:
            logger.error(f"Error extracting prediction: {e}")
            return "0"

    def forward(self, inputs: dict, max_retries: int = 2) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer
            max_retries: Number of retries on LLM failure (default: 2)

        Returns:
            (prediction, msg_history)
        """
        instruction = self._build_prompt(inputs)
        
        # Try LLM call with retries
        msg_history = []
        for attempt in range(max_retries + 1):
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=[],
                )
                break
            except Exception as e:
                logger.error(f"LLM call failed (attempt {attempt + 1}/{max_retries + 1}): {e}")
                if attempt == max_retries:
                    return "0", [{"role": "error", "text": f"LLM call failed after {max_retries + 1} attempts: {str(e)}"}]
                time.sleep(2 ** attempt)  # Exponential backoff

        # Extract prediction
        prediction = self._extract_prediction(msg_history)
        
        # Validate and normalize the score
        prediction = self._validate_score(prediction)
        logger.info(f"Final validated prediction: {prediction}")

        return str(prediction), msg_history
