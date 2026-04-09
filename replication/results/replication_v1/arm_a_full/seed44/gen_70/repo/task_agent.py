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


# Few-shot examples for IMO grading
_FEW_SHOT_EXAMPLES = """
Example 1 (Full Credit - Complete Solution):
Problem: Find all positive integers n such that n^2 + 3n + 2 is divisible by 4.
Solution: We can factor n^2 + 3n + 2 = (n+1)(n+2). For this to be divisible by 4, either (n+1) or (n+2) must be even, and at least one must be divisible by 2. Since n+1 and n+2 are consecutive integers, one is even. For the product to be divisible by 4, we need the even number to be divisible by 4, or both to be even (impossible for consecutive integers). So we need n+1 ≡ 0 (mod 4) or n+2 ≡ 0 (mod 4), meaning n ≡ 3 (mod 4) or n ≡ 2 (mod 4).
Grading Guidelines: Award 7 points for correct answer (n ≡ 2 or 3 mod 4). Award partial credit: 2 points for correct factorization, 2 points for recognizing need for divisibility by 4, 3 points for correct modular analysis.
Student Answer: n^2 + 3n + 2 = (n+1)(n+2). Since these are consecutive, one is even. For divisibility by 4, we need n ≡ 2 or 3 (mod 4).
<json>
{
    "response": "7"
}
</json>

Example 2 (Full Credit - Complete Induction):
Problem: Prove that for any positive integer n, n^3 + 2n is divisible by 3.
Solution: We use induction. Base case: n=1, 1^3 + 2(1) = 3, divisible by 3. Inductive step: assume k^3 + 2k divisible by 3. Then (k+1)^3 + 2(k+1) = k^3 + 3k^2 + 3k + 1 + 2k + 2 = k^3 + 2k + 3(k^2 + k + 1). By induction hypothesis, k^3 + 2k divisible by 3, and 3(k^2 + k + 1) clearly divisible by 3. Thus divisible by 3.
Grading Guidelines: Award 7 points for complete proof. Deduct 2 points for missing base case, 3 points for errors in inductive step, 2 points for lack of clarity.
Student Answer: By induction. Base case n=1 works. Assume true for k, then (k+1)^3 + 2(k+1) = k^3 + 3k^2 + 3k + 1 + 2k + 2 = (k^3 + 2k) + 3(k^2 + k + 1), both terms divisible by 3.
<json>
{
    "response": "7"
}
</json>

Example 3 (Partial Credit - Minor Error):
Problem: Prove that the sum of the first n positive integers is n(n+1)/2.
Solution: Use mathematical induction. Base case: n=1, sum is 1 = 1(2)/2. Inductive step: assume true for k, so 1+2+...+k = k(k+1)/2. Then 1+2+...+k+(k+1) = k(k+1)/2 + (k+1) = (k+1)(k/2 + 1) = (k+1)(k+2)/2, which is the formula for n=k+1.
Grading Guidelines: Award 7 points for complete proof. Award 3 points for correct base case only. Award 4 points for correct inductive step setup but errors in algebra. Award 6 points for minor algebraic error in inductive step.
Student Answer: Base case: n=1, 1 = 1(2)/2 = 1. Inductive step: Assume 1+2+...+k = k(k+1)/2. Then adding (k+1): sum = k(k+1)/2 + (k+1) = (k+1)(k+2)/2.
<json>
{
    "response": "7"
}
</json>

Example 4 (Zero Credit - Completely Wrong):
Problem: Find the remainder when 2^100 is divided by 3.
Solution: Note that 2 ≡ -1 (mod 3), so 2^100 ≡ (-1)^100 ≡ 1 (mod 3). The remainder is 1.
Grading Guidelines: Award 7 points for correct answer with valid reasoning. Award 2 points for recognizing pattern in powers of 2 mod 3. Award 0 points for incorrect answer.
Student Answer: 2^100 is even, so remainder is 2 when divided by 3.
<json>
{
    "response": "0"
}
</json>

Example 5 (Partial Credit - Missing Base Case):
Problem: Prove by induction that 1 + 2 + ... + n = n(n+1)/2 for all positive integers n.
Solution: Base case: n=1, LHS = 1, RHS = 1(2)/2 = 1. Inductive step: Assume true for n=k, so 1+2+...+k = k(k+1)/2. Add (k+1) to both sides: 1+2+...+k+(k+1) = k(k+1)/2 + (k+1) = (k+1)(k+2)/2, which is the formula for n=k+1.
Grading Guidelines: Award 7 points for complete proof. Deduct 2 points for missing base case. Deduct 3 points for errors in inductive step. Deduct 1-2 points for minor errors.
Student Answer: Assume true for n=k, so 1+2+...+k = k(k+1)/2. Then 1+2+...+k+(k+1) = k(k+1)/2 + (k+1) = (k+1)(k+2)/2.
<json>
{
    "response": "5"
}
</json>

Example 6 (Partial Credit - Good Progress but Incomplete):
Problem: Find all prime numbers p such that p^2 + 2 is also prime.
Solution: Test small primes: p=2 gives 2^2+2=6 (not prime), p=3 gives 3^2+2=11 (prime), p=5 gives 5^2+2=27 (not prime), p=7 gives 7^2+2=51 (not prime). For p>3, p ≡ ±1 (mod 3), so p^2 ≡ 1 (mod 3), thus p^2+2 ≡ 0 (mod 3) and divisible by 3. Since p^2+2 > 3 for p>3, it's composite. So only p=3 works.
Grading Guidelines: Award 7 points for complete solution with proof. Award 4 points for correct answer with partial justification. Award 2 points for testing cases without general proof. Award 0 points for incorrect answer.
Student Answer: Testing: p=2 gives 6 (not prime), p=3 gives 11 (prime), p=5 gives 27 (not prime). So p=3 is the answer.
<json>
{
    "response": "4"
}
</json>

Example 7 (Partial Credit - Significant Progress):
Problem: Prove that for any positive integer n, the number n^3 + 5n is divisible by 6.
Solution: Factor: n^3 + 5n = n(n^2 + 5) = n(n^2 - 1 + 6) = n(n-1)(n+1) + 6n. The term n(n-1)(n+1) is product of three consecutive integers, so divisible by both 2 and 3, hence by 6. The term 6n is clearly divisible by 6. Thus the sum is divisible by 6.
Grading Guidelines: Award 7 points for complete proof. Award 5 points for correct factorization and recognizing divisibility by 2. Award 3 points for correct factorization only. Award 1 point for attempting factorization.
Student Answer: n^3 + 5n = n(n^2 + 5). For even n, n is divisible by 2. For odd n, n^2 is odd so n^2+5 is even. So always divisible by 2.
<json>
{
    "response": "5"
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

GRADING INSTRUCTIONS:
1. Carefully compare the student answer to the official solution
2. Identify what parts are correct, partially correct, or incorrect
3. Consider the grading guidelines for partial credit - be generous with partial credit when the student shows understanding
4. Assign an appropriate integer score between 0 and 7 (inclusive)
5. Respond ONLY with the JSON format below - no additional text before or after

PARTIAL CREDIT GUIDELINES:
- 7 points: Complete, correct solution with proper reasoning
- 6 points: Minor error or omission, but essentially correct approach
- 5 points: Good progress, significant correct work, but missing some key elements
- 4 points: Partial solution with some correct ideas but incomplete or has errors
- 3 points: Some correct initial steps or approach, but major gaps
- 2 points: Limited correct work, shows some understanding of the problem
- 1 point: Minimal correct insight or attempt
- 0 points: No correct work, completely wrong, or irrelevant

IMPORTANT: Your response must be in this exact format:
<json>
{{
    "response": "<numerical_score>"
}}
</json>

The "response" field must contain ONLY a single integer between 0 and 7, as a string (e.g., "7", "5", "0", "3"). Do not include any other text, explanation, or formatting outside the JSON block."""

        return prompt

    def _validate_score(self, prediction: str) -> str:
        """Validate and normalize the score prediction.
        
        Handles various formats like "Score: 7", "7 points", "7/7", "7.0", etc.
        Ensures the score is within valid IMO range (0-7).
        """
        if prediction is None:
            return "0"
        
        # Convert to string and strip whitespace
        prediction = str(prediction).strip()
        
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
        decimal_match = re.search(r'\d+\.\d+', prediction)
        if decimal_match:
            try:
                score_float = float(decimal_match.group())
                score_int = int(round(score_float))
                if 0 <= score_int <= 7:
                    return str(score_int)
            except ValueError:
                pass
        
        # Try integer match
        number_match = re.search(r'\d+', prediction)
        if number_match:
            score = number_match.group()
            try:
                score_int = int(score)
                if 0 <= score_int <= 7:
                    return str(score_int)
            except ValueError:
                pass
        
        # If we can't extract a valid number, return "0"
        logger.warning(f"Could not extract valid score from '{prediction}', defaulting to 0")
        return "0"

    def _extract_score_from_text(self, text: str) -> str | None:
        """Extract a score from text using various patterns.
        
        This is a helper method for extracting scores from non-JSON responses.
        Returns the extracted score or None if no valid score found.
        """
        if not text:
            return None
        
        # Look for patterns like "Score: 7" or "The score is 7" or just "7"
        score_patterns = [
            r'[Ss]core[\s:]+(\d+)',
            r'[Rr]esult[\s:]+(\d+)',
            r'[Gg]rade[\s:]+(\d+)',
            r'[Pp]oints?[\s:]+(\d+)',
            r'[Aa]ward[\s:]+(\d+)',
            r'[Gg]ive[\s:]+(\d+)',
            r'^(\d+)$',
            r'\b(\d+)\s*(?:points?|/\s*7)?\b',
            r'"(\d+)"',  # Quoted number
            r"'(\d+)'",  # Single-quoted number
        ]
        for pattern in score_patterns:
            match = re.search(pattern, text)
            if match:
                score = match.group(1)
                try:
                    score_int = int(score)
                    if 0 <= score_int <= 7:
                        return str(score_int)
                except ValueError:
                    continue
        
        # Last resort: just find any digit sequence
        number_match = re.search(r'\d+', text)
        if number_match:
            score = number_match.group()
            try:
                score_int = int(score)
                if 0 <= score_int <= 7:
                    return str(score_int)
            except ValueError:
                pass
        
        return None

    def _extract_prediction(self, msg_history: list[dict]) -> str:
        """Extract prediction from message history.
        
        Tries multiple strategies to extract a valid score:
        1. Extract from JSON response field
        2. Extract from any numeric field in JSON
        3. Extract from text using score-related patterns
        4. Extract any number as last resort
        
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
            
            # Try to extract JSON first
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
                                extracted_score = self._extract_score_from_text(value)
                                if extracted_score:
                                    logger.info(f"Using extracted number from key '{key}': {extracted_score}")
                                    return extracted_score
                        
                        # No numeric value found, use string representation
                        prediction = str(last_json)
                        logger.info(f"No numeric key found, using full JSON string: {prediction}")
                        return prediction
                else:
                    prediction = str(last_json)
                    logger.info(f"JSON is not a dict, using string representation: {prediction}")
                    return prediction
            else:
                # No JSON found, try to extract score from text
                logger.info("No JSON found, attempting to extract number from text")
                
                extracted_score = self._extract_score_from_text(last_text)
                if extracted_score:
                    logger.info(f"Extracted score from text: {extracted_score}")
                    return extracted_score
                
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

    def forward_with_consensus(self, inputs: dict, num_samples: int = 3) -> tuple[str, list[dict]]:
        """Run the task agent multiple times and use consensus for more reliable grading.
        
        This method can be used when higher confidence in the grading is needed.
        It takes multiple samples and uses the median score to reduce variance.
        
        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer
            num_samples: Number of grading samples to take (default: 3)
            
        Returns:
            (prediction, msg_history) - prediction is the consensus score
        """
        scores = []
        all_histories = []
        
        for i in range(num_samples):
            prediction, msg_history = self.forward(inputs, max_retries=1)
            try:
                score = int(prediction)
                if 0 <= score <= 7:
                    scores.append(score)
                    all_histories.append(msg_history)
            except ValueError:
                logger.warning(f"Invalid score in sample {i}: {prediction}")
                continue
        
        if not scores:
            return "0", [{"role": "error", "text": "All consensus samples failed"}]
        
        # Use median for consensus (more robust to outliers than mean)
        scores.sort()
        consensus = scores[len(scores) // 2]
        
        logger.info(f"Consensus scores: {scores}, selected: {consensus}")
        
        # Return the first history for debugging purposes
        return str(consensus), all_histories[0] if all_histories else []
