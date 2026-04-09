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
    Also handles markdown code blocks with json tag.
    """
    results = []
    search_from = 0
    
    # Handle both <json>...</json> and ```json...``` formats
    while True:
        # Try <json> tags first
        start = text.find("<json>", search_from)
        end_marker = "</json>"
        
        # If no <json> found, try markdown code blocks
        if start == -1:
            start = text.find("```json", search_from)
            if start != -1:
                start = start + 7  # Skip past ```json
                end_marker = "```"
        
        if start == -1:
            break
            
        end = text.find(end_marker, start)
        if end == -1:
            break
            
        inner = text[start:end].strip()
        search_from = end + len(end_marker)
        
        # Clean up common formatting issues
        inner = inner.strip()
        if inner.startswith("\n"):
            inner = inner[1:]
        if inner.endswith("\n"):
            inner = inner[:-1]
            
        try:
            results.append(json.loads(inner))
        except json.JSONDecodeError:
            continue
            
    return results or None


# Few-shot examples for IMO grading - Carefully calibrated to match actual grading standards
_FEW_SHOT_EXAMPLES = """
Example 1 (Full Credit - Complete Solution, Score 7):
Problem: Find all positive integers n such that n^2 + 3n + 2 is divisible by 4.
Solution: We can factor n^2 + 3n + 2 = (n+1)(n+2). For this to be divisible by 4, either (n+1) or (n+2) must be even, and at least one must be divisible by 2. Since n+1 and n+2 are consecutive integers, one is even. For the product to be divisible by 4, we need the even number to be divisible by 4, or both to be even (impossible for consecutive integers). So we need n+1 ≡ 0 (mod 4) or n+2 ≡ 0 (mod 4), meaning n ≡ 3 (mod 4) or n ≡ 2 (mod 4).
Grading Guidelines: Award 7 points for correct answer (n ≡ 2 or 3 mod 4). Award partial credit: 2 points for correct factorization, 2 points for recognizing need for divisibility by 4, 3 points for correct modular analysis.
Student Answer: n^2 + 3n + 2 = (n+1)(n+2). Since these are consecutive, one is even. For divisibility by 4, we need n ≡ 2 or 3 (mod 4).
<json>
{
    "response": "7"
}
</json>

Example 2 (Zero Credit - Wrong Answer, Score 0):
Problem: Prove that for any positive integer n, n^3 + 2n is divisible by 3.
Solution: We use induction. Base case: n=1, 1^3 + 2(1) = 3, divisible by 3. Inductive step: assume k^3 + 2k divisible by 3. Then (k+1)^3 + 2(k+1) = k^3 + 3k^2 + 3k + 1 + 2k + 2 = k^3 + 2k + 3(k^2 + k + 1). By induction hypothesis, k^3 + 2k divisible by 3, and 3(k^2 + k + 1) clearly divisible by 3. Thus divisible by 3.
Grading Guidelines: Award 7 points for complete proof. Deduct 2 points for missing base case, 3 points for errors in inductive step, 2 points for lack of clarity.
Student Answer: n^3 + 2n = n(n^2 + 2). For n=2, this is 2(6) = 12, divisible by 3. So it works for all n.
<json>
{
    "response": "0"
}
</json>

Example 3 (Partial Credit - Minor Gap, Score 6):
Problem: Prove that the sum of the first n positive integers is n(n+1)/2.
Solution: Use mathematical induction. Base case: n=1, sum is 1 = 1(2)/2. Inductive step: assume true for k, so 1+2+...+k = k(k+1)/2. Then 1+2+...+k+(k+1) = k(k+1)/2 + (k+1) = (k(k+1) + 2(k+1))/2 = (k+1)(k+2)/2.
Grading Guidelines: Award 7 points for complete proof. Award 6 points for correct approach with minor presentation issue. Award 5 points for correct approach but missing explicit inductive hypothesis statement.
Student Answer: By induction. Base case: 1 = 1(2)/2. Assume for k, then sum to k+1 is k(k+1)/2 + (k+1) = (k+1)(k+2)/2. Therefore the formula holds.
<json>
{
    "response": "6"
}
</json>

Example 4 (Zero Credit - No Meaningful Progress, Score 0):
Problem: Find all primes p such that p^2 + 2 is also prime.
Solution: Check small primes: p=2 gives 4+2=6 (not prime), p=3 gives 9+2=11 (prime), p=5 gives 25+2=27 (not prime), p=7 gives 49+2=51 (not prime). For p>3, p ≡ ±1 (mod 6), so p^2 ≡ 1 (mod 3), thus p^2+2 ≡ 0 (mod 3) and divisible by 3. Only p=3 works.
Grading Guidelines: Award 7 points for complete solution. Award 4 points for checking cases without modular argument. Award 1 point for identifying p=3 as answer without proof. Award 0 points for no meaningful progress.
Student Answer: All primes work because primes are special numbers.
<json>
{
    "response": "0"
}
</json>

Example 5 (Partial Credit - Good Progress but Incomplete, Score 4):
Problem: Prove that among all triangles with given perimeter, the equilateral triangle has maximum area.
Solution: Use Heron's formula: area = √[s(s-a)(s-b)(s-c)] where s = (a+b+c)/2. By AM-GM, (s-a)(s-b)(s-c) ≤ ((3s-(a+b+c))/3)^3 = (s/3)^3. Equality when s-a=s-b=s-c, i.e., a=b=c. For equilateral triangle with side s, area = (√3/4)s^2 and semiperimeter = 3s/2. Substituting gives area = (a+b+c)^2/(12√3).
Grading Guidelines: Award 7 points for complete proof. Award 4 points for correct approach (Heron's + AM-GM) but incomplete execution. Award 2 points for stating equilateral maximizes area without proof.
Student Answer: The equilateral triangle has the maximum area for a given perimeter. If all sides equal s, then a+b+c = 3s, and area = (√3/4)s^2 = (√3/4)((a+b+c)/3)^2 = (a+b+c)^2/(12√3).
<json>
{
    "response": "4"
}
</json>

Example 6 (Partial Credit - Minimal Progress, Score 1):
Problem: Find all functions f: R → R such that f(x+y) = f(x) + f(y) for all x, y ∈ R.
Solution: The solutions are f(x) = cx for some constant c. First show f(0) = 0. Then prove f(nx) = nf(x) for integers n, extend to rationals, and use continuity (if assumed) or other conditions for reals.
Grading Guidelines: Award 7 points for complete solution. Award 2 points for finding f(0)=0. Award 4 points for proving f(nx)=nf(x) for integers. Award 1 point for guessing linear form without proof. Award 0 points for no progress.
Student Answer: f(0) = f(0+0) = f(0) + f(0), so f(0) = 0. Also f(x) = cx seems to work.
<json>
{
    "response": "1"
}
</json>

Example 7 (Score 3 - Some correct elements but significant gaps):
Problem: Prove that for any triangle with sides a, b, c: a^2 + b^2 + c^2 ≥ ab + bc + ca.
Solution: Rearrange to (a-b)^2 + (b-c)^2 + (c-a)^2 ≥ 0, which is always true since squares are non-negative. Equality holds when a=b=c.
Grading Guidelines: Award 7 points for complete proof. Award 3 points for stating the inequality or attempting algebraic manipulation. Award 5 points for correct approach but algebraic error.
Student Answer: We know that (a-b)^2 ≥ 0, so a^2 + b^2 ≥ 2ab. Similarly for other pairs.
<json>
{
    "response": "3"
}
</json>

Example 8 (Score 5 - Significant progress but missing key element):
Problem: Prove that the product of n consecutive positive integers is divisible by n!.
Solution: The product k(k+1)...(k+n-1) = (k+n-1)!/(k-1)!. This equals n! × C(k+n-1, n) where C is binomial coefficient. Since binomial coefficients are integers, the product is divisible by n!.
Grading Guidelines: Award 7 points for complete proof. Award 5 points for recognizing combinatorial interpretation but not completing the argument. Award 3 points for checking small cases.
Student Answer: The product k(k+1)...(k+n-1) can be written as (k+n-1)!/(k-1)!. This is related to binomial coefficients.
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
        """Build a structured prompt for the grading task with enhanced reasoning instructions."""
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        # Truncate very long inputs to prevent token overflow
        max_len = 8000
        problem = problem[:max_len] if len(problem) > max_len else problem
        solution = solution[:max_len] if len(solution) > max_len else solution
        grading_guidelines = grading_guidelines[:max_len] if len(grading_guidelines) > max_len else grading_guidelines
        student_answer = student_answer[:max_len] if len(student_answer) > max_len else student_answer

        prompt = f"""You are an expert mathematics grader for International Mathematical Olympiad (IMO) problems. Your task is to evaluate student answers based on the official solution and grading guidelines.

CRITICAL GRADING PRINCIPLES:
1. Be STRICT and OBJECTIVE - do not give benefit of the doubt
2. The student answer must MATCH the official solution's key insights
3. Partial credit is only awarded for SPECIFIC correct elements mentioned in grading guidelines
4. Wrong answers or answers with fundamental errors get 0 points
5. Vague or incomplete reasoning does NOT earn full credit

Your responsibilities:
1. Carefully read the problem, official solution, and grading guidelines
2. Compare the student's answer POINT BY POINT with the official solution
3. Identify what specific elements the student got RIGHT vs WRONG
4. Assign a score based STRICTLY on the grading guidelines
5. Provide your evaluation in the specified JSON format

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

Evaluation Process (think step by step):
1. **Understand the Problem**: What is being asked? What are the key concepts?
2. **Analyze the Official Solution**: What is the complete correct approach? What are the KEY insights?
3. **Review Grading Guidelines**: What partial credit is available and for what SPECIFIC elements?
4. **Evaluate Student Answer STRICTLY**: 
   - Does the student have the CORRECT final answer?
   - Does the reasoning match the official solution's key steps?
   - Are there errors, gaps, or incorrect statements?
   - Is the logic sound and complete?
5. **Assign Score**: Based STRICTLY on the guidelines - be conservative with partial credit

Scoring Reference:
- 7: Complete, correct solution with clear reasoning matching official solution
- 6: Minor flaw or gap in an otherwise correct solution
- 5: Significant progress but missing one key element
- 3-4: Partial solution with some correct elements but major gaps
- 1-2: Minimal progress or single relevant insight only
- 0: No meaningful progress, wrong answer, or fundamentally flawed reasoning

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

    def _normalize_score(self, raw_score: str) -> str:
        """Normalize a raw score to the valid IMO range (0-7).
        
        This handles cases where the model might output scores outside
        the valid range due to confusion or errors.
        """
        try:
            score_int = int(raw_score)
            # Clamp to valid range
            if score_int < 0:
                logger.warning(f"Score {score_int} below minimum, clamping to 0")
                return "0"
            elif score_int > 7:
                logger.warning(f"Score {score_int} above maximum, clamping to 7")
                return "7"
            return str(score_int)
        except (ValueError, TypeError):
            logger.warning(f"Invalid score '{raw_score}', defaulting to 0")
            return "0"

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
            else:
                # No JSON found, try to extract any number from the response
                logger.info("No JSON found, attempting to extract number from text")
                
                # Look for patterns like "Score: 7" or "The score is 7" or just "7"
                score_patterns = [
                    r'[Ss]core[:\s]+(\d+)',
                    r'[Rr]esult[:\s]+(\d+)',
                    r'[Gg]rade[:\s]+(\d+)',
                    r'[Pp]oints?[:\s]+(\d+)',
                    r'[Aa]ward[:\s]+(\d+)',
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
                
                # Last resort: just find any digit sequence
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
        
        # Additional normalization to ensure valid IMO range
        prediction = self._normalize_score(prediction)
        logger.info(f"Final validated prediction: {prediction}")

        return str(prediction), msg_history
