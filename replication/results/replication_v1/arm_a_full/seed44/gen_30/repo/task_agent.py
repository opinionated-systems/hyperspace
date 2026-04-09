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
    Also handles markdown code blocks and bare JSON objects.
    """
    results = []
    search_from = 0
    
    # First, try to find <json>...</json> blocks
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
            # Try to fix common JSON issues
            try:
                # Remove trailing commas before closing braces/brackets
                fixed = re.sub(r',(\s*[}\]])', r'\1', inner)
                results.append(json.loads(fixed))
            except json.JSONDecodeError:
                continue
    
    # If no <json> blocks found, try markdown code blocks
    if not results:
        # Match ```json ... ``` or ``` ... ``` blocks (non-greedy, multiline)
        md_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
        for match in re.finditer(md_pattern, text, re.DOTALL):
            try:
                inner = match.group(1).strip()
                if inner:
                    results.append(json.loads(inner))
            except json.JSONDecodeError:
                # Try to fix common JSON issues
                try:
                    fixed = re.sub(r',(\s*[}\]])', r'\1', inner)
                    results.append(json.loads(fixed))
                except json.JSONDecodeError:
                    continue
        
        # Also try a simpler pattern for inline code blocks
        if not results:
            simple_md_pattern = r'```\s*\n?([^`]+)\n?```'
            for match in re.finditer(simple_md_pattern, text, re.DOTALL):
                try:
                    inner = match.group(1).strip()
                    if inner and inner.startswith('{'):
                        results.append(json.loads(inner))
                except json.JSONDecodeError:
                    continue
    
    # If still no results, try to find bare JSON objects
    if not results:
        # Look for JSON-like structures with "response" or "reasoning" keys
        json_pattern = r'\{\s*"[^"]+"\s*:[^}]+\}'
        for match in re.finditer(json_pattern, text, re.DOTALL):
            try:
                results.append(json.loads(match.group()))
            except json.JSONDecodeError:
                continue
    
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
Solution: Observe that 2 ≡ -1 (mod 3), so 2^100 ≡ (-1)^100 ≡ 1 (mod 3). The remainder is 1.
Grading Guidelines: Award 7 points for correct answer with valid reasoning. Award 5 points for correct answer with minor computational error. Award 3 points for correct approach but wrong answer.
Student Answer: 2^100 = (2^2)^50 = 4^50. Since 4 ≡ 1 (mod 3), we have 4^50 ≡ 1^50 ≡ 1 (mod 3). So the remainder is 1.
Score: 7
Reasoning: The student used a different but equally valid approach, correctly applying modular arithmetic to arrive at the right answer.

Example 5 (Zero Credit - Completely Wrong):
Problem: Prove that √2 is irrational.
Solution: Assume √2 = p/q in lowest terms. Then 2q^2 = p^2, so p^2 is even, thus p is even. Write p = 2k, then 2q^2 = 4k^2, so q^2 = 2k^2, making q even. Contradiction: p and q both even.
Grading Guidelines: Award 7 points for complete proof. Award 4 points for correct setup but missing contradiction. Award 2 points for stating the assumption correctly.
Student Answer: √2 is irrational because it cannot be written as a fraction. This is because 2 is prime and its square root is not an integer.
Score: 1
Reasoning: The student stated the correct conclusion but provided no valid mathematical reasoning. The explanation about primes is irrelevant to the proof.

Example 6 (Partial Credit - Good Progress):
Problem: Solve the equation x^2 - 5x + 6 = 0.
Solution: Factor: (x-2)(x-3) = 0, so x = 2 or x = 3.
Grading Guidelines: Award 7 points for correct answer with justification. Award 4 points for correct answer without showing work. Award 2 points for correct method but arithmetic error.
Student Answer: Using the quadratic formula: x = (5 ± √(25-24))/2 = (5 ± 1)/2. So x = 3 or x = 2.
Score: 7
Reasoning: The student correctly applied the quadratic formula and arrived at the correct solutions with clear work shown.
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

## Your Grading Responsibilities:
1. Carefully read the problem, official solution, and grading guidelines
2. Evaluate the student's answer against the official solution
3. Identify what parts are correct, partially correct, or incorrect
4. Consider the grading guidelines for partial credit
5. Assign an appropriate score (typically 0-7 points for IMO problems)
6. Provide brief reasoning explaining your grading decision

## Grading Principles:
- Award full credit (7 points) only when the solution is complete and correct
- Award partial credit based on the grading guidelines - look for specific point allocations
- Consider alternative valid approaches - different methods can be equally correct
- Distinguish between minor errors (computation) and major errors (conceptual)
- An incomplete but correct partial solution deserves partial credit
- A wrong answer with no valid reasoning deserves minimal or no credit

{_FEW_SHOT_EXAMPLES}

## Now Evaluate This Submission:

**DOMAIN:** {domain}

**PROBLEM:**
{problem}

**OFFICIAL SOLUTION:**
{solution}

**GRADING GUIDELINES:**
{grading_guidelines}

**STUDENT ANSWER TO EVALUATE:**
{student_answer}

## Step-by-Step Evaluation Process:
1. First, identify the key components required for a complete solution
2. Check which components the student has addressed
3. Verify the correctness of each component
4. Apply the grading guidelines to determine partial credit
5. Sum up the points to get the final score

## Response Format:
You MUST respond in JSON format with the following schema:
<json>
{{
    "response": "<numerical_score>",
    "reasoning": "<brief explanation of grading decision>"
}}
</json>

IMPORTANT:
- The "response" field must contain ONLY the numerical score (e.g., "7", "5", "0")
- The score must be an integer between 0 and 7 (inclusive)
- The "reasoning" field should briefly explain why you assigned that score
- Be specific about what the student got right and what was missing/wrong"""

        return prompt

    def _validate_score(self, prediction: str) -> str:
        """Validate and normalize the score prediction.
        
        Handles various formats including:
        - Plain numbers: "7", "5"
        - Score prefixes: "Score: 7", "Points: 5"
        - Fractions: "7/7", "5/7"
        - Descriptive: "7 points", "full credit (7)"
        - Floats: "6.5" -> "7" (round), "6.0" -> "6"
        """
        if prediction is None:
            return "0"
        
        # Convert to string and strip whitespace
        prediction = str(prediction).strip()
        
        # Handle empty string
        if not prediction:
            return "0"
        
        # Try to extract a number from the prediction
        # First, look for patterns like "X/7" or "Score: X"
        fraction_match = re.search(r'(\d+(?:\.\d+)?)\s*/\s*\d+', prediction)
        if fraction_match:
            try:
                score_float = float(fraction_match.group(1))
                score_int = int(round(score_float))
                return str(max(0, min(7, score_int)))
            except ValueError:
                pass
        
        # Look for "Score: X" or "Points: X" patterns
        prefix_match = re.search(r'(?:score|points?)\s*[:=]?\s*(\d+(?:\.\d+)?)', prediction, re.IGNORECASE)
        if prefix_match:
            try:
                score_float = float(prefix_match.group(1))
                score_int = int(round(score_float))
                return str(max(0, min(7, score_int)))
            except ValueError:
                pass
        
        # Try to find a standalone single digit 0-7 (word boundary)
        number_match = re.search(r'\b([0-7])\b', prediction)
        if number_match:
            return number_match.group(1)
        
        # Try to find any number and validate/clamp it
        # Use a pattern that requires word boundaries to avoid matching parts of larger numbers
        number_match = re.search(r'\b(\d+(?:\.\d+)?)\b', prediction)
        if number_match:
            try:
                score_float = float(number_match.group(1))
                score_int = int(round(score_float))
                # Clamp to valid IMO score range (0-7)
                return str(max(0, min(7, score_int)))
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
        extraction_method = "none"
        
        try:
            # Get the last assistant message
            last_message = None
            for msg in reversed(msg_history):
                if msg.get("role") == "assistant":
                    last_message = msg
                    break
            
            if last_message is None:
                raise ValueError("No assistant message found in history")
            
            last_text = last_message.get("text", "")
            
            # Try to extract JSON
            extracted = _extract_jsons(last_text)
            if extracted:
                last_json = extracted[-1]
                extraction_method = "json"
                
                # Try to get response field
                if "response" in last_json:
                    prediction = last_json["response"]
                    extraction_method = "json_response_field"
                elif "score" in last_json:
                    prediction = last_json["score"]
                    extraction_method = "json_score_field"
                elif isinstance(last_json, dict):
                    # Try to find any numeric-looking value
                    for key, value in last_json.items():
                        if isinstance(value, (int, float)):
                            prediction = str(value)
                            extraction_method = f"json_{key}_field"
                            break
                        elif isinstance(value, str):
                            # Try to extract number from string
                            num_match = re.search(r'\d+', value)
                            if num_match:
                                prediction = num_match.group()
                                extraction_method = f"json_{key}_extracted"
                                break
                
                # Extract reasoning if available
                if "reasoning" in last_json:
                    reasoning = last_json["reasoning"]
            else:
                # No JSON found - try to extract score directly from text
                extraction_method = "fallback_text"
                
                # Look for patterns like "Score: X" or "X points"
                score_patterns = [
                    r'(?:score|points?)\s*[:=]?\s*(\d+(?:\.\d+)?)',
                    r'(?:grade|mark)\s*[:=]?\s*(\d+(?:\.\d+)?)',
                    r'response["\']?\s*[:=]\s*["\']?(\d+(?:\.\d+)?)',
                ]
                for pattern in score_patterns:
                    match = re.search(pattern, last_text, re.IGNORECASE)
                    if match:
                        prediction = match.group(1)
                        extraction_method = f"pattern_{pattern[:20]}"
                        break
                else:
                    # Try to find a standalone single digit 0-7
                    number_match = re.search(r'\b([0-7])\b', last_text)
                    if number_match:
                        prediction = number_match.group(1)
                        extraction_method = "standalone_digit"
                    else:
                        # Fallback to any digit
                        number_match = re.search(r'\d+', last_text)
                        if number_match:
                            prediction = number_match.group()
                            extraction_method = "any_digit"
                            
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            extraction_method = "error"

        # Validate and normalize the score
        original_prediction = prediction
        prediction = self._validate_score(prediction)
        
        # Log extraction details for debugging
        self.log_fn(f"Extraction method: {extraction_method}, raw: {original_prediction}, final: {prediction}")
        
        # Log reasoning if available
        if reasoning:
            self.log_fn(f"Grading reasoning: {reasoning[:200]}")

        return str(prediction), msg_history
