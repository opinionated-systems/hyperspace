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
    Also handles markdown code blocks and inline JSON as fallbacks.
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
            # Try to extract just the response field if full JSON fails
            try:
                response_match = re.search(r'"response"\s*:\s*"([^"]*)"', inner)
                if response_match:
                    results.append({"response": response_match.group(1)})
                else:
                    # Try to find any numeric value as response
                    num_match = re.search(r'"response"\s*:\s*(\d+(?:\.\d+)?)', inner)
                    if num_match:
                        results.append({"response": num_match.group(1)})
            except Exception:
                continue
    
    # If no <json> blocks found, try markdown code blocks
    if not results:
        # Look for ```json ... ``` blocks
        md_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
        for match in re.finditer(md_pattern, text, re.DOTALL):
            inner = match.group(1).strip()
            try:
                results.append(json.loads(inner))
            except json.JSONDecodeError:
                # Try to find a JSON object within the text
                try:
                    json_start = inner.find('{')
                    json_end = inner.rfind('}')
                    if json_start != -1 and json_end != -1 and json_end > json_start:
                        results.append(json.loads(inner[json_start:json_end+1]))
                except Exception:
                    continue
    
    # Last resort: try to find any JSON-like object in the text
    if not results:
        try:
            # Look for patterns like {"response": "7"} or {"reasoning": "...", "response": "5"}
            json_pattern = r'\{\s*"[^"]+"\s*:\s*"[^"]*"(?:\s*,\s*"[^"]+"\s*:\s*"[^"]*")*\s*\}'
            for match in re.finditer(json_pattern, text, re.DOTALL):
                try:
                    results.append(json.loads(match.group()))
                except json.JSONDecodeError:
                    continue
        except Exception:
            pass
    
    return results or None


# Few-shot examples for IMO grading with detailed reasoning
_FEW_SHOT_EXAMPLES = """
Example 1:
Problem: Find all positive integers n such that n^2 + 3n + 2 is divisible by 4.
Solution: We can factor n^2 + 3n + 2 = (n+1)(n+2). For this to be divisible by 4, either (n+1) or (n+2) must be even, and at least one must be divisible by 2. Since n+1 and n+2 are consecutive integers, one is even. For the product to be divisible by 4, we need the even number to be divisible by 4, or both to be even (impossible for consecutive integers). So we need n+1 ≡ 0 (mod 4) or n+2 ≡ 0 (mod 4), meaning n ≡ 3 (mod 4) or n ≡ 2 (mod 4).
Grading Guidelines: Award 7 points for correct answer (n ≡ 2 or 3 mod 4). Award partial credit: 2 points for correct factorization, 2 points for recognizing need for divisibility by 4, 3 points for correct modular analysis.
Student Answer: n^2 + 3n + 2 = (n+1)(n+2). Since these are consecutive, one is even. For divisibility by 4, we need n ≡ 2 or 3 (mod 4).
Reasoning: The student correctly factored the expression and identified the key insight about consecutive integers. They provided the correct final answer with proper modular arithmetic notation. All critical steps from the official solution are present.
Score: 7

Example 2:
Problem: Prove that for any positive integer n, n^3 + 2n is divisible by 3.
Solution: We use induction. Base case: n=1, 1^3 + 2(1) = 3, divisible by 3. Inductive step: assume k^3 + 2k divisible by 3. Then (k+1)^3 + 2(k+1) = k^3 + 3k^2 + 3k + 1 + 2k + 2 = k^3 + 2k + 3(k^2 + k + 1). By induction hypothesis, k^3 + 2k divisible by 3, and 3(k^2 + k + 1) clearly divisible by 3. Thus divisible by 3.
Grading Guidelines: Award 7 points for complete proof. Deduct 2 points for missing base case, 3 points for errors in inductive step, 2 points for lack of clarity.
Student Answer: By induction. Base case n=1 works. Assume true for k, then (k+1)^3 + 2(k+1) = k^3 + 3k^2 + 3k + 1 + 2k + 2 = (k^3 + 2k) + 3(k^2 + k + 1), both terms divisible by 3.
Reasoning: The student provided a complete induction proof with correct base case, proper inductive hypothesis, and correct algebraic manipulation. The final conclusion clearly shows both terms are divisible by 3. All required elements are present and correct.
Score: 7

Example 3 (Partial Credit):
Problem: Prove that the sum of the first n odd numbers is n^2.
Solution: The k-th odd number is 2k-1. Sum = Σ(k=1 to n) (2k-1) = 2Σk - Σ1 = 2(n(n+1)/2) - n = n(n+1) - n = n^2 + n - n = n^2.
Grading Guidelines: 7 points for complete proof. 3 points for identifying the k-th odd number formula. 2 points for setting up the summation. 2 points for correct algebraic simplification.
Student Answer: The odd numbers are 1, 3, 5, ... The sum is n^2 because if you arrange them in a square pattern, you get n rows of n.
Reasoning: The student has the correct intuition about the result (n^2) and provides a geometric visualization, but lacks the formal algebraic proof required. They did not identify the k-th odd number formula (2k-1) or use summation notation. This is a valid observation but not a rigorous proof.
Score: 2

Example 4 (Zero Credit - Incorrect Reasoning):
Problem: Prove that √2 is irrational.
Solution: Assume √2 = p/q in lowest terms with gcd(p,q)=1. Then 2q^2 = p^2, so p^2 is even, thus p is even. Let p = 2k. Then 2q^2 = 4k^2, so q^2 = 2k^2, making q even. Contradiction: p and q both even, not in lowest terms.
Grading Guidelines: 7 points for complete proof. Award 2 points for correct assumption setup, 2 points for showing p is even, 2 points for showing q is even, 1 point for concluding contradiction.
Student Answer: √2 is approximately 1.414, which is not a whole number, so it must be irrational.
Reasoning: The student's reasoning is fundamentally flawed. Being non-integer does not imply irrationality (e.g., 3/2 = 1.5 is rational). The student completely missed the proof by contradiction structure and failed to address the parity argument at the heart of the standard proof. No credit is warranted.
Score: 0
"""


class TaskAgent:
    """Task agent that solves IMO grading problems."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def _build_prompt(self, inputs: dict) -> str:
        """Build a structured prompt for the grading task with step-by-step reasoning."""
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        prompt = f"""You are an expert mathematics grader for International Mathematical Olympiad (IMO) problems. Your task is to evaluate student answers based on the official solution and grading guidelines.

Your responsibilities:
1. Carefully read the problem, official solution, and grading guidelines
2. Analyze the student's answer step-by-step
3. Compare against the official solution to identify correct, partially correct, and incorrect parts
4. Apply the grading guidelines rigorously for partial credit
5. Assign a final score (typically 0-7 points for IMO problems)

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

Evaluation Instructions:
1. First, identify the key claims and proof steps in the official solution
2. Check which of these appear in the student's answer
3. Note any errors, gaps, or incorrect reasoning
4. Consider partial credit according to the guidelines
5. Be fair but rigorous - IMO grading rewards complete, correct proofs

IMPORTANT: You MUST respond using the exact JSON format below. Do not add any text outside the JSON tags.

<json>
{{
    "reasoning": "<your detailed step-by-step evaluation reasoning>",
    "response": "<numerical_score>"
}}
</json>

Requirements:
- The "reasoning" field must contain your detailed analysis of the student's answer
- The "response" field must contain ONLY a single number (0-7) representing the score
- Do not include any text before or after the JSON block
- Do not use markdown formatting inside the JSON values"""

        return prompt

    def _validate_score(self, prediction: str | int | float | None) -> str:
        """Validate and normalize the score prediction to IMO grading range [0, 7].
        
        This method handles various input formats commonly produced by LLMs when
        grading IMO problems. It extracts numeric scores and clamps them to the
        valid range, handling edge cases gracefully.
        
        Supported formats:
        - Plain numbers: "7", "5", 7, 5.5
        - Score prefixes: "Score: 7", "score: 5", "Final score: 6"
        - Fractions: "7/7", "5/7" (extracts numerator)
        - With units: "7 points", "5 pts", "6.5 points"
        - Decimal scores: "6.5", "3.5" (rounded to nearest integer)
        - Text descriptions: "full marks" → 7, "no credit" → 0
        
        Edge cases handled:
        - None or empty input → "0"
        - Negative scores → clamped to 0
        - Scores > 7 → clamped to 7
        - Multi-digit numbers (likely years) → rejected
        - Invalid/malformed input → "0"
        
        Args:
            prediction: Raw score prediction from LLM (string, number, or None)
            
        Returns:
            Normalized score as string in range ["0", "7"]
        """
        # Handle None input
        if prediction is None:
            return "0"
        
        # Convert to string and normalize
        try:
            prediction = str(prediction).strip()
        except (TypeError, ValueError):
            return "0"
        
        if not prediction:
            return "0"
        
        # Handle special text cases first (before numeric extraction)
        lower_pred = prediction.lower()
        full_marks_indicators = ["full marks", "complete", "perfect", "all points", "maximum", "full credit"]
        zero_indicators = ["no credit", "zero", "none", "incorrect", "wrong", "invalid", "no points"]
        
        if any(phrase in lower_pred for phrase in full_marks_indicators):
            return "7"
        if any(phrase in lower_pred for phrase in zero_indicators):
            return "0"
        
        # Try to extract a number from the prediction using multiple strategies
        
        # Strategy 1: Fraction pattern like "5/7" - extract numerator
        fraction_match = re.search(r'(\d+(?:\.\d+)?)\s*/\s*\d+', prediction)
        if fraction_match:
            try:
                score_float = float(fraction_match.group(1))
                score_int = int(round(score_float))
                return str(max(0, min(7, score_int)))
            except (ValueError, TypeError):
                pass
        
        # Strategy 2: Score prefix patterns like "Score: 6.5", "score of 7"
        score_prefix_match = re.search(r'score[:\s]+(\d+(?:\.\d+)?)', prediction, re.IGNORECASE)
        if score_prefix_match:
            try:
                score_float = float(score_prefix_match.group(1))
                score_int = int(round(score_float))
                return str(max(0, min(7, score_int)))
            except (ValueError, TypeError):
                pass
        
        # Strategy 3: Number followed by "points" or "pts"
        points_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:points?|pts?)', prediction, re.IGNORECASE)
        if points_match:
            try:
                score_float = float(points_match.group(1))
                score_int = int(round(score_float))
                return str(max(0, min(7, score_int)))
            except (ValueError, TypeError):
                pass
        
        # Strategy 4: Standalone number at start/end like "The score is 5" or "5 out of 7"
        standalone_match = re.search(r'(?:^|\s)(\d+(?:\.\d+)?)(?:\s*$|\s+out\s+of)', prediction)
        if standalone_match:
            try:
                score_float = float(standalone_match.group(1))
                score_int = int(round(score_float))
                return str(max(0, min(7, score_int)))
            except (ValueError, TypeError):
                pass
        
        # Strategy 5: Fallback - any digit sequence (with validation)
        digit_match = re.search(r'\d+', prediction)
        if digit_match:
            try:
                matched_num = digit_match.group()
                # Reject likely years (4 digits) or very large numbers
                if len(matched_num) <= 2:
                    score_int = int(matched_num)
                    return str(max(0, min(7, score_int)))
            except (ValueError, TypeError):
                pass
        
        # Default: return "0" if no valid score could be extracted
        return "0"

    def _extract_score(self, text: str) -> tuple[str, str]:
        """Extract score from text using multiple strategies.
        
        Returns:
            (prediction, extraction_method)
        """
        # Strategy 1: Try JSON extraction first
        extracted = _extract_jsons(text)
        if extracted:
            last_extracted = extracted[-1]
            
            # Extract reasoning if present
            if "reasoning" in last_extracted:
                reasoning = str(last_extracted["reasoning"])
                self.log_fn(f"Grading reasoning: {reasoning[:200]}...")
            
            # Extract response/score
            if "response" in last_extracted:
                return str(last_extracted["response"]), "json"
            
            # If no "response" key, try to find any numeric value in the dict
            for key, value in last_extracted.items():
                if isinstance(value, (int, float)) or (isinstance(value, str) and value.isdigit()):
                    return str(value), f"json_key:{key}"
            
            # Last resort: use the whole JSON as string
            return str(last_extracted), "json_fallback"
        
        # Strategy 2: Look for score patterns in text
        score_patterns = [
            (r'[Ff]inal\s+[Ss]core[:\s]+(\d+(?:\.\d+)?)', "final_score"),
            (r'[Ss]core[:\s]+(\d+(?:\.\d+)?)', "score"),
            (r'[Ss]core\s+is[:\s]+(\d+(?:\.\d+)?)', "score_is"),
            (r'[Tt]otal[:\s]+(\d+(?:\.\d+)?)', "total"),
            (r'[Gg]rade[:\s]+(\d+(?:\.\d+)?)', "grade"),
        ]
        
        for pattern, method in score_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1), method
        
        # Strategy 3: Try to find any standalone number that could be a score (0-7)
        standalone_match = re.search(r'(?:^|[.\s])\s*(\d)(?:\s*$|[.\s])', text)
        if standalone_match:
            return standalone_match.group(1), "standalone"
        
        # Strategy 4: Last resort - any digit
        digit_match = re.search(r'\d', text)
        if digit_match:
            return digit_match.group(), "digit"
        
        return "0", "none"

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

        # Extract prediction from response
        prediction = "0"
        extraction_method = "none"
        
        try:
            # Get the assistant's response text
            assistant_text = msg_history[-1]["text"] if msg_history else ""
            
            # Extract score using consolidated method
            prediction, extraction_method = self._extract_score(assistant_text)
            
            if extraction_method != "none":
                self.log_fn(f"Extracted score using method '{extraction_method}': {prediction}")
            else:
                self.log_fn("No score found in response, defaulting to 0")
                            
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            # Try to extract any number from the response as fallback
            try:
                if msg_history:
                    last_text = msg_history[-1]["text"]
                    number_match = re.search(r'\d+', last_text)
                    if number_match:
                        prediction = number_match.group()
                        extraction_method = "error_fallback"
            except Exception:
                pass

        # Validate and normalize the score
        original_prediction = prediction
        prediction = self._validate_score(prediction)
        
        if original_prediction != prediction:
            self.log_fn(f"Score normalized: {original_prediction} -> {prediction}")
        
        self.log_fn(f"Final score (extraction: {extraction_method}): {prediction}")

        return str(prediction), msg_history
