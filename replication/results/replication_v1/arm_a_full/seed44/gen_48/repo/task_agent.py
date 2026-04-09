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
    Also handles markdown code blocks as a fallback.
    Includes robust error recovery for malformed JSON.
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
                    continue
            except Exception:
                pass
            # Try to extract reasoning and response separately
            try:
                reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]*)"', inner, re.DOTALL)
                response_match = re.search(r'"response"\s*:\s*"([^"]*)"', inner)
                if reasoning_match or response_match:
                    extracted = {}
                    if reasoning_match:
                        extracted["reasoning"] = reasoning_match.group(1)
                    if response_match:
                        extracted["response"] = response_match.group(1)
                    results.append(extracted)
                    continue
            except Exception:
                pass
            # Try to find any JSON-like structure with braces
            try:
                brace_start = inner.find('{')
                brace_end = inner.rfind('}')
                if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
                    potential_json = inner[brace_start:brace_end+1]
                    # Try to fix common JSON issues
                    potential_json = re.sub(r',\s*}', '}', potential_json)  # Remove trailing commas
                    potential_json = re.sub(r',\s*]', ']', potential_json)  # Remove trailing commas in arrays
                    results.append(json.loads(potential_json))
                    continue
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
                        potential_json = inner[json_start:json_end+1]
                        # Try to fix common JSON issues
                        potential_json = re.sub(r',\s*}', '}', potential_json)
                        potential_json = re.sub(r',\s*]', ']', potential_json)
                        results.append(json.loads(potential_json))
                except Exception:
                    continue
    
    # Final fallback: try to find any JSON-like structure in the entire text
    if not results:
        try:
            json_start = text.find('{')
            json_end = text.rfind('}')
            if json_start != -1 and json_end != -1 and json_end > json_start:
                potential_json = text[json_start:json_end+1]
                # Try to fix common JSON issues
                potential_json = re.sub(r',\s*}', '}', potential_json)
                potential_json = re.sub(r',\s*]', ']', potential_json)
                results.append(json.loads(potential_json))
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

Respond in JSON format with the following schema:
<json>
{{
    "reasoning": "<your detailed step-by-step evaluation reasoning>",
    "response": "<numerical_score>"
}}
</json>

The reasoning field should contain your detailed analysis. The response field should contain only the numerical score (e.g., "7", "5", "0", etc.)."""

        return prompt

    def _validate_score(self, prediction: str) -> str:
        """Validate and normalize the score prediction.
        
        Handles various formats including:
        - Plain numbers: "7", "5"
        - Score prefixes: "Score: 7", "score: 5"
        - Fractions: "7/7", "5/7"
        - With units: "7 points", "5 pts"
        - Decimal scores: "6.5", "3.5" (rounded to nearest int)
        - Negative or out-of-range scores are clamped to valid range
        - Handles edge cases like "0/7", "full marks", "no credit"
        """
        if prediction is None:
            return "0"
        
        # Convert to string and strip whitespace
        prediction = str(prediction).strip()
        
        if not prediction:
            return "0"
        
        # Handle special text cases
        lower_pred = prediction.lower()
        if any(phrase in lower_pred for phrase in ["full", "complete", "perfect", "all points"]):
            return "7"
        if any(phrase in lower_pred for phrase in ["no credit", "zero", "none", "incorrect", "wrong"]):
            return "0"
        
        # Try to extract a number from the prediction
        # First, try to find a fraction pattern like "5/7" and extract numerator
        fraction_match = re.search(r'(\d+(?:\.\d+)?)\s*/\s*\d+', prediction)
        if fraction_match:
            try:
                score_float = float(fraction_match.group(1))
                score_int = int(round(score_float))
                return str(max(0, min(7, score_int)))
            except ValueError:
                pass
        
        # Try to find a decimal or integer number
        # Look for patterns like "Score: 6.5", "7 points", "score: 5", etc.
        number_match = re.search(r'(?:score[:\s]*)?(\d+(?:\.\d+)?)', prediction, re.IGNORECASE)
        if number_match:
            try:
                score_float = float(number_match.group(1))
                score_int = int(round(score_float))
                # Clamp to valid IMO score range (0-7)
                return str(max(0, min(7, score_int)))
            except ValueError:
                pass
        
        # Fallback: try any digit sequence
        digit_match = re.search(r'\d+', prediction)
        if digit_match:
            try:
                score_int = int(digit_match.group())
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
        try:
            extracted = _extract_jsons(msg_history[-1]["text"])
            if extracted:
                last_extracted = extracted[-1]
                # Extract reasoning if present
                if "reasoning" in last_extracted:
                    reasoning = str(last_extracted["reasoning"])
                    self.log_fn(f"Grading reasoning: {reasoning[:200]}...")
                # Extract response/score
                if "response" in last_extracted:
                    prediction = last_extracted["response"]
                else:
                    # If no "response" key, try to use the last JSON object
                    prediction = str(last_extracted)
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            # Try to extract any number from the response as fallback
            try:
                last_text = msg_history[-1]["text"]
                number_match = re.search(r'\d+', last_text)
                if number_match:
                    prediction = number_match.group()
            except Exception:
                pass

        # Validate and normalize the score
        prediction = self._validate_score(prediction)

        return str(prediction), msg_history
