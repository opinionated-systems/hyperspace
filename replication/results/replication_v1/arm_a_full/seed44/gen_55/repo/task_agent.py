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
    Also handles markdown code blocks with json and bare JSON objects.
    """
    results = []
    search_from = 0
    
    # First try to find <json>...</json> blocks
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
    
    # Also try to find ```json code blocks
    search_from = 0
    while True:
        start = text.find("```json", search_from)
        if start == -1:
            break
        end = text.find("```", start + 7)
        if end == -1:
            break
        inner = text[start + 7:end].strip()
        search_from = end + 3
        try:
            results.append(json.loads(inner))
        except json.JSONDecodeError:
            # Try to fix common JSON issues
            try:
                fixed = re.sub(r',(\s*[}\]])', r'\1', inner)
                results.append(json.loads(fixed))
            except json.JSONDecodeError:
                continue
    
    # Also try to find bare JSON objects (for robustness)
    if not results:
        # Try to find JSON objects between curly braces with proper nesting
        import re as _re
        # Improved pattern that handles nested structures better
        json_pattern = _re.compile(r'\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}')
        for match in json_pattern.finditer(text):
            try:
                obj = json.loads(match.group())
                if isinstance(obj, dict):
                    results.append(obj)
            except json.JSONDecodeError:
                # Try to fix common JSON issues
                try:
                    fixed = re.sub(r',(\s*[}\]])', r'\1', match.group())
                    obj = json.loads(fixed)
                    if isinstance(obj, dict):
                        results.append(obj)
                except json.JSONDecodeError:
                    continue
    
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
Solution: Since AB = AC, triangle ABC is isosceles with A at the apex. D is the midpoint of BC. In an isosceles triangle, the median from the apex to the base is also the altitude. Therefore AD ⊥ BC. Alternatively, using coordinates: place D at origin, B at (-a, 0), C at (a, 0). Then A is at (0, h) for some h > 0. Vector AD = (0, -h), vector BC = (2a, 0). Their dot product is 0, so AD ⊥ BC.
Grading Guidelines: Award 7 points for complete proof. Award 3 points for recognizing isosceles property, 4 points for proving perpendicularity. Partial credit: 2 points for setting up coordinate system, 3 points for correct vector calculation.
Student Answer: Triangle ABC is isosceles with AB = AC. The median from A to BC is also the altitude. So AD is perpendicular to BC.
Score: 7

Example 4:
Problem: Find the sum of all positive divisors of 360.
Solution: First factor 360 = 2^3 × 3^2 × 5^1. The sum of divisors formula gives: σ(360) = (2^4-1)/(2-1) × (3^3-1)/(3-1) × (5^2-1)/(5-1) = 15 × 13 × 6 = 1170.
Grading Guidelines: Award 7 points for correct answer (1170). Award partial credit: 2 points for correct prime factorization, 3 points for knowing sum of divisors formula, 2 points for correct calculation.
Student Answer: 360 = 8 × 45 = 2^3 × 3^2 × 5. Sum = (1+2+4+8)(1+3+9)(1+5) = 15 × 13 × 6 = 1170.
Score: 7

Example 5:
Problem: Prove that √2 is irrational.
Solution: Assume √2 = p/q where p, q are coprime integers. Then 2q^2 = p^2, so p^2 is even, thus p is even. Let p = 2k. Then 2q^2 = 4k^2, so q^2 = 2k^2, making q even. But then p and q are both even, contradicting coprimality. Therefore √2 is irrational.
Grading Guidelines: Award 7 points for complete proof. Award partial credit: 2 points for assuming rational form, 2 points for correct algebraic manipulation, 3 points for deriving contradiction.
Student Answer: Assume √2 = a/b in lowest terms. Then 2b^2 = a^2, so a^2 is even, so a is even. Write a = 2c. Then 2b^2 = 4c^2, so b^2 = 2c^2, so b is even. Contradiction since a/b was in lowest terms.
Score: 7
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

Instructions for Grading:
1. First, identify the key mathematical concepts and steps in the official solution
2. Check which of these concepts appear in the student's answer
3. Look for:
   - Correct final answers (even with incomplete reasoning)
   - Valid proof techniques (induction, contradiction, direct proof, etc.)
   - Correct algebraic manipulations
   - Proper logical structure
   - Any errors or misconceptions
4. Consider partial credit generously when the student shows understanding
5. IMO problems are scored 0-7 points:
   - 7: Complete, correct solution
   - 6: Minor flaw in an otherwise correct solution
   - 5-3: Partial progress with significant gaps
   - 2-1: Minor progress or useful observations
   - 0: No meaningful progress or completely wrong

Respond in JSON format with the following schema:
<json>
{{
    "reasoning": "<your step-by-step analysis of the student's answer>",
    "response": "<numerical_score>"
}}
</json>

The reasoning field should contain your detailed analysis. The response field should contain only the numerical score (e.g., "7", "5", "0", etc.)."""

        return prompt

    def _validate_score(self, prediction: str) -> str:
        """Validate and normalize the score prediction."""
        if prediction is None:
            return "0"
        
        # Convert to string and strip whitespace
        prediction = str(prediction).strip()
        
        # Try to extract a number from the prediction
        # Handle cases like "Score: 7", "7 points", "7/7", etc.
        number_match = re.search(r'\d+', prediction)
        if number_match:
            score = number_match.group()
            # Validate it's a reasonable IMO score (0-7)
            try:
                score_int = int(score)
                if 0 <= score_int <= 7:
                    return str(score_int)
                elif score_int > 7:
                    # Cap at 7 if score is too high
                    return "7"
                else:
                    return "0"
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
        try:
            extracted = _extract_jsons(msg_history[-1]["text"])
            if extracted:
                last_json = extracted[-1]
                # Try to get "response" field first
                if isinstance(last_json, dict):
                    if "response" in last_json:
                        prediction = last_json["response"]
                    elif "score" in last_json:
                        prediction = last_json["score"]
                    else:
                        # Try to find any numeric value in the JSON
                        for key, value in last_json.items():
                            if isinstance(value, (int, float)) or (isinstance(value, str) and value.isdigit()):
                                prediction = str(value)
                                break
                else:
                    prediction = str(last_json)
        except Exception as e:
            self.log_fn(f"Error extracting prediction from JSON: {e}")
            # Try to extract any number from the response as fallback
            try:
                last_text = msg_history[-1]["text"]
                # Look for patterns like "Score: 7" or "score is 5"
                score_match = re.search(r'(?:score|response|points?)\s*[:=]\s*(\d+)', last_text, re.IGNORECASE)
                if score_match:
                    prediction = score_match.group(1)
                else:
                    # Just find any digit sequence
                    number_match = re.search(r'\d+', last_text)
                    if number_match:
                        prediction = number_match.group()
            except Exception:
                pass

        # Validate and normalize the score
        prediction = self._validate_score(prediction)

        return str(prediction), msg_history
