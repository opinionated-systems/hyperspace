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
    Also handles markdown code blocks as fallback.
    Includes robust error recovery and nested brace handling.
    """
    results = []
    search_from = 0
    
    def _extract_json_from_text(text: str) -> dict | None:
        """Try to extract a valid JSON object from text with nested brace handling."""
        # Find all possible JSON object start positions
        json_start = text.find("{")
        while json_start != -1:
            # Try to find the matching closing brace by counting braces
            brace_count = 0
            json_end = -1
            for i, char in enumerate(text[json_start:], start=json_start):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_end = i
                        break
            
            if json_end != -1:
                try:
                    candidate = text[json_start:json_end + 1]
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    pass
            # Try next opening brace
            json_start = text.find("{", json_start + 1)
        return None
    
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
        
        # Try direct parsing first
        try:
            results.append(json.loads(inner))
            continue
        except json.JSONDecodeError:
            pass
        
        # Try extracting JSON object with nested brace handling
        extracted = _extract_json_from_text(inner)
        if extracted:
            results.append(extracted)
            continue
        
        # Try cleaning common issues and re-parsing
        try:
            # Remove trailing commas before closing braces
            cleaned = re.sub(r',(\s*[}\]])', r'\1', inner)
            # Fix single quotes to double quotes (common LLM error)
            cleaned = re.sub(r"'([^']*?)'(?=\s*:)", r'"\1"', cleaned)
            results.append(json.loads(cleaned))
        except json.JSONDecodeError:
            pass
    
    # If no <json> blocks found, try markdown code blocks as fallback
    if not results:
        search_from = 0
        while True:
            start = text.find("```json", search_from)
            if start == -1:
                # Try without 'json' specifier
                start = text.find("```", search_from)
                if start == -1:
                    break
                end_marker = "```"
                start += 3
            else:
                end_marker = "```"
                start += 7
            
            end = text.find(end_marker, start)
            if end == -1:
                break
            inner = text[start:end].strip()
            search_from = end + len(end_marker)
            
            # Try direct parsing first
            try:
                results.append(json.loads(inner))
                continue
            except json.JSONDecodeError:
                pass
            
            # Try extracting JSON object with nested brace handling
            extracted = _extract_json_from_text(inner)
            if extracted:
                results.append(extracted)
                continue
            
            # Try cleaning common issues
            try:
                cleaned = re.sub(r',(\s*[}\]])', r'\1', inner)
                cleaned = re.sub(r"'([^']*?)'(?=\s*:)", r'"\1"', cleaned)
                results.append(json.loads(cleaned))
            except json.JSONDecodeError:
                pass
    
    # Final fallback: try to find any JSON-like structure in the entire text
    if not results:
        extracted = _extract_json_from_text(text)
        if extracted:
            results.append(extracted)
    
    return results or None


# Few-shot examples for IMO grading
_FEW_SHOT_EXAMPLES = """
Example 1 - Complete Correct Answer:
Problem: Find all positive integers n such that n^2 + 3n + 2 is divisible by 4.
Solution: We can factor n^2 + 3n + 2 = (n+1)(n+2). For this to be divisible by 4, either (n+1) or (n+2) must be even, and at least one must be divisible by 2. Since n+1 and n+2 are consecutive integers, one is even. For the product to be divisible by 4, we need the even number to be divisible by 4, or both to be even (impossible for consecutive integers). So we need n+1 ≡ 0 (mod 4) or n+2 ≡ 0 (mod 4), meaning n ≡ 3 (mod 4) or n ≡ 2 (mod 4).
Grading Guidelines: Award 7 points for correct answer (n ≡ 2 or 3 mod 4). Award partial credit: 2 points for correct factorization, 2 points for recognizing need for divisibility by 4, 3 points for correct modular analysis.
Student Answer: n^2 + 3n + 2 = (n+1)(n+2). Since these are consecutive, one is even. For divisibility by 4, we need n ≡ 2 or 3 (mod 4).
Evaluation: The student correctly factored the expression, recognized the consecutive integer property, and arrived at the correct answer with proper modular arithmetic.
Score: 7

Example 2 - Complete Correct Answer:
Problem: Prove that for any positive integer n, n^3 + 2n is divisible by 3.
Solution: We use induction. Base case: n=1, 1^3 + 2(1) = 3, divisible by 3. Inductive step: assume k^3 + 2k divisible by 3. Then (k+1)^3 + 2(k+1) = k^3 + 3k^2 + 3k + 1 + 2k + 2 = k^3 + 2k + 3(k^2 + k + 1). By induction hypothesis, k^3 + 2k divisible by 3, and 3(k^2 + k + 1) clearly divisible by 3. Thus divisible by 3.
Grading Guidelines: Award 7 points for complete proof. Deduct 2 points for missing base case, 3 points for errors in inductive step, 2 points for lack of clarity.
Student Answer: By induction. Base case n=1 works. Assume true for k, then (k+1)^3 + 2(k+1) = k^3 + 3k^2 + 3k + 1 + 2k + 2 = (k^3 + 2k) + 3(k^2 + k + 1), both terms divisible by 3.
Evaluation: The student provided a complete induction proof with correct base case and inductive step. The algebraic manipulation is correct.
Score: 7

Example 3 - Partial Credit:
Problem: Prove that the sum of the first n positive integers is n(n+1)/2.
Solution: Use induction. Base case: n=1, sum is 1 = 1(2)/2. Inductive step: assume true for k, so 1+2+...+k = k(k+1)/2. Then 1+2+...+k+(k+1) = k(k+1)/2 + (k+1) = (k+1)(k/2 + 1) = (k+1)(k+2)/2.
Grading Guidelines: Award 7 points for complete proof. Award 3 points for correct base case only. Award 4 points for correct inductive hypothesis but errors in step. Award 6 points for minor algebraic errors in inductive step.
Student Answer: We can prove this by induction. For n=1, the sum is 1. The formula gives 1(2)/2 = 1. Now assume it works for some k. Then for k+1, we add k+1 to both sides.
Evaluation: The student correctly identified the base case and set up the induction framework, but did not complete the algebraic manipulation in the inductive step. This demonstrates understanding of the method but incomplete execution.
Score: 4

Example 4 - Incorrect Answer:
Problem: Find all prime numbers p such that p^2 + 2 is also prime.
Solution: Check small primes: p=2 gives 4+2=6 (not prime), p=3 gives 9+2=11 (prime), p=5 gives 25+2=27 (not prime), p=7 gives 49+2=51 (not prime). For p>3, p ≡ ±1 (mod 3), so p^2 ≡ 1 (mod 3), thus p^2+2 ≡ 0 (mod 3), divisible by 3 and greater than 3, hence not prime. Only p=3 works.
Grading Guidelines: Award 7 points for complete solution with modular arithmetic proof. Award 4 points for checking cases without general proof. Award 2 points for only finding p=3 without justification.
Student Answer: p=3 works because 9+2=11 is prime. Other primes don't work.
Evaluation: The student found the correct answer but provided no mathematical justification or proof. This is a minimal answer with no demonstration of understanding why other primes don't work.
Score: 2

Example 5 - Zero Score:
Problem: Prove there are infinitely many primes.
Solution: Assume finitely many primes p1, p2, ..., pn. Consider N = p1*p2*...*pn + 1. N is not divisible by any pi, so either N is prime or has a prime factor not in our list. Contradiction.
Grading Guidelines: Award 7 points for complete proof. Award 0 points for incorrect or irrelevant answers.
Student Answer: Primes are numbers like 2, 3, 5, 7. There are many of them.
Evaluation: The answer is completely inadequate. It merely lists a few primes and makes an unsupported claim without any mathematical reasoning or proof structure.
Score: 0
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

Instructions for grading:
1. Read the problem carefully and understand what is being asked
2. Study the official solution to understand the correct approach
3. Review the grading guidelines to understand how points should be awarded
4. Compare the student's answer to the official solution:
   - Identify correct steps and reasoning
   - Identify errors or missing steps
   - Assess the overall completeness and correctness
5. Assign a score from 0 to 7 based on the grading guidelines:
   - 7: Complete, correct solution
   - 5-6: Minor errors or omissions
   - 3-4: Partial solution with significant gaps
   - 1-2: Minimal progress or major errors
   - 0: No meaningful progress or completely wrong

You must respond with ONLY a JSON object in the following format:
<json>
{{
    "response": "<numerical_score>"
}}
</json>

IMPORTANT:
- The response field must contain ONLY a single integer from 0 to 7
- Do not include any explanation, reasoning, or additional text outside the JSON
- Do not use quotes around the number (e.g., use "7" not "\"7\"")
- Example of correct response: {{"response": "7"}}
- Example of correct response: {{"response": "3"}}"""

        return prompt

    def _validate_score(self, prediction: str) -> str:
        """Validate and normalize the score prediction.
        
        Handles various formats including:
        - Plain numbers: "7", "5"
        - Score labels: "Score: 7", "Points: 5"
        - Fractions: "7/7", "5/7"
        - Text with numbers: "The student scored 6 points"
        - Decimal scores: "6.5", "7.0"
        """
        if prediction is None:
            return "0"
        
        # Convert to string and strip whitespace
        prediction = str(prediction).strip()
        
        if not prediction:
            return "0"
        
        # Try to extract a number from the prediction
        # First, try to find patterns like "X/Y" (fraction format)
        fraction_match = re.search(r'(\d+(?:\.\d+)?)\s*/\s*\d+', prediction)
        if fraction_match:
            score = fraction_match.group(1)
            try:
                score_float = float(score)
                score_int = int(score_float)
                if 0 <= score_int <= 7:
                    return str(score_int)
            except ValueError:
                pass
        
        # Try to find standalone numbers or numbers after common prefixes
        # Look for patterns like "Score: 7", "7 points", "scored 6", etc.
        number_patterns = [
            r'(?:score|points?|grade|mark|value)\s*[:=]?\s*(\d+(?:\.\d+)?)',  # Score: 7, 7 points
            r'(?:scored|received|got|earned)\s+(\d+(?:\.\d+)?)',  # scored 7, received 6
            r'^(\d+(?:\.\d+)?)$',  # Just a number
            r'\b(\d+(?:\.\d+)?)\s*(?:points?|pts?|/\s*7)?\b',  # 7, 7 points, 7/7
        ]
        
        for pattern in number_patterns:
            match = re.search(pattern, prediction, re.IGNORECASE)
            if match:
                score = match.group(1)
                try:
                    score_float = float(score)
                    score_int = int(score_float)
                    # Validate it's a reasonable IMO score (0-7)
                    if 0 <= score_int <= 7:
                        return str(score_int)
                except ValueError:
                    continue
        
        # Fallback: just find any number in the text
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

        # Extract prediction from JSON with multiple fallback strategies
        prediction = "0"
        extraction_attempts = []
        
        try:
            # Get the last assistant response
            if msg_history and len(msg_history) > 0:
                last_msg = msg_history[-1]
                last_text = last_msg.get("text", "") if isinstance(last_msg, dict) else str(last_msg)
                extraction_attempts.append(("last_message", last_text))
                
                # Try to extract JSON from the response
                extracted = _extract_jsons(last_text)
                
                if extracted:
                    last_json = extracted[-1]
                    extraction_attempts.append(("json_extracted", str(last_json)))
                    
                    # Try to get the response field
                    if isinstance(last_json, dict):
                        if "response" in last_json:
                            prediction = last_json["response"]
                        elif "score" in last_json:
                            prediction = last_json["score"]
                        elif "grade" in last_json:
                            prediction = last_json["grade"]
                        elif "points" in last_json:
                            prediction = last_json["points"]
                        else:
                            # Use the first numeric value found in the dict
                            for key, value in last_json.items():
                                if isinstance(value, (int, float)) or (isinstance(value, str) and value.isdigit()):
                                    prediction = str(value)
                                    break
                    else:
                        prediction = str(last_json)
        except Exception as e:
            self.log_fn(f"Error extracting prediction from JSON: {e}")
        
        # If JSON extraction failed, try direct number extraction from the response
        if prediction == "0" and extraction_attempts:
            try:
                for source, text in extraction_attempts:
                    if text:
                        # Look for patterns like "Score: 7" or just "7"
                        score_match = re.search(r'(?:score|grade|points?)?\s*[:=]?\s*(\d+)', text, re.IGNORECASE)
                        if score_match:
                            prediction = score_match.group(1)
                            break
            except Exception as e:
                self.log_fn(f"Error in fallback extraction: {e}")

        # Validate and normalize the score
        prediction = self._validate_score(prediction)
        
        self.log_fn(f"Final prediction: {prediction}")

        return str(prediction), msg_history
