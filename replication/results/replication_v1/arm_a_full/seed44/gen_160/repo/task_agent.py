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
    
    # Final fallback: try to find any JSON-like object in the text
    if not results:
        try:
            json_start = text.find('{')
            json_end = text.rfind('}')
            if json_start != -1 and json_end != -1 and json_end > json_start:
                candidate = text[json_start:json_end+1]
                # Try to clean up common issues
                candidate = re.sub(r',\s*}', '}', candidate)  # Remove trailing commas
                candidate = re.sub(r',\s*]', ']', candidate)  # Remove trailing commas in arrays
                results.append(json.loads(candidate))
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

Instructions:
- Compare the student answer to the official solution
- Identify what parts are correct, partially correct, or incorrect
- Consider the grading guidelines for partial credit
- Assign an appropriate score

Respond in JSON format with the following schema:
<json>
{{
    "response": "<numerical_score>"
}}
</json>

The response field should contain only the numerical score (e.g., "7", "5", "0", etc.)."""

        return prompt

    def _validate_score(self, prediction: str) -> str:
        """Validate and normalize the score prediction.
        
        Handles various formats including:
        - Plain numbers: "7", "5"
        - Score prefixes: "Score: 7", "score: 5"
        - Fractions: "7/7", "5/7"
        - With units: "7 points", "5 pts"
        - Decimal scores: "6.5", "3.5" (rounded to nearest int)
        - Written numbers: "seven", "five"
        - Negative or out-of-range scores are clamped to valid range
        
        Includes comprehensive logging for debugging extraction issues.
        """
        if prediction is None:
            self.log_fn("Score validation: None prediction, defaulting to 0")
            return "0"
        
        # Convert to string and strip whitespace
        prediction = str(prediction).strip()
        original_prediction = prediction
        
        if not prediction:
            self.log_fn("Score validation: empty prediction, defaulting to 0")
            return "0"
        
        # Map written numbers to digits
        written_numbers = {
            'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
            'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
            'ten': 10
        }
        pred_lower = prediction.lower()
        for word, value in written_numbers.items():
            if word in pred_lower:
                result = str(max(0, min(7, value)))
                self.log_fn(f"Score validation: matched written number '{word}' -> {result}")
                return result
        
        # Try to extract a number from the prediction
        # First, try to find a fraction pattern like "5/7" and extract numerator
        fraction_match = re.search(r'(\d+(?:\.\d+)?)\s*/\s*\d+', prediction)
        if fraction_match:
            try:
                score_float = float(fraction_match.group(1))
                score_int = int(round(score_float))
                result = str(max(0, min(7, score_int)))
                self.log_fn(f"Score validation: extracted fraction '{fraction_match.group()}' -> {result}")
                return result
            except ValueError as e:
                self.log_fn(f"Score validation: fraction extraction failed: {e}")
        
        # Try to find a decimal or integer number
        # Look for patterns like "Score: 6.5", "7 points", "score: 5", etc.
        number_match = re.search(r'(?:score[:\s]*)?(\d+(?:\.\d+)?)', prediction, re.IGNORECASE)
        if number_match:
            try:
                score_float = float(number_match.group(1))
                score_int = int(round(score_float))
                # Clamp to valid IMO score range (0-7)
                result = str(max(0, min(7, score_int)))
                self.log_fn(f"Score validation: extracted number '{number_match.group()}' -> {result}")
                return result
            except ValueError as e:
                self.log_fn(f"Score validation: number extraction failed: {e}")
        
        # Fallback: try any digit sequence
        digit_match = re.search(r'\d+', prediction)
        if digit_match:
            try:
                score_int = int(digit_match.group())
                result = str(max(0, min(7, score_int)))
                self.log_fn(f"Score validation: extracted digit '{digit_match.group()}' -> {result}")
                return result
            except ValueError as e:
                self.log_fn(f"Score validation: digit extraction failed: {e}")
        
        # If we can't extract a valid number, return "0"
        self.log_fn(f"Score validation: could not extract score from '{original_prediction[:100]}...', defaulting to 0")
        return "0"

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Log the problem domain for debugging
        domain = inputs.get("domain", "unknown")
        self.log_fn(f"Processing problem in domain: {domain}")
        
        instruction = self._build_prompt(inputs)

        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
        except Exception as e:
            self.log_fn(f"Error calling LLM: {e}")
            return "0", [{"role": "error", "text": str(e)}]

        # Extract prediction from JSON
        prediction = "0"
        extraction_method = "none"
        
        try:
            if msg_history and len(msg_history) > 0:
                last_message = msg_history[-1]
                last_text = last_message.get("text", "")
                
                extracted = _extract_jsons(last_text)
                if extracted and "response" in extracted[-1]:
                    prediction = extracted[-1]["response"]
                    extraction_method = "json_response_key"
                elif extracted:
                    # If no "response" key, try to use the last JSON object
                    prediction = str(extracted[-1])
                    extraction_method = "json_fallback"
                else:
                    # Try to extract any number from the response as fallback
                    number_match = re.search(r'\d+', last_text)
                    if number_match:
                        prediction = number_match.group()
                        extraction_method = "regex_fallback"
            else:
                self.log_fn("Warning: empty message history from LLM")
                extraction_method = "empty_history"
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            extraction_method = "extraction_error"
            # Try to extract any number from the response as fallback
            try:
                if msg_history and len(msg_history) > 0:
                    last_text = msg_history[-1].get("text", "")
                    number_match = re.search(r'\d+', last_text)
                    if number_match:
                        prediction = number_match.group()
                        extraction_method = "regex_recovery"
            except Exception:
                pass

        # Log extraction method for debugging
        self.log_fn(f"Score extraction method: {extraction_method}, raw prediction: {prediction}")

        # Validate and normalize the score
        validated_prediction = self._validate_score(prediction)
        
        if prediction != validated_prediction:
            self.log_fn(f"Score normalized: {prediction} -> {validated_prediction}")

        return str(validated_prediction), msg_history
