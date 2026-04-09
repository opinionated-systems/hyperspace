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
    Includes robust error recovery for malformed JSON.
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
        
        # Try multiple extraction strategies
        parsed = _try_parse_json(inner)
        if parsed is not None:
            results.append(parsed)
    
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
            
            parsed = _try_parse_json(inner)
            if parsed is not None:
                results.append(parsed)
    
    # Final fallback: try to find any JSON-like structure in the text
    if not results:
        parsed = _try_parse_json(text)
        if parsed is not None:
            results.append(parsed)
    
    return results or None


def _try_parse_json(text: str) -> dict | None:
    """Try to parse JSON from text with multiple recovery strategies.
    
    Returns the parsed dict or None if parsing fails.
    """
    # Strategy 1: Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Strategy 2: Extract JSON object from surrounding text
    try:
        json_start = text.find("{")
        json_end = text.rfind("}")
        if json_start != -1 and json_end != -1 and json_end > json_start:
            return json.loads(text[json_start:json_end + 1])
    except json.JSONDecodeError:
        pass
    
    # Strategy 3: Try to fix common JSON errors
    try:
        # Remove trailing commas before closing braces
        fixed = re.sub(r',(\s*[}\]])', r'\1', text)
        # Fix single quotes to double quotes
        fixed = fixed.replace("'", '"')
        json_start = fixed.find("{")
        json_end = fixed.rfind("}")
        if json_start != -1 and json_end != -1 and json_end > json_start:
            return json.loads(fixed[json_start:json_end + 1])
    except (json.JSONDecodeError, ValueError):
        pass
    
    return None


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
        
        Handles various formats: "7", "Score: 7", "7 points", "7/7", 
        "7.0", partial credit like "5-6", etc.
        """
        if prediction is None:
            return "0"
        
        # Convert to string and strip whitespace
        prediction = str(prediction).strip()
        
        if not prediction:
            return "0"
        
        # Try to extract a number from the prediction
        # Handle cases like "Score: 7", "7 points", "7/7", "7.5", etc.
        
        # First, try to find a decimal or integer number
        number_match = re.search(r'\d+(?:\.\d+)?', prediction)
        if number_match:
            score_str = number_match.group()
            try:
                score_float = float(score_str)
                score_int = int(score_float)
                
                # Validate it's a reasonable IMO score (0-7)
                # IMO allows partial points, but we return integer for compatibility
                if 0 <= score_int <= 7:
                    return str(score_int)
                # Clamp to valid range
                if score_int < 0:
                    return "0"
                if score_int > 7:
                    return "7"
            except ValueError:
                pass
        
        # Handle range notation like "5-6" or "5/6" - take the higher value
        range_match = re.search(r'(\d+)[/-](\d+)', prediction)
        if range_match:
            try:
                high_score = int(range_match.group(2))
                if 0 <= high_score <= 7:
                    return str(high_score)
            except ValueError:
                pass
        
        # If we can't extract a valid number, return "0" as safe default
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
        try:
            # Get the last assistant response
            last_text = msg_history[-1]["text"] if msg_history else ""
            
            # Try to extract JSON
            extracted = _extract_jsons(last_text)
            
            if extracted:
                last_json = extracted[-1]
                
                # Try to get the response field
                if isinstance(last_json, dict):
                    if "response" in last_json:
                        prediction = str(last_json["response"])
                    elif "score" in last_json:
                        prediction = str(last_json["score"])
                    else:
                        # Use the first numeric value found
                        for key, value in last_json.items():
                            if isinstance(value, (int, float)):
                                prediction = str(value)
                                break
                            elif isinstance(value, str) and re.match(r'^\d+(?:\.\d+)?$', value.strip()):
                                prediction = value.strip()
                                break
                else:
                    prediction = str(last_json)
            else:
                # No JSON found - try to extract any number from the response
                number_match = re.search(r'\b(\d+(?:\.\d+)?)\b', last_text)
                if number_match:
                    prediction = number_match.group(1)
                    self.log_fn(f"No JSON found, extracted number: {prediction}")
                    
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        # Validate and normalize the score
        prediction = self._validate_score(prediction)

        return str(prediction), msg_history
