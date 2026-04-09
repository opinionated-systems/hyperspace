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
    Also attempts to extract raw JSON objects if tags are missing.
    
    Enhanced with better error handling and support for nested structures.
    """
    results = []
    search_from = 0
    
    # First, try to find JSON in <json> tags
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
        
        # Try to find JSON object boundaries manually with brace counting
        try:
            brace_start = inner.find('{')
            if brace_start == -1:
                continue
                
            brace_count = 0
            in_string = False
            escape_next = False
            
            for i, char in enumerate(inner[brace_start:]):
                if escape_next:
                    escape_next = False
                    continue
                    
                if char == '\\' and in_string:
                    escape_next = True
                    continue
                    
                if char == '"' and not in_string:
                    in_string = True
                elif char == '"' and in_string:
                    in_string = False
                elif not in_string:
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            json_str = inner[brace_start:brace_start+i+1]
                            results.append(json.loads(json_str))
                            break
        except (json.JSONDecodeError, ValueError):
            logger.debug(f"Failed to parse JSON from <json> block: {inner[:100]}...")
            continue
    
    # If no tagged JSON found, try to extract raw JSON objects
    if not results:
        search_from = 0
        while True:
            start = text.find('{', search_from)
            if start == -1:
                break
                
            # Try to find matching closing brace with proper string handling
            brace_count = 0
            in_string = False
            escape_next = False
            
            for i, char in enumerate(text[start:]):
                if escape_next:
                    escape_next = False
                    continue
                    
                if char == '\\' and in_string:
                    escape_next = True
                    continue
                    
                if char == '"' and not in_string:
                    in_string = True
                elif char == '"' and in_string:
                    in_string = False
                elif not in_string:
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            try:
                                obj = json.loads(text[start:start+i+1])
                                if isinstance(obj, dict):
                                    results.append(obj)
                            except json.JSONDecodeError:
                                pass
                            search_from = start + i + 1
                            break
            else:
                break
    
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
        
        Enhanced to handle more edge cases and provide better validation.
        """
        if prediction is None:
            return "0"
        
        # Convert to string and strip whitespace
        prediction = str(prediction).strip()
        
        # Handle empty string
        if not prediction:
            return "0"
        
        # Try to extract a number from the prediction
        # Handle cases like "Score: 7", "7 points", "7/7", "Score: 7/7", etc.
        
        # First, try to find a fraction pattern like "7/7" and extract numerator
        fraction_match = re.search(r'(\d+)\s*/\s*\d+', prediction)
        if fraction_match:
            score = fraction_match.group(1)
            try:
                score_int = int(score)
                if 0 <= score_int <= 7:
                    return str(score_int)
            except ValueError:
                pass
        
        # Try to find a number at the start (most common case)
        start_match = re.match(r'^(\d+)', prediction)
        if start_match:
            score = start_match.group(1)
            try:
                score_int = int(score)
                if 0 <= score_int <= 7:
                    return str(score_int)
            except ValueError:
                pass
        
        # Try to find any number in the string
        number_match = re.search(r'\d+', prediction)
        if number_match:
            score = number_match.group()
            # Validate it's a reasonable IMO score (0-7)
            try:
                score_int = int(score)
                if 0 <= score_int <= 7:
                    return str(score_int)
            except ValueError:
                pass
        
        # If we can't extract a valid number, return "0"
        logger.debug(f"Could not extract valid score from: {prediction}")
        return "0"

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Validate required inputs
        required_keys = ["problem", "solution", "grading_guidelines", "student_answer"]
        missing_keys = [k for k in required_keys if k not in inputs or not inputs[k]]
        if missing_keys:
            logger.warning(f"Missing required inputs: {missing_keys}")
        
        instruction = self._build_prompt(inputs)
        
        # Log prompt length for debugging
        logger.debug(f"Prompt length: {len(instruction)} characters")

        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return "0", [{"role": "error", "text": str(e)}]

        # Extract prediction from JSON
        prediction = "0"
        try:
            if not msg_history:
                logger.warning("Empty message history from LLM")
                return "0", msg_history
                
            last_message = msg_history[-1]
            if "text" not in last_message:
                logger.warning(f"Last message has no 'text' key: {last_message.keys()}")
                return "0", msg_history
                
            extracted = _extract_jsons(last_message["text"])
            if extracted and "response" in extracted[-1]:
                prediction = extracted[-1]["response"]
                logger.debug(f"Extracted prediction from 'response' key: {prediction}")
            elif extracted:
                # If no "response" key, try to use the last JSON object
                prediction = str(extracted[-1])
                logger.debug(f"Extracted prediction from JSON object: {prediction}")
            else:
                logger.debug("No JSON found in response, trying fallback extraction")
                # Try to extract any number from the response as fallback
                last_text = last_message["text"]
                number_match = re.search(r'\d+', last_text)
                if number_match:
                    prediction = number_match.group()
                    logger.debug(f"Extracted prediction via regex: {prediction}")
        except Exception as e:
            logger.error(f"Error extracting prediction: {e}")
            # Try to extract any number from the response as fallback
            try:
                if msg_history and "text" in msg_history[-1]:
                    last_text = msg_history[-1]["text"]
                    number_match = re.search(r'\d+', last_text)
                    if number_match:
                        prediction = number_match.group()
            except Exception:
                pass

        # Validate and normalize the score
        original_prediction = prediction
        prediction = self._validate_score(prediction)
        
        if original_prediction != prediction:
            logger.debug(f"Score normalized: {original_prediction} -> {prediction}")

        return str(prediction), msg_history
