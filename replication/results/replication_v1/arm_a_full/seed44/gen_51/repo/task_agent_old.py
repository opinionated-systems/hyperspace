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


# Few-shot examples for IMO grading - Simplified format
_FEW_SHOT_EXAMPLES = """
EXAMPLE 1 - Score 7 (Complete Solution):
Problem: Find all positive integers n such that n^2 + 3n + 2 is divisible by 4.
Official Solution: We can factor n^2 + 3n + 2 = (n+1)(n+2). For this to be divisible by 4, either (n+1) or (n+2) must be even, and at least one must be divisible by 2. Since n+1 and n+2 are consecutive integers, one is even. For the product to be divisible by 4, we need n+1 ≡ 0 (mod 4) or n+2 ≡ 0 (mod 4), meaning n ≡ 3 (mod 4) or n ≡ 2 (mod 4).
Grading Guidelines: Award 7 points for correct answer (n ≡ 2 or 3 mod 4). Award partial credit: 2 points for correct factorization, 2 points for recognizing need for divisibility by 4, 3 points for correct modular analysis.
Student Answer: n^2 + 3n + 2 = (n+1)(n+2). Since these are consecutive, one is even. For divisibility by 4, we need n ≡ 2 or 3 (mod 4).
Score: 7

EXAMPLE 2 - Score 0 (Completely Wrong):
Problem: Find the remainder when 2^100 is divided by 3.
Official Solution: Note that 2 ≡ -1 (mod 3), so 2^100 ≡ (-1)^100 ≡ 1 (mod 3). The remainder is 1.
Grading Guidelines: Award 7 points for correct answer with valid reasoning. Award 2 points for recognizing pattern in powers of 2 mod 3. Award 0 points for incorrect answer.
Student Answer: 2^100 is even, so remainder is 2 when divided by 3.
Score: 0

EXAMPLE 3 - Score 4 (Partial Credit):
Problem: Prove that for any triangle with sides a, b, c, the area is at most (a+b+c)^2/(12√3).
Official Solution: Using Heron's formula and AM-GM inequality, we can show the equilateral triangle maximizes area for fixed perimeter. For equilateral triangle with side s, area = (√3/4)s^2 and semiperimeter = 3s/2. Substituting gives area = (a+b+c)^2/(12√3).
Grading Guidelines: Award 7 points for complete proof. Award 4 points for correct approach (Heron's + AM-GM) but incomplete execution. Award 2 points for stating equilateral maximizes area without proof.
Student Answer: The equilateral triangle has the maximum area for a given perimeter. If all sides equal s, then a+b+c = 3s, and area = (√3/4)s^2 = (√3/4)((a+b+c)/3)^2 = (a+b+c)^2/(12√3).
Score: 4

EXAMPLE 4 - Score 2 (Minimal Progress):
Problem: Find all functions f: R → R such that f(x+y) = f(x) + f(y) for all x, y ∈ R.
Official Solution: The solutions are f(x) = cx for some constant c. First show f(0) = 0. Then prove f(nx) = nf(x) for integers n, extend to rationals, and use continuity (if assumed) or other conditions for reals.
Grading Guidelines: Award 7 points for complete solution. Award 2 points for finding f(0)=0. Award 4 points for proving f(nx)=nf(x) for integers. Award 1 point for guessing linear form without proof.
Student Answer: f(0) = f(0+0) = f(0) + f(0), so f(0) = 0. Also f(x) = cx seems to work.
Score: 2
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

        # Truncate very long inputs to prevent token overflow
        max_len = 6000
        problem = problem[:max_len] if len(problem) > max_len else problem
        solution = solution[:max_len] if len(solution) > max_len else solution
        grading_guidelines = grading_guidelines[:max_len] if len(grading_guidelines) > max_len else grading_guidelines
        student_answer = student_answer[:max_len] if len(student_answer) > max_len else student_answer

        prompt = f"""You are an expert mathematics grader for IMO problems. Evaluate the student answer and assign a score from 0-7.

{_FEW_SHOT_EXAMPLES}

NOW EVALUATE THIS PROBLEM:

DOMAIN: {domain}

PROBLEM:
{problem}

OFFICIAL SOLUTION:
{solution}

GRADING GUIDELINES:
{grading_guidelines}

STUDENT ANSWER TO EVALUATE:
{student_answer}

INSTRUCTIONS:
1. Compare the student answer to the official solution
2. Use the grading guidelines to determine the appropriate score
3. Output ONLY the score as a single digit (0-7)

Your response must be in this exact format:
<json>
{{
    "response": "X"
}}
</json>

Where X is a single digit from 0 to 7."""

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

        # Log the raw response for debugging
        if msg_history:
            for msg in reversed(msg_history):
                if msg.get("role") == "assistant":
                    raw_response = msg.get("text", "")
                    logger.info(f"Raw LLM response (first 500 chars): {raw_response[:500]}")
                    break

        # Extract prediction
        prediction = self._extract_prediction(msg_history)
        logger.info(f"Extracted prediction before validation: {prediction}")
        
        # Validate and normalize the score
        prediction = self._validate_score(prediction)
        logger.info(f"Prediction after validation: {prediction}")
        
        # Additional normalization to ensure valid IMO range
        prediction = self._normalize_score(prediction)
        logger.info(f"Final validated prediction: {prediction}")

        return str(prediction), msg_history
