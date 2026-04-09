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
    """Extract JSON objects from <json>...</json> blocks or markdown code blocks."""
    results = []
    search_from = 0
    
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
        
        try:
            results.append(json.loads(inner))
        except json.JSONDecodeError:
            continue
            
    return results or None


# Few-shot examples for IMO grading with detailed reasoning
_FEW_SHOT_EXAMPLES = """
EXAMPLE 1 - Score 7 (Complete Solution):
Problem: Find all positive integers n such that n^2 + 3n + 2 is divisible by 4.
Official Solution: We can factor n^2 + 3n + 2 = (n+1)(n+2). For this to be divisible by 4, either (n+1) or (n+2) must be even, and at least one must be divisible by 2. Since n+1 and n+2 are consecutive integers, one is even. For the product to be divisible by 4, we need n+1 ≡ 0 (mod 4) or n+2 ≡ 0 (mod 4), meaning n ≡ 3 (mod 4) or n ≡ 2 (mod 4).
Grading Guidelines: Award 7 points for correct answer (n ≡ 2 or 3 mod 4). Award partial credit: 2 points for correct factorization, 2 points for recognizing need for divisibility by 4, 3 points for correct modular analysis.
Student Answer: n^2 + 3n + 2 = (n+1)(n+2). Since these are consecutive, one is even. For divisibility by 4, we need n ≡ 2 or 3 (mod 4).
Score: 7
Reasoning: The student correctly factored the expression, recognized the consecutive integer property, and arrived at the correct answer with valid reasoning.

EXAMPLE 2 - Score 0 (Completely Wrong):
Problem: Find the remainder when 2^100 is divided by 3.
Official Solution: Note that 2 ≡ -1 (mod 3), so 2^100 ≡ (-1)^100 ≡ 1 (mod 3). The remainder is 1.
Grading Guidelines: Award 7 points for correct answer with valid reasoning. Award 2 points for recognizing pattern in powers of 2 mod 3. Award 0 points for incorrect answer.
Student Answer: 2^100 is even, so remainder is 2 when divided by 3.
Score: 0
Reasoning: The student's reasoning is fundamentally flawed. Being even has no direct relation to the remainder when dividing by 3. The correct answer is 1.

EXAMPLE 3 - Score 4 (Partial Credit - Good Approach, Incomplete):
Problem: Prove that for any triangle with sides a, b, c, the area is at most (a+b+c)^2/(12√3).
Official Solution: Using Heron's formula and AM-GM inequality, we can show the equilateral triangle maximizes area for fixed perimeter. For equilateral triangle with side s, area = (√3/4)s^2 and semiperimeter = 3s/2. Substituting gives area = (a+b+c)^2/(12√3).
Grading Guidelines: Award 7 points for complete proof. Award 4 points for correct approach (Heron's + AM-GM) but incomplete execution. Award 2 points for stating equilateral maximizes area without proof.
Student Answer: The equilateral triangle has the maximum area for a given perimeter. If all sides equal s, then a+b+c = 3s, and area = (√3/4)s^2 = (√3/4)((a+b+c)/3)^2 = (a+b+c)^2/(12√3).
Score: 4
Reasoning: The student correctly identified that the equilateral triangle maximizes area and verified the formula works for that case, but did not provide a general proof that this is indeed the maximum for all triangles.

EXAMPLE 4 - Score 2 (Minimal Progress):
Problem: Find all functions f: R → R such that f(x+y) = f(x) + f(y) for all x, y ∈ R.
Official Solution: The solutions are f(x) = cx for some constant c. First show f(0) = 0. Then prove f(nx) = nf(x) for integers n, extend to rationals, and use continuity (if assumed) or other conditions for reals.
Grading Guidelines: Award 7 points for complete solution. Award 2 points for finding f(0)=0. Award 4 points for proving f(nx)=nf(x) for integers. Award 1 point for guessing linear form without proof.
Student Answer: f(0) = f(0+0) = f(0) + f(0), so f(0) = 0. Also f(x) = cx seems to work.
Score: 2
Reasoning: The student only found f(0)=0 (2 points) and guessed the linear form without proof (0 points for that part).
"""


class TaskAgent:
    """Task agent that solves IMO grading problems."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def _build_prompt(self, inputs: dict) -> str:
        """Build the grading prompt."""
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        prompt = f"""You are an expert IMO (International Mathematical Olympiad) grader. Your task is to evaluate a student's solution and assign a score from 0 to 7.

Domain: {domain}

Problem:
{problem}

Official Solution:
{solution}

Grading Guidelines:
{grading_guidelines}

Student Answer:
{student_answer}

{_FEW_SHOT_EXAMPLES}

Now evaluate the student answer above. Follow these steps:
1. Carefully read the problem, official solution, and grading guidelines
2. Analyze what the student has written
3. Compare against the official solution and grading guidelines
4. Assign a score from 0 to 7 based on the guidelines

IMPORTANT: You must respond with a JSON object containing exactly these fields:
- "score": an integer from 0 to 7
- "reasoning": a brief explanation of why you gave this score

Format your response like this:
<json>
{{"score": 5, "reasoning": "The student had the right approach but missed some key steps..."}}
</json>
"""
        return prompt

    def _validate_score(self, raw_score: str) -> str:
        """Validate and normalize the score to 0-7 range."""
        try:
            # Extract first number from the string
            match = re.search(r'\d+', str(raw_score))
            if match:
                score_int = int(match.group())
            else:
                return "0"
            
            # Clamp to valid range
            if score_int < 0:
                return "0"
            elif score_int > 7:
                return "7"
            return str(score_int)
        except (ValueError, TypeError):
            return "0"

    def _extract_prediction(self, msg_history: list[dict]) -> str:
        """Extract prediction from message history."""
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
                    # Look for score field
                    if "score" in last_json:
                        score = last_json["score"]
                        logger.info(f"Extracted score from 'score' field: {score}")
                        return str(score)
                    # Try other common field names
                    for key in ["response", "answer", "grade", "points", "result"]:
                        if key in last_json:
                            score = last_json[key]
                            logger.info(f"Extracted score from '{key}' field: {score}")
                            return str(score)
                    # Try to find any numeric value
                    for key, value in last_json.items():
                        if isinstance(value, (int, float)):
                            logger.info(f"Using numeric value from key '{key}': {value}")
                            return str(int(value))
                        elif isinstance(value, str):
                            num_match = re.search(r'\d+', value)
                            if num_match:
                                logger.info(f"Using extracted number from key '{key}': {num_match.group()}")
                                return num_match.group()
                else:
                    # JSON is not a dict, try to convert to string and extract number
                    prediction = str(last_json)
                    num_match = re.search(r'\d+', prediction)
                    if num_match:
                        logger.info(f"Extracted number from non-dict JSON: {num_match.group()}")
                        return num_match.group()
            
            # No JSON found, try to extract score from text patterns
            logger.info("No JSON found, attempting to extract score from text")
            
            # Look for explicit score patterns
            score_patterns = [
                r'["\']score["\']\s*:\s*(\d+)',
                r'[Ss]core\s*[=:]\s*(\d+)',
                r'[Gg]rade\s*[=:]\s*(\d+)',
                r'[Pp]oints?\s*[=:]\s*(\d+)',
                r'[Aa]ward\s+(\d+)\s*points?',
                r'^(\d+)\s*$',  # Just a number on its own line
            ]
            for pattern in score_patterns:
                match = re.search(pattern, last_text, re.MULTILINE)
                if match:
                    prediction = match.group(1)
                    logger.info(f"Extracted score using pattern '{pattern}': {prediction}")
                    return prediction
            
            # Last resort: find any single digit 0-7 that appears to be a score
            # Look for numbers that appear after common score-related words
            context_patterns = [
                r'(?:score|grade|points?|award|assign|give|is|of)\s+(\d)',
                r'(\d)\s*(?:points?|/\s*7)',
            ]
            for pattern in context_patterns:
                match = re.search(pattern, last_text, re.IGNORECASE)
                if match:
                    num = int(match.group(1))
                    if 0 <= num <= 7:
                        logger.info(f"Extracted score from context pattern: {num}")
                        return str(num)
            
            # Very last resort: any digit
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
        
        # Validate the score
        prediction = self._validate_score(prediction)
        logger.info(f"Final validated prediction: {prediction}")

        return str(prediction), msg_history
