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
    
    Enhanced to handle:
    - Nested braces within JSON content
    - Multiple JSON blocks in the same text
    - Common formatting issues (trailing commas, comments)
    - Markdown code blocks containing JSON
    """
    results = []
    search_from = 0
    
    # Also handle markdown code blocks that might contain JSON
    text = _preprocess_json_text(text)
    
    while True:
        start = text.find("<json>", search_from)
        if start == -1:
            break
        end = text.find("</json>", start)
        if end == -1:
            break
        
        # Extract content between tags
        inner = text[start + 6:end].strip()
        search_from = end + 7
        
        # Try to parse the JSON
        parsed = _try_parse_json(inner)
        if parsed is not None:
            results.append(parsed)
            continue
            
        # If direct parsing fails, try to find valid JSON within the content
        # This handles cases where there's extra text before/after the JSON
        parsed = _extract_json_from_text(inner)
        if parsed is not None:
            results.append(parsed)
    
    return results or None


def _preprocess_json_text(text: str) -> str:
    """Preprocess text to normalize JSON markers."""
    # Convert markdown code blocks to <json> tags for uniform handling
    text = re.sub(r'```json\s*', '<json>', text)
    text = re.sub(r'```\s*', '</json>', text)
    return text


def _try_parse_json(text: str) -> dict | None:
    """Attempt to parse JSON with various cleanup strategies."""
    # First try direct parsing
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Try removing trailing commas before closing braces/brackets
    cleaned = re.sub(r',(\s*[}\]])', r'\1', text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    
    # Try removing C-style comments
    cleaned = re.sub(r'//[^\n]*', '', cleaned)
    cleaned = re.sub(r'/\*.*?\*/', '', cleaned, flags=re.DOTALL)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    
    # Try extracting just the object/array portion
    obj_start = text.find('{')
    arr_start = text.find('[')
    
    if obj_start != -1 and (arr_start == -1 or obj_start < arr_start):
        # Find matching closing brace
        brace_count = 0
        for i in range(obj_start, len(text)):
            if text[i] == '{':
                brace_count += 1
            elif text[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    try:
                        return json.loads(text[obj_start:i+1])
                    except json.JSONDecodeError:
                        break
                    
    elif arr_start != -1:
        # Find matching closing bracket
        bracket_count = 0
        for i in range(arr_start, len(text)):
            if text[i] == '[':
                bracket_count += 1
            elif text[i] == ']':
                bracket_count -= 1
                if bracket_count == 0:
                    try:
                        return json.loads(text[arr_start:i+1])
                    except json.JSONDecodeError:
                        break
    
    return None


def _extract_json_from_text(text: str) -> dict | None:
    """Try to extract a valid JSON object from text that may contain extra content."""
    # Look for JSON-like patterns
    patterns = [
        r'\{[^{}]*"response"[^{}]*\}',  # Simple object with response field
        r'\{[^{}]*\}',  # Any simple object
        r'\{.*\}',  # Object (greedy, may need refinement)
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                continue
    
    return None


# Few-shot examples for IMO grading
_FEW_SHOT_EXAMPLES = """
Example 1 (Full Credit - Complete Solution):
Problem: Find all positive integers n such that n^2 + 3n + 2 is divisible by 4.
Solution: We can factor n^2 + 3n + 2 = (n+1)(n+2). For this to be divisible by 4, either (n+1) or (n+2) must be even, and at least one must be divisible by 2. Since n+1 and n+2 are consecutive integers, one is even. For the product to be divisible by 4, we need the even number to be divisible by 4, or both to be even (impossible for consecutive integers). So we need n+1 ≡ 0 (mod 4) or n+2 ≡ 0 (mod 4), meaning n ≡ 3 (mod 4) or n ≡ 2 (mod 4).
Grading Guidelines: Award 7 points for correct answer (n ≡ 2 or 3 mod 4). Award partial credit: 2 points for correct factorization, 2 points for recognizing need for divisibility by 4, 3 points for correct modular analysis.
Student Answer: n^2 + 3n + 2 = (n+1)(n+2). Since these are consecutive, one is even. For divisibility by 4, we need n ≡ 2 or 3 (mod 4).
<json>
{
    "response": "7"
}
</json>

Example 2 (Full Credit - Complete Induction Proof):
Problem: Prove that for any positive integer n, n^3 + 2n is divisible by 3.
Solution: We use induction. Base case: n=1, 1^3 + 2(1) = 3, divisible by 3. Inductive step: assume k^3 + 2k divisible by 3. Then (k+1)^3 + 2(k+1) = k^3 + 3k^2 + 3k + 1 + 2k + 2 = k^3 + 2k + 3(k^2 + k + 1). By induction hypothesis, k^3 + 2k divisible by 3, and 3(k^2 + k + 1) clearly divisible by 3. Thus divisible by 3.
Grading Guidelines: Award 7 points for complete proof. Deduct 2 points for missing base case, 3 points for errors in inductive step, 2 points for lack of clarity.
Student Answer: By induction. Base case n=1 works. Assume true for k, then (k+1)^3 + 2(k+1) = k^3 + 3k^2 + 3k + 1 + 2k + 2 = (k^3 + 2k) + 3(k^2 + k + 1), both terms divisible by 3.
<json>
{
    "response": "7"
}
</json>

Example 3 (Partial Credit - Minor Error):
Problem: Prove that the sum of the first n positive integers is n(n+1)/2.
Solution: Use mathematical induction. Base case: n=1, sum is 1 = 1(2)/2. Inductive step: assume true for k, so 1+2+...+k = k(k+1)/2. Then 1+2+...+k+(k+1) = k(k+1)/2 + (k+1) = (k+1)(k/2 + 1) = (k+1)(k+2)/2, which is the formula for n=k+1.
Grading Guidelines: Award 7 points for complete proof. Award 3 points for correct base case only. Award 4 points for correct inductive step setup but errors in algebra. Award 6 points for minor algebraic error in inductive step.
Student Answer: Base case: n=1, 1 = 1(2)/2 = 1. Inductive step: Assume 1+2+...+k = k(k+1)/2. Then adding (k+1): sum = k(k+1)/2 + (k+1) = (k+1)(k+2)/2.
<json>
{
    "response": "7"
}
</json>

Example 4 (Zero Credit - Incorrect Answer):
Problem: Find the remainder when 2^100 is divided by 3.
Solution: Note that 2 ≡ -1 (mod 3), so 2^100 ≡ (-1)^100 ≡ 1 (mod 3). The remainder is 1.
Grading Guidelines: Award 7 points for correct answer with valid reasoning. Award 2 points for recognizing pattern in powers of 2 mod 3. Award 0 points for incorrect answer.
Student Answer: 2^100 is even, so remainder is 2 when divided by 3.
<json>
{
    "response": "0"
}
</json>

Example 5 (Partial Credit - Good Progress but Incomplete):
Problem: Let ABC be a triangle with AB = AC. Let D be the midpoint of BC. Prove that AD is perpendicular to BC.
Solution: Since AB = AC, triangle ABC is isosceles with A at the apex. In an isosceles triangle, the median from the apex to the base is also the altitude. Therefore AD ⊥ BC.
Grading Guidelines: Award 7 points for complete proof. Award 4-5 points for stating key properties (isosceles, median) but missing rigorous justification. Award 2-3 points for recognizing the triangle is isosceles. Award 0-1 points for minimal progress.
Student Answer: Triangle ABC is isosceles because AB = AC. D is the midpoint so AD is a median. In isosceles triangles, the median from the apex is perpendicular to the base.
<json>
{
    "response": "5"
}
</json>

Example 6 (Partial Credit - Correct Approach, Calculation Error):
Problem: Find the sum of all positive divisors of 60.
Solution: Prime factorization: 60 = 2^2 × 3 × 5. Sum of divisors formula: σ(n) = (2^3-1)/(2-1) × (3^2-1)/(3-1) × (5^2-1)/(5-1) = 7 × 4 × 6 = 168.
Grading Guidelines: Award 7 points for correct answer. Award 5-6 points for correct method with minor calculation error. Award 3-4 points for correct prime factorization but wrong formula application. Award 1-2 points for some correct divisors listed.
Student Answer: 60 = 2^2 × 3 × 5. Using sum formula: (4+1)(3+1)(5+1) = 5×4×6 = 120. Wait, let me recalculate: (2^3-1)/(2-1) = 7, (3^2-1)/(3-1) = 4, (5^2-1)/(5-1) = 6. So 7×4×6 = 168... I got 120 first but the correct answer is 168.
<json>
{
    "response": "6"
}
</json>
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

GRADING PRINCIPLES:
- 7 points: Complete, correct solution with clear reasoning
- 6 points: Minor error or omission in an otherwise complete solution
- 4-5 points: Significant progress with gaps in reasoning or missing key steps
- 2-3 points: Some correct ideas or partial progress toward solution
- 0-1 points: Little to no meaningful progress or completely incorrect

Partial credit should be awarded generously when the student demonstrates:
- Understanding of key concepts
- Correct approach even if execution has errors
- Meaningful progress toward the solution
- Valid mathematical reasoning that may be incomplete

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
1. Carefully compare the student answer to the official solution
2. Identify what parts are correct, partially correct, or incorrect
3. Consider the grading guidelines for partial credit
4. Assign an appropriate integer score between 0 and 7 (inclusive)
5. Respond ONLY with the JSON format below - no additional text before or after

IMPORTANT: Your response must be in this exact format:
<json>
{{
    "response": "<numerical_score>"
}}
</json>

The "response" field must contain ONLY a single integer between 0 and 7, as a string (e.g., "7", "5", "0", "3"). Do not include any other text, explanation, or formatting outside the JSON block."""

        return prompt

    def _validate_score(self, prediction: str) -> str:
        """Validate and normalize the score prediction.
        
        Handles various formats like "Score: 7", "7 points", "7/7", "7.0", etc.
        Ensures the score is within valid IMO range (0-7).
        
        Enhanced to handle:
        - Fractions (e.g., "6/7" -> "6")
        - Decimals (e.g., "6.5" -> "7" or "6")
        - Negative numbers (clamped to 0)
        - Numbers > 7 (clamped to 7)
        - Whitespace and punctuation
        """
        if prediction is None:
            return "0"
        
        # Convert to string and strip whitespace
        prediction = str(prediction).strip()
        
        # Remove common wrappers
        prediction = prediction.strip('"\'<>[]{}')
        
        # Handle fraction format like "7/7" or "6 / 7" - extract numerator
        if "/" in prediction:
            parts = prediction.split("/")
            if len(parts) >= 2:
                numerator = parts[0].strip()
                if numerator.replace('.', '', 1).replace('-', '', 1).isdigit():
                    try:
                        score_float = float(numerator)
                        score_int = int(round(score_float))
                        # Clamp to valid range
                        score_int = max(0, min(7, score_int))
                        return str(score_int)
                    except ValueError:
                        pass
        
        # Try to extract a number from the prediction
        # Handle cases like "Score: 7", "7 points", "7.0", etc.
        # Look for decimal numbers first, then integers
        decimal_match = re.search(r'-?\d+\.\d+', prediction)
        if decimal_match:
            try:
                score_float = float(decimal_match.group())
                # Clamp to valid range before rounding
                score_float = max(0, min(7, score_float))
                score_int = int(round(score_float))
                return str(score_int)
            except ValueError:
                pass
        
        # Try integer match (including negative)
        number_match = re.search(r'-?\d+', prediction)
        if number_match:
            score = number_match.group()
            try:
                score_int = int(score)
                # Clamp to valid range [0, 7]
                score_int = max(0, min(7, score_int))
                return str(score_int)
            except ValueError:
                pass
        
        # If we can't extract a valid number, return "0"
        logger.warning(f"Could not extract valid score from '{prediction}', defaulting to 0")
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
            
            # Try to extract JSON first (using enhanced extraction)
            extracted = _extract_jsons(last_text)
            if extracted:
                # Try each extracted JSON in reverse order (most recent first)
                for last_json in reversed(extracted):
                    if isinstance(last_json, dict):
                        # Priority 1: Look for "response" field
                        if "response" in last_json:
                            prediction = last_json["response"]
                            logger.info(f"Extracted prediction from 'response' field: {prediction}")
                            return str(prediction)
                        
                        # Priority 2: Look for score-related fields
                        score_fields = ["score", "grade", "points", "result", "value", "answer"]
                        for field in score_fields:
                            if field in last_json:
                                value = last_json[field]
                                if isinstance(value, (int, float)):
                                    logger.info(f"Extracted prediction from '{field}' field: {value}")
                                    return str(int(value))
                                elif isinstance(value, str):
                                    num_match = re.search(r'\d+', value)
                                    if num_match:
                                        prediction = num_match.group()
                                        logger.info(f"Extracted number from '{field}' string: {prediction}")
                                        return prediction
                        
                        # Priority 3: Any numeric value in the JSON
                        for key, value in last_json.items():
                            if isinstance(value, (int, float)):
                                prediction = str(int(value))
                                logger.info(f"Using numeric value from key '{key}': {prediction}")
                                return prediction
                            elif isinstance(value, str):
                                num_match = re.search(r'\d+', value)
                                if num_match:
                                    prediction = num_match.group()
                                    logger.info(f"Using extracted number from key '{key}': {prediction}")
                                    return prediction
                        
                        # Priority 4: String representation of the dict
                        prediction = str(last_json)
                        logger.info(f"No numeric key found, using full JSON string: {prediction}")
                        return prediction
                    else:
                        # JSON is not a dict (might be a list or primitive)
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
                r'\b(\d+)\s*/\s*7\b',  # Format: "6/7"
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

        # Extract prediction
        prediction = self._extract_prediction(msg_history)
        
        # Validate and normalize the score
        prediction = self._validate_score(prediction)
        logger.info(f"Final validated prediction: {prediction}")

        return str(prediction), msg_history
