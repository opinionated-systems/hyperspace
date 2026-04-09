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
    Also handles markdown code blocks and bare JSON objects.
    Includes robust error recovery for malformed JSON.
    
    Enhanced to handle:
    - Nested braces within JSON values
    - Multiple JSON objects in a single block
    - Unicode and special characters
    - Common LLM formatting errors
    """
    results = []
    search_from = 0
    
    def _try_parse_json(inner: str) -> dict | None:
        """Try to parse JSON with multiple fallback strategies."""
        # Strategy 1: Direct parsing
        try:
            return json.loads(inner)
        except json.JSONDecodeError:
            pass
        
        # Strategy 2: Fix common JSON issues
        fixed = inner
        try:
            # Remove trailing commas before closing braces/brackets
            fixed = re.sub(r',(\s*[}\]])', r'\1', fixed)
            # Fix single quotes to double quotes for string values (but not apostrophes in words)
            fixed = re.sub(r"(?<!\w)'([^']*)'(?!\w)", r'"\1"', fixed)
            # Fix unquoted keys (but not in values)
            fixed = re.sub(r'([{,]\s*)(\w+)(\s*:)', r'\1"\2"\3', fixed)
            # Remove comments
            fixed = re.sub(r'//[^\n]*', '', fixed)
            fixed = re.sub(r'/\*.*?\*/', '', fixed, flags=re.DOTALL)
            # Normalize whitespace
            fixed = ' '.join(fixed.split())
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass
        
        # Strategy 3: Extract just the response field
        try:
            response_match = re.search(r'"response"\s*:\s*"?(\d+)"?', fixed, re.IGNORECASE)
            if response_match:
                return {"response": response_match.group(1)}
        except Exception:
            pass
        
        # Strategy 4: Look for any key-value pair that looks like a score
        try:
            score_match = re.search(r'"(\w+)"\s*:\s*"?(\d+)"?', fixed)
            if score_match:
                return {score_match.group(1): score_match.group(2)}
        except Exception:
            pass
        
        # Strategy 5: Handle cases where the value is a string containing a number
        try:
            # Match patterns like "response": "7" or "score": "5"
            str_num_match = re.search(r'"(\w+)"\s*:\s*"(\d+)"', fixed, re.IGNORECASE)
            if str_num_match:
                return {str_num_match.group(1): str_num_match.group(2)}
        except Exception:
            pass
        
        # Strategy 6: Handle bare numbers in JSON-like context
        try:
            # If the content looks like just a number, wrap it
            stripped = fixed.strip()
            if stripped.isdigit() or (stripped.startswith('-') and stripped[1:].isdigit()):
                return {"response": stripped}
        except Exception:
            pass
        
        return None
    
    # First, try to find <json>...</json> blocks with proper nesting handling
    while True:
        start = text.find("<json>", search_from)
        if start == -1:
            break
        end = text.find("</json>", start)
        if end == -1:
            break
        
        inner = text[start + 6:end].strip()
        search_from = end + 7
        
        # Try to parse the content
        parsed = _try_parse_json(inner)
        if parsed:
            results.append(parsed)
            continue
        
        # If direct parsing failed, try to extract multiple JSON objects
        # Handle case where LLM output multiple JSON objects concatenated
        brace_count = 0
        json_start = -1
        for i, char in enumerate(inner):
            if char == '{':
                if brace_count == 0:
                    json_start = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and json_start != -1:
                    try:
                        obj = inner[json_start:i+1]
                        parsed = _try_parse_json(obj)
                        if parsed:
                            results.append(parsed)
                    except Exception:
                        pass
                    json_start = -1
    
    # If no <json> blocks found, try markdown code blocks
    if not results:
        md_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
        for match in re.finditer(md_pattern, text, re.DOTALL):
            inner = match.group(1).strip()
            parsed = _try_parse_json(inner)
            if parsed:
                results.append(parsed)
    
    # If still no results, try to find bare JSON objects with improved pattern
    if not results:
        # Look for JSON-like structures, handling nested braces
        brace_count = 0
        json_start = -1
        for i, char in enumerate(text):
            if char == '{':
                if brace_count == 0:
                    json_start = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and json_start != -1:
                    try:
                        obj = text[json_start:i+1]
                        parsed = _try_parse_json(obj)
                        if parsed:
                            results.append(parsed)
                    except Exception:
                        pass
                    json_start = -1
    
    # Last resort: look for any "response": number pattern in raw text
    if not results:
        response_patterns = [
            r'"response"\s*:\s*"(\d+)"',
            r'"response"\s*:\s*(\d+)',
            r'response["\']?\s*[:=]\s*["\']?(\d+)',
            r'"score"\s*:\s*"(\d+)"',
            r'"score"\s*:\s*(\d+)',
            r'"grade"\s*:\s*"(\d+)"',
            r'"grade"\s*:\s*(\d+)',
        ]
        for pattern in response_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                # Determine the key from the pattern
                key = "response"
                if "score" in pattern.lower():
                    key = "score"
                elif "grade" in pattern.lower():
                    key = "grade"
                results.append({key: match.group(1)})
                break
    
    return results or None


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
        
        Enhanced with additional edge case handling for robust grading.
        """
        if prediction is None:
            logger.warning("Prediction is None, defaulting to 0")
            return "0"
        
        # Convert to string and strip whitespace
        prediction = str(prediction).strip()
        
        if not prediction:
            logger.warning("Prediction is empty, defaulting to 0")
            return "0"
        
        # Handle common text patterns that might appear before/after the number
        # Remove common prefixes/suffixes that might interfere with extraction
        cleaned = prediction
        prefixes_to_remove = [
            r'^[Ss]core[:\s]*',
            r'^[Gg]rade[:\s]*',
            r'^[Pp]oints?[:\s]*',
            r'^[Rr]esult[:\s]*',
            r'^[Aa]nswer[:\s]*',
            r'^["\'\(\[]+',
        ]
        suffixes_to_remove = [
            r'\s*[Pp]oints?$',
            r'\s*/\s*7$',
            r'["\'\)\]]+$',
        ]
        
        for prefix in prefixes_to_remove:
            cleaned = re.sub(prefix, '', cleaned)
        for suffix in suffixes_to_remove:
            cleaned = re.sub(suffix, '', cleaned)
        
        cleaned = cleaned.strip()
        
        # Handle fraction format like "7/7" - extract numerator
        if "/" in cleaned:
            parts = cleaned.split("/")
            if parts[0].strip().lstrip("-").isdigit():
                score = parts[0].strip()
                try:
                    score_int = int(score)
                    if 0 <= score_int <= 7:
                        logger.info(f"Extracted score {score_int} from fraction '{prediction}'")
                        return str(score_int)
                except ValueError:
                    pass
        
        # Try to extract a number from the prediction
        # Handle cases like "Score: 7", "7 points", "7.0", etc.
        # Look for decimal numbers first, then integers
        decimal_match = re.search(r'-?\d+\.\d+', cleaned)
        if decimal_match:
            try:
                score_float = float(decimal_match.group())
                score_int = int(round(score_float))
                if 0 <= score_int <= 7:
                    logger.info(f"Extracted score {score_int} from decimal '{prediction}'")
                    return str(score_int)
            except ValueError:
                pass
        
        # Try integer match (including negative numbers)
        number_match = re.search(r'-?\d+', cleaned)
        if number_match:
            score = number_match.group()
            try:
                score_int = int(score)
                # Clamp negative scores to 0, cap scores above 7 at 7
                if score_int < 0:
                    logger.warning(f"Negative score {score_int} clamped to 0")
                    return "0"
                elif score_int > 7:
                    logger.warning(f"Score {score_int} exceeds max 7, capping at 7")
                    return "7"
                else:
                    logger.info(f"Extracted score {score_int} from '{prediction}'")
                    return str(score_int)
            except ValueError:
                pass
        
        # If we can't extract a valid number, return "0"
        logger.warning(f"Could not extract valid score from '{prediction}', defaulting to 0")
        return "0"

    def _extract_with_llm(self, text: str) -> str:
        """Use LLM to extract score when other methods fail.
        
        This is a fallback method for difficult cases.
        """
        try:
            extraction_prompt = f"""Extract the numerical score (0-7) from this text. Respond with ONLY a single digit.

Text: {text[:500]}

Score:"""
            response, _, _ = get_response_from_llm(
                msg=extraction_prompt,
                model=self.model,
                msg_history=[],
            )
            # Try to extract just a number
            num_match = re.search(r'\d+', response)
            if num_match:
                score = num_match.group()
                logger.info(f"LLM extracted score: {score}")
                return score
        except Exception as e:
            logger.warning(f"LLM extraction failed: {e}")
        return "0"

    def _extract_prediction(self, msg_history: list[dict]) -> str:
        """Extract prediction from message history with enhanced robustness.
        
        Tries multiple strategies to extract a valid score:
        1. Extract from JSON response field
        2. Extract from any numeric field in JSON
        3. Extract from text using score-related patterns
        4. Extract any number as last resort
        5. Use regex patterns for common response formats
        
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
            else:
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
                    r'\*\*(\d+)\*\*',  # Bold number in markdown
                    r'\bscore\s+is\s+(\d+)\b',  # "score is 7"
                    r'\bgrade\s+is\s+(\d+)\b',  # "grade is 7"
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
                
                # Ultimate fallback: try LLM extraction
                logger.warning(f"Could not extract any number from text, trying LLM extraction: {last_text[:200]}")
                return self._extract_with_llm(last_text)
                
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
