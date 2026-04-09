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

    Uses a robust brace-tracking algorithm to handle nested JSON objects
    and avoid the lazy regex bug that truncates content.
    Also handles markdown code blocks and plain JSON objects.
    """
    results = []
    
    # Helper: find balanced JSON object starting at position
    def find_json_end(s: str, start: int) -> int:
        """Find the end of a JSON object starting at 'start' (should be '{').
        Returns index of the closing '}' or -1 if not found."""
        if start >= len(s) or s[start] != '{':
            return -1
        
        brace_depth = 1
        in_string = False
        escape_next = False
        
        for i in range(start + 1, len(s)):
            char = s[i]
            
            if escape_next:
                escape_next = False
                continue
            if char == '\\':
                escape_next = True
                continue
            if char == '"' and not in_string:
                in_string = True
            elif char == '"' and in_string:
                in_string = False
            elif not in_string:
                if char == '{':
                    brace_depth += 1
                elif char == '}':
                    brace_depth -= 1
                    if brace_depth == 0:
                        return i
        return -1
    
    # Strategy 1: Find <json>...</json> blocks with proper JSON parsing
    search_from = 0
    while True:
        start_tag = text.find("<json>", search_from)
        if start_tag == -1:
            break
        
        content_start = start_tag + 6
        # Find the JSON object within
        json_start = text.find('{', content_start)
        if json_start == -1:
            search_from = content_start
            continue
            
        json_end = find_json_end(text, json_start)
        if json_end == -1:
            # Fallback: find </json> tag
            end_tag = text.find("</json>", content_start)
            if end_tag == -1:
                break
            inner = text[content_start:end_tag].strip()
        else:
            inner = text[json_start:json_end + 1]
            end_tag = text.find("</json>", json_end)
            if end_tag == -1:
                end_tag = json_end
        
        search_from = end_tag + 7 if end_tag != -1 else json_end + 1
        
        try:
            results.append(json.loads(inner))
        except json.JSONDecodeError as e:
            logger.debug(f"Failed to parse JSON from <json> block: {e}")
            continue
    
    # Strategy 2: Markdown code blocks
    if not results:
        for pattern in [r'```json\s*(.*?)\s*```', r'```\s*(\{.*?\})\s*```']:
            for match in re.finditer(pattern, text, re.DOTALL):
                inner = match.group(1).strip()
                try:
                    results.append(json.loads(inner))
                except json.JSONDecodeError:
                    continue
            if results:
                break
    
    # Strategy 3: Find any JSON object with "response" key
    if not results:
        for match in re.finditer(r'\{[^{}]*"response"[^{}]*\}', text, re.IGNORECASE):
            try:
                results.append(json.loads(match.group()))
            except json.JSONDecodeError:
                continue
    
    # Strategy 4: Find any balanced JSON object
    if not results:
        for match in re.finditer(r'\{', text):
            json_end = find_json_end(text, match.start())
            if json_end != -1:
                try:
                    results.append(json.loads(text[match.start():json_end + 1]))
                except json.JSONDecodeError:
                    continue
    
    return results or None


# Few-shot examples for IMO grading - concise format for better token efficiency
_FEW_SHOT_EXAMPLES = """
Example 1 (Full Credit - 7 points):
Problem: Find all positive integers n such that n^2 + 3n + 2 is divisible by 4.
Grading Guidelines: Award 7 points for correct answer (n ≡ 2 or 3 mod 4). Award partial credit: 2 points for correct factorization, 2 points for recognizing need for divisibility by 4, 3 points for correct modular analysis.
Student Answer: n^2 + 3n + 2 = (n+1)(n+2). Since these are consecutive, one is even. For divisibility by 4, we need n ≡ 2 or 3 (mod 4).
<json>{"response": "7"}</json>

Example 2 (Full Credit - 7 points):
Problem: Prove that for any positive integer n, n^3 + 2n is divisible by 3.
Grading Guidelines: Award 7 points for complete proof. Deduct 2 points for missing base case, 3 points for errors in inductive step, 2 points for lack of clarity.
Student Answer: By induction. Base case n=1 works. Assume true for k, then (k+1)^3 + 2(k+1) = k^3 + 3k^2 + 3k + 1 + 2k + 2 = (k^3 + 2k) + 3(k^2 + k + 1), both terms divisible by 3.
<json>{"response": "7"}</json>

Example 3 (Partial Credit - 4 points):
Problem: Prove that the sum of the first n positive integers is n(n+1)/2.
Grading Guidelines: Award 7 points for complete proof. Award 3 points for correct base case only. Award 4 points for correct inductive step setup but errors in algebra. Award 6 points for minor algebraic error in inductive step.
Student Answer: Base case: n=1, 1 = 1(2)/2 = 1. Inductive step: Assume 1+2+...+k = k(k+1)/2. Then adding (k+1): sum = k(k+1)/2 + (k+1) = (k+1)(k/2 + 1).
<json>{"response": "4"}</json>

Example 4 (Zero Credit - 0 points):
Problem: Find the remainder when 2^100 is divided by 3.
Grading Guidelines: Award 7 points for correct answer with valid reasoning. Award 2 points for recognizing pattern in powers of 2 mod 3. Award 0 points for incorrect answer.
Student Answer: 2^100 is even, so remainder is 2 when divided by 3.
<json>{"response": "0"}</json>

Example 5 (Partial Credit - 2 points):
Problem: Find all positive integers n such that n^2 + 3n + 2 is divisible by 4.
Grading Guidelines: Award 7 points for correct answer (n ≡ 2 or 3 mod 4). Award partial credit: 2 points for correct factorization, 2 points for recognizing need for divisibility by 4, 3 points for correct modular analysis.
Student Answer: n^2 + 3n + 2 = (n+1)(n+2). The product of two consecutive integers.
<json>{"response": "2"}</json>
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

        prompt = f"""You are an expert IMO mathematics grader. Evaluate the student answer against the official solution and grading guidelines.

{_FEW_SHOT_EXAMPLES}

---

NOW EVALUATE:

Domain: {domain}

Problem:
{problem}

Official Solution:
{solution}

Grading Guidelines:
{grading_guidelines}

Student Answer:
{student_answer}

---

Evaluation Steps:
1. Understand what the problem asks and key concepts
2. Analyze the official solution approach
3. Review grading guidelines for partial credit
4. Evaluate the student's answer:
   - What is correct?
   - What errors or omissions exist?
   - Is reasoning sound?
5. Assign score (0-7) based on guidelines

Scoring Scale:
- 7: Complete, correct solution
- 6: Minor flaw in correct solution
- 5: Significant progress, missing key element
- 3-4: Partial solution with correct elements
- 1-2: Minimal progress or relevant insight
- 0: No meaningful progress or wrong

Respond ONLY with:
<json>{{"response": "<score>"}}</json>

Where <score> is a single integer 0-7."""

        return prompt

    def _validate_score(self, prediction: str) -> str:
        """Validate and normalize the score prediction.
        
        Handles various formats like "Score: 7", "7 points", "7/7", "7.0", etc.
        Ensures the score is within valid IMO range (0-7).
        """
        if prediction is None:
            return "0"
        
        prediction = str(prediction).strip()
        
        # Handle fraction format like "7/7" - extract numerator
        if "/" in prediction:
            parts = prediction.split("/")
            if parts[0].strip().isdigit():
                try:
                    score_int = int(parts[0].strip())
                    if 0 <= score_int <= 7:
                        return str(score_int)
                except ValueError:
                    pass
        
        # Try decimal numbers first, then integers
        decimal_match = re.search(r'\d+\.\d+', prediction)
        if decimal_match:
            try:
                score_int = int(round(float(decimal_match.group())))
                if 0 <= score_int <= 7:
                    return str(score_int)
            except ValueError:
                pass
        
        # Try integer match
        number_match = re.search(r'\d+', prediction)
        if number_match:
            try:
                score_int = int(number_match.group())
                if 0 <= score_int <= 7:
                    return str(score_int)
            except ValueError:
                pass
        
        logger.warning(f"Could not extract valid score from '{prediction}', defaulting to 0")
        return "0"

    def _normalize_score(self, raw_score: str) -> str:
        """Normalize a raw score to the valid IMO range (0-7)."""
        try:
            score_int = int(raw_score)
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
            last_text = ""
            for msg in reversed(msg_history):
                if msg.get("role") == "assistant":
                    last_text = msg.get("text", "")
                    break
            
            if not last_text:
                logger.warning("No assistant message or empty text found")
                return "0"
            
            # Try to extract JSON first
            extracted = _extract_jsons(last_text)
            if extracted:
                last_json = extracted[-1]
                if isinstance(last_json, dict):
                    if "response" in last_json:
                        prediction = str(last_json["response"]).strip()
                        logger.info(f"Extracted from 'response' field: {prediction}")
                        return prediction
                    # Try any numeric value in the JSON
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
                else:
                    prediction = str(last_json)
                    logger.info(f"JSON is not a dict, using string: {prediction}")
                    return prediction
            
            # No JSON found, try score patterns
            logger.info("No JSON found, trying text patterns")
            
            # Enhanced patterns for better score extraction
            score_patterns = [
                # Direct score mentions
                r'[Ss]core[\s]*[:=][\s]*(\d+)',
                r'[Rr]esult[\s]*[:=][\s]*(\d+)',
                r'[Gg]rade[\s]*[:=][\s]*(\d+)',
                r'[Pp]oints?[\s]*[:=][\s]*(\d+)',
                r'[Aa]ward[\s]*[:=][\s]*(\d+)',
                r'[Ff]inal[\s]*[Ss]core[\s]*[:=][\s]*(\d+)',
                # Score in parentheses or brackets
                r'\(\s*(\d+)\s*(?:points?)?\s*\)',
                r'\[\s*(\d+)\s*(?:points?)?\s*\]',
                # Score with /7 format
                r'(\d+)\s*/\s*7',
                # Standalone number at end of line (likely a score)
                r'^(\d+)\s*$',
                # Number in quotes
                r'"(\d+)"',
                r"'(\d+)'",
                # Score mentioned in evaluation context
                r'(?:assign|give|is|equals?|of)[\s]+(\d+)',
                r'(?:worth|value)[\s]+(\d+)',
            ]
            for pattern in score_patterns:
                match = re.search(pattern, last_text, re.MULTILINE)
                if match:
                    prediction = match.group(1)
                    logger.info(f"Extracted score using pattern '{pattern}': {prediction}")
                    return prediction
            
            # Last resort: find any single digit 0-7 (most likely to be a score)
            single_digit_match = re.search(r'\b([0-7])\b', last_text)
            if single_digit_match:
                prediction = single_digit_match.group(1)
                logger.info(f"Extracted single digit as score: {prediction}")
                return prediction
            
            # Very last resort: find any digit sequence
            number_match = re.search(r'\d+', last_text)
            if number_match:
                prediction = number_match.group()
                logger.info(f"Extracted raw number: {prediction}")
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
                time.sleep(2 ** attempt)

        # Extract, validate, and normalize the score
        prediction = self._extract_prediction(msg_history)
        prediction = self._validate_score(prediction)
        prediction = self._normalize_score(prediction)
        logger.info(f"Final prediction: {prediction}")

        return str(prediction), msg_history
