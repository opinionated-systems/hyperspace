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
import random
import re
import time

from agent.llm_client import get_response_from_llm, EVAL_MODEL

logger = logging.getLogger(__name__)


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    Also handles markdown code blocks and plain JSON objects.
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
        try:
            results.append(json.loads(inner))
        except json.JSONDecodeError as e:
            # Try to fix common JSON issues
            fixed_inner = _attempt_json_repair(inner)
            if fixed_inner:
                try:
                    results.append(json.loads(fixed_inner))
                    logger.debug(f"Successfully parsed JSON after repair: {fixed_inner[:100]}")
                except json.JSONDecodeError:
                    logger.debug(f"Failed to parse JSON from <json> block even after repair: {e}")
            else:
                logger.debug(f"Failed to parse JSON from <json> block: {e}")
            continue
    
    # Also try markdown code blocks with json
    if not results:
        search_from = 0
        while True:
            start = text.find("```json", search_from)
            if start == -1:
                # Try without language specifier
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
            try:
                results.append(json.loads(inner))
            except json.JSONDecodeError as e:
                # Try to fix common JSON issues
                fixed_inner = _attempt_json_repair(inner)
                if fixed_inner:
                    try:
                        results.append(json.loads(fixed_inner))
                        logger.debug(f"Successfully parsed markdown JSON after repair: {fixed_inner[:100]}")
                    except json.JSONDecodeError:
                        logger.debug(f"Failed to parse JSON from markdown block even after repair: {e}")
                else:
                    logger.debug(f"Failed to parse JSON from markdown block: {e}")
                continue
    
    # Try to find plain JSON objects as last resort
    if not results:
        # Look for JSON-like patterns with nested brace support
        # Use a more sophisticated approach to find complete JSON objects
        json_candidates = _find_json_candidates(text)
        for candidate in json_candidates:
            try:
                results.append(json.loads(candidate))
            except json.JSONDecodeError:
                # Try repair
                fixed = _attempt_json_repair(candidate)
                if fixed:
                    try:
                        results.append(json.loads(fixed))
                    except json.JSONDecodeError:
                        pass
    
    return results or None


def _attempt_json_repair(text: str) -> str | None:
    """Attempt to repair common JSON formatting issues.
    
    Returns repaired JSON string or None if repair failed.
    """
    if not text or not text.strip():
        return None
    
    text = text.strip()
    
    # Fix single quotes to double quotes (common LLM error)
    # Only replace quotes that are not inside strings
    fixed = []
    in_string = False
    escape_next = False
    for i, char in enumerate(text):
        if escape_next:
            fixed.append(char)
            escape_next = False
            continue
        if char == '\\':
            fixed.append(char)
            escape_next = True
            continue
        if char == '"' and not in_string:
            in_string = True
            fixed.append(char)
        elif char == '"' and in_string:
            in_string = False
            fixed.append(char)
        elif char == "'" and not in_string:
            # Check if this looks like a JSON key or value boundary
            # by checking surrounding context
            if i > 0 and text[i-1] in '{,[':
                fixed.append('"')
            elif i < len(text) - 1 and text[i+1] in '}:,]':
                fixed.append('"')
            elif i > 0 and text[i-1] == ' ' and i < len(text) - 1 and text[i+1].isalnum():
                # Likely a string value
                fixed.append('"')
            else:
                fixed.append(char)
        else:
            fixed.append(char)
    
    text = ''.join(fixed)
    
    # Fix trailing commas in objects/arrays
    text = re.sub(r',(\s*[}\]])', r'\1', text)
    
    # Fix missing quotes around keys (simple cases)
    # Pattern: {key: value} -> {"key": value}
    text = re.sub(r'(\{|,\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', text)
    
    # Ensure the text starts with { or [ and ends with } or ]
    text = text.strip()
    if not (text.startswith('{') or text.startswith('[')):
        # Try to find the start of a JSON object
        obj_start = text.find('{')
        arr_start = text.find('[')
        if obj_start != -1 and (arr_start == -1 or obj_start < arr_start):
            text = text[obj_start:]
        elif arr_start != -1:
            text = text[arr_start:]
    
    # Balance braces
    open_braces = text.count('{') - text.count('}')
    open_brackets = text.count('[') - text.count(']')
    
    if open_braces > 0:
        text = text + ('}' * open_braces)
    if open_brackets > 0:
        text = text + (']' * open_brackets)
    
    return text if text else None


def _find_json_candidates(text: str) -> list[str]:
    """Find potential JSON object candidates in text.
    
    Uses brace matching to find complete JSON objects.
    """
    candidates = []
    i = 0
    while i < len(text):
        if text[i] == '{':
            # Try to find matching closing brace
            brace_count = 1
            j = i + 1
            in_string = False
            escape_next = False
            
            while j < len(text) and brace_count > 0:
                if escape_next:
                    escape_next = False
                    j += 1
                    continue
                if text[j] == '\\':
                    escape_next = True
                    j += 1
                    continue
                if text[j] == '"' and not in_string:
                    in_string = True
                elif text[j] == '"' and in_string:
                    in_string = False
                elif not in_string:
                    if text[j] == '{':
                        brace_count += 1
                    elif text[j] == '}':
                        brace_count -= 1
                j += 1
            
            if brace_count == 0:
                candidate = text[i:j]
                candidates.append(candidate)
                i = j
            else:
                i += 1
        else:
            i += 1
    
    return candidates


# Few-shot examples for IMO grading - comprehensive examples with chain-of-thought reasoning
_FEW_SHOT_EXAMPLES = """
Example 1 (Full Credit - 7 points):
Problem: Find all positive integers n such that n^2 + 3n + 2 is divisible by 4.
Grading Guidelines: Award 7 points for correct answer (n ≡ 2 or 3 mod 4). Award partial credit: 2 points for correct factorization, 2 points for recognizing need for divisibility by 4, 3 points for correct modular analysis.
Student Answer: n^2 + 3n + 2 = (n+1)(n+2). Since these are consecutive, one is even. For divisibility by 4, we need n ≡ 2 or 3 (mod 4).

Analysis:
1. Problem asks for all positive integers n where n^2 + 3n + 2 ≡ 0 (mod 4)
2. Official solution uses factorization and modular arithmetic
3. Student correctly factorized: (n+1)(n+2) - this is 2 points
4. Student recognized consecutive integers property - this is 2 points  
5. Student correctly determined n ≡ 2 or 3 (mod 4) - this is 3 points
6. Total: 2 + 2 + 3 = 7 points. Complete solution with correct reasoning.
<json>{"response": "7"}</json>

Example 2 (Full Credit - 7 points):
Problem: Prove that for any positive integer n, n^3 + 2n is divisible by 3.
Grading Guidelines: Award 7 points for complete proof. Deduct 2 points for missing base case, 3 points for errors in inductive step, 2 points for lack of clarity.
Student Answer: By induction. Base case n=1 works. Assume true for k, then (k+1)^3 + 2(k+1) = k^3 + 3k^2 + 3k + 1 + 2k + 2 = (k^3 + 2k) + 3(k^2 + k + 1), both terms divisible by 3.

Analysis:
1. Problem requires proving divisibility by 3 for all positive integers
2. Official solution likely uses induction or modular arithmetic
3. Student stated "Base case n=1 works" - base case present (no deduction)
4. Student set up inductive hypothesis correctly
5. Student expanded (k+1)^3 + 2(k+1) correctly
6. Student grouped terms to show both divisible by 3 - inductive step complete
7. Total: 7 points. Complete, correct proof.
<json>{"response": "7"}</json>

Example 3 (Partial Credit - 4 points):
Problem: Prove that the sum of the first n positive integers is n(n+1)/2.
Grading Guidelines: Award 7 points for complete proof. Award 3 points for correct base case only. Award 4 points for correct inductive step setup but errors in algebra. Award 6 points for minor algebraic error in inductive step.
Student Answer: Base case: n=1, 1 = 1(2)/2 = 1. Inductive step: Assume 1+2+...+k = k(k+1)/2. Then adding (k+1): sum = k(k+1)/2 + (k+1) = (k+1)(k/2 + 1).

Analysis:
1. Problem requires proving formula for sum of first n integers
2. Official solution uses induction with proper algebraic manipulation
3. Student correctly verified base case n=1: 1 = 1(2)/2 = 1 - 3 points
4. Student stated inductive hypothesis correctly
5. Student attempted inductive step but made algebraic error: (k+1)(k/2 + 1) should be (k+1)(k+2)/2
6. Student did not complete the proof to show it equals (k+1)(k+2)/2
7. Total: 4 points. Correct base case (3) + inductive step setup with errors (1).
<json>{"response": "4"}</json>

Example 4 (Zero Credit - 0 points):
Problem: Find the remainder when 2^100 is divided by 3.
Grading Guidelines: Award 7 points for correct answer with valid reasoning. Award 2 points for recognizing pattern in powers of 2 mod 3. Award 0 points for incorrect answer.
Student Answer: 2^100 is even, so remainder is 2 when divided by 3.

Analysis:
1. Problem asks for 2^100 mod 3
2. Official solution: 2 ≡ -1 (mod 3), so 2^100 ≡ (-1)^100 ≡ 1 (mod 3)
3. Student's reasoning is fundamentally flawed: being even doesn't determine remainder mod 3
4. Student's answer (remainder 2) is incorrect - correct answer is 1
5. Student did not recognize pattern 2^1≡2, 2^2≡1, 2^3≡2, 2^4≡1 (mod 3)
6. Total: 0 points. Incorrect answer with invalid reasoning.
<json>{"response": "0"}</json>

Example 5 (Partial Credit - 2 points):
Problem: Find all positive integers n such that n^2 + 3n + 2 is divisible by 4.
Grading Guidelines: Award 7 points for correct answer (n ≡ 2 or 3 mod 4). Award partial credit: 2 points for correct factorization, 2 points for recognizing need for divisibility by 4, 3 points for correct modular analysis.
Student Answer: n^2 + 3n + 2 = (n+1)(n+2). The product of two consecutive integers.

Analysis:
1. Problem asks for n where n^2 + 3n + 2 ≡ 0 (mod 4)
2. Official solution: factor and analyze mod 4
3. Student correctly factorized: (n+1)(n+2) - 2 points
4. Student noted consecutive integers but did not analyze divisibility by 4
5. Student did not determine which values of n satisfy the condition
6. No modular analysis performed - missing 5 points
7. Total: 2 points. Only factorization correct.
<json>{"response": "2"}</json>

Example 6 (Partial Credit - 5 points):
Problem: Prove that for any positive integer n, n^3 + 2n is divisible by 3.
Grading Guidelines: Award 7 points for complete proof. Award 5 points for correct approach with minor gap. Award 3 points for correct approach with significant gaps.
Student Answer: n^3 + 2n = n(n^2 + 2). If n ≡ 0 (mod 3), then n^3 + 2n ≡ 0. If n ≡ 1 (mod 3), then 1 + 2 = 3 ≡ 0. If n ≡ 2 (mod 3), then 8 + 4 = 12 ≡ 0.

Analysis:
1. Problem requires proving 3 | n^3 + 2n for all positive integers n
2. Official solution can use cases or induction
3. Student used case analysis by n mod 3 - valid approach
4. Case n ≡ 0: Correct, n^3 + 2n ≡ 0 + 0 = 0 (mod 3)
5. Case n ≡ 1: Student wrote "1 + 2 = 3" but should be 1^3 + 2(1) = 3 ≡ 0
6. Case n ≡ 2: Student wrote "8 + 4 = 12" but should be 2^3 + 2(2) = 8 + 4 = 12 ≡ 0
7. All cases covered and correct, but presentation slightly informal
8. Total: 5 points. Correct approach with all cases, minor presentation gap.
<json>{"response": "5"}</json>

Example 7 (Partial Credit - 3 points):
Problem: Find all positive integers n such that n^2 + 3n + 2 is divisible by 4.
Grading Guidelines: Award 7 points for correct answer (n ≡ 2 or 3 mod 4). Award partial credit: 2 points for correct factorization, 2 points for recognizing need for divisibility by 4, 3 points for correct modular analysis.
Student Answer: Testing values: n=1: 1+3+2=6, not divisible by 4. n=2: 4+6+2=12, divisible by 4. n=3: 9+9+2=20, divisible by 4. n=4: 16+12+2=30, not divisible by 4. Pattern: works for n=2,3,6,7...

Analysis:
1. Problem asks for all n where n^2 + 3n + 2 ≡ 0 (mod 4)
2. Official solution uses factorization and modular arithmetic
3. Student did not factorize - missing 2 points
4. Student tested specific values and identified pattern
5. Student recognized pattern (n ≡ 2,3 mod 4) but didn't prove it
6. Student showed some understanding but no formal proof
7. Total: 3 points. Pattern recognition without proof or factorization.
<json>{"response": "3"}</json>
"""


class TaskAgent:
    """Task agent that solves IMO grading problems."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def _build_prompt(self, inputs: dict) -> str:
        """Build a structured prompt for the grading task with chain-of-thought reasoning."""
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        prompt = f"""You are an expert IMO mathematics grader. Your task is to evaluate student answers against official solutions using the provided grading guidelines.

{_FEW_SHOT_EXAMPLES}

---

NOW EVALUATE THIS SUBMISSION:

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

REQUIRED EVALUATION PROCESS:

Step 1 - Problem Analysis:
- What is the problem asking for?
- What are the key mathematical concepts?
- What constitutes a complete solution?

Step 2 - Official Solution Analysis:
- What is the core approach/method?
- What are the critical steps that must be present?
- What alternative approaches are valid?

Step 3 - Grading Guidelines Analysis:
- What specific elements earn full credit (7 points)?
- What partial credit opportunities exist?
- How many points for each component?

Step 4 - Student Answer Evaluation:
- What did the student do correctly? (List specific elements)
- What errors, gaps, or omissions exist?
- Is the reasoning mathematically sound where present?
- Does the answer address what the problem asks?

Step 5 - Score Assignment:
Based on the above analysis, assign a score from 0-7 following the IMO grading scale:
- 7: Complete, correct solution with proper reasoning
- 6: Correct solution with very minor flaw or presentation issue
- 5: Correct approach, significant progress, missing one key element
- 4: Substantial partial solution with correct major elements
- 3: Some correct elements but incomplete or significant gaps
- 2: Minimal progress, some relevant insight
- 1: Very minimal progress, minor relevant insight
- 0: No meaningful progress, completely wrong, or irrelevant

IMPORTANT: Be precise and follow the grading guidelines exactly. Award partial credit as specified. Do not be overly harsh - credit what the student correctly demonstrates.

FINAL RESPONSE FORMAT:
Provide your analysis above, then respond with ONLY this JSON:
<json>{{"response": "<score>"}}</json>

Where <score> is a single integer 0-7."""

        return prompt

    def _validate_score(self, prediction: str) -> str:
        """Validate and normalize the score prediction.
        
        Handles various formats like "Score: 7", "7 points", "7/7", "7.0", "seven", etc.
        Ensures the score is within valid IMO range (0-7).
        """
        if prediction is None:
            return "0"
        
        prediction = str(prediction).strip()
        
        # Handle word numbers
        word_to_num = {
            'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
            'five': 5, 'six': 6, 'seven': 7
        }
        pred_lower = prediction.lower()
        for word, num in word_to_num.items():
            if word in pred_lower:
                logger.info(f"Extracted word number '{word}' -> {num}")
                return str(num)
        
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
                score_float = float(decimal_match.group())
                score_int = int(round(score_float))
                if 0 <= score_int <= 7:
                    logger.info(f"Rounded decimal {score_float} to {score_int}")
                    return str(score_int)
            except ValueError:
                pass
        
        # Try integer match - look for all numbers and prefer 0-7 range
        all_numbers = re.findall(r'\d+', prediction)
        valid_scores = [int(n) for n in all_numbers if 0 <= int(n) <= 7]
        
        if valid_scores:
            # Return the last valid score (most likely the final answer)
            result = str(valid_scores[-1])
            logger.info(f"Validated score from '{prediction}': {result}")
            return result
        
        # If no valid 0-7 numbers, check if there's any number
        if all_numbers:
            # Try to use the first number found, clamped to valid range
            try:
                num = int(all_numbers[0])
                if num < 0:
                    logger.warning(f"Score {num} below minimum, clamping to 0")
                    return "0"
                elif num > 7:
                    logger.warning(f"Score {num} above maximum, clamping to 7")
                    return "7"
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
        1. Extract from JSON response field (highest priority)
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
            
            # Try to extract JSON first - look for <json> tags specifically
            json_results = _extract_jsons(last_text)
            if json_results:
                # Use the last JSON found (most likely the final answer)
                for json_obj in reversed(json_results):
                    if isinstance(json_obj, dict):
                        if "response" in json_obj:
                            prediction = str(json_obj["response"]).strip()
                            logger.info(f"Extracted from 'response' field: {prediction}")
                            return prediction
                        # Try other common score field names
                        for key in ["score", "grade", "points", "result", "answer", "value"]:
                            if key in json_obj:
                                prediction = str(json_obj[key]).strip()
                                logger.info(f"Extracted from '{key}' field: {prediction}")
                                return prediction
                        # Try any numeric value in the JSON
                        for key, value in json_obj.items():
                            if isinstance(value, (int, float)):
                                # Check if it's a valid IMO score (0-7)
                                if 0 <= value <= 7:
                                    prediction = str(int(value))
                                    logger.info(f"Using numeric value from key '{key}': {prediction}")
                                    return prediction
                            elif isinstance(value, str):
                                # Try to extract number from string value
                                num_match = re.search(r'\d+', value)
                                if num_match:
                                    num = int(num_match.group())
                                    if 0 <= num <= 7:
                                        prediction = str(num)
                                        logger.info(f"Using extracted number from key '{key}': {prediction}")
                                        return prediction
                    else:
                        # JSON is not a dict, try to convert to string and extract number
                        prediction = str(json_obj)
                        num_match = re.search(r'\d+', prediction)
                        if num_match:
                            num = int(num_match.group())
                            if 0 <= num <= 7:
                                logger.info(f"JSON is not a dict, extracted number: {num}")
                                return str(num)
            
            # No JSON found, try score patterns in order of specificity
            logger.info("No JSON found, trying text patterns")
            
            # Look for explicit score declarations first
            explicit_patterns = [
                r'[Ff]inal\s+[Ss]core[:\s]+(\d+)',
                r'[Ss]core[:\s]+(\d+)\s*(?:points?)?',
                r'[Gg]rade[:\s]+(\d+)',
                r'[Aa]ssign(?:ed)?\s+(?:a\s+)?(?:score\s+)?(?:of\s+)?(\d+)',
                r'[Aa]ward(?:ed)?\s+(\d+)\s*(?:points?)?',
                r'[Rr]esult[:\s]+(\d+)',
            ]
            for pattern in explicit_patterns:
                match = re.search(pattern, last_text)
                if match:
                    prediction = match.group(1)
                    logger.info(f"Extracted score using explicit pattern: {prediction}")
                    return prediction
            
            # Look for standalone numbers that could be scores
            # Prefer numbers that appear after analysis or conclusion
            score_context_patterns = [
                r'(?:[Tt]otal|[Ss]um|[Ff]inal)[^\n]*?(\d+)',
                r'(?:[Tt]herefore|[Hh]ence|[Tt]hus)[^\n]*?(\d+)',
                r'(?:[Gg]ive|[Aa]ssign)[^\n]*?(\d+)',
                r'<json>[^<]*?(\d+)[^<]*?</json>',
            ]
            for pattern in score_context_patterns:
                match = re.search(pattern, last_text)
                if match:
                    prediction = match.group(1)
                    num = int(prediction)
                    if 0 <= num <= 7:
                        logger.info(f"Extracted score using context pattern: {prediction}")
                        return prediction
            
            # Look for quoted numbers
            quoted_match = re.search(r'["\'](\d+)["\']', last_text)
            if quoted_match:
                prediction = quoted_match.group(1)
                num = int(prediction)
                if 0 <= num <= 7:
                    logger.info(f"Extracted quoted number: {prediction}")
                    return prediction
            
            # Last resort: find any digit sequence, prefer 0-7 range
            all_numbers = re.findall(r'\d+', last_text)
            valid_scores = [n for n in all_numbers if 0 <= int(n) <= 7]
            if valid_scores:
                # Use the last valid score found (most likely the final answer)
                prediction = valid_scores[-1]
                logger.info(f"Extracted last valid score from text: {prediction}")
                return prediction
            
            # If no valid 0-7 numbers, use any number
            if all_numbers:
                prediction = all_numbers[-1]
                logger.info(f"Extracted last number as fallback: {prediction}")
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
        
        # Try LLM call with retries and exponential backoff
        msg_history = []
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                logger.info(f"LLM call attempt {attempt + 1}/{max_retries + 1} with model {self.model}")
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=[],
                )
                # Log usage info if available
                if info and "usage" in info:
                    usage = info["usage"]
                    logger.info(f"LLM usage - prompt_tokens: {usage.get('prompt_tokens', 'N/A')}, "
                              f"completion_tokens: {usage.get('completion_tokens', 'N/A')}, "
                              f"total_tokens: {usage.get('total_tokens', 'N/A')}")
                break
            except Exception as e:
                last_error = e
                logger.error(f"LLM call failed (attempt {attempt + 1}/{max_retries + 1}): {e}")
                if attempt == max_retries:
                    logger.error(f"All {max_retries + 1} LLM call attempts exhausted. Last error: {last_error}")
                    return "0", [{"role": "error", "text": f"LLM call failed after {max_retries + 1} attempts: {str(last_error)}"}]
                # Exponential backoff with jitter: 2^attempt + random(0, 1)
                wait_time = (2 ** attempt) + random.random()
                logger.info(f"Retrying in {wait_time:.2f} seconds...")
                time.sleep(wait_time)

        # Extract, validate, and normalize the score
        prediction = self._extract_prediction(msg_history)
        logger.info(f"Raw prediction extracted: {prediction}")
        prediction = self._validate_score(prediction)
        logger.info(f"Validated prediction: {prediction}")
        prediction = self._normalize_score(prediction)
        logger.info(f"Final normalized prediction: {prediction}")

        return str(prediction), msg_history
