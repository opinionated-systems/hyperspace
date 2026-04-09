"""
Task agent: solves a given task with chain-of-thought reasoning and self-reflection.

Reimplemented from facebookresearch/HyperAgents task_agent.py.
Same interface, same JSON output format, same extraction logic.

This is the INITIAL task agent. The meta agent modifies this file
during self-improvement. The evaluation harness loads whatever
task_agent.py exists at the agent's repo path.
"""

from __future__ import annotations

import json
import logging
import math
import re

from agent.llm_client import get_response_from_llm, EVAL_MODEL

logger = logging.getLogger(__name__)


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    Also handles markdown code blocks and bare JSON objects.
    Includes robust error recovery for malformed JSON with detailed logging.
    
    Enhanced with proper string handling for nested braces, common LLM JSON errors,
    and improved performance through compiled regex patterns and safety limits.
    """
    results = []
    search_from = 0
    extraction_attempts = 0
    successful_extractions = 0
    
    # Pre-compile regex patterns for better performance
    _CODE_BLOCK_PATTERN = re.compile(r'```(?:json)?\s*\n?(.*?)\n?```', re.DOTALL)
    
    # Keywords that indicate a JSON object is likely a grading result
    _GRADING_KEYWORDS = ("score", "max_score", "thinking", "rationale", "response", "final_response")
    
    def _find_balanced_braces(text: str, start: int) -> tuple[int, int] | None:
        """Find a balanced brace pair starting from position, respecting string boundaries."""
        if start >= len(text) or text[start] != '{':
            return None
        brace_count = 0
        in_string = False
        escape_next = False
        
        for i, char in enumerate(text[start:], start=start):
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
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        return (start, i)
        return None
    
    def _fix_common_json_errors(json_str: str) -> str:
        """Apply common fixes for malformed JSON from LLMs.
        
        Fixes applied:
        1. Remove trailing commas before closing braces/brackets
        2. Convert unquoted single quotes to double quotes (outside strings)
        """
        # Remove trailing commas before closing braces/brackets
        fixed = re.sub(r',(\s*[}\]])', r'\1', json_str)
        
        # Fix single quotes to double quotes (common LLM error) - but not within strings
        result_fixed = []
        in_str = False
        escape = False
        for c in fixed:
            if escape:
                result_fixed.append(c)
                escape = False
                continue
            if c == '\\':
                result_fixed.append(c)
                escape = True
                continue
            if c == '"':
                in_str = not in_str
                result_fixed.append(c)
            elif c == "'" and not in_str:
                result_fixed.append('"')
            else:
                result_fixed.append(c)
        return ''.join(result_fixed)
    
    def _try_parse_json(json_str: str, source: str, idx: int) -> dict | None:
        """Try to parse JSON string with multiple recovery strategies."""
        nonlocal extraction_attempts, successful_extractions
        extraction_attempts += 1
        
        json_str = json_str.strip()
        if not json_str:
            return None
        
        try:
            result = json.loads(json_str)
            successful_extractions += 1
            logger.debug(f"Successfully extracted JSON from {source} #{idx}")
            return result
        except json.JSONDecodeError as e:
            logger.debug(f"JSON decode error in {source} #{idx}: {e}")
            
            # Strategy 1: Try to find inner JSON object with balanced braces
            try:
                bounds = _find_balanced_braces(json_str, 0)
                if bounds:
                    inner = json_str[bounds[0]:bounds[1]+1]
                    result = json.loads(inner)
                    successful_extractions += 1
                    logger.debug(f"Recovered JSON using balanced brace finder in {source} #{idx}")
                    return result
            except (json.JSONDecodeError, ValueError):
                pass
            
            # Strategy 2: Try common fixes for malformed JSON from LLMs
            try:
                fixed = _fix_common_json_errors(json_str)
                result = json.loads(fixed)
                successful_extractions += 1
                logger.debug(f"Recovered JSON using common fixes in {source} #{idx}")
                return result
            except json.JSONDecodeError:
                pass
            
            logger.warning(f"Failed to recover JSON from {source} #{idx}")
            return None
    
    def _is_grading_result(json_str: str) -> bool:
        """Check if a JSON string contains grading-related keywords."""
        return any(key in json_str for key in _GRADING_KEYWORDS)
    
    # First, try to find <json>...</json> blocks
    max_iterations = 100  # Safety limit to prevent infinite loops on malformed input
    iterations = 0
    while iterations < max_iterations:
        iterations += 1
        start = text.find("<json>", search_from)
        if start == -1:
            break
        end = text.find("</json>", start)
        if end == -1:
            logger.warning("Found <json> tag without closing </json> tag")
            break
        inner = text[start + 6:end].strip()
        search_from = end + 7
        
        result = _try_parse_json(inner, "<json> block", extraction_attempts + 1)
        if result:
            results.append(result)
    
    if iterations >= max_iterations:
        logger.warning(f"Reached max iterations ({max_iterations}) extracting <json> tags")
    
    # If no <json> blocks found, try markdown code blocks
    if not results:
        logger.debug("No <json> blocks found, trying markdown code blocks")
        matches = _CODE_BLOCK_PATTERN.findall(text)
        for i, match in enumerate(matches):
            result = _try_parse_json(match.strip(), "markdown block", i+1)
            if result:
                results.append(result)
    
    # Last resort: try to find bare JSON objects with proper brace balancing
    if not results:
        logger.debug("No code blocks found, trying bare JSON objects")
        i = 0
        max_bare_iterations = 10000  # Safety limit for bare JSON scanning
        bare_iterations = 0
        while i < len(text) and bare_iterations < max_bare_iterations:
            bare_iterations += 1
            if text[i] == '{':
                bounds = _find_balanced_braces(text, i)
                if bounds:
                    json_str = text[bounds[0]:bounds[1]+1]
                    # Only parse if it looks like a grading result
                    if _is_grading_result(json_str):
                        result = _try_parse_json(json_str, "bare JSON", len(results)+1)
                        if result:
                            results.append(result)
                    i = bounds[1] + 1
                else:
                    i += 1
            else:
                i += 1
        
        if bare_iterations >= max_bare_iterations:
            logger.warning(f"Reached max iterations ({max_bare_iterations}) scanning for bare JSON")
    
    logger.info(f"JSON extraction complete: {successful_extractions}/{extraction_attempts} successful, {len(results)} results returned")
    return results or None


def _validate_score(score: any, max_score: any) -> tuple[int, int] | None:
    """Validate and normalize score values.
    
    Returns normalized (score, max_score) tuple or None if invalid.
    Handles various edge cases including string numbers, None values,
    and out-of-range scores.
    """
    try:
        # Handle None values
        if score is None or max_score is None:
            return None
        
        # Handle string representations of numbers
        if isinstance(score, str):
            score = score.strip()
            if not score:
                return None
        if isinstance(max_score, str):
            max_score = max_score.strip()
            if not max_score:
                return None
                
        score_val = float(score)
        max_val = float(max_score)
        
        # Ensure non-negative and max_score > 0
        if score_val < 0 or max_val <= 0:
            return None
            
        # Handle NaN and Inf
        if math.isnan(score_val) or math.isnan(max_val):
            return None
        if math.isinf(max_val) or max_val > 1e9:  # Unreasonably large max_score
            return None
            
        # Cap score at max_score
        if score_val > max_val:
            score_val = max_val
            
        # Round to nearest integer for cleaner output
        return (round(score_val), round(max_val))
    except (ValueError, TypeError, OverflowError):
        return None


# Few-shot examples for IMO grading
FEW_SHOT_EXAMPLES = """
Example 1:
Problem: Find all positive integers n such that n^2 + 3n + 2 is divisible by 4.
Solution: n^2 + 3n + 2 = (n+1)(n+2). For divisibility by 4, either n+1 or n+2 must be even, and one of them must be divisible by 4. This happens when n ≡ 0 or 3 (mod 4).
Grading Guidelines: Award 1 point for factoring, 1 point for analyzing cases, 1 point for correct answer.
Student Answer: "I factored it as (n+1)(n+2). Since these are consecutive integers, one is even. For divisibility by 4, we need one factor divisible by 4. This happens when n=3,7,11,... or n=0,4,8,... So n ≡ 0 or 3 (mod 4)."
Grade: {"score": 3, "max_score": 3, "rationale": "Complete solution with correct factoring, case analysis, and answer."}

Example 2:
Problem: Prove that the sum of two odd numbers is even.
Solution: Let the odd numbers be 2k+1 and 2m+1. Their sum is 2k+1+2m+1 = 2(k+m+1), which is even.
Grading Guidelines: Award 1 point for setting up odd number representation, 1 point for algebraic manipulation, 1 point for concluding the result is even.
Student Answer: "Let two odd numbers be 2a+1 and 2b+1. Adding them: (2a+1) + (2b+1) = 2a + 2b + 2 = 2(a+b+1). This is clearly divisible by 2, so it's even."
Grade: {"score": 3, "max_score": 3, "rationale": "Complete proof with correct setup, algebra, and conclusion."}

Example 3:
Problem: Solve the equation x^2 - 5x + 6 = 0.
Solution: Factoring gives (x-2)(x-3) = 0, so x = 2 or x = 3.
Grading Guidelines: Award 1 point for correct factoring, 1 point for finding both solutions.
Student Answer: "x^2 - 5x + 6 = (x-2)(x-3) = 0, so x = 2."
Grade: {"score": 1, "max_score": 2, "rationale": "Correct factoring but only found one of the two solutions (x=2). Missed x=3."}

Example 3:
Problem: Prove that for any positive integer n, n^3 - n is divisible by 6.
Solution: n^3 - n = n(n-1)(n+1). This is the product of three consecutive integers, so it contains a multiple of 2 and a multiple of 3, making it divisible by 6.
Grading Guidelines: Award 1 point for factoring, 1 point for recognizing consecutive integers, 1 point for divisibility argument, 1 point for conclusion.
Student Answer: "n^3 - n = n(n^2-1) = n(n-1)(n+1). Among any three consecutive integers, one is divisible by 2 and one is divisible by 3. Therefore the product is divisible by 6."
Grade: {"score": 4, "max_score": 4, "rationale": "Excellent solution with correct factoring, clear recognition of consecutive integers property, and complete divisibility argument."}

Example 4:
Problem: Find the remainder when 2^100 is divided by 3.
Solution: Note that 2 ≡ -1 (mod 3), so 2^100 ≡ (-1)^100 ≡ 1 (mod 3). The remainder is 1.
Grading Guidelines: Award 1 point for recognizing the pattern/modular arithmetic, 1 point for correct computation, 1 point for final answer.
Student Answer: "2^1 = 2, 2^2 = 4, 2^3 = 8, 2^4 = 16... The remainders when divided by 3 are 2, 1, 2, 1... So the pattern repeats every 2. Since 100 is even, the remainder is 1."
Grade: {"score": 3, "max_score": 3, "rationale": "Correct answer obtained through pattern recognition. While modular arithmetic would be more elegant, the student's approach is mathematically valid and complete."}
"""


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with reasoning and reflection.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Step 1: Initial grading with chain-of-thought
        instruction = f"""You are an expert mathematical grader for the International Mathematical Olympiad (IMO).

Your task is to grade a student's answer to a mathematical problem.

{FEW_SHOT_EXAMPLES}

Now grade the following problem:

Domain: {inputs.get('domain', 'Mathematics')}

Problem:
{inputs.get('problem', '')}

Official Solution:
{inputs.get('solution', '')}

Grading Guidelines:
{inputs.get('grading_guidelines', '')}

Student Answer:
{inputs.get('student_answer', '')}

Think step by step:
1. Analyze what the student did correctly according to the official solution
2. Identify any errors, gaps, or missing steps
3. Compare against the grading guidelines
4. Determine the score and provide detailed rationale

Respond in JSON format with the following schema:
<json>
{{
    "thinking": "Your detailed step-by-step analysis here",
    "score": <numerical score>,
    "max_score": <maximum possible score>,
    "rationale": "Detailed explanation of why this score was awarded",
    "response": "<score>/<max_score> - <brief summary>"
}}
</json>"""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON
        prediction = "None"
        try:
            extracted = _extract_jsons(msg_history[-1]["text"])
            if extracted:
                result = extracted[-1]
                if "response" in result and result["response"]:
                    prediction = str(result["response"])
                elif "score" in result and "max_score" in result:
                    validated = _validate_score(result["score"], result["max_score"])
                    if validated:
                        prediction = f"{validated[0]}/{validated[1]}"
                    else:
                        prediction = f"{result['score']}/{result['max_score']}"
                elif "final_response" in result and result["final_response"]:
                    prediction = str(result["final_response"])
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        # Step 2: Self-reflection to verify the grade
        if prediction != "None" and len(msg_history) >= 2:
            reflection_msg = f"""Review your grading above carefully. Perform a critical self-check:

1. ACCURACY: Did you award points the student didn't actually earn? Check each claim against the official solution.
2. ERRORS: Did you miss any mathematical errors, logical gaps, or incorrect statements in the student's work?
3. CONSISTENCY: Is your score strictly consistent with the grading guidelines? Are you being too lenient or too harsh?
4. COMPLETENESS: Did the student address all parts of the problem? Are there missing cases or incomplete reasoning?
5. ALTERNATIVE APPROACHES: If the student used a different valid method than the official solution, did you fairly credit it?

Be honest and critical. If you find any issues, provide a corrected grade with clear justification.

Respond in JSON format:
<json>
{{
    "reflection": "Your detailed self-review addressing each point above",
    "revised_score": <score>,
    "revised_max_score": <max_score>,
    "final_response": "<score>/<max_score> - <brief summary of final assessment>"
}}
</json>"""
            
            reflection_response, msg_history, _ = get_response_from_llm(
                msg=reflection_msg,
                model=self.model,
                msg_history=msg_history,
            )
            
            # Try to extract revised prediction
            try:
                extracted = _extract_jsons(msg_history[-1]["text"])
                if extracted:
                    result = extracted[-1]
                    if "final_response" in result and result["final_response"]:
                        prediction = str(result["final_response"])
                    elif "revised_score" in result and "revised_max_score" in result:
                        validated = _validate_score(result["revised_score"], result["revised_max_score"])
                        if validated:
                            prediction = f"{validated[0]}/{validated[1]}"
                        else:
                            prediction = f"{result['revised_score']}/{result['revised_max_score']}"
                    # Fallback: if reflection contains score/max_score but not revised_* fields
                    elif "score" in result and "max_score" in result:
                        validated = _validate_score(result["score"], result["max_score"])
                        if validated:
                            prediction = f"{validated[0]}/{validated[1]}"
                        else:
                            prediction = f"{result['score']}/{result['max_score']}"
            except Exception as e:
                self.log_fn(f"Error extracting revised prediction: {e}")

        return str(prediction), msg_history
