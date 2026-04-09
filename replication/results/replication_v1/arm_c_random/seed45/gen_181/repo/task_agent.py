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
import re

from agent.llm_client import get_response_from_llm, EVAL_MODEL

logger = logging.getLogger(__name__)


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    Also handles markdown code blocks and inline JSON as fallbacks.
    Includes robust error recovery for common LLM formatting issues.
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
            logger.debug(f"JSON decode error in <json> block: {e}")
            # Try to fix common LLM formatting issues
            fixed = _attempt_json_repair(inner)
            if fixed:
                results.append(fixed)
            continue
    
    # Fallback 1: try markdown code blocks if no <json> blocks found
    if not results:
        # Look for ```json ... ``` blocks (with or without json label)
        markdown_pattern = r'```(?:json)?\s*\n(.*?)\n```'
        for match in re.finditer(markdown_pattern, text, re.DOTALL):
            try:
                results.append(json.loads(match.group(1).strip()))
            except json.JSONDecodeError:
                # Try repair on markdown blocks too
                fixed = _attempt_json_repair(match.group(1).strip())
                if fixed:
                    results.append(fixed)
                continue
    
    # Fallback 2: try to find JSON objects directly in the text
    if not results:
        # Look for patterns that look like JSON objects with balanced braces
        # This regex handles nested braces by counting
        def find_json_objects(s: str) -> list[str]:
            """Find all JSON-like objects in text using brace counting."""
            objects = []
            i = 0
            while i < len(s):
                if s[i] == '{':
                    start = i
                    count = 1
                    i += 1
                    in_string = False
                    escape_next = False
                    while i < len(s) and count > 0:
                        if escape_next:
                            escape_next = False
                        elif s[i] == '\\':
                            escape_next = True
                        elif s[i] == '"' and not escape_next:
                            in_string = not in_string
                        elif not in_string:
                            if s[i] == '{':
                                count += 1
                            elif s[i] == '}':
                                count -= 1
                        i += 1
                    if count == 0:
                        objects.append(s[start:i])
                else:
                    i += 1
            return objects
        
        candidates = find_json_objects(text)
        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
                # Only accept if it has expected keys for grading
                if any(key in parsed for key in ['score', 'response', 'thinking', 'rationale', 'max_score']):
                    results.append(parsed)
            except (json.JSONDecodeError, ValueError):
                # Try repair before giving up
                fixed = _attempt_json_repair(candidate)
                if fixed and any(key in fixed for key in ['score', 'response', 'thinking', 'rationale', 'max_score']):
                    results.append(fixed)
                continue
    
    # Fallback 3: try to find JSON-like patterns with score/max_score format
    if not results:
        # Look for patterns like {"score": X, "max_score": Y} or similar
        score_pattern = r'["\']?score["\']?\s*[:=]\s*(\d+)'
        max_score_pattern = r'["\']?max_score["\']?\s*[:=]\s*(\d+)'
        
        score_match = re.search(score_pattern, text, re.IGNORECASE)
        max_score_match = re.search(max_score_pattern, text, re.IGNORECASE)
        
        if score_match:
            score = int(score_match.group(1))
            max_score = int(max_score_match.group(1)) if max_score_match else score
            # Try to extract rationale/thinking if available
            rationale = ""
            thinking = ""
            
            # Look for rationale
            rationale_pattern = r'["\']?rationale["\']?\s*[:=]\s*["\']([^"\']+)["\']'
            rationale_match = re.search(rationale_pattern, text, re.IGNORECASE)
            if rationale_match:
                rationale = rationale_match.group(1)
            
            # Look for thinking
            thinking_pattern = r'["\']?thinking["\']?\s*[:=]\s*["\']([^"\']+)["\']'
            thinking_match = re.search(thinking_pattern, text, re.IGNORECASE)
            if thinking_match:
                thinking = thinking_match.group(1)
            
            results.append({
                'score': score,
                'max_score': max_score,
                'rationale': rationale or "Extracted from pattern match",
                'thinking': thinking or "",
                'response': f"{score}/{max_score}"
            })
    
    return results or None


def _attempt_json_repair(text: str) -> dict | None:
    """Attempt to repair common JSON formatting errors from LLM outputs.
    
    Fixes:
    - Trailing commas before closing braces/brackets
    - Single quotes instead of double quotes
    - Unquoted keys
    - Missing quotes around string values
    - Newlines in string values
    - Comments in JSON
    - Unicode escape sequences
    - Boolean/null lowercase variants
    - Nested quote escaping issues
    - Control characters in strings
    """
    import re
    
    original = text.strip()
    
    # Pre-fix: Remove control characters except common whitespace
    repaired = ''.join(char for char in original if ord(char) >= 32 or char in '\n\t\r')
    
    # Fix 1: Remove trailing commas before } or ]
    repaired = re.sub(r',\s*([}\]])', r'\1', repaired)
    
    # Fix 2: Replace single quotes with double quotes (carefully)
    # Only replace single quotes that appear to be delimiters
    repaired = re.sub(r"(?<!\\)'([^']*?)'(?=\s*[:}\],])", r'"\1"', repaired)
    
    # Fix 3: Add quotes to unquoted keys
    repaired = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', repaired)
    
    # Fix 4: Fix common escape sequence issues - handle double-escaped quotes
    # Be careful not to over-escape
    repaired = repaired.replace('\\"', '\"').replace('\\n', '\n').replace('\\t', '\t')
    
    # Fix 5: Remove C-style comments
    repaired = re.sub(r'//[^\n]*', '', repaired)
    repaired = re.sub(r'/\*.*?\*/', '', repaired, flags=re.DOTALL)
    
    # Fix 6: Normalize boolean and null values (Python True/False/None to JSON true/false/null)
    repaired = re.sub(r'(?<=[\s\[\{,])(True|False)(?=[\s\]\},])', lambda m: m.group(1).lower(), repaired)
    repaired = re.sub(r'(?<=[\s\[\{,])None(?=[\s\]\},])', 'null', repaired)
    
    # Fix 7: Handle unquoted string values (common in LLM outputs)
    # Match values that look like unquoted strings after colon
    def quote_unquoted_strings(match):
        value = match.group(1).strip()
        # Don't quote if it's already a number, boolean, or null
        if re.match(r'^-?\d+(\.\d+)?([eE][+-]?\d+)?$', value):
            return f': {value}'
        if value in ('true', 'false', 'null'):
            return f': {value}'
        if value.startswith('"') and value.endswith('"'):
            return f': {value}'
        # Quote the value, escaping internal quotes
        escaped_value = value.replace('"', '\\"')
        return f': "{escaped_value}"'
    
    repaired = re.sub(r':\s*([^",\[\]{}\s][^,\[\]{}]*?)(?=[,}\]])', quote_unquoted_strings, repaired)
    
    # Fix 8: Handle newlines and tabs in string values - escape them properly
    def escape_special_in_strings(match):
        content = match.group(1)
        # Escape unescaped newlines and tabs
        content = re.sub(r'(?<!\\)\n', r'\\n', content)
        content = re.sub(r'(?<!\\)\t', r'\\t', content)
        content = re.sub(r'(?<!\\)\r', r'\\r', content)
        return f'"{content}"'
    
    repaired = re.sub(r'"((?:[^"\\]|\\.)*?)"', escape_special_in_strings, repaired)
    
    # Fix 9: Handle consecutive string values without comma (common LLM error)
    repaired = re.sub(r'"\s+"', '", "', repaired)
    
    try:
        return json.loads(repaired)
    except json.JSONDecodeError as e:
        logger.debug(f"First repair attempt failed: {e}")
        
        # Second attempt: try to extract just the first JSON-like object
        try:
            # Find content between outermost braces
            start = repaired.find('{')
            end = repaired.rfind('}')
            if start != -1 and end != -1 and end > start:
                candidate = repaired[start:end+1]
                # Clean up the candidate
                candidate = re.sub(r',\s*([}\]])', r'\1', candidate)
                return json.loads(candidate)
        except json.JSONDecodeError as e2:
            logger.debug(f"Second repair attempt failed: {e2}")
            pass
        
        # Third attempt: aggressive cleaning - remove all newlines outside strings
        try:
            # Simple approach: find the JSON structure and clean it
            start = repaired.find('{')
            end = repaired.rfind('}')
            if start != -1 and end != -1:
                candidate = repaired[start:end+1]
                # Remove all whitespace between tokens except within strings
                # This is a simplified tokenizer approach
                tokens = []
                in_string = False
                i = 0
                while i < len(candidate):
                    char = candidate[i]
                    if char == '"' and (i == 0 or candidate[i-1] != '\\'):
                        in_string = not in_string
                        tokens.append(char)
                    elif char in ' \n\t\r' and not in_string:
                        pass  # Skip whitespace outside strings
                    else:
                        tokens.append(char)
                    i += 1
                cleaned = ''.join(tokens)
                return json.loads(cleaned)
        except (json.JSONDecodeError, ValueError) as e3:
            logger.debug(f"Third repair attempt failed: {e3}")
            pass
        
        return None


# Few-shot examples for IMO grading
FEW_SHOT_EXAMPLES = """
Example 1 - Complete Correct Solution:
Problem: Find all positive integers n such that n^2 + 3n + 2 is divisible by 4.
Solution: n^2 + 3n + 2 = (n+1)(n+2). For divisibility by 4, either n+1 or n+2 must be even, and one of them must be divisible by 4. This happens when n ≡ 0 or 3 (mod 4).
Grading Guidelines: Award 1 point for factoring, 1 point for analyzing cases, 1 point for correct answer.
Student Answer: "I factored it as (n+1)(n+2). Since these are consecutive integers, one is even. For divisibility by 4, we need one factor divisible by 4. This happens when n=3,7,11,... or n=0,4,8,... So n ≡ 0 or 3 (mod 4)."
Grade: {"score": 3, "max_score": 3, "rationale": "Complete solution with correct factoring, case analysis, and answer."}

Example 2 - Complete Correct Solution:
Problem: Prove that the sum of two odd numbers is even.
Solution: Let the odd numbers be 2k+1 and 2m+1. Their sum is 2k+1+2m+1 = 2(k+m+1), which is even.
Grading Guidelines: Award 1 point for setting up odd number representation, 1 point for correct algebra, 1 point for concluding evenness.
Student Answer: "Let the numbers be 2k+1 and 2m+1. Adding: 2k+1+2m+1 = 2k+2m+2 = 2(k+m+1), which is divisible by 2, so even."
Grade: {"score": 3, "max_score": 3, "rationale": "Correct representation, algebra, and conclusion."}

Example 3 - No Progress:
Problem: Find the remainder when 2^100 is divided by 3.
Solution: Note 2 ≡ -1 (mod 3), so 2^100 ≡ (-1)^100 ≡ 1 (mod 3).
Grading Guidelines: Award 1 point for recognizing 2 ≡ -1 (mod 3), 1 point for applying exponent, 1 point for final answer.
Student Answer: "2^1 = 2, 2^2 = 4, 2^3 = 8... I see a pattern but can't figure it out."
Grade: {"score": 0, "max_score": 3, "rationale": "No progress toward solution. Pattern observation alone insufficient."}

Example 4 - Partial Credit:
Problem: Prove that for any positive integer n, n^3 - n is divisible by 6.
Solution: Factor as n(n-1)(n+1). Among three consecutive integers, one is divisible by 3 and at least one is even, so the product is divisible by 6.
Grading Guidelines: Award 1 point for factoring, 1 point for divisibility by 2 argument, 1 point for divisibility by 3 argument.
Student Answer: "n^3 - n = n(n^2 - 1) = n(n-1)(n+1). These are three consecutive integers so one must be even."
Grade: {"score": 2, "max_score": 3, "rationale": "Correct factoring and divisibility by 2 argument, but missing divisibility by 3 argument."}

Example 5 - Incorrect Answer with Correct Work:
Problem: Solve x^2 - 5x + 6 = 0.
Solution: Factor as (x-2)(x-3) = 0, so x = 2 or x = 3.
Grading Guidelines: Award 1 point for correct factoring, 1 point for correct solutions.
Student Answer: "(x-2)(x-3) = 0, so x = 2 and x = 4" [student made arithmetic error in final answer]
Grade: {"score": 1, "max_score": 2, "rationale": "Correct factoring approach but arithmetic error in final answer (said 4 instead of 3)."}

Example 6 - Full Credit with Different Approach:
Problem: Find the sum of the first 100 positive integers.
Solution: Use formula n(n+1)/2 = 100(101)/2 = 5050.
Grading Guidelines: Award 1 point for correct approach, 1 point for correct calculation.
Student Answer: "Pair 1+100=101, 2+99=101, ..., 50+51=101. There are 50 pairs, so 50×101 = 5050."
Grade: {"score": 2, "max_score": 2, "rationale": "Alternative valid approach (Gauss's method) with correct calculation."}
"""


class TaskAgent:
    """Task agent that grades mathematical solutions with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, temperature: float = 0.0) -> None:
        self.model = model
        self.temperature = temperature
        self.log_fn = logger.info

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Grade a student answer and return (prediction, msg_history)."""
        # Step 1: Initial grading with chain-of-thought
        instruction = f"""You are an expert mathematical grader for the International Mathematical Olympiad (IMO).

Your task is to grade a student's answer to a mathematical problem with precision and consistency.

{FEW_SHOT_EXAMPLES}

GRADING PRINCIPLES:
1. Award partial credit for correct intermediate steps, even if the final answer is incorrect
2. Accept alternative valid approaches that differ from the official solution
3. Be strict about mathematical errors - deduct points for incorrect reasoning
4. Follow the grading guidelines point allocation exactly
5. Consider the student's demonstrated understanding, not just the final result

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

ANALYSIS STEPS:
1. Read the problem and official solution carefully
2. Identify the key concepts and required steps
3. Analyze the student's answer step-by-step
4. Note what the student did correctly (award points)
5. Identify any errors or gaps (deduct points)
6. Check for alternative valid approaches
7. Calculate the final score based on the guidelines

Respond in JSON format with the following schema:
<json>
{{
    "thinking": "Your detailed step-by-step analysis here. Include: (a) key concepts in the problem, (b) what the student did correctly, (c) any errors or gaps, (d) justification for the score",
    "score": <numerical score>,
    "max_score": <maximum possible score>,
    "rationale": "Clear explanation of why this exact score was awarded, referencing specific parts of the student's answer",
    "response": "<score>/<max_score> - <brief summary>"
}}
</json>

IMPORTANT: Ensure your JSON is valid and properly formatted. All string values must be in double quotes."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            temperature=self.temperature,
            msg_history=[],
        )

        # Extract prediction from JSON with detailed logging
        prediction = "None"
        try:
            last_msg = msg_history[-1]["text"]
            extracted = _extract_jsons(last_msg)
            if extracted:
                result = extracted[-1]
                self.log_fn(f"Extracted JSON result: {result}")
                
                # Validate the result has required fields
                if "response" in result:
                    prediction = result["response"]
                    self.log_fn(f"Using 'response' field: {prediction}")
                elif "score" in result and "max_score" in result:
                    # Validate score values are numeric
                    try:
                        score = float(result["score"])
                        max_score = float(result["max_score"])
                        prediction = f"{int(score)}/{int(max_score)}"
                        self.log_fn(f"Using score/max_score fields: {prediction}")
                    except (ValueError, TypeError) as e:
                        self.log_fn(f"Warning: Invalid score values: {e}")
                        prediction = f"{result['score']}/{result['max_score']}"
                else:
                    self.log_fn(f"Warning: JSON missing expected fields. Keys: {list(result.keys())}")
                    # Try to construct from available fields
                    if "score" in result:
                        score = result["score"]
                        max_score = result.get("max_score", score)
                        prediction = f"{score}/{max_score}"
                        self.log_fn(f"Constructed prediction from partial fields: {prediction}")
            else:
                self.log_fn("Warning: No JSON blocks found in response")
                # Try to extract any numeric score pattern as fallback
                score_match = re.search(r'(\d+)\s*/\s*(\d+)', last_msg)
                if score_match:
                    prediction = f"{score_match.group(1)}/{score_match.group(2)}"
                    self.log_fn(f"Fallback extraction: {prediction}")
                else:
                    # Try to find any numbers that might be scores
                    numbers = re.findall(r'\b(\d+)\b', last_msg)
                    if len(numbers) >= 2:
                        prediction = f"{numbers[0]}/{numbers[1]}"
                        self.log_fn(f"Fallback extraction from numbers: {prediction}")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            import traceback
            self.log_fn(f"Traceback: {traceback.format_exc()}")

        # Step 2: Self-reflection to verify the grade
        if prediction != "None" and len(msg_history) >= 2:
            reflection_msg = f"""Review your grading above carefully. Perform a critical self-assessment:

1. ERROR CHECK: Did you miss any mathematical errors in the student's work?
   - Check each step against the official solution
   - Verify all calculations are correct
   - Look for subtle logical gaps

2. PARTIAL CREDIT: Did you award appropriate partial credit?
   - Award points for correct intermediate steps even if final answer is wrong
   - Consider alternative valid approaches
   - Check if the student demonstrated understanding of key concepts

3. GUIDELINE ALIGNMENT: Is your score consistent with the grading guidelines?
   - Verify you're following the point allocation scheme
   - Ensure consistency with similar problems

4. CONSISTENCY: Would another expert grader agree with your assessment?
   - Consider if your rationale is clear and defensible
   - Check for any bias in your evaluation

DECISION RULES:
- If you find errors in your initial grading, provide the corrected JSON with revised_score
- If your initial grade is correct, confirm it with the same score
- Always provide a clear explanation in the reflection field

Respond in JSON format:
<json>
{{
    "reflection": "Detailed self-review addressing each point above",
    "revised_score": <numerical score>,
    "revised_max_score": <maximum possible score>,
    "final_response": "<score>/<max_score> - <brief summary of final decision>"
}}
</json>"""
            
            try:
                reflection_response, msg_history, _ = get_response_from_llm(
                    msg=reflection_msg,
                    model=self.model,
                    temperature=self.temperature,
                    msg_history=msg_history,
                )
                
                # Try to extract revised prediction with detailed logging
                try:
                    last_msg = msg_history[-1]["text"]
                    extracted = _extract_jsons(last_msg)
                    if extracted:
                        result = extracted[-1]
                        self.log_fn(f"Reflection extracted JSON: {result}")
                        if "final_response" in result:
                            prediction = result["final_response"]
                            self.log_fn(f"Using 'final_response' field: {prediction}")
                        elif "revised_score" in result and "revised_max_score" in result:
                            prediction = f"{result['revised_score']}/{result['revised_max_score']}"
                            self.log_fn(f"Using revised_score/revised_max_score: {prediction}")
                        else:
                            self.log_fn(f"Warning: Reflection JSON missing expected fields. Keys: {list(result.keys())}")
                    else:
                        self.log_fn("Warning: No JSON found in reflection response, keeping initial prediction")
                except Exception as e:
                    self.log_fn(f"Error extracting revised prediction: {e}, keeping initial prediction")
            except Exception as e:
                self.log_fn(f"Error during reflection LLM call: {e}, keeping initial prediction")

        return str(prediction), msg_history
