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
        # Look for ```json ... ``` blocks
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
        # Look for patterns that look like JSON objects
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        for match in re.finditer(json_pattern, text, re.DOTALL):
            try:
                candidate = match.group(0).strip()
                # Only accept if it has expected keys
                parsed = json.loads(candidate)
                if any(key in parsed for key in ['score', 'response', 'thinking', 'rationale']):
                    results.append(parsed)
            except (json.JSONDecodeError, ValueError):
                continue
    
    # Fallback 3: try to extract from nested structures with balanced braces
    if not results:
        results = _extract_nested_json(text)
    
    return results or None


def _extract_nested_json(text: str) -> list[dict]:
    """Extract JSON objects from text with nested braces using stack-based parsing.
    
    This handles cases where JSON objects contain nested braces that confuse
    simple regex patterns.
    """
    results = []
    i = 0
    n = len(text)
    
    while i < n:
        # Find the start of a potential JSON object
        if text[i] == '{':
            # Use stack to find matching closing brace
            stack = 1
            j = i + 1
            in_string = False
            escape_next = False
            
            while j < n and stack > 0:
                char = text[j]
                if escape_next:
                    escape_next = False
                elif char == '\\' and in_string:
                    escape_next = True
                elif char == '"' and not escape_next:
                    in_string = not in_string
                elif not in_string:
                    if char == '{':
                        stack += 1
                    elif char == '}':
                        stack -= 1
                j += 1
            
            if stack == 0:
                # Found a complete JSON object
                candidate = text[i:j].strip()
                try:
                    parsed = json.loads(candidate)
                    if any(key in parsed for key in ['score', 'response', 'thinking', 'rationale', 'reflection']):
                        results.append(parsed)
                except json.JSONDecodeError:
                    # Try repair
                    fixed = _attempt_json_repair(candidate)
                    if fixed:
                        results.append(fixed)
            i = j
        else:
            i += 1
    
    return results


def _attempt_json_repair(text: str) -> dict | None:
    """Attempt to repair common JSON formatting errors from LLM outputs.
    
    Fixes:
    - Trailing commas before closing braces/brackets
    - Single quotes instead of double quotes
    - Unquoted keys
    - Missing quotes around string values
    - Newlines in strings
    - Control characters
    - Unicode escape sequences
    """
    import re
    
    original = text.strip()
    
    # Fix 1: Remove trailing commas before } or ]
    repaired = re.sub(r',\s*([}\]])', r'\1', original)
    
    # Fix 2: Replace single quotes with double quotes (carefully)
    # Only replace single quotes that appear to be delimiters
    repaired = re.sub(r"(?<!\\)'([^']*?)'(?=\s*[:}\],])", r'"\1"', repaired)
    
    # Fix 3: Add quotes to unquoted keys
    repaired = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', repaired)
    
    # Fix 4: Fix common escape sequence issues - but preserve valid escapes
    # First, fix double-escaped quotes
    repaired = repaired.replace('\\"', '"')
    # Then re-escape them properly
    repaired = repaired.replace('"', '\\"')
    # Fix other common issues
    repaired = repaired.replace('\\n', '\n').replace('\\t', '\t')
    
    # Fix 5: Handle newlines in strings by escaping them
    repaired = repaired.replace('\n', '\\n').replace('\r', '\\r')
    
    # Fix 6: Remove control characters
    repaired = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', repaired)
    
    # Fix 7: Handle Python-style True/False/None
    repaired = re.sub(r'(?<=[\s\[\{:,])True(?=[\s\]\},])', 'true', repaired)
    repaired = re.sub(r'(?<=[\s\[\{:,])False(?=[\s\]\},])', 'false', repaired)
    repaired = re.sub(r'(?<=[\s\[\{:,])None(?=[\s\]\},])', 'null', repaired)
    
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        # Last resort: try to extract just the key-value pairs we care about
        return _extract_key_values(repaired)


def _extract_key_values(text: str) -> dict | None:
    """Extract key-value pairs from malformed JSON as a last resort.
    
    This is a fallback when all other repair attempts fail.
    """
    import re
    
    result = {}
    
    # Look for score patterns
    score_match = re.search(r'["\']?score["\']?\s*[:=]\s*(\d+)', text, re.IGNORECASE)
    if score_match:
        result['score'] = int(score_match.group(1))
    
    # Look for max_score patterns
    max_score_match = re.search(r'["\']?max_score["\']?\s*[:=]\s*(\d+)', text, re.IGNORECASE)
    if max_score_match:
        result['max_score'] = int(max_score_match.group(1))
    
    # Look for response patterns
    response_match = re.search(r'["\']?response["\']?\s*[:=]\s*["\']([^"\']+)["\']', text, re.IGNORECASE)
    if response_match:
        result['response'] = response_match.group(1)
    
    # Look for rationale/thinking patterns
    rationale_match = re.search(r'["\']?rationale["\']?\s*[:=]\s*["\']([^"\']+)["\']', text, re.IGNORECASE)
    if rationale_match:
        result['rationale'] = rationale_match.group(1)
    
    thinking_match = re.search(r'["\']?thinking["\']?\s*[:=]\s*["\']([^"\']+)["\']', text, re.IGNORECASE)
    if thinking_match:
        result['thinking'] = thinking_match.group(1)
    
    return result if result else None


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
Grading Guidelines: Award 1 point for setting up odd number representation, 1 point for correct algebra, 1 point for concluding evenness.
Student Answer: "Let the numbers be 2k+1 and 2m+1. Adding: 2k+1+2m+1 = 2k+2m+2 = 2(k+m+1), which is divisible by 2, so even."
Grade: {"score": 3, "max_score": 3, "rationale": "Correct representation, algebra, and conclusion."}

Example 3:
Problem: Find the remainder when 2^100 is divided by 3.
Solution: Note 2 ≡ -1 (mod 3), so 2^100 ≡ (-1)^100 ≡ 1 (mod 3).
Grading Guidelines: Award 1 point for recognizing 2 ≡ -1 (mod 3), 1 point for applying exponent, 1 point for final answer.
Student Answer: "2^1 = 2, 2^2 = 4, 2^3 = 8... I see a pattern but can't figure it out."
Grade: {"score": 0, "max_score": 3, "rationale": "No progress toward solution. Pattern observation alone insufficient."}
"""


class TaskAgent:
    """Task agent that grades mathematical solutions with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, temperature: float = 0.0) -> None:
        self.model = model
        self.temperature = temperature
        self.log_fn = logger.info
        self.max_retries = 2  # Number of retries for failed extractions

    def _validate_grade(self, result: dict, inputs: dict) -> tuple[bool, str]:
        """Validate that a grade result is reasonable.
        
        Returns (is_valid, reason) tuple.
        """
        # Check for required fields
        if 'score' not in result:
            return False, "Missing 'score' field"
        
        score = result.get('score')
        max_score = result.get('max_score')
        
        # Validate score is numeric
        if not isinstance(score, (int, float)):
            return False, f"Score is not numeric: {score}"
        
        # Validate score is non-negative
        if score < 0:
            return False, f"Score is negative: {score}"
        
        # Validate against max_score if available
        if max_score is not None:
            if not isinstance(max_score, (int, float)):
                return False, f"Max score is not numeric: {max_score}"
            if score > max_score:
                return False, f"Score {score} exceeds max_score {max_score}"
        
        # Check for empty rationale/thinking
        rationale = result.get('rationale', '')
        thinking = result.get('thinking', '')
        if not rationale and not thinking:
            return False, "Missing both rationale and thinking"
        
        return True, "Valid"

    def _extract_prediction(self, msg_history: list[dict], inputs: dict, stage: str = "initial") -> tuple[str, dict | None]:
        """Extract prediction from message history with validation.
        
        Returns (prediction, result_dict) tuple.
        """
        prediction = "None"
        result = None
        
        try:
            last_msg = msg_history[-1]["text"]
            extracted = _extract_jsons(last_msg)
            
            if extracted:
                result = extracted[-1]
                self.log_fn(f"[{stage}] Extracted JSON result: {result}")
                
                # Validate the grade
                is_valid, reason = self._validate_grade(result, inputs)
                if not is_valid:
                    self.log_fn(f"[{stage}] Invalid grade: {reason}")
                    # Still try to use it if we can extract basic info
                
                if "response" in result:
                    prediction = result["response"]
                    self.log_fn(f"[{stage}] Using 'response' field: {prediction}")
                elif "score" in result and "max_score" in result:
                    prediction = f"{result['score']}/{result['max_score']}"
                    self.log_fn(f"[{stage}] Using score/max_score fields: {prediction}")
                elif "score" in result:
                    # Try to infer max_score from grading guidelines
                    max_score = self._infer_max_score(inputs)
                    prediction = f"{result['score']}/{max_score}"
                    self.log_fn(f"[{stage}] Using score with inferred max_score: {prediction}")
                else:
                    self.log_fn(f"[{stage}] Warning: JSON missing expected fields. Keys: {list(result.keys())}")
            else:
                self.log_fn(f"[{stage}] Warning: No JSON blocks found in response")
                # Try to extract any numeric score pattern as fallback
                score_match = re.search(r'(\d+)\s*/\s*(\d+)', last_msg)
                if score_match:
                    prediction = f"{score_match.group(1)}/{score_match.group(2)}"
                    self.log_fn(f"[{stage}] Fallback extraction: {prediction}")
        except Exception as e:
            self.log_fn(f"[{stage}] Error extracting prediction: {e}")
            import traceback
            self.log_fn(f"[{stage}] Traceback: {traceback.format_exc()}")
        
        return prediction, result

    def _infer_max_score(self, inputs: dict) -> int:
        """Infer max_score from grading guidelines if not explicitly provided."""
        guidelines = inputs.get('grading_guidelines', '')
        
        # Try to find point values in guidelines
        import re
        points = re.findall(r'(\d+)\s*point', guidelines, re.IGNORECASE)
        if points:
            return sum(int(p) for p in points)
        
        # Default to 3 if we can't determine
        return 3

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Grade a student answer and return (prediction, msg_history)."""
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
            temperature=self.temperature,
            msg_history=[],
        )

        # Extract prediction from JSON with validation
        prediction, initial_result = self._extract_prediction(msg_history, inputs, "initial")

        # Step 2: Self-reflection to verify the grade
        if prediction != "None" and len(msg_history) >= 2:
            reflection_msg = f"""Review your grading above. Check for:
1. Did you award points the student didn't earn?
2. Did you miss any errors in the student's work?
3. Is your score consistent with the grading guidelines?
4. Would another grader agree with your assessment?

If you need to revise your grade, provide the corrected JSON. If your grade is correct, confirm it.

Respond in JSON format:
<json>
{{
    "reflection": "Your self-review here",
    "revised_score": <score>,
    "revised_max_score": <max_score>,
    "final_response": "<score>/<max_score> - <brief summary>"
}}
</json>"""
            
            reflection_response, msg_history, _ = get_response_from_llm(
                msg=reflection_msg,
                model=self.model,
                temperature=self.temperature,
                msg_history=msg_history,
            )
            
            # Extract revised prediction with validation
            revised_prediction, revised_result = self._extract_prediction(msg_history, inputs, "reflection")
            
            # Only use revised prediction if it's valid
            if revised_prediction != "None" and revised_result:
                is_valid, reason = self._validate_grade(revised_result, inputs)
                if is_valid:
                    prediction = revised_prediction
                    self.log_fn(f"Using valid revised prediction: {prediction}")
                else:
                    self.log_fn(f"Revised prediction invalid ({reason}), keeping original: {prediction}")
            elif revised_prediction != "None":
                prediction = revised_prediction
                self.log_fn(f"Using revised prediction without validation: {prediction}")

        return str(prediction), msg_history
