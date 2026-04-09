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
    if not text or not isinstance(text, str):
        return None
        
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
        # Look for patterns that look like JSON objects with nested structures
        # This pattern handles nested braces more carefully
        def find_json_objects(s: str) -> list[str]:
            """Find all JSON-like objects in a string."""
            objects = []
            i = 0
            while i < len(s):
                if s[i] == '{':
                    # Found start of potential JSON object
                    brace_count = 0
                    start = i
                    in_string = False
                    escape_next = False
                    
                    while i < len(s):
                        char = s[i]
                        if escape_next:
                            escape_next = False
                        elif char == '\\' and in_string:
                            escape_next = True
                        elif char == '"' and not escape_next:
                            in_string = not in_string
                        elif not in_string:
                            if char == '{':
                                brace_count += 1
                            elif char == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    # Found complete object
                                    objects.append(s[start:i+1])
                                    break
                        i += 1
                i += 1
            return objects
        
        candidates = find_json_objects(text)
        for candidate in candidates:
            try:
                # Only accept if it has expected keys
                parsed = json.loads(candidate)
                if isinstance(parsed, dict) and any(key in parsed for key in ['score', 'response', 'thinking', 'rationale', 'revised_score']):
                    results.append(parsed)
            except (json.JSONDecodeError, ValueError):
                # Try repair
                fixed = _attempt_json_repair(candidate)
                if fixed and isinstance(fixed, dict) and any(key in fixed for key in ['score', 'response', 'thinking', 'rationale', 'revised_score']):
                    results.append(fixed)
                continue
    
    return results or None


def _attempt_json_repair(text: str) -> dict | None:
    """Attempt to repair common JSON formatting errors from LLM outputs.
    
    Fixes:
    - Trailing commas before closing braces/brackets
    - Single quotes instead of double quotes
    - Unquoted keys
    - Missing quotes around string values
    - Newlines in string values
    - Comments in JSON (// and /* */)
    - Unicode escape sequences
    - Control characters
    - Unescaped backslashes
    - Missing closing braces/brackets
    """
    import re
    
    original = text.strip()
    
    # Fix 1: Remove C-style comments (// and /* */)
    repaired = re.sub(r'//[^\n]*', '', original)
    repaired = re.sub(r'/\*.*?\*/', '', repaired, flags=re.DOTALL)
    
    # Fix 2: Remove trailing commas before } or ]
    repaired = re.sub(r',\s*([}\]])', r'\1', repaired)
    
    # Fix 3: Replace single quotes with double quotes (carefully)
    # Only replace single quotes that appear to be delimiters
    repaired = re.sub(r"(?<!\\)'([^']*?)'(?=\s*[:}\],])", r'"\1"', repaired)
    
    # Fix 4: Add quotes to unquoted keys
    repaired = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', repaired)
    
    # Fix 5: Fix common escape sequence issues
    repaired = repaired.replace('\\"', '"').replace('\\n', '\n').replace('\\t', '\t')
    
    # Fix 6: Handle newlines in string values by escaping them
    # This is a more aggressive fix for multiline strings
    def escape_newlines_in_strings(match):
        content = match.group(1)
        # Escape unescaped newlines
        content = re.sub(r'(?<!\\)\n', r'\\n', content)
        content = re.sub(r'(?<!\\)\t', r'\\t', content)
        return '"' + content + '"'
    
    repaired = re.sub(r'"([^"]*(?:\n[^"]*)*)"', escape_newlines_in_strings, repaired)
    
    # Fix 7: Remove control characters except for valid whitespace
    repaired = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', repaired)
    
    # Fix 8: Fix unescaped backslashes (not followed by valid escape char)
    # First, protect valid escape sequences
    valid_escapes = ['\\n', '\\t', '\\r', '\\b', '\\f', '\\"', '\\\\', '\\/']
    placeholder = '\x00ESC\x00'
    for i, esc in enumerate(valid_escapes):
        repaired = repaired.replace(esc, f'{placeholder}{i}{placeholder}')
    
    # Now escape remaining backslashes
    repaired = re.sub(r'\\(?!n|t|r|b|f|"|\\|/|u[0-9a-fA-F]{4})', r'\\\\', repaired)
    
    # Restore valid escape sequences
    for i, esc in enumerate(valid_escapes):
        repaired = repaired.replace(f'{placeholder}{i}{placeholder}', esc)
    
    # Fix 9: Balance braces and brackets
    open_braces = repaired.count('{') - repaired.count('}')
    open_brackets = repaired.count('[') - repaired.count(']')
    
    if open_braces > 0:
        repaired += '}' * open_braces
    if open_brackets > 0:
        repaired += ']' * open_brackets
    
    # Fix 10: Handle incomplete JSON (missing closing brace at end)
    if repaired and not repaired.endswith('}'):
        # Try to find the last complete object
        last_brace = repaired.rfind('}')
        if last_brace > 0:
            try:
                candidate = repaired[:last_brace + 1]
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass
    
    try:
        return json.loads(repaired)
    except json.JSONDecodeError as e:
        logger.debug(f"JSON repair failed: {e}")
        
        # Last resort: try to extract just the first valid JSON object
        try:
            # Find the first { and try to parse from there
            start = repaired.find('{')
            if start != -1:
                for end in range(len(repaired), start, -1):
                    try:
                        candidate = repaired[start:end]
                        # Balance braces in candidate
                        open_count = candidate.count('{')
                        close_count = candidate.count('}')
                        if open_count > close_count:
                            candidate += '}' * (open_count - close_count)
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        continue
        except Exception:
            pass
        
        return None


def validate_grading_result(result: dict) -> tuple[bool, str]:
    """Validate that a grading result has all required fields.
    
    Returns:
        (is_valid, error_message)
    """
    required_fields = ['score', 'max_score', 'rationale']
    missing = [f for f in required_fields if f not in result]
    
    if missing:
        return False, f"Missing required fields: {missing}"
    
    # Validate score is numeric
    try:
        score = float(result['score'])
        max_score = float(result['max_score'])
        if score < 0 or max_score <= 0:
            return False, "Invalid score values (score must be >= 0, max_score > 0)"
        if score > max_score:
            return False, f"Score {score} exceeds max_score {max_score}"
    except (ValueError, TypeError):
        return False, "Score and max_score must be numeric"
    
    # Validate rationale is non-empty string
    if not isinstance(result['rationale'], str) or not result['rationale'].strip():
        return False, "Rationale must be a non-empty string"
    
    return True, ""


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
Grading Guidelines: Award 1 point for setting up odd number representation, 1 point for correct algebra, 1 point for conclusion.
Student Answer: "Let two odd numbers be 2a+1 and 2b+1. Adding: 2a+1+2b+1 = 2a+2b+2 = 2(a+b+1). This is clearly divisible by 2, so it's even."
Grade: {"score": 3, "max_score": 3, "rationale": "Correct representation, algebra, and conclusion."}
"""


class TaskAgent:
    """Task agent for grading mathematical problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, temperature: float = 0.0) -> None:
        self.model = model
        self.temperature = temperature
        self.log_fn = logger.info
        self.max_retries = 3

    def _extract_prediction(self, msg_history: list[dict], stage: str = "initial") -> str:
        """Extract prediction from message history with robust fallback strategies.
        
        Args:
            msg_history: The message history from LLM calls
            stage: Description of the extraction stage for logging
            
        Returns:
            Extracted prediction string or "None" if extraction fails
        """
        try:
            if not msg_history:
                self.log_fn(f"[{stage}] Empty message history")
                return "None"
                
            last_msg = msg_history[-1].get("text", "")
            if not last_msg:
                self.log_fn(f"[{stage}] Empty last message")
                return "None"
            
            # Strategy 1: Extract from JSON blocks
            extracted = _extract_jsons(last_msg)
            if extracted:
                result = extracted[-1]
                self.log_fn(f"[{stage}] Extracted JSON result: {result}")
                
                # Try different field combinations
                if "response" in result:
                    prediction = result["response"]
                    self.log_fn(f"[{stage}] Using 'response' field: {prediction}")
                    return prediction
                elif "final_response" in result:
                    prediction = result["final_response"]
                    self.log_fn(f"[{stage}] Using 'final_response' field: {prediction}")
                    return prediction
                elif "score" in result and "max_score" in result:
                    prediction = f"{result['score']}/{result['max_score']}"
                    self.log_fn(f"[{stage}] Using score/max_score fields: {prediction}")
                    return prediction
                elif "revised_score" in result and "revised_max_score" in result:
                    prediction = f"{result['revised_score']}/{result['revised_max_score']}"
                    self.log_fn(f"[{stage}] Using revised_score/revised_max_score: {prediction}")
                    return prediction
                else:
                    self.log_fn(f"[{stage}] Warning: JSON missing expected fields. Keys: {list(result.keys())}")
            
            # Strategy 2: Regex pattern matching for score/max_score format
            score_match = re.search(r'(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)', last_msg)
            if score_match:
                prediction = f"{score_match.group(1)}/{score_match.group(2)}"
                self.log_fn(f"[{stage}] Fallback regex extraction: {prediction}")
                return prediction
            
            # Strategy 3: Look for standalone numbers that might be scores
            numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', last_msg)
            if len(numbers) >= 2:
                # Assume first is score, second is max_score
                prediction = f"{numbers[0]}/{numbers[1]}"
                self.log_fn(f"[{stage}] Number pair extraction: {prediction}")
                return prediction
            
            self.log_fn(f"[{stage}] All extraction strategies failed")
            return "None"
            
        except Exception as e:
            self.log_fn(f"[{stage}] Error extracting prediction: {e}")
            import traceback
            self.log_fn(f"[{stage}] Traceback: {traceback.format_exc()}")
            return "None"

    def _grade_with_retry(self, inputs: dict) -> tuple[str, list[dict]]:
        """Perform initial grading with retry logic for robustness.
        
        Args:
            inputs: Dictionary containing problem data
            
        Returns:
            Tuple of (prediction string, message history)
        """
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

        for attempt in range(self.max_retries):
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    temperature=self.temperature,
                    msg_history=[],
                )
                
                prediction = self._extract_prediction(msg_history, stage=f"initial_attempt_{attempt+1}")
                
                if prediction != "None":
                    return prediction, msg_history
                
                self.log_fn(f"[grading] Attempt {attempt + 1} failed to extract valid prediction, retrying...")
                
            except Exception as e:
                self.log_fn(f"[grading] Attempt {attempt + 1} error: {e}")
                if attempt == self.max_retries - 1:
                    raise
        
        return "None", []

    def _reflect_on_grade(self, msg_history: list[dict], current_prediction: str) -> tuple[str, list[dict]]:
        """Perform self-reflection on the grading with retry logic.
        
        Args:
            msg_history: Previous message history
            current_prediction: Current grade prediction
            
        Returns:
            Tuple of (updated prediction, updated message history)
        """
        if not msg_history or len(msg_history) < 2:
            return current_prediction, msg_history
        
        reflection_msg = f"""Review your grading above. Check for:
1. Did you award points the student didn't earn?
2. Did you miss any errors in the student's work?
3. Is your score consistent with the grading guidelines?
4. Would another grader agree with your assessment?

Current grade: {current_prediction}

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
        
        for attempt in range(self.max_retries):
            try:
                reflection_response, updated_history, _ = get_response_from_llm(
                    msg=reflection_msg,
                    model=self.model,
                    temperature=self.temperature,
                    msg_history=msg_history,
                )
                
                new_prediction = self._extract_prediction(updated_history, stage=f"reflection_attempt_{attempt+1}")
                
                if new_prediction != "None":
                    return new_prediction, updated_history
                
                self.log_fn(f"[reflection] Attempt {attempt + 1} failed to extract valid prediction, retrying...")
                
            except Exception as e:
                self.log_fn(f"[reflection] Attempt {attempt + 1} error: {e}")
                if attempt == self.max_retries - 1:
                    break
        
        # Return original prediction if reflection fails
        return current_prediction, msg_history

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Grade a student's answer to a mathematical problem with retry logic.

        Args:
            inputs: Dictionary containing 'problem', 'solution', 'grading_guidelines',
                   'student_answer', and optionally 'domain'

        Returns:
            Tuple of (prediction string, message history)
        """
        self.log_fn(f"[forward] Starting grading for problem: {inputs.get('problem', '')[:50]}...")
        
        # Step 1: Initial grading with retry logic
        prediction, msg_history = self._grade_with_retry(inputs)
        
        if prediction == "None":
            self.log_fn("[forward] Failed to get valid prediction after all retries")
            return "None", msg_history
        
        self.log_fn(f"[forward] Initial prediction: {prediction}")
        
        # Step 2: Self-reflection to verify the grade
        prediction, msg_history = self._reflect_on_grade(msg_history, prediction)
        
        self.log_fn(f"[forward] Final prediction: {prediction}")
        
        return str(prediction), msg_history
