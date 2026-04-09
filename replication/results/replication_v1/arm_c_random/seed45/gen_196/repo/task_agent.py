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
    Includes robust error recovery for common JSON formatting issues.
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
            # Try to fix common JSON issues before giving up
            fixed_json = _attempt_json_repair(inner)
            if fixed_json:
                results.append(fixed_json)
            else:
                logger.debug(f"JSON decode error in <json> block: {e}")
            continue
    
    # Fallback 1: try markdown code blocks if no <json> blocks found
    if not results:
        markdown_pattern = r'```(?:json)?\s*\n(.*?)\n```'
        for match in re.finditer(markdown_pattern, text, re.DOTALL):
            try:
                results.append(json.loads(match.group(1).strip()))
            except json.JSONDecodeError:
                # Try repair on markdown blocks too
                fixed_json = _attempt_json_repair(match.group(1).strip())
                if fixed_json:
                    results.append(fixed_json)
                continue
    
    # Fallback 2: look for inline JSON objects (e.g., {"key": "value"})
    if not results:
        # Find JSON-like structures with balanced braces
        inline_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        for match in re.finditer(inline_pattern, text, re.DOTALL):
            try:
                candidate = match.group(0).strip()
                # Only accept if it looks like a grading result
                if '"score"' in candidate or '"response"' in candidate:
                    results.append(json.loads(candidate))
            except json.JSONDecodeError:
                fixed_json = _attempt_json_repair(match.group(0).strip())
                if fixed_json and ('score' in fixed_json or 'response' in fixed_json):
                    results.append(fixed_json)
                continue
    
    return results or None


def _attempt_json_repair(json_str: str) -> dict | None:
    """Attempt to repair common JSON formatting issues.
    
    Fixes:
    - Trailing commas before closing braces/brackets
    - Single quotes instead of double quotes
    - Unescaped newlines in strings
    - Missing quotes around keys
    - Comments in JSON (// and /* */)
    """
    import re
    
    original = json_str.strip()
    repaired = original
    
    # Fix 1: Remove single-line comments (// ...)
    repaired = re.sub(r'//[^\n]*', '', repaired)
    
    # Fix 2: Remove multi-line comments (/* ... */)
    repaired = re.sub(r'/\*.*?\*/', '', repaired, flags=re.DOTALL)
    
    # Fix 3: Remove trailing commas before } or ]
    repaired = re.sub(r',\s*\}', '}', repaired)
    repaired = re.sub(r',\s*\]', ']', repaired)
    
    # Fix 4: Replace single quotes with double quotes (carefully)
    # Only replace if not inside a string
    repaired = repaired.replace("'", '"')
    
    # Fix 5: Escape unescaped newlines in string values
    # This is a simplified fix - replace newlines between quotes
    repaired = re.sub(r'("[^"]*?)\n([^"]*?")', r'\1\\n\2', repaired)
    
    # Fix 6: Try to handle missing quotes around keys
    # Match word: followed by space or value
    repaired = re.sub(r'(\w+):\s*"', r'"\1": "', repaired)
    repaired = re.sub(r'(\w+):\s*\{', r'"\1": {', repaired)
    repaired = re.sub(r'(\w+):\s*\[', r'"\1": [', repaired)
    repaired = re.sub(r'(\w+):\s*(\d)', r'"\1": \2', repaired)
    
    # Fix 7: Handle escaped quotes that might be malformed
    repaired = re.sub(r'\\"', '"', repaired)
    
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        return None


def _validate_grading_result(result: dict) -> tuple[bool, str]:
    """Validate that a grading result has the required fields and valid values.
    
    Returns:
        (is_valid, error_message)
    """
    if not isinstance(result, dict):
        return False, f"Result must be a dict, got {type(result).__name__}"
    
    required_fields = ["score", "max_score"]
    
    for field in required_fields:
        if field not in result:
            return False, f"Missing required field: {field}"
    
    try:
        score = float(result["score"])
        max_score = float(result["max_score"])
    except (ValueError, TypeError):
        return False, "Score and max_score must be numeric"
    
    if score < 0:
        return False, f"Score cannot be negative: {score}"
    
    if max_score <= 0:
        return False, f"Max score must be positive: {max_score}"
    
    if score > max_score:
        return False, f"Score {score} exceeds max_score {max_score}"
    
    # Additional validation: check for NaN or Inf
    if score != score or max_score != max_score:  # NaN check
        return False, "Score or max_score is NaN"
    
    if score == float('inf') or score == float('-inf'):
        return False, "Score cannot be infinite"
    
    if max_score == float('inf') or max_score == float('-inf'):
        return False, "Max score cannot be infinite"
    
    return True, ""


def _normalize_score(score: float, max_score: float, target_max: float = 10.0) -> float:
    """Normalize a score to a standard scale for comparison.
    
    Args:
        score: The raw score
        max_score: The maximum possible score
        target_max: The target maximum scale (default 10.0)
    
    Returns:
        Normalized score on the target scale, clamped to [0, target_max]
    """
    if max_score <= 0 or target_max <= 0:
        return 0.0
    
    # Handle NaN and Inf
    if score != score or max_score != max_score:  # NaN check
        return 0.0
    
    if score == float('inf') or max_score == float('inf'):
        return target_max
    
    if score == float('-inf'):
        return 0.0
    
    normalized = (score / max_score) * target_max
    # Clamp to valid range
    return max(0.0, min(target_max, normalized))


def _parse_prediction(prediction: str) -> tuple[float | None, float | None]:
    """Parse a prediction string like '3/5' into (score, max_score).
    
    Args:
        prediction: The prediction string to parse
        
    Returns:
        (score, max_score) tuple, or (None, None) if parsing fails
    """
    if not prediction or prediction == "None":
        return None, None
    
    # Try to match patterns like "3/5", "3 / 5", "3.5/10", etc.
    import re
    patterns = [
        r'(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)',  # 3/5, 3.5/10
        r'(\d+(?:\.\d+)?)\s*out of\s*(\d+(?:\.\d+)?)',  # 3 out of 5
        r'score:\s*(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)',  # score: 3/5
    ]
    
    for pattern in patterns:
        match = re.search(pattern, prediction, re.IGNORECASE)
        if match:
            try:
                score = float(match.group(1))
                max_score = float(match.group(2))
                return score, max_score
            except (ValueError, IndexError):
                continue
    
    return None, None


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
Grading Guidelines: Award 1 point for setting up odd number representation, 1 point for algebraic manipulation, 1 point for conclusion.
Student Answer: "Odd numbers end in 1,3,5,7,9. Adding two odd numbers gives an even number. For example, 3+5=8 which is even."
Grade: {"score": 1, "max_score": 3, "rationale": "Student only provided examples without general proof. Missing algebraic representation and general reasoning."}
"""


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def get_normalized_score(self, prediction: str, target_max: float = 10.0) -> float:
        """Get the normalized score from a prediction string.
        
        Args:
            prediction: The prediction string (e.g., "3/5")
            target_max: The target scale for normalization
            
        Returns:
            Normalized score on the target scale, or 0.0 if parsing fails
        """
        score, max_score = _parse_prediction(prediction)
        if score is None or max_score is None:
            return 0.0
        return _normalize_score(score, max_score, target_max)

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

        # Extract prediction from JSON with detailed logging and validation
        prediction = "None"
        try:
            last_msg = msg_history[-1]["text"]
            extracted = _extract_jsons(last_msg)
            if extracted:
                result = extracted[-1]
                self.log_fn(f"Extracted JSON result: {result}")
                
                # Validate the grading result
                is_valid, error_msg = _validate_grading_result(result)
                if not is_valid:
                    self.log_fn(f"Validation failed: {error_msg}")
                    # Try to use response field if available despite validation failure
                    if "response" in result:
                        prediction = result["response"]
                        self.log_fn(f"Using 'response' field despite validation failure: {prediction}")
                elif "response" in result:
                    prediction = result["response"]
                    self.log_fn(f"Using 'response' field: {prediction}")
                elif "score" in result and "max_score" in result:
                    prediction = f"{result['score']}/{result['max_score']}"
                    self.log_fn(f"Using score/max_score fields: {prediction}")
                else:
                    self.log_fn(f"Warning: JSON missing expected fields. Keys: {list(result.keys())}")
            else:
                self.log_fn("Warning: No JSON blocks found in response")
                # Try to extract any numeric score pattern as fallback
                import re
                score_match = re.search(r'(\d+)\s*/\s*(\d+)', last_msg)
                if score_match:
                    prediction = f"{score_match.group(1)}/{score_match.group(2)}"
                    self.log_fn(f"Fallback extraction: {prediction}")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            import traceback
            self.log_fn(f"Traceback: {traceback.format_exc()}")

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
            
            try:
                reflection_response, msg_history, _ = get_response_from_llm(
                    msg=reflection_msg,
                    model=self.model,
                    msg_history=msg_history,
                )
                
                # Try to extract revised prediction with detailed logging and validation
                try:
                    last_msg = msg_history[-1]["text"]
                    extracted = _extract_jsons(last_msg)
                    if extracted:
                        result = extracted[-1]
                        self.log_fn(f"Reflection extracted JSON: {result}")
                        
                        # Validate reflection result before using it
                        # Map reflection fields to standard fields for validation
                        validation_result = result.copy()
                        if "revised_score" in result:
                            validation_result["score"] = result["revised_score"]
                        if "revised_max_score" in result:
                            validation_result["max_score"] = result["revised_max_score"]
                        
                        is_valid, error_msg = _validate_grading_result(validation_result)
                        if not is_valid:
                            self.log_fn(f"Reflection validation failed: {error_msg}, keeping original prediction")
                        elif "final_response" in result:
                            prediction = result["final_response"]
                            self.log_fn(f"Using 'final_response' field: {prediction}")
                        elif "revised_score" in result and "revised_max_score" in result:
                            prediction = f"{result['revised_score']}/{result['revised_max_score']}"
                            self.log_fn(f"Using revised_score/revised_max_score: {prediction}")
                        elif "score" in result and "max_score" in result:
                            # Fallback to standard fields if revision fields missing
                            prediction = f"{result['score']}/{result['max_score']}"
                            self.log_fn(f"Using score/max_score fields: {prediction}")
                        else:
                            self.log_fn(f"Warning: Reflection JSON missing expected fields. Keys: {list(result.keys())}")
                    else:
                        self.log_fn("Warning: No JSON found in reflection response, keeping original prediction")
                except Exception as e:
                    self.log_fn(f"Error extracting revised prediction: {e}")
                    # Keep original prediction on error
                    pass
            except Exception as e:
                self.log_fn(f"Error during reflection LLM call: {e}")
                # Keep original prediction if reflection fails
                pass

        return str(prediction), msg_history
