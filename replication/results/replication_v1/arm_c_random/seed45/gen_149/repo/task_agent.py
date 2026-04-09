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
import time
from typing import Any

from agent.llm_client import get_response_from_llm, EVAL_MODEL

logger = logging.getLogger(__name__)


def log_execution_time(func):
    """Decorator to log function execution time for performance monitoring."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        logger.debug(f"{func.__name__} executed in {elapsed:.3f}s")
        return result
    return wrapper


@log_execution_time
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
    
    return results or None


@log_execution_time
def _attempt_json_repair(text: str) -> dict | None:
    """Attempt to repair common JSON formatting errors from LLM outputs.
    
    Fixes:
    - Trailing commas before closing braces/brackets
    - Single quotes instead of double quotes
    - Unquoted keys
    - Missing quotes around string values
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
    
    # Fix 4: Fix common escape sequence issues
    repaired = repaired.replace('\\"', '"').replace('\\n', '\n').replace('\\t', '\t')
    
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
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
Grading Guidelines: Award 1 point for setting up odd number representation, 1 point for algebraic manipulation, 1 point for conclusion.
Student Answer: "Odd numbers end in 1,3,5,7,9. Adding two odd numbers gives an even number. For example, 3+5=8 which is even."
Grade: {"score": 1, "max_score": 3, "rationale": "Student only provided examples without general proof. Missing algebraic representation and general reasoning."}
"""


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    @log_execution_time
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

        # Extract prediction from JSON with detailed logging
        prediction = "None"
        try:
            last_msg = msg_history[-1]["text"]
            extracted = _extract_jsons(last_msg)
            if extracted:
                result = extracted[-1]
                self.log_fn(f"Extracted JSON result: {result}")
                if "response" in result:
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
5. Is the score within valid bounds (0 ≤ score ≤ max_score)?

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
                msg_history=msg_history,
            )
            
            # Try to extract revised prediction with detailed logging
            try:
                last_msg = msg_history[-1]["text"]
                extracted = _extract_jsons(last_msg)
                if extracted:
                    result = extracted[-1]
                    self.log_fn(f"Reflection extracted JSON: {result}")
                    
                    # Validate score bounds before accepting
                    revised_score = None
                    revised_max = None
                    
                    if "revised_score" in result:
                        revised_score = result["revised_score"]
                    if "revised_max_score" in result:
                        revised_max = result["revised_max_score"]
                    
                    # Validate that scores are numeric and within bounds
                    if revised_score is not None and revised_max is not None:
                        try:
                            score_val = float(revised_score)
                            max_val = float(revised_max)
                            if 0 <= score_val <= max_val:
                                if "final_response" in result:
                                    prediction = result["final_response"]
                                    self.log_fn(f"Using 'final_response' field: {prediction}")
                                else:
                                    prediction = f"{int(score_val)}/{int(max_val)}"
                                    self.log_fn(f"Using validated revised_score/revised_max_score: {prediction}")
                            else:
                                self.log_fn(f"Warning: Score {score_val}/{max_val} out of bounds, keeping original prediction")
                        except (ValueError, TypeError):
                            self.log_fn(f"Warning: Non-numeric score values, keeping original prediction")
                    elif "final_response" in result:
                        prediction = result["final_response"]
                        self.log_fn(f"Using 'final_response' field: {prediction}")
                    else:
                        self.log_fn(f"Warning: Reflection JSON missing expected fields. Keys: {list(result.keys())}")
                else:
                    self.log_fn("Warning: No JSON found in reflection response")
            except Exception as e:
                self.log_fn(f"Error extracting revised prediction: {e}")
                import traceback
                self.log_fn(f"Traceback: {traceback.format_exc()}")

        # Final validation: ensure prediction is in expected format
        prediction = self._validate_prediction(prediction)
        
        return str(prediction), msg_history

    @log_execution_time
    def _validate_prediction(self, prediction: str) -> str:
        """Validate and normalize the prediction format.
        
        Ensures the prediction is in the format 'score/max_score'.
        Returns the original prediction if validation fails.
        """
        if prediction == "None" or not prediction:
            return "None"
        
        # Try to extract score/max_score pattern
        match = re.search(r'(\d+)\s*/\s*(\d+)', str(prediction))
        if match:
            score, max_score = int(match.group(1)), int(match.group(2))
            # Ensure score is within bounds
            if score < 0:
                score = 0
            if max_score > 0 and score > max_score:
                score = max_score
            return f"{score}/{max_score}"
        
        return prediction
