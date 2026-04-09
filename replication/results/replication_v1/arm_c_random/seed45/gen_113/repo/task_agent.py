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
    Also handles markdown code blocks as a fallback.
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
            fixed = _attempt_json_fix(inner)
            if fixed is not None:
                results.append(fixed)
            else:
                logger.debug(f"JSON decode error in <json> block: {e}")
            continue
    
    # Fallback: try markdown code blocks if no <json> blocks found
    if not results:
        # Look for ```json ... ``` blocks
        markdown_pattern = r'```(?:json)?\s*\n(.*?)\n```'
        for match in re.finditer(markdown_pattern, text, re.DOTALL):
            try:
                results.append(json.loads(match.group(1).strip()))
            except json.JSONDecodeError:
                # Try to fix common JSON issues
                fixed = _attempt_json_fix(match.group(1).strip())
                if fixed is not None:
                    results.append(fixed)
                continue
    
    # Second fallback: try to find JSON-like objects with braces
    if not results:
        # Look for content between outermost braces
        brace_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        for match in re.finditer(brace_pattern, text, re.DOTALL):
            try:
                results.append(json.loads(match.group(0)))
            except json.JSONDecodeError:
                fixed = _attempt_json_fix(match.group(0))
                if fixed is not None:
                    results.append(fixed)
                continue
    
    return results or None


def _attempt_json_fix(text: str) -> dict | None:
    """Attempt to fix common JSON formatting issues.
    
    Fixes:
    - Trailing commas before closing braces/brackets
    - Single quotes instead of double quotes
    - Unquoted keys
    - Missing quotes around string values
    """
    import re
    
    original = text.strip()
    fixed = original
    
    # Remove trailing commas before } or ]
    fixed = re.sub(r',(\s*[}\]])', r'\1', fixed)
    
    # Try to parse after basic fixes
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    
    # Replace single quotes with double quotes (carefully)
    # Only replace quotes that are not inside strings
    try:
        # Simple approach: replace ' with " and hope for the best
        # This is a heuristic and may not work for all cases
        fixed_quotes = re.sub(r"(?<!\\)'", '"', fixed)
        return json.loads(fixed_quotes)
    except json.JSONDecodeError:
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

Example 2 - Partial Credit with Missing General Proof:
Problem: Prove that the sum of two odd numbers is even.
Solution: Let the odd numbers be 2k+1 and 2m+1. Their sum is 2k+1+2m+1 = 2(k+m+1), which is even.
Grading Guidelines: Award 1 point for setting up odd number representation, 1 point for algebraic manipulation, 1 point for conclusion.
Student Answer: "Odd numbers end in 1,3,5,7,9. Adding two odd numbers gives an even number. For example, 3+5=8 which is even."
Grade: {"score": 1, "max_score": 3, "rationale": "Student only provided examples without general proof. Missing algebraic representation and general reasoning."}

Example 3 - Incorrect Answer with Some Correct Work:
Problem: Solve x^2 - 5x + 6 = 0.
Solution: Factor as (x-2)(x-3) = 0, so x = 2 or x = 3.
Grading Guidelines: Award 1 point for correct factoring attempt, 1 point for correct roots.
Student Answer: "x^2 - 5x + 6 = (x-1)(x-6) = 0, so x = 1 or x = 6."
Grade: {"score": 0, "max_score": 2, "rationale": "Student attempted to factor but made an error. (x-1)(x-6) = x^2 - 7x + 6, not x^2 - 5x + 6. No correct work shown."}

Example 4 - Full Credit with Alternative Valid Approach:
Problem: Prove that for any positive integer n, n^3 - n is divisible by 6.
Solution: n^3 - n = n(n-1)(n+1), product of three consecutive integers. Among any three consecutive integers, one is divisible by 2 and one by 3, so product divisible by 6.
Grading Guidelines: Award 1 point for algebraic manipulation, 1 point for recognizing consecutive integers property, 1 point for divisibility argument.
Student Answer: "n^3 - n = n(n^2-1) = n(n-1)(n+1). These are three consecutive integers. In any three consecutive integers, there's always a multiple of 2 and a multiple of 3. Therefore the product is divisible by 6."
Grade: {"score": 3, "max_score": 3, "rationale": "Complete and correct proof using factorization and properties of consecutive integers."}

Example 5 - Zero Credit for Completely Wrong Approach:
Problem: Find the area of a circle with radius 5.
Solution: Area = πr^2 = 25π.
Grading Guidelines: Award 1 point for correct formula, 1 point for correct computation.
Student Answer: "Area = 2πr = 10π."
Grade: {"score": 0, "max_score": 2, "rationale": "Student used circumference formula instead of area formula. No correct work shown."}
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

Your task is to grade a student's answer to a mathematical problem with precision and consistency.

GRADING PRINCIPLES:
1. Award points ONLY for correct mathematical reasoning that aligns with the official solution
2. Deduct points for errors, even if the final answer is correct
3. Partial credit is appropriate when some but not all criteria are met
4. Be consistent with the grading guidelines - they define the scoring rubric
5. Consider alternative valid approaches that differ from the official solution but are mathematically sound

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
1. Carefully read the official solution to understand the expected approach
2. Analyze the student's answer line by line
3. Identify what the student did correctly (mathematically valid steps)
4. Identify errors, gaps, or missing steps
5. Compare against the grading guidelines - these define how points are awarded
6. Determine the score based on criteria met, not just the final answer
7. Provide detailed rationale explaining why each point was or wasn't awarded

IMPORTANT: You MUST respond in valid JSON format wrapped in <json>...</json> tags:
<json>
{{
    "thinking": "Your detailed step-by-step analysis here. Be specific about what was correct and what was wrong.",
    "score": <numerical score as integer>,
    "max_score": <maximum possible score as integer>,
    "rationale": "Detailed explanation of why this score was awarded. Reference specific grading criteria.",
    "response": "<score>/<max_score> - <brief summary of key points>"
}}
</json>

Ensure your JSON is valid - no trailing commas, proper quotes, and correct data types."""

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
            reflection_msg = f"""Review your grading above carefully. Perform a critical self-assessment:

1. ACCURACY CHECK: Did you award points for work that contains mathematical errors?
2. COMPLETENESS CHECK: Did you miss any errors or gaps in the student's reasoning?
3. CONSISTENCY CHECK: Is your score consistent with the grading guidelines provided?
4. FAIRNESS CHECK: Would another expert grader award the same score?
5. ALTERNATIVE CHECK: Did the student use a valid alternative approach different from the official solution?

If you identify issues with your initial grading, provide a corrected assessment.
If your initial grading is correct, confirm it with the same score.

IMPORTANT: You MUST respond in valid JSON format wrapped in <json>...</json> tags:
<json>
{{
    "reflection": "Your critical self-review. Be honest about any potential errors in your initial assessment.",
    "revised_score": <score as integer>,
    "revised_max_score": <max_score as integer>,
    "final_response": "<score>/<max_score> - <brief summary>"
}}
</json>

Ensure your JSON is valid - no trailing commas, proper quotes, and correct data types."""
            
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
                    if "final_response" in result:
                        prediction = result["final_response"]
                        self.log_fn(f"Using 'final_response' field: {prediction}")
                    elif "revised_score" in result and "revised_max_score" in result:
                        prediction = f"{result['revised_score']}/{result['revised_max_score']}"
                        self.log_fn(f"Using revised_score/revised_max_score: {prediction}")
                    else:
                        self.log_fn(f"Warning: Reflection JSON missing expected fields. Keys: {list(result.keys())}")
                else:
                    self.log_fn("Warning: No JSON found in reflection response")
            except Exception as e:
                self.log_fn(f"Error extracting revised prediction: {e}")
                import traceback
                self.log_fn(f"Traceback: {traceback.format_exc()}")

        return str(prediction), msg_history
