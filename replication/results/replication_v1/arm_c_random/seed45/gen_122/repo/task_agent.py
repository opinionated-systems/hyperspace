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
            logger.debug(f"JSON decode error in <json> block: {e}")
            # Try to fix common JSON issues
            try:
                # Remove trailing commas before closing braces/brackets
                fixed = re.sub(r',(\s*[}\]])', r'\1', inner)
                # Fix single quotes to double quotes (common LLM error)
                fixed = re.sub(r"(?<!\\)'", '"', fixed)
                # Fix unescaped newlines in strings
                fixed = re.sub(r'(?<=")\n(?=\s*")', '\\n', fixed)
                results.append(json.loads(fixed))
                logger.debug("JSON fixed and parsed successfully")
            except json.JSONDecodeError:
                continue
    
    # Fallback: try markdown code blocks if no <json> blocks found
    if not results:
        # Look for ```json ... ``` blocks
        markdown_pattern = r'```(?:json)?\s*\n(.*?)\n```'
        for match in re.finditer(markdown_pattern, text, re.DOTALL):
            try:
                results.append(json.loads(match.group(1).strip()))
            except json.JSONDecodeError:
                # Try same fixes as above
                try:
                    inner = match.group(1).strip()
                    fixed = re.sub(r',(\s*[}\]])', r'\1', inner)
                    fixed = re.sub(r"(?<!\\)'", '"', fixed)
                    fixed = re.sub(r'(?<=")\n(?=\s*")', '\\n', fixed)
                    results.append(json.loads(fixed))
                except json.JSONDecodeError:
                    continue
    
    # Last resort: try to find any JSON-like structure in the text
    if not results:
        # Look for patterns like {"key": value} or [{...}]
        json_pattern = r'\{[^{}]*"[^"]+"[^{}]*\}'
        for match in re.finditer(json_pattern, text, re.DOTALL):
            try:
                results.append(json.loads(match.group(0)))
            except json.JSONDecodeError:
                continue
    
    return results or None


# Few-shot examples for IMO grading
FEW_SHOT_EXAMPLES = """
Example 1 - Complete Correct Solution:
Problem: Find all positive integers n such that n^2 + 3n + 2 is divisible by 4.
Solution: n^2 + 3n + 2 = (n+1)(n+2). For divisibility by 4, either n+1 or n+2 must be even, and one of them must be divisible by 4. This happens when n ≡ 0 or 3 (mod 4).
Grading Guidelines: Award 1 point for factoring, 1 point for analyzing cases, 1 point for correct answer.
Student Answer: "I factored it as (n+1)(n+2). Since these are consecutive integers, one is even. For divisibility by 4, we need one factor divisible by 4. This happens when n=3,7,11,... or n=0,4,8,... So n ≡ 0 or 3 (mod 4)."
Grade: {"score": 3, "max_score": 3, "rationale": "Complete solution with correct factoring, case analysis, and answer."}

Example 2 - Minimal Credit for Examples Only:
Problem: Prove that the sum of two odd numbers is even.
Solution: Let the odd numbers be 2k+1 and 2m+1. Their sum is 2k+1+2m+1 = 2(k+m+1), which is even.
Grading Guidelines: Award 1 point for setting up odd number representation, 1 point for algebraic manipulation, 1 point for conclusion.
Student Answer: "Odd numbers end in 1,3,5,7,9. Adding two odd numbers gives an even number. For example, 3+5=8 which is even."
Grade: {"score": 0, "max_score": 3, "rationale": "Student only provided examples without general proof. Examples alone do not constitute a mathematical proof. No credit awarded."}

Example 3 - Partial Credit for Correct Approach with Minor Error:
Problem: Solve x^2 - 5x + 6 = 0.
Solution: Factor as (x-2)(x-3) = 0, so x = 2 or x = 3.
Grading Guidelines: Award 2 points for correct factoring, 1 point for finding both roots.
Student Answer: "(x-2)(x-3) = 0, so x = 2 and x = -3"
Grade: {"score": 2, "max_score": 3, "rationale": "Correct factoring (2 points) but arithmetic error in second root - should be 3 not -3. Deducted 1 point for the error."}

Example 4 - Zero Credit for Completely Wrong Approach:
Problem: Prove √2 is irrational.
Solution: Assume √2 = p/q in lowest terms. Then 2q^2 = p^2, so p^2 is even, thus p is even. Let p=2k, then 2q^2 = 4k^2, so q^2 = 2k^2, making q even. Contradiction since p/q not in lowest terms.
Grading Guidelines: Award 1 point for setting up proof by contradiction, 1 point for showing p is even, 1 point for showing q is even, 1 point for concluding contradiction.
Student Answer: "√2 = 1.41421356... which is not a whole number, so it's irrational."
Grade: {"score": 0, "max_score": 4, "rationale": "Student only stated the decimal approximation without any proof. No mathematical reasoning provided."}

Example 5 - Full Credit with Different Valid Approach:
Problem: Find the sum of first n positive integers.
Solution: Use formula n(n+1)/2 or pairing argument.
Grading Guidelines: Award 2 points for correct formula/approach, 1 point for correct derivation/answer.
Student Answer: "Pair 1 with n, 2 with n-1, etc. Each pair sums to n+1. There are n/2 pairs. So sum = n(n+1)/2."
Grade: {"score": 3, "max_score": 3, "rationale": "Excellent alternative solution using pairing method. Correct reasoning and final answer."}

Example 6 - Detecting Subtle Logical Gaps:
Problem: Prove that for any prime p > 3, p^2 ≡ 1 (mod 24).
Solution: Any prime p > 3 is of form 6k±1. Squaring: (6k±1)^2 = 36k^2 ± 12k + 1 = 12k(3k±1) + 1. One of k or (3k±1) is even, so 24 | 12k(3k±1), thus p^2 ≡ 1 (mod 24).
Grading Guidelines: Award 1 point for recognizing p = 6k±1 form, 1 point for correct squaring, 1 point for divisibility argument, 1 point for conclusion.
Student Answer: "Primes greater than 3 are of form 6k±1. Then p^2 = 36k^2 ± 12k + 1. This is 1 mod 24."
Grade: {"score": 2, "max_score": 4, "rationale": "Correct form identification (1 pt) and squaring (1 pt), but missing the crucial divisibility argument showing why 12k(3k±1) is divisible by 24. The student jumped to conclusion without justification."}

Example 7 - Handling Incomplete Solutions:
Problem: Prove the AM-GM inequality for two positive numbers: (a+b)/2 ≥ √(ab).
Solution: Start with (√a - √b)^2 ≥ 0. Expand: a - 2√(ab) + b ≥ 0. Rearrange: a + b ≥ 2√(ab). Divide by 2: (a+b)/2 ≥ √(ab).
Grading Guidelines: Award 1 point for correct starting point, 1 point for expansion, 1 point for rearrangement, 1 point for final inequality.
Student Answer: "We know that (√a - √b)^2 is always non-negative since squares are ≥ 0. Expanding gives a + b - 2√(ab) ≥ 0."
Grade: {"score": 2, "max_score": 4, "rationale": "Correct starting insight (1 pt) and expansion (1 pt), but stopped before reaching the AM-GM inequality. Missing the crucial rearrangement and final conclusion."}

GRADING PRINCIPLES:
1. Award points for correct mathematical reasoning, even if different from official solution
2. Deduct points for logical errors, missing steps, or incorrect conclusions
3. Partial credit is appropriate for incomplete but partially correct solutions
4. Examples alone without general proof receive ZERO credit - not partial credit
5. Correct approach with arithmetic errors receives partial credit
6. Completely wrong approach or no reasoning receives zero credit
7. Missing crucial justification steps should result in point deductions
8. Alternative valid approaches deserve full credit when properly executed
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
1. Carefully read the official solution to understand the expected approach and key proof steps
2. Analyze what the student did correctly - award points for valid reasoning even if different from official solution
3. CRITICAL: Check for these common student errors:
   - Examples without general proof (award 0 points for this section)
   - Missing justification for key claims
   - Logical leaps without explanation
   - Incomplete case analysis
   - Unstated assumptions
4. Compare against the grading guidelines point by point
5. Determine the final score and provide detailed rationale

GRADING STANDARDS:
- Full points: Complete, correct solution with proper justification
- Partial points: Correct approach with minor errors or gaps
- Minimal/zero points: Examples only, major logical gaps, or fundamentally wrong approach

IMPORTANT: You must respond in valid JSON format wrapped in <json>...</json> tags:
<json>
{{
    "thinking": "Your detailed step-by-step analysis here - explicitly mention any errors or gaps found",
    "score": <numerical score as integer>,
    "max_score": <maximum possible score as integer>,
    "rationale": "Detailed explanation of why this score was awarded, referencing specific grading criteria and any errors found",
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
                
                # Validate score and max_score are integers
                score = result.get("score")
                max_score = result.get("max_score")
                
                if score is not None and max_score is not None:
                    try:
                        score = int(score)
                        max_score = int(max_score)
                        prediction = f"{score}/{max_score}"
                        self.log_fn(f"Using score/max_score fields: {prediction}")
                    except (ValueError, TypeError):
                        self.log_fn(f"Warning: score/max_score not valid integers: {score}, {max_score}")
                
                if "response" in result and prediction == "None":
                    prediction = result["response"]
                    self.log_fn(f"Using 'response' field: {prediction}")
                
                if prediction == "None":
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
            reflection_msg = f"""Review your grading above carefully. Perform these specific checks:

COMMON GRADING ERRORS TO WATCH FOR:
1. OVER-GRADING: Did you award points for:
   - Examples without general proof? (Should get 0 points)
   - Missing crucial justification steps?
   - Logical leaps without explanation?
   - Partial work presented as complete?

2. UNDER-GRADING: Did you miss:
   - Valid alternative approaches different from the official solution?
   - Correct reasoning that uses different notation or steps?
   - Partial credit for correct intermediate steps?

3. LOGICAL GAPS: Check if the student's solution has:
   - Unstated assumptions that need justification?
   - Missing case analysis?
   - Incomplete induction steps?
   - Unverified claims?

4. CONSISTENCY: Compare against:
   - The grading guidelines point-by-point
   - The few-shot examples (especially Example 2: examples alone = 0 credit)
   - What an IMO grader would expect

5. ARITHMETIC vs LOGIC: Distinguish between:
   - Minor arithmetic errors (deduct 1 point)
   - Fundamental logical errors (deduct more)
   - Missing entire proof sections (significant deduction)

If you identify issues with your grade, provide a corrected JSON with the revised score and detailed rationale explaining what error you caught.
If your grade is correct, confirm it with the same score and explain why it withstands scrutiny.

IMPORTANT: You must respond in valid JSON format wrapped in <json>...</json> tags:
<json>
{{
    "reflection": "Your detailed self-review - what specific checks did you perform and what did you find?",
    "revised_score": <score as integer>,
    "revised_max_score": <max_score as integer>,
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
                    
                    # Validate revised scores are integers
                    revised_score = result.get("revised_score")
                    revised_max_score = result.get("revised_max_score")
                    
                    if revised_score is not None and revised_max_score is not None:
                        try:
                            revised_score = int(revised_score)
                            revised_max_score = int(revised_max_score)
                            prediction = f"{revised_score}/{revised_max_score}"
                            self.log_fn(f"Using revised_score/revised_max_score: {prediction}")
                        except (ValueError, TypeError):
                            self.log_fn(f"Warning: revised scores not valid integers: {revised_score}, {revised_max_score}")
                    
                    if "final_response" in result and prediction == "None":
                        prediction = result["final_response"]
                        self.log_fn(f"Using 'final_response' field: {prediction}")
                    
                    if prediction == "None":
                        self.log_fn(f"Warning: Reflection JSON missing expected fields. Keys: {list(result.keys())}")
                else:
                    self.log_fn("Warning: No JSON found in reflection response")
            except Exception as e:
                self.log_fn(f"Error extracting revised prediction: {e}")
                import traceback
                self.log_fn(f"Traceback: {traceback.format_exc()}")

        return str(prediction), msg_history
