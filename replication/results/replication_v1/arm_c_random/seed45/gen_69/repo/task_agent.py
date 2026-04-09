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
    Also handles markdown code blocks as fallback.
    """
    results = []
    search_from = 0
    
    # First try <json>...</json> blocks
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
        except json.JSONDecodeError:
            # Try to extract JSON from within the content
            try:
                # Find first { and last }
                json_start = inner.find("{")
                json_end = inner.rfind("}")
                if json_start != -1 and json_end != -1 and json_end > json_start:
                    results.append(json.loads(inner[json_start:json_end + 1]))
            except json.JSONDecodeError:
                continue
    
    # Fallback: try markdown code blocks with json
    if not results:
        json_blocks = re.findall(r'```json\s*(.*?)\s*```', text, re.DOTALL)
        for block in json_blocks:
            try:
                results.append(json.loads(block.strip()))
            except json.JSONDecodeError:
                continue
    
    return results or None


def _validate_grade(result: dict, max_score: int | None = None) -> tuple[bool, str]:
    """Validate that a grade result has valid structure and values.
    
    Args:
        result: The parsed JSON result dict
        max_score: Optional expected max_score to validate against
        
    Returns:
        (is_valid, error_message)
    """
    # Check required fields exist
    if "score" not in result:
        return False, "Missing 'score' field"
    if "max_score" not in result:
        return False, "Missing 'max_score' field"
    
    score = result.get("score")
    result_max = result.get("max_score")
    
    # Validate types
    if not isinstance(score, (int, float)):
        return False, f"Score must be numeric, got {type(score).__name__}"
    if not isinstance(result_max, (int, float)):
        return False, f"Max score must be numeric, got {type(result_max).__name__}"
    
    # Validate ranges
    if score < 0:
        return False, f"Score cannot be negative: {score}"
    if result_max <= 0:
        return False, f"Max score must be positive: {result_max}"
    if score > result_max:
        return False, f"Score {score} exceeds max {result_max}"
    
    # Validate against expected max if provided
    if max_score is not None and result_max != max_score:
        return False, f"Max score mismatch: got {result_max}, expected {max_score}"
    
    # Validate rationale if present (should be non-empty string)
    if "rationale" in result:
        rationale = result.get("rationale")
        if not isinstance(rationale, str) or len(rationale.strip()) < 10:
            return False, "Rationale must be a meaningful string (at least 10 characters)"
    
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
Grading Guidelines: Award 1 point for setting up odd number representation, 1 point for algebraic manipulation, 1 point for conclusion.
Student Answer: "Odd numbers end in 1,3,5,7,9. Adding two odd numbers gives an even number. For example, 3+5=8 which is even."
Grade: {"score": 1, "max_score": 3, "rationale": "Student only provided examples without general proof. Missing algebraic representation and general reasoning."}

Example 3:
Problem: Prove that for any prime p > 3, p^2 - 1 is divisible by 24.
Solution: p^2 - 1 = (p-1)(p+1). Since p is odd and not divisible by 3, one of p-1, p, p+1 is divisible by 3. Since p is prime > 3, neither p-1 nor p+1 is divisible by p. Among three consecutive integers, one is divisible by 3. Since p is odd, both p-1 and p+1 are even, and one of them is divisible by 4. Thus (p-1)(p+1) is divisible by 8×3 = 24.
Grading Guidelines: Award 1 point for factoring, 1 point for divisibility by 8 argument, 1 point for divisibility by 3 argument, 1 point for combining to get 24.
Student Answer: "p^2 - 1 = (p-1)(p+1). Since p is odd, p-1 and p+1 are consecutive even integers, so one is divisible by 4 and the other by 2, giving divisibility by 8. Also, among p-1, p, p+1, one must be divisible by 3, and since p is prime > 3, it's not divisible by 3, so either p-1 or p+1 is. Therefore p^2 - 1 is divisible by 8×3 = 24."
Grade: {"score": 4, "max_score": 4, "rationale": "Excellent proof covering all required elements: factoring, divisibility by 8, divisibility by 3, and final combination."}

Example 4:
Problem: Find the number of ways to arrange 6 people around a circular table where rotations are considered the same.
Solution: For circular arrangements, we fix one person's position to account for rotational equivalence. The remaining 5 people can be arranged in 5! = 120 ways.
Grading Guidelines: Award 1 point for recognizing circular arrangement formula, 1 point for correct calculation, 1 point for final answer.
Student Answer: "There are 6! = 720 ways to arrange 6 people. But since it's a circle, we divide by 6 to get 120 ways."
Grade: {"score": 3, "max_score": 3, "rationale": "Correct answer with valid reasoning. Student correctly applied circular permutation formula (n-1)! or equivalently divided by n to account for rotational symmetry."}

Example 5:
Problem: Prove that √2 is irrational.
Solution: Assume √2 = p/q in lowest terms. Then 2q^2 = p^2, so p^2 is even, thus p is even. Let p = 2k. Then 2q^2 = 4k^2, so q^2 = 2k^2, making q even. Contradiction: p and q both even, not in lowest terms.
Grading Guidelines: Award 1 point for setting up proof by contradiction, 1 point for showing p is even, 1 point for showing q is even, 1 point for reaching contradiction.
Student Answer: "Suppose √2 is rational. Then it can be written as a fraction p/q. Squaring gives 2 = p^2/q^2, so p^2 = 2q^2. This means p^2 is even, so p is even. But then q must also be even, which contradicts p/q being in lowest terms. Therefore √2 is irrational."
Grade: {"score": 4, "max_score": 4, "rationale": "Complete proof by contradiction with all logical steps clearly presented."}
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
1. Analyze what the student did correctly according to the official solution
2. Identify any errors, gaps, or missing steps
3. Compare against the grading guidelines - count specific points awarded
4. Determine the score and provide detailed rationale

IMPORTANT GRADING PRINCIPLES:
- Be consistent with the official solution and grading guidelines
- Award partial credit only when explicitly justified by the guidelines
- Document specific errors or gaps that led to point deductions
- Ensure your rationale directly references the grading criteria

Respond in JSON format with the following schema:
<json>
{{
    "thinking": "Your detailed step-by-step analysis here",
    "score": <numerical score>,
    "max_score": <maximum possible score>,
    "rationale": "Detailed explanation of why this score was awarded, referencing specific grading criteria",
    "response": "<score>/<max_score> - <brief summary>"
}}
</json>"""

        # Step 1: Initial grading with chain-of-thought (with retry for robustness)
        max_retries = 2
        prediction = "None"
        msg_history = []
        
        for attempt in range(max_retries + 1):
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=[] if attempt == 0 else msg_history,
                )
                
                # Extract prediction from JSON with validation
                extracted = _extract_jsons(msg_history[-1]["text"])
                if extracted:
                    result = extracted[-1]
                    # Validate the grade structure
                    is_valid, error_msg = _validate_grade(result)
                    if is_valid:
                        if "response" in result:
                            prediction = result["response"]
                        elif "score" in result and "max_score" in result:
                            prediction = f"{result['score']}/{result['max_score']}"
                        break  # Success, exit retry loop
                    else:
                        self.log_fn(f"Grade validation failed (attempt {attempt + 1}): {error_msg}")
                        # Try to use partial data if available
                        if "score" in result and "max_score" in result:
                            score = result["score"]
                            max_s = result["max_score"]
                            if isinstance(score, (int, float)) and isinstance(max_s, (int, float)):
                                if 0 <= score <= max_s:
                                    prediction = f"{score}/{max_s}"
                                    break  # Partial success, exit retry loop
            except Exception as e:
                self.log_fn(f"Error extracting prediction (attempt {attempt + 1}): {e}")
                if attempt == max_retries:
                    break  # Exhausted retries

        # Step 2: Self-reflection to verify the grade
        if prediction != "None" and len(msg_history) >= 2:
            reflection_msg = f"""Review your grading above carefully. Perform a critical self-assessment:

1. STRICTNESS CHECK: Did you award points too generously? Re-examine the student's work against the official solution point by point.
2. ERROR DETECTION: Did you overlook any mistakes, gaps, or logical flaws in the student's reasoning?
3. GUIDELINE ALIGNMENT: Does your score precisely match the grading guidelines? Count the specific points awarded vs. the guidelines.
4. CONSISTENCY: Would an expert IMO grader give the same score? Be honest about any leniency.
5. PARTIAL CREDIT: Did you award partial credit appropriately? Too much or too little?
6. RATIONALE QUALITY: Is your rationale specific, detailed, and directly tied to the grading guidelines?

IMPORTANT: If your initial grade was too lenient or strict, you MUST revise it. Do not simply confirm your first assessment.

Respond in JSON format:
<json>
{{
    "reflection": "Detailed self-review addressing each check above",
    "grade_revised": true/false,
    "revised_score": <score>,
    "revised_max_score": <max_score>,
    "revision_reason": "Explanation if grade changed, or 'No change - initial grade accurate'",
    "final_response": "<score>/<max_score> - <brief summary>"
}}
</json>"""
            
            try:
                reflection_response, msg_history, _ = get_response_from_llm(
                    msg=reflection_msg,
                    model=self.model,
                    msg_history=msg_history,
                )
                
                # Try to extract revised prediction with validation
                try:
                    extracted = _extract_jsons(msg_history[-1]["text"])
                    if extracted:
                        result = extracted[-1]
                        # Check if grade was revised and validate
                        grade_revised = result.get("grade_revised", False)
                        
                        # Determine which score fields to use
                        if grade_revised and "revised_score" in result and "revised_max_score" in result:
                            score = result["revised_score"]
                            max_s = result["revised_max_score"]
                        elif "score" in result and "max_score" in result:
                            score = result["score"]
                            max_s = result["max_score"]
                        else:
                            score = None
                            max_s = None
                        
                        # Validate the grade
                        if score is not None and max_s is not None:
                            is_valid, error_msg = _validate_grade({"score": score, "max_score": max_s})
                            if is_valid:
                                if "final_response" in result:
                                    prediction = result["final_response"]
                                else:
                                    prediction = f"{score}/{max_s}"
                            else:
                                self.log_fn(f"Revised grade validation failed: {error_msg}")
                except Exception as e:
                    self.log_fn(f"Error extracting revised prediction: {e}")
            except Exception as e:
                self.log_fn(f"Error during reflection: {e}")

        return str(prediction), msg_history
