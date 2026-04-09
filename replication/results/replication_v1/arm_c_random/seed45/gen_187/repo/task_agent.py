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
    Includes robust JSON repair for common LLM output issues.
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
        
        # Try direct parsing first
        try:
            results.append(json.loads(inner))
            continue
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON from within the content
        try:
            json_start = inner.find("{")
            json_end = inner.rfind("}")
            if json_start != -1 and json_end != -1 and json_end > json_start:
                results.append(json.loads(inner[json_start:json_end + 1]))
                continue
        except json.JSONDecodeError:
            pass
        
        # Try JSON repair: fix common issues like trailing commas
        try:
            repaired = _repair_json(inner)
            if repaired:
                results.append(json.loads(repaired))
                continue
        except (json.JSONDecodeError, ValueError):
            pass
    
    # Fallback: try markdown code blocks with json
    if not results:
        json_blocks = re.findall(r'```json\s*(.*?)\s*```', text, re.DOTALL)
        for block in json_blocks:
            try:
                results.append(json.loads(block.strip()))
            except json.JSONDecodeError:
                # Try repair on markdown blocks too
                try:
                    repaired = _repair_json(block.strip())
                    if repaired:
                        results.append(json.loads(repaired))
                except (json.JSONDecodeError, ValueError):
                    continue
    
    # Last resort: try to find any JSON-like structure in the text
    if not results:
        try:
            # Find outermost braces
            json_start = text.find("{")
            json_end = text.rfind("}")
            if json_start != -1 and json_end != -1 and json_end > json_start:
                candidate = text[json_start:json_end + 1]
                try:
                    results.append(json.loads(candidate))
                except json.JSONDecodeError:
                    repaired = _repair_json(candidate)
                    if repaired:
                        results.append(json.loads(repaired))
        except (json.JSONDecodeError, ValueError):
            pass
    
    return results or None


def _repair_json(text: str) -> str | None:
    """Attempt to repair common JSON formatting issues from LLM outputs.
    
    Fixes:
    - Trailing commas before closing braces/brackets
    - Single quotes instead of double quotes
    - Unquoted keys
    - Missing closing braces/brackets
    """
    if not text or not text.strip():
        return None
    
    text = text.strip()
    
    # Remove trailing commas before } or ]
    import re
    text = re.sub(r',(\s*[}\]])', r'\1', text)
    
    # Try to balance braces
    open_braces = text.count('{') - text.count('}')
    open_brackets = text.count('[') - text.count(']')
    
    if open_braces > 0:
        text = text + ('}' * open_braces)
    if open_brackets > 0:
        text = text + (']' * open_brackets)
    
    return text


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

Example 6:
Problem: Find the sum of the first 100 positive integers.
Solution: Using the arithmetic series formula: S = n(n+1)/2 = 100(101)/2 = 5050.
Grading Guidelines: Award 1 point for identifying the correct formula, 1 point for correct substitution, 1 point for final answer.
Student Answer: "I can pair the numbers: 1+100=101, 2+99=101, 3+98=101, and so on. There are 50 such pairs, so the sum is 50 × 101 = 5050."
Grade: {"score": 3, "max_score": 3, "rationale": "Excellent alternative solution using the pairing method (Gauss's approach). Correct reasoning and final answer."}

Example 7:
Problem: Solve the equation 2x + 5 = 13 for x.
Solution: Subtract 5 from both sides: 2x = 8. Divide by 2: x = 4.
Grading Guidelines: Award 1 point for correct algebraic manipulation, 1 point for correct final answer.
Student Answer: "2x = 13 - 5 = 8, so x = 4"
Grade: {"score": 2, "max_score": 2, "rationale": "Correct solution with proper algebraic steps. Student showed intermediate work (2x=8) and final answer."}

Example 8:
Problem: Find the area of a rectangle with length 8 and width 5.
Solution: Area = length × width = 8 × 5 = 40.
Grading Guidelines: Award 1 point for correct formula, 1 point for correct calculation, 1 point for correct units (if mentioned).
Student Answer: "The area is 40 square units. I multiplied 8 by 5."
Grade: {"score": 3, "max_score": 3, "rationale": "Complete answer with correct formula application, calculation, and units."}
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

GRADING INSTRUCTIONS:
1. First, carefully read the official solution and understand the key steps and concepts required.
2. Examine the grading guidelines to understand how points should be allocated.
3. Analyze the student's answer line by line, comparing it to the official solution.
4. Award points ONLY for correct mathematical reasoning that matches the official solution.
5. Do NOT award points for:
   - Correct final answers without proper reasoning
   - Vague or incomplete explanations
   - Incorrect methods even if they accidentally yield the right answer
   - Missing critical steps from the official solution
6. Consider alternative valid approaches if mathematically sound.
7. Be precise with partial credit - award only when genuine progress is demonstrated.

Think step by step:
1. What are the key components of the official solution?
2. Which components did the student address correctly?
3. Which components are missing or incorrect in the student's answer?
4. How do the points add up according to the grading guidelines?
5. What is the final score and detailed rationale?

Respond in JSON format with the following schema:
<json>
{{
    "thinking": "Your detailed step-by-step analysis here - be specific about what the student got right and wrong",
    "score": <numerical score - must be a non-negative number>,
    "max_score": <maximum possible score - must be a positive number>,
    "rationale": "Detailed explanation of why this score was awarded, referencing specific points from the grading guidelines",
    "response": "<score>/<max_score> - <brief summary of the grade>"
}}
</json>"""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON with validation
        prediction = "None"
        try:
            extracted = _extract_jsons(msg_history[-1]["text"])
            if extracted:
                result = extracted[-1]
                
                # Validate score values if present
                score = None
                max_score = None
                
                if "score" in result:
                    try:
                        score = float(result["score"])
                        if score < 0:
                            self.log_fn(f"Warning: Negative score {score}, treating as None")
                            score = None
                    except (ValueError, TypeError):
                        pass
                
                if "max_score" in result:
                    try:
                        max_score = float(result["max_score"])
                        if max_score <= 0:
                            self.log_fn(f"Warning: Invalid max_score {max_score}, treating as None")
                            max_score = None
                    except (ValueError, TypeError):
                        pass
                
                # Use validated values
                if "response" in result and result["response"]:
                    prediction = result["response"]
                elif score is not None and max_score is not None:
                    if score > max_score:
                        self.log_fn(f"Warning: Score {score} exceeds max {max_score}, capping at max")
                        prediction = f"{int(max_score)}/{int(max_score)}"
                    else:
                        prediction = f"{int(score)}/{int(max_score)}"
                elif "response" in result:
                    prediction = result["response"]
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        # Step 2: Self-reflection to verify the grade
        if prediction != "None" and len(msg_history) >= 2:
            reflection_msg = f"""Review your grading above carefully. Perform a rigorous self-assessment:

CRITICAL CHECKS:
1. POINT-BY-POINT VERIFICATION: Go through each point in the grading guidelines. Did the student actually earn each point you awarded? Be specific.
2. ERROR ANALYSIS: Did you miss any mathematical errors, logical gaps, or incorrect assumptions in the student's work?
3. RIGOR CHECK: Are you being consistent with IMO grading standards? IMO graders are strict - partial credit is limited.
4. ALTERNATIVE SOLUTIONS: If the student used a different approach, is it mathematically valid and complete?
5. SCORE CALCULATION: Re-count the points. Does your math match your awarded score?

COMMON MISTAKES TO AVOID:
- Awarding points for "showing work" when the work contains errors
- Giving partial credit for vague or incomplete explanations
- Being swayed by a correct final answer when the reasoning is flawed
- Overlooking missing steps that are required in the official solution

DECISION RULE:
- If you find ANY errors in your initial assessment, you MUST revise the grade
- If the initial grade is accurate, confirm it with specific evidence
- Be decisive: either revise or confirm, don't be ambiguous

Respond in JSON format:
<json>
{{
    "reflection": "Detailed self-review addressing each critical check with specific evidence",
    "grade_revised": true/false,
    "revised_score": <numerical score - must be non-negative>,
    "revised_max_score": <maximum possible score - must be positive>,
    "revision_reason": "Specific explanation of what changed and why, or detailed confirmation that initial grade was accurate",
    "final_response": "<score>/<max_score> - <brief summary of final grade>"
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
                        
                        # Validate and extract scores
                        revised_score = None
                        revised_max = None
                        
                        if "revised_score" in result:
                            try:
                                revised_score = float(result["revised_score"])
                            except (ValueError, TypeError):
                                pass
                        
                        if "revised_max_score" in result:
                            try:
                                revised_max = float(result["revised_max_score"])
                            except (ValueError, TypeError):
                                pass
                        
                        # Validate score bounds
                        if revised_score is not None and revised_max is not None:
                            if revised_score > revised_max:
                                self.log_fn(f"Warning: Revised score {revised_score} exceeds max {revised_max}, using original")
                            elif revised_score < 0:
                                self.log_fn(f"Warning: Revised score {revised_score} is negative, using original")
                            else:
                                # Valid scores - use them
                                if "final_response" in result and result["final_response"]:
                                    prediction = result["final_response"]
                                else:
                                    prediction = f"{int(revised_score)}/{int(revised_max)}"
                        elif "final_response" in result and result["final_response"]:
                            prediction = result["final_response"]
                except Exception as e:
                    self.log_fn(f"Error extracting revised prediction: {e}")
            except Exception as e:
                self.log_fn(f"Error during reflection: {e}")

        return str(prediction), msg_history
