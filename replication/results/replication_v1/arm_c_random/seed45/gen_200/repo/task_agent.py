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
    Also attempts to parse JSON directly if no tags are found.
    """
    results = []
    search_from = 0
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
            # Try to extract JSON from markdown code blocks
            try:
                if "```json" in inner:
                    json_start = inner.find("```json") + 7
                    json_end = inner.find("```", json_start)
                    if json_end > json_start:
                        results.append(json.loads(inner[json_start:json_end].strip()))
                elif "```" in inner:
                    json_start = inner.find("```") + 3
                    json_end = inner.find("```", json_start)
                    if json_end > json_start:
                        results.append(json.loads(inner[json_start:json_end].strip()))
            except json.JSONDecodeError:
                continue
    
    # If no <json> tags found, try to find JSON objects directly
    if not results:
        # Look for JSON objects in the text
        brace_count = 0
        start_idx = -1
        for i, char in enumerate(text):
            if char == '{':
                if brace_count == 0:
                    start_idx = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start_idx != -1:
                    try:
                        results.append(json.loads(text[start_idx:i+1]))
                    except json.JSONDecodeError:
                        pass
                    start_idx = -1
    
    return results or None


def _validate_score(result: dict, max_score: int | None = None) -> dict:
    """Validate and normalize score fields in the result.
    
    Ensures score and max_score are valid integers within reasonable bounds.
    Returns a normalized result dict.
    """
    validated = dict(result)
    
    # Extract max_score from result or use provided default
    if max_score is None:
        max_score = validated.get("max_score", validated.get("revised_max_score", 10))
    
    try:
        max_score = int(max_score)
    except (ValueError, TypeError):
        max_score = 10
    
    # Validate score field
    score_fields = ["score", "revised_score"]
    for field in score_fields:
        if field in validated:
            try:
                score = int(validated[field])
                # Clamp score to valid range [0, max_score]
                score = max(0, min(score, max_score))
                validated[field] = score
            except (ValueError, TypeError):
                validated[field] = 0
    
    # Ensure max_score fields are valid
    max_score_fields = ["max_score", "revised_max_score"]
    for field in max_score_fields:
        if field in validated:
            try:
                validated[field] = max(1, int(validated[field]))
            except (ValueError, TypeError):
                validated[field] = max_score
    
    return validated


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
Solution: p^2 - 1 = (p-1)(p+1). Since p is odd and not divisible by 3, one of p-1, p, p+1 is divisible by 3, and both p-1 and p+1 are even consecutive integers, so one is divisible by 4.
Grading Guidelines: Award 1 point for factoring, 1 point for divisibility by 3 argument, 1 point for divisibility by 8 argument, 1 point for combining to 24.
Student Answer: "p^2 - 1 = (p-1)(p+1). Since p is prime > 3, it's odd. So p-1 and p+1 are even. One of them must be divisible by 4. Also, since p isn't divisible by 3, one of p-1 or p+1 is. So we have factors of 3 and 8, giving 24."
Grade: {"score": 4, "max_score": 4, "rationale": "Complete proof with correct factoring, divisibility by 3 and 8 arguments properly combined."}

Example 4:
Problem: Find the remainder when 2^100 is divided by 3.
Solution: Note that 2 ≡ -1 (mod 3), so 2^100 ≡ (-1)^100 ≡ 1 (mod 3).
Grading Guidelines: Award 1 point for recognizing 2 ≡ -1 (mod 3), 1 point for applying exponent, 1 point for final answer.
Student Answer: "2^1 = 2, 2^2 = 4 ≡ 1, 2^3 = 8 ≡ 2, 2^4 = 16 ≡ 1. The pattern repeats every 2. Since 100 is even, 2^100 ≡ 1 (mod 3)."
Grade: {"score": 3, "max_score": 3, "rationale": "Alternative valid approach using pattern recognition. Correctly identified cycle and applied to get answer."}
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

Think step by step and provide a thorough analysis:

1. **Understanding**: Restate the problem in your own words to ensure clarity.

2. **Official Solution Analysis**: Break down the official solution into key components and scoring points.

3. **Student Work Analysis**: 
   - What did the student do correctly? (Be specific about each correct step)
   - What errors or gaps exist? (Identify specific mistakes or missing reasoning)
   - Did the student use a valid alternative approach?

4. **Guideline Application**: Map the student's work to the grading guidelines point by point.

5. **Score Determination**: Assign a score based on the rubric, with clear justification for each point awarded or deducted.

Respond in JSON format with the following schema:
<json>
{{
    "thinking": "Your detailed step-by-step analysis covering all points above",
    "score": <numerical score>,
    "max_score": <maximum possible score>,
    "rationale": "Detailed explanation of why this score was awarded, referencing specific aspects of the student's work",
    "response": "<score>/<max_score> - <brief summary of key strengths and weaknesses>"
}}
</json>

Important: Ensure your JSON is valid and complete. The score must be an integer between 0 and max_score."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON with enhanced error handling
        prediction = "None"
        try:
            extracted = _extract_jsons(msg_history[-1]["text"])
            if extracted:
                result = _validate_score(extracted[-1])
                
                # Try multiple field combinations for maximum robustness
                if "response" in result:
                    prediction = result["response"]
                elif "final_response" in result:
                    prediction = result["final_response"]
                elif "score" in result and "max_score" in result:
                    prediction = f"{result['score']}/{result['max_score']}"
                elif "score" in result:
                    # Fallback if only score is present
                    max_score = result.get("max_score", 10)
                    prediction = f"{result['score']}/{max_score}"
                
                # Log successful extraction with score details
                self.log_fn(f"Initial grading: {prediction}")
            else:
                self.log_fn("No JSON found in initial response")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        # Step 2: Self-reflection to verify the grade
        if prediction != "None" and len(msg_history) >= 2:
            reflection_msg = f"""Review your grading above carefully. Perform a critical self-assessment:

1. **Accuracy Check**: Did you award points the student didn't actually earn? Look for:
   - Claims without justification
   - Logical gaps or errors
   - Incorrect calculations or conclusions

2. **Completeness Check**: Did you miss any errors or gaps? Consider:
   - Missing steps in the reasoning
   - Unstated assumptions
   - Incomplete case analysis

3. **Guideline Alignment**: Is your score consistent with the grading guidelines?
   - Are you applying the rubric correctly?
   - Are partial credits awarded appropriately?

4. **Peer Consistency**: Would another expert grader agree with your assessment?
   - Is your rationale clear and defensible?
   - Would the grade hold up under scrutiny?

5. **Alternative Validity**: Did the student use a valid alternative approach?
   - Different methods can be equally correct
   - Don't penalize for non-standard but valid reasoning

If you need to revise your grade, provide the corrected JSON with clear reasoning. If your grade is correct, confirm it with justification.

Respond in JSON format:
<json>
{{
    "reflection": "Your detailed self-review addressing each check above",
    "revision_needed": true/false,
    "revised_score": <score>,
    "revised_max_score": <max_score>,
    "revision_rationale": "Explanation for any changes made or why original stands",
    "final_response": "<score>/<max_score> - <brief summary>"
}}
</json>"""
            
            reflection_response, msg_history, _ = get_response_from_llm(
                msg=reflection_msg,
                model=self.model,
                msg_history=msg_history,
            )
            
            # Try to extract revised prediction with enhanced error handling
            try:
                extracted = _extract_jsons(msg_history[-1]["text"])
                if extracted:
                    result = _validate_score(extracted[-1])
                    # Check if revision was explicitly requested
                    revision_needed = result.get("revision_needed", False)
                    
                    if "final_response" in result:
                        prediction = result["final_response"]
                    elif "revised_score" in result:
                        # Use revised score if provided
                        max_score = result.get("revised_max_score", result.get("max_score", 10))
                        prediction = f"{result['revised_score']}/{max_score}"
                    elif revision_needed and "score" in result:
                        # Fallback: if revision_needed is true but no revised_score, use score
                        max_score = result.get("max_score", 10)
                        prediction = f"{result['score']}/{max_score}"
                    
                    self.log_fn(f"Reflection completed. Revision needed: {revision_needed}")
            except Exception as e:
                self.log_fn(f"Error extracting revised prediction: {e}")
                # Keep original prediction on error

        return str(prediction), msg_history
