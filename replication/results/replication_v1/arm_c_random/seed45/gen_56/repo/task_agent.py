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


def _calculate_confidence(result: dict, inputs: dict) -> float:
    """Calculate confidence score for the grading result.
    
    Returns a confidence score between 0.0 and 1.0 based on:
    - Presence and quality of rationale/thinking
    - Score alignment with guidelines
    - Completeness of the response
    - Consistency between score and rationale
    """
    confidence = 0.5  # Base confidence
    
    # Check for detailed rationale
    rationale = result.get("rationale", "")
    if len(rationale) > 100:
        confidence += 0.15
    if len(rationale) > 200:
        confidence += 0.1
    
    # Check for detailed thinking
    thinking = result.get("thinking", "")
    if len(thinking) > 150:
        confidence += 0.1
    
    # Check if score is present and valid
    score = result.get("score")
    max_score = result.get("max_score")
    if score is not None and isinstance(score, (int, float)):
        confidence += 0.1
        
        # Validate score is within bounds
        if max_score is not None and isinstance(max_score, (int, float)):
            if 0 <= score <= max_score:
                confidence += 0.05
            else:
                confidence -= 0.2  # Penalty for out-of-bounds score
    
    # Check if max_score is present and valid
    if max_score is not None and isinstance(max_score, (int, float)):
        confidence += 0.05
    
    # Check if response format is correct
    response = result.get("response", "")
    if "/" in response and len(response.split("/")) == 2:
        confidence += 0.1
        # Verify response matches score/max_score
        try:
            parts = response.split("/")
            resp_score = int(parts[0].strip().split()[-1])  # Handle "X/Y - summary"
            resp_max = int(parts[1].strip().split()[0])     # Handle "X/Y - summary"
            if score is not None and max_score is not None:
                if resp_score == score and resp_max == max_score:
                    confidence += 0.05  # Consistency bonus
        except (ValueError, IndexError):
            pass
    
    # Check alignment with grading guidelines if available
    guidelines = inputs.get("grading_guidelines", "")
    if guidelines and rationale:
        # Simple heuristic: check if rationale mentions key guideline terms
        guideline_terms = set(guidelines.lower().split())
        rationale_terms = set(rationale.lower().split())
        overlap = len(guideline_terms & rationale_terms)
        if overlap > 0:
            confidence += min(0.1, overlap / len(guideline_terms) * 0.1)
    
    return max(0.0, min(1.0, confidence))


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
Problem: Find the remainder when 2^100 is divided by 3.
Solution: Note that 2 ≡ -1 (mod 3), so 2^100 ≡ (-1)^100 ≡ 1 (mod 3). The remainder is 1.
Grading Guidelines: Award 1 point for recognizing the pattern/modular arithmetic, 1 point for correct computation, 1 point for final answer.
Student Answer: "2^1 = 2, 2^2 = 4, 2^3 = 8, 2^4 = 16... The remainders when divided by 3 are 2, 1, 2, 1... The pattern repeats every 2. Since 100 is even, the remainder is 1."
Grade: {"score": 3, "max_score": 3, "rationale": "Student correctly identified the pattern, used modular arithmetic implicitly, and arrived at the correct answer through valid reasoning."}

Example 4:
Problem: Prove that √2 is irrational.
Solution: Assume √2 = p/q in lowest terms. Then 2q^2 = p^2, so p^2 is even, thus p is even. Let p=2k, then 2q^2 = 4k^2, so q^2 = 2k^2, making q even. Contradiction: p and q both even.
Grading Guidelines: Award 1 point for setting up proof by contradiction, 1 point for showing p is even, 1 point for showing q is even, 1 point for concluding contradiction.
Student Answer: "Suppose √2 is rational. Then it can be written as a fraction. Squaring both sides gives 2 = p²/q². This means p² = 2q², so p² is even. Therefore p is even. This leads to a contradiction, so √2 is irrational."
Grade: {"score": 3, "max_score": 4, "rationale": "Student correctly set up proof by contradiction and showed p is even, but did not explicitly show q is even or clearly state the final contradiction. Missing one step in the logical chain."}
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

        # Extract prediction from JSON
        prediction = "None"
        confidence = 0.0
        try:
            extracted = _extract_jsons(msg_history[-1]["text"])
            if extracted:
                result = _validate_score(extracted[-1])
                confidence = _calculate_confidence(result, inputs)
                if "response" in result:
                    prediction = result["response"]
                elif "score" in result and "max_score" in result:
                    prediction = f"{result['score']}/{result['max_score']}"
                elif "score" in result:
                    # Fallback if only score is present
                    max_score = result.get("max_score", 10)
                    prediction = f"{result['score']}/{max_score}"
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        # Step 2: Self-reflection to verify the grade
        if prediction != "None" and len(msg_history) >= 2:
            reflection_msg = f"""Review your grading above carefully. Check for these common errors:

1. PARTIAL CREDIT: Did you award partial credit for correct reasoning even if the final answer is wrong?
2. MISSING STEPS: Did the student skip any steps required by the grading guidelines?
3. ALTERNATIVE METHODS: Did the student use a valid approach different from the official solution?
4. ARITHMETIC ERRORS: Did you penalize appropriately for minor calculation mistakes vs conceptual errors?
5. INCOMPLETE PROOFS: Did the student claim a result without justification?
6. CONSISTENCY: Would another expert grader give the same score?

Guidelines for revision:
- If the student showed correct reasoning but made a minor error, consider partial credit
- If the student used a different valid method, award full credit for correct parts
- If you missed a required step in the guidelines, adjust the score

If you need to revise your grade, provide the corrected JSON with detailed explanation. If your grade is correct, confirm it with the same score.

Respond in JSON format:
<json>
{{
    "reflection": "Your detailed self-review here - explain what you checked and why",
    "revised_score": <score>,
    "revised_max_score": <max_score>,
    "final_response": "<score>/<max_score> - <brief summary of decision>"
}}
</json>"""
            
            reflection_response, msg_history, _ = get_response_from_llm(
                msg=reflection_msg,
                model=self.model,
                msg_history=msg_history,
            )
            
            # Try to extract revised prediction
            try:
                extracted = _extract_jsons(msg_history[-1]["text"])
                if extracted:
                    result = _validate_score(extracted[-1])
                    # Update confidence after reflection
                    confidence = _calculate_confidence(result, inputs)
                    if "final_response" in result:
                        prediction = result["final_response"]
                    elif "revised_score" in result and "revised_max_score" in result:
                        prediction = f"{result['revised_score']}/{result['revised_max_score']}"
                    elif "revised_score" in result:
                        # Fallback if only revised_score is present
                        max_score = result.get("revised_max_score", 10)
                        prediction = f"{result['revised_score']}/{max_score}"
            except Exception as e:
                self.log_fn(f"Error extracting revised prediction: {e}")

        # Log confidence for monitoring
        if confidence > 0:
            self.log_fn(f"Grading confidence: {confidence:.2f}")

        return str(prediction), msg_history
