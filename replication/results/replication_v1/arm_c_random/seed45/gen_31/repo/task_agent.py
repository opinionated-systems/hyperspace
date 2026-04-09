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
    Also handles markdown code blocks and plain JSON.
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
        except json.JSONDecodeError:
            continue
    
    # Also try markdown code blocks
    search_from = 0
    while True:
        start = text.find("```json", search_from)
        if start == -1:
            break
        end = text.find("```", start + 7)
        if end == -1:
            break
        inner = text[start + 7:end].strip()
        search_from = end + 3
        try:
            results.append(json.loads(inner))
        except json.JSONDecodeError:
            continue
    
    # Try to find plain JSON objects as fallback
    if not results:
        try:
            # Look for JSON-like structures with score/max_score fields
            pattern = r'\{\s*"[^"]+":\s*[^}]+\}'
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    parsed = json.loads(match)
                    if any(k in parsed for k in ["score", "max_score", "thinking", "rationale"]):
                        results.append(parsed)
                except json.JSONDecodeError:
                    continue
        except Exception:
            pass
    
    return results or None


def _validate_score(result: dict, default_max: int = 3) -> dict:
    """Validate and normalize score fields in the result."""
    validated = dict(result)
    
    # Ensure score and max_score are integers
    try:
        if "score" in validated:
            validated["score"] = int(float(validated["score"]))
        if "max_score" in validated:
            validated["max_score"] = int(float(validated["max_score"]))
    except (ValueError, TypeError):
        pass
    
    # Set defaults if missing
    if "max_score" not in validated or validated["max_score"] is None:
        validated["max_score"] = default_max
    if "score" not in validated or validated["score"] is None:
        validated["score"] = 0
    
    # Clamp score to valid range
    validated["score"] = max(0, min(validated["score"], validated["max_score"]))
    
    return validated


# Few-shot examples for IMO grading - expanded with more diverse examples
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
Student Answer: "2^1 = 2 ≡ 2 (mod 3), 2^2 = 4 ≡ 1 (mod 3), 2^3 = 8 ≡ 2 (mod 3), 2^4 = 16 ≡ 1 (mod 3). The pattern repeats every 2 powers. Since 100 is even, 2^100 ≡ 1 (mod 3)."
Grade: {"score": 3, "max_score": 3, "rationale": "Excellent solution using pattern recognition. Student correctly identified the cycle and applied it to find the answer."}

Example 4:
Problem: Prove that √2 is irrational.
Solution: Assume √2 = p/q where p,q are coprime integers. Then 2q^2 = p^2, so p^2 is even, thus p is even. Let p=2k. Then 2q^2 = 4k^2, so q^2 = 2k^2, making q even. Contradiction: p and q both even.
Grading Guidelines: Award 1 point for setting up proof by contradiction, 1 point for showing p is even, 1 point for showing q is even, 1 point for reaching contradiction.
Student Answer: "Suppose √2 is rational. Then it can be written as a fraction. Squaring both sides gives 2 = p^2/q^2. This means p^2 = 2q^2. So p^2 is even. Therefore p is even. This leads to a contradiction."
Grade: {"score": 2, "max_score": 4, "rationale": "Student correctly set up the contradiction and showed p is even, but didn't complete the proof by showing q is even and explicitly stating the contradiction."}
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
        # Parse max_score from grading guidelines if available
        default_max = 3
        guidelines = inputs.get('grading_guidelines', '')
        max_score_match = re.search(r'max(?:imum)?\s*score\s*[:=]\s*(\d+)', guidelines, re.IGNORECASE)
        if max_score_match:
            default_max = int(max_score_match.group(1))
        else:
            # Count points mentioned in guidelines
            point_count = len(re.findall(r'(?:award|point|credit)', guidelines, re.IGNORECASE))
            if point_count > 0:
                default_max = max(point_count, default_max)
        
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
3. Compare against the grading guidelines point by point
4. Determine the score (0 to {default_max}) and provide detailed rationale
5. Consider: would another expert grader reach the same conclusion?

Respond in JSON format with the following schema:
<json>
{{
    "thinking": "Your detailed step-by-step analysis here",
    "score": <numerical score between 0 and {default_max}>,
    "max_score": {default_max},
    "rationale": "Detailed explanation of why this score was awarded, referencing specific grading criteria",
    "response": "<score>/{default_max} - <brief summary of key points>"
}}
</json>"""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON
        prediction = "None"
        initial_result = None
        try:
            extracted = _extract_jsons(msg_history[-1]["text"])
            if extracted:
                initial_result = _validate_score(extracted[-1], default_max)
                if "response" in initial_result:
                    prediction = initial_result["response"]
                elif "score" in initial_result and "max_score" in initial_result:
                    prediction = f"{initial_result['score']}/{initial_result['max_score']}"
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        # Step 2: Self-reflection to verify the grade
        if prediction != "None" and len(msg_history) >= 2:
            reflection_msg = f"""Review your grading above carefully. Perform a critical self-assessment:

1. Bias Check: Did you award points the student didn't fully earn? Be conservative.
2. Error Detection: Did you miss any subtle errors or incorrect reasoning?
3. Guideline Alignment: Is your score strictly consistent with the grading guidelines?
4. Peer Consistency: Would another IMO grader award the same score?
5. Completeness: Did the student address all parts of the problem?

Current assessment: Score {initial_result.get('score', '?')}/{initial_result.get('max_score', default_max)}

If you need to revise your grade, provide the corrected JSON with clear reasoning for the change. If your grade is correct, confirm it with the same JSON structure.

Respond in JSON format:
<json>
{{
    "reflection": "Your detailed self-review addressing each check above",
    "confidence": "high/medium/low - how certain are you of this grade?",
    "revised_score": <score>,
    "revised_max_score": {default_max},
    "revision_reason": "Explanation if score changed, or 'No change - grade is accurate'",
    "final_response": "<score>/{default_max} - <brief summary>"
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
                    revised_result = _validate_score(extracted[-1], default_max)
                    if "final_response" in revised_result:
                        prediction = revised_result["final_response"]
                    elif "revised_score" in revised_result:
                        prediction = f"{revised_result['revised_score']}/{revised_result['revised_max_score']}"
                    elif "score" in revised_result:
                        prediction = f"{revised_result['score']}/{revised_result['max_score']}"
            except Exception as e:
                self.log_fn(f"Error extracting revised prediction: {e}")

        return str(prediction), msg_history
