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


def _validate_grading_result(result: dict) -> tuple[bool, str]:
    """Validate that a grading result has required fields and valid values.
    
    Returns (is_valid, error_message).
    """
    # Check for required fields
    if "score" not in result:
        return False, "Missing 'score' field"
    if "max_score" not in result:
        return False, "Missing 'max_score' field"
    
    # Validate score is numeric
    try:
        score_val = float(result["score"])
        max_val = float(result["max_score"])
    except (ValueError, TypeError):
        return False, f"Non-numeric score or max_score: {result.get('score')}, {result.get('max_score')}"
    
    # Validate score is within bounds
    if score_val < 0:
        return False, f"Negative score: {score_val}"
    if max_val <= 0:
        return False, f"Invalid max_score: {max_val}"
    if score_val > max_val:
        return False, f"Score {score_val} exceeds max_score {max_val}"
    
    return True, ""


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    Also handles markdown code blocks and bare JSON objects.
    """
    results = []
    search_from = 0
    
    # First, try to find <json>...</json> blocks
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
            # Try to extract JSON from within the text if it's wrapped in other content
            try:
                # Look for JSON object pattern with balanced braces
                json_start = inner.find("{")
                if json_start != -1:
                    # Find matching closing brace
                    brace_count = 0
                    json_end = -1
                    for i, char in enumerate(inner[json_start:], start=json_start):
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                json_end = i
                                break
                    if json_end != -1:
                        results.append(json.loads(inner[json_start:json_end + 1]))
            except json.JSONDecodeError:
                continue
    
    # If no <json> blocks found, try markdown code blocks
    if not results:
        # Match ```json ... ``` or just ``` ... ``` blocks
        code_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
        matches = re.findall(code_block_pattern, text, re.DOTALL)
        for match in matches:
            try:
                results.append(json.loads(match.strip()))
            except json.JSONDecodeError:
                continue
    
    # Last resort: try to find any JSON object in the text
    if not results:
        # Look for JSON-like patterns with balanced braces
        # This pattern matches complete JSON objects more accurately
        potential_jsons = []
        start_indices = [m.start() for m in re.finditer(r'\{', text)]
        
        for start_idx in start_indices:
            brace_count = 0
            for i, char in enumerate(text[start_idx:], start=start_idx):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        potential_jsons.append(text[start_idx:i+1])
                        break
        
        for match in potential_jsons:
            try:
                parsed = json.loads(match)
                # Only include if it has expected keys
                if any(key in parsed for key in ["score", "max_score", "thinking", "rationale", 
                                                   "reflection", "revised_score", "final_response"]):
                    results.append(parsed)
            except json.JSONDecodeError:
                continue
    
    return results or None


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
    """Task agent that solves IMO grading problems with chain-of-thought reasoning.
    
    Uses two-step process:
    1. Initial grading with chain-of-thought analysis
    2. Self-reflection to verify and potentially revise the grade
    
    Includes validation to ensure grading results have valid score/max_score values
    (non-negative, score <= max_score, numeric values).
    """

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
        try:
            extracted = _extract_jsons(msg_history[-1]["text"])
            if extracted:
                result = extracted[-1]
                # Validate the grading result
                is_valid, error_msg = _validate_grading_result(result)
                if not is_valid:
                    self.log_fn(f"Invalid grading result: {error_msg}")
                
                if "response" in result and result["response"]:
                    prediction = str(result["response"])
                elif "score" in result and "max_score" in result:
                    score = result["score"]
                    max_score = result["max_score"]
                    # Validate score is numeric
                    try:
                        score_val = float(score)
                        max_val = float(max_score)
                        prediction = f"{int(score_val)}/{int(max_val)}"
                    except (ValueError, TypeError):
                        prediction = f"{score}/{max_score}"
                elif "final_response" in result and result["final_response"]:
                    prediction = str(result["final_response"])
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

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
            
            reflection_response, msg_history, _ = get_response_from_llm(
                msg=reflection_msg,
                model=self.model,
                msg_history=msg_history,
            )
            
            # Try to extract revised prediction
            try:
                extracted = _extract_jsons(msg_history[-1]["text"])
                if extracted:
                    result = extracted[-1]
                    if "final_response" in result and result["final_response"]:
                        prediction = str(result["final_response"])
                    elif "revised_score" in result and "revised_max_score" in result:
                        score = result["revised_score"]
                        max_score = result["revised_max_score"]
                        # Validate score is numeric
                        try:
                            score_val = float(score)
                            max_val = float(max_score)
                            prediction = f"{int(score_val)}/{int(max_val)}"
                        except (ValueError, TypeError):
                            prediction = f"{score}/{max_score}"
                    # Fallback: if reflection contains score/max_score but not revised_* fields
                    elif "score" in result and "max_score" in result:
                        score = result["score"]
                        max_score = result["max_score"]
                        try:
                            score_val = float(score)
                            max_val = float(max_score)
                            prediction = f"{int(score_val)}/{int(max_val)}"
                        except (ValueError, TypeError):
                            prediction = f"{score}/{max_score}"
            except Exception as e:
                self.log_fn(f"Error extracting revised prediction: {e}")

        return str(prediction), msg_history
