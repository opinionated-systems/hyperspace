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
            # Try to extract JSON from within the content
            try:
                # Look for JSON object pattern
                json_match = re.search(r'\{[\s\S]*?"score"[\s\S]*?\}', inner)
                if json_match:
                    results.append(json.loads(json_match.group()))
            except (json.JSONDecodeError, AttributeError):
                continue
    
    # If no <json> blocks found, try markdown code blocks
    if not results:
        code_block_pattern = r'```(?:json)?\s*([\s\S]*?)```'
        for match in re.finditer(code_block_pattern, text):
            try:
                content = match.group(1).strip()
                if content:
                    results.append(json.loads(content))
            except json.JSONDecodeError:
                continue
    
    # If still no results, try to find bare JSON objects
    if not results:
        # Look for JSON objects with expected keys
        json_pattern = r'\{\s*"(?:thinking|score|rationale|response|reflection|revised_score)"[\s\S]*?\}'
        for match in re.finditer(json_pattern, text):
            try:
                results.append(json.loads(match.group()))
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

Example 2 - Partial Credit:
Problem: Prove that the sum of two odd numbers is even.
Solution: Let the odd numbers be 2k+1 and 2m+1. Their sum is 2k+1+2m+1 = 2(k+m+1), which is even.
Grading Guidelines: Award 1 point for setting up odd number representation, 1 point for algebraic manipulation, 1 point for conclusion.
Student Answer: "Odd numbers end in 1,3,5,7,9. Adding two odd numbers gives an even number. For example, 3+5=8 which is even."
Grade: {"score": 1, "max_score": 3, "rationale": "Student only provided examples without general proof. Missing algebraic representation and general reasoning."}

Example 3 - Zero Credit:
Problem: Solve x^2 - 5x + 6 = 0.
Solution: Factor as (x-2)(x-3) = 0, so x = 2 or x = 3.
Grading Guidelines: Award 1 point for correct factoring, 1 point for finding both roots.
Student Answer: "x = 5"
Grade: {"score": 0, "max_score": 2, "rationale": "Student gave incorrect answer with no work shown. No credit awarded."}

Example 4 - Full Credit with Different Approach:
Problem: Prove the Pythagorean theorem for right triangles.
Solution: Standard geometric proof using similar triangles.
Grading Guidelines: Award points for valid proof structure, correct reasoning, and conclusion.
Student Answer: "Using the area method: Arrange four copies of the right triangle to form a square with side c. The area is c². It's also equal to 4*(ab/2) + (a-b)² = 2ab + a² - 2ab + b² = a² + b². Therefore c² = a² + b²."
Grade: {"score": 3, "max_score": 3, "rationale": "Valid alternative proof using area method. Correct reasoning and conclusion."}
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

GRADING PRINCIPLES:
1. Award points only for correct mathematical reasoning and results
2. Partial credit is appropriate when some steps are correct but incomplete
3. Alternative valid approaches should receive full credit if mathematically sound
4. No credit for answers without supporting work or reasoning
5. Be consistent with the official solution and grading guidelines

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
1. Carefully read the student's answer and identify all mathematical claims made
2. Compare each claim against the official solution
3. Check for any errors, logical gaps, or missing steps
4. Verify the student didn't make any unjustified assumptions
5. Determine the score based strictly on the grading guidelines
6. Provide a clear rationale explaining each point awarded or deducted

Respond ONLY in JSON format with the following schema:
<json>
{{
    "thinking": "Your detailed step-by-step analysis here - be thorough and specific",
    "score": <numerical score as integer>,
    "max_score": <maximum possible score as integer>,
    "rationale": "Detailed explanation of why this exact score was awarded - reference specific grading criteria",
    "response": "<score>/<max_score> - <brief summary of the grade>"
}}
</json>

IMPORTANT: Ensure your JSON is valid and complete. The score must be an integer between 0 and max_score inclusive."""

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
                # Prefer "response" field if available and non-empty
                if "response" in result and result["response"]:
                    prediction = result["response"]
                elif "score" in result and "max_score" in result:
                    # Validate score values are numeric
                    try:
                        score = float(result["score"])
                        max_score = float(result["max_score"])
                        if max_score > 0 and 0 <= score <= max_score:
                            prediction = f"{int(score)}/{int(max_score)}"
                        else:
                            prediction = f"{result['score']}/{result['max_score']}"
                    except (ValueError, TypeError):
                        prediction = f"{result['score']}/{result['max_score']}"
                else:
                    # Try to construct prediction from available fields
                    score = result.get("score", result.get("revised_score", "?"))
                    max_score = result.get("max_score", result.get("revised_max_score", "?"))
                    if score != "?" and max_score != "?":
                        prediction = f"{score}/{max_score}"
            else:
                self.log_fn("No JSON found in response, using fallback")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        # Step 2: Self-reflection to verify the grade
        if prediction != "None" and len(msg_history) >= 2:
            reflection_msg = f"""Review your grading above. Check for:
1. Did you award points the student didn't earn?
2. Did you miss any errors in the student's work?
3. Is your score consistent with the grading guidelines?
4. Would another grader agree with your assessment?

If you need to revise your grade, provide the corrected JSON with revised_score and revised_max_score.
If your grade is correct, set revised_score and revised_max_score to match your original grade.

Respond in JSON format:
<json>
{{
    "reflection": "Your detailed self-review here - analyze each point above",
    "revised_score": <numerical score>,
    "revised_max_score": <maximum possible score>,
    "final_response": "<score>/<max_score> - <brief summary of decision>"
}}
</json>"""
            
            try:
                reflection_response, msg_history, _ = get_response_from_llm(
                    msg=reflection_msg,
                    model=self.model,
                    msg_history=msg_history,
                )
                
                # Try to extract revised prediction
                extracted = _extract_jsons(msg_history[-1]["text"])
                if extracted:
                    result = extracted[-1]
                    # Validate the extracted result has required fields
                    if "final_response" in result and result["final_response"]:
                        prediction = result["final_response"]
                    elif "revised_score" in result and "revised_max_score" in result:
                        # Validate score is numeric
                        try:
                            score = float(result["revised_score"])
                            max_score = float(result["revised_max_score"])
                            if max_score > 0 and 0 <= score <= max_score:
                                prediction = f"{int(score)}/{int(max_score)}"
                            else:
                                self.log_fn(f"Invalid score range: {score}/{max_score}")
                        except (ValueError, TypeError):
                            self.log_fn(f"Non-numeric score values: {result.get('revised_score')}/{result.get('revised_max_score')}")
            except Exception as e:
                self.log_fn(f"Error during reflection: {e}")
                # Keep original prediction if reflection fails

        return str(prediction), msg_history
