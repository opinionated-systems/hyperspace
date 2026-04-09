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
    Includes robust error recovery for malformed JSON with enhanced validation.
    
    Args:
        text: The input text containing JSON objects in various formats.
        
    Returns:
        A list of parsed JSON dictionaries, or None if no valid JSON found.
        Each result is validated to ensure proper numeric fields for scoring.
    """
    results = []
    search_from = 0
    
    # First, try to find <json>...</json> blocks (preferred format)
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
                    # Find matching closing brace by counting braces
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
                # Log the error for debugging but continue processing
                logger.debug(f"Failed to parse JSON from <json> block: {inner[:100]}...")
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
                # Try to find JSON object within the match
                try:
                    json_start = match.find("{")
                    json_end = match.rfind("}")
                    if json_start != -1 and json_end != -1 and json_end > json_start:
                        results.append(json.loads(match[json_start:json_end + 1].strip()))
                except json.JSONDecodeError:
                    continue
    
    # Last resort: try to find any JSON object in the text
    if not results:
        # Look for JSON-like patterns with improved matching
        # This pattern handles nested braces better
        json_pattern = r'\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}'
        matches = re.findall(json_pattern, text, re.DOTALL)
        for match in matches:
            try:
                parsed = json.loads(match)
                # Only include if it has expected keys
                if any(key in parsed for key in ["score", "max_score", "thinking", "rationale", "response", "final_response"]):
                    results.append(parsed)
            except json.JSONDecodeError:
                continue
    
    # Validate extracted results have required fields for grading
    validated_results = []
    for result in results:
        if isinstance(result, dict):
            # Ensure score and max_score are present and numeric if possible
            if "score" in result:
                try:
                    result["score"] = float(result["score"])
                except (ValueError, TypeError):
                    pass
            if "max_score" in result:
                try:
                    result["max_score"] = float(result["max_score"])
                except (ValueError, TypeError):
                    pass
            validated_results.append(result)
    
    return validated_results or None


# Few-shot examples for IMO grading - using proper <json> format
FEW_SHOT_EXAMPLES = """
Example 1:
Problem: Find all positive integers n such that n^2 + 3n + 2 is divisible by 4.
Solution: n^2 + 3n + 2 = (n+1)(n+2). For divisibility by 4, either n+1 or n+2 must be even, and one of them must be divisible by 4. This happens when n ≡ 0 or 3 (mod 4).
Grading Guidelines: Award 1 point for factoring, 1 point for analyzing cases, 1 point for correct answer.
Student Answer: "I factored it as (n+1)(n+2). Since these are consecutive integers, one is even. For divisibility by 4, we need one factor divisible by 4. This happens when n=3,7,11,... or n=0,4,8,... So n ≡ 0 or 3 (mod 4)."

<json>
{
    "thinking": "The student correctly factored n^2+3n+2 as (n+1)(n+2). They recognized that consecutive integers include one even number. They correctly identified that for divisibility by 4, one factor must be divisible by 4, which occurs when n ≡ 0 or 3 (mod 4). All three required components are present and correct.",
    "score": 3,
    "max_score": 3,
    "rationale": "Complete solution with correct factoring, case analysis, and answer.",
    "response": "3/3 - Complete solution with correct factoring, case analysis, and answer."
}
</json>

Example 2:
Problem: Prove that the sum of two odd numbers is even.
Solution: Let the odd numbers be 2k+1 and 2m+1. Their sum is 2k+1+2m+1 = 2(k+m+1), which is even.
Grading Guidelines: Award 1 point for setting up odd number representation, 1 point for algebraic manipulation, 1 point for conclusion.
Student Answer: "Odd numbers end in 1,3,5,7,9. Adding two odd numbers gives an even number. For example, 3+5=8 which is even."

<json>
{
    "thinking": "The student only provided examples and pattern observation without a general algebraic proof. They did not use the standard representation 2k+1 for odd numbers, did not show algebraic manipulation, and did not provide a general proof. Only 1 point awarded for demonstrating understanding that sum of odds is even.",
    "score": 1,
    "max_score": 3,
    "rationale": "Student only provided examples without general proof. Missing algebraic representation and general reasoning.",
    "response": "1/3 - Examples provided but missing general algebraic proof."
}
</json>

Example 3:
Problem: Prove that for any positive integer n, n^3 - n is divisible by 6.
Solution: n^3 - n = n(n-1)(n+1). This is the product of three consecutive integers, so it contains a multiple of 2 and a multiple of 3, making it divisible by 6.
Grading Guidelines: Award 1 point for factoring, 1 point for recognizing consecutive integers, 1 point for divisibility argument, 1 point for conclusion.
Student Answer: "n^3 - n = n(n^2-1) = n(n-1)(n+1). Among any three consecutive integers, one is divisible by 2 and one is divisible by 3. Therefore the product is divisible by 6."

<json>
{
    "thinking": "The student correctly factored n^3-n as n(n-1)(n+1), which is the product of three consecutive integers. They correctly stated that among three consecutive integers, one is divisible by 2 and one is divisible by 3. They concluded that the product is divisible by 6. All four grading criteria are met.",
    "score": 4,
    "max_score": 4,
    "rationale": "Excellent solution with correct factoring, clear recognition of consecutive integers property, and complete divisibility argument.",
    "response": "4/4 - Excellent complete solution."
}
</json>

Example 4:
Problem: Find the remainder when 2^100 is divided by 3.
Solution: Note that 2 ≡ -1 (mod 3), so 2^100 ≡ (-1)^100 ≡ 1 (mod 3). The remainder is 1.
Grading Guidelines: Award 1 point for recognizing the pattern/modular arithmetic, 1 point for correct computation, 1 point for final answer.
Student Answer: "2^1 = 2, 2^2 = 4, 2^3 = 8, 2^4 = 16... The remainders when divided by 3 are 2, 1, 2, 1... So the pattern repeats every 2. Since 100 is even, the remainder is 1."

<json>
{
    "thinking": "The student recognized the pattern of remainders (2, 1, 2, 1...) when powers of 2 are divided by 3. They correctly identified that the pattern repeats every 2. They correctly concluded that since 100 is even, the remainder is 1. While modular arithmetic would be more elegant, the student's pattern recognition approach is mathematically valid and complete.",
    "score": 3,
    "max_score": 3,
    "rationale": "Correct answer obtained through pattern recognition. While modular arithmetic would be more elegant, the student's approach is mathematically valid and complete.",
    "response": "3/3 - Correct answer via valid pattern recognition."
}
</json>
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
    "score": <numerical_score_as_number>,
    "max_score": <maximum_possible_score_as_number>,
    "rationale": "Detailed explanation of why this score was awarded",
    "response": "<score>/<max_score> - <brief summary>"
}}
</json>

IMPORTANT: score and max_score must be numeric values (integers or floats), NOT strings."""

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
                if "response" in result and result["response"]:
                    prediction = str(result["response"])
                elif "score" in result and "max_score" in result:
                    score = result["score"]
                    max_score = result["max_score"]
                    # Validate score is numeric
                    try:
                        score_val = float(score) if not isinstance(score, (int, float)) else score
                        max_val = float(max_score) if not isinstance(max_score, (int, float)) else max_score
                        prediction = f"{int(score_val)}/{int(max_val)}"
                    except (ValueError, TypeError):
                        prediction = f"{score}/{max_score}"
                elif "final_response" in result and result["final_response"]:
                    prediction = str(result["final_response"])
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        # Step 2: Self-reflection to verify the grade
        if prediction != "None" and len(msg_history) >= 2:
            reflection_msg = f"""Review your grading above carefully. Perform a critical self-check:

1. ACCURACY: Did you award points the student didn't actually earn? Check each claim against the official solution.
2. ERRORS: Did you miss any mathematical errors, logical gaps, or incorrect statements in the student's work?
3. CONSISTENCY: Is your score strictly consistent with the grading guidelines? Are you being too lenient or too harsh?
4. COMPLETENESS: Did the student address all parts of the problem? Are there missing cases or incomplete reasoning?
5. ALTERNATIVE APPROACHES: If the student used a different valid method than the official solution, did you fairly credit it?

IMPORTANT: Your revised_score MUST be a numeric value (integer or float), not a string.
Your revised_max_score MUST also be a numeric value.

Be honest and critical. If you find any issues, provide a corrected grade with clear justification.

Respond in JSON format:
<json>
{{
    "reflection": "Your detailed self-review addressing each point above",
    "revised_score": <numeric_score>,
    "revised_max_score": <numeric_max_score>,
    "final_response": "<score>/<max_score> - <brief summary of final assessment>"
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
                        # Validate score is numeric - handle both numeric and string inputs
                        try:
                            score_val = float(score) if not isinstance(score, (int, float)) else score
                            max_val = float(max_score) if not isinstance(max_score, (int, float)) else max_score
                            prediction = f"{int(score_val)}/{int(max_val)}"
                        except (ValueError, TypeError):
                            prediction = f"{score}/{max_score}"
                    # Fallback: if reflection contains score/max_score but not revised_* fields
                    elif "score" in result and "max_score" in result:
                        score = result["score"]
                        max_score = result["max_score"]
                        try:
                            score_val = float(score) if not isinstance(score, (int, float)) else score
                            max_val = float(max_score) if not isinstance(max_score, (int, float)) else max_score
                            prediction = f"{int(score_val)}/{int(max_val)}"
                        except (ValueError, TypeError):
                            prediction = f"{score}/{max_score}"
            except Exception as e:
                self.log_fn(f"Error extracting revised prediction: {e}")

        return str(prediction), msg_history
