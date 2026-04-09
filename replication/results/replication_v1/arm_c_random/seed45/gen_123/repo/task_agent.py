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
from typing import Any

from agent.llm_client import get_response_from_llm, EVAL_MODEL

logger = logging.getLogger(__name__)


def _find_balanced_braces(text: str, start: int = 0) -> tuple[int, int] | None:
    """Find a balanced pair of braces starting from position.
    
    Returns (start, end) indices of the outermost braces, or None if not found.
    """
    json_start = text.find("{", start)
    if json_start == -1:
        return None
    
    brace_count = 0
    for i, char in enumerate(text[json_start:], start=json_start):
        if char == '{':
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0:
                return (json_start, i)
    return None


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    Also handles markdown code blocks and bare JSON objects.
    Includes robust error recovery for malformed JSON.
    
    Improved to handle multiple JSON blocks and nested structures better.
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
            brace_range = _find_balanced_braces(inner)
            if brace_range:
                try:
                    results.append(json.loads(inner[brace_range[0]:brace_range[1] + 1]))
                except json.JSONDecodeError:
                    # Try to clean up common JSON issues
                    cleaned = _clean_json_string(inner[brace_range[0]:brace_range[1] + 1])
                    try:
                        results.append(json.loads(cleaned))
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
                # Try to find JSON object within the match
                brace_range = _find_balanced_braces(match)
                if brace_range:
                    try:
                        results.append(json.loads(match[brace_range[0]:brace_range[1] + 1]))
                    except json.JSONDecodeError:
                        # Try to clean up common JSON issues
                        cleaned = _clean_json_string(match[brace_range[0]:brace_range[1] + 1])
                        try:
                            results.append(json.loads(cleaned))
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
                # Try to clean up common JSON issues
                cleaned = _clean_json_string(match)
                try:
                    parsed = json.loads(cleaned)
                    if any(key in parsed for key in ["score", "max_score", "thinking", "rationale", "response", "final_response"]):
                        results.append(parsed)
                except json.JSONDecodeError:
                    continue
    
    return results or None


def _clean_json_string(text: str) -> str:
    """Clean up common JSON formatting issues.
    
    Handles trailing commas, unescaped newlines in strings, and other
    common issues that LLMs produce in JSON output.
    """
    # Remove trailing commas before closing braces/brackets
    text = re.sub(r',(\s*[}\]])', r'\1', text)
    
    # Replace unescaped newlines in string values with escaped newlines
    # This is a simplified approach - handles common cases
    result = []
    in_string = False
    i = 0
    while i < len(text):
        char = text[i]
        if char == '"' and (i == 0 or text[i-1] != '\\'):
            in_string = not in_string
            result.append(char)
        elif char == '\n' and in_string:
            result.append('\\n')
        elif char == '\t' and in_string:
            result.append('\\t')
        else:
            result.append(char)
        i += 1
    
    return ''.join(result)


def _format_score_prediction(result: dict) -> str:
    """Format a prediction from a result dictionary.
    
    Extracts score/max_score or response fields to create a formatted prediction string.
    """
    # Priority: response > final_response > score/max_score
    if "response" in result and result["response"]:
        return str(result["response"])
    
    if "final_response" in result and result["final_response"]:
        return str(result["final_response"])
    
    if "score" in result and "max_score" in result:
        try:
            score_val = float(result["score"])
            max_val = float(result["max_score"])
            return f"{int(score_val)}/{int(max_val)}"
        except (ValueError, TypeError):
            return f"{result['score']}/{result['max_score']}"
    
    return "None"


def _extract_prediction_from_history(msg_history: list[dict]) -> str:
    """Extract prediction from the last message in history.
    
    Returns the formatted prediction string or "None" if extraction fails.
    """
    if not msg_history:
        return "None"
    
    try:
        last_text = msg_history[-1].get("text", "")
        extracted = _extract_jsons(last_text)
        if extracted:
            return _format_score_prediction(extracted[-1])
    except Exception as e:
        logger.debug(f"Error extracting prediction: {e}")
    
    return "None"


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
Problem: Prove that for any positive integer n, n^3 - n is divisible by 6.
Solution: n^3 - n = n(n-1)(n+1). This is the product of three consecutive integers, so it contains a multiple of 2 and a multiple of 3, making it divisible by 6.
Grading Guidelines: Award 1 point for factoring, 1 point for recognizing consecutive integers, 1 point for divisibility argument, 1 point for conclusion.
Student Answer: "n^3 - n = n(n^2-1) = n(n-1)(n+1). Among any three consecutive integers, one is divisible by 2 and one is divisible by 3. Therefore the product is divisible by 6."
Grade: {"score": 4, "max_score": 4, "rationale": "Excellent solution with correct factoring, clear recognition of consecutive integers property, and complete divisibility argument."}

Example 4:
Problem: Find the remainder when 2^100 is divided by 3.
Solution: Note that 2 ≡ -1 (mod 3), so 2^100 ≡ (-1)^100 ≡ 1 (mod 3). The remainder is 1.
Grading Guidelines: Award 1 point for recognizing the pattern/modular arithmetic, 1 point for correct computation, 1 point for final answer.
Student Answer: "2^1 = 2, 2^2 = 4, 2^3 = 8, 2^4 = 16... The remainders when divided by 3 are 2, 1, 2, 1... So the pattern repeats every 2. Since 100 is even, the remainder is 1."
Grade: {"score": 3, "max_score": 3, "rationale": "Correct answer obtained through pattern recognition. While modular arithmetic would be more elegant, the student's approach is mathematically valid and complete."}
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
        prediction = _extract_prediction_from_history(msg_history)

        # Step 2: Self-reflection to verify the grade
        if prediction != "None" and len(msg_history) >= 2:
            reflection_msg = f"""Review your grading above carefully. Perform a critical self-check:

1. ACCURACY: Did you award points the student didn't actually earn? Check each claim against the official solution.
2. ERRORS: Did you miss any mathematical errors, logical gaps, or incorrect statements in the student's work?
3. CONSISTENCY: Is your score strictly consistent with the grading guidelines? Are you being too lenient or too harsh?
4. COMPLETENESS: Did the student address all parts of the problem? Are there missing cases or incomplete reasoning?
5. ALTERNATIVE APPROACHES: If the student used a different valid method than the official solution, did you fairly credit it?

Be honest and critical. If you find any issues, provide a corrected grade with clear justification.

Respond in JSON format:
<json>
{{
    "reflection": "Your detailed self-review addressing each point above",
    "revised_score": <score>,
    "revised_max_score": <max_score>,
    "final_response": "<score>/<max_score> - <brief summary of final assessment>"
}}
</json>"""
            
            reflection_response, msg_history, _ = get_response_from_llm(
                msg=reflection_msg,
                model=self.model,
                msg_history=msg_history,
            )
            
            # Try to extract revised prediction
            revised_prediction = _extract_prediction_from_history(msg_history)
            if revised_prediction != "None":
                prediction = revised_prediction

        return str(prediction), msg_history
