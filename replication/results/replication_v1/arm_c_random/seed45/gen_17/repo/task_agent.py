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
    Also handles markdown code blocks and inline JSON as fallbacks.
    Includes robust error recovery for common JSON formatting issues.
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
            # Try to fix common JSON issues before giving up
            fixed_json = _attempt_json_repair(inner)
            if fixed_json:
                results.append(fixed_json)
            else:
                logger.debug(f"JSON decode error in <json> block: {e}")
            continue
    
    # Fallback 1: try markdown code blocks if no <json> blocks found
    if not results:
        markdown_pattern = r'```(?:json)?\s*\n(.*?)\n```'
        for match in re.finditer(markdown_pattern, text, re.DOTALL):
            try:
                results.append(json.loads(match.group(1).strip()))
            except json.JSONDecodeError:
                # Try repair on markdown blocks too
                fixed_json = _attempt_json_repair(match.group(1).strip())
                if fixed_json:
                    results.append(fixed_json)
                continue
    
    # Fallback 2: look for inline JSON objects (e.g., {"key": "value"})
    if not results:
        # Find JSON-like structures with balanced braces
        inline_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        for match in re.finditer(inline_pattern, text, re.DOTALL):
            try:
                candidate = match.group(0).strip()
                # Only accept if it looks like a grading result
                if '"score"' in candidate or '"response"' in candidate:
                    results.append(json.loads(candidate))
            except json.JSONDecodeError:
                fixed_json = _attempt_json_repair(match.group(0).strip())
                if fixed_json and ('score' in fixed_json or 'response' in fixed_json):
                    results.append(fixed_json)
                continue
    
    return results or None


def _attempt_json_repair(json_str: str) -> dict | None:
    """Attempt to repair common JSON formatting issues.
    
    Fixes:
    - Trailing commas before closing braces/brackets
    - Single quotes instead of double quotes
    - Unescaped newlines in strings
    - Missing quotes around keys
    """
    import re
    
    original = json_str.strip()
    repaired = original
    
    # Fix 1: Remove trailing commas before } or ]
    repaired = re.sub(r',\s*\}', '}', repaired)
    repaired = re.sub(r',\s*\]', ']', repaired)
    
    # Fix 2: Replace single quotes with double quotes (carefully)
    # Only replace if not inside a string
    repaired = repaired.replace("'", '"')
    
    # Fix 3: Escape unescaped newlines in string values
    # This is a simplified fix - replace newlines between quotes
    repaired = re.sub(r'("[^"]*?)\n([^"]*?")', r'\1\\n\2', repaired)
    
    # Fix 4: Try to handle missing quotes around keys
    # Match word: followed by space or value
    repaired = re.sub(r'(\w+):\s*"', r'"\1": "', repaired)
    repaired = re.sub(r'(\w+):\s*\{', r'"\1": {', repaired)
    repaired = re.sub(r'(\w+):\s*\[', r'"\1": [', repaired)
    repaired = re.sub(r'(\w+):\s*(\d)', r'"\1": \2', repaired)
    
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        return None


# Few-shot examples for IMO grading
FEW_SHOT_EXAMPLES = """
Example 1:
Problem: Find all positive integers n such that n^2 + 3n + 2 is divisible by 4.
Solution: n^2 + 3n + 2 = (n+1)(n+2). For divisibility by 4, either n+1 or n+2 must be even, and one of them must be divisible by 4. This happens when n ≡ 0 or 3 (mod 4).
Grading Guidelines: Award 1 point for factoring, 1 point for analyzing cases, 1 point for correct answer.
Student Answer: "I factored it as (n+1)(n+2). Since these are consecutive integers, one is even. For divisibility by 4, we need one of them divisible by 4, so n ≡ 0 or 3 (mod 4)."
Grade: 3/3 - Correct factoring, proper case analysis, correct answer.

Example 2:
Problem: Prove that for any prime p > 3, p^2 ≡ 1 (mod 24).
Solution: Any prime p > 3 is odd and not divisible by 3. So p ≡ ±1 (mod 8) giving p^2 ≡ 1 (mod 8), and p ≡ ±1 (mod 3) giving p^2 ≡ 1 (mod 3). By CRT, p^2 ≡ 1 (mod 24).
Grading Guidelines: 1 point for showing p^2 ≡ 1 (mod 8), 1 point for showing p^2 ≡ 1 (mod 3), 1 point for combining via CRT.
Student Answer: "Since p is odd, p = 2k+1, so p^2 = 4k(k+1)+1. Since k(k+1) is even, p^2 ≡ 1 (mod 8). Since p is not divisible by 3, p^2 ≡ 1 (mod 3). Therefore p^2 ≡ 1 (mod 24)."
Grade: 3/3 - Correct modular arithmetic, proper reasoning for both moduli, correct conclusion.
"""


class TaskAgent:
    """Task agent that grades IMO problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, temperature: float = 0.0) -> None:
        self.model = model
        self.temperature = temperature
        self.log_fn = logger.info

    def forward(
        self,
        problem: str,
        solution: str,
        student_answer: str,
        grading_guidelines: str,
        max_score: int,
    ) -> tuple[str, list[dict]]:
        """Grade a student answer for an IMO problem.

        Args:
            problem: The problem statement
            solution: The reference solution
            student_answer: The student's answer to grade
            grading_guidelines: Instructions for how to grade
            max_score: Maximum possible score

        Returns:
            (prediction, msg_history) where prediction is the grade string
        """
        # Step 1: Initial grading with chain-of-thought
        instruction = f"""You are an expert mathematics grader for the International Mathematical Olympiad (IMO).

Your task is to grade a student's solution to a mathematics problem.

{FEW_SHOT_EXAMPLES}

Now grade this submission:

Problem: {problem}

Solution: {solution}

Grading Guidelines: {grading_guidelines}

Student Answer: {student_answer}

Please follow these steps:
1. Analyze the student's solution step by step
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

        # Extract prediction from JSON with detailed logging
        prediction = "None"
        try:
            last_msg = msg_history[-1]["text"]
            extracted = _extract_jsons(last_msg)
            if extracted:
                result = extracted[-1]
                self.log_fn(f"Extracted JSON result: {result}")
                if "response" in result:
                    prediction = result["response"]
                    self.log_fn(f"Using 'response' field: {prediction}")
                elif "score" in result and "max_score" in result:
                    prediction = f"{result['score']}/{result['max_score']}"
                    self.log_fn(f"Using score/max_score fields: {prediction}")
                else:
                    self.log_fn(f"Warning: JSON missing expected fields. Keys: {list(result.keys())}")
            else:
                self.log_fn("Warning: No JSON blocks found in response")
                # Try to extract any numeric score pattern as fallback
                import re
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
            
            try:
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
                        if "final_response" in result:
                            prediction = result["final_response"]
                            self.log_fn(f"Using 'final_response' field: {prediction}")
                        elif "revised_score" in result and "revised_max_score" in result:
                            prediction = f"{result['revised_score']}/{result['revised_max_score']}"
                            self.log_fn(f"Using revised_score/revised_max_score: {prediction}")
                        elif "score" in result and "max_score" in result:
                            # Fallback to standard fields if revision fields missing
                            prediction = f"{result['score']}/{result['max_score']}"
                            self.log_fn(f"Using score/max_score fields: {prediction}")
                        else:
                            self.log_fn(f"Warning: Reflection JSON missing expected fields. Keys: {list(result.keys())}")
                    else:
                        self.log_fn("Warning: No JSON found in reflection response, keeping original prediction")
                except Exception as e:
                    self.log_fn(f"Error extracting revised prediction: {e}")
                    # Keep original prediction on error
                    pass
            except Exception as e:
                self.log_fn(f"Error during reflection LLM call: {e}")
                # Keep original prediction if reflection fails
                pass

        return str(prediction), msg_history
