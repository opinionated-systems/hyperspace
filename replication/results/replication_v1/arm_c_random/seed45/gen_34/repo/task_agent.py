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
    Includes robust error recovery and nested structure handling.
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
            # Try to extract JSON from within the text if it's not pure JSON
            try:
                # Look for JSON-like content with braces using balanced brace matching
                brace_start = inner.find('{')
                if brace_start != -1:
                    # Find matching closing brace by counting
                    brace_count = 0
                    brace_end = -1
                    for i, char in enumerate(inner[brace_start:], start=brace_start):
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                brace_end = i
                                break
                    if brace_end != -1:
                        results.append(json.loads(inner[brace_start:brace_end + 1]))
            except json.JSONDecodeError:
                continue
    
    # If no <json> blocks found, try markdown code blocks
    if not results:
        # Match ```json ... ``` or just ``` ... ``` blocks
        code_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
        for match in re.finditer(code_block_pattern, text, re.DOTALL):
            try:
                content = match.group(1).strip()
                if content:
                    results.append(json.loads(content))
            except json.JSONDecodeError:
                # Try to find JSON object within the content
                try:
                    brace_start = content.find('{')
                    brace_end = content.rfind('}')
                    if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
                        results.append(json.loads(content[brace_start:brace_end + 1]))
                except json.JSONDecodeError:
                    continue
    
    # If still no results, try to find bare JSON objects with balanced braces
    if not results:
        # Look for JSON objects with proper brace balancing
        i = 0
        while i < len(text):
            if text[i] == '{':
                # Found potential start of JSON object
                brace_count = 0
                start = i
                for j, char in enumerate(text[i:], start=i):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            # Found complete JSON object
                            try:
                                obj = json.loads(text[start:j+1])
                                results.append(obj)
                                i = j + 1
                                break
                            except json.JSONDecodeError:
                                i = j + 1
                                break
                else:
                    i += 1
            else:
                i += 1
    
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
        try:
            extracted = _extract_jsons(msg_history[-1]["text"])
            if extracted:
                result = extracted[-1]
                # Validate score fields are numeric before using them
                if "response" in result and isinstance(result["response"], str):
                    # Validate response format contains score pattern
                    resp = result["response"]
                    if "/" in resp and any(c.isdigit() for c in resp):
                        prediction = resp
                    else:
                        # Try to extract from malformed response
                        self.log_fn(f"Response field malformed: {resp}")
                        prediction = "None"
                elif "score" in result and "max_score" in result:
                    try:
                        score = float(result["score"])
                        max_score = float(result["max_score"])
                        if 0 <= score <= max_score and max_score > 0:
                            prediction = f"{int(score)}/{int(max_score)}"
                        else:
                            self.log_fn(f"Invalid score range: {score}/{max_score}")
                    except (ValueError, TypeError):
                        self.log_fn(f"Non-numeric score values: {result.get('score')}/{result.get('max_score')}")
                else:
                    self.log_fn(f"JSON missing required fields: {list(result.keys())}")
            else:
                self.log_fn("No valid JSON found in response")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        # Step 2: Self-reflection to verify the grade
        if prediction != "None" and len(msg_history) >= 2:
            reflection_msg = f"""Review your grading above. Check for:
1. Did you award points the student didn't earn?
2. Did you miss any errors in the student's work?
3. Is your score consistent with the grading guidelines?
4. Would another grader agree with your assessment?

If you need to revise your grade, provide the corrected JSON with "revised_score" and "revised_max_score". 
If your grade is correct, set "revised_score" to the same value as your original score.

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
                    if "revised_score" in result and "revised_max_score" in result:
                        # Ensure scores are numeric
                        try:
                            revised_score = float(result["revised_score"])
                            revised_max = float(result["revised_max_score"])
                            # Validate score range and consistency
                            if 0 <= revised_score <= revised_max and revised_max > 0:
                                # Check if final_response is present and well-formed
                                if "final_response" in result and isinstance(result["final_response"], str):
                                    # Validate final_response format (should contain score/max_score pattern)
                                    final_resp = result["final_response"]
                                    if "/" in final_resp and any(c.isdigit() for c in final_resp):
                                        prediction = final_resp
                                    else:
                                        # Fallback to constructed response if format is invalid
                                        prediction = f"{int(revised_score)}/{int(revised_max)}"
                                        self.log_fn(f"Reflection had invalid final_response format, using constructed: {prediction}")
                                else:
                                    prediction = f"{int(revised_score)}/{int(revised_max)}"
                                self.log_fn(f"Reflection updated prediction to: {prediction}")
                            else:
                                self.log_fn(f"Reflection produced invalid score range: {revised_score}/{revised_max}")
                        except (ValueError, TypeError) as e:
                            self.log_fn(f"Reflection produced non-numeric scores: {e}")
                    else:
                        self.log_fn(f"Reflection response missing required score fields: {list(result.keys())}")
                else:
                    self.log_fn("No valid JSON found in reflection response")
            except Exception as e:
                self.log_fn(f"Error during reflection: {e}")
                # Keep original prediction on error

        return str(prediction), msg_history
