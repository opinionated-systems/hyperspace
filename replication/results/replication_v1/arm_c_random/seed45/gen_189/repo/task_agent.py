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
    Also handles markdown code blocks as a fallback.
    Includes robust error recovery for malformed JSON with multiple fix strategies.
    """
    results = []
    search_from = 0
    
    def _try_fix_json(json_str: str) -> dict | None:
        """Try multiple strategies to fix and parse malformed JSON."""
        fixes = [
            # Strategy 1: Remove trailing commas before closing braces/brackets
            lambda s: re.sub(r',(\s*[}\]])', r'\1', s),
            # Strategy 2: Fix single quotes to double quotes (but not within values)
            lambda s: s.replace("'", '"'),
            # Strategy 3: Remove comments
            lambda s: re.sub(r'//.*?\n', '\n', s),
            # Strategy 4: Fix unquoted keys (simple cases)
            lambda s: re.sub(r'(\{|,\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', s),
        ]
        
        for fix in fixes:
            try:
                fixed = fix(json_str)
                return json.loads(fixed)
            except json.JSONDecodeError:
                continue
        
        # Try combining fixes
        try:
            fixed = json_str
            for fix in fixes[:2]:  # Most common fixes
                fixed = fix(fixed)
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass
        
        return None
    
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
            logger.debug(f"JSON decode error in <json> block: {e}")
            fixed = _try_fix_json(inner)
            if fixed:
                results.append(fixed)
                logger.debug("JSON fixed and parsed successfully")
    
    # Fallback: try markdown code blocks if no <json> blocks found
    if not results:
        # Look for ```json ... ``` blocks (multiline)
        markdown_pattern = r'```(?:json)?\s*\n(.*?)\n```'
        for match in re.finditer(markdown_pattern, text, re.DOTALL):
            content = match.group(1).strip()
            try:
                results.append(json.loads(content))
            except json.JSONDecodeError:
                fixed = _try_fix_json(content)
                if fixed:
                    results.append(fixed)
    
    # Second fallback: try inline markdown blocks ```json ... ```
    if not results:
        inline_pattern = r'```(?:json)?\s*(.*?)\s*```'
        for match in re.finditer(inline_pattern, text):
            content = match.group(1).strip()
            try:
                results.append(json.loads(content))
            except json.JSONDecodeError:
                fixed = _try_fix_json(content)
                if fixed:
                    results.append(fixed)
    
    # Last resort: try to find any JSON-like object with balanced braces
    if not results:
        # Find potential JSON starting with { and track brace balance
        i = 0
        while i < len(text):
            if text[i] == '{':
                start = i
                brace_count = 1
                i += 1
                while i < len(text) and brace_count > 0:
                    if text[i] == '{':
                        brace_count += 1
                    elif text[i] == '}':
                        brace_count -= 1
                    i += 1
                if brace_count == 0:
                    json_candidate = text[start:i]
                    try:
                        results.append(json.loads(json_candidate))
                    except json.JSONDecodeError:
                        fixed = _try_fix_json(json_candidate)
                        if fixed:
                            results.append(fixed)
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
        # Validate required inputs
        required_fields = ['problem', 'solution', 'grading_guidelines', 'student_answer']
        missing_fields = [f for f in required_fields if f not in inputs or not inputs[f]]
        if missing_fields:
            logger.error(f"Missing required fields: {missing_fields}")
            return f"Error: Missing fields {missing_fields}", []
        
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

If you need to revise your grade, provide the corrected JSON. If your grade is correct, confirm it by returning the same score.

Respond in JSON format:
<json>
{{
    "reflection": "Your detailed self-review here - analyze each point above",
    "revised_score": <numerical score>,
    "revised_max_score": <maximum possible score>,
    "final_response": "<score>/<max_score> - <brief summary of decision>"
}}
</json>

Important: revised_score must be a number, not a string."""
            
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
                    
                    # Validate and extract the final prediction
                    final_pred = None
                    if "final_response" in result and result["final_response"]:
                        final_pred = str(result["final_response"])
                        self.log_fn(f"Using 'final_response' field: {final_pred}")
                    elif "revised_score" in result and "revised_max_score" in result:
                        # Ensure scores are numeric
                        try:
                            rev_score = float(result["revised_score"])
                            rev_max = float(result["revised_max_score"])
                            final_pred = f"{int(rev_score)}/{int(rev_max)}"
                            self.log_fn(f"Using revised_score/revised_max_score: {final_pred}")
                        except (ValueError, TypeError) as ve:
                            self.log_fn(f"Invalid numeric score in reflection: {ve}")
                    
                    if final_pred:
                        prediction = final_pred
                    else:
                        self.log_fn(f"Warning: Reflection JSON missing expected fields. Keys: {list(result.keys())}")
                else:
                    self.log_fn("Warning: No JSON found in reflection response, keeping original prediction")
            except Exception as e:
                self.log_fn(f"Error extracting revised prediction: {e}")
                import traceback
                self.log_fn(f"Traceback: {traceback.format_exc()}")
                # Keep original prediction on error

        return str(prediction), msg_history
