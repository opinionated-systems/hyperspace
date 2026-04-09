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
from typing import Callable

from agent.llm_client import get_response_from_llm, EVAL_MODEL

logger = logging.getLogger(__name__)


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    Also handles markdown code blocks and bare JSON objects.
    Includes robust error recovery and validation.
    """
    results = []
    search_from = 0
    extraction_attempts = []
    
    # First try to find <json>...</json> blocks
    while True:
        start = text.find("<json>", search_from)
        if start == -1:
            break
        end = text.find("</json>", start)
        if end == -1:
            extraction_attempts.append(("<json> block", start, "unclosed tag"))
            break
        inner = text[start + 6:end].strip()
        search_from = end + 7
        
        # Try multiple parsing strategies for robustness
        parsed = _try_parse_json(inner)
        if parsed is not None:
            results.append(parsed)
            extraction_attempts.append(("<json> block", start, "success"))
        else:
            extraction_attempts.append(("<json> block", start, "parse error after all strategies"))
    
    # If no <json> blocks found, try markdown code blocks
    if not results:
        # Match ```json ... ``` or just ``` ... ``` blocks
        code_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
        for match in re.finditer(code_block_pattern, text, re.DOTALL):
            try:
                content = match.group(1).strip()
                if content:
                    parsed = _try_parse_json(content)
                    if parsed is not None:
                        results.append(parsed)
                        extraction_attempts.append(("markdown code block", match.start(), "success"))
                    else:
                        extraction_attempts.append(("markdown code block", match.start(), "parse error"))
            except Exception as e:
                extraction_attempts.append(("markdown code block", match.start(), f"error: {e}"))
                continue
    
    # If still no results, try to find bare JSON objects using a stack-based approach
    if not results:
        bare_results = _extract_bare_jsons(text)
        if bare_results:
            results.extend(bare_results)
            extraction_attempts.append(("bare JSON", 0, f"found {len(bare_results)} objects"))
        else:
            extraction_attempts.append(("bare JSON", 0, "no objects found"))
    
    # Log extraction summary for debugging
    if not results:
        logger.debug(f"JSON extraction failed. Attempts: {extraction_attempts}")
    else:
        logger.debug(f"JSON extraction succeeded with {len(results)} results from {len(extraction_attempts)} attempts")
    
    return results or None


def _try_parse_json(text: str) -> dict | None:
    """Try multiple strategies to parse JSON from text.
    
    Returns the parsed dict or None if all strategies fail.
    """
    # Strategy 1: Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Strategy 2: Extract from braces (handles extra text before/after JSON)
    try:
        brace_start = text.find('{')
        brace_end = text.rfind('}')
        if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
            return json.loads(text[brace_start:brace_end + 1])
    except json.JSONDecodeError:
        pass
    
    # Strategy 3: Handle common LLM formatting issues
    # Remove trailing commas before closing braces/brackets
    try:
        cleaned = re.sub(r',(\s*[}\]])', r'\1', text)
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    
    # Strategy 4: Try with unicode escape fixes
    try:
        # Replace common problematic escape sequences
        cleaned = text.replace('\\"', '"').replace("\\'", "'")
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    
    return None


def _validate_and_format_prediction(result: dict, log_fn: Callable) -> str:
    """Validate prediction result and format it consistently.
    
    Args:
        result: The parsed JSON result dict
        log_fn: Logging function for warnings
        
    Returns:
        Formatted prediction string or "None" if invalid
    """
    # Check for direct response field first
    if "response" in result and isinstance(result["response"], str):
        return result["response"]
    
    # Validate and format score/max_score
    if "score" in result and "max_score" in result:
        try:
            score = float(result["score"])
            max_score = float(result["max_score"])
            if 0 <= score <= max_score:
                return f"{int(score)}/{int(max_score)}"
            else:
                log_fn(f"Invalid score range: {score}/{max_score}")
        except (ValueError, TypeError):
            log_fn(f"Non-numeric score values: {result.get('score')}/{result.get('max_score')}")
    else:
        log_fn(f"JSON missing required fields: {list(result.keys())}")
    
    return "None"


def _validate_reflection_prediction(result: dict, log_fn: Callable) -> str:
    """Validate reflection result and format prediction consistently.
    
    Args:
        result: The parsed JSON result dict from reflection
        log_fn: Logging function for warnings
        
    Returns:
        Formatted prediction string or "None" if invalid
    """
    # Check for final_response field first
    if "final_response" in result and isinstance(result["final_response"], str):
        return result["final_response"]
    
    # Validate revised_score and revised_max_score
    if "revised_score" in result and "revised_max_score" in result:
        try:
            revised_score = float(result["revised_score"])
            revised_max = float(result["revised_max_score"])
            if 0 <= revised_score <= revised_max:
                return f"{int(revised_score)}/{int(revised_max)}"
            else:
                log_fn(f"Reflection produced invalid score range: {revised_score}/{revised_max}")
        except (ValueError, TypeError):
            log_fn(f"Reflection produced non-numeric scores")
    else:
        log_fn("Reflection response missing required score fields")
    
    return "None"


def _extract_bare_jsons(text: str) -> list[dict]:
    """Extract bare JSON objects from text using stack-based brace matching.
    
    This is more robust than regex for handling nested JSON objects.
    """
    results = []
    i = 0
    n = len(text)
    
    while i < n:
        # Find the start of a potential JSON object
        if text[i] == '{':
            # Use stack to find matching closing brace
            stack = 1
            j = i + 1
            in_string = False
            escape_next = False
            
            while j < n and stack > 0:
                char = text[j]
                if escape_next:
                    escape_next = False
                elif char == '\\' and in_string:
                    escape_next = True
                elif char == '"' and not in_string:
                    in_string = True
                elif char == '"' and in_string:
                    in_string = False
                elif not in_string:
                    if char == '{':
                        stack += 1
                    elif char == '}':
                        stack -= 1
                j += 1
            
            if stack == 0:
                # Found a complete JSON object
                json_str = text[i:j]
                try:
                    results.append(json.loads(json_str))
                except json.JSONDecodeError:
                    pass
            i = j
        else:
            i += 1
    
    return results


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
                prediction = _validate_and_format_prediction(result, self.log_fn)
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
                    # Use validation helper for consistency
                    revised_prediction = _validate_reflection_prediction(result, self.log_fn)
                    if revised_prediction != "None":
                        prediction = revised_prediction
                        self.log_fn(f"Reflection updated prediction to: {prediction}")
            except Exception as e:
                self.log_fn(f"Error during reflection: {e}")
                # Keep original prediction on error

        return str(prediction), msg_history
