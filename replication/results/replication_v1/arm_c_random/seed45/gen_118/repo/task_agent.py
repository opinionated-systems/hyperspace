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
    Includes enhanced error logging for debugging extraction failures.
    
    Improvements:
    - Handles nested braces correctly using brace counting
    - Supports markdown code blocks within <json> tags
    - Handles escaped characters in JSON strings
    - Validates extracted JSON has required fields for grading
    """
    results = []
    search_from = 0
    extraction_attempts = 0
    
    while True:
        start = text.find("<json>", search_from)
        if start == -1:
            break
        end = text.find("</json>", start)
        if end == -1:
            logger.warning("Found <json> tag but no closing </json> tag")
            break
        inner = text[start + 6:end].strip()
        search_from = end + 7
        extraction_attempts += 1
        
        # Try direct JSON parsing first
        try:
            parsed = json.loads(inner)
            if isinstance(parsed, dict):
                results.append(parsed)
                continue
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON from markdown code blocks within the <json> tag
        try:
            if "```json" in inner:
                json_start = inner.find("```json") + 7
                json_end = inner.find("```", json_start)
                if json_end > json_start:
                    parsed = json.loads(inner[json_start:json_end].strip())
                    if isinstance(parsed, dict):
                        results.append(parsed)
                        logger.debug(f"Successfully extracted JSON from ```json block")
                        continue
            elif "```" in inner:
                json_start = inner.find("```") + 3
                json_end = inner.find("```", json_start)
                if json_end > json_start:
                    parsed = json.loads(inner[json_start:json_end].strip())
                    if isinstance(parsed, dict):
                        results.append(parsed)
                        logger.debug(f"Successfully extracted JSON from ``` block")
                        continue
        except json.JSONDecodeError as e2:
            logger.debug(f"Failed to extract JSON from markdown block: {e2}")
    
    # If no <json> tags found or no valid JSON extracted, try to find JSON objects directly
    if not results:
        logger.debug("No <json> tags found or no valid JSON extracted, attempting direct JSON object extraction")
        # Use a more robust approach with proper brace counting
        i = 0
        while i < len(text):
            # Find the start of a potential JSON object
            if text[i] == '{':
                start_idx = i
                brace_count = 1
                in_string = False
                escape_next = False
                
                # Track through the text to find matching closing brace
                j = i + 1
                while j < len(text) and brace_count > 0:
                    char = text[j]
                    if escape_next:
                        escape_next = False
                    elif char == '\\':
                        escape_next = True
                    elif char == '"' and not escape_next:
                        in_string = not in_string
                    elif not in_string:
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                    j += 1
                
                # If we found a complete JSON object
                if brace_count == 0:
                    try:
                        json_str = text[start_idx:j]
                        parsed = json.loads(json_str)
                        if isinstance(parsed, dict):
                            results.append(parsed)
                    except json.JSONDecodeError:
                        pass
                    i = j  # Continue searching from after this object
                else:
                    i += 1  # Not a valid object, move on
                continue
            i += 1
    
    # Filter results to only include dicts with at least one grading-related field
    valid_results = []
    for r in results:
        if isinstance(r, dict):
            # Check if it has any grading-related fields
            grading_fields = ['score', 'revised_score', 'max_score', 'revised_max_score', 
                            'rationale', 'reflection', 'response', 'final_response']
            if any(field in r for field in grading_fields):
                valid_results.append(r)
            elif len(r) > 0:  # Accept any non-empty dict as fallback
                valid_results.append(r)
    
    if not valid_results:
        logger.warning(f"Failed to extract any valid JSON from text (length: {len(text)}, attempts: {extraction_attempts})")
    else:
        logger.debug(f"Successfully extracted {len(valid_results)} valid JSON object(s)")
    
    return valid_results or None


def _validate_score(result: dict, max_score: int | None = None) -> dict:
    """Validate and normalize score fields in the result.
    
    Ensures score and max_score are valid integers within reasonable bounds.
    Also validates and formats response/final_response fields to ensure they
    follow the expected "score/max_score - summary" format.
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
    
    # Build standardized response string if response fields are missing or malformed
    effective_score = validated.get("revised_score", validated.get("score", 0))
    effective_max = validated.get("revised_max_score", validated.get("max_score", max_score))
    
    # Validate and fix response field format
    response_fields = ["response", "final_response"]
    for field in response_fields:
        if field in validated:
            resp = validated[field]
            if not isinstance(resp, str):
                validated[field] = f"{effective_score}/{effective_max}"
            elif not resp.strip().startswith(f"{effective_score}/"):
                # Response doesn't start with score/max_score format
                # Check if it contains the pattern anywhere
                import re
                pattern = r"(\d+)\s*/\s*(\d+)"
                match = re.search(pattern, resp)
                if match:
                    # Extract existing score pattern and ensure it matches validated score
                    found_score, found_max = int(match.group(1)), int(match.group(2))
                    if found_score != effective_score or found_max != effective_max:
                        # Replace with correct score
                        validated[field] = f"{effective_score}/{effective_max} - {resp[match.end():].strip(' -')}"
                else:
                    # No score pattern found, prepend it
                    validated[field] = f"{effective_score}/{effective_max} - {resp}"
    
    return validated


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

        # Log thinking if available from the LLM
        if info.get("thinking"):
            self.log_fn(f"Initial grading thinking: {info['thinking'][:200]}...")

        # Extract prediction from JSON
        prediction = "None"
        try:
            extracted = _extract_jsons(msg_history[-1]["text"])
            if extracted:
                result = _validate_score(extracted[-1])
                if "response" in result:
                    prediction = result["response"]
                    self.log_fn(f"Initial grade: {prediction}")
                elif "score" in result and "max_score" in result:
                    prediction = f"{result['score']}/{result['max_score']}"
                    self.log_fn(f"Initial grade: {prediction}")
                elif "score" in result:
                    # Fallback if only score is present
                    max_score = result.get("max_score", 10)
                    prediction = f"{result['score']}/{max_score}"
                    self.log_fn(f"Initial grade: {prediction}")
            else:
                self.log_fn("No JSON extracted from initial grading response")
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
            
            try:
                reflection_response, msg_history, reflection_info = get_response_from_llm(
                    msg=reflection_msg,
                    model=self.model,
                    msg_history=msg_history,
                )
                
                # Log reflection thinking if available
                if reflection_info.get("thinking"):
                    self.log_fn(f"Reflection thinking: {reflection_info['thinking'][:200]}...")
                
                # Try to extract revised prediction
                try:
                    extracted = _extract_jsons(msg_history[-1]["text"])
                    if extracted:
                        result = _validate_score(extracted[-1])
                        if "final_response" in result:
                            prediction = result["final_response"]
                            self.log_fn(f"Grade revised after reflection: {prediction}")
                        elif "revised_score" in result and "revised_max_score" in result:
                            prediction = f"{result['revised_score']}/{result['revised_max_score']}"
                            self.log_fn(f"Grade revised after reflection: {prediction}")
                        elif "revised_score" in result:
                            # Fallback if only revised_score is present
                            max_score = result.get("revised_max_score", 10)
                            prediction = f"{result['revised_score']}/{max_score}"
                            self.log_fn(f"Grade revised after reflection: {prediction}")
                        else:
                            self.log_fn("Reflection completed but no grade revision found")
                    else:
                        self.log_fn("No JSON extracted from reflection response")
                except Exception as e:
                    self.log_fn(f"Error extracting revised prediction: {e}")
            except Exception as e:
                self.log_fn(f"Error during reflection step: {e}")

        return str(prediction), msg_history
