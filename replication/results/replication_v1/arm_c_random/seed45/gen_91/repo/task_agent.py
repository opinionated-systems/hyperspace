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
        
        try:
            results.append(json.loads(inner))
        except json.JSONDecodeError as e:
            logger.debug(f"JSON decode error in <json> block {extraction_attempts}: {e}")
            # Try to extract JSON from markdown code blocks
            try:
                if "```json" in inner:
                    json_start = inner.find("```json") + 7
                    json_end = inner.find("```", json_start)
                    if json_end > json_start:
                        results.append(json.loads(inner[json_start:json_end].strip()))
                        logger.debug(f"Successfully extracted JSON from ```json block")
                elif "```" in inner:
                    json_start = inner.find("```") + 3
                    json_end = inner.find("```", json_start)
                    if json_end > json_start:
                        results.append(json.loads(inner[json_start:json_end].strip()))
                        logger.debug(f"Successfully extracted JSON from ``` block")
            except json.JSONDecodeError as e2:
                logger.debug(f"Failed to extract JSON from markdown block: {e2}")
                continue
    
    # If no <json> tags found, try to find JSON objects directly
    if not results:
        logger.debug("No <json> tags found, attempting direct JSON object extraction")
        # Look for JSON objects in the text
        brace_count = 0
        start_idx = -1
        for i, char in enumerate(text):
            if char == '{':
                if brace_count == 0:
                    start_idx = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start_idx != -1:
                    try:
                        results.append(json.loads(text[start_idx:i+1]))
                        logger.debug(f"Successfully extracted JSON object at position {start_idx}")
                    except json.JSONDecodeError:
                        pass
                    start_idx = -1
    
    if not results:
        logger.warning(f"Failed to extract any valid JSON from text (length: {len(text)}, attempts: {extraction_attempts})")
    else:
        logger.debug(f"Successfully extracted {len(results)} JSON object(s)")
    
    return results or None


def _validate_score(result: dict, max_score: int | None = None) -> dict:
    """Validate and normalize score fields in the result.
    
    Ensures score and max_score are valid integers within reasonable bounds.
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

        # Extract prediction from JSON
        prediction = "None"
        try:
            extracted = _extract_jsons(msg_history[-1]["text"])
            if extracted:
                result = _validate_score(extracted[-1])
                if "response" in result:
                    prediction = result["response"]
                elif "score" in result and "max_score" in result:
                    prediction = f"{result['score']}/{result['max_score']}"
                elif "score" in result:
                    # Fallback if only score is present
                    max_score = result.get("max_score", 10)
                    prediction = f"{result['score']}/{max_score}"
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
                            self.log_fn(f"Grade confirmed/revised to: {prediction}")
                        elif "revised_score" in result and "revised_max_score" in result:
                            prediction = f"{result['revised_score']}/{result['revised_max_score']}"
                            self.log_fn(f"Grade revised to: {prediction}")
                        elif "revised_score" in result:
                            # Fallback if only revised_score is present
                            max_score = result.get("revised_max_score", 10)
                            prediction = f"{result['revised_score']}/{max_score}"
                            self.log_fn(f"Grade revised (fallback): {prediction}")
                    else:
                        self.log_fn("No JSON extracted from reflection, keeping original grade")
                except Exception as e:
                    self.log_fn(f"Error extracting revised prediction: {e}")
            except Exception as e:
                self.log_fn(f"Error during reflection step: {e}")
                # Keep original prediction if reflection fails

        return str(prediction), msg_history
