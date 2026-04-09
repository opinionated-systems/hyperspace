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


def _clean_json_string(json_str: str) -> str:
    """Clean and normalize a JSON string for parsing.
    
    Removes common formatting issues that cause JSON parsing to fail,
    such as trailing commas, extra whitespace, and control characters.
    """
    # Remove null bytes and control characters
    cleaned = json_str.replace('\x00', '').replace('\x01', '').replace('\x02', '')
    
    # Remove trailing commas before closing braces/brackets
    cleaned = re.sub(r',(\s*[}\]])', r'\1', cleaned)
    
    # Normalize line endings
    cleaned = cleaned.replace('\r\n', '\n').replace('\r', '\n')
    
    return cleaned.strip()


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
            cleaned = _clean_json_string(inner)
            results.append(json.loads(cleaned))
        except json.JSONDecodeError as e:
            logger.debug(f"JSON decode error in <json> block {extraction_attempts}: {e}")
            # Try to extract JSON from markdown code blocks
            try:
                if "```json" in inner:
                    json_start = inner.find("```json") + 7
                    json_end = inner.find("```", json_start)
                    if json_end > json_start:
                        cleaned = _clean_json_string(inner[json_start:json_end].strip())
                        results.append(json.loads(cleaned))
                        logger.debug(f"Successfully extracted JSON from ```json block")
                elif "```" in inner:
                    json_start = inner.find("```") + 3
                    json_end = inner.find("```", json_start)
                    if json_end > json_start:
                        cleaned = _clean_json_string(inner[json_start:json_end].strip())
                        results.append(json.loads(cleaned))
                        logger.debug(f"Successfully extracted JSON from ``` block")
            except json.JSONDecodeError as e2:
                logger.debug(f"Failed to extract JSON from markdown block: {e2}")
                continue
    
    # If no <json> tags found, try to find JSON objects directly
    if not results:
        logger.debug("No <json> tags found, attempting direct JSON object extraction")
        # Look for JSON objects in the text with improved brace matching
        brace_count = 0
        start_idx = -1
        in_string = False
        escape_next = False
        
        for i, char in enumerate(text):
            if escape_next:
                escape_next = False
                continue
            if char == '\\' and in_string:
                escape_next = True
                continue
            if char == '"' and (i == 0 or text[i-1] != '\\'):
                in_string = not in_string
                continue
            if in_string:
                continue
                
            if char == '{':
                if brace_count == 0:
                    start_idx = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start_idx != -1:
                    try:
                        json_str = text[start_idx:i+1]
                        cleaned = _clean_json_string(json_str)
                        results.append(json.loads(cleaned))
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
    Returns a normalized result dict with enhanced validation and safety checks.
    """
    validated = dict(result)
    
    # Extract max_score from result or use provided default
    if max_score is None:
        max_score = validated.get("max_score", validated.get("revised_max_score", 10))
    
    try:
        max_score = int(max_score)
        # Ensure max_score is reasonable (at least 1, at most 1000)
        max_score = max(1, min(max_score, 1000))
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
                field_value = int(validated[field])
                # Ensure max_score is at least 1 and at most 1000
                validated[field] = max(1, min(field_value, 1000))
            except (ValueError, TypeError):
                validated[field] = max_score
    
    # Ensure score never exceeds its corresponding max_score
    if "score" in validated and "max_score" in validated:
        validated["score"] = min(validated["score"], validated["max_score"])
    if "revised_score" in validated and "revised_max_score" in validated:
        validated["revised_score"] = min(validated["revised_score"], validated["revised_max_score"])
    elif "revised_score" in validated and "max_score" in validated:
        validated["revised_score"] = min(validated["revised_score"], validated["max_score"])
    
    return validated


# Few-shot examples for IMO grading with detailed reasoning
FEW_SHOT_EXAMPLES = """
Example 1 - Correct Answer:
Problem: Find all positive integers n such that n^2 + 3n + 2 is divisible by 4.
Solution: n^2 + 3n + 2 = (n+1)(n+2). For divisibility by 4, either n+1 or n+2 must be even, and one of them must be divisible by 4. This happens when n ≡ 0 or 3 (mod 4).
Grading Guidelines: Award 1 point for factoring, 1 point for analyzing cases, 1 point for correct answer.
Student Answer: "I factored it as (n+1)(n+2). Since these are consecutive integers, one is even. For divisibility by 4, we need one factor divisible by 4. This happens when n=3,7,11,... or n=0,4,8,... So n ≡ 0 or 3 (mod 4)."

Analysis:
- Student correctly factored n^2 + 3n + 2 = (n+1)(n+2) ✓
- Student correctly identified that consecutive integers include one even number ✓
- Student correctly identified the pattern for divisibility by 4 ✓
- Student gave the correct final answer n ≡ 0 or 3 (mod 4) ✓

Grade: {"score": 3, "max_score": 3, "rationale": "Complete solution with correct factoring, case analysis, and answer. All three components from grading guidelines are present and correct.", "response": "3/3 - Complete and correct solution"}

Example 2 - Partial Answer:
Problem: Prove that the sum of two odd numbers is even.
Solution: Let the odd numbers be 2k+1 and 2m+1. Their sum is 2k+1+2m+1 = 2(k+m+1), which is even.
Grading Guidelines: Award 1 point for setting up odd number representation, 1 point for algebraic manipulation, 1 point for conclusion.
Student Answer: "Odd numbers end in 1,3,5,7,9. Adding two odd numbers gives an even number. For example, 3+5=8 which is even."

Analysis:
- Student did NOT use the required algebraic representation (2k+1) ✗
- Student only provided specific examples, not a general proof ✗
- Student stated the correct conclusion but without proper reasoning ✓
- Missing: general algebraic proof, proper representation of odd numbers

Grade: {"score": 1, "max_score": 3, "rationale": "Student only stated the conclusion with examples but provided no general proof. Missing required algebraic representation (2k+1) and general algebraic reasoning. Only the conclusion point can be awarded.", "response": "1/3 - Examples only, no general proof"}

Example 3 - Incorrect Answer:
Problem: Solve x^2 - 5x + 6 = 0
Solution: Factoring: (x-2)(x-3) = 0, so x = 2 or x = 3
Grading Guidelines: Award 1 point for correct factoring, 1 point for finding both roots.
Student Answer: "Using the quadratic formula: x = (5 ± sqrt(25-24))/2 = (5 ± 1)/2, so x = 3 and x = 2"

Analysis:
- Student used quadratic formula correctly ✓
- Student calculated discriminant correctly: 25-24 = 1 ✓
- Student found both roots correctly: (5+1)/2 = 3, (5-1)/2 = 2 ✓
- Method is different from factoring but equally valid

Grade: {"score": 2, "max_score": 2, "rationale": "Correct solution using quadratic formula. Both roots (x=2 and x=3) are correct. Method is valid even though different from suggested factoring approach.", "response": "2/2 - Correct solution with valid method"}
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
        # Step 1: Initial grading with detailed chain-of-thought
        max_score_val = inputs.get('max_score', 10)
        instruction = f"""You are an expert mathematical grader for the International Mathematical Olympiad (IMO).

Your task is to grade a student's answer to a mathematical problem with precision and accuracy.

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

Follow this structured grading process:

STEP 1 - UNDERSTAND THE PROBLEM:
- What is the problem asking for?
- What are the key concepts and techniques needed?

STEP 2 - ANALYZE THE OFFICIAL SOLUTION:
- Break down the solution into key components/steps
- Identify what constitutes a complete correct answer
- Note any alternative valid approaches

STEP 3 - EVALUATE THE STUDENT'S ANSWER:
- Compare each part of the student's answer to the official solution
- Mark each component as: ✓ (correct), ✗ (incorrect), or ~ (partial)
- Identify any errors, gaps, or missing steps
- Check if the student made any unjustified claims
- Note if the student used a valid alternative method

STEP 4 - ASSIGN SCORE (0 to {max_score_val}):
- Use the grading guidelines strictly
- Award points only for correct, justified work
- Do NOT award partial credit for incorrect reasoning
- Consider: Is each point in the guidelines satisfied?

STEP 5 - VERIFY:
- Re-read the student's answer one more time
- Check if you missed any errors
- Ensure your score is consistent with the guidelines

Respond in JSON format with the following schema:
<json>
{{
    "thinking": "Your detailed step-by-step analysis following the 5 steps above",
    "score": <integer score from 0 to {max_score_val}>,
    "max_score": {max_score_val},
    "rationale": "Detailed explanation: what was correct, what was wrong, and why this exact score was assigned",
    "response": "<score>/{max_score_val} - <brief 1-sentence summary of answer quality>"
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

        # Step 2: Rigorous self-reflection to verify the grade
        if prediction != "None" and len(msg_history) >= 2:
            reflection_msg = f"""You previously graded this answer. Now perform a rigorous self-review to ensure accuracy.

Your previous grade: {prediction}

CRITICAL REVIEW QUESTIONS:

1. COMPARISON CHECK:
   - Did you compare EVERY part of the student's answer to the official solution?
   - Did you verify each claim the student made?
   - List any discrepancies you find.

2. ERROR DETECTION:
   - Did you miss any errors in the student's work?
   - Are there any subtle mistakes you overlooked?
   - Did the student make any unjustified assumptions?

3. STRICTNESS CHECK:
   - Were you too generous with partial credit?
   - Did you award points for work that isn't fully correct?
   - Would a strict IMO grader give the same score?

4. GUIDELINE ADHERENCE:
   - Does your score strictly follow the grading guidelines?
   - Are you awarding points only for what the guidelines specify?

5. FINAL VERIFICATION:
   - Re-examine the student's answer one more time
   - Confirm or revise your score

If you find ANY issues, provide a corrected grade with explanation. If your grade is correct, confirm it.

Respond in JSON format:
<json>
{{
    "reflection": "Your detailed self-review addressing each question above. Be critical of your own grading.",
    "revised_score": <integer score from 0 to {max_score_val}>,
    "revised_max_score": {max_score_val},
    "final_response": "<score>/{max_score_val} - <brief 1-sentence summary>"
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
                            self.log_fn(f"Grade after reflection: {prediction}")
                        elif "revised_score" in result and "revised_max_score" in result:
                            prediction = f"{result['revised_score']}/{result['revised_max_score']}"
                            self.log_fn(f"Grade after reflection: {prediction}")
                        elif "revised_score" in result:
                            # Fallback if only revised_score is present
                            max_score_val = result.get("revised_max_score", max_score_val)
                            prediction = f"{result['revised_score']}/{max_score_val}"
                            self.log_fn(f"Grade after reflection: {prediction}")
                        else:
                            self.log_fn("Reflection completed, keeping initial grade")
                    else:
                        self.log_fn("No JSON extracted from reflection response, keeping initial grade")
                except Exception as e:
                    self.log_fn(f"Error extracting revised prediction: {e}, keeping initial grade")
            except Exception as e:
                self.log_fn(f"Error during reflection step: {e}")

        return str(prediction), msg_history
