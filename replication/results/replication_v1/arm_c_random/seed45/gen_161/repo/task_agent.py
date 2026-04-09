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
    Includes robust error recovery for common LLM formatting issues.
    """
    results = []
    search_from = 0
    extraction_attempts = 0
    
    # First try to find <json>...</json> blocks
    while True:
        start = text.find("<json>", search_from)
        if start == -1:
            break
        end = text.find("</json>", start)
        if end == -1:
            logger.warning(f"Found <json> tag at position {start} but no closing </json> tag")
            break
        inner = text[start + 6:end].strip()
        search_from = end + 7
        extraction_attempts += 1
        try:
            results.append(json.loads(inner))
            logger.debug(f"Successfully extracted JSON from <json> block #{extraction_attempts}")
        except json.JSONDecodeError as e:
            logger.debug(f"JSON decode error in <json> block #{extraction_attempts}: {e}")
            # Try to fix common LLM formatting issues
            fixed = _attempt_json_repair(inner)
            if fixed:
                results.append(fixed)
                logger.info(f"Successfully repaired and extracted JSON from <json> block #{extraction_attempts}")
            else:
                logger.warning(f"Failed to repair JSON from <json> block #{extraction_attempts}")
            continue
    
    # Fallback 1: try markdown code blocks if no <json> blocks found
    if not results:
        logger.info("No <json> blocks found, trying markdown code blocks as fallback")
        # Look for ```json ... ``` blocks
        markdown_pattern = r'```(?:json)?\s*\n(.*?)\n```'
        markdown_matches = list(re.finditer(markdown_pattern, text, re.DOTALL))
        logger.debug(f"Found {len(markdown_matches)} markdown code blocks")
        for i, match in enumerate(markdown_matches):
            try:
                results.append(json.loads(match.group(1).strip()))
                logger.debug(f"Successfully extracted JSON from markdown block #{i+1}")
            except json.JSONDecodeError:
                # Try repair on markdown blocks too
                fixed = _attempt_json_repair(match.group(1).strip())
                if fixed:
                    results.append(fixed)
                    logger.info(f"Successfully repaired and extracted JSON from markdown block #{i+1}")
                else:
                    logger.debug(f"Failed to extract JSON from markdown block #{i+1}")
                continue
    
    # Fallback 2: try to find JSON objects directly in the text
    if not results:
        logger.info("No markdown blocks found, trying inline JSON pattern as final fallback")
        # Look for patterns that look like JSON objects
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        inline_matches = list(re.finditer(json_pattern, text, re.DOTALL))
        logger.debug(f"Found {len(inline_matches)} potential inline JSON objects")
        for i, match in enumerate(inline_matches):
            try:
                candidate = match.group(0).strip()
                # Only accept if it has expected keys
                parsed = json.loads(candidate)
                if any(key in parsed for key in ['score', 'response', 'thinking', 'rationale']):
                    results.append(parsed)
                    logger.info(f"Successfully extracted inline JSON object #{i+1} with expected keys")
            except (json.JSONDecodeError, ValueError):
                continue
    
    if results:
        logger.info(f"Successfully extracted {len(results)} JSON object(s) from text")
    else:
        logger.warning(f"Failed to extract any valid JSON from text (attempted {extraction_attempts} <json> blocks)")
    
    return results or None


def _attempt_json_repair(text: str) -> dict | None:
    """Attempt to repair common JSON formatting errors from LLM outputs.
    
    Fixes:
    - Trailing commas before closing braces/brackets
    - Single quotes instead of double quotes
    - Unquoted keys
    - Missing quotes around string values
    """
    import re
    
    original = text.strip()
    
    # Fix 1: Remove trailing commas before } or ]
    repaired = re.sub(r',\s*([}\]])', r'\1', original)
    
    # Fix 2: Replace single quotes with double quotes (carefully)
    # Only replace single quotes that appear to be delimiters
    repaired = re.sub(r"(?<!\\)'([^']*?)'(?=\s*[:}\],])", r'"\1"', repaired)
    
    # Fix 3: Add quotes to unquoted keys
    repaired = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', repaired)
    
    # Fix 4: Fix common escape sequence issues
    repaired = repaired.replace('\\"', '"').replace('\\n', '\n').replace('\\t', '\t')
    
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
Student Answer: "I factored it as (n+1)(n+2). Since these are consecutive integers, one is even. For divisibility by 4, we need one factor divisible by 4. This happens when n=3,7,11,... or n=0,4,8,... So n ≡ 0 or 3 (mod 4)."
Grade: {"score": 3, "max_score": 3, "rationale": "Complete solution with correct factoring, case analysis, and answer.", "confidence": 0.95}

Example 2:
Problem: Prove that the sum of two odd numbers is even.
Solution: Let the odd numbers be 2k+1 and 2m+1. Their sum is 2k+1+2m+1 = 2(k+m+1), which is even.
Grading Guidelines: Award 1 point for setting up odd number representation, 1 point for algebraic manipulation, 1 point for conclusion.
Student Answer: "Odd numbers end in 1,3,5,7,9. Adding two odd numbers gives an even number. For example, 3+5=8 which is even."
Grade: {"score": 1, "max_score": 3, "rationale": "Student only provided examples without general proof. Missing algebraic representation and general reasoning.", "confidence": 0.90}

Example 3:
Problem: Solve the equation x^2 - 5x + 6 = 0.
Solution: Factoring gives (x-2)(x-3) = 0, so x = 2 or x = 3.
Grading Guidelines: Award 1 point for correct factoring, 1 point for finding both roots.
Student Answer: "Using the quadratic formula: x = (5 ± √(25-24))/2 = (5 ± 1)/2, so x = 3 or x = 2."
Grade: {"score": 2, "max_score": 2, "rationale": "Correct solution using alternative method (quadratic formula). Both roots found correctly.", "confidence": 0.98}
"""


def _validate_score_format(prediction: str) -> bool:
    """Validate that prediction follows the expected score format."""
    if not prediction or prediction == "None":
        return False
    # Check for pattern like "3/5" or "3 / 5"
    import re
    return bool(re.match(r'^\d+\s*/\s*\d+', prediction.strip()))


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
5. Assess your confidence in this grade (0.0 to 1.0)

Respond in JSON format with the following schema:
<json>
{{
    "thinking": "Your detailed step-by-step analysis here",
    "score": <numerical score>,
    "max_score": <maximum possible score>,
    "rationale": "Detailed explanation of why this score was awarded",
    "confidence": <confidence score between 0.0 and 1.0>,
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
        initial_confidence = None
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
                
                # Extract confidence if available
                if "confidence" in result:
                    initial_confidence = result["confidence"]
                    self.log_fn(f"Initial confidence: {initial_confidence}")
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
5. Is your confidence level appropriate given the clarity of the student's answer?

If you need to revise your grade, provide the corrected JSON with updated confidence. If your grade is correct, confirm it.

Respond in JSON format:
<json>
{{
    "reflection": "Your self-review here",
    "revised_score": <score>,
    "revised_max_score": <max_score>,
    "revised_confidence": <confidence score between 0.0 and 1.0>,
    "final_response": "<score>/<max_score> - <brief summary>"
}}
</json>"""
            
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
                    else:
                        self.log_fn(f"Warning: Reflection JSON missing expected fields. Keys: {list(result.keys())}")
                    
                    # Extract revised confidence if available
                    if "revised_confidence" in result:
                        revised_confidence = result["revised_confidence"]
                        self.log_fn(f"Revised confidence: {revised_confidence}")
                else:
                    self.log_fn("Warning: No JSON found in reflection response")
            except Exception as e:
                self.log_fn(f"Error extracting revised prediction: {e}")
                import traceback
                self.log_fn(f"Traceback: {traceback.format_exc()}")

        # Validate final prediction format
        if not _validate_score_format(prediction):
            self.log_fn(f"Warning: Final prediction '{prediction}' does not match expected format 'score/max_score'")

        return str(prediction), msg_history
