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
    
    Enhanced with detailed logging and multiple repair strategies for better
    reliability with various LLM output formats.
    """
    results = []
    search_from = 0
    extraction_log = []
    repair_attempts = 0
    
    # First try to find <json>...</json> blocks
    while True:
        start = text.find("<json>", search_from)
        if start == -1:
            break
        end = text.find("</json>", start)
        if end == -1:
            extraction_log.append(f"Unclosed <json> tag at position {start}")
            break
        inner = text[start + 6:end].strip()
        search_from = end + 7
        try:
            parsed = json.loads(inner)
            results.append(parsed)
            extraction_log.append(f"Successfully parsed <json> block at {start}-{end}")
        except json.JSONDecodeError as e:
            logger.debug(f"JSON decode error in <json> block: {e}")
            extraction_log.append(f"JSON decode error at {start}-{end}: {str(e)[:100]}")
            # Try to fix common LLM formatting issues
            fixed = _attempt_json_repair(inner)
            repair_attempts += 1
            if fixed:
                results.append(fixed)
                extraction_log.append(f"Successfully repaired JSON at {start}-{end} (attempt {repair_attempts})")
            continue
    
    # Fallback 1: try markdown code blocks if no <json> blocks found
    if not results:
        # Look for ```json ... ``` blocks (with or without json label)
        markdown_pattern = r'```(?:json)?\s*\n(.*?)\n```'
        for match in re.finditer(markdown_pattern, text, re.DOTALL):
            try:
                results.append(json.loads(match.group(1).strip()))
                extraction_log.append(f"Successfully parsed markdown block")
            except json.JSONDecodeError:
                # Try repair on markdown blocks too
                fixed = _attempt_json_repair(match.group(1).strip())
                repair_attempts += 1
                if fixed:
                    results.append(fixed)
                    extraction_log.append(f"Successfully repaired markdown block (attempt {repair_attempts})")
                continue
    
    # Fallback 2: try to find JSON objects directly in the text
    if not results:
        # Look for patterns that look like JSON objects with balanced braces
        # This pattern handles nested braces better
        def find_json_objects(s: str) -> list[str]:
            """Find potential JSON objects by tracking brace balance."""
            objects = []
            i = 0
            while i < len(s):
                if s[i] == '{':
                    start = i
                    depth = 1
                    i += 1
                    while i < len(s) and depth > 0:
                        if s[i] == '{':
                            depth += 1
                        elif s[i] == '}':
                            depth -= 1
                        i += 1
                    if depth == 0:
                        objects.append(s[start:i])
                else:
                    i += 1
            return objects
        
        for candidate in find_json_objects(text):
            try:
                parsed = json.loads(candidate)
                # Only accept if it has expected keys for grading
                if any(key in parsed for key in ['score', 'response', 'thinking', 'rationale', 'revised_score']):
                    results.append(parsed)
                    extraction_log.append(f"Successfully parsed inline JSON with keys: {list(parsed.keys())}")
            except (json.JSONDecodeError, ValueError):
                # Try repair on inline JSON too
                fixed = _attempt_json_repair(candidate)
                repair_attempts += 1
                if fixed and any(key in fixed for key in ['score', 'response', 'thinking', 'rationale', 'revised_score']):
                    results.append(fixed)
                    extraction_log.append(f"Successfully repaired inline JSON (attempt {repair_attempts})")
                continue
    
    # Log extraction summary for debugging
    if extraction_log:
        logger.debug(f"JSON extraction log: {'; '.join(extraction_log)}")
        if repair_attempts > 0:
            logger.info(f"JSON extraction required {repair_attempts} repair attempts")
    
    return results or None


def _attempt_json_repair(text: str) -> dict | None:
    """Attempt to repair common JSON formatting errors from LLM outputs.
    
    Fixes:
    - Trailing commas before closing braces/brackets
    - Single quotes instead of double quotes
    - Unquoted keys
    - Missing quotes around string values
    - Comments in JSON
    - Newlines in strings
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
    
    # Fix 4: Remove comments (// and /* */)
    repaired = re.sub(r'//[^\n]*', '', repaired)
    repaired = re.sub(r'/\*.*?\*/', '', repaired, flags=re.DOTALL)
    
    # Fix 5: Handle newlines in strings by escaping them
    # First, find all string values and escape newlines within them
    def escape_newlines_in_strings(match):
        content = match.group(1)
        # Escape unescaped newlines
        content = content.replace('\n', '\\n').replace('\t', '\\t')
        return f'"{content}"'
    
    # This is a simplified approach - replace actual newlines that aren't already escaped
    repaired = re.sub(r'"((?:[^"\\]|\\.)*?)"', escape_newlines_in_strings, repaired, flags=re.DOTALL)
    
    # Fix 6: Fix common escape sequence issues (but preserve valid ones)
    # Don't unescape valid escape sequences
    repaired = repaired.replace('\\\\"', '\"')  # Fix double-escaped quotes
    
    # Fix 7: Handle numeric values that might be written as words
    # (rare but some LLMs do this)
    
    try:
        return json.loads(repaired)
    except json.JSONDecodeError as e:
        # One more attempt: try to extract just the object if there's extra text
        try:
            # Find the first { and last }
            start = repaired.find('{')
            end = repaired.rfind('}')
            if start != -1 and end != -1 and end > start:
                return json.loads(repaired[start:end+1])
        except json.JSONDecodeError:
            pass
        logger.debug(f"JSON repair failed: {e}")
        return None


# Few-shot examples for IMO grading
FEW_SHOT_EXAMPLES = """
Example 1:
Problem: Find all positive integers n such that n^2 + 3n + 2 is divisible by 4.
Solution: n^2 + 3n + 2 = (n+1)(n+2). For divisibility by 4, either n+1 or n+2 must be even, and one of them must be divisible by 4. This happens when n ≡ 0 or 3 (mod 4).
Grading Guidelines: Award 1 point for factoring, 1 point for analyzing cases, 1 point for correct answer.
Student Answer: "I factored it as (n+1)(n+2). Since these are consecutive integers, one is even. For divisibility by 4, we need one factor divisible by 4. This happens when n=3,7,11,... or n=0,4,8,... So n ≡ 0 or 3 (mod 4)."

Analysis: The student correctly factored the expression, analyzed the cases properly, and arrived at the correct answer. Full credit awarded.

<json>
{
    "thinking": "The student provided a complete and correct solution. They correctly factored n^2 + 3n + 2 as (n+1)(n+2), recognized that consecutive integers have special divisibility properties, and correctly identified that n ≡ 0 or 3 (mod 4) for the product to be divisible by 4.",
    "score": 3,
    "max_score": 3,
    "rationale": "Complete solution with correct factoring (1 pt), case analysis (1 pt), and final answer (1 pt).",
    "response": "3/3 - Complete and correct solution with proper factoring and case analysis."
}
</json>

Example 2:
Problem: Prove that the sum of two odd numbers is even.
Solution: Let the odd numbers be 2k+1 and 2m+1. Their sum is 2k+1+2m+1 = 2(k+m+1), which is even.
Grading Guidelines: Award 1 point for setting up odd number representation, 1 point for algebraic manipulation, 1 point for conclusion.
Student Answer: "Odd numbers end in 1,3,5,7,9. Adding two odd numbers gives an even number. For example, 3+5=8 which is even."

Analysis: The student only provided examples without a general proof. They did not use the algebraic representation of odd numbers (2k+1) and did not provide general reasoning. Only partial credit for demonstrating understanding through examples.

<json>
{
    "thinking": "The student demonstrated some understanding by providing examples, but this is not a mathematical proof. They did not use the standard algebraic representation of odd numbers (2k+1 form) and did not show that the sum is always even for ALL odd numbers, just gave specific examples.",
    "score": 1,
    "max_score": 3,
    "rationale": "Partial credit for demonstrating understanding through examples, but no general proof provided. Missing algebraic representation (-1 pt) and general reasoning (-1 pt).",
    "response": "1/3 - Examples provided but no general proof. Missing algebraic representation and general reasoning."
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

Your task is to grade a student's answer to a mathematical problem with precision and consistency.

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
1. Carefully read the official solution to understand the correct approach
2. Analyze what the student did correctly and what they missed
3. Identify any errors, gaps, or incorrect statements in the student's work
4. Compare the student's work point-by-point against the grading guidelines
5. Determine the score based strictly on the grading guidelines
6. Provide detailed rationale explaining each point awarded or deducted

IMPORTANT: You must respond in valid JSON format wrapped in <json>...</json> tags with exactly these fields:
- "thinking": Your detailed step-by-step analysis
- "score": The numerical score (integer)
- "max_score": The maximum possible score (integer)
- "rationale": Detailed explanation of the grading decision
- "response": A brief summary in format "<score>/<max_score> - <summary>"

Example format:
<json>
{{
    "thinking": "Your analysis here...",
    "score": 5,
    "max_score": 7,
    "rationale": "Student correctly identified... but missed...",
    "response": "5/7 - Good approach but missing key steps"
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
            reflection_msg = f"""Review your grading above carefully. Be critical and check:

1. Did you award points the student didn't earn? (Over-grading is a common error)
2. Did you miss any errors, gaps, or incorrect statements in the student's work?
3. Is your score consistent with the grading guidelines? Compare point-by-point.
4. Would another expert grader agree with your assessment?
5. Did you misinterpret any part of the student's answer?

IMPORTANT: If you find any issues with your initial grading, you MUST provide a corrected grade. Do not simply confirm your initial grade without critical review.

Respond in JSON format with your reflection and final grade:
<json>
{{
    "reflection": "Your detailed self-review here. Be specific about what you checked and what you found.",
    "revised_score": <numerical score>,
    "revised_max_score": <maximum possible score>,
    "final_response": "<score>/<max_score> - <brief summary of the grade>"
}}
</json>

If your initial grade was correct, set revised_score and revised_max_score to match your initial grade. If you found errors, provide the corrected values."""
            
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
                else:
                    self.log_fn("Warning: No JSON found in reflection response")
            except Exception as e:
                self.log_fn(f"Error extracting revised prediction: {e}")
                import traceback
                self.log_fn(f"Traceback: {traceback.format_exc()}")

        return str(prediction), msg_history
