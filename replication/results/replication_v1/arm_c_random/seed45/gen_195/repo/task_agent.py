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
    extraction_log = []
    
    # First try to find <json>...</json> blocks with proper nested brace handling
    while True:
        start = text.find("<json>", search_from)
        if start == -1:
            break
        
        # Find the closing tag, but be careful about nested content
        json_start = start + 6
        end = text.find("</json>", json_start)
        if end == -1:
            extraction_log.append(f"Unclosed <json> tag at position {start}")
            break
        
        # Extract content and validate brace balance
        inner = text[json_start:end].strip()
        
        # Check for nested braces - if we have more opening than closing braces,
        # we might have found a premature closing tag
        brace_count = 0
        in_string = False
        escape_next = False
        for i, char in enumerate(inner):
            if escape_next:
                escape_next = False
                continue
            if char == '\\':
                escape_next = True
                continue
            if char == '"' and (i == 0 or inner[i-1] != '\\'):
                in_string = not in_string
                continue
            if not in_string:
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
        
        # If braces are unbalanced, try to find a better closing position
        if brace_count > 0:
            # Look for additional closing tags
            extended_end = end
            while brace_count > 0 and extended_end != -1:
                next_end = text.find("</json>", extended_end + 7)
                if next_end == -1:
                    break
                extended_content = text[json_start:next_end]
                # Recount braces in extended content
                brace_count = 0
                in_string = False
                escape_next = False
                for i, char in enumerate(extended_content):
                    if escape_next:
                        escape_next = False
                        continue
                    if char == '\\':
                        escape_next = True
                        continue
                    if char == '"' and (i == 0 or extended_content[i-1] != '\\'):
                        in_string = not in_string
                        continue
                    if not in_string:
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                if brace_count == 0:
                    end = next_end
                    inner = text[json_start:end].strip()
                    break
                extended_end = next_end
        
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
            if fixed:
                results.append(fixed)
                extraction_log.append(f"Successfully repaired JSON at {start}-{end}")
            continue
    
    # Fallback 1: try markdown code blocks if no <json> blocks found
    if not results:
        # Look for ```json ... ``` blocks with better nested brace handling
        # Use a more flexible pattern that handles various whitespace scenarios
        markdown_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
        for match in re.finditer(markdown_pattern, text, re.DOTALL):
            content = match.group(1).strip()
            if not content:
                continue
            try:
                results.append(json.loads(content))
                extraction_log.append(f"Successfully parsed markdown block")
            except json.JSONDecodeError:
                # Try repair on markdown blocks too
                fixed = _attempt_json_repair(content)
                if fixed:
                    results.append(fixed)
                    extraction_log.append(f"Successfully repaired markdown block")
                continue
    
    # Fallback 2: try to find JSON objects directly in the text using balanced brace matching
    if not results:
        # Find all potential JSON object starts
        for match in re.finditer(r'\{', text):
            start_pos = match.start()
            # Try to find the matching closing brace
            brace_count = 1
            in_string = False
            escape_next = False
            for i in range(start_pos + 1, len(text)):
                char = text[i]
                if escape_next:
                    escape_next = False
                    continue
                if char == '\\':
                    escape_next = True
                    continue
                if char == '"' and text[i-1] != '\\':
                    in_string = not in_string
                    continue
                if not in_string:
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            # Found a complete JSON object
                            candidate = text[start_pos:i+1]
                            try:
                                parsed = json.loads(candidate)
                                # Only accept if it has expected keys
                                if any(key in parsed for key in ['score', 'response', 'thinking', 'rationale', 'max_score']):
                                    results.append(parsed)
                                    extraction_log.append(f"Successfully parsed inline JSON at {start_pos}")
                            except (json.JSONDecodeError, ValueError):
                                # Try repair
                                fixed = _attempt_json_repair(candidate)
                                if fixed and any(key in fixed for key in ['score', 'response', 'thinking', 'rationale', 'max_score']):
                                    results.append(fixed)
                                    extraction_log.append(f"Successfully repaired inline JSON at {start_pos}")
                            break
    
    # Log extraction summary for debugging
    if extraction_log:
        logger.debug(f"JSON extraction log: {'; '.join(extraction_log)}")
    
    return results or None


def _attempt_json_repair(text: str) -> dict | None:
    """Attempt to repair common JSON formatting errors from LLM outputs.
    
    Fixes:
    - Trailing commas before closing braces/brackets
    - Single quotes instead of double quotes
    - Unquoted keys
    - Missing quotes around string values
    - Newlines in strings
    - Comments
    - Control characters
    """
    import re
    
    original = text.strip()
    
    # Remove any text before the first { and after the last }
    start_idx = original.find('{')
    end_idx = original.rfind('}')
    if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
        return None
    repaired = original[start_idx:end_idx+1]
    
    # Fix 1: Remove C-style comments
    repaired = re.sub(r'/\*.*?\*/', '', repaired, flags=re.DOTALL)
    repaired = re.sub(r'//.*?$', '', repaired, flags=re.MULTILINE)
    
    # Fix 2: Remove trailing commas before } or ]
    repaired = re.sub(r',\s*([}\]])', r'\1', repaired)
    
    # Fix 3: Replace single quotes with double quotes (carefully)
    # Handle nested quotes by being more selective
    def replace_single_quotes(match):
        content = match.group(1)
        # Don't replace if it looks like an apostrophe in a word
        if "'" in content and not any(c in content for c in '{}[]'):
            return f'"{content}"'
        return match.group(0)
    
    # Replace single-quoted strings that are likely JSON keys/values
    repaired = re.sub(r"(?<![a-zA-Z0-9])'((?:[^'\\]|\\.)*?)'(?=\s*[:}\],])", r'"\1"', repaired)
    
    # Fix 4: Add quotes to unquoted keys (word followed by colon)
    repaired = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)(\s*:)', r'\1"\2"\3', repaired)
    
    # Fix 5: Handle newlines in strings - replace with \n
    # First, find strings and escape newlines within them
    def escape_newlines_in_strings(s):
        result = []
        in_string = False
        escape_next = False
        for char in s:
            if escape_next:
                result.append(char)
                escape_next = False
                continue
            if char == '\\':
                result.append(char)
                escape_next = True
                continue
            if char == '"' and not escape_next:
                in_string = not in_string
                result.append(char)
            elif char == '\n' and in_string:
                result.append('\\n')
            elif char == '\t' and in_string:
                result.append('\\t')
            elif ord(char) < 32 and in_string:
                # Replace other control characters
                result.append(f'\\u{ord(char):04x}')
            else:
                result.append(char)
        return ''.join(result)
    
    repaired = escape_newlines_in_strings(repaired)
    
    # Fix 6: Handle escaped quotes that might be double-escaped
    repaired = repaired.replace('\\"', '"').replace('\\"', '\"')
    
    # Fix 7: Remove BOM and other zero-width characters
    repaired = repaired.replace('\ufeff', '').replace('\u200b', '').replace('\u200c', '').replace('\u200d', '')
    
    try:
        return json.loads(repaired)
    except json.JSONDecodeError as e:
        # Last resort: try to extract just the fields we need
        try:
            # Try to find score and max_score with regex
            score_match = re.search(r'["\']?score["\']?\s*:\s*(\d+)', repaired)
            max_score_match = re.search(r'["\']?max_score["\']?\s*:\s*(\d+)', repaired)
            response_match = re.search(r'["\']?response["\']?\s*:\s*["\']([^"\']+)["\']', repaired)
            thinking_match = re.search(r'["\']?thinking["\']?\s*:\s*["\']([^"\']*(?:\.[^"\']*)*)["\']', repaired, re.DOTALL)
            rationale_match = re.search(r'["\']?rationale["\']?\s*:\s*["\']([^"\']*(?:\.[^"\']*)*)["\']', repaired, re.DOTALL)
            
            if score_match and max_score_match:
                result = {
                    "score": int(score_match.group(1)),
                    "max_score": int(max_score_match.group(1))
                }
                if response_match:
                    result["response"] = response_match.group(1)
                if thinking_match:
                    result["thinking"] = thinking_match.group(1).replace('\\n', '\n')
                if rationale_match:
                    result["rationale"] = rationale_match.group(1).replace('\\n', '\n')
                return result
        except Exception:
            pass
        return None


def _validate_and_normalize_score(result: dict, default_max_score: int = 7) -> dict:
    """Validate and normalize score fields in the result.
    
    Ensures:
    - score and max_score are integers
    - 0 <= score <= max_score
    - max_score is positive
    - response field is properly formatted
    """
    if not isinstance(result, dict):
        return result
    
    # Extract or default max_score
    max_score = result.get('max_score', default_max_score)
    try:
        max_score = int(max_score)
    except (ValueError, TypeError):
        max_score = default_max_score
    
    # Ensure max_score is positive
    if max_score <= 0:
        max_score = default_max_score
    
    # Extract and validate score
    score = result.get('score', 0)
    try:
        score = int(score)
    except (ValueError, TypeError):
        score = 0
    
    # Clamp score to valid range
    score = max(0, min(score, max_score))
    
    # Update result
    result['score'] = score
    result['max_score'] = max_score
    
    # Ensure response field exists and is properly formatted
    if 'response' not in result or not result['response']:
        result['response'] = f"{score}/{max_score}"
    elif not re.search(r'\d+\s*/\s*\d+', str(result['response'])):
        # If response doesn't contain a score/max_score pattern, prepend it
        result['response'] = f"{score}/{max_score} - {result['response']}"
    
    return result


# Few-shot examples for IMO grading - using proper <json> format
FEW_SHOT_EXAMPLES = """
Example 1 - Complete Correct Solution:
Problem: Find all positive integers n such that n^2 + 3n + 2 is divisible by 4.
Solution: n^2 + 3n + 2 = (n+1)(n+2). For divisibility by 4, either n+1 or n+2 must be even, and one of them must be divisible by 4. This happens when n ≡ 0 or 3 (mod 4).
Grading Guidelines: Award 1 point for factoring, 1 point for analyzing cases, 1 point for correct answer.
Student Answer: "I factored it as (n+1)(n+2). Since these are consecutive integers, one is even. For divisibility by 4, we need one factor divisible by 4. This happens when n=3,7,11,... or n=0,4,8,... So n ≡ 0 or 3 (mod 4)."

<json>
{
    "thinking": "The student correctly factored n^2 + 3n + 2 as (n+1)(n+2). They recognized that consecutive integers include one even number, and correctly identified when one factor is divisible by 4. The answer n ≡ 0 or 3 (mod 4) is complete and correct.",
    "score": 3,
    "max_score": 3,
    "rationale": "Complete solution with correct factoring, case analysis, and answer. All grading criteria met.",
    "response": "3/3 - Complete correct solution"
}
</json>

Example 2 - Partial Credit:
Problem: Prove that the sum of two odd numbers is even.
Solution: Let the odd numbers be 2k+1 and 2m+1. Their sum is 2k+1+2m+1 = 2(k+m+1), which is even.
Grading Guidelines: Award 1 point for setting up odd number representation, 1 point for algebraic manipulation, 1 point for conclusion.
Student Answer: "Odd numbers end in 1,3,5,7,9. Adding two odd numbers gives an even number. For example, 3+5=8 which is even."

<json>
{
    "thinking": "The student only provided examples without general proof. They correctly stated that the sum of two odd numbers is even, but failed to provide the algebraic representation (2k+1 and 2m+1) and general reasoning required by the grading guidelines. Examples alone do not constitute a proof.",
    "score": 1,
    "max_score": 3,
    "rationale": "Student only provided examples without general proof. Missing algebraic representation and general reasoning required for full credit.",
    "response": "1/3 - Examples only, no general proof"
}
</json>

Example 3 - Alternative Valid Approach:
Problem: Solve x^2 - 5x + 6 = 0.
Solution: Factoring: (x-2)(x-3) = 0, so x = 2 or x = 3.
Grading Guidelines: Award 2 points for correct factoring, 2 points for finding both roots.
Student Answer: "Using the quadratic formula: x = (5 ± sqrt(25-24))/2 = (5 ± 1)/2, so x = 3 or x = 2."

<json>
{
    "thinking": "The student correctly applied the quadratic formula and found both roots x = 2 and x = 3. While they used a different method than the factoring approach in the official solution, their method is mathematically valid and the answer is correct.",
    "score": 4,
    "max_score": 4,
    "rationale": "Correct solution using quadratic formula. Both roots found accurately. Different valid approach from official solution.",
    "response": "4/4 - Correct solution using quadratic formula"
}
</json>

Example 4 - Zero Score (Completely Wrong):
Problem: Find the derivative of f(x) = x^2 + 3x.
Solution: f'(x) = 2x + 3 using the power rule.
Grading Guidelines: Award 2 points for applying power rule correctly, 1 point for final answer.
Student Answer: "The derivative is x^2 + 3 because you decrease the power by 1."

<json>
{
    "thinking": "The student fundamentally misunderstood the power rule. They stated that the derivative of x^2 is x^2 (incorrect - should be 2x) and the derivative of 3x is 3 (correct, but for wrong reason). The explanation 'decrease the power by 1' is incomplete and misapplied. No correct application of differentiation rules.",
    "score": 0,
    "max_score": 3,
    "rationale": "Fundamental misunderstanding of differentiation. No correct application of power rule. Incorrect final answer.",
    "response": "0/3 - Fundamental errors in applying differentiation rules"
}
</json>

Example 5 - Partial Credit with Calculation Error:
Problem: Compute the integral of 2x from 0 to 3.
Solution: ∫(0 to 3) 2x dx = [x^2](0 to 3) = 9 - 0 = 9.
Grading Guidelines: Award 2 points for correct antiderivative, 2 points for correct evaluation at bounds.
Student Answer: "The antiderivative of 2x is x^2. Evaluating: 3^2 - 0^2 = 6 - 0 = 6."

<json>
{
    "thinking": "The student correctly identified the antiderivative as x^2 (2 points). However, they made an arithmetic error when evaluating: 3^2 = 9, not 6. The setup for evaluation was correct (subtracting lower bound from upper), but the final calculation was wrong.",
    "score": 2,
    "max_score": 4,
    "rationale": "Correct antiderivative (2 points) but arithmetic error in evaluation (0 points for evaluation). 3^2 = 9, not 6.",
    "response": "2/4 - Correct antiderivative but arithmetic error in evaluation"
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
        # Extract max_score from grading guidelines if available
        default_max_score = 7  # IMO problems typically have 7 points
        grading_guidelines = inputs.get('grading_guidelines', '')
        
        # Try to extract max score from grading guidelines
        max_score_match = re.search(r'(\d+)\s*(?:points?|marks?)', grading_guidelines, re.IGNORECASE)
        if max_score_match:
            default_max_score = int(max_score_match.group(1))
        
        # Step 1: Initial grading with chain-of-thought
        instruction = f"""You are an expert mathematical grader for the International Mathematical Olympiad (IMO).

Your task is to grade a student's answer to a mathematical problem with precision and consistency.

GRADING PRINCIPLES:
1. Award points ONLY for correct mathematical reasoning and results
2. Partial credit is allowed when the student makes progress toward the solution
3. Different valid approaches should receive full credit even if they differ from the official solution
4. Calculation errors should result in partial credit, not zero (unless the error is fundamental)
5. Missing steps that are "obvious" to an expert may not need explicit statement
6. Be consistent - similar errors should receive similar deductions

{FEW_SHOT_EXAMPLES}

Now grade the following problem:

Domain: {inputs.get('domain', 'Mathematics')}

Problem:
{inputs.get('problem', '')}

Official Solution:
{inputs.get('solution', '')}

Grading Guidelines:
{grading_guidelines}

Student Answer:
{inputs.get('student_answer', '')}

Think step by step:
1. First, understand what the problem is asking and what the official solution provides
2. Analyze the student's approach - is it valid even if different from the official solution?
3. Check each step of the student's work for correctness
4. Identify any errors, gaps, or missing steps
5. Compare against the grading guidelines point by point
6. Determine the score based on what the student earned, not what they missed
7. Provide detailed rationale explaining the score

IMPORTANT: You MUST respond in valid JSON format wrapped in <json>...</json> tags exactly as shown in the examples.

<json>
{{
    "thinking": "Your detailed step-by-step analysis here. Be thorough and specific about what the student did right and wrong.",
    "score": <numerical score - must be an integer>,
    "max_score": <maximum possible score - must be an integer>,
    "rationale": "Detailed explanation of why this score was awarded. Reference specific parts of the student's work and the grading guidelines.",
    "response": "<score>/<max_score> - <brief summary of the grade>"
}}
</json>"""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON with detailed logging and validation
        prediction = "None"
        initial_result = None
        try:
            last_msg = msg_history[-1]["text"]
            extracted = _extract_jsons(last_msg)
            if extracted:
                result = extracted[-1]
                # Validate and normalize the score
                result = _validate_and_normalize_score(result, default_max_score)
                initial_result = result
                self.log_fn(f"Extracted and validated JSON result: {result}")
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
                score_match = re.search(r'(\d+)\s*/\s*(\d+)', last_msg)
                if score_match:
                    score = int(score_match.group(1))
                    max_score = int(score_match.group(2))
                    # Validate the fallback extraction
                    if max_score > 0 and 0 <= score <= max_score:
                        prediction = f"{score}/{max_score}"
                        self.log_fn(f"Fallback extraction: {prediction}")
                    else:
                        self.log_fn(f"Fallback extraction invalid: {score}/{max_score}")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            import traceback
            self.log_fn(f"Traceback: {traceback.format_exc()}")

        # Step 2: Self-reflection to verify the grade
        if prediction != "None" and len(msg_history) >= 2 and initial_result:
            reflection_msg = f"""Review your grading above carefully. Your initial assessment was:
- Score: {initial_result.get('score', 'N/A')}/{initial_result.get('max_score', 'N/A')}
- Rationale: {initial_result.get('rationale', 'N/A')}

Please verify your grading by checking:
1. CONSISTENCY: Would another expert grader give the same score? If not, why?
2. ACCURACY: Did you miss any errors in the student's work? List any errors found.
3. COMPLETENESS: Did the student earn all points you awarded? Which points are questionable?
4. GUIDELINES: Is your score strictly consistent with the grading guidelines provided?
5. BIAS CHECK: Are you being too lenient or too harsh? Compare to the examples provided.

If your grade is correct, confirm it with the same score. If you need to revise, provide the corrected grade with explanation.

IMPORTANT: You MUST respond in valid JSON format wrapped in <json>...</json> tags.

<json>
{{
    "reflection": "Your detailed self-review addressing each check above. Be critical of your own assessment.",
    "revised_score": <score - must be an integer>,
    "revised_max_score": <max_score - must be an integer>,
    "revision_rationale": "Explanation for any changes made, or confirmation that original grade was correct",
    "final_response": "<score>/<max_score> - <brief summary>"
}}
</json>"""
            
            reflection_response, msg_history, _ = get_response_from_llm(
                msg=reflection_msg,
                model=self.model,
                msg_history=msg_history,
            )
            
            # Try to extract revised prediction with detailed logging and validation
            try:
                last_msg = msg_history[-1]["text"]
                extracted = _extract_jsons(last_msg)
                if extracted:
                    result = extracted[-1]
                    # Validate and normalize the revised score
                    result = _validate_and_normalize_score(result, initial_result.get('max_score', default_max_score))
                    self.log_fn(f"Reflection extracted and validated JSON: {result}")
                    if "final_response" in result:
                        prediction = result["final_response"]
                        self.log_fn(f"Using 'final_response' field: {prediction}")
                    elif "revised_score" in result and "revised_max_score" in result:
                        prediction = f"{result['revised_score']}/{result['revised_max_score']}"
                        self.log_fn(f"Using revised_score/revised_max_score: {prediction}")
                    else:
                        self.log_fn(f"Warning: Reflection JSON missing expected fields. Keys: {list(result.keys())}")
                else:
                    self.log_fn("Warning: No JSON found in reflection response, keeping initial prediction")
            except Exception as e:
                self.log_fn(f"Error extracting revised prediction: {e}, keeping initial prediction")
                import traceback
                self.log_fn(f"Traceback: {traceback.format_exc()}")

        return str(prediction), msg_history
