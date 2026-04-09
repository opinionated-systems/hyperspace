"""
Task agent: solves a given task with a single LLM call.

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
    Also handles nested JSON objects within the content.
    """
    results = []
    search_from = 0
    while True:
        start = text.find("<json>", search_from)
        if start == -1:
            break
        end = text.find("</json>", start)
        if end == -1:
            break
        inner = text[start + 6:end].strip()
        search_from = end + 7
        
        # Try to parse the inner content as JSON
        try:
            results.append(json.loads(inner))
        except json.JSONDecodeError:
            # If direct parsing fails, try to extract JSON using brace matching
            # This handles cases where there might be nested structures
            try:
                json_obj = _extract_json_with_brace_matching(inner)
                if json_obj:
                    results.append(json_obj)
            except Exception:
                continue
    
    # Also try to find JSON in markdown code blocks as fallback
    if not results:
        results = _extract_json_from_markdown(text)
    
    return results or None


def _extract_json_from_markdown(text: str) -> list[dict]:
    """Extract JSON from markdown code blocks as fallback."""
    results = []
    # Look for ```json or ``` blocks
    import re
    pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    matches = re.findall(pattern, text, re.DOTALL)
    for match in matches:
        try:
            data = json.loads(match.strip())
            if isinstance(data, dict):
                results.append(data)
        except json.JSONDecodeError:
            # Try brace matching for nested structures
            try:
                json_obj = _extract_json_with_brace_matching(match)
                if json_obj:
                    results.append(json_obj)
            except Exception:
                continue
    return results


def _extract_json_with_brace_matching(text: str) -> dict | None:
    """Extract a JSON object from text using brace counting.
    
    This handles nested JSON objects properly by tracking brace depth.
    """
    start_idx = text.find('{')
    if start_idx == -1:
        return None
    
    brace_count = 1
    i = start_idx + 1
    while i < len(text) and brace_count > 0:
        if text[i] == '{':
            brace_count += 1
        elif text[i] == '}':
            brace_count -= 1
        i += 1
    
    if brace_count == 0:
        candidate = text[start_idx:i]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    return None


def _extract_json_with_regex(text: str) -> list[dict] | None:
    """Fallback JSON extraction using regex for malformed outputs.
    
    Handles nested JSON objects by using a stack-based brace matching approach
    instead of simple regex that fails on nested structures.
    """
    results = []
    # Try to find JSON objects in code blocks
    code_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    matches = re.findall(code_block_pattern, text, re.DOTALL)
    for match in matches:
        try:
            results.append(json.loads(match.strip()))
        except json.JSONDecodeError:
            continue
    
    # Try to find JSON objects with curly braces using stack-based matching
    # This handles nested objects properly
    if not results:
        i = 0
        while i < len(text):
            # Find the start of a potential JSON object
            if text[i] == '{':
                start = i
                brace_count = 1
                i += 1
                # Track braces to find the matching closing brace
                while i < len(text) and brace_count > 0:
                    if text[i] == '{':
                        brace_count += 1
                    elif text[i] == '}':
                        brace_count -= 1
                    i += 1
                # If we found a complete object with "response" key, try to parse it
                if brace_count == 0:
                    candidate = text[start:i]
                    if '"response"' in candidate:
                        try:
                            results.append(json.loads(candidate))
                        except json.JSONDecodeError:
                            continue
            else:
                i += 1
    
    return results or None


def _extract_response_value(text: str) -> str | None:
    """Last-resort extraction: try to find a response value directly.
    
    This handles cases where the JSON is malformed but we can still
    extract the value after "response": using regex patterns.
    
    Enhanced to handle:
    - Numbers (integers, floats, negative values)
    - Strings in double or single quotes
    - Boolean values (true/false)
    - Null values
    - Nested quotes within strings
    - Whitespace variations
    """
    # Pattern to match "response": followed by a value (number, string, or boolean)
    # Handles: "response": 7, "response": "7", "response": true, etc.
    patterns = [
        # Number (integer or float) - capture without quotes
        (r'"response"\s*:\s*(-?\d+(?:\.\d+)?)(?:\s*[,}\]])', "number"),
        # String in double quotes - handle escaped quotes inside
        (r'"response"\s*:\s*"((?:[^"\\]|\\.)*)"', "string_double"),
        # String in single quotes
        (r"'response'\s*:\s*'([^']*)'", "string_single"),
        # Boolean or null (unquoted literals)
        (r'"response"\s*:\s*(true|false|null)(?:\s*[,}\]])', "literal"),
    ]
    
    for pattern, pattern_type in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = match.group(1)
            
            # Handle based on pattern type
            if pattern_type == "number":
                # Return numbers as-is (they're already strings)
                return value
            elif pattern_type == "literal":
                # Return boolean/null literals as lowercase strings
                return value.lower()
            elif pattern_type in ("string_double", "string_single"):
                # For strings, unescape any escaped characters
                value = value.replace('\\"', '"').replace("\\'", "'").replace("\\n", "\n").replace("\\t", "\t")
                # Try to convert to number if it looks like one
                try:
                    if '.' in value:
                        return str(float(value))
                    else:
                        return str(int(value))
                except ValueError:
                    # Return as string
                    return value
    
    return None


class TaskAgent:
    """Task agent that solves IMO grading problems."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Extract key fields for better prompt construction
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = f"""You are an expert mathematical grader evaluating student solutions to competition mathematics problems.

Your task is to grade a student's answer to an IMO-level mathematics problem. You must carefully analyze all provided materials and provide a rigorous, fair evaluation.

PROBLEM DOMAIN: {domain}

PROBLEM STATEMENT:
```
{problem}
```

OFFICIAL SOLUTION:
```
{solution}
```

GRADING GUIDELINES (RUBRIC):
```
{grading_guidelines}
```

STUDENT'S ANSWER:
```
{student_answer}
```

GRADING PROCESS - Follow these steps carefully:

STEP 1: PROBLEM ANALYSIS
- Identify the key mathematical concepts and techniques required
- Note any constraints, special cases, or edge conditions
- Understand what constitutes a complete solution

STEP 2: OFFICIAL SOLUTION MAPPING
- Break down the official solution into key steps/claims
- Identify which steps are essential vs. optional
- Note the scoring distribution if provided in guidelines

STEP 3: STUDENT SOLUTION EVALUATION
- Check if the student understood the problem (correct interpretation)
- Verify the approach is valid (even if different from official)
- Examine each claim: is it stated or proven?
- Check calculations for correctness
- Identify any logical gaps or missing cases
- Note creative insights or alternative valid approaches

STEP 4: ERROR ANALYSIS (if applicable)
- Distinguish between minor errors (computation, notation) and major errors (logic, missing cases)
- Determine if errors invalidate the solution or just reduce credit
- Check if partial progress deserves partial credit

STEP 5: SCORE ASSIGNMENT
- Apply the grading guidelines rigorously but fairly
- Award partial credit for substantial progress even if incomplete
- Penalize unjustified claims, logical gaps, or missing cases appropriately
- Consider: Does the solution stand as a rigorous mathematical proof?

GRADING PRINCIPLES:
1. RIGOR: A solution must be a complete proof, not just the right answer
2. FAIRNESS: Award credit for valid alternative approaches
3. PARTIAL CREDIT: Recognize substantial progress toward solution
4. CLARITY: Penalize solutions that are unclear or poorly explained
5. JUSTIFICATION: Every non-trivial claim needs justification

IMPORTANT: Your response MUST be a valid JSON object wrapped in <json> tags. Do not include any text outside the JSON tags.

Respond in the following exact format:
<json>
{{
    "response": <your numerical score or evaluation>,
    "reasoning": "<detailed explanation of your grading decision>"
}}
</json>

The "response" field should contain your final grading decision (typically a number or specific evaluation string as specified in the grading guidelines).
The "reasoning" field should contain a detailed explanation (2-4 sentences) explaining:
- What the student did correctly
- What errors or gaps exist
- Why this score is appropriate based on the rubric

Examples of valid responses:
- For a numeric score: <json>{{"response": 7, "reasoning": "Complete solution with rigorous proof. All key claims justified, no logical gaps found."}}</json>
- For partial credit: <json>{{"response": 3, "reasoning": "Correct approach identified and significant progress made, but missing proof for the key lemma in step 3."}}</json>
- For string evaluation: <json>{{"response": "Correct", "reasoning": "All steps verified correctly with proper justification."}}</json>
- For boolean: <json>{{"response": true, "reasoning": "Answer matches expected result with complete reasoning."}}</json>

Ensure your JSON is properly formatted with no trailing commas or syntax errors.

CRITICAL: Always wrap your response in <json>...</json> tags. Do not output raw JSON without the tags."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON with multiple fallback strategies
        prediction = "None"
        reasoning = None
        raw_text = msg_history[-1]["text"]
        
        # Log the raw response for debugging
        self.log_fn(f"Raw LLM response (first 1000 chars): {raw_text[:1000]}")
        
        # Try all extraction methods in order of preference
        extraction_attempts = [
            ("_extract_jsons", lambda: _extract_jsons(raw_text)),
            ("_extract_json_with_regex", lambda: _extract_json_with_regex(raw_text)),
        ]
        
        for attempt_name, attempt_fn in extraction_attempts:
            try:
                extracted = attempt_fn()
                if extracted and len(extracted) > 0:
                    # Check all extracted JSON objects for "response" key, prefer later ones
                    for obj in reversed(extracted):
                        if isinstance(obj, dict) and "response" in obj:
                            prediction = obj["response"]
                            reasoning = obj.get("reasoning", None)
                            self.log_fn(f"Successfully extracted prediction using {attempt_name}: {prediction}")
                            if reasoning:
                                self.log_fn(f"Grading reasoning: {reasoning}")
                            break
                    if prediction != "None":
                        break
            except Exception as e:
                self.log_fn(f"Extraction attempt {attempt_name} failed: {e}")
                continue
        
        # Last resort: try to extract response value directly from malformed JSON
        if prediction == "None":
            direct_value = _extract_response_value(raw_text)
            if direct_value is not None:
                prediction = direct_value
                self.log_fn(f"Extracted prediction using direct value extraction: {prediction}")
        
        # Final fallback: try to find any numeric value that looks like a score
        if prediction == "None":
            # Look for patterns like "score: 7" or "grade: 5" or just standalone numbers
            score_patterns = [
                r'["\']?score["\']?\s*[:=]\s*["\']?(\d+)["\']?',
                r'["\']?grade["\']?\s*[:=]\s*["\']?(\d+)["\']?',
                r'["\']?points?["\']?\s*[:=]\s*["\']?(\d+)["\']?',
            ]
            for pattern in score_patterns:
                match = re.search(pattern, raw_text, re.IGNORECASE)
                if match:
                    prediction = match.group(1)
                    self.log_fn(f"Extracted prediction using score pattern: {prediction}")
                    break
        
        if prediction == "None":
            self.log_fn(f"Failed to extract prediction from response. Raw text: {raw_text[:500]}")

        return str(prediction), msg_history
