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
    return results or None


def _extract_json_with_brace_matching(text: str) -> dict | None:
    """Extract a JSON object from text using brace counting.
    
    This handles nested JSON objects properly by tracking brace depth.
    Also handles strings that may contain braces.
    """
    start_idx = text.find('{')
    if start_idx == -1:
        return None
    
    brace_count = 1
    i = start_idx + 1
    in_string = False
    escape_next = False
    
    while i < len(text) and brace_count > 0:
        char = text[i]
        
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
                brace_count += 1
            elif char == '}':
                brace_count -= 1
        
        i += 1
    
    if brace_count == 0:
        candidate = text[start_idx:i]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            # Try sanitizing before giving up
            try:
                sanitized = _sanitize_json_string(candidate)
                return json.loads(sanitized)
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
    
    This handles cases where the JSON is malformed but contains a response value.
    Tries multiple patterns to extract numeric or string responses.
    """
    import re
    
    # Pattern 1: Look for "response": <number>
    num_pattern = r'"response"\s*:\s*(\d+(?:\.\d+)?)'
    match = re.search(num_pattern, text)
    if match:
        return match.group(1)
    
    # Pattern 2: Look for "response": "<string>"
    str_pattern = r'"response"\s*:\s*"([^"]*)"'
    match = re.search(str_pattern, text)
    if match:
        return match.group(1)
    
    # Pattern 3: Look for "response": true/false
    bool_pattern = r'"response"\s*:\s*(true|false)'
    match = re.search(bool_pattern, text, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    
    # Pattern 4: Look for response in various quote styles
    alt_pattern = r'[\'"`]response[\'"`]\s*:\s*([\'"`]?)([^\'"`,}\]]*)\1'
    match = re.search(alt_pattern, text)
    if match:
        return match.group(2).strip()
    
    return None


def _validate_numeric_score(prediction: str, max_score: float = 7.0) -> str:
    """Validate and normalize a numeric score prediction.
    
    Args:
        prediction: The extracted prediction value
        max_score: Maximum allowed score (default 7.0 for IMO problems)
        
    Returns:
        Validated and normalized score as string
    """
    try:
        # Try to parse as float
        score = float(prediction)
        # Clamp to valid range [0, max_score]
        score = max(0.0, min(score, max_score))
        # Return as integer if it's a whole number
        if score == int(score):
            return str(int(score))
        return str(score)
    except (ValueError, TypeError):
        # Not a valid number, return original
        return str(prediction)


def _format_grading_reasoning(problem: str, solution: str, student_answer: str, 
                               grading_guidelines: str) -> str:
    """Format a structured reasoning section for the LLM to follow.
    
    This helps the LLM provide more consistent and thorough grading.
    """
    return f"""
STEP-BY-STEP GRADING PROCESS:

1. PROBLEM ANALYSIS:
   - What is the problem asking for?
   - What are the key concepts and techniques needed?
   - What would a complete solution look like?

2. OFFICIAL SOLUTION REVIEW:
   - What is the core insight/technique in the official solution?
   - What are the critical steps that must be present?
   - How many points is each step worth according to the rubric?

3. STUDENT ANSWER EVALUATION:
   - Did the student identify the correct approach?
   - Which steps did they complete correctly?
   - Where did they make errors or skip steps?
   - Is their final answer correct?

4. SCORE ASSIGNMENT:
   - Based on the rubric, what partial credit is deserved?
   - Is the reasoning sound even if the final answer is wrong?
   - Assign a numerical score (typically 0-{grading_guidelines.split('max')[1].split()[0] if 'max' in grading_guidelines.lower() else '7'})

Remember: Be fair but rigorous. Award partial credit for correct reasoning even if the final answer is incorrect.
"""


def _sanitize_json_string(text: str) -> str:
    """Sanitize a JSON string by fixing common formatting issues.
    
    This helps recover from malformed JSON that LLMs sometimes produce.
    """
    # Remove trailing commas before closing braces/brackets
    text = re.sub(r',(\s*[}\]])', r'\1', text)
    
    # Fix single quotes to double quotes for JSON compatibility
    # Only replace quotes that appear to be delimiting keys/values
    text = re.sub(r"(?<=[{,\s])'([^']+)'(?=\s*:)", r'"\1"', text)
    
    # Remove comments (both // and /* */ styles)
    text = re.sub(r'//[^\n]*', '', text)
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    
    return text.strip()


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

        # Get structured reasoning guidance
        reasoning_guide = _format_grading_reasoning(problem, solution, student_answer, grading_guidelines)
        
        instruction = f"""You are an expert mathematical grader evaluating student solutions to competition mathematics problems.

Your task is to grade a student's answer to an IMO-level mathematics problem. You must carefully analyze:
1. The problem statement
2. The official solution
3. The grading guidelines (rubric)
4. The student's submitted answer

Then provide your evaluation in the specified JSON format.

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

{reasoning_guide}

IMPORTANT: Your response MUST be a valid JSON object wrapped in <json> tags. Do not include any text outside the JSON tags.

Respond in the following exact format:
<json>
{{
    "response": <your numerical score or evaluation>
}}
</json>

The "response" field should contain your final grading decision (typically a number or specific evaluation string as specified in the grading guidelines).

Examples of valid responses:
- For a numeric score: <json>{{"response": 7}}</json>
- For a string evaluation: <json>{{"response": "Correct"}}</json>
- For a boolean: <json>{{"response": true}}</json>

Ensure your JSON is properly formatted with no trailing commas or syntax errors."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON with multiple fallback strategies
        prediction = "None"
        raw_text = msg_history[-1]["text"]
        
        # Log the raw response for debugging
        self.log_fn(f"Raw LLM response (first 1000 chars): {raw_text[:1000]}")
        
        extraction_attempts = [
            ("_extract_jsons", lambda: _extract_jsons(raw_text)),
            ("_extract_json_with_regex", lambda: _extract_json_with_regex(raw_text)),
        ]
        
        for attempt_name, attempt_fn in extraction_attempts:
            try:
                extracted = attempt_fn()
                if extracted and len(extracted) > 0:
                    # Check the last extracted JSON object for "response" key
                    last_obj = extracted[-1]
                    if isinstance(last_obj, dict) and "response" in last_obj:
                        prediction = last_obj["response"]
                        self.log_fn(f"Successfully extracted prediction using {attempt_name}: {prediction}")
                        break
            except Exception as e:
                self.log_fn(f"Extraction attempt {attempt_name} failed: {e}")
                continue
        
        # Try sanitizing and re-parsing JSON before giving up
        if prediction == "None":
            try:
                sanitized = _sanitize_json_string(raw_text)
                # Try to find JSON objects in the sanitized text
                sanitized_extracted = _extract_jsons(sanitized)
                if sanitized_extracted and len(sanitized_extracted) > 0:
                    last_obj = sanitized_extracted[-1]
                    if isinstance(last_obj, dict) and "response" in last_obj:
                        prediction = last_obj["response"]
                        self.log_fn(f"Successfully extracted prediction using JSON sanitization: {prediction}")
            except Exception as e:
                self.log_fn(f"JSON sanitization attempt failed: {e}")
        
        # Last resort: try to extract response value directly from malformed JSON
        if prediction == "None":
            direct_value = _extract_response_value(raw_text)
            if direct_value is not None:
                prediction = direct_value
                self.log_fn(f"Extracted prediction using direct value extraction: {prediction}")
        
        if prediction == "None":
            self.log_fn(f"Failed to extract prediction from response. Raw text: {raw_text[:500]}")
        
        # Validate and normalize numeric scores
        prediction = _validate_numeric_score(str(prediction))
        
        return str(prediction), msg_history
