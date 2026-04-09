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
    Also handles markdown code blocks with json tag and inline JSON.
    """
    results = []
    search_from = 0
    
    def _try_parse_json(json_str: str) -> dict | None:
        """Try to parse JSON with common fix attempts."""
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # Try to fix common JSON issues
            # Fix 1: Remove trailing commas before closing braces/brackets
            fixed = re.sub(r',(\s*[}\]])', r'\1', json_str)
            try:
                return json.loads(fixed)
            except json.JSONDecodeError:
                pass
            
            # Fix 2: Fix unescaped newlines in strings
            fixed = re.sub(r'(?<!\\)\n', '\\n', fixed)
            try:
                return json.loads(fixed)
            except json.JSONDecodeError:
                pass
            
            # Fix 3: Fix unescaped tabs in strings
            fixed = re.sub(r'(?<!\\)\t', '\\t', fixed)
            try:
                return json.loads(fixed)
            except json.JSONDecodeError:
                pass
            
            # Fix 4: Fix unescaped carriage returns
            fixed = re.sub(r'(?<!\\)\r', '\\r', fixed)
            try:
                return json.loads(fixed)
            except json.JSONDecodeError:
                pass
            
            # Fix 5: Remove control characters
            fixed = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', fixed)
            try:
                return json.loads(fixed)
            except json.JSONDecodeError:
                pass
        
        return None
    
    # First try <json>...</json> tags
    while True:
        start = text.find("<json>", search_from)
        if start == -1:
            break
        end = text.find("</json>", start)
        if end == -1:
            break
        inner = text[start + 6:end].strip()
        search_from = end + 7
        
        parsed = _try_parse_json(inner)
        if parsed:
            results.append(parsed)
    
    # Also try markdown code blocks ```json ... ```
    search_from = 0
    while True:
        start = text.find("```json", search_from)
        if start == -1:
            break
        start = start + 7  # Skip past ```json
        end = text.find("```", start)
        if end == -1:
            break
        inner = text[start:end].strip()
        search_from = end + 3
        
        parsed = _try_parse_json(inner)
        if parsed:
            results.append(parsed)
    
    # Also try plain markdown code blocks ``` ... ```
    search_from = 0
    while True:
        start = text.find("```", search_from)
        if start == -1:
            break
        # Skip if this is ```json which we already handled
        if text[start:start+7] == "```json":
            search_from = start + 7
            continue
        start = start + 3  # Skip past ```
        end = text.find("```", start)
        if end == -1:
            break
        inner = text[start:end].strip()
        search_from = end + 3
        
        # Only try if it looks like JSON (starts with {)
        if inner.startswith('{'):
            parsed = _try_parse_json(inner)
            if parsed:
                results.append(parsed)
    
    return results or None


def _extract_any_json(text: str) -> list[dict] | None:
    """Fallback JSON extraction that looks for any JSON objects in text.
    
    Uses proper brace counting with string awareness to handle nested objects
    and braces inside strings correctly. Also applies common JSON fixes.
    """
    results = []
    brace_count = 0
    start_idx = -1
    in_string = False
    escape_next = False
    
    def _try_parse_with_fixes(json_str: str) -> dict | None:
        """Try to parse JSON with common fix attempts."""
        fixes = [
            lambda s: s,  # Try raw first
            lambda s: re.sub(r',(\s*[}\]])', r'\1', s),  # Remove trailing commas
            lambda s: re.sub(r'(?<!\\)\n', '\\n', s),  # Fix newlines
            lambda s: re.sub(r'(?<!\\)\t', '\\t', s),  # Fix tabs
            lambda s: re.sub(r'(?<!\\)\r', '\\r', s),  # Fix carriage returns
            lambda s: re.sub(r'[\x00-\x1f\x7f-\x9f]', '', s),  # Remove control chars
        ]
        
        for fix in fixes:
            try:
                fixed = fix(json_str)
                return json.loads(fixed)
            except json.JSONDecodeError:
                continue
        return None
    
    for i, char in enumerate(text):
        if escape_next:
            escape_next = False
            continue
        
        if char == '\\' and in_string:
            escape_next = True
            continue
        
        if char == '"' and not in_string:
            in_string = True
        elif char == '"' and in_string:
            in_string = False
        elif not in_string:
            if char == '{':
                if brace_count == 0:
                    start_idx = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start_idx != -1:
                    json_str = text[start_idx:i+1]
                    obj = _try_parse_with_fixes(json_str)
                    if obj is not None:
                        results.append(obj)
                    start_idx = -1
    
    return results or None


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

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
        # Extract task components for better prompting
        domain = inputs.get("domain", "Unknown")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = f"""You are an expert mathematical grader for International Mathematical Olympiad (IMO) problems.

Your task is to evaluate a student's answer to a mathematics problem and provide a grade.

DOMAIN: {domain}

PROBLEM:
{problem}

OFFICIAL SOLUTION:
{solution}

GRADING GUIDELINES:
{grading_guidelines}

STUDENT'S ANSWER:
{student_answer}

Think step by step:
1. Analyze what the problem is asking - identify key concepts and required steps
2. Review the official solution approach - understand the expected reasoning
3. Compare the student's answer to the official solution - check for correctness and completeness
4. Check if the student followed the grading guidelines - look for partial credit criteria
5. Determine the appropriate grade - be precise and consistent with guidelines

IMPORTANT: Your response MUST be valid JSON wrapped in <json> tags. Do not include any text outside the JSON tags.

Respond in this exact format:
<json>
{{
    "reasoning": "Your detailed chain-of-thought analysis here. Be thorough and specific about what the student did right or wrong.",
    "response": "The final grade/prediction. Use the exact format specified in the grading guidelines (e.g., 'Correct', 'Incorrect', 'Partial', or a numeric score)."
}}
</json>

JSON formatting rules:
- Use double quotes for all strings
- Escape any quotes inside strings with backslash
- Do not use trailing commas
- Keep all text on single lines within the JSON (use \\n for newlines if needed)"""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON with fallback mechanisms
        prediction = "None"
        try:
            last_message = msg_history[-1]["text"]
            
            # Try primary extraction method
            extracted = _extract_jsons(last_message)
            
            # Fallback to generic JSON extraction if primary fails
            if extracted is None:
                extracted = _extract_any_json(last_message)
            
            if extracted:
                # Prefer response field, but accept other common field names
                last_json = extracted[-1]
                
                # Priority order for grade fields
                grade_fields = [
                    "response", "grade", "answer", "result", 
                    "evaluation", "score", "verdict", "prediction",
                    "output", "decision", "assessment", "mark"
                ]
                
                for field in grade_fields:
                    if field in last_json:
                        value = last_json[field]
                        if isinstance(value, str):
                            prediction = value.strip()
                        elif isinstance(value, (int, float, bool)):
                            prediction = str(value)
                        elif isinstance(value, list) and len(value) > 0:
                            # Handle case where grade is a list
                            prediction = str(value[0])
                        break
                else:
                    # If no known field, use the first string or numeric value found
                    for key, value in last_json.items():
                        if key == "reasoning":
                            continue  # Skip reasoning field
                        if isinstance(value, str) and value.strip():
                            prediction = value.strip()
                            break
                        elif isinstance(value, (int, float)):
                            prediction = str(value)
                            break
                        elif isinstance(value, bool):
                            prediction = "Correct" if value else "Incorrect"
                            break
            else:
                # Last resort: try to find grade-like text in the response
                # Look for common grade patterns
                text_lower = last_message.lower()
                if "grade: correct" in text_lower or '"correct"' in text_lower:
                    prediction = "Correct"
                elif "grade: incorrect" in text_lower or '"incorrect"' in text_lower:
                    prediction = "Incorrect"
                elif "grade: partial" in text_lower or '"partial"' in text_lower:
                    prediction = "Partial"
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
