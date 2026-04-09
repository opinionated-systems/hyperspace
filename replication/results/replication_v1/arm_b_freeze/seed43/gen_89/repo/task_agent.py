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
    Collects from ALL sources, not just the first one that yields results.
    """
    results = []
    search_from = 0
    
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
        if parsed is not None:
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
        if parsed is not None:
            results.append(parsed)
    
    # Try ``` ... ``` without json tag
    search_from = 0
    while True:
        start = text.find("```", search_from)
        if start == -1:
            break
        start = start + 3  # Skip past ```
        end = text.find("```", start)
        if end == -1:
            break
        inner = text[start:end].strip()
        search_from = end + 3
        # Only try if it looks like JSON
        if inner.startswith("{") or inner.startswith("["):
            parsed = _try_parse_json(inner)
            if parsed is not None:
                results.append(parsed)
    
    return results or None


def _try_parse_json(text: str) -> dict | None:
    """Try to parse JSON with multiple fallback fixes.
    
    Returns the parsed dict or None if parsing fails.
    """
    # First try direct parsing
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Try progressively more aggressive fixes
    fixes = [
        # Fix 1: Remove trailing commas before closing braces/brackets
        lambda t: re.sub(r',(\s*[}\]])', r'\1', t),
        # Fix 2: Fix single quotes to double quotes (common LLM mistake)
        lambda t: re.sub(r"'([^']*?)'", r'"\1"', t),
        # Fix 3: Fix unescaped newlines in strings
        lambda t: re.sub(r'(?<!\\)\n', r'\\n', t),
        # Fix 4: Fix unescaped tabs in strings
        lambda t: re.sub(r'(?<!\\)\t', r'\\t', t),
        # Fix 5: Fix unescaped carriage returns
        lambda t: re.sub(r'(?<!\\)\r', r'\\r', t),
        # Fix 6: Remove comments (// and /* */)
        lambda t: re.sub(r'//[^\n]*', '', t),
        lambda t: re.sub(r'/\*.*?\*/', '', t, flags=re.DOTALL),
        # Fix 7: Fix missing quotes around keys
        lambda t: re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', t),
        # Fix 8: Remove control characters
        lambda t: re.sub(r'[\x00-\x1f\x7f-\x9f]', '', t),
        # Fix 9: Fix escaped single quotes
        lambda t: t.replace("\\'", "'"),
        # Fix 10: Remove BOM if present
        lambda t: t.lstrip('\ufeff'),
    ]
    
    current = text
    for fix in fixes:
        try:
            current = fix(current)
            return json.loads(current)
        except json.JSONDecodeError:
            continue
    
    # Try all combinations of fixes
    try:
        # Apply all fixes at once
        fixed = text
        for fix in fixes:
            fixed = fix(fixed)
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    
    # Final attempt: try to extract just the first valid JSON object
    try:
        # Find the first { and last } that might form a valid object
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and start < end:
            candidate = text[start:end+1]
            # Apply all fixes to the candidate
            for fix in fixes:
                candidate = fix(candidate)
            return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    
    return None


def _extract_any_json(text: str) -> list[dict] | None:
    """Fallback JSON extraction that looks for any JSON objects in text.
    
    Uses a stack-based approach to properly handle nested braces.
    """
    results = []
    # Try to find JSON objects between curly braces
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
                    parsed = _try_parse_json(json_str)
                    if parsed is not None:
                        results.append(parsed)
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

GRADING PRINCIPLES:
- Be fair and consistent with the official grading guidelines
- Award partial credit when the student shows correct reasoning but makes minor errors
- Consider alternative valid approaches that differ from the official solution
- Check for logical structure and mathematical rigor
- Note any missing steps or unjustified claims
- IMO problems are typically graded on a scale of 0-7 points
- A complete correct solution receives 7 points
- Partial credit (1-6 points) is awarded for partial progress

IMPORTANT: Your response MUST be valid JSON wrapped in <json> tags. Do not include any text outside the JSON tags.

Respond in this exact format:
<json>
{{
    "reasoning": "Your detailed chain-of-thought analysis here. Be thorough and specific about what the student did right or wrong.",
    "response": "The final grade/prediction. Use the exact format specified in the grading guidelines (e.g., 'Correct', 'Incorrect', 'Partial', or a numeric score like '7' or '0')."
}}
</json>"""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON with multiple fallback mechanisms
        prediction = self._extract_prediction(msg_history)

        return str(prediction), msg_history

    def _extract_prediction(self, msg_history: list[dict]) -> str:
        """Extract prediction from message history with robust fallback mechanisms.
        
        Args:
            msg_history: List of message dicts from LLM conversation
            
        Returns:
            Extracted prediction string or "None" if extraction fails
        """
        if not msg_history:
            return "None"
        
        try:
            last_message = msg_history[-1]["text"]
            
            # Try primary extraction method (tagged JSON blocks)
            extracted = _extract_jsons(last_message)
            
            # Fallback to generic JSON extraction if primary fails
            if not extracted:
                extracted = _extract_any_json(last_message)
            
            if extracted:
                return self._get_prediction_from_json(extracted[-1])
            
            # Final fallback: look for common grade patterns in raw text
            return self._extract_from_raw_text(last_message)
            
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            return "None"

    def _get_prediction_from_json(self, json_obj: dict) -> str:
        """Extract prediction value from a JSON object.
        
        Tries multiple common field names in order of preference.
        """
        # Ordered list of field names to try (most specific first)
        field_names = [
            "response", "grade", "evaluation", "verdict", 
            "result", "answer", "score", "prediction", "output"
        ]
        
        for field in field_names:
            if field in json_obj:
                value = json_obj[field]
                if isinstance(value, str):
                    return value.strip()
                elif isinstance(value, (int, float)):
                    return str(value)
                elif isinstance(value, bool):
                    return "Correct" if value else "Incorrect"
        
        # If no known field, use the first string or numeric value found
        for key, value in json_obj.items():
            if isinstance(value, str):
                return value.strip()
            elif isinstance(value, (int, float)):
                return str(value)
        
        return "None"

    def _extract_from_raw_text(self, text: str) -> str:
        """Extract grade from raw text when JSON parsing fails.
        
        Looks for common grade patterns and keywords with improved accuracy.
        Uses a priority-based approach to determine the most likely grade.
        """
        text_lower = text.lower()
        
        # Priority 1: Check for explicit grade statements with comprehensive patterns
        grade_patterns = [
            (r'final[\s]+grade[\s]*[:=][\s]*([\w\s-]+?)(?:\n|$|\.|\,)', 1),
            (r'grade[\s]*[:=][\s]*([\w\s-]+?)(?:\n|$|\.|\,)', 1),
            (r'evaluation[\s]*[:=][\s]*([\w\s-]+?)(?:\n|$|\.|\,)', 1),
            (r'response[\s]*[:=][\s]*([\w\s-]+?)(?:\n|$|\.|\,)', 1),
            (r'verdict[\s]*[:=][\s]*([\w\s-]+?)(?:\n|$|\.|\,)', 1),
            (r'score[\s]*[:=][\s]*([\d/\.]+)(?:\n|$|\.|\,)', 1),
            (r'(?:^|\n)[\s]*([\d]+)[\s]*(?:out of|/)[\s]*([\d]+)', 0),  # "7 out of 7" or "7/7"
        ]
        
        for pattern, group in grade_patterns:
            match = re.search(pattern, text_lower)
            if match:
                result = match.group(group).strip()
                # Clean up common artifacts
                result = re.sub(r'[\'"\n\r]', '', result)
                # Validate result - should not be empty and should have some substance
                if result and len(result) > 0 and result not in ['', ' ', '.', ',']:
                    return result
        
        # Priority 2: Check for numeric scores (e.g., "7/7", "5 points", "score: 10")
        numeric_patterns = [
            r'\b(\d{1,2})\s*/\s*(\d{1,2})\b',  # "7/7", "3/7"
            r'\b(\d{1,2})\s*points?\b',  # "7 points", "5 point"
            r'\bscore\s*[:=]?\s*(\d+)\b',  # "score: 7", "score 5"
        ]
        for pattern in numeric_patterns:
            match = re.search(pattern, text_lower)
            if match:
                return match.group(1)
        
        # Priority 3: Check for IMO-specific grade patterns
        imo_patterns = [
            r'\b(0|1|2|3|4|5|6|7)\s*/\s*7\b',  # IMO scores are typically out of 7
            r'\bscore[d]?\s+(0|1|2|3|4|5|6|7)\b',
            r'\b(0|1|2|3|4|5|6|7)\s*point',
        ]
        for pattern in imo_patterns:
            match = re.search(pattern, text_lower)
            if match:
                return match.group(1)
        
        # Priority 4: Check for binary correct/incorrect indicators with context awareness
        has_correct = re.search(r'\bcorrect\b', text_lower)
        has_incorrect = re.search(r'\bincorrect\b|\bwrong\b|\berror\b', text_lower)
        has_partial = re.search(r'\bpartial\b|\bpartially\b|\bincomplete\b', text_lower)
        
        # Determine based on presence and context
        if has_partial and (has_correct or has_incorrect):
            return "Partial"
        if has_incorrect and not has_correct:
            return "Incorrect"
        if has_correct and not has_incorrect:
            return "Correct"
        
        # Priority 5: Check for explicit true/false indicators
        if re.search(r'\btrue\b|\byes\b|\bvalid\b', text_lower) and not re.search(r'\bfalse\b|\bno\b|\binvalid\b', text_lower):
            return "Correct"
        if re.search(r'\bfalse\b|\bno\b|\binvalid\b', text_lower) and not re.search(r'\btrue\b|\byes\b|\bvalid\b', text_lower):
            return "Incorrect"
        
        # Priority 6: Check for pass/fail indicators
        if re.search(r'\bpass\b|\bpassed\b|\baccept\b|\baccepted\b', text_lower) and not re.search(r'\bfail\b|\bfailed\b|\breject\b|\brejected\b', text_lower):
            return "Correct"
        if re.search(r'\bfail\b|\bfailed\b|\breject\b|\brejected\b', text_lower) and not re.search(r'\bpass\b|\bpassed\b|\baccept\b|\baccepted\b', text_lower):
            return "Incorrect"
        
        return "None"
