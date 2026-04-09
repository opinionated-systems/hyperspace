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
    Also handles nested JSON objects and common formatting issues.
    Enhanced to handle multiple tag formats and edge cases.
    
    Args:
        text: The input text containing <json> tags.
        
    Returns:
        A list of parsed JSON objects, or None if no valid JSON found.
    """
    if not text or not isinstance(text, str):
        return None
    
    # Pre-process: clean up escaped quotes that might confuse parsing
    text = text.replace('\\"', '"').replace("\\'", "'")
        
    results = []
    search_from = 0
    max_iterations = 100  # Safety limit to prevent infinite loops
    iterations = 0
    
    # Support multiple tag formats
    tag_formats = [
        ("<json>", "</json>"),
        ("<JSON>", "</JSON>"),
        ("<json>", "</JSON>"),
        ("<JSON>", "</json>"),
    ]
    
    while iterations < max_iterations:
        iterations += 1
        
        # Try all tag formats
        found = False
        for open_tag, close_tag in tag_formats:
            start = text.find(open_tag, search_from)
            if start != -1:
                end = text.find(close_tag, start)
                if end != -1:
                    inner = text[start + len(open_tag):end].strip()
                    search_from = end + len(close_tag)
                    found = True
                    
                    # Skip empty content
                    if not inner:
                        continue
                    
                    # Try to parse the JSON with multiple recovery strategies
                    parsed = _try_parse_json_with_recovery(inner)
                    if parsed is not None:
                        results.append(parsed)
                    break
        
        if not found:
            break
    
    return results or None


def _try_parse_json_with_recovery(text: str) -> dict | list | None:
    """Attempt to parse JSON with multiple recovery strategies.
    
    Args:
        text: The JSON string to parse.
        
    Returns:
        The parsed JSON object, or None if all strategies fail.
    """
    if not text:
        return None
    
    # Pre-process: strip whitespace and common prefixes
    text = text.strip()
    
    # Strategy 1: Direct parsing
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Strategy 2: Remove trailing commas before closing braces/brackets
    try:
        fixed = re.sub(r',(\s*[}\]])', r'\1', text)
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    
    # Strategy 3: Extract first valid JSON object using brace counting
    try:
        brace_start = text.find('{')
        if brace_start != -1:
            brace_count = 0
            for i, char in enumerate(text[brace_start:]):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        return json.loads(text[brace_start:brace_start+i+1])
    except json.JSONDecodeError:
        pass
    
    # Strategy 4: Extract first valid JSON array using bracket counting
    try:
        bracket_start = text.find('[')
        if bracket_start != -1:
            bracket_count = 0
            for i, char in enumerate(text[bracket_start:]):
                if char == '[':
                    bracket_count += 1
                elif char == ']':
                    bracket_count -= 1
                    if bracket_count == 0:
                        return json.loads(text[bracket_start:bracket_start+i+1])
    except json.JSONDecodeError:
        pass
    
    # Strategy 5: Fix single quotes to double quotes (carefully)
    try:
        # Only replace single quotes that are not inside double-quoted strings
        fixed = ""
        in_double_quotes = False
        for char in text:
            if char == '"' and (not fixed or fixed[-1] != '\\'):
                in_double_quotes = not in_double_quotes
                fixed += char
            elif char == "'" and not in_double_quotes:
                fixed += '"'
            else:
                fixed += char
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    
    # Strategy 6: Remove comments and retry
    try:
        cleaned = re.sub(r'//.*?\n', '\n', text)
        cleaned = re.sub(r'/\*.*?\*/', '', cleaned, flags=re.DOTALL)
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    
    # Strategy 7: Fix common JSON syntax errors
    try:
        # Fix unquoted keys (simple cases only)
        fixed = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', text)
        # Fix trailing commas
        fixed = re.sub(r',(\s*[}\]])', r'\1', fixed)
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    
    # Strategy 8: Extract JSON from mixed content (find first { or [)
    try:
        # Find the first occurrence of { or [
        brace_idx = text.find('{')
        bracket_idx = text.find('[')
        
        if brace_idx != -1 and (bracket_idx == -1 or brace_idx < bracket_idx):
            # Start with brace
            start = brace_idx
            count = 0
            for i, char in enumerate(text[start:]):
                if char == '{':
                    count += 1
                elif char == '}':
                    count -= 1
                    if count == 0:
                        return json.loads(text[start:start+i+1])
        elif bracket_idx != -1:
            # Start with bracket
            start = bracket_idx
            count = 0
            for i, char in enumerate(text[start:]):
                if char == '[':
                    count += 1
                elif char == ']':
                    count -= 1
                    if count == 0:
                        return json.loads(text[start:start+i+1])
    except json.JSONDecodeError:
        pass
    
    # All strategies failed
    return None


def _extract_json_from_markdown(text: str) -> list[dict] | None:
    """Extract JSON objects from markdown code blocks.

    Fallback for when <json> tags are not used but markdown code blocks are.
    Enhanced to handle nested braces and common formatting issues.
    Uses the shared _try_parse_json_with_recovery helper for consistency.
    
    Args:
        text: The input text containing markdown code blocks.
        
    Returns:
        A list of parsed JSON objects, or None if no valid JSON found.
    """
    if not text or not isinstance(text, str):
        return None
    
    # Pre-process: clean up escaped quotes
    text = text.replace('\\"', '"').replace("\\'", "'")
        
    results = []
    
    # Match various markdown code block formats
    patterns = [
        r'```json\s*\n?(.*?)\n?```',  # ```json ... ```
        r'```JSON\s*\n?(.*?)\n?```',  # ```JSON ... ```
        r'```\s*\n?(.*?)\n?```',       # ``` ... ``` (any content)
        r'`\s*\n?(.*?)\n?`',            # ` ... ` (inline code)
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        
        for match in matches:
            match = match.strip()
            if not match:
                continue
                
            # Use the shared recovery helper for consistent parsing
            parsed = _try_parse_json_with_recovery(match)
            if parsed is not None:
                results.append(parsed)
    
    return results or None


# Cache for normalized grades to improve performance
_normalize_grade_cache: dict[str, str] = {}


def _normalize_grade(grade: str) -> str:
    """Normalize grade to a standard format.

    Handles various grade formats and normalizes them.
    Enhanced to handle numeric grades (IMO-style 0-7 scale) and more variations.
    Includes improved handling for edge cases and ambiguous inputs.
    Uses caching to improve performance for repeated calls.
    """
    if not isinstance(grade, str):
        grade = str(grade)
    
    original = grade.strip()
    # Remove surrounding quotes and common punctuation
    original = original.strip('"\'.,;:')
    
    # Check cache first for performance
    cache_key = original.lower()
    if cache_key in _normalize_grade_cache:
        return _normalize_grade_cache[cache_key]
    
    grade = cache_key
    
    # Handle empty strings
    if not grade:
        _normalize_grade_cache[cache_key] = original
        return original
    
    # First, check for numeric grades (0, 1, 2, etc.)
    # IMO-style grading: 0-7 scale where 7 is full marks
    try:
        numeric_grade = float(grade)
        # IMO scale: 0 = no progress, 1-6 = partial progress, 7 = full solution
        # For binary classification: 0 = Incorrect, 1-7 = Correct
        # For ternary: 0 = Incorrect, 1-6 = Partial, 7 = Correct
        if numeric_grade == 0:
            _normalize_grade_cache[cache_key] = 'Incorrect'
            return 'Incorrect'
        elif numeric_grade >= 6:  # 6 or 7 on IMO scale = essentially correct
            _normalize_grade_cache[cache_key] = 'Correct'
            return 'Correct'
        else:  # 1-5 = partial credit
            _normalize_grade_cache[cache_key] = 'Partial'
            return 'Partial'
    except ValueError:
        pass
    
    # Map common variations to standard formats
    correct_variations = [
        'correct', 'right', 'true', 'yes', 'full', 'full credit', 
        'full marks', 'complete', 'valid', 'accepted', 'pass', 'solved',
        'success', 'accurate', 'perfect', 'excellent', 'good', '7', '6',
        'fully correct', 'entirely correct', 'completely correct',
        'correct answer', 'correct solution', 'is correct', 'correct.',
        'right answer', 'valid solution', 'acceptable', 'satisfactory',
        'meets requirements', 'meets criteria', 'full points', 'full score'
    ]
    incorrect_variations = [
        'incorrect', 'wrong', 'false', 'no', 'none', 'zero', '0',
        'invalid', 'rejected', 'fail', 'error', 'mistake', 'unsolved',
        'failure', 'bad', 'unacceptable', 'unsatisfactory',
        'completely wrong', 'entirely wrong', 'totally wrong',
        'is incorrect', 'not correct', 'not right', 'incorrect.',
        'wrong answer', 'invalid solution', 'does not meet',
        'fails', 'failed', 'no credit', 'zero points', 'zero score'
    ]
    partial_variations = [
        'partial', 'partial credit', 'half', 'incomplete',
        'partially correct', 'partially right', 'some credit',
        'partially solved', 'almost', 'nearly', 'minor errors',
        '1', '2', '3', '4', '5', 'partially', 'somewhat',
        'mostly correct', 'mostly right', 'incomplete solution',
        'partial solution', 'partial answer', 'partial.',
        'partially acceptable', 'partially valid', 'partial success',
        'some progress', 'limited success', 'incomplete work',
        'partial points', 'partial score', 'half credit', 'half marks'
    ]
    
    # Check for exact matches first, then substring matches
    if grade in correct_variations:
        _normalize_grade_cache[cache_key] = 'Correct'
        return 'Correct'
    if grade in incorrect_variations:
        _normalize_grade_cache[cache_key] = 'Incorrect'
        return 'Incorrect'
    if grade in partial_variations:
        _normalize_grade_cache[cache_key] = 'Partial'
        return 'Partial'
    
    # Substring matching for more flexibility (but be careful not to over-match)
    # Check for strong indicators at word boundaries
    import re
    
    # Check for "correct" as a standalone word (but not "incorrect")
    if re.search(r'\bcorrect\b', grade) and not re.search(r'\b(in|not|partially|mostly)\s+correct\b', grade):
        _normalize_grade_cache[cache_key] = 'Correct'
        return 'Correct'
    
    # Check for "incorrect" or "wrong"
    if re.search(r'\b(incorrect|wrong|false)\b', grade):
        _normalize_grade_cache[cache_key] = 'Incorrect'
        return 'Incorrect'
    
    # Check for partial indicators
    if re.search(r'\b(partial|incomplete|partially|mostly|almost|nearly|some)\b', grade):
        _normalize_grade_cache[cache_key] = 'Partial'
        return 'Partial'
    
    # Check for negation patterns that indicate incorrect
    if re.search(r'\b(not|no|never|none|without|fails?|failed)\s+(?:correct|right|valid|pass|solve|solution)\b', grade):
        _normalize_grade_cache[cache_key] = 'Incorrect'
        return 'Incorrect'
    
    # Check for affirmation patterns that indicate correct
    if re.search(r'\b(is|was|seems|appears|looks)\s+(?:correct|right|valid|good|acceptable)\b', grade):
        _normalize_grade_cache[cache_key] = 'Correct'
        return 'Correct'
    
    # Return original if no normalization applied (preserve case)
    _normalize_grade_cache[cache_key] = original
    return original


class TaskAgent:
    """Task agent that solves IMO grading problems with enhanced reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def _build_grading_prompt(self, inputs: dict) -> str:
        """Build a structured grading prompt with clear evaluation criteria."""
        domain = inputs.get("domain", "")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        return f"""You are an expert mathematical grader specializing in {domain} problems.

Your task is to evaluate a student's answer to a mathematics problem and assign a grade.

## Problem Statement:
{problem}

## Official Solution:
{solution}

## Grading Guidelines:
{grading_guidelines}

## Student's Answer:
{student_answer}

## Evaluation Framework:

Follow this structured evaluation process carefully:

### Step 1: Problem Understanding
- What is the problem asking for? Identify the core question.
- What are the key constraints, conditions, and given information?
- What is the expected answer format (numeric, proof, construction, etc.)?

### Step 2: Solution Analysis
- What is the correct approach according to the official solution?
- What are the critical steps, theorems, or techniques that must be present?
- What constitutes a complete solution vs. an incomplete one?
- Are there multiple valid approaches? If so, does the student's approach align with any valid method?

### Step 3: Student Answer Evaluation
- Did the student understand the problem correctly?
- What approach did the student take? Is it mathematically sound?
- Are the student's steps logically valid and well-reasoned?
- Did the student show sufficient work and reasoning, or just state answers?
- Is the final answer mathematically correct?
- Did the student make any computational errors, conceptual errors, or logical gaps?

### Step 4: Grade Assignment
Based on the grading guidelines, assign the appropriate grade considering:
- Correctness of the final answer (most important)
- Validity and clarity of the reasoning process
- Completeness of the solution (all required steps present)
- Adherence to mathematical rigor and notation

## Grade Definitions (STRICT - USE ONLY THESE):

- **Correct**: The student's answer is completely correct with valid reasoning, correct final answer, and complete solution. Minor notation issues that don't affect correctness are acceptable.
- **Incorrect**: The student's answer is completely wrong, has no valid reasoning, the final answer is incorrect, or shows no meaningful progress toward the solution.
- **Partial**: The student's answer has some valid elements (correct approach, partial progress, or correct answer with flawed reasoning) but is incomplete, has significant errors, or lacks proper justification.

For IMO-style problems with numeric grades (0-7 scale):
- **0**: No progress or completely wrong → Grade as "Incorrect"
- **1-5**: Some progress but incomplete, has errors, or incorrect final answer → Grade as "Partial"
- **6-7**: Correct or nearly correct solution with valid reasoning → Grade as "Correct"

## Response Format (CRITICAL - FOLLOW EXACTLY):

You MUST respond in JSON format wrapped in <json> tags. The response field must contain ONLY one of: "Correct", "Incorrect", or "Partial".

<json>
{{
    "reasoning": "Your detailed step-by-step analysis following the Evaluation Framework above. Be specific about what the student did right/wrong.",
    "response": "Correct"
}}
</json>

OR

<json>
{{
    "reasoning": "Your detailed step-by-step analysis following the Evaluation Framework above. Be specific about what the student did right/wrong.",
    "response": "Incorrect"
}}
</json>

OR

<json>
{{
    "reasoning": "Your detailed step-by-step analysis following the Evaluation Framework above. Be specific about what the student did right/wrong.",
    "response": "Partial"
}}
</json>

CRITICAL RULES - FOLLOW EXACTLY:
1. The "response" field must contain ONLY one of these exact values: "Correct", "Incorrect", or "Partial" (case-sensitive)
2. Do NOT include explanations, quotes, or additional text in the response field
3. Do NOT use numeric grades (0, 1, 2, etc.) in the response field - use ONLY the text labels above
4. The reasoning field should contain your full analysis with specific details
5. Always wrap your entire response in <json>...</json> tags
6. Ensure the JSON is valid - no trailing commas, proper quotes, etc."""

    def _extract_prediction(self, msg_history: list[dict]) -> str:
        """Extract prediction from message history with multiple fallback strategies.
        
        Uses a prioritized extraction approach with early termination on success.
        Includes enhanced pattern matching for edge cases and malformed responses.
        """
        if not msg_history:
            return "None"
        
        last_message = msg_history[-1].get("text", "")
        if not last_message:
            return "None"
        
        # Pre-process: clean up common formatting issues
        cleaned_message = last_message.replace('\\"', '"').replace("\\'", "'")
        
        # Strategy 1: Extract from <json> tags (most reliable)
        extracted = _extract_jsons(cleaned_message)
        if extracted:
            result = self._get_grade_from_json(extracted[-1])
            if result != "None":
                return result
        
        # Strategy 2: Extract from markdown code blocks
        extracted = _extract_json_from_markdown(cleaned_message)
        if extracted:
            result = self._get_grade_from_json(extracted[-1])
            if result != "None":
                return result
        
        # Strategy 3: Look for grade patterns in plain text
        result = self._extract_grade_from_text(cleaned_message)
        if result != "None":
            return result
        
        # Strategy 4: Look for explicit grade statements with more patterns
        explicit_patterns = [
            r'["\']?(?:the\s+)?(?:final\s+)?grade\s*(?:is|was|should\s+be)\s*["\']?\s*(Correct|Incorrect|Partial)\s*["\']?',
            r'["\']?(?:the\s+)?(?:final\s+)?response\s*(?:is|was|should\s+be)\s*["\']?\s*(Correct|Incorrect|Partial)\s*["\']?',
            r'["\']?(?:the\s+)?(?:student\s+)?(?:answer|solution)\s*(?:is|was)\s*["\']?\s*(Correct|Incorrect|Partial)\s*["\']?',
            r'\b(Correct|Incorrect|Partial)\b\s*(?:grade|assessment|evaluation)',
            r'(?:assessment|evaluation|verdict)\s*:\s*["\']?\s*(Correct|Incorrect|Partial)\s*["\']?',
            r'["\']?(?:I\s+)?(?:would\s+)?(?:assign|give|award)\s*["\']?\s*(Correct|Incorrect|Partial)',
            r'(?:therefore|thus|hence|so)\s*,?\s*(?:the\s+)?(?:grade|answer|result)\s*(?:is|should\s+be)\s*["\']?\s*(Correct|Incorrect|Partial)',
        ]
        for pattern in explicit_patterns:
            match = re.search(pattern, cleaned_message, re.IGNORECASE)
            if match:
                grade = match.group(1) if match.lastindex else match.group(0)
                normalized = _normalize_grade(grade)
                if normalized in ['Correct', 'Incorrect', 'Partial']:
                    return normalized
        
        # Strategy 5: Last resort - check if the entire message is just a grade
        stripped = cleaned_message.strip()
        if stripped in ['0', '1', '2', '3', '4', '5', '6', '7']:
            return _normalize_grade(stripped)
        if stripped.lower() in ['correct', 'incorrect', 'partial']:
            return _normalize_grade(stripped)
        
        # Strategy 6: Check for grade at the very end of the message
        last_lines = stripped.split('\n')[-5:]  # Check last 5 lines for more coverage
        for line in reversed(last_lines):
            line_clean = line.strip().strip('"\'.,;:')
            if line_clean.lower() in ['correct', 'incorrect', 'partial']:
                return _normalize_grade(line_clean)
            # Check for "Grade: X" or "Response: X" at end
            if ':' in line_clean:
                parts = line_clean.split(':', 1)
                if len(parts) == 2:
                    key = parts[0].strip().lower()
                    value = parts[1].strip().strip('"\'.,;:')
                    if key in ['grade', 'response', 'result', 'answer', 'evaluation', 'assessment', 'verdict']:
                        normalized = _normalize_grade(value)
                        if normalized in ['Correct', 'Incorrect', 'Partial']:
                            return normalized
        
        # Strategy 7: Check for JSON-like structures without proper tags
        # Look for {"response": "..."} or {"grade": "..."} patterns
        json_like_patterns = [
            r'\{\s*["\']?response["\']?\s*:\s*["\']?(Correct|Incorrect|Partial)["\']?\s*\}',
            r'\{\s*["\']?grade["\']?\s*:\s*["\']?(Correct|Incorrect|Partial)["\']?\s*\}',
            r'\{\s*["\']?result["\']?\s*:\s*["\']?(Correct|Incorrect|Partial)["\']?\s*\}',
            r'["\']?response["\']?\s*:\s*["\']?(Correct|Incorrect|Partial)["\']?',
            r'["\']?grade["\']?\s*:\s*["\']?(Correct|Incorrect|Partial)["\']?',
        ]
        for pattern in json_like_patterns:
            match = re.search(pattern, cleaned_message, re.IGNORECASE)
            if match:
                grade = match.group(1)
                normalized = _normalize_grade(grade)
                if normalized in ['Correct', 'Incorrect', 'Partial']:
                    return normalized
        
        return "None"

    def _get_grade_from_json(self, json_obj: dict) -> str:
        """Extract grade from JSON object with field priority.
        
        Uses a comprehensive field priority system with type-aware handling.
        Includes special handling for nested structures and common variations.
        """
        if not isinstance(json_obj, dict):
            return "None"
        
        # Priority order for grade fields (most specific to least specific)
        priority_fields = [
            "response",   # The task agent's expected output field (highest priority)
            "grade",      # Most specific - explicitly a grade
            "result",     # Generic result field
            "answer",     # Answer field
            "evaluation", # Evaluation result
            "verdict",    # Verdict/assessment
            "assessment", # Assessment field
            "conclusion", # Conclusion field
            "determination", # Determination field
            "score",      # Numeric score
            "rating",     # Rating field
            "status",     # Status field
            "outcome",    # Outcome field
        ]
        
        # First pass: check priority fields
        for field in priority_fields:
            if field in json_obj:
                value = json_obj[field]
                if isinstance(value, str):
                    normalized = _normalize_grade(value)
                    if normalized in ['Correct', 'Incorrect', 'Partial']:
                        return normalized
                elif isinstance(value, (int, float)):
                    # Convert numeric to string and normalize
                    normalized = _normalize_grade(str(value))
                    if normalized in ['Correct', 'Incorrect', 'Partial']:
                        return normalized
                elif isinstance(value, bool):
                    # Handle boolean values
                    return 'Correct' if value else 'Incorrect'
                elif isinstance(value, list) and len(value) > 0:
                    # Handle list values - try first element
                    first_val = value[0]
                    if isinstance(first_val, str):
                        normalized = _normalize_grade(first_val)
                        if normalized in ['Correct', 'Incorrect', 'Partial']:
                            return normalized
        
        # Second pass: check for nested grade objects
        for key, value in json_obj.items():
            if isinstance(value, dict):
                # Recursively check nested objects
                nested_result = self._get_grade_from_json(value)
                if nested_result != "None":
                    return nested_result
        
        # Third pass: use the first string or numeric value found
        for key, value in json_obj.items():
            if isinstance(value, str):
                normalized = _normalize_grade(value)
                if normalized in ['Correct', 'Incorrect', 'Partial']:
                    return normalized
            elif isinstance(value, (int, float)):
                normalized = _normalize_grade(str(value))
                if normalized in ['Correct', 'Incorrect', 'Partial']:
                    return normalized
            elif isinstance(value, bool):
                return 'Correct' if value else 'Incorrect'
        
        return "None"

    def _extract_grade_from_text(self, text: str) -> str:
        """Extract grade from plain text using pattern matching.
        
        Uses multiple pattern strategies with priority ordering for robust extraction.
        """
        # Pre-process text: clean up escaped quotes and normalize whitespace
        text = text.replace('\\"', '"').replace("\\'", "'").replace('  ', ' ')
        
        # Priority 1: Look for explicit grade statements with improved patterns
        patterns = [
            r'grade[\s]*[:=][\s]*["\']?([^"\'\n]+)["\']?',
            r'response[\s]*[:=][\s]*["\']?([^"\'\n]+)["\']?',
            r'final grade[\s]*[:=][\s]*["\']?([^"\'\n]+)["\']?',
            r'assign[\s]+["\']?([^"\'\n]+)["\']?',
            r'["\']?grade["\']?\s*is\s*["\']?([^"\'\n]+)["\']?',
            r'["\']?score["\']?\s*[:=]\s*["\']?([^"\'\n]+)["\']?',
            r'["\']?result["\']?\s*[:=]\s*["\']?([^"\'\n]+)["\']?',
            r'["\']?evaluation["\']?\s*[:=]\s*["\']?([^"\'\n]+)["\']?',
            r'["\']?answer["\']?\s*[:=]\s*["\']?([^"\'\n]+)["\']?',
            r'["\']?verdict["\']?\s*[:=]\s*["\']?([^"\'\n]+)["\']?',
            r'["\']?assessment["\']?\s*[:=]\s*["\']?([^"\'\n]+)["\']?',
            r'["\']?conclusion["\']?\s*[:=]\s*["\']?([^"\'\n]+)["\']?',
            r'["\']?determination["\']?\s*[:=]\s*["\']?([^"\'\n]+)["\']?',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                result = _normalize_grade(match.group(1).strip())
                if result in ['Correct', 'Incorrect', 'Partial']:
                    return result
        
        # Priority 2: Look for standalone grades at the end of the text (common pattern)
        lines = text.strip().split('\n')
        for line in reversed(lines[-20:]):  # Check last 20 lines for more coverage
            line = line.strip()
            # Skip empty lines and common non-grade lines
            if not line or line.startswith('#') or line.startswith('//') or line.startswith('*') or line.startswith('{'):
                continue
            # Look for common grade patterns
            grade_patterns = [
                r'^\s*([0-7])\s*$',  # Single digit 0-7 (IMO scale)
                r'^\s*["\']?([0-7])["\']?\s*$',  # Digit in quotes
                r'^\s*["\']?(Correct|Incorrect|Partial)["\']?\s*$',  # Standard grades
                r'^\s*["\']?(correct|incorrect|partial)["\']?\s*$',  # Lowercase
                r'^\s*Grade:\s*["\']?([^"\'\n]+)["\']?\s*$',  # "Grade: X" format
                r'^\s*The grade is\s*["\']?([^"\'\n]+)["\']?\s*$',  # "The grade is X"
                r'^\s*Response:\s*["\']?([^"\'\n]+)["\']?\s*$',  # "Response: X" format
                r'^\s*Result:\s*["\']?([^"\'\n]+)["\']?\s*$',  # "Result: X" format
                r'^\s*Answer:\s*["\']?([^"\'\n]+)["\']?\s*$',  # "Answer: X" format
                r'^\s*Evaluation:\s*["\']?([^"\'\n]+)["\']?\s*$',  # "Evaluation: X" format
                r'^\s*Assessment:\s*["\']?([^"\'\n]+)["\']?\s*$',  # "Assessment: X" format
                r'^\s*Verdict:\s*["\']?([^"\'\n]+)["\']?\s*$',  # "Verdict: X" format
                r'^\s*Conclusion:\s*["\']?([^"\'\n]+)["\']?\s*$',  # "Conclusion: X" format
            ]
            for gp in grade_patterns:
                match = re.search(gp, line, re.IGNORECASE)
                if match:
                    result = _normalize_grade(match.group(1).strip())
                    if result in ['Correct', 'Incorrect', 'Partial']:
                        return result
        
        # Priority 3: Look for grade words in context (e.g., "the answer is correct")
        context_patterns = [
            r'(?:answer|solution|response)\s+(?:is|was)\s+["\']?(Correct|Incorrect|Partial)["\']?',
            r'(?:student|they)\s+(?:is|was|got|received)\s+["\']?(Correct|Incorrect|Partial)["\']?',
            r'(?:this|that)\s+(?:is|was)\s+["\']?(Correct|Incorrect|Partial)["\']?',
            r'(?:I|we)\s+(?:would|will|should)\s+(?:say|call|rate|grade)\s+(?:this|that|it)\s+["\']?(Correct|Incorrect|Partial)["\']?',
        ]
        for pattern in context_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                grade = match.group(1)
                normalized = _normalize_grade(grade)
                if normalized in ['Correct', 'Incorrect', 'Partial']:
                    return normalized
        
        return "None"

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with enhanced error handling.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Validate required inputs
        required_fields = ["problem", "solution", "student_answer"]
        missing_fields = [f for f in required_fields if not inputs.get(f)]
        if missing_fields:
            self.log_fn(f"Missing required fields: {missing_fields}")
            return "None", []
        
        instruction = self._build_grading_prompt(inputs)

        # Retry mechanism with exponential backoff for transient failures
        max_retries = 3
        base_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=[],
                )
                # Success - break out of retry loop
                break
            except Exception as e:
                error_msg = str(e).lower()
                # Check for transient errors that warrant retry
                transient_indicators = [
                    'timeout', 'connection', 'rate limit', 'too many requests',
                    'temporarily unavailable', 'service unavailable', 'overloaded',
                    'internal server error', 'bad gateway', 'gateway timeout'
                ]
                is_transient = any(indicator in error_msg for indicator in transient_indicators)
                
                if attempt < max_retries - 1 and is_transient:
                    delay = base_delay * (2 ** attempt)  # Exponential backoff: 1s, 2s, 4s
                    self.log_fn(f"LLM call failed with transient error (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {delay}s...")
                    import time
                    time.sleep(delay)
                    continue
                else:
                    self.log_fn(f"LLM call failed after {attempt + 1} attempt(s): {e}")
                    return "None", []
        else:
            # All retries exhausted
            self.log_fn("All LLM call retries exhausted")
            return "None", []

        # Extract prediction using multiple strategies
        prediction = self._extract_prediction(msg_history)
        
        if prediction == "None":
            # Log more details for debugging
            last_msg = msg_history[-1].get("text", "") if msg_history else ""
            self.log_fn(f"Failed to extract prediction. Last message preview: {last_msg[:1000] if last_msg else 'empty'}")
            # Try to extract from the raw response as well
            if response and response != last_msg:
                prediction = self._extract_prediction([{"text": response}])
                if prediction != "None":
                    self.log_fn(f"Successfully extracted from raw response: {prediction}")
            
            # Final fallback: try to find any grade-like text in the response
            if prediction == "None" and response:
                # Look for the grade words directly in the response
                response_lower = response.lower()
                if 'incorrect' in response_lower and 'correct' not in response_lower.replace('incorrect', ''):
                    prediction = "Incorrect"
                    self.log_fn(f"Fallback extraction: Incorrect")
                elif 'partial' in response_lower:
                    prediction = "Partial"
                    self.log_fn(f"Fallback extraction: Partial")
                elif 'correct' in response_lower and 'incorrect' not in response_lower:
                    prediction = "Correct"
                    self.log_fn(f"Fallback extraction: Correct")

        return str(prediction), msg_history
