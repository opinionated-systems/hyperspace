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
    """Extract JSON objects from <json>...</json> blocks and markdown code blocks.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    Handles multiple JSON extraction strategies with cleanup fallback.
    """
    results = []
    
    # Define extraction patterns: (start_marker, end_marker, start_offset, end_offset)
    patterns = [
        ("<json>", "</json>", 6, 7),  # <json>...</json>
        ("```json", "```", 7, 3),      # ```json...```
        ("```", "```", 3, 3),          # ```...``` (generic, processed last)
    ]
    
    processed_regions = set()  # Track regions to avoid double-processing
    
    for start_marker, end_marker, start_offset, end_offset in patterns:
        search_from = 0
        while True:
            start = text.find(start_marker, search_from)
            if start == -1:
                break
            
            # Skip already processed regions for generic ``` blocks
            if start_marker == "```" and any(start >= s and start < e for s, e in processed_regions):
                search_from = start + start_offset
                continue
                
            end = text.find(end_marker, start + start_offset)
            if end == -1:
                break
            
            # Track this region
            region_end = end + end_offset
            processed_regions.add((start, region_end))
            
            inner = text[start + start_offset:end].strip()
            search_from = region_end
            
            # Skip if doesn't look like JSON (for generic blocks)
            if start_marker == "```" and not (inner.startswith("{") or inner.startswith("[")):
                continue
            
            # Try to parse JSON
            parsed = _try_parse_json(inner)
            if parsed:
                results.extend(parsed) if isinstance(parsed, list) else results.append(parsed)
    
    return results or None


def _try_parse_json(text: str) -> dict | list[dict] | None:
    """Attempt to parse JSON with cleanup fallback.
    
    Args:
        text: JSON string to parse
        
    Returns:
        Parsed dict, list of dicts, or None if parsing fails
    """
    # Direct parse attempt
    try:
        result = json.loads(text)
        return result if isinstance(result, (dict, list)) else None
    except json.JSONDecodeError:
        pass
    
    # Try with cleanup
    try:
        cleaned = _clean_json_string(text)
        result = json.loads(cleaned)
        return result if isinstance(result, (dict, list)) else None
    except json.JSONDecodeError:
        pass
    
    # Try nested extraction for complex cases
    try:
        nested = _extract_nested_json(text)
        return nested if nested else None
    except Exception:
        return None


def _extract_nested_json(text: str) -> list[dict] | None:
    """Extract multiple JSON objects from text with nested brace handling.
    
    This handles cases where JSON objects are concatenated or nested
    within each other in complex ways. Uses a stack-based brace counter
    for accurate boundary detection.
    """
    results = []
    text_len = len(text)
    i = 0
    
    while i < text_len:
        # Find the start of a potential JSON object
        if text[i] != '{':
            i += 1
            continue
            
        start = i
        brace_count = 0
        in_string = False
        escape_next = False
        
        for j in range(i, text_len):
            char = text[j]
            
            # Handle string context
            if escape_next:
                escape_next = False
                continue
            if char == '\\':
                escape_next = True
                continue
            if char == '"' and not in_string:
                in_string = True
                continue
            if char == '"' and in_string:
                in_string = False
                continue
                
            # Only count braces outside of strings
            if not in_string:
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        # Found a complete object
                        candidate = text[start:j+1]
                        parsed = _try_parse_json(candidate)
                        if parsed and isinstance(parsed, dict):
                            results.append(parsed)
                        i = j + 1
                        break
        else:
            # No complete object found from this start position
            i += 1
    
    return results or None


def _clean_json_string(text: str) -> str:
    """Clean up common JSON formatting issues from LLM outputs.
    
    Handles:
    - Trailing commas before closing braces/brackets
    - Single quotes instead of double quotes (outside strings)
    - Comments (// and /* */)
    - Control characters and invalid whitespace
    
    Args:
        text: Raw text potentially containing JSON
        
    Returns:
        Cleaned text ready for JSON parsing
    """
    cleaned = text.strip()
    
    # Remove control characters except valid whitespace
    cleaned = ''.join(char for char in cleaned if ord(char) >= 32 or char in '\n\r\t')
    
    # Remove single-line comments (outside strings)
    cleaned = _remove_comments(cleaned)
    
    # Remove trailing commas before closing braces/brackets (outside strings)
    cleaned = _remove_trailing_commas(cleaned)
    
    # Replace single quotes with double quotes (only outside strings)
    cleaned = _normalize_quotes(cleaned)
    
    return cleaned.strip()


def _remove_comments(text: str) -> str:
    """Remove // and /* */ comments from text, respecting string boundaries."""
    result = []
    i = 0
    text_len = len(text)
    in_string = False
    escape_next = False
    
    while i < text_len:
        char = text[i]
        
        if escape_next:
            result.append(char)
            escape_next = False
            i += 1
            continue
            
        if char == '\\':
            result.append(char)
            escape_next = True
            i += 1
            continue
            
        if char == '"' and not in_string:
            in_string = True
            result.append(char)
            i += 1
            continue
            
        if char == '"' and in_string:
            in_string = False
            result.append(char)
            i += 1
            continue
        
        # Only process comments outside strings
        if not in_string:
            # Check for single-line comment
            if char == '/' and i + 1 < text_len and text[i + 1] == '/':
                # Skip to end of line
                while i < text_len and text[i] not in '\n\r':
                    i += 1
                continue
                
            # Check for multi-line comment
            if char == '/' and i + 1 < text_len and text[i + 1] == '*':
                # Skip to end of comment
                i += 2
                while i < text_len - 1 and not (text[i] == '*' and text[i + 1] == '/'):
                    i += 1
                i += 2  # Skip past */
                continue
        
        result.append(char)
        i += 1
    
    return ''.join(result)


def _remove_trailing_commas(text: str) -> str:
    """Remove trailing commas before } or ], respecting string boundaries."""
    # Pattern: comma followed by whitespace and closing bracket
    result = []
    i = 0
    text_len = len(text)
    in_string = False
    escape_next = False
    
    while i < text_len:
        char = text[i]
        
        if escape_next:
            result.append(char)
            escape_next = False
            i += 1
            continue
            
        if char == '\\':
            result.append(char)
            escape_next = True
            i += 1
            continue
            
        if char == '"' and not in_string:
            in_string = True
            result.append(char)
            i += 1
            continue
            
        if char == '"' and in_string:
            in_string = False
            result.append(char)
            i += 1
            continue
        
        # Check for trailing comma outside strings
        if not in_string and char == ',':
            # Look ahead for closing bracket (skipping whitespace)
            j = i + 1
            while j < text_len and text[j] in ' \t\n\r':
                j += 1
            if j < text_len and text[j] in '}]':
                # Skip the comma (trailing comma)
                i += 1
                continue
        
        result.append(char)
        i += 1
    
    return ''.join(result)


def _normalize_quotes(text: str) -> str:
    """Replace single quotes with double quotes outside of strings."""
    result = []
    i = 0
    text_len = len(text)
    in_string = False
    string_char = None
    escape_next = False
    
    while i < text_len:
        char = text[i]
        
        if escape_next:
            result.append(char)
            escape_next = False
            i += 1
            continue
            
        if char == '\\':
            result.append(char)
            escape_next = True
            i += 1
            continue
        
        # Handle string boundaries
        if not in_string and char in '"\'':
            in_string = True
            string_char = char
            result.append('"')  # Always use double quotes
            i += 1
            continue
            
        if in_string and char == string_char:
            in_string = False
            string_char = None
            result.append('"')  # Always use double quotes
            i += 1
            continue
        
        result.append(char)
        i += 1
    
    return ''.join(result)


def _extract_any_json(text: str) -> list[dict] | None:
    """Fallback JSON extraction that looks for any JSON objects in text.
    
    This is a simpler version that uses the same robust nested extraction
    logic as _extract_nested_json for consistency.
    """
    return _extract_nested_json(text)


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

        # Check for empty student answer early
        if not student_answer or not student_answer.strip():
            return "Incorrect", [{"role": "system", "text": "Empty student answer detected, returning Incorrect"}]

        instruction = f"""You are an expert mathematical grader for International Mathematical Olympiad (IMO) problems.

Your task is to evaluate a student's answer to a mathematics problem and provide a grade with detailed reasoning.

DOMAIN: {domain}

PROBLEM:
{problem}

OFFICIAL SOLUTION:
{solution}

GRADING GUIDELINES:
{grading_guidelines}

STUDENT'S ANSWER:
{student_answer}

Think step by step and provide a thorough analysis following this structure:

1. PROBLEM ANALYSIS: Identify the key mathematical concepts, theorems, and techniques required to solve this problem.

2. SOLUTION REVIEW: Analyze the official solution's approach, key steps, and expected answer format. Note what constitutes a complete and correct solution.

3. STUDENT WORK ANALYSIS: 
   - Identify what approach the student took
   - Note any correct steps, valid insights, or correct intermediate results
   - Identify errors, gaps, misconceptions, or missing steps
   - Check if the final answer matches the expected form

4. GRADING CRITERIA CHECK:
   - Systematically verify if the student met each criterion in the grading guidelines
   - Award partial credit for incomplete but valid reasoning
   - Note any specific point deductions for errors or omissions

5. FINAL DETERMINATION: Assign a clear grade based on:
   - Completeness of the solution
   - Mathematical correctness
   - Adherence to grading guidelines
   - Quality of reasoning shown

Respond ONLY in JSON format with the following exact schema:
<json>
{{
    "reasoning": "Your detailed chain-of-thought analysis covering all 5 steps above. Be specific about what the student did right and wrong.",
    "response": "The final grade/prediction. Use exactly one of: 'Correct', 'Incorrect', 'Partial', or a numeric score if specified in guidelines."
}}
</json>

Important guidelines:
- Be objective, consistent, and thorough in your analysis
- Award partial credit when the student shows valid reasoning even if the final answer is incorrect
- If the student answer is empty, completely irrelevant, or shows no understanding, grade as 'Incorrect'
- If the student shows significant correct work but has minor errors, consider 'Partial'
- 'Correct' should only be used when the solution is essentially complete and correct
- 'Partial' is for solutions with significant correct work but notable gaps or errors
- 'Incorrect' is for solutions that are wrong, empty, or show fundamental misunderstanding
- Only output the JSON block, no additional text before or after
- Ensure your JSON is valid with no trailing commas
- Do not include markdown code blocks around the JSON"""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON with fallback mechanisms
        prediction = self._extract_prediction(msg_history)

        return str(prediction), msg_history

    def _extract_prediction(self, msg_history: list[dict]) -> str:
        """Extract prediction from message history with multiple fallback strategies.
        
        Args:
            msg_history: List of message dicts with 'role' and 'text' keys
            
        Returns:
            Extracted prediction string or "None" if extraction fails
        """
        if not msg_history:
            self.log_fn("[Extract] No message history provided")
            return "None"
        
        try:
            last_message = msg_history[-1]["text"]
            msg_preview = last_message[:200].replace('\n', ' ')
            self.log_fn(f"[Extract] Processing message: {msg_preview}...")
            
            # Check if the response indicates an LLM error
            if last_message.startswith("Error:") or last_message.startswith("Error "):
                self.log_fn(f"[Extract] LLM returned error response: {last_message[:100]}...")
                # Try to extract any grade information from the error message
                error_grade = self._extract_from_text(last_message)
                if error_grade != "None":
                    self.log_fn(f"[Extract] Recovered grade from error message: {error_grade}")
                    return error_grade
                return "None"
            
            # Try primary extraction method
            extracted = _extract_jsons(last_message)
            self.log_fn(f"[Extract] Primary extraction found {len(extracted) if extracted else 0} JSON objects")
            
            # Fallback to generic JSON extraction if primary fails
            if extracted is None:
                self.log_fn("[Extract] Primary extraction failed, trying fallback...")
                extracted = _extract_any_json(last_message)
                if extracted:
                    self.log_fn(f"[Extract] Fallback found {len(extracted)} JSON objects")
            
            if not extracted:
                # Try to extract grade from plain text as last resort
                self.log_fn("[Extract] No JSON found, attempting text extraction...")
                text_grade = self._extract_from_text(last_message)
                self.log_fn(f"[Extract] Text extraction result: {text_grade}")
                return text_grade
            
            # Prefer response field, but accept other common field names
            last_json = extracted[-1]
            self.log_fn(f"[Extract] Using JSON with keys: {list(last_json.keys())}")
            
            # Priority order for grade fields
            grade_fields = ["response", "grade", "answer", "result", "evaluation", 
                           "prediction", "score", "verdict", "assessment"]
            
            for field in grade_fields:
                if field in last_json:
                    field_value = last_json[field]
                    self.log_fn(f"[Extract] Found grade field '{field}': {field_value}")
                    # Handle both string and numeric values
                    if isinstance(field_value, (str, int, float, bool)):
                        normalized = self._normalize_grade(str(field_value))
                        self.log_fn(f"[Extract] Normalized grade: {normalized}")
                        return normalized
            
            # If no known field, use the first string or numeric value found
            for key, value in last_json.items():
                if isinstance(value, (str, int, float)) and key != "reasoning":
                    self.log_fn(f"[Extract] Using first valid field '{key}': {value}")
                    return self._normalize_grade(str(value))
            
            # Last resort: check if there's a value that looks like a grade
            for key, value in last_json.items():
                if isinstance(value, str):
                    normalized = self._normalize_grade(value)
                    if normalized in ["Correct", "Incorrect", "Partial"]:
                        self.log_fn(f"[Extract] Found grade-like value in '{key}': {normalized}")
                        return normalized
                    
        except Exception as e:
            self.log_fn(f"[Extract] Error extracting prediction: {e}")
        
        self.log_fn("[Extract] All extraction methods failed, returning 'None'")
        return "None"
    
    def _normalize_grade(self, grade: str) -> str:
        """Normalize a grade string to standard format.
        
        Args:
            grade: Raw grade string from LLM
            
        Returns:
            Normalized grade: 'Correct', 'Incorrect', 'Partial', or original string
        """
        grade_lower = grade.lower().strip()
        
        # Map common variations to standard grades
        if grade_lower in ["correct", "true", "right", "yes", "pass", "1", "100"]:
            return "Correct"
        elif grade_lower in ["incorrect", "false", "wrong", "no", "fail", "0", "0/100"]:
            return "Incorrect"
        elif grade_lower in ["partial", "partially correct", "partial credit", "incomplete", "0.5", "50"]:
            return "Partial"
        
        # Check for numeric scores that might indicate partial credit
        try:
            numeric = float(grade_lower)
            if numeric >= 0.8 or numeric >= 80:
                return "Correct"
            elif numeric <= 0.2 or numeric <= 20:
                return "Incorrect"
            elif 0.2 < numeric < 0.8 or 20 < numeric < 80:
                return "Partial"
        except ValueError:
            pass
        
        return grade
    
    def _extract_from_text(self, text: str) -> str:
        """Extract grade from plain text when JSON parsing fails.
        
        Args:
            text: Raw text to search for grade indicators
            
        Returns:
            Extracted grade or "None"
        """
        text_lower = text.lower()
        
        # Look for explicit grade statements with more comprehensive patterns
        grade_patterns = [
            (r'grade\s*[:=]\s*["\']?(correct|incorrect|partial|true|false|right|wrong)["\']?', 1),
            (r'final\s*(?:grade|determination|verdict)\s*[:=]\s*["\']?(correct|incorrect|partial|true|false|right|wrong)["\']?', 1),
            (r'response\s*[:=]\s*["\']?(correct|incorrect|partial|true|false|right|wrong)["\']?', 1),
            (r'(?:the\s+)?(?:answer|grade|result|verdict)\s+is\s+["\']?(correct|incorrect|partial|true|false|right|wrong)["\']?', 1),
            (r'["\']?(correct|incorrect|partial|true|false|right|wrong)["\']?\s*(?:grade|score|verdict|result)', 1),
            (r'(?:graded?\s+as|marked\s+as)\s+["\']?(correct|incorrect|partial|true|false|right|wrong)["\']?', 1),
            (r'(?:student|answer)\s+(?:is|was)\s+["\']?(correct|incorrect|partial|true|false|right|wrong)["\']?', 1),
        ]
        
        for pattern, group in grade_patterns:
            match = re.search(pattern, text_lower)
            if match:
                grade = match.group(group).lower()
                normalized = self._normalize_grade(grade)
                if normalized in ["Correct", "Incorrect", "Partial"]:
                    return normalized
        
        # Count mentions as fallback with weighted scoring
        correct_count = len(re.findall(r'\bcorrect\b|\bright\b|\btrue\b|\bpass\b', text_lower))
        incorrect_count = len(re.findall(r'\bincorrect\b|\bwrong\b|\bfalse\b|\bfail\b', text_lower))
        partial_count = len(re.findall(r'\bpartial\b|\bincomplete\b|\bpartially\b', text_lower))
        
        # Apply weights based on context
        if "not correct" in text_lower or "not right" in text_lower:
            correct_count -= 2
        if "not incorrect" in text_lower or "not wrong" in text_lower:
            incorrect_count -= 2
        
        if incorrect_count > correct_count and incorrect_count > partial_count:
            return "Incorrect"
        elif partial_count > correct_count and partial_count > incorrect_count:
            return "Partial"
        elif correct_count > 0 and correct_count >= incorrect_count:
            return "Correct"
        
        return "None"
