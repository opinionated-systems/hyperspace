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
    Also handles markdown code blocks with json tag.
    Includes advanced cleanup for common LLM output issues.
    
    Args:
        text: The text to extract JSON from
        
    Returns:
        List of parsed JSON objects, or None if no valid JSON found
    """
    if not text or not isinstance(text, str):
        return None
        
    results = []
    search_from = 0
    
    # First try to find explicit <json> tags
    while True:
        start = text.find("<json>", search_from)
        if start == -1:
            break
        end = text.find("</json>", start)
        if end == -1:
            break
        inner = text[start + 6:end].strip()
        search_from = end + 7
        
        # Skip empty content
        if not inner:
            continue
            
        try:
            results.append(json.loads(inner))
        except json.JSONDecodeError:
            # Try to clean up common issues
            try:
                cleaned = _clean_json_string(inner)
                results.append(json.loads(cleaned))
            except json.JSONDecodeError:
                continue
    
    # Also try markdown code blocks (even if we found <json> tags, for completeness)
    search_from = 0
    while True:
        start = text.find("```json", search_from)
        if start == -1:
            # Try without 'json' specifier
            start = text.find("```", search_from)
            if start == -1:
                break
            end = text.find("```", start + 3)
            offset = 3
        else:
            end = text.find("```", start + 7)
            offset = 7
            
        if end == -1:
            break
        inner = text[start + offset:end].strip()
        search_from = end + 3
        
        # Skip empty content
        if not inner:
            continue
            
        try:
            obj = json.loads(inner)
            if obj not in results:  # Avoid duplicates
                results.append(obj)
        except json.JSONDecodeError:
            try:
                cleaned = _clean_json_string(inner)
                obj = json.loads(cleaned)
                if obj not in results:
                    results.append(obj)
            except json.JSONDecodeError:
                continue
    
    return results if results else None


def _clean_json_string(text: str) -> str:
    """Clean up common JSON formatting issues from LLM outputs.
    
    Handles:
    - Trailing commas before closing braces/brackets
    - Single quotes instead of double quotes (smart handling)
    - Unescaped newlines in strings
    - Comments (// and /* */)
    - Unescaped quotes in strings
    - Control characters
    - BOM markers
    
    Args:
        text: The JSON string to clean
        
    Returns:
        Cleaned JSON string ready for parsing
    """
    if not text or not isinstance(text, str):
        return "{}"
    
    # Remove BOM if present
    cleaned = text.lstrip('\ufeff')
    
    # Remove trailing commas before closing braces/brackets
    cleaned = re.sub(r',\s*}', '}', cleaned)
    cleaned = re.sub(r',\s*]', ']', cleaned)
    
    # Remove single-line comments
    cleaned = re.sub(r'//[^\n]*', '', cleaned)
    
    # Remove multi-line comments
    cleaned = re.sub(r'/\*.*?\*/', '', cleaned, flags=re.DOTALL)
    
    # Smart handling of single quotes: only replace those that appear to be 
    # JSON string delimiters (not apostrophes within words)
    def replace_json_quotes(match):
        before = match.string[max(0, match.start()-1):match.start()]
        after = match.string[match.end():min(len(match.string), match.end()+20)]
        
        # If preceded by backslash, it's escaped - keep as is
        if before == '\\':
            return "'"
        
        # If followed by word chars and then quote and colon/brace, it's a JSON key
        if re.match(r'\w+[\'"]\s*[:},\]]', after):
            return '"'
        # If at start of string or after punctuation/brace, likely a JSON string start
        if not before or before in ' \t\n{[,':
            return '"'
        # If followed by word chars and then end quote and comma/brace, it's a JSON value
        # Get the full string context for this check
        full_str = match.string
        start_pos = match.start()
        end_pos = min(len(full_str), match.end() + 30)
        context = full_str[start_pos:end_pos]
        if re.search(r'\w*\'\s*[,}\]]', context):
            return '"'
        
        return "'"
    
    # Apply smart quote replacement
    cleaned = re.sub(r"(?<!\\)'", replace_json_quotes, cleaned)
    
    # Fix unescaped newlines in string values (common LLM error)
    def escape_newlines_in_strings(content):
        result = []
        in_string = False
        string_char = None
        i = 0
        while i < len(content):
            char = content[i]
            
            # Handle escape sequences
            if char == '\\' and i + 1 < len(content):
                result.append(char)
                result.append(content[i + 1])
                i += 2
                continue
            
            # String start/end
            if char in '"' and not in_string:
                in_string = True
                string_char = char
                result.append(char)
            elif char == string_char and in_string:
                in_string = False
                string_char = None
                result.append(char)
            elif char == '\n' and in_string:
                # Replace newline with escaped version
                result.append('\\n')
            elif char == '\r' and in_string:
                # Replace carriage return with escaped version
                result.append('\\r')
            elif char == '\t' and in_string:
                # Tab is already valid in JSON strings, keep as is
                result.append(char)
            elif ord(char) < 32 and in_string:
                # Escape other control characters
                result.append(f'\\u{ord(char):04x}')
            else:
                result.append(char)
            
            i += 1
        
        return ''.join(result)
    
    cleaned = escape_newlines_in_strings(cleaned)
    
    # Remove control characters outside of strings (they shouldn't be there)
    cleaned = ''.join(char for char in cleaned if ord(char) >= 32 or char in '\n\r\t')
    
    # Normalize whitespace
    cleaned = cleaned.strip()
    
    return cleaned


def _extract_any_json(text: str) -> list[dict] | None:
    """Fallback JSON extraction that looks for any JSON objects in text.
    
    Uses a robust brace-matching algorithm that handles nested objects.
    Also attempts to clean and repair malformed JSON before parsing.
    
    Args:
        text: The text to search for JSON objects
        
    Returns:
        List of parsed JSON objects, or None if no valid JSON found
    """
    if not text or not isinstance(text, str):
        return None
        
    results = []
    # Try to find JSON objects between curly braces
    brace_count = 0
    start_idx = -1
    potential_objects = []
    in_string = False
    string_char = None
    escape_next = False
    
    for i, char in enumerate(text):
        # Handle string context to avoid counting braces inside strings
        if escape_next:
            escape_next = False
            continue
        
        if char == '\\' and in_string:
            escape_next = True
            continue
            
        if char in '"\'' and not in_string:
            in_string = True
            string_char = char
        elif char == string_char and in_string:
            in_string = False
            string_char = None
        elif not in_string:
            if char == '{':
                if brace_count == 0:
                    start_idx = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start_idx != -1:
                    potential_objects.append((start_idx, i + 1))
                    start_idx = -1
    
    # Try to parse each potential object
    for start, end in potential_objects:
        json_str = text[start:end]
        
        # Skip if too short to be a valid object
        if len(json_str) < 2:
            continue
        
        # First try direct parsing
        try:
            obj = json.loads(json_str)
            if obj not in results:  # Avoid duplicates
                results.append(obj)
            continue
        except json.JSONDecodeError:
            pass
        
        # Try with cleaning
        try:
            cleaned = _clean_json_string(json_str)
            obj = json.loads(cleaned)
            if obj not in results:
                results.append(obj)
            continue
        except json.JSONDecodeError:
            pass
        
        # Try extracting just the outermost object by finding balanced braces
        try:
            for j in range(end - 1, start + 1, -1):
                if text[j] == '}':
                    test_str = text[start:j+1]
                    try:
                        obj = json.loads(test_str)
                        if obj not in results:
                            results.append(obj)
                        break
                    except json.JSONDecodeError:
                        # Try with cleaning
                        try:
                            cleaned = _clean_json_string(test_str)
                            obj = json.loads(cleaned)
                            if obj not in results:
                                results.append(obj)
                            break
                        except json.JSONDecodeError:
                            continue
        except Exception:
            pass
    
    return results if results else None


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
        # Validate inputs
        if not isinstance(inputs, dict):
            self.log_fn(f"Invalid inputs type: {type(inputs)}")
            return "None", [{"role": "system", "text": f"Invalid inputs type: {type(inputs)}"}]
        
        # Extract task components for better prompting
        domain = inputs.get("domain", "Unknown")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        # Check for empty student answer early
        if not student_answer or not isinstance(student_answer, str) or not student_answer.strip():
            return "Incorrect", [{"role": "system", "text": "Empty or invalid student answer detected, returning Incorrect"}]

        # Truncate very long inputs to prevent context overflow
        max_input_len = 8000
        problem = problem[:max_input_len] if len(problem) > max_input_len else problem
        solution = solution[:max_input_len] if len(solution) > max_input_len else solution
        grading_guidelines = grading_guidelines[:max_input_len] if len(grading_guidelines) > max_input_len else grading_guidelines
        student_answer = student_answer[:max_input_len] if len(student_answer) > max_input_len else student_answer

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
- If the student answer is empty or completely irrelevant, grade as 'Incorrect'
- If the student shows significant correct work but has minor errors, consider 'Partial'
- Only output the JSON block, no additional text before or after
- Ensure your JSON is valid with no trailing commas"""

        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
        except Exception as e:
            self.log_fn(f"Error calling LLM: {e}")
            return "None", [{"role": "system", "text": f"LLM call failed: {e}"}]

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
        if not msg_history or not isinstance(msg_history, list):
            return "None"
        
        try:
            # Get the last message with text content
            last_message = None
            for msg in reversed(msg_history):
                if isinstance(msg, dict) and msg.get("text"):
                    last_message = msg["text"]
                    break
            
            if not last_message:
                return "None"
            
            # Try primary extraction method
            extracted = _extract_jsons(last_message)
            
            # Fallback to generic JSON extraction if primary fails
            if not extracted:
                extracted = _extract_any_json(last_message)
            
            if not extracted:
                # Try to extract grade from plain text as last resort
                return self._extract_from_text(last_message)
            
            # Prefer response field, but accept other common field names
            last_json = extracted[-1]
            
            if not isinstance(last_json, dict):
                # If it's not a dict, try to convert or use as-is
                if isinstance(last_json, (str, int, float, bool)):
                    return str(last_json)
                return self._extract_from_text(last_message)
            
            # Priority order for grade fields
            grade_fields = ["response", "grade", "answer", "result", "evaluation", 
                           "prediction", "score", "verdict", "assessment", "decision",
                           "outcome", "judgment", "ruling", "determination", "final_grade"]
            
            for field in grade_fields:
                if field in last_json:
                    field_value = last_json[field]
                    # Handle both string and numeric values
                    if isinstance(field_value, (str, int, float, bool)):
                        return str(field_value)
            
            # If no known field, use the first string or numeric value found (excluding reasoning)
            for key, value in last_json.items():
                if isinstance(value, (str, int, float, bool)) and key.lower() not in ["reasoning", "explanation", "analysis", "thoughts"]:
                    return str(value)
            
            # Last resort: check if there's a value that looks like a grade
            for key, value in last_json.items():
                if isinstance(value, str):
                    value_lower = value.lower().strip()
                    if value_lower in ["correct", "incorrect", "partial", "true", "false", "wrong", "right", "pass", "fail"]:
                        return value
                    # Check for numeric grades (0-100 scale)
                    try:
                        num = float(value)
                        if 0 <= num <= 100:
                            return value
                    except ValueError:
                        pass
                    
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
        
        return "None"
    
    def _extract_from_text(self, text: str) -> str:
        """Extract grade from plain text when JSON parsing fails.
        
        Args:
            text: Raw text to search for grade indicators
            
        Returns:
            Extracted grade or "None"
        """
        if not text or not isinstance(text, str):
            return "None"
            
        text_lower = text.lower()
        
        # Look for explicit grade statements with more comprehensive patterns
        grade_patterns = [
            # Direct grade assignments
            (r'grade\s*[:=]\s*["\']?(correct|incorrect|partial|true|false|wrong|right)["\']?', 1),
            (r'final\s*(?:grade|determination|verdict|assessment|evaluation)\s*[:=]\s*["\']?(correct|incorrect|partial|true|false|wrong|right)["\']?', 1),
            (r'response\s*[:=]\s*["\']?(correct|incorrect|partial|true|false|wrong|right)["\']?', 1),
            (r'(?:the\s+)?(?:answer|grade|result|verdict|assessment)\s+is\s+["\']?(correct|incorrect|partial|true|false|wrong|right)["\']?', 1),
            (r'["\']?(correct|incorrect|partial|true|false|wrong|right)["\']?\s*(?:grade|score|verdict|result|assessment)', 1),
            # Conclusion statements
            (r'(?:therefore|thus|hence|conclusion|in\s+conclusion)[,:]?\s*(?:the\s+)?(?:answer|grade|result)\s+is\s+["\']?(correct|incorrect|partial|true|false|wrong|right)["\']?', 1),
            (r'(?:i\s+)?(?:conclude|determine|assess|judge|find)\s+(?:that\s+)?(?:the\s+)?(?:answer|grade|result)\s+is\s+["\']?(correct|incorrect|partial|true|false|wrong|right)["\']?', 1),
            # Status indicators
            (r'status\s*[:=]\s*["\']?(correct|incorrect|partial|true|false|wrong|right|pass|fail)["\']?', 1),
            (r'evaluation\s*[:=]\s*["\']?(correct|incorrect|partial|true|false|wrong|right)["\']?', 1),
            # Additional patterns for edge cases
            (r'(?:final\s+)?(?:determination|verdict)\s*[:=]\s*["\']?(correct|incorrect|partial|true|false|wrong|right)["\']?', 1),
            (r'(?:the\s+)?student\s+(?:answer|response|work)\s+is\s+["\']?(correct|incorrect|partial|true|false|wrong|right)["\']?', 1),
        ]
        
        for pattern, group in grade_patterns:
            match = re.search(pattern, text_lower)
            if match:
                grade = match.group(group).lower()
                # Normalize grade
                if grade in ["correct", "true", "right", "pass"]:
                    return "Correct"
                elif grade in ["incorrect", "false", "wrong", "fail"]:
                    return "Incorrect"
                elif grade == "partial":
                    return "Partial"
        
        # Look for grade in the last sentence (often where conclusion is)
        sentences = re.split(r'[.!?]+', text_lower)
        if sentences:
            # Check last few sentences for grade indicators
            for sentence in reversed(sentences[-3:]):
                sentence = sentence.strip()
                if not sentence:
                    continue
                    
                last_patterns = [
                    r'\b(correct|incorrect|partial)\b',
                    r'\b(true|false)\b',
                    r'\b(wrong|right)\b',
                    r'\b(pass|fail)\b',
                ]
                for pattern in last_patterns:
                    match = re.search(pattern, sentence)
                    if match:
                        grade = match.group(1).lower()
                        if grade in ["correct", "true", "right", "pass"]:
                            return "Correct"
                        elif grade in ["incorrect", "false", "wrong", "fail"]:
                            return "Incorrect"
                        elif grade == "partial":
                            return "Partial"
        
        # Count mentions as fallback with weighted scoring
        correct_count = len(re.findall(r'\bcorrect\b', text_lower))
        correct_count += len(re.findall(r'\bright\b', text_lower))
        correct_count += len(re.findall(r'\btrue\b', text_lower))
        correct_count += len(re.findall(r'\bpass\b', text_lower))
        
        incorrect_count = len(re.findall(r'\bincorrect\b', text_lower))
        incorrect_count += len(re.findall(r'\bwrong\b', text_lower))
        incorrect_count += len(re.findall(r'\bfalse\b', text_lower))
        incorrect_count += len(re.findall(r'\bfail\b', text_lower))
        
        partial_count = len(re.findall(r'\bpartial\b', text_lower))
        partial_count += len(re.findall(r'\bpartially\s+correct\b', text_lower))
        
        # Check for negations that might flip the meaning
        negation_patterns = [
            r'not\s+correct',
            r'not\s+right',
            r'not\s+true',
            r'not\s+pass',
            r'incorrectly',
            r'wrongly',
        ]
        negation_count = sum(len(re.findall(p, text_lower)) for p in negation_patterns)
        
        # Adjust counts based on negations
        correct_count -= negation_count
        
        # Determine final grade with clear thresholds
        if incorrect_count > correct_count and incorrect_count > partial_count:
            return "Incorrect"
        elif partial_count > correct_count and partial_count > incorrect_count:
            return "Partial"
        elif correct_count > 0 and correct_count >= incorrect_count:
            return "Correct"
        
        return "None"
