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
        text: Raw text potentially containing JSON blocks
        
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
                if cleaned and cleaned != inner:  # Only try if cleaning changed something
                    results.append(json.loads(cleaned))
            except json.JSONDecodeError:
                continue
    
    # Also try markdown code blocks
    if not results:
        search_from = 0
        while True:
            start = text.find("```json", search_from)
            if start == -1:
                # Try without language specifier
                start = text.find("```", search_from)
                if start == -1:
                    break
                offset = 3
            else:
                offset = 7  # len("```json")
                
            end = text.find("```", start + offset)
            if end == -1:
                break
            inner = text[start + offset:end].strip()
            search_from = end + 3
            
            # Skip empty content
            if not inner:
                continue
                
            try:
                results.append(json.loads(inner))
            except json.JSONDecodeError:
                try:
                    cleaned = _clean_json_string(inner)
                    if cleaned and cleaned != inner:
                        results.append(json.loads(cleaned))
                except json.JSONDecodeError:
                    continue
    
    return results if results else None


def _clean_json_string(text: str) -> str:
    """Clean up common JSON formatting issues from LLM outputs.
    
    Handles:
    - Trailing commas before closing braces/brackets
    - Single quotes instead of double quotes
    - Unescaped newlines in strings
    - Comments (// and /* */)
    - Control characters
    
    Args:
        text: Raw text potentially containing JSON with formatting issues
        
    Returns:
        Cleaned text that should be valid JSON
    """
    if not text:
        return ""
        
    cleaned = text
    
    # Remove control characters except tab, newline, carriage return
    cleaned = ''.join(char for char in cleaned if ord(char) >= 32 or char in '\t\n\r')
    
    # Remove trailing commas before closing braces/brackets (handles nested cases)
    # Use a loop to handle multiple consecutive trailing commas
    for _ in range(5):  # Limit iterations to prevent infinite loops
        new_cleaned = re.sub(r',\s*}', '}', cleaned)
        new_cleaned = re.sub(r',\s*]', ']', new_cleaned)
        if new_cleaned == cleaned:
            break
        cleaned = new_cleaned
    
    # Remove single-line comments (but not inside strings)
    # Simple approach: remove // comments that aren't inside quotes
    lines = cleaned.split('\n')
    cleaned_lines = []
    for line in lines:
        # Find // that's not inside quotes
        in_string = False
        escape_next = False
        comment_start = -1
        for i, char in enumerate(line):
            if escape_next:
                escape_next = False
                continue
            if char == '\\':
                escape_next = True
                continue
            if char == '"' and not in_string:
                in_string = True
            elif char == '"' and in_string:
                in_string = False
            elif not in_string and char == '/' and i + 1 < len(line) and line[i + 1] == '/':
                comment_start = i
                break
        if comment_start >= 0:
            line = line[:comment_start]
        cleaned_lines.append(line)
    cleaned = '\n'.join(cleaned_lines)
    
    # Remove multi-line comments
    cleaned = re.sub(r'/\*.*?\*/', '', cleaned, flags=re.DOTALL)
    
    # Replace single quotes with double quotes (only outside strings)
    # This is tricky - we only want to replace quotes that are acting as delimiters
    # A simple heuristic: replace ' that appear at start of values or after structural chars
    result = []
    in_string = False
    escape_next = False
    for i, char in enumerate(cleaned):
        if escape_next:
            result.append(char)
            escape_next = False
            continue
        if char == '\\':
            result.append(char)
            escape_next = True
            continue
        if char == '"':
            in_string = not in_string
            result.append(char)
        elif char == "'" and not in_string:
            # Check if this looks like a JSON delimiter (after structural chars or whitespace)
            if i == 0 or cleaned[i-1] in ':[,{\n\t ':
                result.append('"')
            elif i + 1 < len(cleaned) and cleaned[i+1] in '}\],:\n\t \'"':
                result.append('"')
            else:
                result.append(char)
        else:
            result.append(char)
    cleaned = ''.join(result)
    
    # Normalize whitespace
    cleaned = cleaned.strip()
    
    return cleaned


def _extract_any_json(text: str) -> list[dict] | None:
    """Fallback JSON extraction that looks for any JSON objects in text.
    
    This is a more aggressive extraction method that tries to find JSON-like
    structures even when they're not properly wrapped in tags.
    
    Args:
        text: Raw text potentially containing JSON objects
        
    Returns:
        List of parsed JSON objects, or None if no valid JSON found
    """
    if not text or not isinstance(text, str):
        return None
        
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
        if char == '\\':
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
                    try:
                        obj = json.loads(text[start_idx:i+1])
                        results.append(obj)
                    except json.JSONDecodeError:
                        # Try cleaning and re-parsing
                        try:
                            cleaned = _clean_json_string(text[start_idx:i+1])
                            if cleaned:
                                obj = json.loads(cleaned)
                                results.append(obj)
                        except json.JSONDecodeError:
                            pass
                    start_idx = -1
    
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
            return "None"
        
        try:
            last_message = msg_history[-1]["text"]
            
            # Check if the response indicates an LLM error
            if last_message.startswith("Error:") or last_message.startswith("Error "):
                self.log_fn(f"LLM returned error response: {last_message[:100]}...")
                # Try to extract any grade information from the error message
                error_grade = self._extract_from_text(last_message)
                if error_grade != "None":
                    return error_grade
                return "None"
            
            # Try primary extraction method
            extracted = _extract_jsons(last_message)
            
            # Fallback to generic JSON extraction if primary fails
            if extracted is None:
                extracted = _extract_any_json(last_message)
            
            if not extracted:
                # Try to extract grade from plain text as last resort
                return self._extract_from_text(last_message)
            
            # Prefer response field, but accept other common field names
            last_json = extracted[-1]
            
            # Priority order for grade fields
            grade_fields = ["response", "grade", "answer", "result", "evaluation", 
                           "prediction", "score", "verdict", "assessment"]
            
            for field in grade_fields:
                if field in last_json:
                    field_value = last_json[field]
                    # Handle both string and numeric values
                    if isinstance(field_value, (str, int, float, bool)):
                        return self._normalize_grade(str(field_value))
            
            # If no known field, use the first string or numeric value found
            for key, value in last_json.items():
                if isinstance(value, (str, int, float)) and key != "reasoning":
                    return self._normalize_grade(str(value))
            
            # Last resort: check if there's a value that looks like a grade
            for key, value in last_json.items():
                if isinstance(value, str):
                    normalized = self._normalize_grade(value)
                    if normalized in ["Correct", "Incorrect", "Partial"]:
                        return normalized
                    
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
        
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
        
        Uses multiple strategies: explicit grade statements, context analysis,
        and weighted keyword counting.
        
        Args:
            text: Raw text to search for grade indicators
            
        Returns:
            Extracted grade or "None"
        """
        if not text or not isinstance(text, str):
            return "None"
            
        text_lower = text.lower()
        
        # Look for explicit grade statements with more comprehensive patterns
        # These patterns look for grade declarations in various formats
        grade_patterns = [
            # Direct grade assignments
            (r'grade\s*[:=]\s*["\']?(correct|incorrect|partial|true|false|right|wrong)["\']?', 1),
            (r'response\s*[:=]\s*["\']?(correct|incorrect|partial|true|false|right|wrong)["\']?', 1),
            (r'["\']?(correct|incorrect|partial|true|false|right|wrong)["\']?\s*(?:grade|score|verdict|result)', 1),
            
            # Final determination patterns
            (r'final\s*(?:grade|determination|verdict|assessment|evaluation)\s*[:=]\s*["\']?(correct|incorrect|partial|true|false|right|wrong)["\']?', 1),
            (r'(?:the\s+)?(?:final\s+)?(?:answer|grade|result|verdict|assessment)\s+is\s+["\']?(correct|incorrect|partial|true|false|right|wrong)["\']?', 1),
            
            # Status patterns
            (r'(?:graded?\s+as|marked\s+as|classified\s+as|considered)\s+["\']?(correct|incorrect|partial|true|false|right|wrong)["\']?', 1),
            (r'(?:student|answer|solution|work)\s+(?:is|was|should\s+be)\s+["\']?(correct|incorrect|partial|true|false|right|wrong)["\']?', 1),
            
            # Conclusion patterns
            (r'(?:conclusion|determination|assessment)\s*[:=]\s*["\']?(correct|incorrect|partial|true|false|right|wrong)["\']?', 1),
            (r'(?:therefore|thus|hence|so)\s*,?\s*(?:the\s+)?(?:answer|grade|result)\s+(?:is\s+)?["\']?(correct|incorrect|partial|true|false|right|wrong)["\']?', 1),
        ]
        
        for pattern, group in grade_patterns:
            match = re.search(pattern, text_lower)
            if match:
                grade = match.group(group).lower()
                normalized = self._normalize_grade(grade)
                if normalized in ["Correct", "Incorrect", "Partial"]:
                    return normalized
        
        # Count mentions as fallback with weighted scoring
        # Use word boundaries to avoid partial matches
        correct_keywords = r'\bcorrect\b|\bright\b|\btrue\b|\bpass\b|\bvalid\b|\baccurate\b'
        incorrect_keywords = r'\bincorrect\b|\bwrong\b|\bfalse\b|\bfail\b|\binvalid\b|\berror\b'
        partial_keywords = r'\bpartial\b|\bincomplete\b|\bpartially\b|\bhalf\b|\bsome\b'
        
        correct_count = len(re.findall(correct_keywords, text_lower))
        incorrect_count = len(re.findall(incorrect_keywords, text_lower))
        partial_count = len(re.findall(partial_keywords, text_lower))
        
        # Apply context-based weights
        # Negations flip the meaning
        negation_patterns = [
            (r'\bnot\s+correct\b|\bisn\'t\s+correct\b|\bnot\s+right\b|\bisn\'t\s+right\b', 'correct'),
            (r'\bnot\s+incorrect\b|\bisn\'t\s+incorrect\b|\bnot\s+wrong\b|\bisn\'t\s+wrong\b', 'incorrect'),
            (r'\bnot\s+partial\b|\bisn\'t\s+partial\b', 'partial'),
        ]
        
        for pattern, target in negation_patterns:
            matches = len(re.findall(pattern, text_lower))
            if target == 'correct':
                correct_count -= matches * 3  # Strong penalty for negated correct
            elif target == 'incorrect':
                incorrect_count -= matches * 3  # Strong penalty for negated incorrect
            elif target == 'partial':
                partial_count -= matches * 3
        
        # Boost scores for explicit conclusions
        if re.search(r'\bconclusion\b|\bdetermination\b|\bfinal\s+assessment\b', text_lower):
            # Find which grade appears near conclusion keywords
            conclusion_section = re.search(r'(?:conclusion|determination|final\s+assessment).*?(?:\n\n|$)', text_lower, re.DOTALL)
            if conclusion_section:
                section_text = conclusion_section.group(0)
                correct_in_conclusion = len(re.findall(correct_keywords, section_text))
                incorrect_in_conclusion = len(re.findall(incorrect_keywords, section_text))
                partial_in_conclusion = len(re.findall(partial_keywords, section_text))
                
                # Boost conclusion section scores
                correct_count += correct_in_conclusion * 2
                incorrect_count += incorrect_in_conclusion * 2
                partial_count += partial_in_conclusion * 2
        
        # Determine final grade based on weighted counts
        if incorrect_count > correct_count and incorrect_count > partial_count:
            return "Incorrect"
        elif partial_count > correct_count and partial_count > incorrect_count:
            return "Partial"
        elif correct_count > 0 and correct_count >= incorrect_count:
            return "Correct"
        
        return "None"
