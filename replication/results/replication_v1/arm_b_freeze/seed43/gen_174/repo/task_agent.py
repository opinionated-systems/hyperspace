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
    """
    results = []
    
    # Try multiple extraction strategies in order of reliability
    extraction_strategies = [
        _extract_from_xml_tags,
        _extract_from_markdown_blocks,
        _extract_any_json,
    ]
    
    for strategy in extraction_strategies:
        try:
            extracted = strategy(text)
            if extracted:
                results.extend(extracted)
                # If we found valid JSON, we can stop
                break
        except Exception:
            continue
    
    return results or None


def _extract_from_xml_tags(text: str) -> list[dict] | None:
    """Extract JSON from <json>...</json> XML-style tags."""
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
        
        # Try parsing with increasing levels of cleanup
        for parser in [_parse_clean, _parse_with_cleanup, _parse_flexible]:
            try:
                result = parser(inner)
                if result is not None:
                    results.append(result)
                    break
            except Exception:
                continue
    
    return results or None


def _extract_from_markdown_blocks(text: str) -> list[dict] | None:
    """Extract JSON from ```json markdown code blocks."""
    results = []
    search_from = 0
    
    while True:
        start = text.find("```json", search_from)
        if start == -1:
            # Also try without 'json' specifier
            start = text.find("```", search_from)
            if start == -1:
                break
            offset = 3
        else:
            offset = 7
        
        end = text.find("```", start + offset)
        if end == -1:
            break
        
        inner = text[start + offset:end].strip()
        search_from = end + 3
        
        # Try parsing with increasing levels of cleanup
        for parser in [_parse_clean, _parse_with_cleanup, _parse_flexible]:
            try:
                result = parser(inner)
                if result is not None:
                    results.append(result)
                    break
            except Exception:
                continue
    
    return results or None


def _parse_clean(text: str) -> dict | None:
    """Parse clean JSON."""
    return json.loads(text)


def _parse_with_cleanup(text: str) -> dict | None:
    """Parse JSON with standard cleanup."""
    cleaned = _clean_json_string(text)
    return json.loads(cleaned)


def _parse_flexible(text: str) -> dict | None:
    """Parse JSON with more aggressive cleanup for edge cases."""
    # Remove any text before the first '{' and after the last '}'
    start = text.find('{')
    end = text.rfind('}')
    if start == -1 or end == -1 or start >= end:
        return None
    
    core = text[start:end+1]
    cleaned = _clean_json_string(core)
    
    # Try to fix common LLM JSON errors
    # Fix unquoted keys
    cleaned = re.sub(r'(\{|,\s*)(\w+)(\s*:)', r'\1"\2"\3', cleaned)
    
    return json.loads(cleaned)


def _clean_json_string(text: str) -> str:
    """Clean up common JSON formatting issues from LLM outputs.
    
    Handles:
    - Trailing commas before closing braces/brackets
    - Single quotes instead of double quotes (smart handling)
    - Unescaped newlines in strings
    - Comments (// and /* */)
    - Unescaped quotes in strings
    """
    # Remove trailing commas before closing braces/brackets
    cleaned = re.sub(r',\s*}', '}', text)
    cleaned = re.sub(r',\s*]', ']', cleaned)
    
    # Remove single-line comments
    cleaned = re.sub(r'//[^\n]*', '', cleaned)
    
    # Remove multi-line comments
    cleaned = re.sub(r'/\*.*?\*/', '', cleaned, flags=re.DOTALL)
    
    # Smart handling of single quotes: only replace those that appear to be 
    # JSON string delimiters (not apostrophes within words)
    # Pattern: single quote followed by word chars, then colon or comma or brace
    # This avoids converting apostrophes in contractions like "don't"
    def replace_json_quotes(match):
        # Check if this looks like a JSON key (quote followed by word chars then quote-colon)
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
        if re.search(r'\w*\'\s*[,}\]]', match.string[match.start():match.end()+30]):
            return '"'
        
        return "'"
    
    # Apply smart quote replacement
    cleaned = re.sub(r"(?<!\\)'", replace_json_quotes, cleaned)
    
    # Fix unescaped newlines in string values (common LLM error)
    # This is a best-effort fix - replace newlines between quotes with \n
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
            else:
                result.append(char)
            
            i += 1
        
        return ''.join(result)
    
    cleaned = escape_newlines_in_strings(cleaned)
    
    # Normalize whitespace
    cleaned = cleaned.strip()
    
    return cleaned


def _extract_any_json(text: str) -> list[dict] | None:
    """Fallback JSON extraction that looks for any JSON objects in text.
    
    Uses a robust brace-matching algorithm that handles nested objects.
    Also attempts to clean and repair malformed JSON before parsing.
    """
    results = []
    
    # Find all potential JSON objects using brace matching
    potential_objects = _find_brace_pairs(text)
    
    # Try to parse each potential object with multiple strategies
    for start, end in potential_objects:
        json_str = text[start:end]
        
        # Try multiple parsing strategies
        for parser in [_parse_clean, _parse_with_cleanup, _parse_flexible]:
            try:
                result = parser(json_str)
                if result is not None:
                    results.append(result)
                    break
            except Exception:
                continue
    
    return results or None


def _find_brace_pairs(text: str) -> list[tuple[int, int]]:
    """Find all balanced brace pairs in text.
    
    Returns list of (start, end) tuples where start is index of '{'
    and end is index after the matching '}'.
    
    This finds ALL valid JSON objects, not just top-level ones.
    """
    pairs = []
    stack = []
    
    for i, char in enumerate(text):
        if char == '{':
            stack.append(i)
        elif char == '}':
            if stack:
                start = stack.pop()
                # Record this pair - it's a valid balanced brace
                pairs.append((start, i + 1))
    
    # Also return any incomplete pairs if we have unmatched opens
    while stack:
        start = stack.pop(0)  # Get the outermost remaining
        # Find a reasonable end point - look for next brace or end of text
        end = len(text)
        # Try to find a closing brace after this start
        for j in range(start + 1, len(text)):
            if text[j] == '}':
                end = j + 1
                break
        pairs.append((start, end))
    
    # Sort by start position for consistent ordering
    pairs.sort(key=lambda x: x[0])
    
    # Remove duplicates and overlapping pairs (keep larger ones)
    filtered_pairs = []
    for start, end in pairs:
        # Check if this pair is contained within an existing pair
        contained = False
        for existing_start, existing_end in filtered_pairs:
            if start >= existing_start and end <= existing_end:
                contained = True
                break
        if not contained:
            # Remove any pairs that this one contains
            filtered_pairs = [(s, e) for s, e in filtered_pairs 
                            if not (s >= start and e <= end)]
            filtered_pairs.append((start, end))
    
    return sorted(filtered_pairs, key=lambda x: x[0])


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
- If the student answer is empty or completely irrelevant, grade as 'Incorrect'
- If the student shows significant correct work but has minor errors, consider 'Partial'
- Only output the JSON block, no additional text before or after
- Ensure your JSON is valid with no trailing commas"""

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
            
            # Priority order for grade fields
            grade_fields = ["response", "grade", "answer", "result", "evaluation", 
                           "prediction", "score", "verdict", "assessment", "decision",
                           "outcome", "judgment", "ruling", "determination"]
            
            for field in grade_fields:
                if field in last_json:
                    field_value = last_json[field]
                    # Handle both string and numeric values
                    if isinstance(field_value, (str, int, float, bool)):
                        return str(field_value)
            
            # If no known field, use the first string or numeric value found
            for key, value in last_json.items():
                if isinstance(value, (str, int, float)) and key != "reasoning":
                    return str(value)
            
            # Last resort: check if there's a value that looks like a grade
            for key, value in last_json.items():
                if isinstance(value, str):
                    value_lower = value.lower()
                    if value_lower in ["correct", "incorrect", "partial", "true", "false", "wrong", "right"]:
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
        
        Uses a multi-tier approach:
        1. Look for explicit grade statements with context
        2. Check the final sentence for conclusion indicators
        3. Use weighted scoring with negation handling
        
        Args:
            text: Raw text to search for grade indicators
            
        Returns:
            Extracted grade or "None"
        """
        text_lower = text.lower()
        
        # Tier 1: Explicit grade statements with strong context
        grade = self._extract_explicit_grade(text_lower)
        if grade:
            return grade
        
        # Tier 2: Check final sentences for conclusion indicators
        grade = self._extract_from_conclusion(text_lower)
        if grade:
            return grade
        
        # Tier 3: Weighted scoring with negation handling
        return self._extract_by_scoring(text_lower)
    
    def _extract_explicit_grade(self, text_lower: str) -> str | None:
        """Extract grade from explicit statements."""
        # Priority-ordered patterns with their expected grade groups
        grade_patterns = [
            # Direct grade assignments (highest priority)
            (r'(?:final|the)\s+(?:grade|verdict|assessment|evaluation|determination)\s*[:=]\s*["\']?(correct|incorrect|partial|true|false|wrong|right|pass|fail)["\']?', 1),
            (r'grade\s*[:=]\s*["\']?(correct|incorrect|partial|true|false|wrong|right|pass|fail)["\']?', 1),
            (r'response\s*[:=]\s*["\']?(correct|incorrect|partial|true|false|wrong|right|pass|fail)["\']?', 1),
            # "X is Y" patterns
            (r'(?:the\s+)?(?:answer|grade|result|verdict|assessment|evaluation)\s+is\s+["\']?(correct|incorrect|partial|true|false|wrong|right|pass|fail)["\']?', 1),
            # Conclusion statements
            (r'(?:therefore|thus|hence|conclusion|in\s+conclusion)[,:]?\s*(?:the\s+)?(?:answer|grade|result|verdict|assessment)\s+is\s+["\']?(correct|incorrect|partial|true|false|wrong|right|pass|fail)["\']?', 1),
            (r'(?:i\s+)?(?:conclude|determine|assess|judge|find)\s+(?:that\s+)?(?:the\s+)?(?:answer|grade|result|verdict|assessment)\s+is\s+["\']?(correct|incorrect|partial|true|false|wrong|right|pass|fail)["\']?', 1),
            # Status indicators
            (r'status\s*[:=]\s*["\']?(correct|incorrect|partial|true|false|wrong|right|pass|fail)["\']?', 1),
            (r'evaluation\s*[:=]\s*["\']?(correct|incorrect|partial|true|false|wrong|right|pass|fail)["\']?', 1),
        ]
        
        for pattern, group in grade_patterns:
            match = re.search(pattern, text_lower)
            if match:
                return self._normalize_grade(match.group(group))
        
        return None
    
    def _extract_from_conclusion(self, text_lower: str) -> str | None:
        """Extract grade from the final sentences."""
        # Split into sentences and check from the end
        sentences = re.split(r'[.!?]+', text_lower)
        
        # Check last 3 sentences (conclusion is often at the end)
        for sentence in reversed(sentences[-3:]):
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Look for grade indicators in this sentence
            patterns = [
                (r'\b(correct|incorrect|partial)\b', 1),
                (r'\b(true|false)\b', 1),
                (r'\b(wrong|right)\b', 1),
                (r'\b(pass|fail)\b', 1),
            ]
            
            for pattern, group in patterns:
                match = re.search(pattern, sentence)
                if match:
                    return self._normalize_grade(match.group(group))
        
        return None
    
    def _extract_by_scoring(self, text_lower: str) -> str:
        """Extract grade by weighted scoring of mentions."""
        # Define word groups with their weights
        correct_words = [
            (r'\bcorrect\b', 2.0),
            (r'\bright\b', 1.5),
            (r'\btrue\b', 1.5),
            (r'\bvalid\b', 1.0),
            (r'\baccurate\b', 1.0),
            (r'\bproper\b', 0.5),
            (r'\bpass\b', 1.0),
        ]
        
        incorrect_words = [
            (r'\bincorrect\b', 2.0),
            (r'\bwrong\b', 2.0),
            (r'\bfalse\b', 1.5),
            (r'\binvalid\b', 1.0),
            (r'\berror\b', 0.5),
            (r'\bmistake\b', 0.5),
            (r'\bfail\b', 1.0),
        ]
        
        partial_words = [
            (r'\bpartial\b', 2.0),
            (r'\bpartially\s+correct\b', 2.0),
            (r'\bincomplete\b', 1.0),
            (r'\bsome\s+correct\b', 1.0),
        ]
        
        # Calculate weighted scores
        correct_score = sum(weight * len(re.findall(pattern, text_lower)) 
                             for pattern, weight in correct_words)
        incorrect_score = sum(weight * len(re.findall(pattern, text_lower)) 
                               for pattern, weight in incorrect_words)
        partial_score = sum(weight * len(re.findall(pattern, text_lower)) 
                             for pattern, weight in partial_words)
        
        # Check for negations that flip meaning
        negation_patterns = [
            r'not\s+correct', r'not\s+right', r'not\s+true', r'not\s+valid',
            r'not\s+accurate', r'incorrectly', r'not\s+proper',
        ]
        negation_count = sum(len(re.findall(p, text_lower)) for p in negation_patterns)
        
        # Apply negation penalty to correct score
        correct_score -= negation_count * 2.0
        
        # Determine final grade with thresholds
        if incorrect_score > correct_score and incorrect_score > partial_score:
            return "Incorrect"
        elif partial_score > correct_score and partial_score > incorrect_score:
            return "Partial"
        elif correct_score > 0 and correct_score >= incorrect_score:
            return "Correct"
        
        return "None"
    
    def _normalize_grade(self, grade: str) -> str | None:
        """Normalize a grade string to standard values."""
        grade_lower = grade.lower()
        
        if grade_lower in ["correct", "true", "right", "pass", "valid", "accurate"]:
            return "Correct"
        elif grade_lower in ["incorrect", "false", "wrong", "fail", "invalid"]:
            return "Incorrect"
        elif grade_lower in ["partial", "partially"]:
            return "Partial"
        
        return None
