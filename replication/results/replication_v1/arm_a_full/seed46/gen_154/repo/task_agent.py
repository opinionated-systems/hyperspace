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
from typing import Callable

from agent.llm_client import get_response_from_llm, EVAL_MODEL

logger = logging.getLogger(__name__)


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks with enhanced robustness.

    Uses a multi-pass approach with intelligent parsing and repair.
    Handles nested braces, multiple objects, and various formatting issues.
    """
    if not text or not isinstance(text, str):
        return None
    
    results = []
    
    # Helper function to validate if a dict contains expected grading fields
    def is_valid_grading_object(obj: dict) -> bool:
        expected_keys = {"response", "grade", "score", "analysis", "understanding", 
                        "partial_credit_reasoning", "evaluation", "result"}
        return any(key in obj for key in expected_keys)
    
    # Pass 1: Extract from <json>...</json> blocks (highest priority)
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
        
        if not inner:
            continue
        
        # Try direct parsing first
        try:
            parsed = json.loads(inner)
            if isinstance(parsed, dict) and is_valid_grading_object(parsed):
                results.append(parsed)
                continue
        except json.JSONDecodeError:
            pass
        
        # Try repair
        repaired = _repair_json(inner)
        if repaired and isinstance(repaired, dict) and is_valid_grading_object(repaired):
            results.append(repaired)
            continue
        
        # Try to extract multiple JSON objects from the block
        brace_positions = []
        brace_count = 0
        for i, char in enumerate(inner):
            if char == "{":
                if brace_count == 0:
                    brace_positions.append(i)
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0 and brace_positions:
                    start_pos = brace_positions.pop()
                    json_str = inner[start_pos:i+1]
                    try:
                        parsed = json.loads(json_str)
                        if isinstance(parsed, dict) and is_valid_grading_object(parsed):
                            results.append(parsed)
                    except json.JSONDecodeError:
                        repaired = _repair_json(json_str)
                        if repaired and isinstance(repaired, dict) and is_valid_grading_object(repaired):
                            results.append(repaired)
    
    # Pass 2: Extract from markdown code blocks
    if not results:
        code_block_patterns = [
            (r'```json\s*(.*?)\s*```', re.DOTALL),
            (r'```\s*(.*?)\s*```', re.DOTALL),
        ]
        
        for pattern, flags in code_block_patterns:
            for match in re.finditer(pattern, text, flags):
                inner = match.group(1).strip()
                if not inner:
                    continue
                
                try:
                    parsed = json.loads(inner)
                    if isinstance(parsed, dict) and is_valid_grading_object(parsed):
                        results.append(parsed)
                        continue
                except json.JSONDecodeError:
                    pass
                
                # Try repair
                repaired = _repair_json(inner)
                if repaired and isinstance(repaired, dict) and is_valid_grading_object(repaired):
                    results.append(repaired)
    
    # Pass 3: Find JSON objects directly in text (last resort)
    if not results:
        # Find all potential JSON object boundaries
        brace_pairs = []
        brace_count = 0
        start_pos = None
        
        for i, char in enumerate(text):
            if char == "{":
                if brace_count == 0:
                    start_pos = i
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0 and start_pos is not None:
                    brace_pairs.append((start_pos, i + 1))
                    start_pos = None
        
        # Try to parse each potential JSON object
        for start, end in brace_pairs:
            json_str = text[start:end]
            if len(json_str) < 20:  # Skip very short strings (likely not valid JSON)
                continue
            
            try:
                parsed = json.loads(json_str)
                if isinstance(parsed, dict) and is_valid_grading_object(parsed):
                    results.append(parsed)
            except json.JSONDecodeError:
                repaired = _repair_json(json_str)
                if repaired and isinstance(repaired, dict) and is_valid_grading_object(repaired):
                    results.append(repaired)
    
    # Deduplicate results while preserving order
    seen = set()
    unique_results = []
    for r in results:
        # Create a hashable representation for deduplication
        try:
            key = json.dumps(r, sort_keys=True)
            if key not in seen:
                seen.add(key)
                unique_results.append(r)
        except (TypeError, ValueError):
            # If we can't serialize, include it anyway
            unique_results.append(r)
    
    return unique_results if unique_results else None


def _repair_json(text: str) -> dict | None:
    """Attempt to repair common JSON formatting issues with enhanced robustness.
    
    Handles a wide variety of formatting issues including:
    - Trailing/leading commas in objects/arrays
    - Single quotes instead of double quotes
    - Unescaped newlines, tabs, and special characters in strings
    - Missing quotes around keys
    - Unescaped quotes within strings
    - Comments (// and /* */ style)
    - Unicode BOM and control characters
    - Malformed escape sequences
    - Concatenated JSON objects
    """
    import re
    
    if not text or not isinstance(text, str):
        return None
    
    original_text = text
    
    # Remove Unicode BOM if present
    text = text.lstrip('\ufeff')
    
    # Remove C-style comments first
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    text = re.sub(r'//.*?$', '', text, flags=re.MULTILINE)
    
    # Remove Python-style comments
    text = re.sub(r'#.*?$', '', text, flags=re.MULTILINE)
    
    # Remove trailing commas before closing braces/brackets
    text = re.sub(r',(\s*[}\]])', r'\1', text)
    
    # Remove extra commas at the beginning of objects/arrays
    text = re.sub(r'\{(\s*),', '{', text)
    text = re.sub(r'\[(\s*),', '[', text)
    
    # Replace single quotes with double quotes (carefully, only for JSON keys/values)
    # Use a more careful approach: replace quotes that appear to be delimiters
    def replace_quotes(match):
        before = match.group(1)
        content = match.group(2)
        after = match.group(3)
        # Only replace if it looks like a JSON delimiter context
        if re.match(r'^[\s:,\[\{]*$', before) or re.match(r'^[\s:,\}\]]*$', after):
            return f'{before}"{content}"{after}'
        return match.group(0)
    
    # Pattern: look for single quotes that are likely JSON string delimiters
    text = re.sub(r'([\s:,\[\{]*)\'([^\']*)\'([\s:,\}\]]*)', replace_quotes, text)
    
    # Fix missing quotes around keys (e.g., {key: "value"} -> {"key": "value"})
    text = re.sub(r'(\{|,\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', text)
    
    # Fix unquoted numeric and boolean values that should be strings
    # (be careful not to break actual numbers/booleans)
    
    # Escape unescaped newlines, carriage returns, and tabs in strings
    def escape_special_chars(match):
        content = match.group(1)
        if not content:
            return '""'
        
        # Use placeholders to preserve already-escaped sequences
        placeholders = {
            '\\n': '\x00NEWLINE\x00',
            '\\r': '\x00CARRIAGE\x00',
            '\\t': '\x00TAB\x00',
            '\\\\': '\x00BACKSLASH\x00',
            '\\"': '\x00QUOTE\x00',
        }
        
        for escaped, placeholder in placeholders.items():
            content = content.replace(escaped, placeholder)
        
        # Now escape the actual characters
        content = content.replace('\n', '\\n')
        content = content.replace('\r', '\\r')
        content = content.replace('\t', '\\t')
        content = content.replace('"', '\\"')
        
        # Restore the already-escaped sequences
        for escaped, placeholder in placeholders.items():
            content = content.replace(placeholder, escaped)
        
        return f'"{content}"'
    
    # Match quoted strings and escape special characters within them
    # Use a more robust pattern that handles escaped quotes
    text = re.sub(r'"((?:[^"\\]|\\.)*?)"', escape_special_chars, text)
    
    # Remove all control characters except common whitespace
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)
    
    # Try to parse the repaired JSON
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
        return None
    except json.JSONDecodeError:
        pass
    
    # More aggressive repair: try to extract the largest valid JSON object
    # by progressively removing problematic characters from the ends
    for end_trim in range(0, min(50, len(text) // 4)):
        for start_trim in range(0, min(50, len(text) // 4)):
            try:
                trimmed = text[start_trim:len(text) - end_trim]
                result = json.loads(trimmed)
                if isinstance(result, dict):
                    return result
            except json.JSONDecodeError:
                continue
    
    # Last resort: try to extract key fields using regex
    result = {}
    
    # Extract response/grade field with multiple patterns
    response_patterns = [
        r'["\']?response["\']?\s*[:=]\s*["\']?([^"\'},\n]+)["\']?',
        r'["\']?grade["\']?\s*[:=]\s*["\']?([^"\'},\n]+)["\']?',
        r'["\']?score["\']?\s*[:=]\s*["\']?([^"\'},\n]+)["\']?',
        r'["\']?result["\']?\s*[:=]\s*["\']?([^"\'},\n]+)["\']?',
        r'["\']?evaluation["\']?\s*[:=]\s*["\']?([^"\'},\n]+)["\']?',
    ]
    for pattern in response_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = match.group(1).strip()
            # Clean up the value
            value = re.sub(r'[,}\]]+$', '', value)  # Remove trailing JSON chars
            result["response"] = value
            break
    
    # Extract analysis field with multiline support
    analysis_patterns = [
        r'["\']?analysis["\']?\s*[:=]\s*["\']((?:[^"\']|\\["\'])*?)["\']',
        r'["\']?analysis["\']?\s*[:=]\s*"((?:[^"\\]|\\.)*?)"',
    ]
    for pattern in analysis_patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            analysis_text = match.group(1)
            # Clean up escaped characters
            analysis_text = analysis_text.replace('\\n', '\n').replace('\\"', '"').replace("\\'", "'")
            result["analysis"] = analysis_text.strip()
            break
    
    # Extract understanding field
    understanding_patterns = [
        r'["\']?understanding["\']?\s*[:=]\s*["\']((?:[^"\']|\\["\'])*?)["\']',
        r'["\']?understanding["\']?\s*[:=]\s*"((?:[^"\\]|\\.)*?)"',
    ]
    for pattern in understanding_patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            understanding_text = match.group(1)
            understanding_text = understanding_text.replace('\\n', '\n').replace('\\"', '"').replace("\\'", "'")
            result["understanding"] = understanding_text.strip()
            break
    
    # Extract partial_credit_reasoning field
    partial_patterns = [
        r'["\']?partial_credit_reasoning["\']?\s*[:=]\s*["\']((?:[^"\']|\\["\'])*?)["\']',
        r'["\']?partial_credit_reasoning["\']?\s*[:=]\s*"((?:[^"\\]|\\.)*?)"',
    ]
    for pattern in partial_patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            partial_text = match.group(1)
            partial_text = partial_text.replace('\\n', '\n').replace('\\"', '"').replace("\\'", "'")
            result["partial_credit_reasoning"] = partial_text.strip()
            break
    
    if result:
        return result
    
    return None


def _log_extraction_details(text: str, results: list[dict] | None, logger_fn: Callable) -> None:
    """Log detailed information about JSON extraction for debugging.
    
    Args:
        text: The raw text that was processed
        results: The extracted JSON results (or None if failed)
        logger_fn: Logging function to use
    """
    if results:
        logger_fn(f"Successfully extracted {len(results)} JSON object(s)")
        for i, obj in enumerate(results):
            keys = list(obj.keys())
            logger_fn(f"  Object {i+1}: keys={keys}")
    else:
        # Log hints about why extraction might have failed
        has_json_tags = "<json>" in text and "</json>" in text
        has_code_blocks = "```" in text
        has_braces = "{" in text and "}" in text
        
        logger_fn("JSON extraction failed - diagnostic info:")
        logger_fn(f"  - Contains <json> tags: {has_json_tags}")
        logger_fn(f"  - Contains code blocks: {has_code_blocks}")
        logger_fn(f"  - Contains curly braces: {has_braces}")
        logger_fn(f"  - Text length: {len(text)} chars")
        
        # Show a snippet of the text for debugging
        snippet = text[:200].replace("\n", " ")
        logger_fn(f"  - Text preview: {snippet}...")


def _validate_grade_value(grade: str) -> tuple[bool, str]:
    """Validate that a grade value is in an acceptable format.
    
    Args:
        grade: The grade string to validate
        
    Returns:
        Tuple of (is_valid, normalized_grade)
    """
    if not grade or grade == "None":
        return False, "None"
    
    # Handle numeric grades (0-7)
    valid_grades = {"Correct", "Partial", "Incorrect", "0", "1", "2", "3", "4", "5", "6", "7"}
    
    if grade in valid_grades:
        return True, grade
    
    # Try to normalize and check again
    normalized = _normalize_grade(grade)
    if normalized in valid_grades:
        return True, normalized
    
    # Check if it's a numeric string that can be mapped
    try:
        num = int(grade)
        if 0 <= num <= 7:
            return True, str(num)
    except (ValueError, TypeError):
        pass
    
    return False, grade


def _normalize_grade(grade: str) -> str:
    """Normalize grade to standard format.
    
    Converts various grade formats to a consistent representation.
    Handles both string grades (Correct/Partial/Incorrect) and numeric scores (0-7).
    """
    if not grade:
        return "None"
    
    # Handle non-string types (int, float, etc.)
    if not isinstance(grade, str):
        grade = str(grade)
    
    grade = grade.strip()
    grade_lower = grade.lower()
    
    # Map common variations to standard grades (using lowercase keys)
    grade_map = {
        "correct": "Correct",
        "right": "Correct",
        "true": "Correct",
        "yes": "Correct",
        "full": "Correct",
        "full credit": "Correct",
        "complete": "Correct",
        "solved": "Correct",
        "perfect": "Correct",
        "excellent": "Correct",
        "partial": "Partial",
        "partially correct": "Partial",
        "partial credit": "Partial",
        "partially": "Partial",
        "some credit": "Partial",
        "incomplete": "Partial",
        "mostly correct": "Partial",
        "mostly right": "Partial",
        "incorrect": "Incorrect",
        "wrong": "Incorrect",
        "false": "Incorrect",
        "no": "Incorrect",
        "none": "Incorrect",
        "zero": "Incorrect",
        "0": "0",
        "1": "1",
        "2": "2",
        "3": "3",
        "4": "4",
        "5": "5",
        "6": "6",
        "7": "7",
    }
    
    # Check for exact match first (case-insensitive)
    if grade_lower in grade_map:
        return grade_map[grade_lower]
    
    # Try to extract numeric score from strings like "Score: 5" or "Grade: 3"
    numeric_match = re.search(r'\b([0-7])\b', grade)
    if numeric_match:
        return numeric_match.group(1)
    
    # Check for numeric ranges and pick appropriate grade
    # e.g., "5-6 points" -> "5" (lower bound for partial credit)
    range_match = re.search(r'(\d+)\s*-\s*(\d+)', grade)
    if range_match:
        lower = int(range_match.group(1))
        upper = int(range_match.group(2))
        if 0 <= lower <= 7:
            return str(lower)
    
    # Check if grade contains keywords (order matters - check partial first)
    if "partial" in grade_lower:
        return "Partial"
    if any(word in grade_lower for word in ["incomplete", "unfinished", "missing", "half"]):
        return "Partial"
    if any(word in grade_lower for word in ["mostly correct", "mostly right", "nearly correct"]):
        return "Partial"
    if any(word in grade_lower for word in ["correct", "right", "true", "full", "complete", "solved", "perfect", "excellent"]):
        return "Correct"
    if any(word in grade_lower for word in ["incorrect", "wrong", "false", "none", "zero", "error", "invalid", "fail", "failed"]):
        return "Incorrect"
    
    # Default: capitalize first letter
    return grade.capitalize()


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with structured reasoning.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Extract fields for structured prompting
        domain = inputs.get("domain", "")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = f"""You are an expert mathematical grader for International Mathematical Olympiad (IMO) problems.

Your task is to evaluate a student's answer by comparing it against the official solution and grading guidelines.

## Problem Domain
{domain}

## Problem Statement
{problem}

## Official Solution
{solution}

## Grading Guidelines
{grading_guidelines}

## Student's Answer
{student_answer}

## Instructions
Follow this structured evaluation process:

1. **Understanding Check**: Briefly summarize what the problem is asking and what the correct approach should be.

2. **Step-by-Step Analysis**: Go through the student's answer carefully:
   - Identify each key step or claim they make
   - Check if each step is mathematically valid
   - Note any errors, gaps, or incorrect assumptions
   - Compare their approach to the official solution
   - Check if they used the correct mathematical notation and terminology

3. **Partial Credit Assessment**: Based on the grading guidelines, determine:
   - Which parts of the solution they completed correctly
   - What partial credit they deserve for incomplete or partially correct work
   - Whether they demonstrated understanding of key concepts
   - Whether they made significant progress toward the solution

4. **Final Grade Decision**: Assign a grade that reflects:
   - Full correctness (Correct/7) if completely solved with valid reasoning
   - Partial credit (Partial/1-6) for incomplete or partially correct solutions
   - Incorrect (Incorrect/0) for fundamentally wrong or empty answers

## CRITICAL: Distinguishing Partial from Incorrect
This is the most common grading error. Use these guidelines:

**Assign "Partial" when the student:**
- Made any significant progress toward the solution
- Used a correct approach but had minor errors or gaps
- Demonstrated understanding of key concepts
- Completed some steps correctly even if the final answer is wrong
- Had the right idea but incomplete execution

**Assign "Incorrect" ONLY when:**
- The answer is completely blank or empty
- The student made no meaningful progress
- The approach is fundamentally wrong with no redeeming elements
- There is no evidence of understanding the problem

**When in doubt, prefer "Partial" over "Incorrect"** - partial credit is awarded for ANY meaningful progress.

## Grading Rubric Reference
- 7 points: Complete, correct solution with clear reasoning
- 5-6 points: Minor gaps or unclear reasoning, but essentially correct approach
- 3-4 points: Significant progress, correct approach but incomplete or with errors
- 1-2 points: Some relevant work or correct initial steps
- 0 points: No meaningful progress or completely wrong approach

## IMPORTANT: Response Format
You MUST respond in valid JSON format wrapped in <json>...</json> tags.
The "response" field MUST contain exactly one of these values:
- "Correct" (for complete, correct solutions worth 7 points)
- "Partial" (for incomplete or partially correct solutions worth 1-6 points)
- "Incorrect" (for fundamentally wrong or empty answers worth 0 points)
- Or a numeric string "0" through "7" for specific point values

Example response format:
<json>
{{
    "understanding": "This problem asks to find the sum of digits...",
    "analysis": "The student correctly identified... but made an error in...",
    "partial_credit_reasoning": "The student showed understanding of... but missed...",
    "response": "Partial"
}}
</json>

Respond in JSON format with the following schema:
<json>
{{
    "understanding": "Brief summary of the problem and correct approach",
    "analysis": "Detailed step-by-step analysis of the student's answer, including what they got right and wrong",
    "partial_credit_reasoning": "Explanation of partial credit based on grading guidelines and rubric - BE SPECIFIC about why Partial vs Incorrect",
    "response": "Your final grade/score (use: 'Correct', 'Partial', 'Incorrect', or numeric 0-7)"
}}
</json>"""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON with enhanced error handling
        prediction = "None"
        try:
            # Get the last assistant message
            last_assistant_msg = None
            for msg in reversed(msg_history):
                if msg.get("role") == "assistant" or "text" in msg:
                    last_assistant_msg = msg.get("text", msg.get("content", ""))
                    break
            
            if not last_assistant_msg:
                self.log_fn("Warning: No assistant message found in history")
                return str(prediction), msg_history
            
            extracted = _extract_jsons(last_assistant_msg)
            
            # Log extraction details for debugging
            _log_extraction_details(last_assistant_msg, extracted, self.log_fn)
            
            if extracted:
                # Try to get response field first, fallback to other common fields
                last_extract = extracted[-1]
                
                # Priority order for grade extraction
                grade_fields = ["response", "grade", "score", "result", "final_grade", "evaluation"]
                for field in grade_fields:
                    if field in last_extract:
                        prediction = last_extract[field]
                        break
                
                # Normalize the grade for consistency
                prediction = _normalize_grade(prediction)
                
                # Validate the grade and log if invalid
                is_valid, validated_grade = _validate_grade_value(prediction)
                if not is_valid:
                    self.log_fn(f"Warning: Extracted grade '{prediction}' is not in standard format")
                    # Try to infer from analysis text if grade is invalid
                    if "analysis" in last_extract:
                        inferred = _infer_grade_from_analysis(last_extract["analysis"])
                        if inferred:
                            self.log_fn(f"Inferred grade from analysis: {inferred}")
                            prediction = inferred
                            is_valid = True
                prediction = validated_grade
                
                # Log detailed analysis for debugging
                if "analysis" in last_extract:
                    self.log_fn(f"Analysis: {last_extract['analysis'][:200]}...")
                if "partial_credit_reasoning" in last_extract:
                    self.log_fn(f"Partial Credit: {last_extract['partial_credit_reasoning'][:200]}...")
                if "understanding" in last_extract:
                    self.log_fn(f"Understanding: {last_extract['understanding'][:200]}...")
                
                self.log_fn(f"Extracted grade: {prediction} (valid: {is_valid})")
            else:
                self.log_fn("Warning: No JSON blocks found in response")
                # Try to extract grade directly from text as last resort
                prediction = _extract_grade_from_text(last_assistant_msg)
                if prediction != "None":
                    is_valid, prediction = _validate_grade_value(prediction)
                    self.log_fn(f"Extracted grade from text: {prediction} (valid: {is_valid})")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            import traceback
            self.log_fn(f"Traceback: {traceback.format_exc()}")

        return str(prediction), msg_history


def _extract_grade_from_text(text: str) -> str:
    """Extract grade from plain text when JSON parsing fails.
    
    Uses a multi-pass approach with weighted pattern matching for robust extraction.
    Enhanced to better handle partial vs incorrect distinction and edge cases.
    """
    import re
    
    text_lower = text.lower()
    
    # Pass 1: Look for explicit grade/score patterns with numeric values (highest priority)
    numeric_patterns = [
        (r'(?:grade|score|result|evaluation)\s*[:=]\s*["\']?([0-7])\s*(?:points?|pts?)?["\']?', 3),
        (r'(?:final|overall|total)\s+(?:grade|score|result)\s*[:=]\s*["\']?([0-7])\s*(?:points?|pts?)?["\']?', 3),
        (r'(?:assigned|given|awarded)\s*[:=]\s*["\']?([0-7])\s*(?:points?|pts?)?["\']?', 3),
        (r'\bgrade\s+is\s+["\']?([0-7])["\']?', 3),
        (r'\bscore\s+is\s+["\']?([0-7])["\']?', 3),
        (r'\b(?:worth|deserves|earns?)\s+([0-7])\s*(?:points?)?\b', 2),
        (r'\b([0-7])\s*/\s*7\s*(?:points?)?\b', 2),
    ]
    
    best_numeric_match = None
    best_numeric_priority = 0
    
    for pattern, priority in numeric_patterns:
        match = re.search(pattern, text_lower, re.IGNORECASE)
        if match and priority > best_numeric_priority:
            best_numeric_match = match.group(1)
            best_numeric_priority = priority
    
    if best_numeric_match:
        return best_numeric_match
    
    # Pass 2: Look for explicit text grade assignments
    text_grade_patterns = [
        (r'(?:grade|score|result|evaluation)\s*[:=]\s*["\']?(correct|partial|incorrect)["\']?', 3),
        (r'(?:final|overall|total)\s+(?:grade|score|result)\s*[:=]\s*["\']?(correct|partial|incorrect)["\']?', 3),
        (r'\bgrade\s+is\s+["\']?(correct|partial|incorrect)["\']?', 3),
        (r'\bscore\s+is\s+["\']?(correct|partial|incorrect)["\']?', 3),
        (r'(?:the student|this solution)\s+(?:is|should be|deserves)\s+["\']?(correct|partial|incorrect)["\']?', 2),
    ]
    
    best_text_match = None
    best_text_priority = 0
    
    for pattern, priority in text_grade_patterns:
        match = re.search(pattern, text_lower, re.IGNORECASE)
        if match and priority > best_text_priority:
            grade = match.group(1).strip().capitalize()
            is_valid, validated = _validate_grade_value(grade)
            if is_valid:
                best_text_match = validated
                best_text_priority = priority
    
    if best_text_match:
        return best_text_match
    
    # Pass 3: Contextual keyword analysis with weighted scoring
    score = 0
    
    # Strong partial indicators (check first - more specific)
    strong_partial_patterns = [
        r'\bpartial(ly)?\s+correct\b',
        r'\bpartial\s+credit\b',
        r'\bsome\s+credit\b',
        r'\bpartial\s+points\b',
        r'\bincomplete\s+solution\b',
        r'\bmissing\s+step\b',
        r'\bminor\s+error\b',
        r'\bsmall\s+mistake\b',
        r'\bmostly\s+(?:correct|right)\b',
        r'\bon\s+the\s+right\s+track\b',
        r'\bsome\s+progress\b',
        r'\bsignificant\s+progress\b',
        r'\bgood\s+start\b',
        r'\bcorrect\s+approach\b',
        r'\bsome\s+understanding\b',
        r'\bnearly\s+correct\b',
        r'\balmost\s+correct\b',
    ]
    
    for pattern in strong_partial_patterns:
        if re.search(pattern, text_lower):
            score += 2
    
    # Strong correct indicators
    strong_correct_patterns = [
        r'\bcomplete\s+solution\b',
        r'\bfully\s+solved\b',
        r'\bcorrect\s+solution\b',
        r'\bvalid\s+proof\b',
        r'\bcorrectly\s+proved\b',
        r'\bcorrect\s+answer\b',
        r'\bfully\s+correct\b',
        r'\bentirely\s+correct\b',
        r'\bsolution\s+is\s+correct\b',
        r'\banswer\s+is\s+correct\b',
        r'\bperfect\s+solution\b',
        r'\bexcellent\s+work\b',
    ]
    
    for pattern in strong_correct_patterns:
        if re.search(pattern, text_lower):
            score += 3
    
    # Strong incorrect indicators
    strong_incorrect_patterns = [
        r'\bcompletely\s+wrong\b',
        r'\bfundamentally\s+incorrect\b',
        r'\bno\s+understanding\b',
        r'\bno\s+progress\b',
        r'\bno\s+relevant\s+work\b',
        r'\bentirely\s+incorrect\b',
        r'\btotally\s+wrong\b',
        r'\bcompletely\s+incorrect\b',
        r'\bdoes\s+not\s+understand\b',
        r'\bfundamental\s+error\b',
        r'\bmajor\s+misconception\b',
        r'\bno\s+attempt\b',
        r'\bblank\s+answer\b',
        r'\bempty\s+answer\b',
    ]
    
    for pattern in strong_incorrect_patterns:
        if re.search(pattern, text_lower):
            score -= 3
    
    # Weak indicators
    if re.search(r'\bcorrect\b', text_lower) and not re.search(r'\bnot\s+correct\b|\bpartial(ly)?\s+correct\b|\bincorrect\b', text_lower):
        score += 1
    
    if re.search(r'\bincorrect\b|\bwrong\b|\bno\s+credit\b', text_lower):
        score -= 1
    
    # Look for numeric grades in context (lower priority than explicit patterns)
    numeric_match = re.search(r'\b([0-7])\s*(?:points?|pts?|/\s*7)?\b', text_lower)
    if numeric_match:
        num = int(numeric_match.group(1))
        if num == 7:
            score += 2
        elif num >= 5:
            score += 1
        elif num <= 2:
            score -= 1
        elif num == 0:
            score -= 2
    
    # Decision based on score
    if score >= 3:
        return "Correct"
    elif score > 0:
        return "Partial"
    elif score <= -2:
        return "Incorrect"
    elif score < 0:
        # Borderline - check for any redeeming qualities
        if any(word in text_lower for word in ["some", "partial", "progress", "correct"]):
            return "Partial"
        return "Incorrect"
    
    # Score is 0 or no clear signals - default to Partial for safety
    return "Partial"


def _infer_grade_from_analysis(analysis: str) -> str | None:
    """Infer grade from analysis text when explicit grade is missing or invalid.
    
    Uses weighted keyword matching with contextual scoring for more accurate grading.
    Enhanced to better handle edge cases and nuanced mathematical reasoning.
    """
    if not analysis:
        return None
    
    analysis_lower = analysis.lower()
    
    # Weighted scoring system for more nuanced grading
    score = 0
    
    # Tier 1: Definitive complete correctness indicators (+3 each)
    definitive_correct = [
        "complete solution", "fully solved", "correct solution", "valid proof",
        "correctly proved", "correct answer", "fully correct", "entirely correct",
        "solution is correct", "answer is correct", "correctly derived",
        "perfect solution", "excellent work", "complete proof"
    ]
    
    # Tier 2: Strong correctness indicators (+2 each)
    strong_correct = [
        "correct approach", "valid reasoning", "correct method", "right idea",
        "correctly identified", "properly solved", "correctly applied",
        "sound logic", "valid argument", "correct derivation"
    ]
    
    # Tier 3: Partial credit indicators (+1 each)
    partial_indicators = [
        "partial credit", "partially correct", "some progress", "minor error",
        "mostly correct", "significant progress", "on the right track",
        "good start", "some understanding", "incomplete proof",
        "missing step", "small error", "slight mistake", "nearly correct",
        "almost correct", "minor mistake", "partial solution",
        "incomplete solution", "some correct steps", "partial understanding",
        "correct initial", "correct first step", "some valid reasoning",
        "demonstrated understanding", "made progress", "substantial work"
    ]
    
    # Tier 4: Weak/error indicators (-1 each)
    weak_indicators = [
        "minor error", "small mistake", "slight issue", "unclear",
        "could be better", "needs improvement", "lacks clarity"
    ]
    
    # Tier 5: Strong negative indicators (-2 each)
    strong_negative = [
        "incorrect", "wrong", "error", "mistake", "flawed", "invalid",
        "not correct", "not valid", "does not work", "failed",
        "confused", "misunderstood", "missing key", "lacks understanding"
    ]
    
    # Tier 6: Definitive failure indicators (-3 each)
    definitive_wrong = [
        "completely wrong", "fundamentally incorrect", "no understanding",
        "no progress", "no relevant work", "entirely incorrect", "totally wrong",
        "completely incorrect", "does not understand", "failed to",
        "fundamental error", "major misconception", "no attempt",
        "blank", "empty answer", "irrelevant"
    ]
    
    # Calculate weighted score
    for indicator in definitive_correct:
        if indicator in analysis_lower:
            score += 3
    
    for indicator in strong_correct:
        if indicator in analysis_lower:
            score += 2
    
    for indicator in partial_indicators:
        if indicator in analysis_lower:
            score += 1
    
    for indicator in weak_indicators:
        if indicator in analysis_lower:
            score -= 1
    
    for indicator in strong_negative:
        if indicator in analysis_lower:
            score -= 2
    
    for indicator in definitive_wrong:
        if indicator in analysis_lower:
            score -= 3
    
    # Additional contextual analysis
    # Check for numeric score mentions
    numeric_patterns = [
        r'\b([0-7])\s*(?:points?|pts?|/\s*7)\b',
        r'(?:score|grade)\s*[:=]\s*([0-7])\b',
        r'\b(?:worth|deserves|earns?)\s+([0-7])\b'
    ]
    
    for pattern in numeric_patterns:
        match = re.search(pattern, analysis_lower)
        if match:
            num_score = int(match.group(1))
            if num_score == 7:
                score += 2  # Boost for explicit 7
            elif num_score >= 5:
                score += 1  # Slight boost for high scores
            elif num_score <= 2:
                score -= 1  # Slight penalty for low scores
            elif num_score == 0:
                score -= 2  # Penalty for explicit 0
            break
    
    # Check for conclusion statements (often indicate final assessment)
    conclusion_patterns = [
        r'(?:in conclusion|therefore|thus|overall|final assessment|in summary)[,:]\s*(.+?)(?:\.|$)',
        r'(?:the student|this solution)\s+(?:is|should be|deserves)\s+(.+?)(?:\.|$)'
    ]
    
    for pattern in conclusion_patterns:
        match = re.search(pattern, analysis_lower)
        if match:
            conclusion = match.group(1)
            if any(word in conclusion for word in ["correct", "complete", "full credit", "7"]):
                score += 2
            elif any(word in conclusion for word in ["partial", "some credit", "incomplete"]):
                score += 0  # Neutral - already captured by partial indicators
            elif any(word in conclusion for word in ["incorrect", "wrong", "no credit", "0"]):
                score -= 2
            break
    
    # Decision based on cumulative score
    if score >= 4:
        return "Correct"
    elif score >= 1:
        return "Partial"
    elif score <= -3:
        return "Incorrect"
    elif score <= -1:
        # Borderline negative - check for any redeeming qualities
        if "some" in analysis_lower or "partial" in analysis_lower or "progress" in analysis_lower:
            return "Partial"
        return "Incorrect"
    else:
        # Score is 0 - ambiguous case
        # Check for explicit grade mentions as tiebreaker
        if re.search(r'\b(?:grade|score)\s*[:=]\s*["\']?(?:partial|partially)', analysis_lower):
            return "Partial"
        if re.search(r'\b(?:grade|score)\s*[:=]\s*["\']?(?:incorrect|wrong|0)', analysis_lower):
            return "Incorrect"
        if re.search(r'\b(?:grade|score)\s*[:=]\s*["\']?(?:correct|7)', analysis_lower):
            return "Correct"
        
        # Default to Partial for ambiguous cases (safer for student)
        return "Partial"
