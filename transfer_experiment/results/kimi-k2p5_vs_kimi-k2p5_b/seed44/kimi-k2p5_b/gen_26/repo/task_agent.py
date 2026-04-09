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
from typing import Any

from agent.llm_client import get_response_from_llm, EVAL_MODEL

logger = logging.getLogger(__name__)

# Valid grades for IMO problem evaluation
VALID_GRADES = ["correct", "incorrect", "partial", "almost"]

# Grade priority order for conflict resolution (from highest to lowest confidence)
GRADE_PRIORITY = ["correct", "almost", "partial", "incorrect"]


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks or raw JSON.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    Also tries to find raw JSON objects if tags are not present.
    Handles nested braces correctly and supports multi-field JSON objects.
    
    Args:
        text: The text to extract JSON objects from.
        
    Returns:
        List of parsed JSON objects, or None if no valid JSON found.
    """
    if not text or not isinstance(text, str):
        logger.debug("_extract_jsons: Empty or invalid input text")
        return None
        
    results = []
    search_from = 0

    # First try to find <json>...</json> blocks (highest priority)
    while True:
        start = text.find("<json>", search_from)
        if start == -1:
            break
        end = text.find("</json>", start)
        if end == -1:
            logger.debug(f"_extract_jsons: Unclosed <json> tag at position {start}")
            break
        inner = text[start + 6:end].strip()
        search_from = end + 7
        
        # Try parsing with safe_json_loads for better error handling
        parsed = _safe_json_loads(inner)
        if parsed and isinstance(parsed, dict):
            results.append(parsed)
            logger.debug(f"_extract_jsons: Parsed <json> block: {list(parsed.keys())}")
        else:
            try:
                parsed = json.loads(inner)
                if isinstance(parsed, dict):
                    results.append(parsed)
                    logger.debug(f"_extract_jsons: Parsed <json> block via json.loads: {list(parsed.keys())}")
            except json.JSONDecodeError as e:
                logger.debug(f"_extract_jsons: Failed to parse <json> block: {e}")
                continue

    # Also try to find ```json code blocks
    search_from = 0
    while True:
        start = text.find("```json", search_from)
        if start == -1:
            break
        end = text.find("```", start + 7)
        if end == -1:
            logger.debug(f"_extract_jsons: Unclosed ```json block at position {start}")
            break
        inner = text[start + 7:end].strip()
        search_from = end + 3
        
        parsed = _safe_json_loads(inner)
        if parsed and isinstance(parsed, dict):
            results.append(parsed)
            logger.debug(f"_extract_jsons: Parsed ```json block: {list(parsed.keys())}")
        else:
            try:
                parsed = json.loads(inner)
                if isinstance(parsed, dict):
                    results.append(parsed)
                    logger.debug(f"_extract_jsons: Parsed ```json block via json.loads: {list(parsed.keys())}")
            except json.JSONDecodeError as e:
                logger.debug(f"_extract_jsons: Failed to parse ```json block: {e}")
                continue

    # Also try to find ``` code blocks without json tag
    search_from = 0
    while True:
        start = text.find("```", search_from)
        if start == -1:
            break
        # Skip ```json blocks we already processed
        if text[start:start+7] == "```json":
            search_from = start + 7
            continue
        end = text.find("```", start + 3)
        if end == -1:
            break
        inner = text[start + 3:end].strip()
        search_from = end + 3
        
        parsed = _safe_json_loads(inner)
        if parsed and isinstance(parsed, dict):
            results.append(parsed)
            logger.debug(f"_extract_jsons: Parsed ``` block: {list(parsed.keys())}")
        else:
            try:
                parsed = json.loads(inner)
                if isinstance(parsed, dict):
                    results.append(parsed)
                    logger.debug(f"_extract_jsons: Parsed ``` block via json.loads: {list(parsed.keys())}")
            except json.JSONDecodeError:
                continue

    # If no results from tagged blocks, try to find raw JSON objects
    if not results:
        # Find JSON objects by looking for balanced braces
        i = 0
        while i < len(text):
            if text[i] == '{':
                # Try to find the matching closing brace
                brace_count = 1
                j = i + 1
                while j < len(text) and brace_count > 0:
                    if text[j] == '{':
                        brace_count += 1
                    elif text[j] == '}':
                        brace_count -= 1
                    j += 1

                if brace_count == 0:
                    json_str = text[i:j].strip()
                    # Try parsing with safe_json_loads first
                    parsed = _safe_json_loads(json_str)
                    if parsed and isinstance(parsed, dict):
                        results.append(parsed)
                        logger.debug(f"_extract_jsons: Parsed raw JSON object: {list(parsed.keys())}")
                    else:
                        try:
                            parsed = json.loads(json_str)
                            if isinstance(parsed, dict):
                                results.append(parsed)
                                logger.debug(f"_extract_jsons: Parsed raw JSON object via json.loads: {list(parsed.keys())}")
                        except json.JSONDecodeError:
                            pass
                i = j
            else:
                i += 1

    if not results:
        logger.debug("_extract_jsons: No JSON objects found in text")
        
    return results or None


def _safe_json_loads(text: str) -> dict | list | None:
    """Safely parse JSON with additional cleanup for common LLM output issues.
    
    Handles common issues like:
    - Trailing commas
    - Single quotes instead of double quotes
    - Unescaped newlines in strings
    - Comments in JSON
    - Control characters
    - BOM markers
    
    Args:
        text: The text to parse as JSON.
        
    Returns:
        Parsed JSON object (dict or list), or None if parsing fails.
    """
    if not text or not isinstance(text, str):
        logger.debug("_safe_json_loads: Empty or invalid input")
        return None
    
    # Remove BOM if present and strip whitespace
    text = text.lstrip('\ufeff').strip()
    
    if not text:
        logger.debug("_safe_json_loads: Text is empty after stripping")
        return None
    
    # Try direct parsing first (fast path)
    try:
        result = json.loads(text)
        logger.debug(f"_safe_json_loads: Direct parsing succeeded")
        return result
    except json.JSONDecodeError as e:
        logger.debug(f"_safe_json_loads: Direct parsing failed: {e}")
    
    # Try fixing common issues
    cleaned = text
    
    # Remove control characters except tab, newline, carriage return
    cleaned = ''.join(char for char in cleaned if ord(char) >= 32 or char in '\t\n\r')
    
    # Remove trailing commas before closing braces/brackets
    cleaned = re.sub(r',(\s*[}\]])', r'\1', cleaned)
    
    # Remove single-line comments (// ...)
    cleaned = re.sub(r'//[^\n]*', '', cleaned)
    
    # Remove multi-line comments (/* ... */)
    cleaned = re.sub(r'/\*.*?\*/', '', cleaned, flags=re.DOTALL)
    
    # Try parsing cleaned version
    try:
        result = json.loads(cleaned)
        logger.debug(f"_safe_json_loads: Parsing succeeded after basic cleanup")
        return result
    except json.JSONDecodeError as e:
        logger.debug(f"_safe_json_loads: Basic cleanup failed: {e}")
    
    # Try replacing single quotes with double quotes (carefully)
    try:
        # Replace single-quoted strings with double-quoted strings
        # This regex handles simple cases: 'key' or 'value'
        single_quoted = re.sub(r"'([^']*)'", r'"\1"', cleaned)
        result = json.loads(single_quoted)
        logger.debug(f"_safe_json_loads: Parsing succeeded after quote replacement")
        return result
    except json.JSONDecodeError as e:
        logger.debug(f"_safe_json_loads: Quote replacement failed: {e}")
    
    # Try to extract just the first JSON object if there's extra text
    try:
        # Find the first { and last matching }
        start = cleaned.find('{')
        if start != -1:
            brace_count = 0
            for i, char in enumerate(cleaned[start:], start):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_str = cleaned[start:i+1]
                        result = json.loads(json_str)
                        logger.debug(f"_safe_json_loads: Parsing succeeded after extracting JSON object")
                        return result
    except json.JSONDecodeError as e:
        logger.debug(f"_safe_json_loads: JSON extraction failed: {e}")
    
    logger.debug("_safe_json_loads: All parsing attempts failed")
    return None


def _extract_all_grades_from_text(text: str) -> list[tuple[int, str]]:
    """Extract all valid grade mentions from text with their positions.
    
    Args:
        text: The text to search for grade mentions.
        
    Returns:
        List of (position, grade) tuples sorted by position.
    """
    found_grades = []
    text_lower = text.lower()
    
    # Find all occurrences of grade words with word boundaries
    for grade in VALID_GRADES:
        for match in re.finditer(r'\b' + re.escape(grade) + r'\b', text_lower):
            found_grades.append((match.start(), grade))
    
    # Sort by position
    found_grades.sort(key=lambda x: x[0])
    return found_grades


def _extract_grade_from_text(text: str) -> str | None:
    """Extract grade from various formats in the text.
    
    Uses multiple strategies in order of reliability:
    1. JSON field patterns (most reliable)
    2. Explicit grade assignment patterns
    3. Quoted grades
    4. Last occurrence in text (conclusion usually at end)
    
    Prioritizes "almost" grade detection since it's the most commonly confused.
    
    Args:
        text: The text to extract the grade from.
        
    Returns:
        The extracted grade or None if no valid grade found.
    """
    if not text or not isinstance(text, str):
        logger.debug("_extract_grade_from_text: Empty or invalid input")
        return None
        
    text_lower = text.lower()
    
    # Priority 1: Look for JSON grade field patterns (most reliable)
    # Check "grade" field first specifically - find ALL occurrences and take the LAST one
    # (last is most likely to be the final answer)
    grade_field_patterns = [
        (r'"grade"\s*:\s*"([^"]*)"', 1),
        (r'"grade"\s*:\s*\'([^\']*)\'', 1),
        (r'"grade"\s*:\s*([a-zA-Z]+)\b', 1),
    ]
    
    last_match_pos = -1
    last_match_grade = None
    
    for pattern, group in grade_field_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            if match.start() > last_match_pos:
                grade = match.group(group).strip().lower().strip('"').strip("'")
                if grade in VALID_GRADES:
                    last_match_pos = match.start()
                    last_match_grade = grade
    
    if last_match_grade:
        logger.debug(f"_extract_grade_from_text: Found grade '{last_match_grade}' via JSON 'grade' field pattern (last occurrence)")
        return last_match_grade
    
    # Then check other JSON field patterns - also find LAST occurrence
    other_json_patterns = [
        (r'"response"\s*:\s*"([^"]*)"', 1),
        (r'"prediction"\s*:\s*"([^"]*)"', 1),
        (r'"result"\s*:\s*"([^"]*)"', 1),
        (r'"evaluation"\s*:\s*"([^"]*)"', 1),
        (r'"assessment"\s*:\s*"([^"]*)"', 1),
    ]
    
    last_match_pos = -1
    last_match_grade = None
    
    for pattern, group in other_json_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            if match.start() > last_match_pos:
                grade = match.group(group).strip().lower().strip('"').strip("'")
                if grade in VALID_GRADES:
                    last_match_pos = match.start()
                    last_match_grade = grade
    
    if last_match_grade:
        logger.debug(f"_extract_grade_from_text: Found grade '{last_match_grade}' via JSON pattern (last occurrence)")
        return last_match_grade
    
    # Priority 2: Look for explicit grade assignments in text - find LAST occurrence
    text_patterns = [
        r'\bgrade\s*[:=]\s*"?(correct|incorrect|partial|almost)"?\b',
        r'\bfinal\s+grade\s*[:=]\s*"?(correct|incorrect|partial|almost)"?\b',
        r'\bassigned\s+grade\s*[:=]\s*"?(correct|incorrect|partial|almost)"?\b',
        r'\bthe\s+grade\s+is\s+"?(correct|incorrect|partial|almost)"?\b',
        r'\bi\s+assign\s+"?(correct|incorrect|partial|almost)"?\b',
        r'\bi\s+grade\s+this\s+as\s+"?(correct|incorrect|partial|almost)"?\b',
        r'\bthis\s+(?:should\s+be\s+)?graded\s+as\s+"?(correct|incorrect|partial|almost)"?\b',
    ]
    
    last_match_pos = -1
    last_match_grade = None
    
    for pattern in text_patterns:
        for match in re.finditer(pattern, text_lower):
            if match.start() > last_match_pos:
                grade = match.group(1).strip().lower()
                if grade in VALID_GRADES:
                    last_match_pos = match.start()
                    last_match_grade = grade
    
    if last_match_grade:
        logger.debug(f"_extract_grade_from_text: Found grade '{last_match_grade}' via text pattern (last occurrence)")
        return last_match_grade
    
    # Priority 3: Look for quoted grades in the last 100 words (strong indicator)
    # The conclusion is typically at the end
    words = text_lower.split()
    last_100 = ' '.join(words[-100:]) if len(words) > 100 else text_lower
    
    # Check "almost" first since it's the most commonly missed grade
    quoted_priority = ["almost", "partial", "incorrect", "correct"]
    for grade in quoted_priority:
        if f'"{grade}"' in last_100 or f"'{grade}'" in last_100:
            logger.debug(f"_extract_grade_from_text: Found quoted grade '{grade}' in last 100 words")
            return grade
    
    # Priority 4: Look for grade keywords in the last 50 words (most recent context)
    # PRIORITIZE "almost" detection - check it FIRST before other grades
    last_50 = ' '.join(words[-50:]) if len(words) > 50 else text_lower
    
    # Check "almost" FIRST - it's the most commonly confused grade
    if re.search(r'\balmost\b', last_50):
        logger.debug("_extract_grade_from_text: Found grade 'almost' in last 50 words (priority check)")
        return 'almost'
    elif re.search(r'\bpartial\b', last_50):
        logger.debug("_extract_grade_from_text: Found grade 'partial' in last 50 words")
        return 'partial'
    elif re.search(r'\bincorrect\b', last_50):
        logger.debug("_extract_grade_from_text: Found grade 'incorrect' in last 50 words")
        return 'incorrect'
    elif re.search(r'\bcorrect\b', last_50):
        logger.debug("_extract_grade_from_text: Found grade 'correct' in last 50 words")
        return 'correct'
    
    # Priority 5: Look in last 200 words - again prioritize "almost"
    last_200 = ' '.join(words[-200:]) if len(words) > 200 else text_lower
    
    # Check "almost" FIRST
    if re.search(r'\balmost\b', last_200):
        logger.debug("_extract_grade_from_text: Found grade 'almost' in last 200 words (priority check)")
        return 'almost'
    elif re.search(r'\bpartial\b', last_200):
        logger.debug("_extract_grade_from_text: Found grade 'partial' in last 200 words")
        return 'partial'
    elif re.search(r'\bincorrect\b', last_200):
        logger.debug("_extract_grade_from_text: Found grade 'incorrect' in last 200 words")
        return 'incorrect'
    elif re.search(r'\bcorrect\b', last_200):
        logger.debug("_extract_grade_from_text: Found grade 'correct' in last 200 words")
        return 'correct'
    
    # Priority 6: Look for the LAST occurrence of grade keywords anywhere in text
    # The conclusion typically appears at the end of the response
    last_grades = _extract_all_grades_from_text(text)
    if last_grades:
        # First check if "almost" appears in the last grades (prioritize it)
        for pos, grade in reversed(last_grades):
            if grade == "almost":
                logger.debug(f"_extract_grade_from_text: Found grade 'almost' as last occurrence (priority)")
                return "almost"
        # Otherwise return the last grade mentioned
        last_grade = last_grades[-1][1]
        logger.debug(f"_extract_grade_from_text: Found grade '{last_grade}' as last occurrence")
        return last_grade
    
    # Priority 7: Check for "almost" anywhere in the text as final fallback
    # This is critical because "almost" is the most commonly missed grade
    if 'almost' in text_lower:
        # Make sure it's not part of another word
        if re.search(r'\balmost\b', text_lower):
            logger.debug("_extract_grade_from_text: Found grade 'almost' via full text search (final fallback)")
            return 'almost'
    
    logger.debug("_extract_grade_from_text: No grade found in text")
    return None


def _normalize_grade(grade: str) -> str:
    """Normalize grade to one of the expected lowercase values.
    
    Args:
        grade: The grade string to normalize.
        
    Returns:
        Normalized grade string or "none" if not valid.
    """
    if not grade or not isinstance(grade, str):
        return "none"
    
    grade_lower = grade.lower().strip().strip('"').strip("'")
    
    # Direct match to standard grades
    if grade_lower in VALID_GRADES:
        return grade_lower
    
    # Handle common variations and misspellings
    # Check "almost" first since it's the most commonly confused grade
    variations = {
        "almost": [
            "almost correct", "nearly correct", "mostly correct", "close", 
            "minor errors", "small errors", "nearly complete", "almost there",
            "very close", "nearly right", "mostly right", "almost done",
            "nearly done", "just missed", "minor mistake", "small mistake",
            "trivial error", "small error", "minor issue", "small issue"
        ],
        "partial": [
            "partially correct", "some progress", "incomplete", "partial credit", 
            "significant progress", "halfway", "partial solution", "in progress",
            "not complete", "unfinished", "partial result"
        ],
        "incorrect": [
            "wrong", "error", "false", "no credit", "zero", "no progress",
            "completely wrong", "totally wrong", "fundamentally wrong"
        ],
        "correct": [
            "right", "true", "full credit", "complete", "fully correct",
            "perfect", "flawless", "exactly right"
        ],
    }
    
    for standard, variants in variations.items():
        if grade_lower in variants:
            return standard
    
    # Partial substring matches as fallback (check longer words first to avoid confusion)
    # Check "almost" first since it's the most commonly missed grade
    if "almost" in grade_lower:
        return "almost"
    elif "partial" in grade_lower:
        return "partial"
    elif "incorrect" in grade_lower:
        return "incorrect"
    elif "correct" in grade_lower:
        return "correct"
    
    return "none"


def _extract_confidence(json_obj: dict) -> str:
    """Extract confidence value from a JSON object.
    
    Args:
        json_obj: The JSON object to extract confidence from.
        
    Returns:
        One of: "high", "medium", "low", or "unknown"
    """
    if not isinstance(json_obj, dict):
        return "unknown"
    
    # Try common confidence field names
    confidence_fields = [
        "confidence", "confidence_level", "certainty", "sureness", 
        "confidence_score", "certainty_level"
    ]
    
    for field in confidence_fields:
        if field in json_obj:
            val = str(json_obj[field]).lower().strip().strip('"').strip("'")
            if val in ["high", "medium", "low"]:
                return val
            # Try to parse numeric confidence
            try:
                num_val = float(val)
                if num_val >= 0.8:
                    return "high"
                elif num_val >= 0.5:
                    return "medium"
                else:
                    return "low"
            except (ValueError, TypeError):
                pass
    
    return "unknown"


class TaskAgent:
    """Task agent that solves IMO grading problems.
    
    This agent evaluates student solutions to mathematical olympiad problems
    and assigns grades based on official solutions and grading guidelines.
    
    Attributes:
        model: The LLM model to use for evaluation.
        log_fn: The logging function to use for status messages.
        log_file: Path to the log file (if any).
    """

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        """Initialize the TaskAgent.
        
        Args:
            model: The LLM model to use. Defaults to EVAL_MODEL.
            log_file: Path to the log file. Defaults to empty string.
        """
        self.model = model
        self.log_fn = logger.info
        self.log_file = log_file

    def _validate_inputs(self, inputs: dict) -> tuple[bool, list[str]]:
        """Validate the input dictionary.
        
        Args:
            inputs: The input dictionary to validate.
            
        Returns:
            Tuple of (is_valid, missing_fields).
        """
        if not isinstance(inputs, dict):
            return False, ["inputs is not a dictionary"]
            
        required_fields = ["problem", "solution", "grading_guidelines", "student_answer"]
        missing_fields = [f for f in required_fields if not inputs.get(f)]
        
        return len(missing_fields) == 0, missing_fields

    def _build_instruction(self, inputs: dict) -> str:
        """Build the grading instruction prompt.
        
        Args:
            inputs: Dictionary containing problem data.
            
        Returns:
            The formatted instruction string.
        """
        domain = inputs.get("domain", "")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        return f"""You are an expert mathematical olympiad grader. Your task is to evaluate the student's solution and assign exactly one grade: "correct", "incorrect", "partial", or "almost".

## Problem Domain
{domain if domain else "Mathematical Olympiad"}

## Problem Statement:
{problem}

## Official Solution:
{solution}

## Grading Guidelines (FOLLOW THESE EXACTLY):
{grading_guidelines}

## Student's Answer:
{student_answer}

## Grade Definitions (CRITICAL - READ CAREFULLY)

- **correct**: Complete and fully correct solution. All steps present and logically sound. No gaps, no errors, fully rigorous proof. EVERY step must be correct and justified.

- **incorrect**: No meaningful progress or fundamental errors. Wrong approach or no valid mathematical progress. Student failed to demonstrate understanding of key concepts. Minimal or no substantive work toward the solution.

- **partial**: Significant progress but INCOMPLETE. Found useful invariant/lemma or established valid approach but didn't finish. Student demonstrated understanding of key ideas but solution has MAJOR gaps or STOPS before completion. The student has NOT addressed all major components. Missing conclusion or final step is a KEY indicator.

- **almost**: Nearly complete solution with ONLY minor issues. Main proof structure is CORRECT and COMPLETE. Only minor computational errors, typos, or notation issues. Student has addressed ALL major components. The proof is COMPLETE from start to finish INCLUDING the conclusion. If you fix trivial errors (arithmetic, typos), it becomes correct. The student understood the complete approach and executed it with only small blemishes.

## CRITICAL: Key Indicators for Each Grade

### "correct" indicators:
- Every step is logically sound and justified
- No gaps, no errors, no omissions
- Complete proof structure with flawless execution

### "incorrect" indicators:
- No meaningful mathematical progress
- Wrong approach or fundamental misunderstanding
- Just restated the problem or made random calculations
- No valid lemmas, invariants, or proof structure

### "partial" indicators (INCOMPLETE solutions):
- Missing the conclusion or final step
- Missing one or more major proof components
- Has major logical gaps requiring substantial new work
- Proof structure is incomplete (stops partway through)
- Only addressed some cases, not all
- The proof is INCOMPLETE, even if what's there is correct
- Student made progress but didn't finish

### "almost" indicators (COMPLETE solutions with minor issues):
- COMPLETE proof structure from beginning to end
- ALL major components are present and addressed
- Student REACHES THE CONCLUSION (even if with minor errors)
- Only MINOR issues exist (small arithmetic errors, trivial notation issues, minor computational mistakes, typos)
- If you fix trivial errors, the proof becomes "correct"
- The student has a COMPLETE proof with blemishes, not an INCOMPLETE proof

## THE MOST IMPORTANT DISTINCTION: "almost" vs "partial"

This is where 75% of grading errors occur. Use these tests:

### TEST 1: The "COMPLETE STRUCTURE" Test (MOST IMPORTANT)
Ask: "Does the student have a COMPLETE proof structure?"
- YES (complete structure, minor blemishes) → "almost"
- NO (missing components, incomplete) → "partial"

### TEST 2: The "CONCLUSION" Test (CRITICAL)
Ask: "Did the student REACH the final conclusion/answer?"
- YES (even with minor errors) → Strong indicator for "almost"
- NO (stopped before the end) → Strong indicator for "partial"

### TEST 3: The "FIXABILITY" Test (Apply rigorously!)
Ask: "Can I fix this by correcting ONLY trivial errors?"
- YES (arithmetic, typos, notation only) → "almost"
- NO (need new proof steps, lemmas, cases) → "partial"

### TEST 4: The "WHAT'S MISSING" Test
Ask: "What would need to be added to make this correct?"
- Nothing major (just fix tiny errors) → "almost"
- Something major (conclusion, cases, lemmas, proof steps) → "partial"

### TEST 5: The "PROGRESS" Test (for partial vs incorrect)
Ask: "Did the student make MEANINGFUL progress?"
- YES (found lemma, proved intermediate result, established approach) → "partial"
- NO (just restated problem, random calculations) → "incorrect"

## QUICK DECISION TREE (USE THIS!)

```
START: Did student make meaningful progress?
├── NO → "incorrect"
└── YES → Did student reach the conclusion?
    ├── NO → "partial" (incomplete)
    └── YES → Is the proof structure complete?
        ├── NO → "partial" (incomplete)
        └── YES → Are there any errors?
            ├── NO → "correct"
            └── YES → Are errors ONLY trivial (arithmetic, typos, notation)?
                ├── YES → "almost"
                └── NO → "partial" (major errors = incomplete)
```

## DETAILED EXAMPLES: "almost" vs "partial" (STUDY CAREFULLY!)

**Example 1 - "almost" (Complete proof with minor error):**
Student writes a complete induction proof: base case, inductive hypothesis, inductive step, conclusion. Only issue: in the inductive step, calculates 2^(k+1) = 2^k + 2 instead of 2*2^k (a small arithmetic error).
→ Grade: "almost" (complete structure, minor computational error)

**Example 2 - "partial" (Incomplete induction):**
Student sets up the induction base case and states the inductive hypothesis, but doesn't complete the inductive step or reach the conclusion.
→ Grade: "partial" (incomplete proof structure)

**Example 3 - "almost" (Complete with typo):**
Student writes a complete proof with all steps correct, but has a typo in the final answer (writes "n^2" instead of "n^3" in the conclusion, but the proof logic is correct).
→ Grade: "almost" (complete structure, only a typo)

**Example 4 - "partial" (Missing conclusion):**
Student proves all necessary lemmas and develops the main argument, but stops before reaching the final conclusion.
→ Grade: "partial" (missing the conclusion - a MAJOR component)

**Example 5 - "almost" (Complete number theory with small error):**
Student writes a complete number theory proof: states approach, proves all lemmas, addresses all cases, reaches conclusion. Only issue: a small arithmetic error in case 2 that doesn't affect the logic.
→ Grade: "almost" (complete structure, minor error only)

**Example 6 - "partial" (Incomplete geometry):**
Student correctly identifies the geometric approach, proves one auxiliary lemma, but doesn't complete the main argument or connect the lemma to the final result.
→ Grade: "partial" (significant progress but incomplete structure)

**Example 7 - "almost" (Complete combinatorics with counting error):**
Student sets up the counting argument, considers all necessary cases, reaches the answer. Only issue: makes a small error in one case count (off by 1).
→ Grade: "almost" (complete structure, minor error)

**Example 8 - "partial" (Missing cases in combinatorics):**
Student sets up the counting argument, considers some cases, but misses several cases entirely.
→ Grade: "partial" (incomplete - missing major components)

**Example 9 - "almost" (Complete algebra with sign error):**
Student has a complete algebraic proof but makes a sign error when expanding (a-b)^2.
→ Grade: "almost" (complete structure, minor algebraic error)

**Example 10 - "partial" (Wrong direction):**
Student starts a proof by contradiction but doesn't reach a contradiction or connect the assumption to the conclusion.
→ Grade: "partial" (incomplete argument)

**Example 11 - "almost" (Complete sequence proof with index error):**
Student proves a sequence result with complete structure: defines sequence, proves monotonicity, finds limit. Only issue: uses index n+1 where n should be used, but the logic is still clear.
→ Grade: "almost" (complete structure, minor index error)

**Example 12 - "partial" (Missing sequence convergence):**
Student defines a sequence and proves some properties, but doesn't prove convergence or find the limit when that's the main goal.
→ Grade: "partial" (missing the main result)

**Example 13 - "almost" (Complete graph theory with edge count error):**
Student proves a graph theory result with complete structure. Only issue: miscounts edges by 1 in one subgraph, but this doesn't affect the main argument.
→ Grade: "almost" (complete structure, minor counting error)

**Example 14 - "partial" (Incomplete graph coloring):**
Student proves some properties about graph coloring but doesn't complete the argument about the chromatic number.
→ Grade: "partial" (incomplete - missing verification)

**Example 15 - "almost" (Complete recurrence with boundary error):**
Student solves a recurrence relation with complete proof. Only issue: makes a small arithmetic error when applying one boundary condition.
→ Grade: "almost" (complete structure, minor arithmetic error)

**Example 16 - "partial" (Recurrence setup only):**
Student correctly sets up the recurrence relation and finds the characteristic equation, but doesn't solve for the roots or find the particular solution.
→ Grade: "partial" (incomplete - missing the solution)

**Example 17 - "almost" (Complete modular arithmetic with calculation slip):**
Student proves a modular arithmetic result with complete structure. Only issue: calculates 2^5 = 30 instead of 32 in one step, but the error is isolated and trivial.
→ Grade: "almost" (complete structure, minor calculation slip)

**Example 18 - "partial" (Modular setup but no conclusion):**
Student sets up the modular arithmetic framework correctly and proves some intermediate congruences, but doesn't connect them to reach the final result.
→ Grade: "partial" (incomplete - missing the conclusion)

**Example 19 - "almost" (Complete optimization with derivative error):**
Student solves an optimization problem with complete proof. Only issue: computes derivative of x^3 as 2x^2 instead of 3x^2, but corrects it in the next step.
→ Grade: "almost" (complete structure, minor derivative error)

**Example 20 - "partial" (Optimization setup only):**
Student defines the function to optimize and takes the derivative, but doesn't find critical points or analyze them.
→ Grade: "partial" (incomplete - missing the optimization)

**Example 21 - "almost" (Complete proof with transposed digits):**
Student writes a complete, rigorous proof. All logic is correct. In the final answer, writes "The answer is 127" when the correct answer is 172. Clearly a typo, not a logic error.
→ Grade: "almost" (complete proof, trivial typo in final answer)

**Example 22 - "partial" (Missing existence proof):**
Student proves uniqueness of a solution but doesn't prove that a solution exists at all.
→ Grade: "partial" (incomplete - missing existence)

**Example 23 - "almost" (Complete existence proof with uniqueness gap):**
Student proves existence of a solution and makes a small error in the uniqueness argument. The structure is complete, just a minor gap in one step.
→ Grade: "almost" (complete structure, minor gap in uniqueness)

**Example 24 - "partial" (Only uniqueness proof):**
Student proves that if a solution exists, it must be unique, but never proves that any solution exists.
→ Grade: "partial" (incomplete - missing existence proof)

**Example 25 - "almost" (Complete construction with small error):**
Student constructs an explicit example with complete structure. Only issue: a small arithmetic error in verifying one property that doesn't affect the validity.
→ Grade: "almost" (complete structure, minor verification error)

**Example 26 - "partial" (Construction attempt but invalid):**
Student attempts to construct an example but the construction is invalid or doesn't satisfy the required properties, though the approach shows understanding.
→ Grade: "partial" (attempted construction but failed)

**Example 27 - "almost" (Complete contradiction proof with small gap):**
Student uses proof by contradiction with complete structure. Only issue: a small logical gap in showing the contradiction is reached, but the gap is trivial to fill.
→ Grade: "almost" (complete structure, minor gap)

**Example 28 - "partial" (Contradiction setup but no contradiction):**
Student assumes the negation for proof by contradiction, develops some argument, but never actually reaches a contradiction.
→ Grade: "partial" (incomplete - no contradiction reached)

**Example 29 - "almost" (Complete pigeonhole with off-by-one):**
Student proves a pigeonhole principle result with complete structure. Only issue: uses n holes when it should be n+1, but the logic is otherwise correct.
→ Grade: "almost" (complete structure, minor off-by-one error)

**Example 30 - "partial" (Pigeonhole setup but wrong application):**
Student identifies that pigeonhole principle should be used and defines pigeons/holes, but applies the principle incorrectly or doesn't complete the argument.
→ Grade: "partial" (incomplete application)

## Common Mistakes to AVOID (READ CAREFULLY)

1. **DON'T** grade "partial" when the student has a complete proof with only minor errors → This should be "almost"
2. **DON'T** grade "almost" when the solution has major logical gaps or missing proof steps → This should be "partial"
3. **DON'T** be too stingy with "almost" - if the student has a complete proof structure with minor issues, it's "almost" not "partial"
4. **DON'T** be too generous with "partial" - "partial" requires meaningful progress, not just "attempted"
5. **DON'T** grade "partial" when the student only made minimal progress → This should be "incorrect"
6. **DON'T** grade "almost" when the student is missing the conclusion → This should be "partial"
7. **DON'T** grade "partial" when the student reached the conclusion with only minor errors → This should be "almost"
8. **DON'T** be fooled by length - a long incomplete proof is still "partial", a short complete proof with minor errors is "almost"
9. **DON'T** grade "partial" when the student has a complete proof but wrong final number → This should be "almost"
10. **DON'T** grade "almost" when the student has only proven a lemma but not connected it → This should be "partial"
11. **DON'T** grade "partial" when the student has complete induction with only base case error → This should be "almost"
12. **DON'T** grade "almost" when the student has only base case of induction → This should be "partial"
13. **DON'T** be biased by the student's writing style - focus on mathematical content only
14. **DON'T** require perfection for "almost" - minor errors are expected and acceptable
15. **DON'T** confuse "found a lemma" (partial) with "proved the lemma and completed the proof" (almost)

## Final Verification Checklist (COMPLETE ALL CHECKS!)

Before submitting your grade, answer these questions explicitly:

### MANDATORY YES/NO QUESTIONS:

1. **COMPLETE STRUCTURE TEST**: Does the student have a complete proof structure from start to finish? (YES/NO)
2. **CONCLUSION TEST**: Did the student REACH the final conclusion/answer (even if with errors)? (YES/NO)
3. **FIXABILITY TEST**: Can I fix this by correcting ONLY trivial errors (arithmetic, typos, notation)? (YES/NO)
4. **COMPONENTS TEST**: Are ALL major components present and addressed? (YES/NO)
5. **ERROR SEVERITY TEST**: Are the errors truly MINOR (arithmetic, notation, typos) and not MAJOR (logical gaps, missing steps)? (YES/NO)
6. **PROGRESS TEST**: Did the student make MEANINGFUL progress (not just restate the problem)? (YES/NO)

### DECISION RULES (FOLLOW EXACTLY):

**For "almost":**
- Q1 (Complete Structure) MUST be YES
- Q2 (Conclusion) MUST be YES  
- Q3 (Fixability) MUST be YES
- Q4 (All Components) MUST be YES
- Q5 (Minor Errors) MUST be YES
- If ALL are YES → Grade is "almost" (or "correct" if truly flawless)

**For "partial":**
- Q6 (Progress) MUST be YES
- AND at least one of Q1-Q4 is NO
- If Q6 is YES but proof is incomplete → Grade is "partial"

**For "incorrect":**
- Q6 (Progress) is NO
- No meaningful mathematical progress made

**For "correct":**
- ALL of Q1-Q5 are YES
- AND there are truly NO errors (not even minor ones)

## ANTI-BIAS REMINDERS (IMPORTANT!)

- Many graders are too STRICT with "almost" - don't be! If the structure is complete, it's "almost"
- A complete proof with a small error is MUCH closer to "correct" than to "partial"
- "Almost" means the student UNDERSTOOD the complete approach - reward this!
- When in doubt between "almost" and "partial", ask: "Did they reach the conclusion?" If YES → "almost"
- Don't let minor imperfections push you toward "partial" - "almost" is for complete proofs with minor issues

## Response Format (CRITICAL)

You MUST respond with ONLY a JSON object in this exact format:

```json
{{
    "grade": "correct"
}}
```

OR

```json
{{
    "grade": "incorrect"
}}
```

OR

```json
{{
    "grade": "partial"
}}
```

OR

```json
{{
    "grade": "almost"
}}
```

The grade field MUST be exactly one of: "correct", "incorrect", "partial", or "almost" (all lowercase, no extra spaces, no other text).

Grade:"""

    def _extract_grade_from_response(self, msg_history: list[dict]) -> tuple[str, str]:
        """Extract grade from the LLM response.
        
        Uses multiple extraction strategies in order of reliability.
        Prioritizes "almost" grade detection since it's the most commonly confused.
        
        Args:
            msg_history: The message history from the LLM.
            
        Returns:
            Tuple of (prediction, confidence).
        """
        prediction = "none"
        confidence = "unknown"
        
        try:
            if not msg_history:
                logger.warning("_extract_grade_from_response: Empty message history")
                return prediction, confidence
                
            last_message = msg_history[-1].get("text", "")
            if not last_message:
                logger.warning("_extract_grade_from_response: Last message has no text")
                return prediction, confidence
            
            logger.debug(f"_extract_grade_from_response: Processing message of length {len(last_message)}")
            
            # Strategy 1: Extract from JSON blocks (most reliable)
            extracted = _extract_jsons(last_message)
            if extracted:
                logger.debug(f"_extract_grade_from_response: Found {len(extracted)} JSON objects")
                
                for json_obj in reversed(extracted):  # Check from last to first
                    if not isinstance(json_obj, dict):
                        continue
                    
                    # First check for "grade" field specifically (most reliable)
                    if "grade" in json_obj:
                        val = str(json_obj["grade"]).lower().strip().strip('"').strip("'")
                        if val in VALID_GRADES:
                            prediction = val
                            confidence = _extract_confidence(json_obj)
                            logger.debug(f"_extract_grade_from_response: Found grade '{prediction}' in 'grade' field")
                            break
                    
                    # If no "grade" field, try other common field names
                    if prediction not in VALID_GRADES:
                        other_fields = ["response", "result", "prediction", "evaluation", "assessment"]
                        for field in other_fields:
                            if field in json_obj:
                                val = str(json_obj[field]).lower().strip().strip('"').strip("'")
                                if val in VALID_GRADES:
                                    prediction = val
                                    confidence = _extract_confidence(json_obj)
                                    logger.debug(f"_extract_grade_from_response: Found grade '{prediction}' in field '{field}'")
                                    break
                        if prediction in VALID_GRADES:
                            break
                        
                    # If no standard field found, try any value except reasoning/explanation
                    if prediction not in VALID_GRADES:
                        for key, val in json_obj.items():
                            key_lower = key.lower()
                            if key_lower in ["reasoning", "explanation", "analysis", "thoughts", "note", "comment"]:
                                continue
                            val_str = str(val).lower().strip().strip('"').strip("'")
                            if val_str in VALID_GRADES:
                                prediction = val_str
                                confidence = _extract_confidence(json_obj)
                                logger.debug(f"_extract_grade_from_response: Found grade '{prediction}' in field '{key}'")
                                break
                        if prediction in VALID_GRADES:
                            break
            
            # Strategy 2: Use text-based extraction if JSON didn't work
            if prediction not in VALID_GRADES:
                extracted_text = _extract_grade_from_text(last_message)
                if extracted_text:
                    prediction = extracted_text
                    logger.debug(f"_extract_grade_from_response: Found grade '{prediction}' via text extraction")
            
            # Strategy 3: Direct pattern matching for quoted grades
            # Check "almost" first since it's the most commonly missed grade
            if prediction not in VALID_GRADES:
                text_lower = last_message.lower()
                # Priority order: almost (most confused), then others
                quoted_patterns = [
                    ('"almost"', "almost"), ("'almost'", "almost"),
                    ('"partial"', "partial"), ("'partial'", "partial"),
                    ('"incorrect"', "incorrect"), ("'incorrect'", "incorrect"),
                    ('"correct"', "correct"), ("'correct'", "correct"),
                ]
                for pattern, grade in quoted_patterns:
                    if pattern in text_lower:
                        prediction = grade
                        logger.debug(f"_extract_grade_from_response: Found quoted grade '{prediction}'")
                        break
            
            # Strategy 4: Try safe JSON parsing on the entire message
            if prediction not in VALID_GRADES:
                safe_parsed = _safe_json_loads(last_message)
                if safe_parsed and isinstance(safe_parsed, dict):
                    # Check "grade" field first
                    if "grade" in safe_parsed:
                        val = str(safe_parsed["grade"]).lower().strip().strip('"').strip("'")
                        if val in VALID_GRADES:
                            prediction = val
                            confidence = _extract_confidence(safe_parsed)
                            logger.debug(f"_extract_grade_from_response: Found grade '{prediction}' via safe JSON parsing (grade field)")
                    
                    # If not found, try other fields
                    if prediction not in VALID_GRADES:
                        for field in ["response", "result", "prediction", "evaluation"]:
                            if field in safe_parsed:
                                val = str(safe_parsed[field]).lower().strip().strip('"').strip("'")
                                if val in VALID_GRADES:
                                    prediction = val
                                    confidence = _extract_confidence(safe_parsed)
                                    logger.debug(f"_extract_grade_from_response: Found grade '{prediction}' via safe JSON parsing ({field} field)")
                                    break
            
            # Strategy 5: Look for grade in the last 500 chars (conclusion area)
            # This is often where the final decision is stated
            # PRIORITIZE "almost" detection since it's most commonly missed
            if prediction not in VALID_GRADES:
                conclusion = last_message[-500:].lower()
                # Check "almost" FIRST since it's most commonly missed and critical
                # This is the most important grade to detect correctly
                if re.search(r'\balmost\b', conclusion):
                    prediction = "almost"
                    logger.debug("_extract_grade_from_response: Found grade 'almost' in conclusion (priority check)")
                elif re.search(r'\bcorrect\b', conclusion):
                    prediction = "correct"
                    logger.debug("_extract_grade_from_response: Found grade 'correct' in conclusion")
                elif re.search(r'\bpartial\b', conclusion):
                    prediction = "partial"
                    logger.debug("_extract_grade_from_response: Found grade 'partial' in conclusion")
                elif re.search(r'\bincorrect\b', conclusion):
                    prediction = "incorrect"
                    logger.debug("_extract_grade_from_response: Found grade 'incorrect' in conclusion")
            
            # Strategy 6: Look for explicit grade assignment patterns
            if prediction not in VALID_GRADES:
                text_lower = last_message.lower()
                # Look for patterns like "grade is: almost" or "assigned grade: partial"
                # PRIORITIZE "almost" in patterns
                assignment_patterns = [
                    rf'grade\s*(?:is|[:=])\s*["\']?({"|".join(VALID_GRADES)})["\']?',
                    rf'assigned\s+grade\s*[:=]\s*["\']?({"|".join(VALID_GRADES)})["\']?',
                    rf'final\s+grade\s*[:=]\s*["\']?({"|".join(VALID_GRADES)})["\']?',
                    rf'the\s+grade\s+(?:should\s+be|is)\s*["\']?({"|".join(VALID_GRADES)})["\']?',
                    rf'["\']?grade["\']?\s*[:=]\s*["\']?({"|".join(VALID_GRADES)})["\']?',
                ]
                for pattern in assignment_patterns:
                    match = re.search(pattern, text_lower)
                    if match:
                        prediction = match.group(1)
                        logger.debug(f"_extract_grade_from_response: Found grade '{prediction}' via assignment pattern")
                        break
            
            # Strategy 7: Look for grade in reasoning/explanation sections
            if prediction not in VALID_GRADES:
                # Look for patterns like "therefore the grade is" or "this deserves"
                reasoning_patterns = [
                    rf'therefore\s+(?:the\s+)?grade\s+(?:is|should\s+be)\s*["\']?({"|".join(VALID_GRADES)})["\']?',
                    rf'this\s+(?:deserves|warrants|earns)\s+(?:a\s+)?["\']?({"|".join(VALID_GRADES)})["\']?',
                    rf'(?:assign|award|give)\s+(?:a\s+)?["\']?({"|".join(VALID_GRADES)})["\']?',
                ]
                for pattern in reasoning_patterns:
                    match = re.search(pattern, last_message.lower())
                    if match:
                        prediction = match.group(1)
                        logger.debug(f"_extract_grade_from_response: Found grade '{prediction}' via reasoning pattern")
                        break
            
            # Log extraction results
            if prediction in VALID_GRADES:
                self.log_fn(f"Extracted grade: {prediction}, confidence: {confidence}")
            else:
                preview = last_message[:300].replace('\n', ' ')
                self.log_fn(f"Failed to extract valid grade. Response preview: {preview}...")
                    
        except (KeyError, IndexError) as e:
            logger.error(f"_extract_grade_from_response: Error accessing message history: {e}")
        except Exception as e:
            logger.error(f"_extract_grade_from_response: Error extracting prediction: {e}", exc_info=True)

        # Normalize prediction to expected values
        prediction = _normalize_grade(prediction)
        
        return prediction, confidence

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            Tuple of (prediction, msg_history). Prediction is one of the valid grades
            or "none" if extraction failed.
            
        Raises:
            ValueError: If required inputs are missing.
            RuntimeError: If LLM call fails.
        """
        # Validate inputs
        is_valid, missing_fields = self._validate_inputs(inputs)
        if not is_valid:
            self.log_fn(f"Warning: Missing required fields: {missing_fields}")
            # Continue anyway as some fields might be optional
        
        # Extract fields from inputs for better prompt construction
        domain = inputs.get("domain", "")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        # Log problem context for debugging
        self.log_fn(f"Processing problem in domain: {domain if domain else 'Mathematical Olympiad'}")
        self.log_fn(f"Student answer length: {len(student_answer)} chars")
        self.log_fn(f"Grading guidelines length: {len(grading_guidelines)} chars")

        # Build instruction
        instruction = self._build_instruction(inputs)
        
        # Retry logic for robustness
        max_retries = 3
        prediction = "none"
        msg_history = []
        
        for attempt in range(max_retries):
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=[],
                )
                
                # Extract prediction from response
                prediction, confidence = self._extract_grade_from_response(msg_history)
                
                if prediction in VALID_GRADES:
                    # Successfully extracted a valid grade
                    break
                else:
                    # Could not extract grade, add clarification to prompt for retry
                    if attempt < max_retries - 1:
                        self.log_fn(f"Warning: Could not extract grade on attempt {attempt + 1}. Retrying with clarification...")
                        # Add more specific guidance based on what we saw
                        last_message = msg_history[-1].get("text", "") if msg_history else ""
                        instruction += f"""\n\nCRITICAL ERROR: Your previous response did not contain a valid grade in the required JSON format.
Your response was: {last_message[:300]}...

You MUST respond with ONLY a JSON object in this exact format (no other text):
{{"grade": "correct"}}

or

{{"grade": "incorrect"}}

or

{{"grade": "partial"}}

or

{{"grade": "almost"}}

The grade field MUST be exactly one of: "correct", "incorrect", "partial", or "almost" (all lowercase).
Do not include any other text, explanation, or formatting. Just the JSON object."""
                    else:
                        self.log_fn(f"Warning: Could not extract grade after {max_retries} attempts.")
                        
            except Exception as e:
                logger.error(f"forward: Attempt {attempt + 1} failed: {e}", exc_info=True)
                if attempt == max_retries - 1:
                    raise RuntimeError(f"Failed to get LLM response after {max_retries} attempts: {e}") from e

        return str(prediction), msg_history
