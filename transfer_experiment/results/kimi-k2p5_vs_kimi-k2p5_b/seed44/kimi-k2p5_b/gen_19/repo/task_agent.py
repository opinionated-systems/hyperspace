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
    last_50 = ' '.join(words[-50:]) if len(words) > 50 else text_lower
    
    # Check in order of specificity (most specific first)
    # "almost" and "partial" are more specific than "correct" and "incorrect"
    if re.search(r'\balmost\b', last_50):
        logger.debug("_extract_grade_from_text: Found grade 'almost' in last 50 words")
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
    
    # Priority 5: Look in last 200 words
    last_200 = ' '.join(words[-200:]) if len(words) > 200 else text_lower
    
    if re.search(r'\balmost\b', last_200):
        logger.debug("_extract_grade_from_text: Found grade 'almost' in last 200 words")
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
        "almost": ["almost correct", "nearly correct", "mostly correct", "close", "minor errors", "small errors", "nearly complete"],
        "partial": ["partially correct", "some progress", "incomplete", "partial credit", "significant progress"],
        "incorrect": ["wrong", "error", "false", "no credit", "zero", "no progress"],
        "correct": ["right", "true", "full credit", "complete", "fully correct"],
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
        
        return f"""You are an expert mathematical olympiad grader. Evaluate the student's solution and assign exactly one grade.

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

- **correct**: Complete and fully correct solution. All steps present and logically sound. No gaps, no errors, fully rigorous proof.

- **incorrect**: No meaningful progress or fundamental errors. Wrong approach or no valid mathematical progress. Student failed to demonstrate understanding of key concepts.

- **partial**: Significant progress but incomplete. Found useful invariant/lemma or established valid approach but didn't finish. Student demonstrated understanding of key ideas but solution has major gaps or stops before completion.

- **almost**: Nearly complete with only minor issues. Main proof structure correct, only small computational errors or minor omissions. Student understood the complete approach and executed it with only trivial mistakes that don't affect the main argument.

## CRITICAL DISTINCTION: "almost" vs "partial" (MOST IMPORTANT - READ TWICE!)

This is the most commonly confused distinction. The evaluation data shows this is where graders make the most errors. Pay EXTREME attention:

**"almost" means:**
- The student has a COMPLETE proof structure from start to finish
- All main ideas and proof steps are present and correct
- Only MINOR issues exist (small arithmetic errors, trivial notation issues, minor computational mistakes)
- The issues are COSMETIC and don't affect the mathematical validity
- If the minor issues were fixed, the solution would be immediately "correct"
- The student clearly understood and executed the full solution approach
- The solution reads like a complete proof with tiny blemishes
- The student has addressed ALL major components of the problem

**"partial" means:**
- The student made SIGNIFICANT progress but the solution is INCOMPLETE
- Key lemmas or invariants were found but NOT fully connected to the main proof
- Major logical gaps exist that would require substantial additional work to fix
- The solution is missing critical components needed for a complete proof
- Even after fixing minor errors, substantial work would still be needed
- The student understood key ideas but didn't complete the argument
- The student has NOT addressed all major components of the problem

**THE KEY QUESTION - ASK THIS EXPLICITLY:**
"Does the student have a COMPLETE proof structure with only minor blemishes, or is the proof MISSING substantial components?"

**DECISION TEST (USE THIS EXACT TEST):**
1. Imagine you are fixing the student's solution
2. If you can fix it by correcting only trivial errors (arithmetic, notation, typos) → "almost"
3. If the solution needs substantial new arguments, proof steps, or major components to be complete → "partial"
4. If you're unsure, ask: "Did the student complete the ENTIRE proof structure?" Yes → "almost", No → "partial"

## Grading Decision Process (FOLLOW THIS ORDER EXACTLY)

1. **FIRST** - Read the grading guidelines carefully. Identify what specific criteria are listed under each grade category. These are your PRIMARY criteria.

2. **SECOND** - Analyze the student's answer against the guidelines:
   - Match the student's work to the specific criteria in the guidelines
   - Check which grade's criteria best describes what the student achieved

3. **THIRD** - Apply the "COMPLETE STRUCTURE" test for "almost" vs "partial":
   - Does the student have a COMPLETE proof structure from beginning to end?
   - Are ALL major proof components present (even if with minor errors)?
   - If YES to both → "almost" (or "correct" if flawless)
   - If NO (missing components) → "partial" (or "incorrect" if no progress)

4. **FOURTH** - Apply the "FIXABILITY" test:
   - Ask: "If I fixed only trivial errors (arithmetic, typos), would this be a complete correct proof?"
   - If YES → "almost"
   - If NO (needs substantial new work) → "partial"

5. **FIFTH** - Verify your decision:
   - Does the student have a complete proof structure? (Yes = almost/correct, No = partial/incorrect)
   - Are the errors minor/cosmetic or major/substantial?

6. **FINAL** - Your grade must match the explicit criteria in the guidelines

## Detailed Examples of Correct Grading

**Example 1 - "almost" grading:**
Grading Guidelines say:
(Almost)
 1. Complete proof with minor computational error.

Student's answer: Full rigorous proof with a small arithmetic mistake in one calculation (e.g., wrote 2+2=5 in one line) that doesn't affect the main argument structure.
→ CORRECT grade: "almost" (complete structure, only minor error)

**Example 2 - "partial" grading:**
Grading Guidelines say:
(Partial)
 1. Found a useful invariant.
 2. Proved a key lemma.

Student's answer: Correctly identifies and proves the invariant and key lemma, but stops there without connecting to the main problem or completing the proof.
→ CORRECT grade: "partial" (significant progress but incomplete - needs more work to finish)

**Example 3 - "almost" vs "partial" distinction (CRITICAL):**
Grading Guidelines say:
(Partial)
 1. Found the key lemma.
(Almost)
 1. Proved the key lemma and set up the main argument.

Student's answer: States and proves the key lemma, sets up the framework for the main proof, outlines all remaining steps, but has a small gap in the final algebraic simplification.
→ CORRECT grade: "almost" (student went beyond Partial - they proved the lemma AND set up the complete argument. Only a minor gap remains that doesn't require substantial new work)

**Example 4 - "almost" with minor error:**
Student provides a complete induction proof but makes a small error in the base case calculation (e.g., computes 2^3=7 instead of 8). The inductive step is fully correct.
→ CORRECT grade: "almost" (complete proof structure, only minor computational error)

**Example 5 - "partial" with incomplete argument:**
Student correctly identifies that the problem requires induction and sets up the inductive hypothesis, but fails to complete the inductive step or verify the base case.
→ CORRECT grade: "partial" (significant progress - knows to use induction - but incomplete proof)

**Example 6 - "almost" (NOT "partial"):**
Grading Guidelines: (Partial) Found key lemma. (Almost) Complete proof with minor errors.
Student: States and proves the key lemma, develops the complete proof structure, addresses all cases, but makes a small error in one calculation.
→ CORRECT grade: "almost" (complete proof structure with minor error, not just "found lemma")

**Example 7 - "partial" (NOT "almost"):**
Grading Guidelines: (Partial) Found key lemma. (Almost) Complete proof with minor errors.
Student: States the key lemma, begins to develop the proof, but stops before completing all cases or addressing the main conclusion.
→ CORRECT grade: "partial" (incomplete proof structure, missing substantial components)

**Example 8 - "almost" (NOT "correct"):**
Grading Guidelines: (Correct) Flawless proof. (Almost) Complete proof with minor errors.
Student: Complete proof with rigorous structure, but has one small logical gap in a lemma or minor computational error.
→ CORRECT grade: "almost" (has a gap/error, so not "correct")

**Example 9 - "partial" vs "incorrect":**
Grading Guidelines: (Partial) Considered a prime p|xy+1.
Student: Mentions considering primes dividing xy+1, sets up some equations, demonstrates understanding of the approach, but doesn't complete the proof.
→ CORRECT grade: "partial" (made meaningful progress on the core approach, met Partial criterion)

**Example 10 - "almost" with complete structure:**
Student writes a complete number theory proof: states the approach, proves all necessary lemmas, addresses all cases, reaches the conclusion. Only issue: a small arithmetic error in case 2 that doesn't affect the logic.
→ CORRECT grade: "almost" (complete structure, minor error only)

## Common Mistakes to Avoid (READ CAREFULLY)

1. **DON'T** grade "partial" when the student has a complete proof with only minor errors → This should be "almost"
2. **DON'T** grade "almost" when the solution has major logical gaps or missing proof steps → This should be "partial"
3. **DON'T** ignore the specific criteria in the grading guidelines - they take precedence
4. **DON'T** grade based on your own judgment alone - follow the guidelines criteria
5. **DON'T** be too stingy with "almost" - if the student has a complete proof structure with minor issues, it's "almost" not "partial"
6. **DON'T** be too generous with "partial" - "partial" requires meaningful progress, not just "attempted"

## Final Verification Checklist

Before submitting your grade, verify:
- [ ] Did I check if the student has a COMPLETE proof structure?
- [ ] Did I apply the "fixability test" (trivial errors only = "almost")?
- [ ] Does my grade match the specific criteria in the guidelines?
- [ ] Am I being too strict with "almost" or too generous with "partial"?

## Response Format (CRITICAL - FOLLOW EXACTLY)

Respond ONLY with a JSON object. The grade field must be exactly one of: "correct", "incorrect", "partial", or "almost".

<json>
{{
    "grade": "correct" | "incorrect" | "partial" | "almost",
    "reasoning": "Brief explanation of why this grade was assigned, referencing specific criteria from the guidelines"
}}
</json>

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
            
            # Strategy 5: Look for grade in the last 300 chars (conclusion area)
            # This is often where the final decision is stated
            if prediction not in VALID_GRADES:
                conclusion = last_message[-300:].lower()
                # Check "almost" first since it's most commonly missed
                for grade in ["almost", "partial", "incorrect", "correct"]:
                    if re.search(rf'\b{grade}\b', conclusion):
                        prediction = grade
                        logger.debug(f"_extract_grade_from_response: Found grade '{prediction}' in conclusion")
                        break
            
            # Strategy 6: Look for explicit grade assignment patterns
            if prediction not in VALID_GRADES:
                text_lower = last_message.lower()
                # Look for patterns like "grade is: almost" or "assigned grade: partial"
                assignment_patterns = [
                    rf'grade\s*(?:is|[:=])\s*["\']?({"|".join(VALID_GRADES)})["\']?',
                    rf'assigned\s+grade\s*[:=]\s*["\']?({"|".join(VALID_GRADES)})["\']?',
                    rf'final\s+grade\s*[:=]\s*["\']?({"|".join(VALID_GRADES)})["\']?',
                    rf'the\s+grade\s+(?:should\s+be|is)\s*["\']?({"|".join(VALID_GRADES)})["\']?',
                ]
                for pattern in assignment_patterns:
                    match = re.search(pattern, text_lower)
                    if match:
                        prediction = match.group(1)
                        logger.debug(f"_extract_grade_from_response: Found grade '{prediction}' via assignment pattern")
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

        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
        except Exception as e:
            logger.error(f"forward: LLM call failed: {e}", exc_info=True)
            raise RuntimeError(f"Failed to get LLM response: {e}") from e

        # Extract prediction from response
        prediction, confidence = self._extract_grade_from_response(msg_history)

        return str(prediction), msg_history
