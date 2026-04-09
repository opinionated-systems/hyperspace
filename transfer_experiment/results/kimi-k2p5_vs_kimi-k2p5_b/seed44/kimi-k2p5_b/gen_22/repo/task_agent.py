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

- **correct**: Complete and fully correct solution. All steps present and logically sound. No gaps, no errors, fully rigorous proof.

- **incorrect**: No meaningful progress or fundamental errors. Wrong approach or no valid mathematical progress. Student failed to demonstrate understanding of key concepts.

- **partial**: Significant progress but incomplete. Found useful invariant/lemma or established valid approach but didn't finish. Student demonstrated understanding of key ideas but solution has major gaps or stops before completion. The student has NOT addressed all major components.

- **almost**: Nearly complete with only minor issues. Main proof structure correct, only small computational errors or minor omissions. Student understood the complete approach. The student HAS addressed ALL major components - the proof is complete from start to finish.

## THE DECISION FRAMEWORK: "COMPLETE STRUCTURE" TEST (MOST CRITICAL)

Before assigning any grade, ask yourself this ONE question:

**"Does the student have a COMPLETE proof structure from start to finish?"**

- **YES** → The grade is either "correct" (if flawless) or "almost" (if minor issues)
- **NO** → The grade is either "partial" (if significant progress) or "incorrect" (if minimal progress)

This is THE most important test. Do not skip it.

## CRITICAL DISTINCTION: "almost" vs "partial" (READ THIS SECTION THREE TIMES!)

This is where most grading errors occur. The evaluation data shows 75% of errors happen here. Pay EXTREME attention:

### "almost" means ALL of these are true:
1. ✅ COMPLETE proof structure from beginning to end
2. ✅ ALL major components are present and addressed
3. ✅ Student has a full argument that reaches the conclusion
4. ✅ Only MINOR issues exist (small arithmetic errors, trivial notation issues, minor computational mistakes, typos)
5. ✅ If you fix the trivial errors, the proof becomes "correct"

### "partial" means ANY of these are true:
1. ❌ Missing the conclusion or final step
2. ❌ Missing one or more major proof components
3. ❌ Has major logical gaps that require substantial new work to fill
4. ❌ Proof structure is incomplete (stops partway through)
5. ❌ Only addressed some cases, not all

### THE "FIXABILITY" TEST (Apply this rigorously!):
- Can you fix the solution by correcting ONLY trivial errors (arithmetic, typos, notation)? → **"almost"**
- Would you need to add new proof steps, lemmas, or cases? → **"partial"**

## DETAILED EXAMPLES: "almost" vs "partial" (STUDY CAREFULLY!)

**Example 1 - "almost" (NOT "partial"):**
Grading Guidelines: (Partial) Found key lemma. (Almost) Complete proof with minor errors.
Student: States AND proves the key lemma, develops the complete proof, addresses all cases, reaches the conclusion. Only issue: a small arithmetic error in one calculation.
→ CORRECT grade: "almost" (complete proof structure with minor error)

**Example 2 - "partial" (NOT "almost"):**
Grading Guidelines: (Partial) Found key lemma. (Almost) Complete proof with minor errors.
Student: States the key lemma, begins to develop the proof, but stops before completing all cases or addressing the main conclusion.
→ CORRECT grade: "partial" (incomplete proof structure, missing substantial components)

**Example 3 - "almost" (NOT "correct"):**
Grading Guidelines: (Correct) Flawless proof. (Almost) Complete proof with minor errors.
Student: Complete proof with rigorous structure, but has one small logical gap in a lemma or minor computational error.
→ CORRECT grade: "almost" (has a gap/error, so not "correct")

**Example 4 - "partial" (NOT "almost") - Missing conclusion:**
Student proves all necessary lemmas and develops the main argument, but stops before reaching the final conclusion or verifying the result applies to all cases.
→ CORRECT grade: "partial" (missing the conclusion - a MAJOR component)

**Example 5 - "almost" (NOT "partial") - Complete with typo:**
Student writes a complete proof with all steps correct, but has a typo in the final answer (e.g., writes "n^2" instead of "n^3" in the conclusion, but the proof logic is correct).
→ CORRECT grade: "almost" (complete structure, only a typo - trivial to fix)

**Example 6 - "partial" (NOT "incorrect") - Good progress:**
Student correctly identifies the approach, proves one of two required lemmas, but doesn't address the second lemma or connect to the main result.
→ CORRECT grade: "partial" (significant progress on the approach, but incomplete structure)

**Example 7 - "incorrect" (NOT "partial") - No progress:**
Student writes some calculations that are mathematically invalid, uses wrong theorems, or makes claims without any justification.
→ CORRECT grade: "incorrect" (no meaningful progress)

**Example 8 - "almost" vs "partial" - The critical difference:**
Student A: Proves the key lemma, sets up the main proof, completes 90% of the argument, but has a small gap in the final step that could be fixed with a sentence.
→ "almost" (complete structure, minor gap)

Student B: Proves the key lemma, sets up the main proof, but only completes 50% of the main argument with major gaps remaining that would require several paragraphs to fill.
→ "partial" (incomplete structure, substantial work needed)

**Example 9 - "almost" with complete number theory proof:**
Student writes a complete number theory proof: states the approach, proves all necessary lemmas, addresses all cases, reaches the conclusion. Only issue: a small arithmetic error in case 2 that doesn't affect the logic.
→ CORRECT grade: "almost" (complete structure, minor error only)

**Example 10 - "partial" with incomplete geometry proof:**
Student correctly identifies the geometric approach, proves one auxiliary lemma, but doesn't complete the main argument or connect the lemma to the final result.
→ CORRECT grade: "partial" (significant progress but incomplete structure)

**Example 11 - "almost" with minor logical gap:**
Student has a complete proof but omits a trivial verification step (e.g., "it's easy to see that X > 0" when X is clearly positive from context).
→ CORRECT grade: "almost" (complete structure, minor omission)

**Example 12 - "partial" with missing case:**
Student proves the result for most cases but completely omits one case that requires different reasoning.
→ CORRECT grade: "partial" (missing a major component - the omitted case)

**Example 13 - "almost" with computational error:**
Student has a complete induction proof but makes a small error in the inductive step calculation (e.g., 2^(k+1) = 2^k + 2 instead of 2*2^k).
→ CORRECT grade: "almost" (complete structure, minor computational error)

**Example 14 - "partial" with incomplete induction:**
Student sets up the induction base case and states the inductive hypothesis, but doesn't complete the inductive step.
→ CORRECT grade: "partial" (incomplete proof structure)

**Example 15 - "almost" with notation issue:**
Student writes a complete and correct proof but uses inconsistent notation (e.g., switches between n and N for the same variable).
→ CORRECT grade: "almost" (complete structure, minor notation issue)

**Example 16 - "partial" vs "almost" - Combinatorics problem:**
Student A: Sets up the counting argument, considers all necessary cases, reaches the answer, but makes a small error in one case count.
→ "almost" (complete structure, minor error)

Student B: Sets up the counting argument, considers some cases, but misses several cases entirely.
→ "partial" (incomplete - missing major components)

**Example 17 - "almost" with algebraic manipulation error:**
Student has a complete algebraic proof but makes a sign error when expanding (a-b)^2.
→ CORRECT grade: "almost" (complete structure, minor algebraic error)

**Example 18 - "partial" with wrong direction:**
Student starts a proof by contradiction but doesn't reach a contradiction or connect the assumption to the conclusion.
→ CORRECT grade: "partial" (incomplete argument structure)

**Example 19 - "almost" with inequality error:**
Student proves an inequality with correct structure but makes a small error in applying AM-GM (e.g., wrong weights).
→ CORRECT grade: "almost" (complete structure, minor technique error)

**Example 20 - "partial" with only setup:**
Student restates the problem, defines variables correctly, but doesn't make progress on the actual proof.
→ CORRECT grade: "partial" (some setup but no substantial progress)

**Example 21 - "almost" with complete proof but wrong final number:**
Student writes a complete, rigorous proof of a number theory problem. All logic is correct, all cases covered, but in the very last line writes "Therefore n = 5" when the correct answer is n = 7.
→ CORRECT grade: "almost" (complete proof structure, only the final number is wrong - a trivial error)

**Example 22 - "partial" with missing final verification:**
Student proves all necessary lemmas and develops the argument, but doesn't verify that the result satisfies the original problem conditions.
→ CORRECT grade: "partial" (missing the final verification step - a major component)

**Example 23 - "almost" with complete functional equation proof:**
Student solves a functional equation problem with complete proof: correctly guesses the solution, proves it's the only solution, verifies it works. Only issue: a small algebra error when substituting that doesn't affect the conclusion.
→ CORRECT grade: "almost" (complete structure, minor algebra error)

**Example 24 - "partial" with incomplete functional equation:**
Student correctly guesses the solution to a functional equation and verifies it works, but doesn't prove it's the only solution.
→ CORRECT grade: "partial" (incomplete - missing the uniqueness proof)

**Example 25 - "almost" with geometry diagram error:**
Student writes a complete geometry proof with correct logic, but mislabels one angle in the diagram (e.g., labels angle A as 30° when it should be 40°, but this doesn't affect the proof logic).
→ CORRECT grade: "almost" (complete structure, minor diagram labeling error)

**Example 26 - "partial" with missing key step:**
Student has a good approach and proves some lemmas, but skips a crucial step in the main argument that would require a paragraph to justify.
→ CORRECT grade: "partial" (missing a major proof component)

**Example 27 - "almost" with complete inequality chain:**
Student proves an inequality with a complete chain of reasoning, but makes a small error in one AM-GM application (uses wrong equality condition but the inequality direction is still correct).
→ CORRECT grade: "almost" (complete structure, minor technique error)

**Example 28 - "partial" with only base case:**
Student proves the base case of an induction problem and states the inductive hypothesis, but provides no argument for the inductive step.
→ CORRECT grade: "partial" (incomplete proof structure)

**Example 29 - "almost" with complete induction but small error:**
Student completes the entire induction proof including base case, inductive hypothesis, and inductive step. Only issue: a small calculation error in the algebra of the inductive step that doesn't invalidate the approach.
→ CORRECT grade: "almost" (complete structure, minor calculation error)

**Example 30 - "partial" vs "almost" - Polynomial problem:**
Student A: Finds all roots of the polynomial, verifies they work, but makes a small sign error when writing the final factorization.
→ "almost" (complete work, minor error in final presentation)

Student B: Finds some roots but misses others, or doesn't verify the found roots actually satisfy the equation.
→ "partial" (incomplete - missing major components)

## Common Mistakes to Avoid (READ CAREFULLY)

1. **DON'T** grade "partial" when the student has a complete proof with only minor errors → This should be "almost"
2. **DON'T** grade "almost" when the solution has major logical gaps or missing proof steps → This should be "partial"
3. **DON'T** ignore the specific criteria in the grading guidelines - they take precedence
4. **DON'T** grade based on your own judgment alone - follow the guidelines criteria
5. **DON'T** be too stingy with "almost" - if the student has a complete proof structure with minor issues, it's "almost" not "partial"
6. **DON'T** be too generous with "partial" - "partial" requires meaningful progress, not just "attempted"
7. **DON'T** grade "partial" when the student only made minimal progress → This should be "incorrect"
8. **DON'T** grade "almost" when the student is missing the conclusion or major proof sections → This should be "partial"
9. **DON'T** confuse "found a lemma" (partial) with "proved the lemma and completed the proof" (almost)
10. **DON'T** grade based on effort - grade based on the actual mathematical content
11. **DON'T** be biased against "almost" - many solutions that look "partial" are actually "almost" because they have complete structure
12. **DON'T** require perfection for "almost" - minor errors are expected and acceptable

## Final Verification Checklist (COMPLETE ALL CHECKS BEFORE SUBMITTING!)

Before submitting your grade, you MUST verify:
- [ ] **COMPLETE STRUCTURE TEST**: Does the student have a complete proof structure (beginning, middle, end)?
- [ ] **FIXABILITY TEST**: Can I fix this by correcting only trivial errors (arithmetic, typos)?
- [ ] **ALL COMPONENTS TEST**: Are ALL major components present (not just some)?
- [ ] **GUIDELINES MATCH**: Does my grade match the specific criteria in the guidelines?
- [ ] **BIAS CHECK**: Am I being too strict with "almost" or too generous with "partial"?
- [ ] **CONCLUSION CHECK**: Did the student reach the conclusion (for "almost") or stop early (for "partial")?
- [ ] **ERROR SEVERITY CHECK**: Are the errors truly minor (almost) or major (partial)?
- [ ] **COMPLETENESS CHECK**: Is the proof complete from start to finish?

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
            
            # Strategy 5: Look for grade in the last 500 chars (conclusion area)
            # This is often where the final decision is stated
            # PRIORITIZE "almost" detection since it's most commonly missed
            if prediction not in VALID_GRADES:
                conclusion = last_message[-500:].lower()
                # Check "almost" FIRST since it's most commonly missed and critical
                for grade in ["almost", "correct", "partial", "incorrect"]:
                    if re.search(rf'\b{grade}\b', conclusion):
                        prediction = grade
                        logger.debug(f"_extract_grade_from_response: Found grade '{prediction}' in conclusion (last 500 chars)")
                        break
            
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
