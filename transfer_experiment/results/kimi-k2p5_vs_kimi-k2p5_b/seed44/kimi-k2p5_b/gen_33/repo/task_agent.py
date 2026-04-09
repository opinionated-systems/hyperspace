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
    
    # Handle very long text - truncate to avoid performance issues
    # but keep the end portion where the grade is most likely to be
    max_text_length = 50000
    if len(text) > max_text_length:
        logger.debug(f"_extract_jsons: Text too long ({len(text)} chars), truncating to last {max_text_length}")
        text = text[-max_text_length:]
        
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
        # Start from the end of the text where the grade is most likely to be
        i = len(text) - 1
        found_positions = set()
        
        while i >= 0:
            if text[i] == '}':
                # Try to find the matching opening brace by going backwards
                brace_count = 1
                j = i - 1
                while j >= 0 and brace_count > 0:
                    if text[j] == '}':
                        brace_count += 1
                    elif text[j] == '{':
                        brace_count -= 1
                    j -= 1

                if brace_count == 0:
                    json_str = text[j+1:i+1].strip()
                    # Skip if we've already found this position
                    if j+1 in found_positions:
                        i = j
                        continue
                    
                    # Try parsing with safe_json_loads first
                    parsed = _safe_json_loads(json_str)
                    if parsed and isinstance(parsed, dict):
                        results.append(parsed)
                        found_positions.add(j+1)
                        logger.debug(f"_extract_jsons: Parsed raw JSON object: {list(parsed.keys())}")
                    else:
                        try:
                            parsed = json.loads(json_str)
                            if isinstance(parsed, dict):
                                results.append(parsed)
                                found_positions.add(j+1)
                                logger.debug(f"_extract_jsons: Parsed raw JSON object via json.loads: {list(parsed.keys())}")
                        except json.JSONDecodeError:
                            pass
                i = j
            else:
                i -= 1
        
        # Reverse results so they're in order of appearance
        results.reverse()

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
    - Unicode escape issues
    - Missing quotes around keys
    - Unescaped quotes in strings
    
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
    
    # Try to fix unescaped newlines in strings
    try:
        # Replace newlines within string values with escaped newlines
        # This is a more aggressive fix
        fixed_newlines = re.sub(r'(?<=")([^"]*)\n([^"]*)"', lambda m: '"' + m.group(1).replace('\n', '\\n') + m.group(2).replace('\n', '\\n') + '"', cleaned)
        result = json.loads(fixed_newlines)
        logger.debug(f"_safe_json_loads: Parsing succeeded after fixing newlines")
        return result
    except (json.JSONDecodeError, re.error) as e:
        logger.debug(f"_safe_json_loads: Newline fix failed: {e}")
    
    # Try to fix missing quotes around keys (common LLM issue)
    try:
        # Add quotes around unquoted keys
        # Pattern: match word characters followed by colon at start of object or after comma
        fixed_keys = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', cleaned)
        result = json.loads(fixed_keys)
        logger.debug(f"_safe_json_loads: Parsing succeeded after fixing unquoted keys")
        return result
    except json.JSONDecodeError as e:
        logger.debug(f"_safe_json_loads: Unquoted key fix failed: {e}")
    
    # Try to handle grade-only responses (e.g., just "correct" or {"grade": "correct"})
    try:
        # Check if it's just a grade word
        text_lower = cleaned.lower().strip()
        for grade in VALID_GRADES:
            if text_lower == grade or text_lower == f'"{grade}"':
                return {"grade": grade}
    except Exception as e:
        logger.debug(f"_safe_json_loads: Grade-only check failed: {e}")
    
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
    
    # Handle very long text - focus on the end where the grade is most likely
    max_text_length = 10000
    if len(text) > max_text_length:
        text = text[-max_text_length:]
        
    text_lower = text.lower()
    
    # Priority 1: Look for JSON grade field patterns (most reliable)
    # Check "grade" field first specifically - find ALL occurrences and take the LAST one
    # (last is most likely to be the final answer)
    grade_field_patterns = [
        (r'"grade"\s*:\s*"([^"]*)"', 1),
        (r'"grade"\s*:\s*\'([^\']*)\'', 1),
        (r'"grade"\s*:\s*([a-zA-Z]+)\b', 1),
        (r"'grade'\s*:\s*'([^']*)'", 1),
        (r"'grade'\s*:\s*\"([^\"]*)\"", 1),
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
        (r'"answer"\s*:\s*"([^"]*)"', 1),
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
        r'\bgrade\s+is\s*[:=]\s*"?(correct|incorrect|partial|almost)"?\b',
        r'\bassigned\s*[:=]\s*"?(correct|incorrect|partial|almost)"?\b',
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
    
    # Priority 8: Check for "partial" anywhere in the text
    if 'partial' in text_lower:
        if re.search(r'\bpartial\b', text_lower):
            logger.debug("_extract_grade_from_text: Found grade 'partial' via full text search")
            return 'partial'
    
    # Priority 9: Check for "incorrect" anywhere in the text
    if 'incorrect' in text_lower:
        if re.search(r'\bincorrect\b', text_lower):
            logger.debug("_extract_grade_from_text: Found grade 'incorrect' via full text search")
            return 'incorrect'
    
    # Priority 10: Check for "correct" anywhere in the text
    if 'correct' in text_lower:
        if re.search(r'\bcorrect\b', text_lower):
            logger.debug("_extract_grade_from_text: Found grade 'correct' via full text search")
            return 'correct'
    
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
    # CRITICAL: Check "almost" first since it has the lowest precision/recall
    variations = {
        "almost": [
            "almost correct", "nearly correct", "mostly correct", "close", 
            "minor errors", "small errors", "nearly complete", "almost there",
            "very close", "nearly right", "mostly right", "almost done",
            "nearly done", "just missed", "minor mistake", "small mistake",
            "trivial error", "small error", "minor issue", "small issue",
            "nearly there", "mostly complete", "essentially correct", "practically correct",
            "minor blemish", "small blemish", "tiny error", "slight error",
            "nearly solved", "almost solved", "virtually correct", "substantially correct",
            "complete with minor errors", "complete with small errors", "correct except",
            "one small error", "one minor error", "few minor errors", "few small errors",
            "arithmetic error", "computation error", "calculation error", "typo",
            "notational error", "notation error", "sign error", "coefficient error",
            "minor gap", "small gap", "trivial gap", "insignificant error",
            "negligible error", "cosmetic error", "formatting error", "presentation error",
            # Additional variations for better "almost" detection
            "complete proof with error", "complete proof with minor error",
            "complete solution with error", "complete solution with minor error",
            "finished with minor error", "finished with small error",
            "reached conclusion with error", "reached answer with minor error",
            "minor slip", "small slip", "trivial mistake", "minor oversight",
            "small oversight", "negligible mistake", "insignificant mistake",
            "minor flaw", "small flaw", "trivial flaw", "cosmetic flaw",
            "complete except", "finished except", "done except",
            "minor miscalculation", "small miscalculation", "trivial miscalculation",
            "minor arithmetic slip", "small arithmetic slip", "calculation slip",
            "transcription error", "copying error", "writing error",
            "complete structure", "full structure", "entire proof",
            "all steps present", "all components addressed",
            "minor technical error", "small technical error", "technical slip",
            "proof complete with", "solution complete with",
            "essentially complete", "practically complete", "virtually complete",
            "complete but", "finished but", "done but",
            "only minor", "only small", "just a minor", "just a small",
            "single error", "single mistake", "one error", "one mistake",
            "isolated error", "isolated mistake", "minor isolated",
            "trivial issue", "small issue", "minor issue only",
            "complete work", "full work", "entire work",
            "all parts addressed", "all cases covered", "all steps correct",
            "minor correction needed", "small correction needed", "trivial correction",
            "almost perfect", "nearly perfect", "virtually perfect",
            "minor improvement needed", "small improvement needed",
            "complete reasoning", "full reasoning", "sound reasoning",
            "valid approach", "correct approach", "right approach",
            "minor error only", "small error only", "trivial error only",
            "complete with blemish", "complete with minor blemish",
            "finished with blemish", "finished with minor blemish",
            "substantially complete", "largely complete", "mostly complete",
            "complete proof structure", "complete solution structure",
            "all major components", "all key steps", "all essential parts",
            "minor gap only", "small gap only", "trivial gap only",
            "complete argument", "full argument", "entire argument",
            "minor adjustment needed", "small adjustment needed",
            "complete modulo", "complete up to", "complete except for",
            "minor detail error", "small detail error", "trivial detail error",
            "complete derivation", "full derivation", "entire derivation",
            "minor step error", "small step error", "single step error",
            "complete analysis", "full analysis", "entire analysis",
            "minor case error", "small case error", "trivial case error",
            "complete induction", "full induction", "entire induction",
            "minor base case error", "small base case error",
            "complete proof by contradiction", "full proof by contradiction",
            "minor contradiction error", "small contradiction error",
            "complete construction", "full construction", "entire construction",
            "minor construction error", "small construction error",
            "complete case analysis", "full case analysis", "entire case analysis",
            "minor case oversight", "small case oversight",
            "complete pigeonhole", "full pigeonhole", "entire pigeonhole",
            "minor pigeonhole error", "small pigeonhole error",
            "complete invariant", "full invariant", "entire invariant",
            "minor invariant error", "small invariant error",
            "complete recurrence", "full recurrence", "entire recurrence",
            "minor recurrence error", "small recurrence error",
            "complete graph proof", "full graph proof", "entire graph proof",
            "minor graph error", "small graph error",
            "complete number theory", "full number theory", "entire number theory",
            "minor nt error", "small nt error", "number theory slip",
            "complete algebra", "full algebra", "entire algebra",
            "minor algebra error", "small algebra error", "algebra slip",
            "complete geometry", "full geometry", "entire geometry",
            "minor geometry error", "small geometry error", "geometry slip",
            "complete combinatorics", "full combinatorics", "entire combinatorics",
            "minor combinatorics error", "small combinatorics error",
            "complete sequence", "full sequence", "entire sequence",
            "minor sequence error", "small sequence error",
            "complete functional equation", "full functional equation",
            "minor fe error", "small fe error", "functional equation slip",
            "complete inequality", "full inequality", "entire inequality",
            "minor inequality error", "small inequality error",
            "complete polynomial", "full polynomial", "entire polynomial",
            "minor polynomial error", "small polynomial error",
            "complete modular arithmetic", "full modular arithmetic",
            "minor modular error", "small modular error", "modular slip",
            "complete diophantine", "full diophantine", "entire diophantine",
            "minor diophantine error", "small diophantine error",
        ],
        "partial": [
            "partially correct", "some progress", "incomplete", "partial credit", 
            "significant progress", "halfway", "partial solution", "in progress",
            "not complete", "unfinished", "partial result", "some credit",
            "meaningful progress", "substantial progress", "good progress",
            "incomplete proof", "partial proof", "missing parts", "missing components",
            "not finished", "incomplete solution", "partially solved",
            "missing conclusion", "missing final step", "incomplete argument",
            "missing cases", "missing lemma", "missing step", "incomplete work",
            # Additional variations for better "partial" detection
            "incomplete structure", "unfinished proof", "unfinished solution",
            "partial argument", "partial reasoning", "partial analysis",
            "some parts complete", "some steps correct", "some progress made",
            "meaningful work", "substantive progress", "real progress",
            "valid partial solution", "valid incomplete solution",
            "partially valid", "partially right", "partially correct work",
            "incomplete but valid", "unfinished but correct approach",
            "missing key step", "missing crucial step", "missing main step",
            "missing final answer", "missing result", "missing solution",
            "stopped early", "stopped before end", "did not finish",
            "incomplete case analysis", "partial case analysis",
            "missing some cases", "some cases only", "partial cases",
            "incomplete induction", "partial induction",
            "base case only", "inductive step only", "partial inductive proof",
            "incomplete lemma", "partial lemma", "lemma stated not proved",
            "incomplete construction", "partial construction",
            "construction started", "construction incomplete",
            "incomplete invariant", "partial invariant",
            "invariant identified", "invariant not fully used",
            "incomplete recurrence", "partial recurrence",
            "recurrence set up", "recurrence not solved",
            "incomplete graph argument", "partial graph proof",
            "some graph properties", "graph analysis incomplete",
            "incomplete number theory", "partial number theory",
            "some nt progress", "nt approach started",
            "incomplete algebra", "partial algebra",
            "algebraic manipulation", "algebra incomplete",
            "incomplete geometry", "partial geometry",
            "geometric insight", "geometry incomplete",
            "incomplete combinatorics", "partial combinatorics",
            "counting argument started", "combinatorics incomplete",
            "incomplete sequence", "partial sequence",
            "sequence analysis started", "sequence incomplete",
            "incomplete functional equation", "partial functional equation",
            "fe substitution made", "fe analysis incomplete",
            "incomplete inequality", "partial inequality",
            "inequality setup", "inequality not proved",
            "incomplete polynomial", "partial polynomial",
            "polynomial analysis", "polynomial incomplete",
            "incomplete modular arithmetic", "partial modular arithmetic",
            "modular setup", "modular analysis incomplete",
            "incomplete diophantine", "partial diophantine",
            "diophantine approach", "diophantine incomplete",
            "valid approach", "correct approach started", "right direction",
            "good start", "promising start", "on the right track",
            "key insight", "important observation", "useful lemma found",
            "partial result obtained", "intermediate result", "partial answer",
            "some cases solved", "some values found", "partial characterization",
            "necessary condition", "sufficient condition not proved",
            "lower bound proved", "upper bound not proved", "partial bound",
            "existence proved", "uniqueness not proved", "partial existence",
            "construction attempted", "construction partial",
            "algorithm described", "algorithm not fully analyzed",
            "proof structure started", "proof outline", "proof sketch",
            "main idea correct", "key step identified", "crucial observation made",
            "significant work done", "substantial work", "real mathematical work",
            "not just restating", "beyond problem statement", "genuine attempt",
            "valid mathematical reasoning", "logical progress made",
            "partial success", "partial achievement", "some credit deserved",
            "deserves partial credit", "warrants partial credit",
            "incomplete but meaningful", "unfinished but substantial",
            "partial proof valid", "partial solution valid",
            "correct partial work", "valid partial argument",
        ],
        "incorrect": [
            "wrong", "error", "false", "no credit", "zero", "no progress",
            "completely wrong", "totally wrong", "fundamentally wrong",
            "no meaningful progress", "invalid", "incorrect solution",
            "wrong approach", "no valid approach", "fundamental error",
            "logical error", "no attempt", "blank", "no solution",
            "failed", "unsuccessful", "invalid proof", "erroneous",
            # Additional variations for better "incorrect" detection
            "fundamentally incorrect", "completely incorrect", "totally incorrect",
            "no valid solution", "invalid solution", "nonsensical",
            "no mathematical content", "no mathematical reasoning",
            "just restated problem", "only restated problem", "mere restatement",
            "no work shown", "empty answer", "blank submission",
            "random guessing", "random work", "nonsense work",
            "irrelevant work", "unrelated work", "off topic",
            "mathematically invalid", "logically invalid", "contradictory",
            "self-contradictory", "contains contradiction", "logically flawed",
            "fundamentally flawed", "seriously flawed", "deeply flawed",
            "no understanding shown", "no comprehension", "misunderstood problem",
            "wrong interpretation", "misinterpreted problem", "wrong problem",
            "irrelevant calculations", "pointless calculations", "wasted effort",
            "no valid reasoning", "fallacious reasoning", "spurious reasoning",
            "bogus proof", "fake proof", "invalid argument",
            "non sequitur", "does not follow", "unwarranted conclusion",
            "unjustified claim", "baseless claim", "unfounded assertion",
            "mathematical nonsense", "mathematical gibberish", "incoherent",
            "incomprehensible", "garbled", "confused", "confusing",
            "circular reasoning", "circular argument", "begging the question",
            "false premise", "incorrect assumption", "unwarranted assumption",
            "invalid deduction", "invalid inference", "faulty logic",
            "no logical connection", "missing logical link", "logical gap",
            "severe logical gap", "critical logical gap", "fatal logical gap",
            "wrong theorem used", "misapplied theorem", "theorem misused",
            "incorrect formula", "wrong formula", "formula misapplied",
            "calculation nonsense", "arithmetic nonsense", "algebraic nonsense",
            "no valid method", "invalid method", "inappropriate method",
            "method doesn't apply", "method cannot work", "doomed approach",
            "fundamental misunderstanding", "basic misunderstanding",
            "conceptual error", "conceptual misunderstanding",
            "no valid proof technique", "wrong proof technique",
            "inappropriate technique", "technique misapplied",
            "no progress toward solution", "no headway", "stuck at start",
            "failed attempt", "abortive attempt", "unsuccessful attempt",
            "completely unsuccessful", "totally unsuccessful", "utterly failed",
            "no credit deserved", "deserves no credit", "warrants no credit",
            "zero credit", "no marks", "no points",
            "irrelevant", "immaterial", "inapplicable",
            "not applicable", "does not apply", "cannot apply",
            "meaningless", "without meaning", "lacks meaning",
            "no substance", "lacks substance", "empty of content",
            "trivial in bad way", "vacuous", "vacuously true but useless",
            "not even wrong", "worse than wrong", "meaningless symbols",
            "symbol manipulation without meaning", "form without content",
        ],
        "correct": [
            "right", "true", "full credit", "complete", "fully correct",
            "perfect", "flawless", "exactly right", "valid proof",
            "correct solution", "complete proof", "rigorous proof",
            "sound proof", "valid solution", "correct answer",
            # Additional variations for better "correct" detection
            "completely correct", "totally correct", "entirely correct",
            "absolutely correct", "perfectly correct", "100% correct",
            "fully rigorous", "completely rigorous", "totally rigorous",
            "mathematically correct", "logically correct", "strictly correct",
            "correct and complete", "complete and correct", "correct complete proof",
            "valid complete proof", "sound complete argument",
            "correct solution complete", "complete correct solution",
            "perfect solution", "ideal solution", "exemplary solution",
            "no errors", "no mistakes", "no flaws", "no gaps",
            "error-free", "mistake-free", "flawless", "gap-free",
            "fully justified", "completely justified", "all steps justified",
            "rigorous", "airtight", "watertight", "bulletproof",
            "correct proof", "valid complete proof", "sound complete proof",
            "correct answer derived", "correct result", "correct conclusion",
            "correct final answer", "correctly solved", "correctly proved",
            "solution verified", "proof verified", "result verified",
            "correct approach", "valid approach", "sound approach",
            "correct method", "valid method", "appropriate method",
            "correct technique", "valid technique", "proper technique",
            "all cases correct", "all steps correct", "all parts correct",
            "correct case analysis", "correct induction", "correct contradiction",
            "correct construction", "correct invariant", "correct recurrence",
            "correct graph argument", "correct number theory", "correct algebra",
            "correct geometry", "correct combinatorics", "correct sequence",
            "correct functional equation", "correct inequality",
            "correct polynomial", "correct modular arithmetic", "correct diophantine",
            "proof complete", "solution complete", "argument complete",
            "work complete", "reasoning complete", "analysis complete",
            "fully worked", "completely worked", "thoroughly worked",
            "correctly reasoned", "logically sound", "mathematically sound",
            "correct derivation", "correct computation", "correct calculation",
            "correctly derived", "correctly computed", "correctly calculated",
            "exact answer", "precise answer", "accurate answer",
            "correct characterization", "complete characterization",
            "correct classification", "complete classification",
            "correct formula", "correct equation", "correct identity",
            "correctly established", "correctly shown", "correctly proved",
            "correctly demonstrated", "correctly verified", "correctly checked",
            "proof valid", "argument valid", "reasoning valid",
            "solution valid", "answer valid", "result valid",
            "mathematically rigorous", "logically rigorous", "fully rigorous",
            "complete rigorous proof", "full rigorous proof", "entire rigorous proof",
            "no logical gaps", "no missing steps", "no unjustified claims",
            "every step correct", "every step valid", "every step justified",
            "all reasoning correct", "all logic correct", "all mathematics correct",
        ],
    }
    
    for standard, variants in variations.items():
        if grade_lower in variants:
            return standard
    
    # Partial substring matches as fallback (check longer words first to avoid confusion)
    # CRITICAL: Check "almost" first since it has the lowest precision/recall
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
        "confidence_score", "certainty_level", "confidence_value"
    ]
    
    for field in confidence_fields:
        if field in json_obj:
            try:
                val = str(json_obj[field]).lower().strip().strip('"').strip("'")
                if val in ["high", "medium", "low"]:
                    return val
                if val in ["very high", "extremely high", "certain"]:
                    return "high"
                if val in ["somewhat", "moderate", "fair"]:
                    return "medium"
                if val in ["very low", "uncertain", "unsure"]:
                    return "low"
                # Try to parse numeric confidence
                try:
                    num_val = float(val)
                    if num_val >= 0.8 or num_val >= 80:  # Handle percentage (80%)
                        return "high"
                    elif num_val >= 0.5 or num_val >= 50:  # Handle percentage (50%)
                        return "medium"
                    else:
                        return "low"
                except (ValueError, TypeError):
                    pass
            except Exception as e:
                logger.debug(f"_extract_confidence: Error processing field {field}: {e}")
                continue
    
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

# ============================================================
# GRADE DEFINITIONS - READ CAREFULLY
# ============================================================

## "correct" - FLAWLESS SOLUTION
- Complete and fully correct solution
- All steps present and logically sound
- No gaps, no errors, fully rigorous proof
- EVERY step must be correct and justified

## "incorrect" - NO MEANINGFUL PROGRESS
- No meaningful progress or fundamental errors
- Wrong approach or no valid mathematical progress
- Student failed to demonstrate understanding of key concepts
- Minimal or no substantive work toward the solution
- Just restating the problem is NOT progress

## "partial" - INCOMPLETE (Significant progress but NOT FINISHED)
- Significant progress but INCOMPLETE
- Found useful invariant/lemma or established valid approach but didn't finish
- Student demonstrated understanding of key ideas but solution has MAJOR gaps
- STOPS before completion or missing the conclusion
- The student has NOT addressed all major components
- Missing conclusion or final step is a KEY indicator
- Proof structure is incomplete (stops partway through)
- Needs substantial new work to be complete
- Has NOT reached the final answer/conclusion

## "almost" - COMPLETE WITH MINOR ISSUES (CRITICAL: This is NOT "partial"!)
- Nearly complete solution with ONLY minor issues
- Main proof structure is CORRECT and COMPLETE
- Only minor computational errors, typos, or notation issues
- Student has addressed ALL major components
- The proof is COMPLETE from start to finish INCLUDING the conclusion
- STUDENT REACHED THE CONCLUSION (even if with minor errors)
- If you fix trivial errors (arithmetic, typos), it becomes correct
- The student understood the complete approach and executed it with only small blemishes
- **KEY DISTINCTION**: "almost" = COMPLETE proof with minor errors, "partial" = INCOMPLETE proof

# ============================================================
# THE MOST IMPORTANT DISTINCTION: "almost" vs "partial"
# ============================================================
# THIS IS WHERE 90% OF GRADING ERRORS OCCUR!
# 
# KEY INSIGHT: "almost" = COMPLETE proof with minor blemishes
#              "partial" = INCOMPLETE proof (missing major parts)
#
# Ask yourself: "Did the student finish the proof or stop halfway?"
# - Finished with minor errors -> "almost"
# - Stopped before finishing -> "partial"

## TEST 1: The "COMPLETE STRUCTURE" Test (MOST IMPORTANT)
Ask: "Does the student have a COMPLETE proof structure from start to finish?"
- YES (complete structure, minor blemishes only) -> "almost"
- NO (missing components, incomplete, stopped early) -> "partial"

## TEST 2: The "CONCLUSION" Test (CRITICAL - USE THIS!)
Ask: "Did the student REACH the final conclusion/answer?"
- YES (even with minor errors in the answer) -> Strong indicator for "almost"
- NO (stopped before the end, missing final step) -> Strong indicator for "partial"

## TEST 3: The "FIXABILITY" Test (Apply rigorously!)
Ask: "Can I fix this by correcting ONLY trivial errors?"
- YES (arithmetic, typos, notation only - no new proof steps needed) -> "almost"
- NO (need new proof steps, lemmas, cases, or major additions) -> "partial"

## TEST 4: The "WHAT'S MISSING" Test
Ask: "What would need to be added to make this correct?"
- Nothing major (just fix tiny errors) -> "almost"
- Something major (conclusion, cases, lemmas, proof steps) -> "partial"

## TEST 5: The "PROGRESS" Test (for partial vs incorrect)
Ask: "Did the student make MEANINGFUL progress?"
- YES (found lemma, proved intermediate result, established valid approach) -> "partial"
- NO (just restated problem, random calculations) -> "incorrect"

# ============================================================
# QUICK DECISION TREE (USE THIS!)
# ============================================================

START: Did student make meaningful progress?
├── NO -> "incorrect" (no meaningful progress)
└── YES -> Did student reach the conclusion?
    ├── NO -> "partial" (incomplete - stopped before end)
    └── YES -> Is the proof structure complete?
        ├── NO -> "partial" (incomplete structure)
        └── YES -> Are errors only minor (fixable)?
            ├── YES -> "almost" (complete with minor issues)
            └── NO -> "partial" (needs major fixes)

# ============================================================
# DETAILED EXAMPLES - STUDY THESE CAREFULLY!
# 70+ examples showing all grade boundaries
# ============================================================

## "ALMOST" EXAMPLES (Complete proofs with minor issues)

**Example A1 - "almost" (Complete induction with base case error):**
Student writes a complete induction proof: base case, inductive hypothesis, inductive step, conclusion. Only issue: in the base case, calculates 2^3 = 7 instead of 8, but the logic is correct.
-> Grade: "almost" (complete structure, only minor arithmetic error)

**Example A2 - "almost" (Complete proof with typo in final answer):**
Student writes a complete proof with all steps correct, but has a typo in the final answer (writes "n^2" instead of "n^3" in the conclusion, but the proof logic is correct).
-> Grade: "almost" (complete structure, only a typo)

**Example A3 - "almost" (Complete number theory with small error):**
Student writes a complete number theory proof: states approach, proves all lemmas, addresses all cases, reaches conclusion. Only issue: a small arithmetic error in case 2 that doesn't affect the logic.
-> Grade: "almost" (complete structure, minor error only)

**Example A4 - "almost" (Complete combinatorics with counting error):**
Student sets up the counting argument, considers all necessary cases, reaches the answer. Only issue: makes a small error in one case count (off by 1).
-> Grade: "almost" (complete structure, minor error)

**Example A5 - "almost" (Complete algebra with sign error):**
Student has a complete algebraic proof but makes a sign error when expanding (a-b)^2.
-> Grade: "almost" (complete structure, minor algebraic error)

**Example A6 - "almost" (Complete sequence proof with index error):**
Student proves a sequence result with complete structure: defines sequence, proves monotonicity, finds limit. Only issue: uses index n+1 where n should be used, but the logic is still clear.
-> Grade: "almost" (complete structure, minor index error)

**Example A7 - "almost" (Complete graph theory with edge count error):**
Student proves a graph theory result with complete structure. Only issue: miscounts edges by 1 in one subgraph, but this doesn't affect the main argument.
-> Grade: "almost" (complete structure, minor counting error)

**Example A8 - "almost" (Complete recurrence with boundary error):**
Student solves a recurrence relation with complete proof. Only issue: makes a small arithmetic error when applying one boundary condition.
-> Grade: "almost" (complete structure, minor arithmetic error)

**Example A9 - "almost" (Complete modular arithmetic with calculation slip):**
Student proves a modular arithmetic result with complete structure. Only issue: calculates 2^5 = 30 instead of 32 in one step, but the error is isolated and trivial.
-> Grade: "almost" (complete structure, minor calculation slip)

**Example A10 - "almost" (Complete optimization with derivative error):**
Student solves an optimization problem with complete proof. Only issue: computes derivative of x^3 as 2x^2 instead of 3x^2, but corrects it in the next step.
-> Grade: "almost" (complete structure, minor derivative error)

**Example A11 - "almost" (Complete proof with transposed digits):**
Student writes a complete, rigorous proof. All logic is correct. In the final answer, writes "The answer is 127" when the correct answer is 172. Clearly a typo, not a logic error.
-> Grade: "almost" (complete proof, trivial typo in final answer)

**Example A12 - "almost" (Complete existence proof with uniqueness gap):**
Student proves existence of a solution and makes a small error in the uniqueness argument. The structure is complete, just a minor gap in one step.
-> Grade: "almost" (complete structure, minor gap in uniqueness)

**Example A13 - "almost" (Complete construction with small error):**
Student constructs an explicit example with complete structure. Only issue: a small arithmetic error in verifying one property that doesn't affect the validity.
-> Grade: "almost" (complete structure, minor verification error)

**Example A14 - "almost" (Complete contradiction proof with small gap):**
Student uses proof by contradiction with complete structure. Only issue: a small logical gap in showing the contradiction is reached, but the gap is trivial to fill.
-> Grade: "almost" (complete structure, minor gap)

**Example A15 - "almost" (Complete pigeonhole with off-by-one):**
Student proves a pigeonhole principle result with complete structure. Only issue: uses n holes when it should be n+1, but the logic is otherwise correct.
-> Grade: "almost" (complete structure, minor off-by-one error)

**Example A16 - "almost" (Complete functional equation with domain error):**
Student solves a functional equation with complete proof structure. Only issue: forgets to check x=0 case, but the domain specification is the only issue.
-> Grade: "almost" (complete structure, minor domain error)

**Example A17 - "almost" (Complete geometry proof with diagram error):**
Student proves a geometry result with complete logical structure. Only issue: mislabels one angle in the diagram description, but the proof logic is correct.
-> Grade: "almost" (complete structure, minor diagram error)

**Example A18 - "almost" (Complete inequality with coefficient error):**
Student proves an inequality with complete structure. Only issue: writes 2ab instead of 3ab when expanding, but the inequality direction is correct.
-> Grade: "almost" (complete structure, minor coefficient error)

**Example A19 - "almost" (Complete proof with notation switch):**
Student writes a complete proof but switches notation mid-proof (uses n then m for same variable). The logic is correct, just inconsistent notation.
-> Grade: "almost" (complete structure, minor notation issue)

**Example A20 - "almost" (Complete divisibility proof with factor error):**
Student proves a divisibility result with complete structure. Only issue: writes "divisible by 4" when it should be "divisible by 8", but the proof logic is correct.
-> Grade: "almost" (complete structure, minor factor error)

**Example A21 - "almost" (Complete polynomial with coefficient error):**
Student solves a polynomial problem with complete proof. Only issue: writes coefficient as 5 instead of 6 in expansion, but the factorization logic is correct.
-> Grade: "almost" (complete structure, minor coefficient error)

**Example A22 - "almost" (Complete inequality with AM-GM error):**
Student proves an inequality using AM-GM with complete structure. Only issue: applies AM-GM to wrong number of terms in one step, but the approach is correct.
-> Grade: "almost" (complete structure, minor application error)

**Example A23 - "almost" (Complete geometry with angle measure error):**
Student proves a geometry result with complete structure. Only issue: states one angle is 60° when it should be 45°, but the proof logic is correct.
-> Grade: "almost" (complete structure, minor angle error)

**Example A24 - "almost" (Complete number theory with mod calculation error):**
Student proves a number theory result with complete structure. Only issue: calculates 17 mod 5 = 3 instead of 2, but the modular logic is correct.
-> Grade: "almost" (complete structure, minor mod error)

**Example A25 - "almost" (Complete combinatorics with binomial coefficient error):**
Student proves a combinatorics result with complete structure. Only issue: calculates C(5,2) = 11 instead of 10, but the counting logic is correct.
-> Grade: "almost" (complete structure, minor binomial error)

## "PARTIAL" EXAMPLES (Incomplete solutions with meaningful progress)

**Example P1 - "partial" (Missing conclusion):**
Student proves all necessary lemmas and develops the main argument, but stops before reaching the final conclusion.
-> Grade: "partial" (missing the conclusion - a MAJOR component)

**Example P2 - "partial" (Incomplete geometry):**
Student correctly identifies the geometric approach, proves one auxiliary lemma, but doesn't complete the main argument or connect the lemma to the final result.
-> Grade: "partial" (significant progress but incomplete structure)

**Example P3 - "partial" (Missing cases in combinatorics):**
Student sets up the counting argument, considers some cases, but misses several cases entirely.
-> Grade: "partial" (incomplete - missing major components)

**Example P4 - "partial" (Wrong direction):**
Student starts a proof by contradiction but doesn't reach a contradiction or connect the assumption to the conclusion.
-> Grade: "partial" (incomplete argument)

**Example P5 - "partial" (Missing sequence convergence):**
Student defines a sequence and proves some properties, but doesn't prove convergence or find the limit when that's the main goal.
-> Grade: "partial" (missing the main result)

**Example P6 - "partial" (Incomplete graph coloring):**
Student proves some properties about graph coloring but doesn't complete the argument about the chromatic number.
-> Grade: "partial" (incomplete - missing verification)

**Example P7 - "partial" (Recurrence setup only):**
Student correctly sets up the recurrence relation and finds the characteristic equation, but doesn't solve for the roots or find the particular solution.
-> Grade: "partial" (incomplete - missing the solution)

**Example P8 - "partial" (Modular setup but no conclusion):**
Student sets up the modular arithmetic framework correctly and proves some intermediate congruences, but doesn't connect them to reach the final result.
-> Grade: "partial" (incomplete - missing the conclusion)

**Example P9 - "partial" (Optimization setup only):**
Student defines the function to optimize and takes the derivative, but doesn't find critical points or analyze them.
-> Grade: "partial" (incomplete - missing the optimization)

**Example P10 - "partial" (Missing existence proof):**
Student proves uniqueness of a solution but doesn't prove that a solution exists at all.
-> Grade: "partial" (incomplete - missing existence)

**Example P11 - "partial" (Only uniqueness proof):**
Student proves that if a solution exists, it must be unique, but never proves that any solution exists.
-> Grade: "partial" (incomplete - missing existence proof)

**Example P12 - "partial" (Construction attempt but invalid):**
Student attempts to construct an example but the construction is invalid or doesn't satisfy the required properties, though the approach shows understanding.
-> Grade: "partial" (attempted construction but failed)

**Example P13 - "partial" (Contradiction setup but no contradiction):**
Student assumes the negation for proof by contradiction, develops some argument, but never actually reaches a contradiction.
-> Grade: "partial" (incomplete - no contradiction reached)

**Example P14 - "partial" (Pigeonhole setup but wrong application):**
Student identifies that pigeonhole principle should be used and defines pigeons/holes, but applies the principle incorrectly or doesn't complete the argument.
-> Grade: "partial" (incomplete application)

**Example P15 - "partial" (Missing final connections):**
Student proves all the necessary lemmas and has all the pieces, but doesn't connect them to reach the final conclusion.
-> Grade: "partial" (incomplete - missing final connections)

**Example P16 - "partial" (Incomplete induction):**
Student proves the base case and states the inductive hypothesis, but doesn't complete the inductive step.
-> Grade: "partial" (incomplete - missing inductive step)

**Example P17 - "partial" (Incomplete case analysis):**
Student analyzes some cases correctly but misses one or more cases entirely.
-> Grade: "partial" (incomplete - missing cases)

**Example P18 - "partial" (Recurrence setup only):**
Student sets up the recurrence correctly but doesn't solve it or verify the solution.
-> Grade: "partial" (incomplete - missing solution)

**Example P19 - "partial" (Only base case of induction):**
Student proves only the base case of an induction proof, missing the inductive step entirely.
-> Grade: "partial" (incomplete - only base case)

**Example P20 - "partial" (Missing existence proof):**
Student proves properties about a solution but never shows a solution actually exists.
-> Grade: "partial" (incomplete - missing existence)

**Example P21 - "partial" (Incomplete induction step):**
Student has correct base case and inductive hypothesis, but the inductive step is incomplete or has a major logical gap.
-> Grade: "partial" (incomplete inductive step)

**Example P22 - "partial" (Missing existence proof):**
Student proves uniqueness but doesn't prove existence, or vice versa.
-> Grade: "partial" (missing major component - existence or uniqueness)

**Example P23 - "partial" (Incomplete inequality chain):**
Student starts an inequality chain but breaks off before reaching the final bound.
-> Grade: "partial" (incomplete inequality proof)

**Example P24 - "partial" (Missing verification):**
Student constructs a candidate solution but doesn't verify it satisfies all conditions.
-> Grade: "partial" (missing verification step)

**Example P25 - "partial" (Incomplete case in number theory):**
Student handles most cases in a number theory problem but misses one or more critical cases (e.g., p=2 case).
-> Grade: "partial" (missing critical cases)

## "INCORRECT" EXAMPLES (No meaningful progress)

**Example I1 - "incorrect" (No progress):**
Student just restates the problem without any attempt at solution.
-> Grade: "incorrect" (no meaningful progress)

**Example I2 - "incorrect" (Random calculations):**
Student makes random calculations with no valid mathematical approach.
-> Grade: "incorrect" (no valid approach)

**Example I3 - "incorrect" (Wrong approach):**
Student uses a completely wrong approach that cannot lead to the solution.
-> Grade: "incorrect" (fundamentally wrong approach)

**Example I4 - "incorrect" (Fundamental misunderstanding):**
Student demonstrates a fundamental misunderstanding of the problem concepts.
-> Grade: "incorrect" (fundamental error)

**Example I5 - "incorrect" (No valid lemmas):**
Student claims lemmas that are false or irrelevant.
-> Grade: "incorrect" (no valid mathematical progress)

**Example I6 - "incorrect" (Trivial restatement):**
Student just rearranges the problem statement algebraically without progress.
-> Grade: "incorrect" (no meaningful progress)

**Example I7 - "incorrect" (Wrong theorem application):**
Student applies a theorem in a context where it doesn't apply.
-> Grade: "incorrect" (invalid theorem application)

**Example I8 - "incorrect" (Circular reasoning):**
Student uses circular reasoning where the conclusion is assumed in the proof.
-> Grade: "incorrect" (logical fallacy)

**Example I9 - "incorrect" (Computational only):**
Student only computes small cases without any general argument or proof structure.
-> Grade: "incorrect" (no general proof, just examples)

**Example I10 - "incorrect" (Misunderstood problem):**
Student solves a completely different problem than what was asked.
-> Grade: "incorrect" (wrong problem solved)

## "CORRECT" EXAMPLES (Flawless solutions)

**Example C1 - "correct" (Complete proof):**
Student writes a complete, rigorous proof with all steps justified, no errors, perfect logic.
-> Grade: "correct"

**Example C2 - "correct" (Flawless induction):**
Student writes a perfect induction proof with correct base case, inductive hypothesis, and inductive step.
-> Grade: "correct"

# ============================================================
# CRITICAL BOUNDARY CASES - PAY EXTRA ATTENTION!
# ============================================================

## "almost" vs "partial" - THE MOST COMMON ERROR

**CASE 1: Student has complete proof with wrong final number**
- Student writes complete proof, all logic correct, but final answer is 42 instead of 24
- This is "almost" NOT "partial" - the structure is complete, just a typo

**CASE 2: Student stops before conclusion**
- Student proves lemmas, sets up framework, but never reaches the conclusion
- This is "partial" NOT "almost" - incomplete structure

**CASE 3: Student has complete induction with base case error**
- Student has base case, inductive hypothesis, inductive step, conclusion
- Base case has arithmetic error: 2+3=6 instead of 5
- This is "almost" NOT "partial" - complete structure with minor error

**CASE 4: Student has only base case of induction**
- Student proves only the base case but nothing else
- This is "partial" NOT "almost" - incomplete structure

**CASE 5: Student has complete proof with notation switch**
- Student switches from n to m mid-proof for same variable
- Logic is correct, just notation inconsistency
- This is "almost" NOT "partial" - complete structure with minor issue

**CASE 6: Student has complete proof with missing domain check**
- Student solves functional equation but forgets to check x=0
- This is "almost" NOT "partial" - complete structure with minor omission

## "partial" vs "incorrect" - BE GENEROUS WITH PARTIAL

**CASE 7: Student considers prime p|xy+1**
- Guidelines say "Considered a prime p|xy+1" is partial credit
- Student mentions this and sets up some equations
- This is "partial" NOT "incorrect" - meaningful progress made

**CASE 8: Student just restates problem**
- Student just writes the problem statement in different words
- This is "incorrect" NOT "partial" - no meaningful progress

**CASE 9: Student found key lemma**
- Student identifies and states the key lemma needed
- This is "partial" NOT "incorrect" - meaningful progress

**CASE 10: Student has complete proof with sign error**
- Student writes complete proof but has (-3)^2 = -9 in one step
- This is "almost" NOT "partial" - complete structure with minor arithmetic error

**CASE 11: Student has complete proof with transposed digits**
- Student writes complete proof, final answer is 127 instead of 172
- This is "almost" NOT "partial" - complete structure with typo

**CASE 12: Student has complete proof with coefficient error**
- Student writes complete proof but uses coefficient 5 instead of 6 in expansion
- This is "almost" NOT "partial" - complete structure with minor error

**CASE 13: Student has complete proof with index shift**
- Student writes complete sequence proof but uses a_n+1 where a_n should be
- This is "almost" NOT "partial" - complete structure with minor index error

**CASE 14: Student has random calculations**
- Student performs calculations that don't lead anywhere
- This is "incorrect" NOT "partial" - no valid approach

**CASE 15: Student has wrong approach**
- Student uses an approach that can't solve the problem
- This is "incorrect" NOT "partial" - fundamentally wrong

**CASE 16: Student has setup but no proof**
- Student defines variables and states goal but makes no proof progress
- This is "partial" NOT "incorrect" - valid setup is progress

# ============================================================
# COMMON MISTAKES TO AVOID
# ============================================================

1. **DON'T** grade "partial" when the student has a complete proof with only minor errors -> This should be "almost"
2. **DON'T** grade "almost" when the solution has major logical gaps or missing proof steps -> This should be "partial"
3. **DON'T** be too stingy with "almost" - if the student has a complete proof structure with minor issues, it's "almost" not "partial"
4. **DON'T** be too generous with "partial" - "partial" requires meaningful progress, not just "attempted"
5. **DON'T** grade "partial" when the student only made minimal progress -> This should be "incorrect"
6. **DON'T** grade "almost" when the student is missing the conclusion -> This should be "partial"
7. **DON'T** grade "partial" when the student reached the conclusion with only minor errors -> This should be "almost"
8. **DON'T** be fooled by length - a long incomplete proof is still "partial", a short complete proof with minor errors is "almost"
9. **DON'T** grade "partial" when the student has a complete proof but wrong final number -> This should be "almost"
10. **DON'T** grade "almost" when the student has only proven a lemma but not connected it -> This should be "partial"
11. **DON'T** grade "partial" when the student has complete induction with only base case error -> This should be "almost"
12. **DON'T** grade "almost" when the student has only base case of induction -> This should be "partial"
13. **DON'T** be biased by the student's writing style - focus on mathematical content only
14. **DON'T** require perfection for "almost" - minor errors are expected and acceptable
15. **DON'T** confuse "found a lemma" (partial) with "proved the lemma and completed the proof" (almost)
16. **DON'T** grade "partial" when student has complete proof with sign error -> This should be "almost"
17. **DON'T** grade "almost" when student is missing major proof components -> This should be "partial"
18. **DON'T** grade "incorrect" when student made meaningful progress -> This should be "partial"
19. **DON'T** grade "partial" when student just restated problem -> This should be "incorrect"
20. **DON'T** be too strict - "almost" is for complete proofs with minor issues only
21. **DON'T** grade "partial" when student has complete proof with arithmetic error -> This should be "almost"
22. **DON'T** grade "almost" when student is missing the final step -> This should be "partial"
23. **DON'T** grade "incorrect" when student has valid setup -> This should be "partial"
24. **DON'T** grade "partial" when student has no valid approach -> This should be "incorrect"
25. **DON'T** be too generous - "partial" requires SUBSTANTIVE progress

# ============================================================
# ANTI-BIAS REMINDERS (IMPORTANT!)
# ============================================================

- Many graders are too STRICT with "almost" - don't be! If the structure is complete, it's "almost"
- A complete proof with a small error is MUCH closer to "correct" than to "partial"
- "Almost" means the student UNDERSTOOD the complete approach - reward this!
- When in doubt between "almost" and "partial", ask: "Did they reach the conclusion?" If YES -> "almost"
- Don't let minor imperfections push you toward "partial" - "almost" is for complete proofs with minor issues
- "Partial" is for INCOMPLETE proofs, not complete proofs with errors
- The default for complete proofs with minor errors should be "almost", not "partial"
- "Incorrect" is for NO MEANINGFUL PROGRESS, not for "attempted but failed"
- "Partial" requires SUBSTANTIVE progress - just "trying" is not enough
- When in doubt between "partial" and "incorrect", ask: "Did they find a valid lemma/approach?" If YES -> "partial"

# ============================================================
# FINAL VERIFICATION CHECKLIST
# ============================================================

Before submitting your grade, you MUST answer these questions explicitly:

## MANDATORY YES/NO QUESTIONS:

1. **COMPLETE STRUCTURE TEST**: Does the student have a complete proof structure from start to finish? (YES/NO)
2. **CONCLUSION TEST**: Did the student REACH the final conclusion/answer (even if with errors)? (YES/NO)
3. **FIXABILITY TEST**: Can I fix this by correcting ONLY trivial errors (arithmetic, typos, notation)? (YES/NO)
4. **COMPONENTS TEST**: Are ALL major components present and addressed? (YES/NO)
5. **ERROR SEVERITY TEST**: Are the errors truly MINOR (arithmetic, notation, typos) and not MAJOR (logical gaps, missing steps)? (YES/NO)
6. **PROGRESS TEST**: Did the student make MEANINGFUL progress (not just restate the problem)? (YES/NO)

## DECISION RULES (FOLLOW EXACTLY):

**For "almost":**
- Q1 (Complete Structure) MUST be YES
- Q2 (Conclusion) MUST be YES  
- Q3 (Fixability) MUST be YES
- Q4 (All Components) MUST be YES
- Q5 (Minor Errors) MUST be YES
- If ALL are YES -> Grade is "almost" (or "correct" if truly flawless)

**For "partial":**
- Q6 (Progress) MUST be YES
- AND at least one of Q1-Q4 is NO
- If Q6 is YES but proof is incomplete -> Grade is "partial"

**For "incorrect":**
- Q6 (Progress) is NO
- No meaningful mathematical progress made

**For "correct":**
- ALL of Q1-Q5 are YES
- AND there are truly NO errors (not even minor ones)

## FINAL ANTI-BIAS CHECK:
- Are you being too strict with "almost"? If the structure is complete, it should be "almost" not "partial"
- Are you being too generous with "partial"? If there's no meaningful progress, it should be "incorrect"
- Did you check if the student reached the conclusion? This is the key differentiator for "almost" vs "partial"

# ============================================================
# RESPONSE FORMAT (CRITICAL)
# ============================================================

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
                    rf'(?:i\s+)?(?:would\s+)?(?:say|call|label)\s+(?:this\s+)?["\']?({"|".join(VALID_GRADES)})["\']?',
                    rf'(?:the\s+)?(?:final\s+)?grade\s+(?:should\s+)?(?:be\s+)?["\']?({"|".join(VALID_GRADES)})["\']?',
                ]
                for pattern in reasoning_patterns:
                    match = re.search(pattern, last_message.lower())
                    if match:
                        prediction = match.group(1)
                        logger.debug(f"_extract_grade_from_response: Found grade '{prediction}' via reasoning pattern")
                        break
            
            # Strategy 8: Look for grade at the very end of the message (last resort)
            if prediction not in VALID_GRADES:
                # Check last 100 chars for any grade mention
                last_100 = last_message[-100:].lower()
                for grade in VALID_GRADES:
                    if re.search(rf'\b{grade}\b', last_100):
                        prediction = grade
                        logger.debug(f"_extract_grade_from_response: Found grade '{prediction}' at end of message")
                        break
            
            # Strategy 9: Check for grade in the last line specifically
            if prediction not in VALID_GRADES:
                lines = last_message.strip().split('\n')
                if lines:
                    last_line = lines[-1].lower()
                    for grade in VALID_GRADES:
                        if re.search(rf'\b{grade}\b', last_line):
                            prediction = grade
                            logger.debug(f"_extract_grade_from_response: Found grade '{prediction}' in last line")
                            break
            
            # Strategy 10: Try to find JSON-like patterns with regex
            if prediction not in VALID_GRADES:
                # Look for patterns like {"grade": "correct"} or {"grade": "almost"}
                json_grade_pattern = r'["\']?grade["\']?\s*:\s*["\']?(correct|incorrect|partial|almost)["\']?'
                matches = list(re.finditer(json_grade_pattern, last_message.lower()))
                if matches:
                    # Take the last match (most likely to be the final answer)
                    prediction = matches[-1].group(1)
                    logger.debug(f"_extract_grade_from_response: Found grade '{prediction}' via regex JSON pattern")
            
            # Strategy 11: Look for grade in the final decision/conclusion section
            # This is often where the LLM states its final answer after reasoning
            if prediction not in VALID_GRADES:
                # Look for common conclusion markers
                conclusion_markers = [
                    "final answer", "final grade", "my decision", "i conclude",
                    "therefore", "in conclusion", "to summarize", "in summary",
                    "the grade should be", "the correct grade is", "i determine"
                ]
                text_lower = last_message.lower()
                for marker in conclusion_markers:
                    idx = text_lower.rfind(marker)
                    if idx != -1:
                        # Check the text after the marker (up to 200 chars)
                        after_marker = text_lower[idx:idx+200]
                        # Check "almost" first since it's most commonly missed
                        if re.search(r'\balmost\b', after_marker):
                            prediction = "almost"
                            logger.debug(f"_extract_grade_from_response: Found grade 'almost' after marker '{marker}'")
                            break
                        elif re.search(r'\bcorrect\b', after_marker):
                            prediction = "correct"
                            logger.debug(f"_extract_grade_from_response: Found grade 'correct' after marker '{marker}'")
                            break
                        elif re.search(r'\bpartial\b', after_marker):
                            prediction = "partial"
                            logger.debug(f"_extract_grade_from_response: Found grade 'partial' after marker '{marker}'")
                            break
                        elif re.search(r'\bincorrect\b', after_marker):
                            prediction = "incorrect"
                            logger.debug(f"_extract_grade_from_response: Found grade 'incorrect' after marker '{marker}'")
                            break
                
            # Strategy 12: Check for grade in the last sentence/paragraph
            if prediction not in VALID_GRADES:
                # Split by common sentence delimiters and check the last few
                sentences = re.split(r'[.!?\n]+', last_message.lower())
                # Check last 3 sentences
                for sentence in reversed(sentences[-3:]):
                    sentence = sentence.strip()
                    if sentence:
                        # Check "almost" first
                        if re.search(r'\balmost\b', sentence):
                            prediction = "almost"
                            logger.debug(f"_extract_grade_from_response: Found grade 'almost' in last sentence")
                            break
                        elif re.search(r'\bcorrect\b', sentence):
                            prediction = "correct"
                            logger.debug(f"_extract_grade_from_response: Found grade 'correct' in last sentence")
                            break
                        elif re.search(r'\bpartial\b', sentence):
                            prediction = "partial"
                            logger.debug(f"_extract_grade_from_response: Found grade 'partial' in last sentence")
                            break
                        elif re.search(r'\bincorrect\b', sentence):
                            prediction = "incorrect"
                            logger.debug(f"_extract_grade_from_response: Found grade 'incorrect' in last sentence")
                            break
            
            # Strategy 13: Check for grade in code blocks or formatted sections
            if prediction not in VALID_GRADES:
                # Look for grades in backtick-quoted sections
                code_pattern = r'`(correct|incorrect|partial|almost)`'
                matches = list(re.finditer(code_pattern, last_message.lower()))
                if matches:
                    prediction = matches[-1].group(1)
                    logger.debug(f"_extract_grade_from_response: Found grade '{prediction}' in backticks")
            
            # Strategy 14: Check for grade in bold or emphasized text
            if prediction not in VALID_GRADES:
                # Look for **grade** or __grade__ patterns
                bold_patterns = [
                    (r'\*\*(correct|incorrect|partial|almost)\*\*', "bold"),
                    (r'__(correct|incorrect|partial|almost)__', "underline"),
                    (r'\*(correct|incorrect|partial|almost)\*', "italic"),
                    (r'_(correct|incorrect|partial|almost)_', "italic"),
                ]
                for pattern, style in bold_patterns:
                    match = re.search(pattern, last_message.lower())
                    if match:
                        prediction = match.group(1)
                        logger.debug(f"_extract_grade_from_response: Found grade '{prediction}' in {style} text")
                        break
            
            # Strategy 15: Check for grade in parentheses or brackets
            if prediction not in VALID_GRADES:
                bracket_patterns = [
                    rf'\((correct|incorrect|partial|almost)\)',
                    rf'\[(correct|incorrect|partial|almost)\]',
                    rf'\{{(correct|incorrect|partial|almost)\}}',
                ]
                for pattern in bracket_patterns:
                    matches = list(re.finditer(pattern, last_message.lower()))
                    if matches:
                        prediction = matches[-1].group(1)
                        logger.debug(f"_extract_grade_from_response: Found grade '{prediction}' in brackets")
                        break
            
            # Strategy 16: Check for grade after colons (common in lists)
            if prediction not in VALID_GRADES:
                colon_pattern = rf':\s*(correct|incorrect|partial|almost)\b'
                matches = list(re.finditer(colon_pattern, last_message.lower()))
                if matches:
                    prediction = matches[-1].group(1)
                    logger.debug(f"_extract_grade_from_response: Found grade '{prediction}' after colon")
            
            # Strategy 17: Check for grade in table cells or structured data
            if prediction not in VALID_GRADES:
                # Look for | grade | patterns (markdown tables)
                table_pattern = rf'\|\s*(correct|incorrect|partial|almost)\s*\|'
                matches = list(re.finditer(table_pattern, last_message.lower()))
                if matches:
                    prediction = matches[-1].group(1)
                    logger.debug(f"_extract_grade_from_response: Found grade '{prediction}' in table")
            
            # Strategy 18: Check for grade in HTML-like tags
            if prediction not in VALID_GRADES:
                html_patterns = [
                    rf'<grade>(correct|incorrect|partial|almost)</grade>',
                    rf'<b>(correct|incorrect|partial|almost)</b>',
                    rf'<strong>(correct|incorrect|partial|almost)</strong>',
                ]
                for pattern in html_patterns:
                    match = re.search(pattern, last_message.lower())
                    if match:
                        prediction = match.group(1)
                        logger.debug(f"_extract_grade_from_response: Found grade '{prediction}' in HTML-like tag")
                        break
            
            # Strategy 19: Check for grade in the context of decision words
            if prediction not in VALID_GRADES:
                decision_patterns = [
                    rf'(?:decision|verdict|ruling|judgment)\s*[:=]?\s*["\']?(correct|incorrect|partial|almost)["\']?',
                    rf'(?:determine|conclude|decide|assess)\s+(?:that\s+)?(?:it\s+is\s+)?["\']?(correct|incorrect|partial|almost)["\']?',
                ]
                for pattern in decision_patterns:
                    match = re.search(pattern, last_message.lower())
                    if match:
                        prediction = match.group(1)
                        logger.debug(f"_extract_grade_from_response: Found grade '{prediction}' via decision pattern")
                        break
            
            # Strategy 20: Final fallback - check entire message for any grade
            if prediction not in VALID_GRADES:
                text_lower = last_message.lower()
                # Priority order: almost, partial, incorrect, correct
                for grade in ["almost", "partial", "incorrect", "correct"]:
                    if re.search(rf'\b{grade}\b', text_lower):
                        prediction = grade
                        logger.debug(f"_extract_grade_from_response: Found grade '{prediction}' via full text search (final fallback)")
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
        last_error = None
        
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
                    self.log_fn(f"Successfully extracted grade '{prediction}' on attempt {attempt + 1}")
                    break
                else:
                    # Could not extract grade, add clarification to prompt for retry
                    if attempt < max_retries - 1:
                        self.log_fn(f"Warning: Could not extract grade on attempt {attempt + 1}. Retrying with clarification...")
                        # Add more specific guidance based on what we saw
                        last_message = msg_history[-1].get("text", "") if msg_history else ""
                        # Truncate to avoid overly long prompts
                        last_message_preview = last_message[:300] if len(last_message) > 300 else last_message
                        instruction += f"""\n\nCRITICAL ERROR: Your previous response did not contain a valid grade in the required JSON format.
Your response was: {last_message_preview}...

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
                        self.log_fn(f"Warning: Could not extract grade after {max_retries} attempts. Returning 'none'.")
                        
            except Exception as e:
                last_error = e
                logger.error(f"forward: Attempt {attempt + 1} failed: {e}", exc_info=True)
                if attempt == max_retries - 1:
                    # Return "none" instead of raising to maintain backward compatibility
                    self.log_fn(f"Error: Failed to get LLM response after {max_retries} attempts: {e}")
                    return "none", msg_history if msg_history else []

        return str(prediction), msg_history
