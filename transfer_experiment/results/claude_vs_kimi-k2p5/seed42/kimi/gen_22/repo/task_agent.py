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

# Valid grade labels
_VALID_GRADES = {"correct", "partial", "incorrect"}


def _extract_guideline_label(guidelines: str) -> str | None:
    """Extract explicit grade label from grading guidelines.
    
    The guidelines may contain labels like:
    - "(Partial)" → should be graded as "partial"
    - "(Almost)" → should be graded as "partial" (minor but non-negligible mistakes)
    - "(Incorrect)" → should be graded as "incorrect"
    - "(Correct)" → should be graded as "correct"
    
    Returns the expected grade if a label is found, None otherwise.
    
    Priority order: (Incorrect) > (Almost) > (Partial) > (Correct)
    This ensures we catch the most restrictive label first.
    """
    if not guidelines:
        return None
    
    guidelines_lower = guidelines.lower()
    
    # Priority 1: (Incorrect) - most restrictive
    if '(incorrect)' in guidelines_lower:
        return "incorrect"
    
    # Priority 2: (Almost) - means minor but NON-NEGLIGIBLE mistakes → must be "partial"
    # This is the most important edge case - models often confuse "almost" with "correct"
    if '(almost)' in guidelines_lower:
        return "partial"
    
    # Priority 3: (Partial) - explicit partial credit
    if '(partial)' in guidelines_lower:
        return "partial"
    
    # Priority 4: (Correct) - explicit full credit (less common)
    if '(correct)' in guidelines_lower:
        return "correct"
    
    # Additional check: look for label patterns with brackets or other delimiters
    # e.g., [Almost], [Partial], [Incorrect], [Correct], {Almost}, <Almost>
    if re.search(r'[\[\{<«]almost[\]\}>»]', guidelines_lower):
        return "partial"
    if re.search(r'[\[\{<«]partial[\]\}>»]', guidelines_lower):
        return "partial"
    if re.search(r'[\[\{<«]incorrect[\]\}>»]', guidelines_lower):
        return "incorrect"
    if re.search(r'[\[\{<«]correct[\]\}>»]', guidelines_lower):
        return "correct"
    
    # Check for label at the start of any line
    for line in guidelines_lower.split('\n'):
        line_stripped = line.strip()
        if line_stripped.startswith('(almost)') or line_stripped.startswith('almost'):
            return "partial"
        if line_stripped.startswith('(partial)') or line_stripped.startswith('partial'):
            return "partial"
        if line_stripped.startswith('(incorrect)') or line_stripped.startswith('incorrect'):
            return "incorrect"
        if line_stripped.startswith('(correct)') or line_stripped.startswith('correct'):
            return "correct"
    
    # Check for label anywhere in the text (not just at start of line)
    # This catches cases like "This is (Almost) correct" or "Grade: (Partial)"
    almost_patterns = [
        r'\(almost\)',
        r'\[almost\]',
        r'\{almost\}',
        r'\balmost\b.*\bcorrect\b',  # "almost correct" pattern
        r'\balmost\b.*\bright\b',     # "almost right" pattern
    ]
    for pattern in almost_patterns:
        if re.search(pattern, guidelines_lower):
            return "partial"
    
    partial_patterns = [
        r'\(partial\)',
        r'\[partial\]',
        r'\{partial\}',
        r'\bpartial\s+credit\b',
        r'\bpartially\s+correct\b',
    ]
    for pattern in partial_patterns:
        if re.search(pattern, guidelines_lower):
            return "partial"
    
    incorrect_patterns = [
        r'\(incorrect\)',
        r'\[incorrect\]',
        r'\{incorrect\}',
        r'\bzero\s+credit\b',
        r'\bno\s+credit\b',
    ]
    for pattern in incorrect_patterns:
        if re.search(pattern, guidelines_lower):
            return "incorrect"
    
    correct_patterns = [
        r'\(correct\)',
        r'\[correct\]',
        r'\{correct\}',
        r'\bfull\s+credit\b',
        r'\bcomplete\s+solution\b',
    ]
    for pattern in correct_patterns:
        if re.search(pattern, guidelines_lower):
            return "correct"
    
    return None


def _validate_grade_with_guidelines(predicted_grade: str, guideline_hint: str | None) -> str:
    """Validate and potentially correct the predicted grade based on guideline hints.
    
    This function enforces hard constraints from guideline labels to prevent
    common errors like grading an "(Almost)" answer as "correct".
    
    Hard rules:
    - If guidelines contain (Almost) → MUST be "partial" (never "correct")
    - If guidelines contain (Partial) → should be "partial"
    - If guidelines contain (Incorrect) → should be "incorrect"
    - If guidelines contain (Correct) → should be "correct"
    
    Returns the validated (and possibly corrected) grade.
    """
    if not guideline_hint:
        return predicted_grade
    
    # Hard constraint: (Almost) answers can NEVER be "correct"
    # They have minor but non-negligible mistakes → must be "partial"
    if guideline_hint == "partial" and predicted_grade == "correct":
        # This is a common error - override to partial
        return "partial"
    
    # If the guideline strongly suggests a grade and we predicted something else,
    # we might want to trust the guideline for clear-cut cases
    # But we keep the model's prediction if it's reasonable
    
    return predicted_grade


def _normalize_grade(grade: str) -> str | None:
    """Normalize a grade string to one of the valid grades.
    
    Handles common variations and misspellings.
    """
    if not grade:
        return None
    
    # Clean up the grade string: strip whitespace, quotes, backticks, and common punctuation
    grade_clean = grade.lower().strip().strip('"\'`)').strip('.,;:!?')
    grade_lower = grade_clean.strip()
    
    # Direct match
    if grade_lower in _VALID_GRADES:
        return grade_lower
    
    # Common variations - expanded for better coverage
    variations = {
        "correct": [
            "correct", "right", "true", "valid", "full", "full credit", "complete",
            "fully correct", "all correct", "perfect", "excellent", "good",
            "accurate", "proper", "sound", "flawless", "error-free", "100%",
            "full marks", "full score", "correct answer", "correct solution",
            "entirely correct", "completely correct", "totally correct",
            "yes", "pass", "accepted", "approved", "satisfactory", "success",
            "successful", "complete solution", "valid solution", "right answer",
            "true answer", "accurate answer", "proper answer", "sound answer",
            "flawless answer", "perfect answer", "excellent answer", "good answer",
            "correct work", "right work", "valid work", "sound work",
            "full points", "all points", "maximum points", "max points",
            "100 points", "100% correct", "completely right", "totally right",
            "entirely right", "fully right", "all right", "alright",
        ],
        "partial": [
            "partial", "part", "partially correct", "almost", "some credit", "half",
            "partial credit", "incomplete", "mostly correct", "nearly correct",
            "some progress", "partial solution", "partial answer", "half credit",
            "partially right", "partially valid", "minor errors", "small mistakes",
            "almost correct", "almost right", "nearly right", "mostly right",
            "partial marks", "partial score", "some marks", "in progress",
            "half correct", "half right", "50%", "partially done", "partially",
            "mostly", "nearly", "close", "incomplete solution", "incomplete answer",
            "almost there", "nearly there", "close to correct", "close to right",
            "mostly accurate", "mostly valid", "mostly sound", "mostly proper",
            "some correct", "some right", "some valid", "some proper",
            "partial work", "incomplete work", "unfinished", "unfinished solution",
            "unfinished answer", "unfinished work", "half done", "half finished",
            "partial success", "limited success", "some success", "moderate success",
            "50 percent", "fifty percent", "half marks", "half points",
            "partial points", "some points", "limited points", "moderate points",
        ],
        "incorrect": [
            "incorrect", "wrong", "false", "invalid", "no credit", "none", "zero",
            "fully wrong", "completely wrong", "totally wrong", "bad", "poor",
            "erroneous", "mistaken", "flawed", "unsound", "no progress", "failed",
            "0%", "zero credit", "no marks", "zero marks", "wrong answer",
            "incorrect answer", "wrong solution", "incorrect solution",
            "not correct", "not right", "not valid", "entirely wrong",
            "fail", "rejected", "unacceptable", "unsatisfactory",
            "wrong work", "bad work", "poor work", "flawed work",
            "erroneous work", "mistaken work", "unsound work", "invalid work",
            "false answer", "false solution", "false work",
            "invalid answer", "invalid solution", "invalid work",
            "zero points", "no points", "0 points", "zero percent",
            "0 percent", "none correct", "none right", "none valid",
            "not accurate", "not proper", "not sound", "not flawless",
            "completely false", "totally false", "entirely false",
            "fully false", "all wrong", "all false", "all incorrect",
            "fail answer", "fail solution", "fail work",
            "rejected answer", "rejected solution", "rejected work",
            "unacceptable answer", "unacceptable solution", "unacceptable work",
        ],
    }
    
    for canonical, variants in variations.items():
        if grade_lower in variants:
            return canonical
    
    # Check for partial matches with word boundaries
    # These patterns help catch variations like "partially correct" → "partial"
    if re.search(r'\bpartial\w*\b', grade_lower) or re.search(r'\balmost\b', grade_lower):
        return "partial"
    
    # Check for "correct" but exclude cases with negation or "incorrect"
    if re.search(r'\bcorrect\b', grade_lower):
        if "incorrect" not in grade_lower and "not " not in grade_lower:
            return "correct"
    
    # Check for incorrect/wrong patterns
    if re.search(r'\b(incorrect|wrong|false)\b', grade_lower):
        return "incorrect"
    
    # Check for numeric patterns (0, 50, 100, etc.)
    if re.search(r'\b0\b', grade_lower) or re.search(r'\b0%\b', grade_lower) or re.search(r'\bzero\b', grade_lower):
        return "incorrect"
    if re.search(r'\b100\b', grade_lower) or re.search(r'\b100%\b', grade_lower) or re.search(r'\bfull\b', grade_lower):
        return "correct"
    if re.search(r'\b50\b', grade_lower) or re.search(r'\b50%\b', grade_lower) or re.search(r'\bhalf\b', grade_lower):
        return "partial"
    
    # Check for common phrases that indicate grades
    if re.search(r'\b(full|complete|perfect)\s+(?:credit|marks|score|solution)\b', grade_lower):
        return "correct"
    if re.search(r'\b(partial|some|half)\s+(?:credit|marks|score|progress)\b', grade_lower):
        return "partial"
    if re.search(r'\b(no|zero)\s+(?:credit|marks|score|progress)\b', grade_lower):
        return "incorrect"
    
    return None


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    Also handles markdown code blocks with ```json or ```.
    Includes robust error recovery for common JSON formatting issues.
    """
    results = []
    
    def _try_parse_json(inner: str) -> dict | None:
        """Try to parse JSON with various fixes for common issues."""
        inner = inner.strip()
        if not inner:
            return None
            
        # Try direct parse first
        try:
            return json.loads(inner)
        except json.JSONDecodeError:
            pass
        
        # Fix 1: Handle trailing commas
        try:
            fixed = re.sub(r',\s*}', '}', inner)
            fixed = re.sub(r',\s*]', ']', fixed)
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass
        
        # Fix 2: Handle single quotes (convert to double)
        try:
            # Replace single quotes around keys and string values
            fixed = re.sub(r"'([^']*?)'\s*:", r'"\1":', inner)
            fixed = re.sub(r":\s*'([^']*?)'", r': "\1"', fixed)
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass
        
        # Fix 3: Handle unquoted keys
        try:
            fixed = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', inner)
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass
        
        # Fix 4: Handle Python-style True/False/None
        try:
            fixed = inner.replace('True', 'true').replace('False', 'false').replace('None', 'null')
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass
        
        # Fix 5: Handle escaped newlines in string values
        try:
            fixed = re.sub(r'("[^"]*?)\\n([^"]*?")', r'\1\\n\2', inner)
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass
        
        # Fix 6-20: Extract response field with various patterns (consolidated)
        # These handle cases where the JSON is malformed but the response field is present
        response_patterns = [
            # Standard double-quoted response
            r'"response"\s*:\s*"(correct|partial|incorrect)"',
            # Single-quoted response
            r"'response'\s*:\s*'(correct|partial|incorrect)'",
            # Response with escaped quotes
            r'"response"\s*:\s*\\"(correct|partial|incorrect)\\"',
            # Bare response without quotes
            r'"response"\s*:\s*(correct|partial|incorrect)\b',
            # Response at end of JSON (handles trailing content)
            r'"response"\s*:\s*"(correct|partial|incorrect)"[^}]*\}',
            # Response with grade/verdict/result field names
            r'"(?:grade|verdict|result|evaluation)"\s*:\s*"(correct|partial|incorrect)"',
            r"'(?:grade|verdict|result|evaluation)'\s*:\s*'(correct|partial|incorrect)'",
            # Response with answer field
            r'"answer"\s*:\s*"(correct|partial|incorrect)"',
            # Response with whitespace around value
            r'"response"\s*:\s*"\s*(correct|partial|incorrect)\s*"',
            # Response with backticks
            r'"response"\s*:\s*`(correct|partial|incorrect)`',
            # Response with triple quotes
            r'"response"\s*:\s*"""(correct|partial|incorrect)"""',
            # Response with single quotes and whitespace
            r"'response'\s*:\s*'\s*(correct|partial|incorrect)\s*'",
        ]
        for pattern in response_patterns:
            try:
                response_match = re.search(pattern, inner, re.IGNORECASE)
                if response_match:
                    return {"response": response_match.group(1).lower()}
            except Exception:
                pass
        
        # Fix 9: Handle JSON with newlines in string values
        try:
            # Replace newlines within string values with escaped newlines
            fixed = re.sub(r'("[^"]*?)\n([^"]*?")', r'\1\\n\2', inner)
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass
        
        # Fix 10: Handle single-quoted JSON with response field
        try:
            response_match = re.search(r"'response'\s*:\s*'(correct|partial|incorrect)'", inner, re.IGNORECASE)
            if response_match:
                return {"response": response_match.group(1).lower()}
        except Exception:
            pass
        
        # Fix 11: Handle bare words without quotes
        try:
            response_match = re.search(r'"response"\s*:\s*(correct|partial|incorrect)\b', inner, re.IGNORECASE)
            if response_match:
                return {"response": response_match.group(1).lower()}
        except Exception:
            pass
        
        # Fix 12: Handle JSON with escaped quotes in reasoning
        try:
            # Try to find response field even with complex escaped content
            response_match = re.search(r'"response"\s*:\s*"(correct|partial|incorrect)"', inner, re.IGNORECASE)
            if response_match:
                return {"response": response_match.group(1).lower()}
        except Exception:
            pass
        
        # Fix 13: Handle JSON with reasoning field containing newlines
        try:
            # Extract just the response field from malformed JSON
            response_match = re.search(r'"response"\s*:\s*"([^"]+)"', inner, re.IGNORECASE)
            if response_match:
                grade = response_match.group(1).lower().strip()
                if grade in _VALID_GRADES:
                    return {"response": grade}
        except Exception:
            pass
        
        # Fix 14: Handle case where response value has extra whitespace
        try:
            response_match = re.search(r'"response"\s*:\s*"\s*(correct|partial|incorrect)\s*"', inner, re.IGNORECASE)
            if response_match:
                return {"response": response_match.group(1).lower()}
        except Exception:
            pass
        
        # Fix 15: Handle case with backtick-quoted response
        try:
            response_match = re.search(r'"response"\s*:\s*`(correct|partial|incorrect)`', inner, re.IGNORECASE)
            if response_match:
                return {"response": response_match.group(1).lower()}
        except Exception:
            pass
        
        # Fix 16: Handle case with triple-quoted response
        try:
            response_match = re.search(r'"response"\s*:\s*"""(correct|partial|incorrect)"""', inner, re.IGNORECASE)
            if response_match:
                return {"response": response_match.group(1).lower()}
        except Exception:
            pass
        
        # Fix 17: Handle case with response field but no quotes around value
        try:
            response_match = re.search(r'["\']response["\']\s*:\s*(correct|partial|incorrect)(?:\s*[,}\]])', inner, re.IGNORECASE)
            if response_match:
                return {"response": response_match.group(1).lower()}
        except Exception:
            pass
        
        # Fix 18: Handle case with response field and any value (extract and validate)
        try:
            response_match = re.search(r'["\']response["\']\s*:\s*["\']?([^"\']+)["\']?', inner, re.IGNORECASE)
            if response_match:
                grade = response_match.group(1).lower().strip()
                normalized = _normalize_grade(grade)
                if normalized:
                    return {"response": normalized}
        except Exception:
            pass
        
        # Fix 19: Handle case with nested quotes in reasoning field
        try:
            # Find response field even if reasoning has complex nested content
            response_match = re.search(r'"response"\s*:\s*"(correct|partial|incorrect)"', inner, re.IGNORECASE)
            if response_match:
                return {"response": response_match.group(1).lower()}
        except Exception:
            pass
        
        # Fix 20: Handle JSON with comments (remove them)
        try:
            # Remove single-line comments
            fixed = re.sub(r'//.*?\n', '\n', inner)
            # Remove multi-line comments
            fixed = re.sub(r'/\*.*?\*/', '', fixed, flags=re.DOTALL)
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass
        
        return None
    
    # First, try to find <json>...</json> blocks
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
        
        parsed = _try_parse_json(inner)
        if parsed:
            results.append(parsed)
    
    # Also try markdown code blocks: ```json ... ``` or ``` ... ```
    code_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    for match in re.finditer(code_block_pattern, text, re.DOTALL | re.IGNORECASE):
        inner = match.group(1).strip()
        parsed = _try_parse_json(inner)
        if parsed:
            results.append(parsed)
    
    # Try to find bare JSON objects (not in any tags)
    # Look for patterns like {"response": "..."} or {response: "..."}
    bare_json_pattern = r'\{\s*["\']?response["\']?\s*:\s*["\']?(correct|partial|incorrect)["\']?\s*\}'
    for match in re.finditer(bare_json_pattern, text, re.IGNORECASE):
        try:
            parsed = _try_parse_json(match.group(0))
            if parsed:
                results.append(parsed)
        except Exception:
            pass
    
    # Try to find JSON with reasoning and response fields
    full_json_pattern = r'\{\s*"reasoning"\s*:\s*"[^"]*(?:\.[^"]*)*"\s*,\s*"response"\s*:\s*"(correct|partial|incorrect)"\s*\}'
    for match in re.finditer(full_json_pattern, text, re.IGNORECASE | re.DOTALL):
        try:
            parsed = _try_parse_json(match.group(0))
            if parsed:
                results.append(parsed)
        except Exception:
            pass
    
    # Try to find any JSON-like structure with response field
    # This handles cases where the JSON is malformed but contains a valid response
    response_field_pattern = r'"response"\s*:\s*"?(correct|partial|incorrect)"?'
    for match in re.finditer(response_field_pattern, text, re.IGNORECASE):
        try:
            grade = match.group(1).lower()
            if grade in _VALID_GRADES:
                results.append({"response": grade})
        except Exception:
            pass
    
    # Try to find response field with single quotes
    response_field_pattern2 = r"'response'\s*:\s*'(correct|partial|incorrect)'"
    for match in re.finditer(response_field_pattern2, text, re.IGNORECASE):
        try:
            grade = match.group(1).lower()
            if grade in _VALID_GRADES:
                results.append({"response": grade})
        except Exception:
            pass
    
    # Try to find response field with any value and normalize it
    response_field_pattern3 = r'["\']response["\']\s*:\s*["\']?([^"\']{3,20})["\']?'
    for match in re.finditer(response_field_pattern3, text, re.IGNORECASE):
        try:
            grade = match.group(1).lower().strip()
            normalized = _normalize_grade(grade)
            if normalized:
                results.append({"response": normalized})
        except Exception:
            pass
    
    return results or None


def _extract_grade_fallback(text: str) -> str | None:
    """Fallback: scan the text for a bare grade keyword.

    Uses multiple strategies to extract the grade, prioritizing:
    1. JSON-like structures with grade fields (highest priority)
    2. Explicit grade declarations near the end of text
    3. Pattern-based extraction with negation handling
    4. Last-resort standalone grade words
    
    Enhanced to better handle edge cases and ambiguous responses.
    """
    if not text:
        return None
        
    # Clean up the text: normalize whitespace but preserve structure
    text_clean = re.sub(r'[ \t]+', ' ', text).strip()
    text_lower = text_clean.lower()
    
    # Strategy 1: JSON-like structures (with or without proper formatting)
    # These have highest priority as they're the intended output format
    json_patterns = [
        # Standard double-quoted JSON with response field
        r'"response"\s*:\s*"(correct|partial|incorrect)"',
        # Single-quoted JSON
        r"'response'\s*:\s*'(correct|partial|incorrect)'",
        # Bare JSON-like: {response: correct}
        r'\{\s*response\s*:\s*(correct|partial|incorrect)\s*\}',
        # Other common field names
        r'"(?:grade|verdict|result|answer|evaluation)"\s*:\s*"(correct|partial|incorrect)"',
        r"'(?:grade|verdict|result|answer|evaluation)'\s*:\s*'(correct|partial|incorrect)'",
        # Response with backticks
        r'"response"\s*:\s*`(correct|partial|incorrect)`',
        # Response with triple quotes
        r'"response"\s*:\s*"""(correct|partial|incorrect)"""',
        # Response with single backticks
        r'"response"\s*:\s*`(correct|partial|incorrect)`',
        # Response without quotes (bare word)
        r'"response"\s*:\s*(correct|partial|incorrect)\s*[,}]',
        # Response field with whitespace
        r'"response"\s*:\s*"\s*(correct|partial|incorrect)\s*"',
    ]
    
    for pattern in json_patterns:
        matches = list(re.finditer(pattern, text_lower))
        if matches:
            # Return the last JSON match (most likely the final answer)
            return matches[-1].group(1).lower()
    
    # Strategy 2: Look for explicit final decision patterns near the end
    final_patterns = [
        # "Grade: X" or "Final grade: X" at end
        r'(?:final\s+)?(?:grade|verdict|assessment|decision|evaluation)\s*[:\-=]\s*(correct|partial|incorrect)(?:\s*$|\s*\.\s*$|\s*[,;])',
        # "I assign/grade X" at end
        r'(?:i\s+)?(?:assign|grade|give|award)\s+(?:a\s+)?(?:grade\s+of\s+)?["\']?(correct|partial|incorrect)["\']?(?:\s*$|\s*\.\s*$)',
        # "The grade/verdict is X" at end
        r'(?:the\s+)?(?:grade|verdict|assessment|result|evaluation)\s+is\s*["\']?(correct|partial|incorrect)["\']?(?:\s*$|\s*\.\s*$|\s*[,;])',
        # "Therefore/Thus/So the answer is X"
        r'(?:therefore|thus|so|hence)\s*,?\s*(?:the\s+)?(?:grade|answer|result|verdict)\s+is\s*["\']?(correct|partial|incorrect)["\']?',
        # "I conclude that the answer is X"
        r'(?:i\s+)?(?:conclude|determine|decide)\s+(?:that\s+)?(?:the\s+)?(?:grade|answer|result|verdict)\s+is\s*["\']?(correct|partial|incorrect)["\']?',
        # "Final answer: X"
        r'final\s+answer\s*[:\-]?\s*["\']?(correct|partial|incorrect)["\']?',
        # "Decision: X"
        r'decision\s*[:\-]?\s*["\']?(correct|partial|incorrect)["\']?',
    ]
    
    for pattern in final_patterns:
        matches = list(re.finditer(pattern, text_lower))
        if matches:
            return matches[-1].group(1).lower()
    
    # Strategy 3: Explicit statements with context validation
    # Look for "is correct/partial/incorrect" but check for negation
    is_matches = list(re.finditer(r'\bis\s+(correct|partial|incorrect)\b', text_lower))
    for match in reversed(is_matches):  # Check from end
        grade = match.group(1).lower()
        start_pos = match.start()
        context = text_lower[max(0, start_pos-50):start_pos]
        
        # Check for negation - if negated, skip this match
        negation_indicators = ['not ', 'isn\'t', 'is not', 'never', 'hardly', 'barely', 'not', 'neither', 'nor']
        if any(neg in context for neg in negation_indicators):
            if grade == 'correct':
                continue  # "is not correct" - skip
            elif grade == 'partial':
                continue  # "is not partial" - skip
            elif grade == 'incorrect':
                # "is not incorrect" could mean correct, but let's be conservative
                continue
        return grade
    
    # Strategy 4: Look for grade in the last 500 characters (increased from 300)
    # The final answer is often at the very end
    last_section = text_lower[-500:] if len(text_lower) > 500 else text_lower
    
    # First try to find a grade after common conclusion phrases
    conclusion_patterns = [
        r'(?:conclusion|final|summary|decision|verdict)\s*[:\-]?\s*(correct|partial|incorrect)',
        r'(?:grade|score|mark)\s*[:\-]?\s*(correct|partial|incorrect)',
        r'(?:the\s+student\s+(?:answer|solution|work)\s+is)\s*(correct|partial|incorrect)',
        r'(?:this\s+(?:answer|solution|work)\s+is)\s*(correct|partial|incorrect)',
        r'(?:i\s+rate\s+this|this\s+deserves|this\s+gets)\s+(?:as\s+)?["\']?(correct|partial|incorrect)["\']?',
    ]
    for pattern in conclusion_patterns:
        match = re.search(pattern, last_section)
        if match:
            return match.group(1).lower()
    
    # Then try simple word match in last section, but check for negation context
    last_grade_matches = list(re.finditer(r'\b(correct|partial|incorrect)\b', last_section))
    for match in reversed(last_grade_matches):
        grade = match.group(1).lower()
        start_pos = match.start()
        context = last_section[max(0, start_pos-30):start_pos]
        
        # Check for negation
        negation_indicators = ['not ', 'isn\'t', 'is not', 'never', 'neither', 'nor']
        if any(neg in context for neg in negation_indicators):
            continue  # Skip negated grades
        return grade
    
    # Strategy 5: Look for grade after reasoning/explanation sections
    # Often the grade appears after "Based on this analysis..." or similar
    analysis_patterns = [
        r'(?:based\s+on|given|considering)\s+(?:this|the\s+above|these)\s+(?:analysis|reasoning|evaluation|assessment).*?(correct|partial|incorrect)',
        r'(?:i\s+conclude|my\s+conclusion|the\s+student\s+(?:deserves|should\s+receive|gets)).*?\b(correct|partial|incorrect)\b',
        r'(?:after\s+reviewing|having\s+reviewed)\s+(?:the\s+)?(?:student\s+)?(?:answer|solution|work).*?(correct|partial|incorrect)',
        r'(?:in\s+summary|to\s+summarize|overall)\s*,?\s*(?:the\s+grade\s+is)?\s*(correct|partial|incorrect)',
    ]
    for pattern in analysis_patterns:
        matches = list(re.finditer(pattern, text_lower, re.DOTALL))
        if matches:
            return matches[-1].group(1).lower()
    
    # Strategy 6: Look for grade in specific output sections
    output_section_patterns = [
        r'<json>.*?"response"\s*:\s*"(correct|partial|incorrect)".*?</json>',
        r'```json.*?"response"\s*:\s*"(correct|partial|incorrect)".*?```',
        r'```.*?"response"\s*:\s*"(correct|partial|incorrect)".*?```',
        r'<output>.*?(correct|partial|incorrect).*?</output>',
        r'<grade>.*?(correct|partial|incorrect).*?</grade>',
    ]
    for pattern in output_section_patterns:
        matches = list(re.finditer(pattern, text_lower, re.DOTALL))
        if matches:
            return matches[-1].group(1).lower()
    
    # Strategy 7: Look for grade in parentheses or brackets (common in reasoning)
    bracket_patterns = [
        r'\((correct|partial|incorrect)\)',
        r'\[(correct|partial|incorrect)\]',
        r'\{(correct|partial|incorrect)\}',
    ]
    for pattern in bracket_patterns:
        matches = list(re.finditer(pattern, text_lower))
        if matches:
            # Return the last match in the last 200 chars
            last_section = text_lower[-200:] if len(text_lower) > 200 else text_lower
            last_matches = list(re.finditer(pattern, last_section))
            if last_matches:
                return last_matches[-1].group(1).lower()
    
    # Strategy 8: Last resort - find the last standalone grade word in full text
    # But be careful about negation
    standalone_matches = list(re.finditer(r'\b(correct|partial|incorrect)\b', text_lower))
    for match in reversed(standalone_matches):
        grade = match.group(1).lower()
        start_pos = match.start()
        context = text_lower[max(0, start_pos-30):start_pos]
        
        # Check for negation
        negation_indicators = ['not ', 'isn\'t', 'is not', 'never', 'neither', 'nor']
        if any(neg in context for neg in negation_indicators):
            continue  # Skip negated grades
        return grade
    
    # If all negation checks failed, just return the last match
    if standalone_matches:
        return standalone_matches[-1].group(1).lower()

    return None


class TaskAgent:
    """Task agent that solves IMO grading problems."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_file = log_file
        self.log_fn = logger.info

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines,
                    student_answer

        Returns:
            (prediction, msg_history)
        """
        problem = inputs.get("problem", "")
        official_solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")


        # Add guideline hint context if available
        hint_section = ""
        if guideline_hint:
            # Special handling for (Almost) - this is a critical edge case
            if guideline_hint == "partial" and '(almost)' in (grading_guidelines or "").lower():
                hint_section = f"""
════════════════════════════════════════════════════════════════
⚠️  CRITICAL: GRADING GUIDELINE CONSTRAINT
════════════════════════════════════════════════════════════════

The grading guidelines contain the label: (ALMOST)

HARD RULE: Answers labeled "(Almost)" can NEVER be graded as "correct".
They MUST be graded as "partial" because they contain minor but NON-NEGLIGIBLE mistakes.

This is a CONSTRAINT, not a hint. You MUST output "partial" for (Almost) answers.
"""
            else:
                hint_section = f"""
════════════════════════════════════════════════════════════════
GRADING GUIDELINE HINT (IMPORTANT CONTEXT)
════════════════════════════════════════════════════════════════

The grading guidelines contain the label: ({guideline_hint.upper()})

This label indicates what the student answer CONTAINS, but you must still
independently evaluate the answer against the official solution to determine
the final grade:

- (Partial) or (Almost) in guidelines → Student has partial progress, but you
  must verify if it's truly "partial" or could be "correct" or "incorrect"
- (Correct) in guidelines → Student claims a correct solution, verify it's truly perfect
- (Incorrect) in guidelines → Student has errors, verify if they're major (incorrect)
  or minor (partial)

DO NOT blindly follow this hint - use your own judgment based on the actual content.
"""

        instruction = f"""You are an expert mathematics competition grader. Your task is to evaluate the student's answer and assign exactly ONE grade: "correct", "partial", or "incorrect".

════════════════════════════════════════════════════════════════
GRADE DEFINITIONS - READ CAREFULLY
════════════════════════════════════════════════════════════════

【correct】= FULL CREDIT (100%) - USE ONLY WHEN PERFECT
  • The solution is COMPLETE and FLAWLESS
  • Every logical step is valid and correctly executed
  • Final answer exactly matches the official solution
  • NO errors, NO missing steps, NO approximations when exact answer required
  → Use ONLY when the student deserves FULL credit

【incorrect】= ZERO CREDIT (0%) - USE FOR FUNDAMENTALLY WRONG
  • The approach is completely wrong or misguided
  • No meaningful mathematical progress toward the solution
  • Major conceptual errors that invalidate the entire solution
  • Answer is nonsense or unrelated to the problem
  → Use when the student deserves ZERO credit

【partial】= PARTIAL CREDIT (1-99%) - USE FOR MEANINGFUL BUT INCOMPLETE PROGRESS
  • The student made MEANINGFUL progress but has SIGNIFICANT issues
  • Correct approach started but INCOMPLETE or contains IMPORTANT ERRORS
  • Has non-negligible mistakes that affect the final result
  • "Almost" correct but missing critical final steps or has substantive errors
  → Use when the student deserves MORE than zero but LESS than full credit

════════════════════════════════════════════════════════════════
CRITICAL DECISION RULES
════════════════════════════════════════════════════════════════

1. When in doubt between "correct" and "partial", choose "partial"
   - "correct" requires PERFECTION - any flaw makes it "partial"

2. When in doubt between "partial" and "incorrect", ask:
   - Did the student demonstrate understanding of the problem? 
   - If YES → "partial"
   - If NO → "incorrect"

3. SPECIAL RULE for "(Almost)" labeled answers:
   - If the grading guidelines contain "(Almost)", the answer has MINOR but NON-NEGLIGIBLE mistakes
   - "(Almost)" answers can NEVER be "correct" - they MUST be graded as "partial"
   - This is a HARD CONSTRAINT: (Almost) → always "partial", never "correct"

4. The grading guidelines may contain labels like (Almost), (Partial), (Incorrect), or (Correct)
   - These are HINTS about what the student answer contains, NOT the final grade
   - You must independently verify by comparing the student answer to the official solution
   - Do NOT blindly follow these labels - use your own expert judgment
   - EXCEPTION: "(Almost)" is a strong hint for "partial" - see rule 3 above{hint_section}

════════════════════════════════════════════════════════════════
PROBLEM
════════════════════════════════════════════════════════════════
{problem}

════════════════════════════════════════════════════════════════
OFFICIAL SOLUTION
════════════════════════════════════════════════════════════════
{official_solution}

════════════════════════════════════════════════════════════════
GRADING GUIDELINES
════════════════════════════════════════════════════════════════
{grading_guidelines}

════════════════════════════════════════════════════════════════
STUDENT'S ANSWER
════════════════════════════════════════════════════════════════
{student_answer}

════════════════════════════════════════════════════════════════
YOUR TASK
════════════════════════════════════════════════════════════════

Step 1: Carefully read the official solution and understand the correct approach
Step 2: Compare the student's answer step-by-step with the official solution
Step 3: Identify ALL errors, missing steps, or incorrect conclusions
Step 4: Check if the grading guidelines contain "(Almost)" - if yes, the grade MUST be "partial"
Step 5: Classify the severity of errors:
   - Minor/negligible → may still be "correct" or "partial"
   - Significant/non-negligible → "partial" 
   - Fundamental/major → "incorrect"
Step 6: Choose the grade based on the definitions and rules above
Step 7: Provide your decision in the required JSON format

════════════════════════════════════════════════════════════════
OUTPUT FORMAT (STRICT)
════════════════════════════════════════════════════════════════

You MUST provide your decision in <json> tags with exactly these fields:
- "reasoning": A clear explanation of your grading decision, citing specific errors or correct steps
- "response": MUST be exactly one of: "correct", "partial", or "incorrect" (lowercase, no quotes)

<json>
{{
    "reasoning": "Your detailed explanation here. Cite specific issues found or confirm perfection.",
    "response": "correct" | "partial" | "incorrect"
}}
</json>

IMPORTANT: The "response" field must contain ONLY the word correct, partial, or incorrect.
Do not add extra text, explanations, or formatting in the response field.

════════════════════════════════════════════════════════════════
NOW GRADE THE ANSWER
════════════════════════════════════════════════════════════════"""

        # Check for explicit guideline labels to inform the LLM's decision
        # The labels (Almost), (Partial), (Incorrect), (Correct) in guidelines
        # indicate what the student answer contains, but the final grade must
        # be determined by evaluating the actual student answer against the
        # official solution. We pass this info to the LLM but don't short-circuit.
        guideline_hint = _extract_guideline_label(grading_guidelines)
        
        if guideline_hint:
            self.log_fn(f"Guideline label detected: '{guideline_hint}' - using as hint for LLM evaluation")
        else:
            self.log_fn("No explicit label in guidelines - calling LLM for evaluation")
        
        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # ------------------------------------------------------------------ #
        # Extract prediction from the assistant's response.                   #
        # Strategy:                                                            #
        #   1. Look for <json>...</json> blocks in the last assistant message. #
        #   2. If that fails, try a regex fallback on the full response text.  #
        #   3. Apply guideline-based validation to catch edge cases.          #
        #   4. Default to "incorrect" for safety if no valid grade found.      #
        # ------------------------------------------------------------------ #
        prediction = "incorrect"  # Default to incorrect for safety
        
        # Get assistant text from message history or response
        assistant_text = ""
        try:
            if msg_history:
                last_msg = msg_history[-1]
                # Handle both "content" and "text" keys
                assistant_text = last_msg.get("content", "") or last_msg.get("text", "") or ""
                # Handle Anthropic-style content blocks
                if isinstance(assistant_text, list):
                    parts = []
                    for block in assistant_text:
                        if isinstance(block, dict) and isinstance(block.get("text"), str):
                            parts.append(block["text"])
                        elif isinstance(block, str):
                            parts.append(block)
                    assistant_text = "\n".join(parts) if parts else ""
            if not assistant_text and response:
                assistant_text = response
        except Exception:
            assistant_text = response or ""

        # Try to extract from JSON blocks first
        if assistant_text:
            try:
                extracted = _extract_jsons(assistant_text)
                if extracted:
                    # Prefer the last JSON block that contains a valid "response" key
                    for obj in reversed(extracted):
                        grade = str(obj.get("response", "")).strip().lower()
                        normalized = _normalize_grade(grade)
                        if normalized:
                            prediction = normalized
                            break
                    else:
                        # No valid grade found in any block; take the last block's value if present
                        last_response = str(extracted[-1].get("response", "")).strip().lower()
                        normalized = _normalize_grade(last_response)
                        if normalized:
                            prediction = normalized
                    
            except Exception as e:
                self.log_fn(f"Error extracting prediction from JSON blocks: {e}")

        # Fallback: scan the raw text for a grade keyword
        if prediction not in _VALID_GRADES:
            try:
                fallback = _extract_grade_fallback(assistant_text)
                normalized = _normalize_grade(fallback)
                if normalized:
                    prediction = normalized
                    self.log_fn(f"Used fallback grade extraction: {prediction}")
            except Exception as e:
                self.log_fn(f"Error in fallback grade extraction: {e}")
        
        # Apply guideline-based validation to catch edge cases
        # This enforces hard constraints like: (Almost) → must be "partial"
        original_prediction = prediction
        prediction = _validate_grade_with_guidelines(prediction, guideline_hint)
        if prediction != original_prediction:
            self.log_fn(f"Grade corrected by guideline validation: {original_prediction} → {prediction}")
        
        # Final validation: ensure we always return a valid grade
        if prediction not in _VALID_GRADES:
            prediction = "incorrect"
            self.log_fn("No valid grade found, defaulting to 'incorrect' for safety")

        return str(prediction), msg_history
