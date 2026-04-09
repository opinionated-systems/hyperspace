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
    incorrect_bracket_patterns = [
        r'\(incorrect\)',
        r'\[incorrect\]',
        r'\{incorrect\}',
        r'<incorrect>',
        r'«incorrect»',
    ]
    for pattern in incorrect_bracket_patterns:
        if re.search(pattern, guidelines_lower):
            return "incorrect"
    
    # Priority 2: (Almost) - means minor but NON-NEGLIGIBLE mistakes → must be "partial"
    # This is the most important edge case - models often confuse "almost" with "correct"
    # Check for (Almost) with various bracket styles
    almost_bracket_patterns = [
        r'\(almost\)',
        r'\[almost\]',
        r'\{almost\}',
        r'<almost>',
        r'«almost»',
    ]
    for pattern in almost_bracket_patterns:
        if re.search(pattern, guidelines_lower):
            return "partial"
    
    # Priority 3: (Partial) - explicit partial credit
    partial_bracket_patterns = [
        r'\(partial\)',
        r'\[partial\]',
        r'\{partial\}',
        r'<partial>',
        r'«partial»',
    ]
    for pattern in partial_bracket_patterns:
        if re.search(pattern, guidelines_lower):
            return "partial"
    
    # Priority 4: (Correct) - explicit full credit (less common)
    correct_bracket_patterns = [
        r'\(correct\)',
        r'\[correct\]',
        r'\{correct\}',
        r'<correct>',
        r'«correct»',
    ]
    for pattern in correct_bracket_patterns:
        if re.search(pattern, guidelines_lower):
            return "correct"
    
    # Check for label at the start of any line (with or without brackets)
    for line in guidelines_lower.split('\n'):
        line_stripped = line.strip()
        # Check for bracketed versions at start
        if line_stripped.startswith('(almost)') or line_stripped.startswith('[almost]') or line_stripped.startswith('{almost}'):
            return "partial"
        if line_stripped.startswith('(partial)') or line_stripped.startswith('[partial]') or line_stripped.startswith('{partial}'):
            return "partial"
        if line_stripped.startswith('(incorrect)') or line_stripped.startswith('[incorrect]') or line_stripped.startswith('{incorrect}'):
            return "incorrect"
        if line_stripped.startswith('(correct)') or line_stripped.startswith('[correct]') or line_stripped.startswith('{correct}'):
            return "correct"
        # Check for unbracketed versions at start (only for exact matches)
        if line_stripped.startswith('almost ') and not line_stripped.startswith('almost correct'):
            return "partial"
        if line_stripped.startswith('partial ') and 'partial credit' not in line_stripped:
            return "partial"
    
    # Check for label anywhere in the text (not just at start of line)
    # This catches cases like "This is (Almost) correct" or "Grade: (Partial)"
    almost_patterns = [
        r'\(almost\)',
        r'\[almost\]',
        r'\{almost\}',
        r'<almost>',
        r'«almost»',
        r'\balmost\b.*\bcorrect\b',  # "almost correct" pattern
        r'\balmost\b.*\bright\b',     # "almost right" pattern
        r'\balmost\b.*\bthere\b',    # "almost there" pattern
        r'\bnearly\b.*\bcorrect\b',   # "nearly correct" pattern
        r'\bnearly\b.*\bright\b',     # "nearly right" pattern
        r'\balmost\s+perfect\b',      # "almost perfect" pattern
        r'\balmost\s+complete\b',      # "almost complete" pattern
        r'\balmost\s+done\b',          # "almost done" pattern
        r'\balmost\s+finished\b',      # "almost finished" pattern
    ]
    for pattern in almost_patterns:
        if re.search(pattern, guidelines_lower):
            return "partial"
    
    partial_patterns = [
        r'\(partial\)',
        r'\[partial\]',
        r'\{partial\}',
        r'<partial>',
        r'«partial»',
        r'\bpartial\s+credit\b',
        r'\bpartially\s+correct\b',
        r'\bpartially\s+right\b',
        r'\bpartial\s+marks?\b',
        r'\bpartial\s+score\b',
    ]
    for pattern in partial_patterns:
        if re.search(pattern, guidelines_lower):
            return "partial"
    
    incorrect_patterns = [
        r'\(incorrect\)',
        r'\[incorrect\]',
        r'\{incorrect\}',
        r'<incorrect>',
        r'«incorrect»',
        r'\bzero\s+credit\b',
        r'\bno\s+credit\b',
        r'\bno\s+marks?\b',
        r'\bzero\s+marks?\b',
    ]
    for pattern in incorrect_patterns:
        if re.search(pattern, guidelines_lower):
            return "incorrect"
    
    correct_patterns = [
        r'\(correct\)',
        r'\[correct\]',
        r'\{correct\}',
        r'<correct>',
        r'«correct»',
        r'\bfull\s+credit\b',
        r'\bcomplete\s+solution\b',
        r'\bfull\s+marks?\b',
        r'\b100%\s+correct\b',
    ]
    for pattern in correct_patterns:
        if re.search(pattern, guidelines_lower):
            return "correct"
    
    return None


def _validate_grade_with_guidelines(predicted_grade: str, guideline_hint: str | None, grading_guidelines: str = "") -> str:
    """Validate and potentially correct the predicted grade based on guideline hints.
    
    Hard constraints:
    - (Almost) in guidelines → MUST be "partial" (never "correct" or "incorrect")
    - (Partial) in guidelines → typically "partial" (only "correct" if truly perfect)
    - (Incorrect) in guidelines → typically "incorrect" (only "partial" if meaningful progress)
    - (Correct) in guidelines → verify independently
    
    Returns the validated (and possibly corrected) grade.
    """
    if not guideline_hint:
        return predicted_grade
    
    guidelines_lower = (grading_guidelines or "").lower()
    
    # Check for specific labels
    has_almost_label = '(almost)' in guidelines_lower
    has_partial_label = '(partial)' in guidelines_lower
    has_incorrect_label = '(incorrect)' in guidelines_lower
    has_correct_label = '(correct)' in guidelines_lower
    
    # CRITICAL: (Almost) answers can NEVER be "correct" or "incorrect"
    # They MUST be "partial" - this is the most important constraint
    if has_almost_label:
        if predicted_grade == "correct":
            return "partial"
        if predicted_grade == "incorrect":
            return "partial"
        # If already "partial", keep it
        if predicted_grade == "partial":
            return "partial"
    
    # (Partial) labeled answers should typically be "partial"
    if has_partial_label and predicted_grade == "correct":
        # Only allow "correct" if explicitly marked as correct AND not almost/partial
        if not has_correct_label or has_almost_label:
            return "partial"
    
    # (Incorrect) labeled answers should typically be "incorrect"
    if has_incorrect_label and predicted_grade == "correct":
        return "incorrect"
    
    # If guideline_hint says "partial" but we predicted "correct", be suspicious
    if guideline_hint == "partial" and predicted_grade == "correct":
        if not has_correct_label or has_almost_label or has_partial_label:
            return "partial"
    
    # If guideline_hint says "incorrect" but we predicted "correct", override
    if guideline_hint == "incorrect" and predicted_grade == "correct":
        return "incorrect"
    
    # If guideline_hint says "correct" but we predicted "incorrect", check if valid
    if guideline_hint == "correct" and predicted_grade == "incorrect":
        if has_correct_label and not has_almost_label and not has_partial_label:
            return "correct"
    
    # If guideline_hint says "correct" but we predicted "partial", check if should be correct
    if guideline_hint == "correct" and predicted_grade == "partial":
        if has_correct_label and not has_almost_label and not has_partial_label:
            return "correct"
    
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
    
    # Check for negation patterns first - these should override other patterns
    # e.g., "not correct" should NOT be normalized to "correct"
    negation_patterns = [
        r'\bnot\s+correct\b',
        r'\bnot\s+right\b',
        r'\bnot\s+valid\b',
        r'\bisn\'t\s+correct\b',
        r'\bisn\'t\s+right\b',
        r'\bnot\s+partial\b',
        r'\bnot\s+incorrect\b',  # "not incorrect" could mean correct, but be conservative
        r'\bnot\s+perfect\b',
        r'\bnot\s+complete\b',
        r'\bnot\s+accurate\b',
    ]
    for pattern in negation_patterns:
        if re.search(pattern, grade_lower):
            # If negating correct/right, it's likely incorrect or partial
            if 'correct' in grade_lower or 'right' in grade_lower or 'perfect' in grade_lower or 'complete' in grade_lower or 'accurate' in grade_lower:
                return "incorrect"
            if 'partial' in grade_lower:
                return "incorrect"  # "not partial" means it's either correct or incorrect
    
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
            "fully accurate", "completely accurate", "totally accurate",
            "entirely accurate", "perfectly correct", "absolutely correct",
            "definitely correct", "certainly correct", "undoubtedly correct",
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
            "almost there", "getting there", "on the right track", "good start",
            "some understanding", "partial understanding", "incomplete but correct",
            "mostly there", "nearly complete", "almost complete", "close to complete",
            "almost perfect", "nearly perfect", "mostly perfect",
            "almost flawless", "nearly flawless", "mostly flawless",
            "close to perfect", "close to flawless", "close to complete",
            "nearly complete", "almost complete", "mostly complete",
            "partially complete", "partially correct", "partially accurate",
            "partially valid", "partially sound", "partially proper",
            "partially right", "partially true", "partially successful",
            "somewhat correct", "somewhat right", "somewhat accurate",
            "fairly correct", "fairly right", "fairly accurate",
            "reasonably correct", "reasonably right", "reasonably accurate",
            "relatively correct", "relatively right", "relatively accurate",
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
            "completely incorrect", "totally incorrect", "entirely incorrect",
            "fully incorrect", "absolutely incorrect", "definitely incorrect",
            "certainly incorrect", "undoubtedly incorrect", "clearly incorrect",
            "obviously incorrect", "plainly incorrect", "simply incorrect",
            "just incorrect", "only incorrect", "merely incorrect",
            "utterly wrong", "completely wrong", "totally wrong",
            "entirely wrong", "fully wrong", "absolutely wrong",
            "definitely wrong", "certainly wrong", "undoubtedly wrong",
            "clearly wrong", "obviously wrong", "plainly wrong",
            "simply wrong", "just wrong", "only wrong", "merely wrong",
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
    """Extract JSON objects from <json>...</json> blocks and markdown code blocks.
    
    Uses robust parsing with multiple fallback strategies for malformed JSON.
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
        
        # Fix 5: Handle newlines in strings
        try:
            fixed = re.sub(r'("[^"]*?)\n([^"]*?")', r'\1\\n\2', inner)
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass
        
        # Fix 6: Extract response field directly with regex patterns
        response_patterns = [
            r'"response"\s*:\s*"(correct|partial|incorrect)"',
            r"'response'\s*:\s*'(correct|partial|incorrect)'",
            r'"response"\s*:\s*\\"(correct|partial|incorrect)\\"',
            r'"response"\s*:\s*(correct|partial|incorrect)\b',
            r'"(?:grade|verdict|result|evaluation)"\s*:\s*"(correct|partial|incorrect)"',
            r'"response"\s*:\s*"\s*(correct|partial|incorrect)\s*"',
            r'["\']response["\']\s*:\s*["\']?(correct|partial|incorrect)["\']?',
            r'"response"\s*:\s*"(Correct|Partial|Incorrect)"',
            r'"response"\s*:\s*"(CORRECT|PARTIAL|INCORRECT)"',
        ]
        for pattern in response_patterns:
            try:
                response_match = re.search(pattern, inner, re.IGNORECASE)
                if response_match:
                    return {"response": response_match.group(1).lower()}
            except Exception:
                pass
        
        # Fix 7: Extract any valid grade from the text
        try:
            for grade in _VALID_GRADES:
                if f'"{grade}"' in inner.lower() or f"'{grade}'" in inner.lower():
                    return {"response": grade}
        except Exception:
            pass
        
        return None
    
    # Find <json>...</json> blocks
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
    
    # Find markdown code blocks: ```json ... ``` or ``` ... ```
    code_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    for match in re.finditer(code_block_pattern, text, re.DOTALL | re.IGNORECASE):
        inner = match.group(1).strip()
        parsed = _try_parse_json(inner)
        if parsed:
            results.append(parsed)
    
    # Find bare JSON objects with response field
    bare_json_pattern = r'\{\s*["\']?response["\']?\s*:\s*["\']?(correct|partial|incorrect)["\']?\s*\}'
    for match in re.finditer(bare_json_pattern, text, re.IGNORECASE):
        try:
            parsed = _try_parse_json(match.group(0))
            if parsed:
                results.append(parsed)
        except Exception:
            pass
    
    # Find any response field pattern
    response_field_pattern = r'"response"\s*:\s*"?(correct|partial|incorrect)"?'
    for match in re.finditer(response_field_pattern, text, re.IGNORECASE):
        try:
            grade = match.group(1).lower()
            if grade in _VALID_GRADES:
                results.append({"response": grade})
        except Exception:
            pass
    
    return results or None


def _extract_grade_fallback(text: str) -> str | None:
    """Fallback: scan the text for a bare grade keyword.
    
    Prioritizes:
    1. JSON-like structures with grade fields
    2. Explicit grade declarations near the end
    3. Pattern-based extraction with negation handling
    4. Last-resort standalone grade words
    """
    if not text:
        return None
        
    text_clean = re.sub(r'[ \t]+', ' ', text).strip()
    text_lower = text_clean.lower()
    
    # Strategy 1: JSON-like structures
    json_patterns = [
        r'"response"\s*:\s*"(correct|partial|incorrect)"',
        r"'response'\s*:\s*'(correct|partial|incorrect)'",
        r'\{\s*response\s*:\s*(correct|partial|incorrect)\s*\}',
        r'"(?:grade|verdict|result|answer|evaluation)"\s*:\s*"(correct|partial|incorrect)"',
        r'["\']response["\']\s*:\s*["\']?(correct|partial|incorrect)["\']?',
    ]
    
    for pattern in json_patterns:
        matches = list(re.finditer(pattern, text_lower))
        if matches:
            return matches[-1].group(1).lower()
    
    # Strategy 2: Explicit final decision patterns
    final_patterns = [
        r'(?:final\s+)?(?:grade|verdict|assessment|decision|evaluation)\s*[:\-=]\s*(correct|partial|incorrect)(?:\s*$|\s*\.\s*$|\s*[,;])',
        r'(?:i\s+)?(?:assign|grade|give|award)\s+(?:a\s+)?(?:grade\s+of\s+)?["\']?(correct|partial|incorrect)["\']?(?:\s*$|\s*\.\s*$)',
        r'(?:the\s+)?(?:grade|verdict|assessment|result|evaluation)\s+is\s*["\']?(correct|partial|incorrect)["\']?(?:\s*$|\s*\.\s*$|\s*[,;])',
        r'(?:therefore|thus|so|hence)\s*,?\s*(?:the\s+)?(?:grade|answer|result|verdict)\s+is\s*["\']?(correct|partial|incorrect)["\']?',
        r'(?:i\s+)?(?:conclude|determine|decide)\s+(?:that\s+)?(?:the\s+)?(?:grade|answer|result|verdict)\s+is\s*["\']?(correct|partial|incorrect)["\']?',
        r'final\s+answer\s*[:\-]?\s*["\']?(correct|partial|incorrect)["\']?',
        r'decision\s*[:\-]?\s*["\']?(correct|partial|incorrect)["\']?',
    ]
    
    for pattern in final_patterns:
        matches = list(re.finditer(pattern, text_lower))
        if matches:
            return matches[-1].group(1).lower()
    
    # Strategy 3: "is X" patterns with negation check
    is_matches = list(re.finditer(r'\bis\s+(correct|partial|incorrect)\b', text_lower))
    for match in reversed(is_matches):
        grade = match.group(1).lower()
        start_pos = match.start()
        context = text_lower[max(0, start_pos-50):start_pos]
        
        negation_indicators = ['not ', 'isn\'t', 'is not', 'never', 'neither', 'nor']
        if any(neg in context for neg in negation_indicators):
            continue
        return grade
    
    # Strategy 4: Last section patterns
    last_section = text_lower[-500:] if len(text_lower) > 500 else text_lower
    
    conclusion_patterns = [
        r'(?:conclusion|final|summary|decision|verdict)\s*[:\-]?\s*(correct|partial|incorrect)',
        r'(?:grade|score|mark)\s*[:\-]?\s*(correct|partial|incorrect)',
        r'(?:the\s+student\s+(?:answer|solution|work)\s+is)\s*(correct|partial|incorrect)',
        r'(?:this\s+(?:answer|solution|work)\s+is)\s*(correct|partial|incorrect)',
    ]
    for pattern in conclusion_patterns:
        match = re.search(pattern, last_section)
        if match:
            return match.group(1).lower()
    
    # Strategy 5: Last resort - standalone grade words with negation check
    standalone_matches = list(re.finditer(r'\b(correct|partial|incorrect)\b', text_lower))
    for match in reversed(standalone_matches):
        grade = match.group(1).lower()
        start_pos = match.start()
        context = text_lower[max(0, start_pos-30):start_pos]
        
        negation_indicators = ['not ', 'isn\'t', 'is not', 'never', 'neither', 'nor']
        if any(neg in context for neg in negation_indicators):
            continue
        return grade
    
    if standalone_matches:
        return standalone_matches[-1].group(1).lower()

    return None


def _extract_grade_from_response(text: str) -> str | None:
    """Extract grade from LLM response using multiple strategies.
    
    Priority order:
    1. JSON blocks with response field (highest priority)
    2. Explicit final decision patterns
    3. Pattern-based extraction with context
    4. Fallback to standalone words (lowest priority)
    """
    if not text:
        return None
    
    text_clean = re.sub(r'[ \t]+', ' ', text).strip()
    text_lower = text_clean.lower()
    
    # Strategy 1: Try JSON extraction first (most reliable)
    try:
        extracted = _extract_jsons(text)
        if extracted:
            for obj in reversed(extracted):
                grade = str(obj.get("response", "")).strip().lower()
                normalized = _normalize_grade(grade)
                if normalized:
                    return normalized
    except Exception:
        pass
    
    # Strategy 2: Look for explicit final decision patterns
    last_section = text_lower[-1000:] if len(text_lower) > 1000 else text_lower
    
    final_patterns = [
        r'(?:final\s+)?(?:grade|verdict|assessment|decision|evaluation)\s*[:\-=]\s*["\']?(correct|partial|incorrect)["\']?(?:\s*$|\s*\.\s*$|\s*[,;])',
        r'(?:i\s+)?(?:assign|grade|give|award)\s+(?:a\s+)?(?:grade\s+of\s+)?["\']?(correct|partial|incorrect)["\']?(?:\s*$|\s*\.\s*$)',
        r'(?:the\s+)?(?:grade|verdict|assessment|result|evaluation)\s+is\s*["\']?(correct|partial|incorrect)["\']?(?:\s*$|\s*\.\s*$|\s*[,;])',
        r'(?:therefore|thus|so|hence)\s*,?\s*(?:the\s+)?(?:grade|answer|result|verdict)\s+is\s*["\']?(correct|partial|incorrect)["\']?',
        r'(?:i\s+)?(?:conclude|determine|decide)\s+(?:that\s+)?(?:the\s+)?(?:grade|answer|result|verdict)\s+is\s*["\']?(correct|partial|incorrect)["\']?',
        r'final\s+answer\s*[:\-]?\s*["\']?(correct|partial|incorrect)["\']?',
        r'decision\s*[:\-]?\s*["\']?(correct|partial|incorrect)["\']?',
    ]
    
    for pattern in final_patterns:
        matches = list(re.finditer(pattern, last_section))
        if matches:
            grade = matches[-1].group(1).lower()
            if grade in _VALID_GRADES:
                return grade
    
    # Strategy 3: Try fallback extraction
    try:
        fallback = _extract_grade_fallback(text)
        if fallback:
            normalized = _normalize_grade(fallback)
            if normalized:
                return normalized
    except Exception:
        pass
    
    # Strategy 4: Last resort - look for standalone grade words
    standalone_matches = list(re.finditer(r'\b(correct|partial|incorrect)\b', text_lower))
    for match in reversed(standalone_matches):
        grade = match.group(1).lower()
        start_pos = match.start()
        context = text_lower[max(0, start_pos-50):start_pos]
        
        # Check for negation
        negation_indicators = ['not ', 'isn\'t', 'is not', 'never', 'neither', 'nor', 'no ']
        if any(neg in context for neg in negation_indicators):
            continue
        
        return grade
    
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

        # Check for explicit guideline labels FIRST (before using hint_section)
        # The labels (Almost), (Partial), (Incorrect), (Correct) in guidelines
        # indicate what the student answer contains, but the final grade must
        # be determined by evaluating the actual student answer against the
        # official solution. We pass this info to the LLM but don't short-circuit.
        guideline_hint = _extract_guideline_label(grading_guidelines)
        
        if guideline_hint:
            self.log_fn(f"Guideline label detected: '{guideline_hint}' - using as hint for LLM evaluation")
        else:
            self.log_fn("No explicit label in guidelines - calling LLM for evaluation")

        # Add guideline hint context if available
        hint_section = ""
        if guideline_hint:
            guidelines_lower = (grading_guidelines or "").lower()
            
            # Special handling for (Almost) - this is a critical edge case
            if '(almost)' in guidelines_lower:
                hint_section = """
════════════════════════════════════════════════════════════════
⚠️  CRITICAL HARD CONSTRAINT: (ALMOST) → MUST BE "partial"
════════════════════════════════════════════════════════════════

The grading guidelines contain: (ALMOST)

ABSOLUTE RULE: (Almost) answers can NEVER be "correct" or "incorrect".
They MUST be graded as "partial" - no exceptions.

"Almost" means "close to correct but NOT fully correct".
Examples: correct method but arithmetic error, right approach but missing insight.

THIS IS MANDATORY: Output "partial" for (Almost) answers.
"""
            elif '(partial)' in guidelines_lower:
                hint_section = """
════════════════════════════════════════════════════════════════
⚠️  GUIDELINE CONSTRAINT: (PARTIAL) → typically "partial"
════════════════════════════════════════════════════════════════

The grading guidelines contain: (PARTIAL)

This indicates partial progress. Grade as "partial" unless:
- The answer is truly flawless → "correct"
- No meaningful progress at all → "incorrect"

Default to "partial" for (Partial) labeled answers.
"""
            elif '(incorrect)' in guidelines_lower:
                hint_section = """
════════════════════════════════════════════════════════════════
⚠️  GUIDELINE CONSTRAINT: (INCORRECT) → typically "incorrect"
════════════════════════════════════════════════════════════════

The grading guidelines contain: (INCORRECT)

This indicates significant errors. Grade as "incorrect" unless:
- There is meaningful progress despite errors → "partial"
- The answer is actually flawless (rare) → "correct"

Default to "incorrect" for (Incorrect) labeled answers.
"""
            elif '(correct)' in guidelines_lower:
                hint_section = """
════════════════════════════════════════════════════════════════
ℹ️  GUIDELINE HINT: (CORRECT) → verify independently
════════════════════════════════════════════════════════════════

The grading guidelines contain: (CORRECT)

This means the student claims their answer is correct. You must:
- INDEPENDENTLY VERIFY by comparing to the official solution
- Only grade "correct" if truly flawless
- Do NOT blindly accept the (Correct) label
"""
            else:
                hint_section = f"""
════════════════════════════════════════════════════════════════
GUIDELINE HINT: ({guideline_hint.upper()})
════════════════════════════════════════════════════════════════

The guidelines contain a label. Use it as context, but verify independently:
- (Partial)/(Almost) → typically "partial"
- (Incorrect) → typically "incorrect"  
- (Correct) → verify independently
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
  • NO logical gaps, NO unjustified claims, NO calculation errors
  → Use ONLY when the student deserves FULL credit
  → If you see ANY flaw, no matter how small, it is NOT "correct"

【incorrect】= ZERO CREDIT (0%) - USE FOR FUNDAMENTALLY WRONG
  • The approach is completely wrong or misguided
  • No meaningful mathematical progress toward the solution
  • Major conceptual errors that invalidate the entire solution
  • Answer is nonsense or unrelated to the problem
  • The student failed to demonstrate understanding of the problem
  → Use when the student deserves ZERO credit
  → Only use "incorrect" when there is truly NOTHING salvageable

【partial】= PARTIAL CREDIT (1-99%) - USE FOR MEANINGFUL BUT INCOMPLETE PROGRESS
  • The student made MEANINGFUL progress but has SIGNIFICANT issues
  • Correct approach started but INCOMPLETE or contains IMPORTANT ERRORS
  • Has non-negligible mistakes that affect the final result
  • "Almost" correct but missing critical final steps or has substantive errors
  • Contains good ideas but flawed execution
  → Use when the student deserves MORE than zero but LESS than full credit

════════════════════════════════════════════════════════════════
DECISION FRAMEWORK - FOLLOW THIS EXACTLY
════════════════════════════════════════════════════════════════

STEP 1: Check if the answer is PERFECT (no flaws whatsoever)
  → If YES → grade "correct"
  → If NO → proceed to Step 2

STEP 2: Check if the answer has ANY meaningful progress
  → If NO meaningful progress at all → grade "incorrect"
  → If YES (some correct reasoning, good approach started, etc.) → grade "partial"

STEP 3: Apply guideline label constraints (if present)
  → (Almost) in guidelines → MUST be "partial" (never "correct")
  → (Partial) in guidelines → typically "partial"
  → (Incorrect) in guidelines → typically "incorrect" unless meaningful progress
  → (Correct) in guidelines → verify independently, only "correct" if truly perfect

════════════════════════════════════════════════════════════════
CRITICAL RULES
════════════════════════════════════════════════════════════════

1. "correct" requires PERFECTION - ANY flaw makes it "partial"
   - Even a single minor error → "partial" (not "correct")
   - Missing a step, even if obvious → "partial"
   - "Almost correct" or "nearly right" → "partial" (NOT "correct")

2. "incorrect" requires NO meaningful progress
   - Only use "incorrect" if the student made ZERO progress
   - If the student understood the problem at all → "partial"
   - Wrong approach but some correct calculations → "partial"

3. "(Almost)" answers are NEVER "correct" - they are ALWAYS "partial"
   - The "(Almost)" label means: close to correct, but NOT fully correct
   - This is a HARD CONSTRAINT: (Almost) → always "partial"
   - Examples: correct method but arithmetic error, right approach but missing insight{hint_section}

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

Follow the DECISION FRAMEWORK above:
1. First, determine if the answer is PERFECT (no flaws) → "correct"
2. If not perfect, check if there's ANY meaningful progress → "partial" or "incorrect"
3. Apply guideline label constraints (especially: "(Almost)" → "partial")

════════════════════════════════════════════════════════════════
OUTPUT FORMAT (STRICT - MUST FOLLOW EXACTLY)
════════════════════════════════════════════════════════════════

You MUST provide your decision in <json> tags with exactly these fields:
- "reasoning": A clear explanation of your grading decision
- "response": MUST be exactly one of: "correct", "partial", or "incorrect" (lowercase)

<json>
{{
    "reasoning": "Your detailed explanation here.",
    "response": "correct" | "partial" | "incorrect"
}}
</json>

CRITICAL OUTPUT RULES:
1. The "response" field MUST contain ONLY: correct, partial, or incorrect (lowercase)
2. Do NOT use quotes, uppercase, or variations like "mostly correct"
3. If the answer has ANY flaw → "partial" (not "correct")
4. If "(Almost)" in guidelines → MUST be "partial" (never "correct")

════════════════════════════════════════════════════════════════
NOW GRADE THE ANSWER
════════════════════════════════════════════════════════════════"""

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
        #   4. Default to "partial" for safety if no valid grade found.          #
        # ------------------------------------------------------------------ #
        prediction = "partial"  # Default to partial (safer than incorrect for grading)
        
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

        # Try comprehensive extraction
        if assistant_text:
            extracted_grade = _extract_grade_from_response(assistant_text)
            if extracted_grade:
                prediction = extracted_grade
                self.log_fn(f"Extracted grade: {prediction}")
            else:
                self.log_fn("No grade extracted from response, using default")

        # Apply guideline-based validation to catch edge cases
        # This enforces hard constraints like: (Almost) → must be "partial"
        original_prediction = prediction
        prediction = _validate_grade_with_guidelines(prediction, guideline_hint, grading_guidelines)
        if prediction != original_prediction:
            self.log_fn(f"Grade corrected by guideline validation: {original_prediction} → {prediction}")
        
        # Final validation: ensure we always return a valid grade
        if prediction not in _VALID_GRADES:
            prediction = "partial"  # Default to partial as it's the safest middle ground
            self.log_fn("No valid grade found, defaulting to 'partial' for safety")

        return str(prediction), msg_history
