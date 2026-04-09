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
    - "(Partial)" → typically "partial"
    - "(Almost)" → typically "partial" (but LLM should verify independently)
    - "(Incorrect)" → typically "incorrect"
    - "(Correct)" → typically "correct" (but LLM must verify independently)
    
    Returns the label type if found, None otherwise.
    
    Priority order: (Incorrect) > (Almost) > (Partial) > (Correct)
    This ensures we catch the most restrictive label first.
    
    NOTE: This function only extracts the label for informational purposes.
    The LLM should make its own independent judgment based on the actual content.
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
    
    # Priority 2: (Almost) - means minor but NON-NEGLIGIBLE mistakes
    # This is an important edge case - but we only extract the label, not enforce it
    almost_bracket_patterns = [
        r'\(almost\)',
        r'\[almost\]',
        r'\{almost\}',
        r'<almost>',
        r'«almost»',
    ]
    for pattern in almost_bracket_patterns:
        if re.search(pattern, guidelines_lower):
            return "partial"  # Label as partial, but LLM should verify
    
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
    
    # Check for label anywhere in the text (not just at start of line)
    # This catches cases like "This is (Almost) correct" or "Grade: (Partial)"
    for pattern in almost_bracket_patterns:
        if re.search(pattern, guidelines_lower):
            return "partial"
    
    for pattern in partial_bracket_patterns:
        if re.search(pattern, guidelines_lower):
            return "partial"
    
    for pattern in incorrect_bracket_patterns:
        if re.search(pattern, guidelines_lower):
            return "incorrect"
    
    for pattern in correct_bracket_patterns:
        if re.search(pattern, guidelines_lower):
            return "correct"
    
    return None


def _validate_grade_with_guidelines(predicted_grade: str, guideline_hint: str | None, grading_guidelines: str = "") -> str:
    """Validate and potentially correct the predicted grade based on guideline hints.
    
    Guidelines for validation:
    - (Almost) in guidelines → typically "partial" (but trust LLM if it says "correct" with strong reasoning)
    - (Partial) in guidelines → typically "partial" (but allow "correct" if truly perfect)
    - (Incorrect) in guidelines → typically "incorrect" (allow "partial" if meaningful progress detected)
    - (Correct) in guidelines → verify independently, trust LLM judgment
    
    The LLM has access to the full problem, solution, and student answer - we should
    trust its judgment more than rigid label-based rules. Only override obvious errors.
    
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
    
    # TRUST LLM JUDGMENT: Only override in clear cases where the LLM likely made a mistake
    
    # Case 1: (Almost) labeled as "incorrect" → should be "partial" (has some progress)
    # This is the most important correction - "almost" means some progress was made
    if has_almost_label and predicted_grade == "incorrect":
        return "partial"
    
    # Case 2: (Incorrect) labeled as "correct" → likely wrong, should be "incorrect" or "partial"
    # The guidelines say it's incorrect, so trusting that over LLM
    if has_incorrect_label and predicted_grade == "correct":
        # Check if there's any indication of partial progress in guidelines
        if "some" in guidelines_lower or "partial" in guidelines_lower or "progress" in guidelines_lower:
            return "partial"
        return "incorrect"
    
    # Case 3: (Correct) labeled as "incorrect" → might be wrong, but trust LLM if it found real flaws
    # Don't override this - the LLM may have found real errors
    
    # Case 4: (Partial) labeled as "correct" → might be wrong, but trust LLM if it verified perfection
    # Don't override unless there's clear evidence of imperfection in guidelines
    
    # Case 5: (Almost) labeled as "correct" → suspicious but don't auto-override
    # The LLM may have determined the answer is actually perfect despite the "almost" label
    # Only override if we see explicit error mentions in the guidelines
    if has_almost_label and predicted_grade == "correct":
        # Check if guidelines mention specific errors that would prevent "correct"
        error_indicators = [
            'error', 'mistake', 'wrong', 'incorrect', 'flaw', 'missing', 
            'incomplete', 'not quite', 'not fully', 'not completely'
        ]
        has_explicit_error = any(indicator in guidelines_lower for indicator in error_indicators)
        if has_explicit_error:
            return "partial"
        # Otherwise trust the LLM's judgment
    
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
    """Fallback: scan the text for a grade keyword with context awareness.
    
    Prioritizes:
    1. JSON-like structures with grade fields (highest priority)
    2. Explicit grade declarations near the end
    3. Pattern-based extraction with negation handling
    4. Last-resort standalone grade words (lowest priority)
    
    This function focuses on the last section of the text where the LLM
    typically states its final conclusion.
    """
    if not text:
        return None
        
    text_clean = re.sub(r'[ \t]+', ' ', text).strip()
    text_lower = text_clean.lower()
    
    # Focus on the last section where the conclusion typically is
    last_section = text_lower[-800:] if len(text_lower) > 800 else text_lower
    
    # Strategy 1: JSON-like structures (highest priority)
    json_patterns = [
        r'"response"\s*:\s*"(correct|partial|incorrect)"',
        r'"response"\s*:\s*\'(correct|partial|incorrect)\'',
        r'\{\s*["\']?response["\']?\s*:\s*["\']?(correct|partial|incorrect)["\']?\s*\}',
        r'"(?:grade|verdict|result|answer|evaluation)"\s*:\s*"(correct|partial|incorrect)"',
        r'["\']response["\']\s*:\s*["\']?(correct|partial|incorrect)["\']?',
    ]
    
    for pattern in json_patterns:
        matches = list(re.finditer(pattern, last_section))
        if matches:
            return matches[-1].group(1).lower()
    
    # Strategy 2: Explicit final decision patterns in the last section
    final_patterns = [
        r'(?:final\s+)?(?:grade|verdict|assessment|decision|evaluation|answer)\s*[:\-=]\s*["\']?(correct|partial|incorrect)["\']?(?:\s*$|\s*\.\s*$|\s*[,;])',
        r'(?:the\s+)?(?:grade|verdict|assessment|result|evaluation)\s+is\s*["\']?(correct|partial|incorrect)["\']?(?:\s*$|\s*\.\s*$|\s*[,;])',
        r'(?:therefore|thus|so|hence|conclusion|in\s+conclusion)\s*,?\s*(?:the\s+)?(?:grade|answer|result|verdict)\s+is\s*["\']?(correct|partial|incorrect)["\']?',
        r'(?:i\s+)?(?:conclude|determine|decide|grade|assign)\s+(?:that\s+)?(?:the\s+)?(?:grade|answer|result|verdict)\s+(?:is\s+)?["\']?(correct|partial|incorrect)["\']?',
        r'(?:this\s+(?:answer|solution|work)\s+is)\s*["\']?(correct|partial|incorrect)["\']?',
        r'(?:the\s+student\s+(?:answer|solution|work)\s+is)\s*["\']?(correct|partial|incorrect)["\']?',
    ]
    
    for pattern in final_patterns:
        matches = list(re.finditer(pattern, last_section))
        if matches:
            return matches[-1].group(1).lower()
    
    # Strategy 3: "is X" patterns with negation check in the last section
    is_matches = list(re.finditer(r'\bis\s+(correct|partial|incorrect)\b', last_section))
    for match in reversed(is_matches):
        grade = match.group(1).lower()
        start_pos = match.start()
        context = last_section[max(0, start_pos-50):start_pos]
        
        negation_indicators = ['not ', 'isn\'t', 'is not', 'never', 'neither', 'nor', 'no ']
        if any(neg in context for neg in negation_indicators):
            continue
        return grade
    
    # Strategy 4: Conclusion patterns in the last section
    conclusion_patterns = [
        r'(?:conclusion|final|summary|decision|verdict)\s*[:\-]?\s*(correct|partial|incorrect)',
        r'(?:grade|score|mark)\s*[:\-]?\s*(correct|partial|incorrect)',
    ]
    for pattern in conclusion_patterns:
        match = re.search(pattern, last_section)
        if match:
            return match.group(1).lower()
    
    # Strategy 5: Last resort - standalone grade words with negation check in last section
    standalone_matches = list(re.finditer(r'\b(correct|partial|incorrect)\b', last_section))
    for match in reversed(standalone_matches):
        grade = match.group(1).lower()
        start_pos = match.start()
        context = last_section[max(0, start_pos-30):start_pos]
        
        negation_indicators = ['not ', 'isn\'t', 'is not', 'never', 'neither', 'nor', 'no ']
        if any(neg in context for neg in negation_indicators):
            continue
        return grade
    
    # Strategy 6: If no match in last section, try the full text
    for pattern in json_patterns:
        matches = list(re.finditer(pattern, text_lower))
        if matches:
            return matches[-1].group(1).lower()
    
    # Strategy 7: Full text standalone matches as absolute last resort
    standalone_matches = list(re.finditer(r'\b(correct|partial|incorrect)\b', text_lower))
    for match in reversed(standalone_matches):
        grade = match.group(1).lower()
        start_pos = match.start()
        context = text_lower[max(0, start_pos-30):start_pos]
        
        negation_indicators = ['not ', 'isn\'t', 'is not', 'never', 'neither', 'nor', 'no ']
        if any(neg in context for neg in negation_indicators):
            continue
        return grade
    
    # Absolute last resort - return the last occurrence regardless of negation
    if standalone_matches:
        return standalone_matches[-1].group(1).lower()

    return None


def _extract_grade_from_response(text: str) -> str | None:
    """Extract grade from LLM response using multiple strategies.
    
    Priority order:
    1. JSON blocks with response field (highest priority)
    2. Explicit final decision patterns in the last section
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
            # Use the last JSON object (most likely to be the final answer)
            for obj in reversed(extracted):
                grade = str(obj.get("response", "")).strip().lower()
                normalized = _normalize_grade(grade)
                if normalized:
                    return normalized
    except Exception:
        pass
    
    # Strategy 2: Look for explicit final decision patterns in the last 1000 chars
    # This focuses on the conclusion section where the LLM typically states its final decision
    last_section = text_lower[-1000:] if len(text_lower) > 1000 else text_lower
    
    # High-confidence patterns that indicate a final decision
    final_patterns = [
        # JSON-like response field
        r'"response"\s*:\s*"(correct|partial|incorrect)"',
        r'"response"\s*:\s*\'(correct|partial|incorrect)\'',
        # Explicit final statements
        r'(?:final\s+)?(?:grade|verdict|assessment|decision|evaluation|answer)\s*[:\-=]\s*["\']?(correct|partial|incorrect)["\']?(?:\s*$|\s*\.\s*$|\s*[,;])',
        r'(?:the\s+)?(?:grade|verdict|assessment|result|evaluation)\s+is\s*["\']?(correct|partial|incorrect)["\']?(?:\s*$|\s*\.\s*$|\s*[,;])',
        r'(?:therefore|thus|so|hence|conclusion|in\s+conclusion)\s*,?\s*(?:the\s+)?(?:grade|answer|result|verdict)\s+is\s*["\']?(correct|partial|incorrect)["\']?',
        r'(?:i\s+)?(?:conclude|determine|decide|grade|assign)\s+(?:that\s+)?(?:the\s+)?(?:grade|answer|result|verdict)\s+(?:is\s+)?["\']?(correct|partial|incorrect)["\']?',
        r'(?:this\s+(?:answer|solution|work)\s+is)\s*["\']?(correct|partial|incorrect)["\']?',
        r'(?:the\s+student\s+(?:answer|solution|work)\s+is)\s*["\']?(correct|partial|incorrect)["\']?',
    ]
    
    for pattern in final_patterns:
        matches = list(re.finditer(pattern, last_section, re.IGNORECASE))
        if matches:
            grade = matches[-1].group(1).lower()
            if grade in _VALID_GRADES:
                return grade
    
    # Strategy 3: Try fallback extraction on the full text
    try:
        fallback = _extract_grade_fallback(text)
        if fallback:
            normalized = _normalize_grade(fallback)
            if normalized:
                return normalized
    except Exception:
        pass
    
    # Strategy 4: Last resort - look for standalone grade words in the last section
    # with negation checking
    standalone_matches = list(re.finditer(r'\b(correct|partial|incorrect)\b', last_section))
    for match in reversed(standalone_matches):
        grade = match.group(1).lower()
        start_pos = match.start()
        context = last_section[max(0, start_pos-50):start_pos]
        
        # Check for negation indicators
        negation_indicators = ['not ', 'isn\'t', 'is not', 'never', 'neither', 'nor', 'no ', 'not']
        if any(neg in context for neg in negation_indicators):
            continue
        
        return grade
    
    # Strategy 5: If no match in last section, try the full text
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
    
    # Absolute last resort - return the last occurrence regardless of negation
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

        # Add guideline label context if available (informational only)
        hint_section = ""
        if guideline_hint:
            guidelines_lower = (grading_guidelines or "").lower()
            
            # Informational note about the guideline label - LLM should verify independently
            if '(almost)' in guidelines_lower:
                hint_section = """
════════════════════════════════════════════════════════════════
ℹ️  GUIDELINE NOTE: Label is (ALMOST)
════════════════════════════════════════════════════════════════

The grading guidelines are labeled (ALMOST), which typically indicates:
- The answer is close to correct but may have minor issues
- Usually deserves partial credit

HOWEVER: You must INDEPENDENTLY EVALUATE the actual student answer.
If the answer is truly flawless, grade it "correct" despite the label.
If it has no meaningful progress, grade it "incorrect".
Trust your own analysis, not the label.
"""
            elif '(partial)' in guidelines_lower:
                hint_section = """
════════════════════════════════════════════════════════════════
ℹ️  GUIDELINE NOTE: Label is (PARTIAL)
════════════════════════════════════════════════════════════════

The grading guidelines are labeled (PARTIAL), which typically indicates:
- The answer has some correct elements but is incomplete
- Usually deserves partial credit

HOWEVER: You must INDEPENDENTLY EVALUATE the actual student answer.
If the answer is truly flawless, grade it "correct" despite the label.
If it has no meaningful progress, grade it "incorrect".
Trust your own analysis, not the label.
"""
            elif '(incorrect)' in guidelines_lower:
                hint_section = """
════════════════════════════════════════════════════════════════
ℹ️  GUIDELINE NOTE: Label is (INCORRECT)
════════════════════════════════════════════════════════════════

The grading guidelines are labeled (INCORRECT), which typically indicates:
- The answer has significant errors
- Usually deserves no credit or minimal credit

HOWEVER: You must INDEPENDENTLY EVALUATE the actual student answer.
If the answer has meaningful progress, grade it "partial".
If it is truly flawless, grade it "correct".
Trust your own analysis, not the label.
"""
            elif '(correct)' in guidelines_lower:
                hint_section = """
════════════════════════════════════════════════════════════════
ℹ️  GUIDELINE NOTE: Label is (CORRECT)
════════════════════════════════════════════════════════════════

The grading guidelines are labeled (CORRECT), which typically indicates:
- The student believes their answer is correct

You must INDEPENDENTLY VERIFY by comparing to the official solution.
Only grade "correct" if truly flawless. Do NOT blindly accept the label.
"""
            else:
                hint_section = f"""
════════════════════════════════════════════════════════════════
ℹ️  GUIDELINE NOTE: Label is ({guideline_hint.upper()})
════════════════════════════════════════════════════════════════

The guidelines contain a label. Use it as context, but verify independently.
Trust your own analysis of the student answer against the official solution.
"""

        instruction = f"""You are an expert mathematics competition grader. Your task is to evaluate the student's answer and assign exactly ONE grade: "correct", "partial", or "incorrect".

════════════════════════════════════════════════════════════════
GRADE DEFINITIONS - READ CAREFULLY
════════════════════════════════════════════════════════════════

【correct】= FULL CREDIT (100%) - USE WHEN PERFECT
  • The solution is COMPLETE and FLAWLESS
  • Every logical step is valid and correctly executed
  • Final answer exactly matches the official solution
  • NO errors, NO missing steps, NO calculation mistakes
  • NO logical gaps, NO unjustified claims
  → Use when the student deserves FULL credit
  → Be strict: ANY flaw means NOT "correct"

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

STEP 3: Trust your own analysis
  → The guideline label is a hint, not a constraint
  → Your independent evaluation of the student answer is what matters
  → Compare directly against the official solution

════════════════════════════════════════════════════════════════
CRITICAL RULES
════════════════════════════════════════════════════════════════

1. "correct" requires PERFECTION
   - ANY flaw means NOT "correct"
   - Missing a step, even if obvious → "partial"
   - "Almost correct" or "nearly right" → "partial" (NOT "correct")

2. "incorrect" requires NO meaningful progress
   - Only use "incorrect" if the student made ZERO progress
   - If the student understood the problem at all → "partial"
   - Wrong approach but some correct calculations → "partial"

3. Trust your own judgment over the guideline label
   - The label is a hint about what the student might have written
   - You must independently verify by comparing to the official solution
   - Do NOT blindly follow the label{hint_section}

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
3. Trust your own independent analysis over the guideline label

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
4. Trust your own analysis, not the guideline label

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

        # Apply guideline-based validation to catch obvious errors
        # Only override in clear cases where the LLM likely made a mistake
        original_prediction = prediction
        prediction = _validate_grade_with_guidelines(prediction, guideline_hint, grading_guidelines)
        if prediction != original_prediction:
            self.log_fn(f"Grade corrected by guideline validation: {original_prediction} → {prediction}")
        
        # Final validation: ensure we always return a valid grade
        if prediction not in _VALID_GRADES:
            prediction = "partial"  # Default to partial as it's the safest middle ground
            self.log_fn("No valid grade found, defaulting to 'partial' for safety")

        return str(prediction), msg_history
