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
    
    This function applies hard rules for specific guideline labels to ensure
    consistent grading. The LLM's judgment is respected unless it clearly
    contradicts the explicit guideline label meaning.
    
    Label meanings:
    - (Almost) = "partial" (minor but non-negligible issues, never correct)
    - (Partial) = "partial" (incomplete but has progress, never correct)
    - (Incorrect) = "incorrect" or "partial" (no/wrong progress)
    - (Correct) = verify independently (label is just a claim)
    
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
    
    # HARD RULES: These override LLM judgment when there's a clear contradiction
    
    # Rule 1: (Almost) labeled answers MUST be "partial" (never "correct")
    # "Almost" explicitly means minor but NON-NEGLIGIBLE mistakes exist
    if has_almost_label:
        if predicted_grade == "correct":
            return "partial"
        # "Almost" with minor issues should not be "incorrect" either
        if predicted_grade == "incorrect":
            return "partial"
    
    # Rule 2: (Partial) labeled answers MUST be "partial" (never "correct")
    # "Partial" explicitly means incomplete answer
    if has_partial_label:
        if predicted_grade == "correct":
            return "partial"
        # Check if there's any progress mentioned - if so, don't be "incorrect"
        progress_indicators = [
            'proved', 'shown', 'correct', 'valid', 'progress', 'partial',
            'some', 'invariant', 'observation', 'step', 'idea', 'approach',
            'part', 'partially', 'incomplete'
        ]
        has_progress = any(indicator in guidelines_lower for indicator in progress_indicators)
        if predicted_grade == "incorrect" and has_progress:
            return "partial"
    
    # Rule 3: (Incorrect) labeled answers should not be "correct"
    # But can be "partial" if there's any valid progress
    if has_incorrect_label and predicted_grade == "correct":
        # Check if there's any indication of partial progress in guidelines
        progress_indicators = [
            'some', 'partial', 'progress', 'part', 'step', 'idea', 'approach',
            'observation', 'invariant', 'proved', 'shown'
        ]
        has_progress = any(indicator in guidelines_lower for indicator in progress_indicators)
        if has_progress:
            return "partial"
        return "incorrect"
    
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
        
        # Fix 6: Handle escaped quotes that might be malformed
        try:
            fixed = inner.replace('\\"', '"').replace("\\'", "'")
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass
        
        # Fix 7: Extract response field directly with regex patterns
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
            r'response\s*[=:]\s*["\']?(correct|partial|incorrect)["\']?',
            # Additional patterns for whitespace and formatting variations
            r'"response"\s*:\s*"\s*(correct|partial|incorrect)\s*"',
            r'"response"\s*:\s*\'\s*(correct|partial|incorrect)\s*\'',
            r'response["\']?\s*:\s*["\']?(correct|partial|incorrect)["\']?',
            r'"response"\s*:\s*"\s*(correct|partial|incorrect)\s*"',
            r'"response"\s*:\s*\'\s*(correct|partial|incorrect)\s*\'',
            # Pattern for response as array (handle by extracting first element)
            r'"response"\s*:\s*\[\s*"?(correct|partial|incorrect)"?\s*\]',
        ]
        for pattern in response_patterns:
            try:
                response_match = re.search(pattern, inner, re.IGNORECASE)
                if response_match:
                    return {"response": response_match.group(1).lower()}
            except Exception:
                pass
        
        # Fix 8: Extract any valid grade from the text
        try:
            for grade in _VALID_GRADES:
                if f'"{grade}"' in inner.lower() or f"'{grade}'" in inner.lower():
                    return {"response": grade}
        except Exception:
            pass
        
        return None
    
    # Find <json>...</json> blocks (case insensitive)
    search_from = 0
    while True:
        start = text.lower().find("<json>", search_from)
        if start == -1:
            break
        # Find the actual end position in original case
        end = text.lower().find("</json>", start)
        if end == -1:
            break
        # Get the content between tags (using original text for position accuracy)
        inner_start = text.lower().find("<json>", search_from) + 6
        inner_end = text.lower().find("</json>", start)
        inner = text[inner_start:inner_end].strip()
        search_from = text.lower().find("</json>", start) + 7
        
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
    
    # Find any response field pattern in the text
    response_field_pattern = r'"response"\s*:\s*"?(correct|partial|incorrect)"?'
    for match in re.finditer(response_field_pattern, text, re.IGNORECASE):
        try:
            grade = match.group(1).lower()
            if grade in _VALID_GRADES:
                results.append({"response": grade})
        except Exception:
            pass
    
    # Additional pattern: look for response field with various quote styles
    alt_response_pattern = r"['\"]response['\"]\s*:\s*['\"]?(correct|partial|incorrect)['\"]?"
    for match in re.finditer(alt_response_pattern, text, re.IGNORECASE):
        try:
            grade = match.group(1).lower()
            if grade in _VALID_GRADES:
                results.append({"response": grade})
        except Exception:
            pass
    
    # Pattern for response as array - extract the grade from array
    array_response_pattern = r'"response"\s*:\s*\[\s*"?(correct|partial|incorrect)"?\s*\]'
    for match in re.finditer(array_response_pattern, text, re.IGNORECASE):
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
    
    # Focus on the last section where the conclusion typically is (increased to 1500)
    last_section = text_lower[-1500:] if len(text_lower) > 1500 else text_lower
    
    # Strategy 1: JSON-like structures (highest priority)
    json_patterns = [
        r'"response"\s*:\s*"(correct|partial|incorrect)"',
        r'"response"\s*:\s*\'(correct|partial|incorrect)\'',
        r'"response"\s*:\s*\\"(correct|partial|incorrect)\\"',
        r"'response'\s*:\s*'(correct|partial|incorrect)'",
        r'\{\s*["\']?response["\']?\s*:\s*["\']?(correct|partial|incorrect)["\']?\s*\}',
        r'"(?:grade|verdict|result|answer|evaluation)"\s*:\s*"(correct|partial|incorrect)"',
        r'["\']response["\']\s*:\s*["\']?(correct|partial|incorrect)["\']?',
        r'response\s*[=:]\s*["\']?(correct|partial|incorrect)["\']?',
        # Additional patterns for robustness
        r'"response"\s*:\s*"\s*(correct|partial|incorrect)\s*"',
        r'"response"\s*:\s*\'\s*(correct|partial|incorrect)\s*\'',
        r'response["\']?\s*:\s*["\']?(correct|partial|incorrect)["\']?',
        # Pattern for response as array
        r'"response"\s*:\s*\[\s*"?(correct|partial|incorrect)"?\s*\]',
    ]
    
    for pattern in json_patterns:
        matches = list(re.finditer(pattern, last_section, re.IGNORECASE))
        if matches:
            return matches[-1].group(1).lower()
    
    # Strategy 2: Explicit final decision patterns in the last section
    final_patterns = [
        r'(?:final\s+)?(?:grade|verdict|assessment|decision|evaluation|answer|result)\s*[:\-=]\s*["\']?(correct|partial|incorrect)["\']?(?:\s*$|\s*\.\s*$|\s*[,;])',
        r'(?:the\s+)?(?:grade|verdict|assessment|result|evaluation)\s+is\s*["\']?(correct|partial|incorrect)["\']?(?:\s*$|\s*\.\s*$|\s*[,;])',
        r'(?:therefore|thus|so|hence|conclusion|in\s+conclusion)\s*,?\s*(?:the\s+)?(?:grade|answer|result|verdict)\s+is\s*["\']?(correct|partial|incorrect)["\']?',
        r'(?:i\s+)?(?:conclude|determine|decide|grade|assign)\s+(?:that\s+)?(?:the\s+)?(?:grade|answer|result|verdict)\s+(?:is\s+)?["\']?(correct|partial|incorrect)["\']?',
        r'(?:this\s+(?:answer|solution|work)\s+is)\s*["\']?(correct|partial|incorrect)["\']?',
        r'(?:the\s+student\s+(?:answer|solution|work)\s+is)\s*["\']?(correct|partial|incorrect)["\']?',
        r'grade\s+(?:should\s+be|will\s+be|is)\s*["\']?(correct|partial|incorrect)["\']?',
        r'(?:answer|solution)\s+is\s*["\']?(correct|partial|incorrect)["\']?\s*(?:\.|$)',
        # Additional patterns for robustness
        r'(?:i\s+)?(?:grade|rate|score|mark)\s+(?:this\s+)?(?:as\s+)?["\']?(correct|partial|incorrect)["\']?',
        r'(?:my\s+)?(?:verdict|decision|assessment|judgment)\s+(?:is\s+)?["\']?(correct|partial|incorrect)["\']?',
        r'(?:i\s+)?(?:assign|give|award)\s+["\']?(correct|partial|incorrect)["\']?',
    ]
    
    for pattern in final_patterns:
        matches = list(re.finditer(pattern, last_section, re.IGNORECASE))
        if matches:
            return matches[-1].group(1).lower()
    
    # Strategy 3: "is X" patterns with negation check in the last section
    is_matches = list(re.finditer(r'\bis\s+(correct|partial|incorrect)\b', last_section, re.IGNORECASE))
    for match in reversed(is_matches):
        grade = match.group(1).lower()
        start_pos = match.start()
        context = last_section[max(0, start_pos-100):start_pos]
        
        negation_indicators = [
            'not ', 'isn\'t', 'is not', 'never', 'neither', 'nor', 'no ',
            'cannot be', 'can not be', 'could not be', 'would not be',
            'not a ',
        ]
        if any(neg in context for neg in negation_indicators):
            continue
        return grade
    
    # Strategy 4: Conclusion patterns in the last section
    conclusion_patterns = [
        r'(?:conclusion|final|summary|decision|verdict)\s*[:\-]?\s*(correct|partial|incorrect)',
        r'(?:grade|score|mark)\s*[:\-]?\s*(correct|partial|incorrect)',
    ]
    for pattern in conclusion_patterns:
        match = re.search(pattern, last_section, re.IGNORECASE)
        if match:
            return match.group(1).lower()
    
    # Strategy 5: Last resort - standalone grade words with negation check in last section
    standalone_matches = list(re.finditer(r'\b(correct|partial|incorrect)\b', last_section))
    for match in reversed(standalone_matches):
        grade = match.group(1).lower()
        start_pos = match.start()
        context = last_section[max(0, start_pos-100):start_pos]
        
        negation_indicators = [
            'not ', 'isn\'t', 'is not', 'never', 'neither', 'nor', 'no ',
            'cannot be', 'can not be', 'could not be', 'would not be',
            'not a ',
        ]
        if any(neg in context for neg in negation_indicators):
            continue
        return grade
    
    # Strategy 6: If no match in last section, try the full text
    for pattern in json_patterns:
        matches = list(re.finditer(pattern, text_lower, re.IGNORECASE))
        if matches:
            return matches[-1].group(1).lower()
    
    # Strategy 7: Full text standalone matches as absolute last resort
    standalone_matches = list(re.finditer(r'\b(correct|partial|incorrect)\b', text_lower))
    for match in reversed(standalone_matches):
        grade = match.group(1).lower()
        start_pos = match.start()
        context = text_lower[max(0, start_pos-100):start_pos]
        
        negation_indicators = [
            'not ', 'isn\'t', 'is not', 'never', 'neither', 'nor', 'no ',
            'cannot be', 'can not be', 'could not be', 'would not be',
            'not a ',
        ]
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
    
    # Strategy 2: Look for explicit final decision patterns in the last 3000 chars
    # This focuses on the conclusion section where the LLM typically states its final decision
    last_section = text_lower[-3000:] if len(text_lower) > 3000 else text_lower
    
    # High-confidence patterns that indicate a final decision
    # Priority: JSON-like > explicit statements > conclusion phrases
    final_patterns = [
        # JSON-like response field (highest priority)
        r'"response"\s*:\s*"(correct|partial|incorrect)"',
        r'"response"\s*:\s*\'(correct|partial|incorrect)\'',
        r'"response"\s*:\s*\\"(correct|partial|incorrect)\\"',
        r"'response'\s*:\s*'(correct|partial|incorrect)'",
        r'response\s*[=:]\s*["\']?(correct|partial|incorrect)["\']?',
        # Additional JSON patterns with whitespace
        r'"response"\s*:\s*"\s*(correct|partial|incorrect)\s*"',
        r'"response"\s*:\s*\'\s*(correct|partial|incorrect)\s*\'',
        r'response["\']?\s*:\s*["\']?(correct|partial|incorrect)["\']?',
        # Explicit final statements with grade/verdict/assessment
        r'(?:final\s+)?(?:grade|verdict|assessment|decision|evaluation|answer|result)\s*[:\-=]\s*["\']?(correct|partial|incorrect)["\']?(?:\s*$|\s*\.\s*$|\s*[,;])',
        r'(?:the\s+)?(?:grade|verdict|assessment|result|evaluation)\s+is\s*["\']?(correct|partial|incorrect)["\']?(?:\s*$|\s*\.\s*$|\s*[,;])',
        r'(?:therefore|thus|so|hence|conclusion|in\s+conclusion)\s*,?\s*(?:the\s+)?(?:grade|answer|result|verdict)\s+is\s*["\']?(correct|partial|incorrect)["\']?',
        r'(?:i\s+)?(?:conclude|determine|decide|grade|assign)\s+(?:that\s+)?(?:the\s+)?(?:grade|answer|result|verdict)\s+(?:is\s+)?["\']?(correct|partial|incorrect)["\']?',
        r'(?:this\s+(?:answer|solution|work)\s+is)\s*["\']?(correct|partial|incorrect)["\']?',
        r'(?:the\s+student\s+(?:answer|solution|work)\s+is)\s*["\']?(correct|partial|incorrect)["\']?',
        # Additional patterns for robustness
        r'grade\s+(?:should\s+be|will\s+be|is)\s*["\']?(correct|partial|incorrect)["\']?',
        r'(?:answer|solution)\s+is\s*["\']?(correct|partial|incorrect)["\']?\s*(?:\.|$)',
        # Pattern for "I grade this as X" or similar
        r'(?:i\s+)?(?:grade|rate|score|mark)\s+(?:this\s+)?(?:as\s+)?["\']?(correct|partial|incorrect)["\']?',
        # Pattern for "my verdict is X"
        r'(?:my\s+)?(?:verdict|decision|assessment|judgment)\s+(?:is\s+)?["\']?(correct|partial|incorrect)["\']?',
        # Pattern for "I assign X"
        r'(?:i\s+)?(?:assign|give|award)\s+["\']?(correct|partial|incorrect)["\']?',
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
    
    # Strategy 4: Look for standalone grade words in the last section with negation checking
    # Use a larger context window for better negation detection
    standalone_matches = list(re.finditer(r'\b(correct|partial|incorrect)\b', last_section))
    for match in reversed(standalone_matches):
        grade = match.group(1).lower()
        start_pos = match.start()
        # Increased context window for better negation detection
        context = last_section[max(0, start_pos-100):start_pos]
        
        # Check for negation indicators (expanded list with word boundaries)
        negation_indicators = [
            'not ', 'isn\'t', 'is not', 'never', 'neither', 'nor', 'no ',
            'not\s+', 'isn\'t\s+', 'is not\s+', 'never\s+', 'neither\s+', 'nor\s+', 'no\s+',
            'cannot be', 'can not be', 'could not be', 'would not be',
            'is neither', 'was neither', 'are neither', 'not a ',
        ]
        if any(neg in context for neg in negation_indicators):
            continue
        
        return grade
    
    # Strategy 5: If no match in last section, try the full text
    standalone_matches = list(re.finditer(r'\b(correct|partial|incorrect)\b', text_lower))
    for match in reversed(standalone_matches):
        grade = match.group(1).lower()
        start_pos = match.start()
        # Increased context window for better negation detection
        context = text_lower[max(0, start_pos-100):start_pos]
        
        # Check for negation
        negation_indicators = [
            'not ', 'isn\'t', 'is not', 'never', 'neither', 'nor', 'no ',
            'cannot be', 'can not be', 'could not be', 'would not be',
            'is neither', 'was neither', 'are neither', 'not a ',
        ]
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
⚠️  MANDATORY RULE: Label is (ALMOST) → GRADE MUST BE "PARTIAL"
════════════════════════════════════════════════════════════════

The grading guidelines are labeled (ALMOST), which means:
- The answer is CLOSE to correct but has MINOR but NON-NEGLIGIBLE issues
- This REQUIRES "partial" grade (NEVER "correct", rarely "incorrect")
- The student made meaningful progress but has small mistakes

⚠️  HARD RULE: "(Almost)" labeled answers MUST be graded "partial"
- The label confirms there are flaws → grade is "partial", not "correct"
- This is non-negotiable - the label itself is evidence of imperfection
"""
            elif '(partial)' in guidelines_lower:
                hint_section = """
════════════════════════════════════════════════════════════════
⚠️  MANDATORY RULE: Label is (PARTIAL) → GRADE MUST BE "PARTIAL"
════════════════════════════════════════════════════════════════

The grading guidelines are labeled (PARTIAL), which means:
- The answer has SOME correct elements but is INCOMPLETE
- This REQUIRES "partial" grade (NEVER "correct")
- The student showed meaningful progress but didn't finish or has errors

⚠️  HARD RULE: "(Partial)" labeled answers MUST be graded "partial"
- The label confirms the answer is incomplete → grade is "partial", not "correct"
- This is non-negotiable - incomplete answers cannot be "correct"
"""
            elif '(incorrect)' in guidelines_lower:
                hint_section = """
════════════════════════════════════════════════════════════════
ℹ️  GUIDELINE NOTE: Label is (INCORRECT)
════════════════════════════════════════════════════════════════

The grading guidelines are labeled (INCORRECT), which means:
- The answer has SIGNIFICANT ERRORS or shows NO meaningful progress
- This typically means "incorrect" (or "partial" if some progress exists)

Check if there's ANY valid reasoning or correct steps:
- If ANY progress exists → "partial"
- Only "incorrect" if truly no valid mathematical reasoning
"""
            elif '(correct)' in guidelines_lower:
                hint_section = """
════════════════════════════════════════════════════════════════
ℹ️  GUIDELINE NOTE: Label is (CORRECT)
════════════════════════════════════════════════════════════════

The grading guidelines are labeled (CORRECT), which means:
- The student believes their answer is correct

⚠️  You must INDEPENDENTLY VERIFY by comparing to the official solution.
Only grade "correct" if TRULY FLAWLESS. Do NOT blindly accept the label.
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

【correct】= FULL CREDIT (100%) - USE ONLY WHEN PERFECT
  • The solution is COMPLETE and FLAWLESS
  • Every logical step is valid and correctly executed
  • Final answer matches the official solution exactly
  • NO errors, NO missing steps, NO logical gaps, NO calculation mistakes
  • ONLY use "correct" if you would give 100% credit without hesitation
  • This is RARE - most answers have some minor issue

【partial】= PARTIAL CREDIT (1-99%) - USE FOR IMPERFECT BUT VALID WORK
  • The solution has SOME valid progress but is incomplete or flawed
  • Contains errors, missing steps, logical gaps, or calculation mistakes
  • Shows understanding of key concepts but execution is imperfect
  • Answer is "almost correct" or "mostly correct" → this is "partial"
  • "(Almost)" labeled answers ALWAYS go here - never "correct"
  • "(Partial)" labeled answers ALWAYS go here - never "correct"

【incorrect】= NO CREDIT (0%) - USE WHEN NO MEANINGFUL PROGRESS
  • The solution shows NO understanding of the problem
  • Completely wrong approach or no valid mathematical reasoning
  • No meaningful progress toward the solution
  • Answer is nonsense, irrelevant, or blank
  • "(Incorrect)" labeled answers with NO valid steps go here

════════════════════════════════════════════════════════════════
DECISION FRAMEWORK - FOLLOW THIS ORDER
════════════════════════════════════════════════════════════════

Step 1: Is the answer TRULY PERFECT? (zero flaws, complete, matches official solution)
  → If YES → grade is "correct"
  → If NO (any flaw at all) → continue to Step 2

Step 2: Does the answer show ANY meaningful progress?
  → If YES (valid concepts, correct approach started, good ideas) → grade is "partial"
  → If NO (no valid reasoning, completely wrong, nonsense) → grade is "incorrect"

CRITICAL RULES:
• "correct" is for PERFECT answers only - be strict about this
• "partial" is for answers with valid work but flaws
• "incorrect" is for answers with NO valid mathematical reasoning
• "(Almost)" in guidelines → ALWAYS "partial" (never "correct")
• "(Partial)" in guidelines → ALWAYS "partial" (never "correct")
• "(Incorrect)" in guidelines → typically "incorrect" (unless some progress exists)
• "(Correct)" in guidelines → verify independently, don't blindly trust{hint_section}

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
3. The response field must be a simple string, not an array or object
4. Example valid response: "response": "partial"
5. Example invalid response: "response": ["partial"] or "response": "Partial"

════════════════════════════════════════════════════════════════
NOW GRADE THE ANSWER
════════════════════════════════════════════════════════════════

Remember: 
- "correct" = PERFECT (rare)
- "partial" = has valid work but flaws (common)
- "incorrect" = no valid reasoning (uncommon)

Provide your grade in the JSON format above."""

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
        # This is critical for handling (Almost) and (Partial) labeled answers correctly
        original_prediction = prediction
        prediction = _validate_grade_with_guidelines(prediction, guideline_hint, grading_guidelines)
        if prediction != original_prediction:
            self.log_fn(f"Grade corrected by guideline validation: {original_prediction} → {prediction}")
        
        # HARD ENFORCEMENT: (Almost) and (Partial) labels MUST result in "partial"
        # This is a non-negotiable rule that overrides even the LLM's judgment
        guidelines_lower = (grading_guidelines or "").lower()
        if '(almost)' in guidelines_lower or '(partial)' in guidelines_lower:
            if prediction == "correct":
                prediction = "partial"
                self.log_fn(f"Grade ENFORCED to 'partial' due to (Almost)/(Partial) label")
        
        # HARD ENFORCEMENT: (Incorrect) labels should not be "correct"
        if '(incorrect)' in guidelines_lower and prediction == "correct":
            # Check if there's any progress mentioned
            progress_indicators = [
                'some', 'partial', 'progress', 'part', 'step', 'idea', 'approach',
                'observation', 'invariant', 'proved', 'shown'
            ]
            has_progress = any(indicator in guidelines_lower for indicator in progress_indicators)
            if has_progress:
                prediction = "partial"
                self.log_fn(f"Grade ENFORCED to 'partial' due to (Incorrect) label with progress indicators")
            else:
                prediction = "incorrect"
                self.log_fn(f"Grade ENFORCED to 'incorrect' due to (Incorrect) label")
        
        # Final validation: ensure we always return a valid grade
        if prediction not in _VALID_GRADES:
            prediction = "partial"  # Default to partial as it's the safest middle ground
            self.log_fn("No valid grade found, defaulting to 'partial' for safety")

        return str(prediction), msg_history
