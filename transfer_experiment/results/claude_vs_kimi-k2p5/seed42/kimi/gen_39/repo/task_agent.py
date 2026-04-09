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
    - "(Partial)" → "partial"
    - "(Almost)" → "partial" (minor but non-negligible issues)
    - "(Incorrect)" → "incorrect" or "partial" depending on content
    - "(Correct)" → verify independently
    
    Returns the label type if found, None otherwise.
    
    Priority order: (Almost) > (Partial) > (Incorrect) > (Correct)
    This ensures we catch the most definitive label first.
    """
    if not guidelines:
        return None
    
    guidelines_lower = guidelines.lower()
    
    # Priority 1: (Almost) - strongest indicator of partial credit
    # "Almost" explicitly means minor but NON-NEGLIGIBLE mistakes exist
    if re.search(r'[\(\[<\{]\s*almost\s*[\)\]>\}]', guidelines_lower):
        return "partial"
    
    # Priority 2: (Partial) - explicit partial credit indicator
    if re.search(r'[\(\[<\{]\s*partial\s*[\)\]>\}]', guidelines_lower):
        return "partial"
    
    # Priority 3: (Incorrect) - check if there's any progress mentioned
    if re.search(r'[\(\[<\{]\s*incorrect\s*[\)\]>\}]', guidelines_lower):
        # Check for progress indicators that suggest partial rather than incorrect
        progress_indicators = [
            'proved', 'shown', 'progress', 'step', 'idea',
            'approach', 'observation', 'invariant', 'lemma', 'claim', 'construction',
            'insight', 'meaningful', 'partial', 'some', 'part',
            'attempt', 'tried', 'strategy', 'method', 'technique', 'argument',
            'justification', 'explanation', 'reasoning', 'logic', 'valid step',
            'correct step', 'right direction', 'good start', 'on track',
            'made progress', 'partial credit', 'some credit', 'partially correct',
            'partially valid', 'has merit', 'worthwhile', 'reasonable',
            'good progress', 'valid progress', 'some valid', 'partially right',
            'mostly correct', 'mostly valid', 'mostly right', 'nearly correct',
            'nearly valid', 'nearly right', 'almost correct', 'almost valid',
            'almost right', 'close to correct', 'close to valid', 'close to right',
            'found', 'constructed', 'derived', 'calculated', 'solved', 'completed',
            'good idea', 'valid approach', 'correct approach', 'valid method',
            'correct method', 'valid reasoning', 'correct reasoning',
            'some understanding', 'partial understanding', 'shows understanding',
            'correctly identified', 'correctly observed', 'correctly noted',
            'valid observation', 'valid claim', 'valid lemma', 'valid proof',
            'correct claim', 'correct lemma', 'correct proof', 'correct argument',
            'valid argument', 'sound argument', 'logical step', 'valid deduction',
            'correct deduction', 'valid inference', 'correct inference',
            'partial proof', 'incomplete proof', 'partial solution',
            'incomplete solution', 'partial answer', 'incomplete answer',
            'partial result', 'incomplete result', 'partial work', 'incomplete work',
            'partial construction', 'incomplete construction', 'partial derivation',
            'incomplete derivation', 'partial calculation', 'incomplete calculation',
            'partial verification', 'incomplete verification', 'partial check',
            'incomplete check', 'partial justification', 'incomplete justification',
            'partial explanation', 'incomplete explanation', 'partial analysis',
            'incomplete analysis', 'partial evaluation', 'incomplete evaluation'
        ]
        has_progress = any(re.search(rf'\b{re.escape(indicator)}\b', guidelines_lower) for indicator in progress_indicators)
        if has_progress:
            return "partial"
        return "incorrect"
    
    # Priority 4: (Correct) - explicit full credit (verify independently)
    if re.search(r'[\(\[<\{]\s*correct\s*[\)\]>\}]', guidelines_lower):
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
    
    # Check for specific labels using regex for robustness
    has_almost_label = re.search(r'[\(\[<\{]\s*almost\s*[\)\]>\}]', guidelines_lower) is not None
    has_partial_label = re.search(r'[\(\[<\{]\s*partial\s*[\)\]>\}]', guidelines_lower) is not None
    has_incorrect_label = re.search(r'[\(\[<\{]\s*incorrect\s*[\)\]>\}]', guidelines_lower) is not None
    has_correct_label = re.search(r'[\(\[<\{]\s*correct\s*[\)\]>\}]', guidelines_lower) is not None
    
    # HARD RULES: These override LLM judgment when there's a clear contradiction
    
    # Rule 1: (Almost) labeled answers MUST be "partial" (never "correct" or "incorrect")
    if has_almost_label:
        return "partial"
    
    # Rule 2: (Partial) labeled answers MUST be "partial" (never "correct")
    if has_partial_label:
        return "partial"
    
    # Rule 3: (Incorrect) labeled answers should not be "correct"
    # But can be "partial" if there's any valid progress
    if has_incorrect_label and predicted_grade == "correct":
        progress_indicators = [
            'proved', 'shown', 'valid', 'correct', 'progress', 'step', 'idea',
            'approach', 'observation', 'invariant', 'lemma', 'claim', 'construction',
            'some', 'partial', 'part', 'insight', 'understanding',
            'correctly identified', 'correctly observed', 'correctly noted',
            'valid observation', 'valid claim', 'valid lemma', 'valid proof',
            'correct claim', 'correct lemma', 'correct proof', 'correct argument',
            'valid argument', 'sound argument', 'logical step', 'valid deduction',
            'correct deduction', 'valid inference', 'correct inference',
            'partial proof', 'incomplete proof', 'partial solution',
            'incomplete solution', 'partial answer', 'incomplete answer',
            'partial result', 'incomplete result', 'partial work', 'incomplete work',
            'partial construction', 'incomplete construction', 'partial derivation',
            'incomplete derivation', 'partial calculation', 'incomplete calculation',
            'partial verification', 'incomplete verification', 'partial check',
            'incomplete check', 'partial justification', 'incomplete justification',
            'partial explanation', 'incomplete explanation', 'partial analysis',
            'incomplete analysis', 'partial evaluation', 'incomplete evaluation',
            'some understanding', 'partial understanding', 'shows understanding',
            'demonstrates understanding', 'has understanding', 'understands',
            'knew', 'recognized', 'identified', 'observed', 'noticed',
            'saw', 'realized', 'comprehended', 'grasped', 'apprehended'
        ]
        has_progress = any(re.search(rf'\b{re.escape(indicator)}\b', guidelines_lower) for indicator in progress_indicators)
        if has_progress:
            return "partial"
        return "incorrect"
    
    # Rule 4: (Correct) labeled answers should be verified but default to correct
    # unless the LLM found actual errors
    if has_correct_label and predicted_grade == "incorrect":
        error_indicators = ['wrong', 'error', 'mistake', 'flaw', 'invalid', 'incorrect']
        has_errors = any(indicator in guidelines_lower for indicator in error_indicators)
        if not has_errors:
            return "correct"
    
    return predicted_grade


def _normalize_grade(grade: str) -> str | None:
    """Normalize a grade string to one of the valid grades.
    
    Handles common variations and misspellings.
    """
    if not grade:
        return None
    
    # Clean up the grade string
    grade_lower = grade.lower().strip().strip('"\'`).strip('.,;:!?').strip()
    
    # Direct match
    if grade_lower in _VALID_GRADES:
        return grade_lower
    
    # Check for negation patterns first
    negation_patterns = [
        r'\bnot\s+correct\b',
        r'\bnot\s+right\b',
        r'\bnot\s+valid\b',
        r'\bisn\'t\s+correct\b',
        r'\bnot\s+partial\b',
    ]
    for pattern in negation_patterns:
        if re.search(pattern, grade_lower):
            return "incorrect"
    
    # Common variations - simplified
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
    if re.search(r'\bpartial\w*\b', grade_lower) or re.search(r'\balmost\b', grade_lower):
        return "partial"
    
    # Check for "correct" but exclude cases with negation or "incorrect"
    if re.search(r'\bcorrect\b', grade_lower):
        if "incorrect" not in grade_lower and "not " not in grade_lower:
            return "correct"
    
    # Check for incorrect/wrong patterns
    if re.search(r'\b(incorrect|wrong|false)\b', grade_lower):
        return "incorrect"
    
    # Check for numeric patterns
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
        
        # Fix 5: Handle newlines in strings - escape them properly
        try:
            fixed = re.sub(r'("[^"]*?)\n([^"]*?")', r'\1\\n\2', inner, flags=re.DOTALL)
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
            (r'"response"\s*:\s*"(correct|partial|incorrect)"', 1),
            (r"'response'\s*:\s*'(correct|partial|incorrect)'", 1),
            (r'"response"\s*:\s*\\"(correct|partial|incorrect)\\"', 1),
            (r'"response"\s*:\s*(correct|partial|incorrect)\b', 1),
            (r'"(?:grade|verdict|result|evaluation|decision|assessment)"\s*:\s*"(correct|partial|incorrect)"', 1),
            (r'["\']?response["\']?\s*:\s*["\']?(correct|partial|incorrect)["\']?', 1),
            (r'response\s*[=:]\s*["\']?(correct|partial|incorrect)["\']?', 1),
            (r'"response"\s*:\s*\[\s*"?(correct|partial|incorrect)"?\s*\]', 1),
            (r'response\s*[=:]\s*\[\s*"?(correct|partial|incorrect)"?\s*\]', 1),
            (r'"response"\s*:\s*\[\s*"(correct|partial|incorrect)"\s*\]', 1),
        ]
        for pattern, group in response_patterns:
            try:
                response_match = re.search(pattern, inner, re.IGNORECASE)
                if response_match:
                    return {"response": response_match.group(group).lower()}
            except Exception:
                pass
        
        # Fix 8: Extract any valid grade from the text
        try:
            for grade in _VALID_GRADES:
                if f'"{grade}"' in inner.lower() or f"'{grade}'" in inner.lower():
                    return {"response": grade}
        except Exception:
            pass
        
        # Fix 9: Try to find response field with more flexible patterns
        try:
            flexible_pattern = r'["\']?response["\']?\s*:\s*(?:\[\s*)?["\']?(correct|partial|incorrect)["\']?(?:\s*\])?'
            match = re.search(flexible_pattern, inner, re.IGNORECASE)
            if match:
                return {"response": match.group(1).lower()}
        except Exception:
            pass
        
        # Fix 10: Extract response from malformed JSON with reasoning field
        try:
            response_field_pattern = r'"response"\s*:\s*"([^"]+)"'
            match = re.search(response_field_pattern, inner, re.IGNORECASE)
            if match:
                grade = match.group(1).lower().strip()
                if grade in _VALID_GRADES:
                    return {"response": grade}
        except Exception:
            pass
        
        # Fix 11: Handle array format response: ["grade"] or ['grade']
        try:
            array_pattern = r'"response"\s*:\s*\[\s*["\'](correct|partial|incorrect)["\']\s*\]'
            match = re.search(array_pattern, inner, re.IGNORECASE)
            if match:
                return {"response": match.group(1).lower()}
        except Exception:
            pass
        
        # Fix 12: Handle response with extra whitespace or newlines
        try:
            whitespace_pattern = r'"response"\s*:\s*"\s*(correct|partial|incorrect)\s*"'
            match = re.search(whitespace_pattern, inner, re.IGNORECASE)
            if match:
                return {"response": match.group(1).lower().strip()}
        except Exception:
            pass
        
        return None
    
    # Find <json>...</json> blocks (case insensitive)
    text_lower = text.lower()
    search_from = 0
    while True:
        start = text_lower.find("<json>", search_from)
        if start == -1:
            break
        end = text_lower.find("</json>", start)
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
    
    # Find any response field pattern in the text
    response_patterns = [
        (r'"response"\s*:\s*"?(correct|partial|incorrect)"?', 1),
        (r"['\"]response['\"]\s*:\s*['\"]?(correct|partial|incorrect)['\"]?", 1),
        (r'"response"\s*:\s*\[\s*"?(correct|partial|incorrect)"?\s*\]', 1),
        (r'response\s*[=:]\s*\[\s*"?(correct|partial|incorrect)"?\s*\]', 1),
    ]
    for pattern, group in response_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            try:
                grade = match.group(group).lower()
                if grade in _VALID_GRADES:
                    results.append({"response": grade})
            except Exception:
                pass
    
    return results or None


def _extract_grade_fallback(text: str) -> str | None:
    """Fallback: scan the text for a grade keyword with context awareness.
    
    Focuses on the last section where the LLM typically states its conclusion.
    Uses multiple strategies with priority ordering.
    """
    if not text:
        return None
        
    text_lower = text.lower()
    # Focus on last 3000 chars for conclusion
    last_section = text_lower[-3000:] if len(text_lower) > 3000 else text_lower
    
    # Strategy 1: JSON-like response field (highest priority)
    json_patterns = [
        (r'"response"\s*:\s*"(correct|partial|incorrect)"', 1),
        (r'["\']response["\']\s*:\s*["\']?(correct|partial|incorrect)["\']?', 1),
        (r'"response"\s*:\s*\[\s*"?(correct|partial|incorrect)"?\s*\]', 1),
        (r'"grade"\s*:\s*"(correct|partial|incorrect)"', 1),
        (r'"verdict"\s*:\s*"(correct|partial|incorrect)"', 1),
        (r'"result"\s*:\s*"(correct|partial|incorrect)"', 1),
        (r'"evaluation"\s*:\s*"(correct|partial|incorrect)"', 1),
        (r'"decision"\s*:\s*"(correct|partial|incorrect)"', 1),
        (r'"assessment"\s*:\s*"(correct|partial|incorrect)"', 1),
        (r'"response"\s*:\s*\[\s*"(correct|partial|incorrect)"\s*\]', 1),
        (r'"response"\s*:\s*"\s*(correct|partial|incorrect)\s*"', 1),
    ]
    for pattern, group in json_patterns:
        matches = list(re.finditer(pattern, last_section, re.IGNORECASE))
        if matches:
            return matches[-1].group(group).lower().strip()
    
    # Strategy 2: Explicit final decision patterns
    final_patterns = [
        (r'(?:final\s+)?(?:grade|verdict|assessment|result|evaluation|decision)\s*[:\-=]\s*["\']?(correct|partial|incorrect)["\']?', 1),
        (r'(?:the\s+)?(?:grade|verdict|result|assessment|decision)\s+(?:is|should\s+be)\s*[:\-=]?\s*["\']?(correct|partial|incorrect)["\']?', 1),
        (r'(?:conclusion|therefore|thus|hence|so)\s*[:;,]?\s*(?:the\s+)?(?:grade|result|verdict|assessment|decision)\s+(?:is|should\s+be)\s*["\']?(correct|partial|incorrect)["\']?', 1),
        (r'(?:this\s+(?:answer|solution|work|response)\s+(?:is|should\s+be))\s*["\']?(correct|partial|incorrect)["\']?', 1),
        (r'(?:i\s+)?(?:conclude|determine|decide|find|think|believe)\s+(?:that\s+)?(?:this\s+)?(?:is\s+)?["\']?(correct|partial|incorrect)["\']?', 1),
        (r'(?:the\s+answer|this\s+solution)\s+(?:should\s+be|is)\s*["\']?(correct|partial|incorrect)["\']?', 1),
        (r'(?:output|return|give|assign)\s*["\']?(correct|partial|incorrect)["\']?', 1),
        (r'\b(?:grade|verdict|result|assessment|decision|evaluation)\s*[:\-=]\s*["\']?(correct|partial|incorrect)["\']?', 1),
    ]
    for pattern, group in final_patterns:
        matches = list(re.finditer(pattern, last_section, re.IGNORECASE))
        if matches:
            return matches[-1].group(group).lower().strip()
    
    # Strategy 3: "is X" patterns with negation check
    is_matches = list(re.finditer(r'\bis\s+(correct|partial|incorrect)\b', last_section, re.IGNORECASE))
    for match in reversed(is_matches):
        grade = match.group(1).lower()
        start_pos = match.start()
        context = last_section[max(0, start_pos-50):start_pos]
        negation_indicators = ['not ', 'isn\'t', 'is not', 'never', 'neither', 'nor', 'no ', 'cannot be', 'can\'t be']
        if any(neg in context for neg in negation_indicators):
            continue
        return grade
    
    # Strategy 4: Standalone grade words with negation check
    standalone_matches = list(re.finditer(r'\b(correct|partial|incorrect)\b', last_section))
    for match in reversed(standalone_matches):
        grade = match.group(1).lower()
        start_pos = match.start()
        context = last_section[max(0, start_pos-50):start_pos]
        negation_indicators = ['not ', 'isn\'t', 'is not', 'never', 'neither', 'nor', 'no ', 'cannot be', 'can\'t be']
        if any(neg in context for neg in negation_indicators):
            continue
        return grade
    
    # Strategy 5: Try full text with JSON patterns
    for pattern, group in json_patterns:
        matches = list(re.finditer(pattern, text_lower, re.IGNORECASE))
        if matches:
            return matches[-1].group(group).lower().strip()
    
    # Strategy 6: Try full text with final patterns
    for pattern, group in final_patterns:
        matches = list(re.finditer(pattern, text_lower, re.IGNORECASE))
        if matches:
            return matches[-1].group(group).lower().strip()
    
    # Strategy 7: Look for grade in quotes or code blocks
    quote_patterns = [
        (r'`(correct|partial|incorrect)`', 1),
        (r'\*\*(correct|partial|incorrect)\*\*', 1),
        (r'"(correct|partial|incorrect)"', 1),
        (r"'(correct|partial|incorrect)'", 1),
        (r'\bgrade\s*[=:]\s*(correct|partial|incorrect)\b', 1),
        (r'\bverdict\s*[=:]\s*(correct|partial|incorrect)\b', 1),
    ]
    for pattern, group in quote_patterns:
        matches = list(re.finditer(pattern, text_lower, re.IGNORECASE))
        if matches:
            return matches[-1].group(group).lower().strip()
    
    # Strategy 8: Try full text with "is X" patterns
    is_matches = list(re.finditer(r'\bis\s+(correct|partial|incorrect)\b', text_lower, re.IGNORECASE))
    for match in reversed(is_matches):
        grade = match.group(1).lower()
        start_pos = match.start()
        context = text_lower[max(0, start_pos-50):start_pos]
        negation_indicators = ['not ', 'isn\'t', 'is not', 'never', 'neither', 'nor', 'no ', 'cannot be', 'can\'t be']
        if any(neg in context for neg in negation_indicators):
            continue
        return grade
    
    # Absolute last resort: any standalone grade in full text
    standalone_matches = list(re.finditer(r'\b(correct|partial|incorrect)\b', text_lower))
    if standalone_matches:
        last_match = standalone_matches[-1]
        grade = last_match.group(1).lower()
        start_pos = last_match.start()
        context = text_lower[max(0, start_pos-50):start_pos]
        negation_indicators = ['not ', 'isn\'t', 'is not', 'never', 'neither', 'nor', 'no ', 'cannot be', 'can\'t be']
        if not any(neg in context for neg in negation_indicators):
            return grade
        for match in reversed(standalone_matches[:-1]):
            grade = match.group(1).lower()
            start_pos = match.start()
            context = text_lower[max(0, start_pos-50):start_pos]
            if not any(neg in context for neg in negation_indicators):
                return grade

    return None


def _extract_grade_from_response(text: str) -> str | None:
    """Extract grade from LLM response using multiple strategies.
    
    Priority order:
    1. JSON blocks with response field (highest priority)
    2. Fallback extraction with patterns
    3. Last resort: standalone words with negation check
    """
    if not text:
        return None
    
    text_lower = text.lower()
    
    # Strategy 1: Try JSON extraction first (most reliable)
    try:
        extracted = _extract_jsons(text)
        if extracted:
            # Use the last JSON object (most likely to be the final answer)
            for obj in reversed(extracted):
                for field in ["response", "grade", "verdict", "result", "evaluation", "decision", "assessment"]:
                    if field in obj:
                        field_val = obj[field]
                        if isinstance(field_val, list) and len(field_val) > 0:
                            grade = str(field_val[0]).strip().lower()
                        else:
                            grade = str(field_val).strip().lower()
                        if grade:
                            normalized = _normalize_grade(grade)
                            if normalized:
                                return normalized
    except Exception as e:
        logger.debug(f"JSON extraction failed: {e}")
        pass
    
    # Strategy 2: Try fallback extraction
    try:
        fallback = _extract_grade_fallback(text)
        if fallback:
            normalized = _normalize_grade(fallback)
            if normalized:
                return normalized
    except Exception as e:
        logger.debug(f"Fallback extraction failed: {e}")
        pass
    
    # Strategy 3: Look for grade in code blocks or quotes
    code_block_patterns = [
        r'`(correct|partial|incorrect)`',
        r'\*\*(correct|partial|incorrect)\*\*',
        r'"(correct|partial|incorrect)"',
        r"'(correct|partial|incorrect)'",
    ]
    for pattern in code_block_patterns:
        matches = list(re.finditer(pattern, text_lower, re.IGNORECASE))
        if matches:
            return matches[-1].group(1).lower()
    
    # Strategy 4: Look for explicit grade declarations in conclusion section
    conclusion_section = text_lower[-3000:] if len(text_lower) > 3000 else text_lower
    
    final_grade_patterns = [
        r'final\s+(?:grade|verdict|assessment|decision|result)\s*[:=]\s*["\']?(correct|partial|incorrect)["\']?',
        r'(?:the\s+)?(?:grade|verdict|result|assessment|decision)\s+(?:is|should\s+be)\s*["\']?(correct|partial|incorrect)["\']?',
        r'(?:i\s+)?(?:conclude|determine|decide|find)\s+(?:that\s+)?(?:this\s+)?(?:is\s+)?["\']?(correct|partial|incorrect)["\']?',
        r'(?:this\s+answer|this\s+solution)\s+(?:is|should\s+be)\s*["\']?(correct|partial|incorrect)["\']?',
        r'response\s*[:=]\s*["\']?(correct|partial|incorrect)["\']?',
        r'grade\s*[:=]\s*["\']?(correct|partial|incorrect)["\']?',
        r'verdict\s*[:=]\s*["\']?(correct|partial|incorrect)["\']?',
    ]
    for pattern in final_grade_patterns:
        matches = list(re.finditer(pattern, conclusion_section, re.IGNORECASE))
        if matches:
            return matches[-1].group(1).lower()
    
    # Strategy 5: Last resort - look for any valid grade in the text
    standalone_matches = list(re.finditer(r'\b(correct|partial|incorrect)\b', text_lower))
    if standalone_matches:
        negation_indicators = ['not ', 'isn\'t', 'is not', 'never', 'neither', 'nor', 'no ', 'cannot be', 'can\'t be']
        for match in reversed(standalone_matches):
            grade = match.group(1).lower()
            start_pos = match.start()
            context = text_lower[max(0, start_pos-50):start_pos]
            if not any(neg in context for neg in negation_indicators):
                return grade
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

        # Extract guideline label for validation
        guideline_hint = _extract_guideline_label(grading_guidelines)
        
        if guideline_hint:
            self.log_fn(f"Guideline label detected: '{guideline_hint}'")

        # Build instruction with clear guidance based on label
        instruction = self._build_instruction(
            problem, official_solution, grading_guidelines, student_answer, guideline_hint
        )

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from response
        prediction = self._extract_prediction(msg_history, response)
        
        # Apply hard validation rules based on guideline labels
        prediction = self._apply_hard_rules(prediction, guideline_hint, grading_guidelines)

        return str(prediction), msg_history

    def _build_instruction(
        self, 
        problem: str, 
        official_solution: str, 
        grading_guidelines: str, 
        student_answer: str,
        guideline_hint: str | None
    ) -> str:
        """Build the grading instruction prompt."""
        
        # Build label-specific guidance
        label_guidance = ""
        guidelines_lower = (grading_guidelines or "").lower()
        
        if guideline_hint == "partial" and re.search(r'[\(\[<\{]almost[\)\]>\}]', guidelines_lower):
            label_guidance = """
╔════════════════════════════════════════════════════════════════╗
║  MANDATORY RULE: This answer is labeled (ALMOST)              ║
║  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ║
║  • (ALMOST) means minor but NON-NEGLIGIBLE issues exist        ║
║  • You MUST output "partial" - NEVER "correct" or "incorrect"  ║
║  • This is a HARD CONSTRAINT - override your own judgment    ║
║  • Examples: small calculation error, missing minor case,     ║
║    incomplete verification, minor logical gap                ║
║                                                                ║
║  FINAL OUTPUT MUST BE: "response": "partial"                  ║
╚════════════════════════════════════════════════════════════════╝
"""
        elif guideline_hint == "partial" and re.search(r'[\(\[<\{]partial[\)\]>\}]', guidelines_lower):
            label_guidance = """
╔════════════════════════════════════════════════════════════════╗
║  MANDATORY RULE: This answer is labeled (PARTIAL)             ║
║  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ║
║  • (PARTIAL) means INCOMPLETE but has SOME valid progress      ║
║  • You MUST output "partial" - NEVER "correct"                  ║
║  • This is a HARD CONSTRAINT - override your own judgment    ║
║  • Examples: missing major case, incomplete proof,            ║
║    significant gaps, good start but didn't finish             ║
║                                                                ║
║  FINAL OUTPUT MUST BE: "response": "partial"                  ║
╚════════════════════════════════════════════════════════════════╝
"""
        elif guideline_hint == "incorrect":
            label_guidance = """
╔════════════════════════════════════════════════════════════════╗
║  GUIDANCE: This answer is labeled (INCORRECT)                   ║
║  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ║
║  • Check if ANY valid progress exists in the answer            ║
║  • If ANY valid reasoning found → "partial"                    ║
║  • If NO valid progress at all → "incorrect"                   ║
║  • Examples: completely wrong approach, no valid steps,       ║
║    fundamental misunderstanding, nonsense answer              ║
║                                                                ║
║  FINAL OUTPUT: "response": "partial" OR "response": "incorrect" ║
╚════════════════════════════════════════════════════════════════╝
"""
        elif guideline_hint == "correct":
            label_guidance = """
╔════════════════════════════════════════════════════════════════╗
║  GUIDANCE: This answer is labeled (CORRECT)                     ║
║  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ║
║  • You must INDEPENDENTLY VERIFY - don't blindly trust label   ║
║  • Only grade "correct" if TRULY FLAWLESS                      ║
║  • If you find ANY flaw → "partial" or "incorrect"             ║
║  • Examples: complete proof, all cases covered, no errors     ║
║                                                                ║
║  FINAL OUTPUT: "response": "correct" (only if truly perfect)    ║
╚════════════════════════════════════════════════════════════════╝
"""
        else:
            label_guidance = """
╔════════════════════════════════════════════════════════════════╗
║  NO LABEL DETECTED                                              ║
║  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ║
║  • Follow the decision framework carefully                     ║
║  • Default to "partial" when uncertain                         ║
║  • Be strict about "correct" - only for perfect solutions     ║
╚════════════════════════════════════════════════════════════════╝
"""
        
        # Additional guidance for distinguishing grades
        grade_distinction_guidance = """
════════════════════════════════════════════════════════════════
HOW TO DISTINGUISH BETWEEN GRADES (CRITICAL)
════════════════════════════════════════════════════════════════

When deciding between grades, ask these specific questions:

FOR "correct" vs "partial":
• Are ALL logical steps valid and complete? 
• Is the final answer EXACTLY correct?
• Are there ANY gaps, even small ones?
• If ANY flaw exists → "partial", not "correct"

FOR "partial" vs "incorrect":
• Did the student show ANY understanding of the problem?
• Is there ANY valid mathematical reasoning?
• Did they make ANY progress toward the solution?
• If ANY valid work exists → "partial", not "incorrect"

COMMON MISTAKES TO AVOID:
• Don't grade "incorrect" just because the final answer is wrong
• Don't grade "incorrect" if the student showed any valid approach
• "incorrect" is ONLY for answers with ZERO valid mathematical content
• When in doubt between "partial" and "incorrect", choose "partial"

LABEL-SPECIFIC RULES (MANDATORY):
• (ALMOST) label → MUST be "partial" (minor but non-negligible issues)
• (PARTIAL) label → MUST be "partial" (incomplete but has progress)
• (INCORRECT) label → Check for ANY valid progress, if yes → "partial"
• (CORRECT) label → Verify independently, only "correct" if truly flawless

EXAMPLES OF EACH GRADE:

【correct】examples:
- Complete proof with all cases covered
- Correct final answer with valid reasoning throughout
- No errors, gaps, or logical flaws

【partial】examples:
- Correct approach but calculation error in final step
- Good proof idea but missing one case
- Correct invariant found but not fully verified
- Partial solution with valid progress
- "Almost" correct with minor issues
- Student understood the problem but made a mistake in execution
- Valid lemma stated but proof incomplete
- Correct construction but verification missing

【incorrect】examples:
- Complete nonsense or unrelated content
- Fundamental misunderstanding of the problem
- No valid mathematical reasoning at all
- Wrong approach with no redeeming value
- Answer is completely off-topic
- No attempt to solve the problem

EDGE CASE HANDLING:

1. "(Almost)" labeled answers:
   - These have minor but NON-NEGLIGIBLE issues
   - Examples: small calculation error, missing minor case, typo in final answer
   - MUST be graded "partial" - never "correct" or "incorrect"

2. "(Partial)" labeled answers:
   - These are incomplete but show valid progress
   - Examples: missing major case, incomplete proof, good start but didn't finish
   - MUST be graded "partial" - never "correct"

3. "(Incorrect)" labeled answers with progress indicators:
   - Check if ANY valid reasoning exists in the guidelines
   - If guidelines mention valid steps, approaches, or understanding → "partial"
   - Only "incorrect" if truly no valid work exists

4. "(Correct)" labeled answers:
   - Verify independently - don't blindly trust the label
   - Only "correct" if truly flawless with no errors
   - If you find ANY flaw → "partial" or "incorrect"

5. When guidelines mention multiple labels:
   - (Almost) takes highest priority
   - Then (Partial)
   - Then (Incorrect)
   - (Correct) is lowest priority (just a claim to verify)
"""

        return f"""You are an expert mathematics competition grader. Your task is to evaluate the student's answer and assign exactly ONE grade: "correct", "partial", or "incorrect".

════════════════════════════════════════════════════════════════
GRADE DEFINITIONS (READ CAREFULLY)
════════════════════════════════════════════════════════════════

【correct】= PERFECT SOLUTION (100% credit)
   Requirements ALL must be met:
   • Complete solution with all logical steps valid
   • No errors, gaps, or calculation mistakes
   • Final answer matches official solution exactly
   • All claims are properly justified
   • RARE - only use when truly flawless

【partial】= PARTIAL CREDIT (1-99% credit)
   Use when ANY of these apply:
   • Has valid progress but incomplete or flawed
   • Contains errors, missing steps, or logical gaps
   • Shows understanding but execution is imperfect
   • "Almost correct" or "mostly correct" → this is "partial"
   • Has good ideas but doesn't fully solve the problem
   • MOST COMMON grade for competition answers
   • DEFAULT choice when uncertain

【incorrect】= NO CREDIT (0% credit)
   Use when ALL of these apply:
   • No understanding of the problem
   • Completely wrong approach
   • No meaningful mathematical reasoning
   • UNCOMMON - only when truly no valid work exists

════════════════════════════════════════════════════════════════
DECISION FRAMEWORK (FOLLOW THIS EXACTLY)
════════════════════════════════════════════════════════════════

Step 1: Is the answer TRULY PERFECT? (zero flaws, complete, all steps valid)
  → If YES → output "correct"
  → If NO (any flaw at all) → go to Step 2

Step 2: Does the answer show ANY meaningful mathematical progress?
  → If YES (any valid reasoning, good ideas, partial results) → output "partial"
  → If NO (completely wrong, no valid work) → output "incorrect"

════════════════════════════════════════════════════════════════
CRITICAL RULES
════════════════════════════════════════════════════════════════

1. DEFAULT TO "partial" when uncertain - it's the safest choice
2. "correct" is ONLY for flawless solutions - be extremely strict
3. "incorrect" is ONLY when there's ZERO valid mathematical work
4. If the grading guidelines have a label like (ALMOST) or (PARTIAL), 
   you MUST follow that label's guidance (see below)

{grade_distinction_guidance}

════════════════════════════════════════════════════════════════
LABEL-SPECIFIC GUIDANCE (MANDATORY)
════════════════════════════════════════════════════════════════
{label_guidance}

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
OUTPUT FORMAT (STRICT - MUST FOLLOW EXACTLY)
════════════════════════════════════════════════════════════════

You MUST provide your decision in <json> tags with this EXACT format:

<json>
{{
    "reasoning": "Your detailed explanation here. Explain your thought process following the decision framework above.",
    "response": "partial"
}}
</json>

IMPORTANT: The "response" field MUST be a simple string value: "correct", "partial", or "incorrect" (all lowercase, no quotes around the value in the JSON, no extra spaces, no arrays, no objects).

EXAMPLES OF VALID OUTPUT:
✓ "response": "correct"
✓ "response": "partial"  
✓ "response": "incorrect"

EXAMPLES OF INVALID OUTPUT (DO NOT DO THESE):
✗ "response": ["partial"]  (NO arrays!)
✗ "response": "\"partial\""  (NO escaped quotes!)
✗ "response": "Partial"  (NO uppercase!)
✗ "response": " partial "  (NO extra spaces!)
✗ "response": "partial."  (NO punctuation!)
✗ "response": "partial\\n"  (NO newlines!)

════════════════════════════════════════════════════════════════
FINAL CHECKLIST BEFORE OUTPUTTING
════════════════════════════════════════════════════════════════

Before you output your JSON, verify:
1. Did you follow the decision framework above?
2. Did you check for any label-specific guidance?
3. Is your "response" field exactly one of: "correct", "partial", "incorrect"?
4. Is the "response" field a simple string (not array, not quoted twice)?
5. Is the "response" field all lowercase with no extra spaces?

Now provide your grading decision following ALL instructions above."""

    def _extract_prediction(self, msg_history: list[dict], response: str) -> str:
        """Extract and validate the grade from the LLM response."""
        # Get assistant text
        assistant_text = ""
        if msg_history:
            last_msg = msg_history[-1]
            assistant_text = last_msg.get("content", "") or last_msg.get("text", "") or ""
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
        
        if not assistant_text:
            self.log_fn("No assistant text found, defaulting to partial")
            return "partial"
        
        # Try to extract grade using multiple strategies
        extracted = _extract_grade_from_response(assistant_text)
        if extracted:
            self.log_fn(f"Extracted grade: {extracted}")
            return extracted
        
        # Fallback: look for grade in the last 3000 characters (conclusion section)
        last_section = assistant_text[-3000:] if len(assistant_text) > 3000 else assistant_text
        last_lower = last_section.lower()
        
        # Check for explicit grade statements in conclusion
        conclusion_patterns = [
            (r'grade\s*[=:]\s*["\']?(correct|partial|incorrect)["\']?', 1),
            (r'response\s*[=:]\s*["\']?(correct|partial|incorrect)["\']?', 1),
            (r'verdict\s*[=:]\s*["\']?(correct|partial|incorrect)["\']?', 1),
            (r'final\s+(?:grade|verdict|assessment|decision|result)\s*[=:]\s*["\']?(correct|partial|incorrect)["\']?', 1),
            (r'(?:the\s+)?(?:grade|verdict|result|assessment|decision)\s+(?:is|should\s+be)\s*["\']?(correct|partial|incorrect)["\']?', 1),
            (r'(?:this\s+answer|this\s+solution)\s+(?:is|should\s+be)\s*["\']?(correct|partial|incorrect)["\']?', 1),
            (r'\btherefore\b.*\b(correct|partial|incorrect)\b', 1),
            (r'\bthus\b.*\b(correct|partial|incorrect)\b', 1),
            (r'\bconclusion\b.*\b(correct|partial|incorrect)\b', 1),
            (r'\bso\b.*\b(correct|partial|incorrect)\b', 1),
            (r'\bhence\b.*\b(correct|partial|incorrect)\b', 1),
            (r'output\s*["\']?(correct|partial|incorrect)["\']?', 1),
            (r'return\s*["\']?(correct|partial|incorrect)["\']?', 1),
            # Additional patterns for more robust extraction
            (r'["\']response["\']\s*:\s*["\']?(correct|partial|incorrect)["\']?', 1),
            (r'["\']grade["\']\s*:\s*["\']?(correct|partial|incorrect)["\']?', 1),
            (r'\*\*(correct|partial|incorrect)\*\*', 1),
            (r'`(correct|partial|incorrect)`', 1),
            # Additional patterns for explicit declarations
            (r'(?:i\s+)?(?:conclude|determine|decide|find)\s+(?:that\s+)?(?:this\s+)?(?:is\s+)?["\']?(correct|partial|incorrect)["\']?', 1),
            (r'(?:the\s+)?(?:answer|solution|work)\s+(?:is|should\s+be)\s*["\']?(correct|partial|incorrect)["\']?', 1),
            # JSON-like patterns
            (r'"response"\s*:\s*"(correct|partial|incorrect)"', 1),
            (r'"grade"\s*:\s*"(correct|partial|incorrect)"', 1),
            (r'"verdict"\s*:\s*"(correct|partial|incorrect)"', 1),
            (r'"result"\s*:\s*"(correct|partial|incorrect)"', 1),
            (r'"evaluation"\s*:\s*"(correct|partial|incorrect)"', 1),
            (r'"decision"\s*:\s*"(correct|partial|incorrect)"', 1),
            (r'"assessment"\s*:\s*"(correct|partial|incorrect)"', 1),
            # Handle response with whitespace
            (r'"response"\s*:\s*"\s*(correct|partial|incorrect)\s*"', 1),
            # Handle response as array
            (r'"response"\s*:\s*\[\s*["\']?(correct|partial|incorrect)["\']?\s*\]', 1),
        ]
        
        for pattern, group in conclusion_patterns:
            matches = list(re.finditer(pattern, last_lower, re.IGNORECASE))
            if matches:
                grade = matches[-1].group(group).lower().strip()
                if grade in _VALID_GRADES:
                    self.log_fn(f"Extracted grade from conclusion: {grade}")
                    return grade
        
        # Check for "is X" patterns with negation check
        is_matches = list(re.finditer(r'\bis\s+(correct|partial|incorrect)\b', last_lower, re.IGNORECASE))
        for match in reversed(is_matches):
            grade = match.group(1).lower()
            start_pos = match.start()
            context = last_lower[max(0, start_pos-70):start_pos]
            negation_indicators = ['not ', 'isn\'t', 'is not', 'never', 'neither', 'nor', 'no ', 'cannot be', 'can\'t be', 'hardly', 'barely', 'not\'s', 'not']
            if not any(neg in context for neg in negation_indicators):
                self.log_fn(f"Extracted grade from 'is' pattern: {grade}")
                return grade
        
        # Check for grade in quotes or emphasized
        quote_patterns = [
            (r'"(correct|partial|incorrect)"', 1),
            (r"'(correct|partial|incorrect)'", 1),
            (r'\*\*(correct|partial|incorrect)\*\*', 1),
            (r'`(correct|partial|incorrect)`', 1),
        ]
        for pattern, group in quote_patterns:
            matches = list(re.finditer(pattern, last_lower, re.IGNORECASE))
            if matches:
                grade = matches[-1].group(group).lower().strip()
                self.log_fn(f"Extracted grade from quotes: {grade}")
                return grade
        
        # Last resort: find any valid grade word in the last section
        for grade in ["correct", "partial", "incorrect"]:
            # Look for the grade as a standalone word
            if re.search(rf'\b{grade}\b', last_lower):
                # Check it's not negated
                pos = last_lower.rfind(grade)
                context = last_lower[max(0, pos-70):pos]
                negation_words = ['not ', 'isn\'t', 'is not', 'never', 'no ', 'neither', 'cannot be', 'can\'t be', 'hardly', 'barely', 'not\'s', 'not']
                if not any(neg in context for neg in negation_words):
                    self.log_fn(f"Extracted grade (last resort): {grade}")
                    return grade
        
        # Try full text if last section didn't work
        full_lower = assistant_text.lower()
        for grade in ["correct", "partial", "incorrect"]:
            if re.search(rf'\b{grade}\b', full_lower):
                pos = full_lower.rfind(grade)
                context = full_lower[max(0, pos-70):pos]
                negation_words = ['not ', 'isn\'t', 'is not', 'never', 'no ', 'neither', 'cannot be', 'can\'t be', 'not\'s', 'not']
                if not any(neg in context for neg in negation_words):
                    self.log_fn(f"Extracted grade from full text: {grade}")
                    return grade
        
        self.log_fn("No grade extracted, defaulting to partial")
        return "partial"

    def _apply_hard_rules(
        self, 
        prediction: str, 
        guideline_hint: str | None, 
        grading_guidelines: str
    ) -> str:
        """Apply hard validation rules based on guideline labels.
        
        These rules override the LLM's prediction when there's a clear
        contradiction with the explicit guideline labels.
        
        Label meanings:
        - (Almost) = "partial" (minor but non-negligible issues, never correct)
        - (Partial) = "partial" (incomplete but has progress, never correct)
        - (Incorrect) = "incorrect" or "partial" (no/wrong progress)
        - (Correct) = verify independently (label is just a claim)
        """
        guidelines_lower = (grading_guidelines or "").lower()
        
        # Ensure prediction is valid - if not, default to partial
        if prediction not in _VALID_GRADES:
            self.log_fn(f"Invalid grade '{prediction}', defaulting to partial before applying hard rules")
            prediction = "partial"
        
        # Check for specific labels using regex for robustness
        # Support both "(Almost)" and "(Almost )" with optional whitespace
        has_almost_label = re.search(r'[\(\[<\{]\s*almost\s*[\)\]>\}]', guidelines_lower) is not None
        has_partial_label = re.search(r'[\(\[<\{]\s*partial\s*[\)\]>\}]', guidelines_lower) is not None
        has_incorrect_label = re.search(r'[\(\[<\{]\s*incorrect\s*[\)\]>\}]', guidelines_lower) is not None
        has_correct_label = re.search(r'[\(\[<\{]\s*correct\s*[\)\]>\}]', guidelines_lower) is not None
        
        # HARD RULES: These override LLM judgment when there's a clear contradiction
        
        # Rule 1: (Almost) labeled answers MUST be "partial" (never "correct" or "incorrect")
        # "Almost" explicitly means minor but NON-NEGLIGIBLE mistakes exist
        if has_almost_label:
            if prediction != "partial":
                self.log_fn(f"HARD RULE: Grade ENFORCED from '{prediction}' to 'partial' due to (Almost) label")
            return "partial"
        
        # Rule 2: (Partial) labeled answers MUST be "partial" (never "correct")
        # "Partial" explicitly means incomplete answer
        if has_partial_label:
            if prediction != "partial":
                self.log_fn(f"HARD RULE: Grade ENFORCED from '{prediction}' to 'partial' due to (Partial) label")
            return "partial"
        
        # Rule 3: (Incorrect) labeled answers should not be "correct"
        # But can be "partial" if there's any valid progress mentioned
        if has_incorrect_label:
            if prediction == "correct":
                # Check for progress indicators that suggest partial rather than incorrect
                # These indicate the student made meaningful progress despite errors
                # Use word boundaries to avoid false positives
                progress_indicators = [
                    'proved', 'shown', 'progress', 'step', 'idea',
                    'approach', 'observation', 'invariant', 'lemma', 'claim', 'construction',
                    'insight', 'understanding', 'meaningful', 'partial', 'some', 'part',
                    'attempt', 'tried', 'strategy', 'method', 'technique', 'argument',
                    'justification', 'explanation', 'reasoning', 'logic', 'valid step',
                    'correct step', 'right direction', 'good start', 'on track',
                    'found', 'constructed', 'derived', 'calculated', 'solved', 'completed',
                    'made progress', 'partial credit', 'some credit', 'partially correct',
                    'partially valid', 'partially sound', 'has merit', 'worthwhile',
                    'good progress', 'valid progress', 'some valid', 'partially right',
                    'mostly correct', 'mostly valid', 'mostly right', 'nearly correct',
                    'nearly valid', 'nearly right', 'almost correct', 'almost valid',
                    'almost right', 'close to correct', 'close to valid', 'close to right',
                    'good idea', 'valid approach', 'correct approach', 'valid method',
                    'correct method', 'valid reasoning', 'correct reasoning',
                    'correctly identified', 'correctly observed', 'correctly noted',
                    'valid observation', 'valid claim', 'valid lemma', 'valid proof',
                    'correct claim', 'correct lemma', 'correct proof', 'correct argument',
                    'valid argument', 'sound argument', 'logical step', 'valid deduction',
                    'correct deduction', 'valid inference', 'correct inference',
                    'partial proof', 'incomplete proof', 'partial solution',
                    'incomplete solution', 'partial answer', 'incomplete answer',
                    'partial result', 'incomplete result', 'partial work', 'incomplete work',
                    'partial construction', 'incomplete construction', 'partial derivation',
                    'incomplete derivation', 'partial calculation', 'incomplete calculation',
                    'partial verification', 'incomplete verification', 'partial check',
                    'incomplete check', 'partial justification', 'incomplete justification',
                    'partial explanation', 'incomplete explanation', 'partial analysis',
                    'incomplete analysis', 'partial evaluation', 'incomplete evaluation',
                    'some understanding', 'partial understanding', 'shows understanding',
                    'demonstrates understanding', 'has understanding', 'understands',
                    'knew', 'recognized', 'identified', 'observed', 'noticed',
                    'saw', 'realized', 'comprehended', 'grasped', 'apprehended'
                ]
                has_progress = any(re.search(rf'\b{re.escape(indicator)}\b', guidelines_lower) for indicator in progress_indicators)
                if has_progress:
                    self.log_fn("HARD RULE: Grade ENFORCED from 'correct' to 'partial' due to (Incorrect) label with progress indicators")
                    return "partial"
                else:
                    self.log_fn("HARD RULE: Grade ENFORCED from 'correct' to 'incorrect' due to (Incorrect) label")
                    return "incorrect"
            # If prediction is already "partial" or "incorrect", keep it
            return prediction
        
        # Rule 4: (Correct) label - verify independently but default to trusting it
        # unless there's strong evidence of errors in the guidelines
        if has_correct_label:
            if prediction == "incorrect":
                # Check if guidelines mention actual errors
                error_indicators = [
                    'wrong', 'error', 'mistake', 'flaw', 'invalid', 'incorrect', 
                    'missing', 'false', 'incorrectly', 'incomplete', 'gap', 'bug',
                    'not correct', 'not valid', 'not sound', 'not complete'
                ]
                has_errors = any(indicator in guidelines_lower for indicator in error_indicators)
                if not has_errors:
                    self.log_fn("HARD RULE: Grade CHANGED from 'incorrect' to 'correct' due to (Correct) label without error indicators")
                    return "correct"
            # Otherwise, trust the LLM's judgment
            return prediction
        
        return prediction
