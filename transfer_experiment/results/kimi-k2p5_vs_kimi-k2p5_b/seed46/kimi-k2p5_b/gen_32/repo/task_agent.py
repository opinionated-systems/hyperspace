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

# Valid grading labels for IMO evaluation - 4 class system (including "almost")
VALID_LABELS = ["correct", "almost", "partial", "incorrect"]

# Grade hierarchy for disambiguation (from highest to lowest quality)
GRADE_HIERARCHY = ["correct", "almost", "partial", "incorrect"]

# Confidence thresholds for grade classification
CONFIDENCE_THRESHOLDS = {
    "correct": 0.95,   # Very high confidence required for correct
    "almost": 0.80,   # High confidence for almost
    "partial": 0.70,  # Moderate confidence for partial
    "incorrect": 0.60, # Lower threshold for incorrect (safer default)
}


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks or raw JSON with enhanced robustness.
    
    Uses multiple strategies to find JSON content with improved handling of:
    - Nested braces and brackets
    - Trailing commas
    - Single quotes vs double quotes
    - Unicode characters
    - Malformed JSON structures
    - Markdown code blocks
    - Nested JSON objects
    - LLM-specific formatting quirks
    - Empty or whitespace-only responses
    - Multiple JSON objects in one response
    """
    if not text or not isinstance(text, str):
        return None
    
    # Handle empty or whitespace-only text
    text = text.strip()
    if not text:
        return None
        
    results = []
    found_spans = []  # Track (start, end) of found JSON to avoid duplicates
    
    def _is_duplicate(start: int, end: int) -> bool:
        """Check if this span overlaps with any already found."""
        for s, e in found_spans:
            if not (end <= s or start >= e):  # Overlapping
                return True
        return False
    
    def _add_result(parsed: dict, start: int, end: int) -> bool:
        """Add result if not duplicate and has expected keys."""
        if _is_duplicate(start, end):
            return False
        if isinstance(parsed, dict):
            # Check for grading-related keys (expanded list)
            grading_keys = ['grade', 'response', 'label', 'evaluation', 'reasoning', 'result', 
                          'score', 'verdict', 'assessment', 'classification', 'decision']
            if any(key in parsed for key in grading_keys):
                results.append(parsed)
                found_spans.append((start, end))
                return True
        return False
    
    # Strategy 1: Look for <json>...</json> tags (most reliable)
    search_from = 0
    while True:
        start = text.find("<json>", search_from)
        if start == -1:
            break
        end = text.find("</json>", start)
        if end == -1:
            break
        inner_start = start + 6
        inner_end = end
        inner = text[inner_start:inner_end].strip()
        search_from = end + 7
        
        # Skip if inner content is empty
        if not inner:
            continue
        
        # Handle case where there might be nested JSON tags
        json_candidates = _generate_json_candidates(inner)
        
        for candidate in json_candidates:
            try:
                parsed = json.loads(candidate)
                if _add_result(parsed, start, end + 7):
                    break  # Success, move to next tag
            except json.JSONDecodeError:
                continue
    
    # Strategy 2: Look for JSON objects in markdown code blocks
    code_block_patterns = [
        (r'```json\s*\n?(.*?)\n?```', True),  # ```json ... ```
        (r'```\s*\n?(.*?)\n?```', False),     # ``` ... ```
        (r'`\s*\n?(.*?)\n?`', False),          # ` ... ` (inline code)
    ]
    for pattern, is_json_block in code_block_patterns:
        for match in re.finditer(pattern, text, re.DOTALL):
            block = match.group(1).strip()
            if not block:
                continue
            
            # For non-json blocks, skip if it doesn't look like JSON
            if not is_json_block and not (block.startswith('{') or block.startswith('[')):
                continue
                
            json_candidates = _generate_json_candidates(block)
            
            for candidate in json_candidates:
                try:
                    parsed = json.loads(candidate)
                    if _add_result(parsed, match.start(), match.end()):
                        break
                except json.JSONDecodeError:
                    continue
    
    # Strategy 3: Look for raw JSON objects with balanced braces
    for start, end, json_str in _find_json_objects_with_spans(text):
        # Skip if this overlaps with already found spans
        if _is_duplicate(start, end):
            continue
        
        # Skip if the JSON string is too short to be meaningful
        if len(json_str) < 10:
            continue
            
        json_candidates = _generate_json_candidates(json_str)
        
        for candidate in json_candidates:
            try:
                parsed = json.loads(candidate)
                if _add_result(parsed, start, end):
                    break
            except json.JSONDecodeError:
                continue
    
    # Strategy 3b: Look for JSON objects with nested structures
    # Sometimes LLMs output multiple nested JSON objects
    if not results:
        # Try to find JSON objects that might be nested or concatenated
        nested_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        for match in re.finditer(nested_pattern, text, re.DOTALL):
            json_str = match.group(0)
            if len(json_str) < 10:
                continue
            # Check if it contains grading-related keys
            if any(key in json_str.lower() for key in ['grade', 'label', 'evaluation', 'reasoning']):
                json_candidates = _generate_json_candidates(json_str)
                for candidate in json_candidates:
                    try:
                        parsed = json.loads(candidate)
                        if _add_result(parsed, match.start(), match.end()):
                            break
                    except json.JSONDecodeError:
                        continue
    
    # Strategy 4: Look for JSON-like structures that might be malformed
    # Try to extract key-value pairs that look like grade assignments
    if not results:
        grade_pattern = r'["\']?(?:grade|label|evaluation|result|response)["\']?\s*[:=]\s*["\']?([a-z]+)["\']?'
        for match in re.finditer(grade_pattern, text, re.IGNORECASE):
            potential_grade = match.group(1).lower()
            if potential_grade in VALID_LABELS:
                # Create a synthetic result
                synthetic = {"grade": potential_grade, "reasoning": "Extracted from text pattern"}
                results.append(synthetic)
                break
    
    # Strategy 5: Handle case where response is just a grade word
    # Check if the entire text (after cleaning) is just a valid grade
    if not results:
        clean_text = text.lower().strip().strip('"\'`.,;:!?()[]{}')
        if clean_text in VALID_LABELS:
            synthetic = {"grade": clean_text, "reasoning": "Grade extracted from plain text"}
            results.append(synthetic)
    
    # Strategy 6: Look for grade in various delimiters
    if not results:
        delimiter_patterns = [
            r'\bgrade\s*[=:]\s*["\']?(correct|almost|partial|incorrect)["\']?\b',
            r'\blabel\s*[=:]\s*["\']?(correct|almost|partial|incorrect)["\']?\b',
            r'\bevaluation\s*[=:]\s*["\']?(correct|almost|partial|incorrect)["\']?\b',
            r'\bresult\s*[=:]\s*["\']?(correct|almost|partial|incorrect)["\']?\b',
            r'\bverdict\s*[=:]\s*["\']?(correct|almost|partial|incorrect)["\']?\b',
            r'\bassessment\s*[=:]\s*["\']?(correct|almost|partial|incorrect)["\']?\b',
        ]
        for pattern in delimiter_patterns:
            match = re.search(pattern, text_lower)
            if match:
                synthetic = {"grade": match.group(1), "reasoning": "Extracted from delimiter pattern"}
                results.append(synthetic)
                break
    
    # Strategy 7: Look for grade at the end of the response
    if not results:
        # Common pattern: reasoning followed by grade at the end
        end_patterns = [
            r'["\']?(correct|almost|partial|incorrect)["\']?\s*$',
            r'grade\s+is\s+["\']?(correct|almost|partial|incorrect)["\']?\s*$',
            r'(?:therefore|thus|hence|so)\s*,?\s*["\']?(correct|almost|partial|incorrect)["\']?\s*$',
        ]
        for pattern in end_patterns:
            match = re.search(pattern, text_lower)
            if match:
                synthetic = {"grade": match.group(1), "reasoning": "Extracted from end of response"}
                results.append(synthetic)
                break
    
    return results if results else None


def _generate_json_candidates(text: str) -> list[str]:
    """Generate multiple candidate versions of potentially malformed JSON.
    
    Returns candidates in order of likelihood to be valid JSON.
    Uses comprehensive fixes for common LLM JSON errors with enhanced robustness.
    """
    candidates = [text]
    seen = {text}
    
    def add_candidate(candidate: str) -> None:
        if candidate not in seen and len(candidate) > 0:
            candidates.append(candidate)
            seen.add(candidate)
    
    # Fix 1: Remove trailing commas before closing braces/brackets
    fixed = re.sub(r',(\s*[}\]])', r'\1', text)
    add_candidate(fixed)
    
    # Fix 2: Replace single quotes with double quotes (carefully)
    fixed = re.sub(r"(?<!\\)'", '"', text)
    add_candidate(fixed)
    
    # Fix 3: Remove comments (both // and /* */)
    fixed = re.sub(r'//.*?\n', '\n', text)
    fixed = re.sub(r'/\*.*?\*/', '', fixed, flags=re.DOTALL)
    add_candidate(fixed)
    
    # Fix 4: Handle escaped quotes that might be double-escaped
    fixed = text.replace('\\"', '"').replace("\\'", "'")
    add_candidate(fixed)
    
    # Fix 5: Remove control characters
    fixed = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)
    add_candidate(fixed)
    
    # Fix 6: Handle newlines in strings (replace with spaces)
    fixed = text.replace('\n', ' ').replace('\r', ' ')
    add_candidate(fixed)
    
    # Fix 7: Handle multiple trailing commas
    fixed = re.sub(r',+\s*([}\]])', r'\1', text)
    add_candidate(fixed)
    
    # Fix 8: Handle missing quotes around keys (simple cases)
    fixed = re.sub(r'\{(\s*)(\w+)(\s*):', r'{\1"\2"\3:', text)
    add_candidate(fixed)
    
    # Fix 9: Handle backticks as quotes
    fixed = text.replace('`', '"')
    add_candidate(fixed)
    
    # Fix 10: Handle smart quotes (curly quotes)
    fixed = text.replace('"', '"').replace('"', '"').replace(''', "'").replace(''', "'")
    add_candidate(fixed)
    
    # Fix 11: Handle unescaped quotes within string values
    # This is a more careful approach - only fix quotes that appear to be inside strings
    def escape_internal_quotes(match):
        content = match.group(1)
        # Escape unescaped quotes
        return '"' + re.sub(r'(?<!\\)"', r'\\"', content) + '"'
    
    fixed = re.sub(r'"([^"]*)"', escape_internal_quotes, text)
    add_candidate(fixed)
    
    # Fix 12: Handle missing closing braces
    open_braces = text.count('{') - text.count('}')
    if open_braces > 0:
        fixed = text + ('}' * open_braces)
        add_candidate(fixed)
    
    # Fix 13: Handle missing closing brackets
    open_brackets = text.count('[') - text.count(']')
    if open_brackets > 0:
        fixed = text + (']' * open_brackets)
        add_candidate(fixed)
    
    # Fix 14: Handle BOM (Byte Order Mark)
    fixed = text.lstrip('\ufeff')
    add_candidate(fixed)
    
    # Fix 15: Handle escaped newlines
    fixed = text.replace('\\n', '\n').replace('\\r', '\r').replace('\\t', '\t')
    add_candidate(fixed)
    
    # Fix 16: Handle double curly braces (common in template languages)
    fixed = text.replace('{{', '{').replace('}}', '}')
    add_candidate(fixed)
    
    # Fix 17: Handle trailing content after JSON object
    # Find the first complete JSON object and extract just that
    brace_count = 0
    in_string = False
    escape_next = False
    for i, char in enumerate(text):
        if escape_next:
            escape_next = False
            continue
        if char == '\\':
            escape_next = True
            continue
        if char == '"' and not in_string:
            in_string = True
        elif char == '"' and in_string:
            in_string = False
        elif not in_string:
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    fixed = text[:i+1]
                    add_candidate(fixed)
                    break
    
    # Fix 18: Handle leading content before JSON object
    json_start = text.find('{')
    if json_start > 0:
        fixed = text[json_start:]
        add_candidate(fixed)
    
    # Fix 19: Handle multiple JSON objects - try to find the one with grade key
    # Look for patterns like "grade": or 'grade':
    grade_match = re.search(r'["\']?grade["\']?\s*:', text, re.IGNORECASE)
    if grade_match:
        # Try to extract from 100 chars before to end, or from start
        start_pos = max(0, grade_match.start() - 100)
        fixed = text[start_pos:]
        add_candidate(fixed)
    
    # Fix 20: Handle JSON with unquoted string values for grade
    # Pattern: "grade": word (without quotes)
    fixed = re.sub(r'("grade"\s*:\s*)([a-z]+)(\s*[,}])', r'\1"\2"\3', text, flags=re.IGNORECASE)
    add_candidate(fixed)
    
    # Fix 21: Handle concatenated JSON objects (common LLM error)
    # Split on }{ and take the first valid one
    if '}{' in text:
        parts = text.split('}{')
        for part in parts:
            fixed = part if part.startswith('{') else '{' + part
            fixed = fixed if part.endswith('}') else fixed + '}'
            add_candidate(fixed)
    
    # Fix 22: Handle escaped quotes that might be double-escaped in JSON strings
    fixed = text.replace('\\"', '"')
    add_candidate(fixed)
    
    # Fix 23: Handle JSON with unquoted keys (common LLM error)
    # Pattern: key: value -> "key": value
    fixed = re.sub(r'(\s*)(\w+)(\s*):(\s*)', r'\1"\2"\3:\4', text)
    add_candidate(fixed)
    
    # Fix 24: Handle trailing commas in arrays
    fixed = re.sub(r',(\s*\])', r'\1', text)
    add_candidate(fixed)
    
    # Fix 25: Handle multiple consecutive commas
    fixed = re.sub(r',+', ',', text)
    add_candidate(fixed)
    
    # Fix 26: Handle missing commas between key-value pairs
    # Pattern: "key": "value" "key2": -> "key": "value", "key2":
    fixed = re.sub(r'("\s*:\s*[^,\{\}\[\]]+)(\s+"\w+"\s*:)', r'\1,\2', text)
    add_candidate(fixed)
    
    return candidates


def _find_json_objects_with_spans(text: str) -> list[tuple[int, int, str]]:
    """Find potential JSON objects by tracking brace balance.
    
    Returns:
        List of (start_index, end_index, json_string) tuples
    """
    objects = []
    i = 0
    n = len(text)
    
    while i < n:
        if text[i] == '{':
            start = i
            brace_count = 1
            i += 1
            in_string = False
            escape_next = False
            
            while i < n and brace_count > 0:
                char = text[i]
                
                if escape_next:
                    escape_next = False
                    i += 1
                    continue
                
                if char == '\\':
                    escape_next = True
                    i += 1
                    continue
                
                if char == '"' and not in_string:
                    in_string = True
                elif char == '"' and in_string:
                    in_string = False
                elif not in_string:
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                
                i += 1
                
            if brace_count == 0:
                end = i
                candidate = text[start:end]
                # Check if it looks like our expected JSON (has grading-related keys)
                lower_candidate = candidate.lower()
                if any(key in lower_candidate for key in ['grade', 'response', 'label', 'evaluation', 'reasoning', 'result']):
                    objects.append((start, end, candidate))
        else:
            i += 1
            
    return objects


def _extract_label_from_text(text: str) -> str | None:
    """Extract grading label from raw text using multiple strategies with priority ordering.
    
    Priority order (most specific to least specific):
    1. Explicit grade assignments ("grade is X", "grade: X")
    2. JSON-like structures without proper parsing
    3. Word boundary matches for each label (using GRADE_HIERARCHY priority)
    4. Contextual indicators in reasoning
    """
    if not text or not isinstance(text, str):
        return None
    
    # Handle empty or whitespace-only text
    text = text.strip()
    if not text:
        return None
        
    text_lower = text.lower()
    
    # Build regex pattern for all valid labels
    labels_pattern = '|'.join(VALID_LABELS)
    
    # Strategy 1: Look for explicit grade assignments with high confidence patterns
    # These patterns indicate a deliberate assignment of grade
    grade_patterns = [
        rf'grade\s+is\s+["\']?({labels_pattern})["\']?',
        rf'grade\s*[:=]\s*["\']?({labels_pattern})["\']?',
        rf'assigned\s+grade\s*[:=]\s*["\']?({labels_pattern})["\']?',
        rf'final\s+grade\s*[:=]\s*["\']?({labels_pattern})["\']?',
        rf'evaluation\s*[:=]\s*["\']?({labels_pattern})["\']?',
        rf'result\s*[:=]\s*["\']?({labels_pattern})["\']?',
        rf'response\s*[:=]\s*["\']?({labels_pattern})["\']?',
        rf'label\s*[:=]\s*["\']?({labels_pattern})["\']?',
        rf'["\']?({labels_pattern})["\']?\s+is\s+appropriate',
        rf'["\']?({labels_pattern})["\']?\s+is\s+the\s+(?:grade|evaluation)',
        rf'["\']?({labels_pattern})["\']?\s+is\s+assigned',
        rf'therefore\s*,?\s*(?:the\s+)?(?:grade|evaluation|answer)\s+is\s+["\']?({labels_pattern})["\']?',
        rf'conclusion\s*:?\s*["\']?({labels_pattern})["\']?',
        rf'determination\s*:?\s*["\']?({labels_pattern})["\']?',
        rf'verdict\s*:?\s*["\']?({labels_pattern})["\']?',
        rf'assessment\s*:?\s*["\']?({labels_pattern})["\']?',
        rf'classification\s*:?\s*["\']?({labels_pattern})["\']?',
        rf'decision\s*:?\s*["\']?({labels_pattern})["\']?',
        rf'award\s*["\']?({labels_pattern})["\']?',
        rf'assign\s*["\']?({labels_pattern})["\']?',
        rf'rate\s+as\s*["\']?({labels_pattern})["\']?',
        rf'score\s+as\s*["\']?({labels_pattern})["\']?',
        rf'mark\s+as\s*["\']?({labels_pattern})["\']?',
        rf'grade[d]?\s*["\']?({labels_pattern})["\']?',
        rf'["\']?({labels_pattern})["\']?\s*grade',
    ]
    
    for pattern in grade_patterns:
        match = re.search(pattern, text_lower)
        if match:
            # Find the first group that matched (could be any of the alternatives)
            for group_num in range(1, len(match.groups()) + 1):
                val = match.group(group_num)
                if val:
                    val = val.lower().strip()
                    if val in VALID_LABELS:
                        return val
    
    # Strategy 2: Look for label in quotes after specific keywords
    quote_patterns = [
        rf'"({labels_pattern})"',
        rf"'({labels_pattern})'",
        rf'`({labels_pattern})`',
        rf'\[({labels_pattern})\]',
        rf'\(({labels_pattern})\)',
    ]
    for pattern in quote_patterns:
        match = re.search(pattern, text_lower)
        if match:
            val = match.group(1).lower().strip()
            if val in VALID_LABELS:
                return val
    
    # Strategy 3: Look for explicit label mentions with word boundaries
    # Use GRADE_HIERARCHY priority (most specific first)
    for label in GRADE_HIERARCHY:
        if re.search(rf'\b{label}\b', text_lower):
            return label
    
    # Strategy 4: Look for grade indicators in reasoning text
    # Check for phrases that indicate a specific grade
    indicator_patterns = {
        "correct": [
            r'\bfull\s+(?:marks?|credit|score|points?)\b',
            r'\bcomplete\s+solution\b',
            r'\bcorrect\s+(?:proof|solution|answer)\b',
            r'\b7\s*/\s*7\b',
            r'\bfull\s+marks?\s*\(?7\b',
            r'\bsolution\s+is\s+(?:correct|valid|complete)\b',
            r'\bproof\s+is\s+(?:correct|valid|complete)\b',
        ],
        "almost": [
            r'\bminor\s+(?:gap|error|mistake|issue)\b',
            r'\bnearly\s+complete\b',
            r'\balmost\s+(?:correct|complete|there)\b',
            r'\b5\s*/\s*7\b',
            r'\b6\s*/\s*7\b',
            r'\bsmall\s+(?:gap|error|omission)\b',
            r'\bvery\s+close\s+to\b',
        ],
        "partial": [
            r'\bsignificant\s+(?:progress|work|insight)\b',
            r'\bpartial\s+(?:credit|marks?|solution)\b',
            r'\bkey\s+lemma\b',
            r'\bmeaningful\s+progress\b',
            r'\b[1-4]\s*/\s*7\b',
            r'\bsubstantial\s+progress\b',
        ],
        "incorrect": [
            r'\bno\s+(?:valid|meaningful)\s+(?:progress|solution|work)\b',
            r'\bfundamentally\s+flawed\b',
            r'\bwrong\s+approach\b',
            r'\b0\s*/\s*7\b',
            r'\bno\s+credit\b',
            r'\bdoes\s+not\s+(?:solve|address)\b',
        ],
    }
    
    for label, patterns in indicator_patterns.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                return label
    
    return None


def _extract_grade_from_reasoning(text: str) -> str | None:
    """Extract grade from reasoning text by analyzing the content.
    
    This function looks for linguistic patterns in the reasoning that indicate
    the grader's assessment, even when explicit grade labels aren't present.
    """
    if not text or not isinstance(text, str):
        return None
    
    text_lower = text.lower()
    
    # Strong indicators for each grade
    grade_indicators = {
        "correct": [
            r'\bfull\s+(?:marks?|credit|score|points?)\b',
            r'\bcomplete\s+(?:proof|solution|answer)\b',
            r'\bcorrect\s+(?:proof|solution|answer|conclusion)\b',
            r'\b7\s*/\s*7\b',
            r'\bfull\s+marks?\s*\(?7\b',
            r'\bsolution\s+is\s+(?:correct|valid|complete)\b',
            r'\bproof\s+is\s+(?:correct|valid|complete)\b',
            r'\ball\s+necessary\s+steps\b',
            r'\bfully\s+(?:correct|solved|proven)\b',
            r'\bmeets\s+all\s+requirements\b',
            r'\bcompletely\s+correct\b',
            r'\bentirely\s+correct\b',
            r'\bwholly\s+correct\b',
            r'\bperfectly\s+valid\b',
            r'\bfully\s+justified\b',
        ],
        "almost": [
            r'\bminor\s+(?:gap|error|mistake|issue|flaw)\b',
            r'\bnearly\s+complete\b',
            r'\balmost\s+(?:correct|complete|there|perfect)\b',
            r'\b5\s*/\s*7\b',
            r'\b6\s*/\s*7\b',
            r'\bsmall\s+(?:gap|error|omission|issue)\b',
            r'\btiny\s+(?:error|mistake|gap)\b',
            r'\bvery\s+close\s+to\s+(?:correct|complete)\b',
            r'\bjust\s+(?:missing|needs)\b',
            r'\bminor\s+correction\b',
            r'\bslight\s+(?:error|issue|problem)\b',
            r'\bminimal\s+(?:gap|error|issue)\b',
            r'\bnegligible\s+(?:error|issue)\b',
            r'\bmostly\s+correct\b',
            r'\bnearly\s+there\b',
        ],
        "partial": [
            r'\bsignificant\s+(?:progress|work|insight|contribution)\b',
            r'\bpartial\s+(?:credit|marks?|solution|progress)\b',
            r'\bkey\s+lemma\b',
            r'\bmeaningful\s+progress\b',
            r'\b[1-4]\s*/\s*7\b',
            r'\bsubstantial\s+progress\b',
            r'\bgood\s+insight\b',
            r'\bcorrect\s+approach\s+but\b',
            r'\bsolved\s+(?:a|some|part)\b',
            r'\bincomplete\s+but\s+(?:valid|correct|good)\b',
            r'\bpartially\s+(?:correct|complete|solved)\b',
            r'\bsome\s+progress\b',
            r'\bnon-trivial\s+progress\b',
            r'\bvaluable\s+insight\b',
            r'\bimportant\s+step\b',
        ],
        "incorrect": [
            r'\bno\s+(?:valid|meaningful)\s+(?:progress|solution|work)\b',
            r'\bfundamentally\s+flawed\b',
            r'\bwrong\s+approach\b',
            r'\b0\s*/\s*7\b',
            r'\bno\s+credit\b',
            r'\bdoes\s+not\s+(?:solve|address|answer)\b',
            r'\bincorrect\s+(?:approach|method|reasoning)\b',
            r'\bonly\s+restates\s+(?:the\s+)?problem\b',
            r'\bno\s+valid\s+mathematical\s+content\b',
            r'\btrivial\s+observations?\b',
            r'\bcompletely\s+wrong\b',
            r'\bentirely\s+incorrect\b',
            r'\bwholly\s+incorrect\b',
            r'\bno\s+real\s+progress\b',
            r'\bfails\s+to\s+(?:solve|address|prove)\b',
        ],
    }
    
    # Count matches for each grade
    grade_scores = {grade: 0 for grade in VALID_LABELS}
    
    for grade, patterns in grade_indicators.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                grade_scores[grade] += 1
    
    # Find the grade with the highest score
    max_score = max(grade_scores.values())
    if max_score > 0:
        # Return the first grade with max score (following hierarchy priority)
        for grade in GRADE_HIERARCHY:
            if grade_scores[grade] == max_score:
                return grade
    
    return None


def _extract_grade_from_edge_cases(text: str) -> str | None:
    """Extract grade from edge cases and malformed responses.
    
    Handles cases where the LLM output doesn't follow expected formats but
    still contains grade information that can be extracted.
    """
    if not text or not isinstance(text, str):
        return None
    
    text_lower = text.lower().strip()
    
    # Handle case where response is just a grade word with punctuation
    clean_text = re.sub(r'[^\w\s]', '', text_lower).strip()
    if clean_text in VALID_LABELS:
        return clean_text
    
    # Handle case where grade is in parentheses
    paren_match = re.search(r'\((correct|almost|partial|incorrect)\)', text_lower)
    if paren_match:
        return paren_match.group(1)
    
    # Handle case where grade is in brackets
    bracket_match = re.search(r'\[(correct|almost|partial|incorrect)\]', text_lower)
    if bracket_match:
        return bracket_match.group(1)
    
    # Handle case where grade is in curly braces
    brace_match = re.search(r'\{(correct|almost|partial|incorrect)\}', text_lower)
    if brace_match:
        return brace_match.group(1)
    
    # Handle case where grade is in angle brackets
    angle_match = re.search(r'<(correct|almost|partial|incorrect)>', text_lower)
    if angle_match:
        return angle_match.group(1)
    
    # Handle case where grade is followed by punctuation
    for label in VALID_LABELS:
        if re.search(rf'^{label}[.!?,;:\s]', text_lower):
            return label
        if re.search(rf'[.!?,;:\s]{label}$', text_lower):
            return label
    
    # Handle case where grade is in a sentence
    sentence_patterns = [
        rf'the\s+grade\s+is\s+(?:"|\')?(correct|almost|partial|incorrect)(?:"|\')?',
        rf'the\s+answer\s+is\s+(?:"|\')?(correct|almost|partial|incorrect)(?:"|\')?',
        rf'the\s+evaluation\s+is\s+(?:"|\')?(correct|almost|partial|incorrect)(?:"|\')?',
        rf'the\s+result\s+is\s+(?:"|\')?(correct|almost|partial|incorrect)(?:"|\')?',
        rf'the\s+student\s+(?:gets|receives|earns)\s+(?:a\s+)?(?:"|\')?(correct|almost|partial|incorrect)(?:"|\')?',
        rf'i\s+(?:would\s+)?(?:assign|give|award)\s+(?:a\s+)?(?:"|\')?(correct|almost|partial|incorrect)(?:"|\')?',
        rf'(?:this|the\s+solution)\s+is\s+(?:"|\')?(correct|almost|partial|incorrect)(?:"|\')?',
    ]
    
    for pattern in sentence_patterns:
        match = re.search(pattern, text_lower)
        if match:
            return match.group(1)
    
    return None


def _normalize_grade(value: str) -> str | None:
    """Normalize a grade value to one of the valid labels with improved logic.
    
    Uses GRADE_HIERARCHY for priority ordering to handle overlapping terms.
    Handles various edge cases including punctuation, whitespace, and common variations.
    """
    if value is None:
        return None
        
    if not isinstance(value, str):
        value = str(value)
    
    # Handle empty or whitespace-only values
    value = value.strip()
    if not value:
        return None
    
    value = value.lower().strip()
    
    # Remove common prefixes/suffixes that might appear
    value = re.sub(r'^(is\s+|graded\s+as\s+|marked\s+as\s+|assigned\s+|final\s+|the\s+|a\s+|an\s+)', '', value)
    value = re.sub(r'\s*(grade|score|mark|evaluation|result|classification|verdict|assessment)$', '', value)
    
    # Remove surrounding quotes and punctuation
    value = value.strip('"\'`,;:.!()[]{} ')
    
    # Handle double-quoted strings (common LLM error)
    value = re.sub(r'^"+', '', value)
    value = re.sub(r'"+$', '', value)
    
    # Handle single-quoted strings
    value = re.sub(r"^'+", '', value)
    value = re.sub(r"'+$", '', value)
    
    # Handle backtick-quoted strings
    value = re.sub(r'^`+', '', value)
    value = re.sub(r'`+$', '', value)
    
    # Direct match
    if value in VALID_LABELS:
        return value
    
    # Check for exact matches with punctuation removed
    clean_value = re.sub(r'[^\w]', '', value)
    if clean_value in VALID_LABELS:
        return clean_value
    
    # Check for partial matches with priority (using GRADE_HIERARCHY - most specific first)
    # This ensures "almost" is checked before "partial", and "incorrect" before "correct"
    for label in GRADE_HIERARCHY:
        if label in value:
            return label
    
    # Handle common variations and synonyms (expanded list)
    variations_map = {
        "correct": [
            'right', 'true', 'valid', 'full', 'complete', '7', '7/7', 'full marks', 'perfect', 
            'solved', 'done', 'success', 'accurate', 'proper', 'sound', 'flawless', 'excellent',
            'full credit', '100%', 'seven', 'full score'
        ],
        "almost": [
            'nearly', 'close', 'mostly', 'almost there', 'minor gap', 'small error', 
            '5/7', '6/7', 'almost correct', 'nearly complete', 'minor issue', 'tiny error',
            'slight mistake', 'minimal gap', 'five', 'six', 'near complete'
        ],
        "partial": [
            'partially', 'part', 'some', 'incomplete', 'half', 'partial credit', 'progress', 
            'attempt', '1/7', '2/7', '3/7', '4/7', 'started', 'beginning', 'in progress',
            'one', 'two', 'three', 'four', 'partial solution', 'significant progress',
            'meaningful work', 'substantial progress'
        ],
        "incorrect": [
            'wrong', 'false', 'invalid', 'error', '0', '0/7', 'no credit', 'none', 'fail', 
            'failed', 'unsolved', 'no solution', 'zero', 'incorrect answer', 'wrong answer',
            'invalid solution', 'no marks', 'zero marks', 'failed attempt', 'unsuccessful'
        ],
    }
    
    for label, variations in variations_map.items():
        if value in variations:
            return label
    
    # Handle numeric scores (IMO 7-point scale)
    # Try to extract numeric scores and map them
    numeric_match = re.search(r'(\d+)\s*/\s*7', value)
    if numeric_match:
        score = int(numeric_match.group(1))
        if score == 7:
            return "correct"
        elif score >= 5:
            return "almost"
        elif score >= 1:
            return "partial"
        else:
            return "incorrect"
    
    # Handle percentage scores
    percent_match = re.search(r'(\d+)%', value)
    if percent_match:
        percent = int(percent_match.group(1))
        if percent >= 95:
            return "correct"
        elif percent >= 70:
            return "almost"
        elif percent >= 15:
            return "partial"
        else:
            return "incorrect"
    
    return None


def _validate_student_answer(student_answer: str) -> tuple[bool, str]:
    """Validate if the student answer contains meaningful content with expanded checks.
    
    Returns:
        (is_valid, reason): Tuple indicating if answer is valid and why it might not be
    """
    if not student_answer:
        return False, "empty"
    
    # Check if it's just whitespace or very short
    stripped = student_answer.strip()
    if not stripped:
        return False, "empty"
    
    if len(stripped) < 3:
        return False, "too_short"
    
    # Check for common "no answer" patterns (expanded list)
    no_answer_patterns = [
        r'^\s*no\s+answer\s*$',
        r'^\s*none\s*$',
        r'^\s*n/a\s*$',
        r'^\s*n\.a\.\s*$',
        r'^\s*not\s+applicable\s*$',
        r'^\s*skip\s*$',
        r'^\s*skipped\s*$',
        r'^\s*blank\s*$',
        r'^\s*empty\s*$',
        r'^\s*null\s*$',
        r'^\s*undefined\s*$',
        r'^\s*nil\s*$',
        r'^\s*idk\s*$',
        r'^\s*i\s+don\'t\s+know\s*$',
        r'^\s*i\s+do\s+not\s+know\s*$',
        r'^\s*no\s+solution\s*$',
        r'^\s*can\'t\s+solve\s*$',
        r'^\s*cannot\s+solve\s*$',
        r'^\s*unsure\s*$',
        r'^\s*unknown\s*$',
        r'^\s*unclear\s*$',
        r'^\s*pass\s*$',
        r'^\s*\?+\s*$',  # Just question marks
        r'^\s*\.+\s*$',  # Just dots
        r'^\s*-+\s*$',   # Just dashes
        r'^\s*_+\s*$',   # Just underscores
        r'^\s*\*+\s*$',  # Just asterisks
        r'^\s*#+\s*$',   # Just hashes
        r'^\s*\$+\s*$',  # Just dollar signs
        r'^\s*\\+\s*$',  # Just backslashes
        r'^\s*\/+\s*$',  # Just forward slashes
        r'^\s*\|+\s*$',  # Just pipes
        r'^\s*~+\s*$',   # Just tildes
        r'^\s*`+\s*$',   # Just backticks
        r'^\s*\^+\s*$',  # Just carets
        r'^\s*&+\s*$',   # Just ampersands
        r'^\s*%+\s*$',   # Just percent signs
        r'^\s*@+\s*$',   # Just at signs
        r'^\s*!+\s*$',   # Just exclamation marks
        r'^\s*\(+\s*$',  # Just opening parens
        r'^\s*\)+\s*$',  # Just closing parens
        r'^\s*\[+\s*$',  # Just opening brackets
        r'^\s*\]+\s*$',  # Just closing brackets
        r'^\s*\{+\s*$',  # Just opening braces
        r'^\s*\}+\s*$',  # Just closing braces
        r'^\s*<+\s*$',   # Just less than
        r'^\s*>+\s*$',   # Just greater than
        r'^\s*=+\s*$',   # Just equals
        r'^\s*\++\s*$',  # Just plus
    ]
    
    for pattern in no_answer_patterns:
        if re.search(pattern, stripped, re.IGNORECASE):
            return False, "no_answer"
    
    # Check if answer is just whitespace and punctuation
    content_only = re.sub(r'[\s\W]', '', stripped)
    if len(content_only) < 2:
        return False, "insufficient_content"
    
    # Check for repeated single characters (like "aaaaaa" or "111111")
    if len(set(content_only.lower())) == 1:
        return False, "repetitive_content"
    
    # Check for excessive repetition of short patterns
    if len(content_only) > 10:
        # Check if the same 2-3 character pattern repeats
        for pattern_len in [2, 3]:
            if len(content_only) >= pattern_len * 3:
                first_pattern = content_only[:pattern_len].lower()
                if all(content_only[i:i+pattern_len].lower() == first_pattern 
                       for i in range(0, min(len(content_only), pattern_len * 5), pattern_len)):
                    return False, "repetitive_pattern"
    
    # Check for placeholder text patterns
    placeholder_patterns = [
        r'\[insert.*?\]',
        r'\[your answer.*?\]',
        r'\[solution.*?\]',
        r'\[work.*?\]',
        r'\[proof.*?\]',
        r'<insert.*?>',
        r'<your answer.*?>',
        r'<solution.*?>',
        r'\(insert.*?\)',
        r'\(your answer.*?\)',
        r'\(solution.*?\)',
        r'answer\s+goes\s+here',
        r'solution\s+goes\s+here',
        r'type\s+your\s+answer',
        r'enter\s+your\s+answer',
        r'write\s+your\s+solution',
    ]
    
    for pattern in placeholder_patterns:
        if re.search(pattern, stripped, re.IGNORECASE):
            return False, "placeholder_text"
    
    return True, "valid"


class TaskAgent:
    """Task agent that solves IMO grading problems with enhanced accuracy."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        # Validate student answer first
        is_valid, validation_reason = _validate_student_answer(student_answer)
        if not is_valid:
            # Return incorrect for empty/invalid answers
            return "incorrect", [{"role": "system", "text": f"Invalid answer detected: {validation_reason}"}]
        
        instruction = f"""You are an expert mathematics grader for International Mathematical Olympiad (IMO) problems. Your task is to evaluate student answers and assign exactly one of four grades.

## GRADE DEFINITIONS (IMO SCORING SYSTEM):

**"correct"** (7 points): The answer is fully correct and complete. It contains a valid proof or solution with all necessary steps, logical reasoning, and reaches the correct conclusion. The student demonstrates full understanding of the problem. Only award this grade if the solution would receive full marks (7/7) in an actual IMO competition.

**"almost"** (5-6 points): The answer is nearly complete with only minor gaps or errors. The student has the right approach, made substantial progress, and is very close to a full solution. The missing piece is small (e.g., a minor case not handled, a small logical gap, or a computational error in an otherwise correct approach). This is stronger than "partial" but not quite "correct".

**"partial"** (1-4 points): The answer shows meaningful progress toward the solution with genuine mathematical insight, but has significant gaps or is incomplete. The student made non-trivial progress beyond just understanding the problem statement. Examples: found a key lemma but didn't complete the proof, solved a significant special case, or made substantial progress on a multi-part problem but missing critical components.

**"incorrect"** (0 points): The answer is wrong, fundamentally flawed, trivial, or shows no valid mathematical progress. The approach is completely wrong, the answer only restates the problem, contains only trivial observations without real mathematical progress, or demonstrates a fundamental misunderstanding of the problem.

## GRADING GUIDELINES CONTEXT:
{grading_guidelines}

## PROBLEM:
{problem}

## OFFICIAL SOLUTION:
{solution}

## STUDENT'S ANSWER TO EVALUATE:
{student_answer}

## YOUR TASK:

Analyze the student's answer carefully following these steps:

1. **Understand the Problem**: Verify you understand what the problem is asking and what constitutes a correct solution.

2. **Review the Official Solution**: Note the key steps, techniques, and final answer required.

3. **Analyze the Student's Answer**: 
   - Check if the student addressed the actual problem asked
   - Identify any correct mathematical insights, lemmas, or techniques
   - Note any errors, gaps, or logical flaws
   - Assess the completeness of the solution
   - Check if the student reached a valid conclusion
   - Evaluate the quality and depth of mathematical reasoning

4. **Compare to Guidelines**: Use the grading guidelines to determine the appropriate grade.

5. **Assign Grade**: Choose exactly one of: "correct", "almost", "partial", or "incorrect"

## GRADING CRITERIA SUMMARY:

- **Award "correct" ONLY if**: The solution is complete, rigorous, and would receive full marks (7/7)
- **Award "almost" if**: Nearly complete solution with only minor gaps/errors (5-6/7)
- **Award "partial" if**: Genuine mathematical progress but significant gaps remain (1-4/7)
- **Award "incorrect" if**: Wrong approach, no meaningful progress, trivial observations only, or fundamental flaws (0/7)

## KEY DISTINCTIONS:

- "almost" vs "partial": "almost" means the student is very close to the full solution with only minor issues. "partial" means significant work remains.
- "partial" vs "incorrect": "partial" requires genuine non-trivial mathematical insight or progress. "incorrect" means little to no valid progress.

## COMMON PITFALLS TO AVOID:

1. **Don't over-grade**: A solution with significant gaps should not be "almost" - use "partial" instead.
2. **Don't under-grade**: If the student made genuine mathematical progress (found a key lemma, solved a special case), use "partial" not "incorrect".
3. **Check for completeness**: "correct" requires ALL necessary steps, not just the right idea.
4. **Watch for restatements**: Answers that only restate the problem without new mathematical content are "incorrect".
5. **Consider partial credit**: Even incomplete solutions with good insights deserve "partial".

## RESPONSE FORMAT (STRICT JSON REQUIRED):

You MUST respond with ONLY a JSON object wrapped in <json> tags. Do NOT include any other text, markdown formatting, or explanations outside the JSON tags.

<json>
{{
    "reasoning": "Detailed explanation of your evaluation including specific strengths and weaknesses found. Cite specific parts of the student's answer. Explain why you chose this specific grade over the alternatives.",
    "grade": "correct"
}}
</json>

CRITICAL INSTRUCTIONS:
1. The "grade" field MUST be exactly one of: "correct", "almost", "partial", or "incorrect" (all lowercase, no quotes around the value in the JSON field itself)
2. Do NOT include any text before or after the <json> tags
3. Do NOT use markdown formatting like ```json inside the <json> tags
4. The JSON must be valid - ensure proper escaping of quotes and newlines
5. Double-check that your grade is one of the four valid options before responding
6. Be conservative with "correct" - only use it for truly complete solutions
7. The grade value must be a plain string without quotes: "correct" not ""correct"" or "'correct'"

## DETAILED GRADING EXAMPLES:

**Example of "correct"**: Student proves all cases, handles all edge cases, provides rigorous justification for each step, and arrives at the correct answer with a complete logical chain.

**Example of "almost"**: Student has the right approach, proves the main case, but misses one small sub-case or has a minor computational error that doesn't affect the overall approach. The fix would be simple.

**Example of "partial"**: Student identifies the key lemma or makes significant progress on the problem (e.g., solves a special case, finds a useful invariant, proves a necessary condition) but doesn't complete the full solution.

**Example of "incorrect"**: Student's approach is completely wrong, or they only rewrite the problem statement without adding mathematical content, or they make trivial observations that don't advance toward the solution.

Example valid responses:
<json>{{"reasoning": "The student provided a complete proof with all necessary steps.", "grade": "correct"}}</json>
<json>{{"reasoning": "The solution is nearly complete with only a minor case missing.", "grade": "almost"}}</json>
<json>{{"reasoning": "The student found a key lemma but didn't complete the proof.", "grade": "partial"}}</json>
<json>{{"reasoning": "The approach is fundamentally flawed.", "grade": "incorrect"}}</json>"""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction using multiple strategies with priority
        prediction = "None"
        extraction_log = []
        try:
            raw_text = msg_history[-1]["text"] if msg_history else ""
            
            if not raw_text:
                self.log_fn("Empty response from LLM")
                return "incorrect", msg_history
            
            # Strategy 1: Try to extract from JSON tags (highest priority)
            extracted = _extract_jsons(raw_text)
            if extracted:
                extraction_log.append(f"JSON extraction found {len(extracted)} objects")
                # Sort by likelihood of being the main grade object (prefer objects with 'grade' key)
                sorted_extracted = sorted(extracted, key=lambda x: 0 if 'grade' in x else 1)
                for json_obj in sorted_extracted:
                    # Try multiple possible keys in order of preference
                    for key in ["grade", "label", "evaluation", "result", "response", 
                               "score", "verdict", "assessment", "classification", "decision"]:
                        if key in json_obj:
                            val = json_obj[key]
                            if val is not None:
                                normalized = _normalize_grade(str(val))
                                if normalized:
                                    prediction = normalized
                                    extraction_log.append(f"Found grade '{prediction}' from key '{key}'")
                                    break
                    if prediction != "None":
                        break
            else:
                extraction_log.append("JSON extraction found no objects")
            
            # Strategy 2: If JSON extraction failed, try text extraction
            if prediction == "None":
                text_pred = _extract_label_from_text(raw_text)
                if text_pred:
                    prediction = text_pred
                    extraction_log.append(f"Found grade '{prediction}' from text extraction")
                else:
                    extraction_log.append("Text extraction found no grade")
            
            # Strategy 3: Last resort - look for any valid label in the text
            # Use GRADE_HIERARCHY for priority (most specific first)
            if prediction == "None":
                text_lower = raw_text.lower()
                for label in GRADE_HIERARCHY:
                    if re.search(rf'\b{label}\b', text_lower):
                        prediction = label
                        extraction_log.append(f"Found grade '{prediction}' from fallback search")
                        break
            
            # Strategy 4: Try to extract from reasoning text patterns
            if prediction == "None":
                reasoning_pred = _extract_grade_from_reasoning(raw_text)
                if reasoning_pred:
                    prediction = reasoning_pred
                    extraction_log.append(f"Found grade '{prediction}' from reasoning analysis")
            
            # Strategy 5: Try edge case extraction for malformed responses
            if prediction == "None":
                edge_pred = _extract_grade_from_edge_cases(raw_text)
                if edge_pred:
                    prediction = edge_pred
                    extraction_log.append(f"Found grade '{prediction}' from edge case analysis")
            
            # Log extraction details for debugging
            if extraction_log:
                self.log_fn(f"Grade extraction: {'; '.join(extraction_log)}")
                    
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            import traceback
            self.log_fn(f"Traceback: {traceback.format_exc()}")

        # Default to incorrect if we couldn't extract anything
        if prediction == "None":
            prediction = "incorrect"
            self.log_fn("No grade found, defaulting to 'incorrect'")
        elif prediction not in VALID_LABELS:
            # If we got an invalid grade, try to normalize it one more time
            normalized = _normalize_grade(prediction)
            if normalized:
                prediction = normalized
            else:
                self.log_fn(f"Invalid grade '{prediction}' found, defaulting to 'incorrect'")
                prediction = "incorrect"
            
        return str(prediction), msg_history
