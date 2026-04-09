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
# Order matters: more specific/distinguishable terms first to avoid misclassification
GRADE_HIERARCHY = ["almost", "partial", "incorrect", "correct"]

# Confidence thresholds for grade classification
CONFIDENCE_THRESHOLDS = {
    "correct": 0.95,   # Very high confidence required for correct
    "almost": 0.85,   # High confidence for almost
    "partial": 0.75,  # Moderate confidence for partial
    "incorrect": 0.70, # Higher threshold for incorrect to reduce false negatives
}

# Grade confusion matrix - which grades are commonly confused
GRADE_CONFUSION_PAIRS = [
    ("almost", "partial"),  # Most commonly confused
    ("partial", "incorrect"),  # Second most common
    ("correct", "almost"),  # Sometimes confused
]

# Keywords that strongly indicate specific grades (for disambiguation)
GRADE_KEYWORDS = {
    "correct": [
        "complete proof", "full solution", "correct answer", "valid proof",
        "all cases", "fully justified", "rigorous proof", "7/7", "full marks",
        "perfect solution", "fully correct", "entirely correct", "wholly correct"
    ],
    "almost": [
        "minor gap", "small error", "nearly complete", "almost correct",
        "minor omission", "tiny mistake", "5/7", "6/7", "close to complete",
        "very close", "mostly correct", "nearly there", "almost there",
        "minimal gap", "slight error", "minor correction", "small fix"
    ],
    "partial": [
        "significant progress", "key lemma", "partial credit", "meaningful work",
        "substantial progress", "incomplete but valid", "1/7", "2/7", "3/7", "4/7",
        "genuine insight", "important step", "correct approach", "solved special case",
        "proved lemma", "found invariant", "meaningful contribution", "non-trivial progress"
    ],
    "incorrect": [
        "fundamentally flawed", "wrong approach", "no valid progress",
        "completely wrong", "trivial observation", "0/7", "no credit",
        "no real progress", "does not solve", "fails to prove", "incorrect assumption",
        "only restates", "invalid reasoning", "wrong answer"
    ]
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
    - Nested <json> tags within JSON content
    - Escaped characters and special encodings
    """
    if not text or not isinstance(text, str):
        return None
    
    # Handle empty or whitespace-only text
    text = text.strip()
    if not text:
        return None
    
    # Pre-process: remove common LLM artifacts that break JSON parsing
    # Remove markdown bold/italic markers that might appear in JSON
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # **bold**
    text = re.sub(r'\*([^*]+)\*', r'\1', text)  # *italic*
    
    # Remove HTML tags that might appear in JSON content (except our <json> tags)
    text = re.sub(r'<(?!json>|/json>)([^>]+)>', r'\1', text)
    
    # Handle common LLM output prefixes that break JSON
    text = re.sub(r'^(?:Here is|Here\'s|The|My|This is|I\'ll provide|Let me provide)\s+(?:the\s+)?(?:json\s+)?(?:response|answer|grade|evaluation|result)[:;]?\s*', '', text, flags=re.IGNORECASE)
    
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
                          'score', 'verdict', 'assessment', 'classification', 'decision', 'answer']
            if any(key in parsed for key in grading_keys):
                results.append(parsed)
                found_spans.append((start, end))
                return True
        return False
    
    # Strategy 1: Look for <json>...</json> tags (most reliable)
    # Handle nested tags by finding the outermost complete pair
    search_from = 0
    while True:
        start = text.find("<json>", search_from)
        if start == -1:
            break
        
        # Find the matching </json> - handle nested <json> tags
        end = start + 6
        json_depth = 1
        while json_depth > 0 and end < len(text):
            next_open = text.find("<json>", end)
            next_close = text.find("</json>", end)
            
            if next_close == -1:
                break
            if next_open != -1 and next_open < next_close:
                json_depth += 1
                end = next_open + 6
            else:
                json_depth -= 1
                end = next_close + 7
        
        if json_depth != 0:
            # Unbalanced tags, try simple approach
            end = text.find("</json>", start)
            if end == -1:
                break
            end = end + 7
        
        inner_start = start + 6
        inner_end = end - 7 if text[end-7:end] == "</json>" else end
        inner = text[inner_start:inner_end].strip()
        search_from = end
        
        # Skip if inner content is empty
        if not inner:
            continue
        
        # Handle case where there might be nested JSON tags in content
        # Remove any remaining <json> or </json> tags from inner content
        inner = re.sub(r'</?json>', '', inner)
        
        # Remove common prefixes that LLMs add inside JSON tags
        inner = re.sub(r'^(?:json\s*)?\n+', '', inner, flags=re.IGNORECASE)
        
        json_candidates = _generate_json_candidates(inner)
        
        for candidate in json_candidates:
            try:
                parsed = json.loads(candidate)
                if _add_result(parsed, start, end):
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
            
            # Remove any markdown formatting from inside the block
            block = re.sub(r'\*\*([^*]+)\*\*', r'\1', block)
            block = re.sub(r'\*([^*]+)\*', r'\1', block)
                
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
    text_lower = text.lower()
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
    
    # Fix 27: Handle JSON with unquoted grade values (e.g., "grade": correct)
    fixed = re.sub(r'("grade"\s*:\s*)(correct|almost|partial|incorrect)(\s*[,}])', r'\1"\2"\3', text, flags=re.IGNORECASE)
    add_candidate(fixed)
    
    # Fix 28: Handle JSON with unquoted label values
    fixed = re.sub(r'("label"\s*:\s*)(correct|almost|partial|incorrect)(\s*[,}])', r'\1"\2"\3', text, flags=re.IGNORECASE)
    add_candidate(fixed)
    
    # Fix 29: Handle JSON with unquoted evaluation values
    fixed = re.sub(r'("evaluation"\s*:\s*)(correct|almost|partial|incorrect)(\s*[,}])', r'\1"\2"\3', text, flags=re.IGNORECASE)
    add_candidate(fixed)
    
    # Fix 30: Handle JSON with unquoted result values
    fixed = re.sub(r'("result"\s*:\s*)(correct|almost|partial|incorrect)(\s*[,}])', r'\1"\2"\3', text, flags=re.IGNORECASE)
    add_candidate(fixed)
    
    # Fix 31: Handle JSON with unquoted response values
    fixed = re.sub(r'("response"\s*:\s*)(correct|almost|partial|incorrect)(\s*[,}])', r'\1"\2"\3', text, flags=re.IGNORECASE)
    add_candidate(fixed)
    
    # Fix 32: Handle JSON with unquoted verdict values
    fixed = re.sub(r'("verdict"\s*:\s*)(correct|almost|partial|incorrect)(\s*[,}])', r'\1"\2"\3', text, flags=re.IGNORECASE)
    add_candidate(fixed)
    
    # Fix 33: Handle JSON with unquoted assessment values
    fixed = re.sub(r'("assessment"\s*:\s*)(correct|almost|partial|incorrect)(\s*[,}])', r'\1"\2"\3', text, flags=re.IGNORECASE)
    add_candidate(fixed)
    
    # Fix 34: Handle JSON with unquoted classification values
    fixed = re.sub(r'("classification"\s*:\s*)(correct|almost|partial|incorrect)(\s*[,}])', r'\1"\2"\3', text, flags=re.IGNORECASE)
    add_candidate(fixed)
    
    # Fix 35: Handle JSON with unquoted decision values
    fixed = re.sub(r'("decision"\s*:\s*)(correct|almost|partial|incorrect)(\s*[,}])', r'\1"\2"\3', text, flags=re.IGNORECASE)
    add_candidate(fixed)
    
    # Fix 36: Handle JSON with unquoted answer values
    fixed = re.sub(r'("answer"\s*:\s*)(correct|almost|partial|incorrect)(\s*[,}])', r'\1"\2"\3', text, flags=re.IGNORECASE)
    add_candidate(fixed)
    
    # Fix 37: Handle JSON with single-quoted grade values
    fixed = re.sub(r"('grade'\s*:\s*)'([a-z]+)'(\s*[,}])", r'\1"\2"\3', text, flags=re.IGNORECASE)
    add_candidate(fixed)
    
    # Fix 38: Handle JSON with single-quoted label values
    fixed = re.sub(r"('label'\s*:\s*)'([a-z]+)'(\s*[,}])", r'\1"\2"\3', text, flags=re.IGNORECASE)
    add_candidate(fixed)
    
    # Fix 39: Handle JSON with single-quoted evaluation values
    fixed = re.sub(r"('evaluation'\s*:\s*)'([a-z]+)'(\s*[,}])", r'\1"\2"\3', text, flags=re.IGNORECASE)
    add_candidate(fixed)
    
    # Fix 40: Handle JSON with single-quoted result values
    fixed = re.sub(r"('result'\s*:\s*)'([a-z]+)'(\s*[,}])", r'\1"\2"\3', text, flags=re.IGNORECASE)
    add_candidate(fixed)
    
    # Fix 41: Handle JSON with single-quoted response values
    fixed = re.sub(r"('response'\s*:\s*)'([a-z]+)'(\s*[,}])", r'\1"\2"\3', text, flags=re.IGNORECASE)
    add_candidate(fixed)
    
    # Fix 42: Handle JSON with single-quoted verdict values
    fixed = re.sub(r"('verdict'\s*:\s*)'([a-z]+)'(\s*[,}])", r'\1"\2"\3', text, flags=re.IGNORECASE)
    add_candidate(fixed)
    
    # Fix 43: Handle JSON with single-quoted assessment values
    fixed = re.sub(r"('assessment'\s*:\s*)'([a-z]+)'(\s*[,}])", r'\1"\2"\3', text, flags=re.IGNORECASE)
    add_candidate(fixed)
    
    # Fix 44: Handle JSON with single-quoted classification values
    fixed = re.sub(r"('classification'\s*:\s*)'([a-z]+)'(\s*[,}])", r'\1"\2"\3', text, flags=re.IGNORECASE)
    add_candidate(fixed)
    
    # Fix 45: Handle JSON with single-quoted decision values
    fixed = re.sub(r"('decision'\s*:\s*)'([a-z]+)'(\s*[,}])", r'\1"\2"\3', text, flags=re.IGNORECASE)
    add_candidate(fixed)
    
    # Fix 46: Handle JSON with single-quoted answer values
    fixed = re.sub(r"('answer'\s*:\s*)'([a-z]+)'(\s*[,}])", r'\1"\2"\3', text, flags=re.IGNORECASE)
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
    # Priority order: more specific patterns first
    grade_patterns = [
        # Highest priority: explicit "grade is X" patterns
        rf'\bgrade\s+is\s+["\']?({labels_pattern})["\']?\b',
        rf'\bthe\s+grade\s+is\s+["\']?({labels_pattern})["\']?\b',
        rf'\bfinal\s+grade\s+is\s+["\']?({labels_pattern})["\']?\b',
        rf'\bassigned\s+grade\s+is\s+["\']?({labels_pattern})["\']?\b',
        # Second priority: colon/equals assignment patterns
        rf'\bgrade\s*[:=]\s*["\']?({labels_pattern})["\']?\b',
        rf'\bassigned\s+grade\s*[:=]\s*["\']?({labels_pattern})["\']?\b',
        rf'\bfinal\s+grade\s*[:=]\s*["\']?({labels_pattern})["\']?\b',
        rf'\bevaluation\s*[:=]\s*["\']?({labels_pattern})["\']?\b',
        rf'\bresult\s*[:=]\s*["\']?({labels_pattern})["\']?\b',
        rf'\bresponse\s*[:=]\s*["\']?({labels_pattern})["\']?\b',
        rf'\blabel\s*[:=]\s*["\']?({labels_pattern})["\']?\b',
        # Third priority: conclusion/determination patterns
        rf'\btherefore\s*,?\s*(?:the\s+)?(?:grade|evaluation|answer)\s+is\s+["\']?({labels_pattern})["\']?\b',
        rf'\bconclusion\s*:?\s*["\']?({labels_pattern})["\']?\b',
        rf'\bdetermination\s*:?\s*["\']?({labels_pattern})["\']?\b',
        rf'\bverdict\s*:?\s*["\']?({labels_pattern})["\']?\b',
        rf'\bassessment\s*:?\s*["\']?({labels_pattern})["\']?\b',
        rf'\bclassification\s*:?\s*["\']?({labels_pattern})["\']?\b',
        rf'\bdecision\s*:?\s*["\']?({labels_pattern})["\']?\b',
        # Fourth priority: action patterns (award/assign/rate)
        rf'\baward\s+["\']?({labels_pattern})["\']?\b',
        rf'\bassign\s+["\']?({labels_pattern})["\']?\b',
        rf'\brate\s+as\s+["\']?({labels_pattern})["\']?\b',
        rf'\bscore\s+as\s+["\']?({labels_pattern})["\']?\b',
        rf'\bmark\s+as\s+["\']?({labels_pattern})["\']?\b',
        rf'\bgrade[d]?\s+["\']?({labels_pattern})["\']?\b',
        # Fifth priority: appropriateness patterns
        rf'\b["\']?({labels_pattern})["\']?\s+is\s+appropriate\b',
        rf'\b["\']?({labels_pattern})["\']?\s+is\s+the\s+(?:grade|evaluation)\b',
        rf'\b["\']?({labels_pattern})["\']?\s+is\s+assigned\b',
        rf'\b["\']?({labels_pattern})["\']?\s*grade\b',
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
    
    # Strategy 2: Look for label in quotes/brackets after specific keywords
    # These are high-confidence patterns
    quote_patterns = [
        rf'"grade"\s*:\s*"({labels_pattern})"',
        rf"'grade'\s*:\s*'({labels_pattern})'",
        rf'"label"\s*:\s*"({labels_pattern})"',
        rf"'label'\s*:\s*'({labels_pattern})'",
        rf'"evaluation"\s*:\s*"({labels_pattern})"',
        rf"'evaluation'\s*:\s*'({labels_pattern})'",
        rf'"result"\s*:\s*"({labels_pattern})"',
        rf"'result'\s*:\s*'({labels_pattern})'",
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
    # This ensures we don't match "partial" when "almost" is present
    for label in GRADE_HIERARCHY:
        if re.search(rf'\b{label}\b', text_lower):
            return label
    
    # Strategy 4: Look for grade indicators in reasoning text
    # Check for phrases that indicate a specific grade
    # These are lower confidence but can help when explicit labels are missing
    indicator_patterns = {
        "correct": [
            r'\bfull\s+(?:marks?|credit|score|points?)\b',
            r'\bcomplete\s+solution\b',
            r'\bcorrect\s+(?:proof|solution|answer)\b',
            r'\b7\s*/\s*7\b',
            r'\bfull\s+marks?\s*\(?7\b',
            r'\bsolution\s+is\s+(?:correct|valid|complete)\b',
            r'\bproof\s+is\s+(?:correct|valid|complete)\b',
            r'\bfully\s+correct\b',
            r'\bentirely\s+correct\b',
            r'\bperfect\s+solution\b',
        ],
        "almost": [
            r'\bminor\s+(?:gap|error|mistake|issue)\b',
            r'\bnearly\s+complete\b',
            r'\balmost\s+(?:correct|complete|there)\b',
            r'\b5\s*/\s*7\b',
            r'\b6\s*/\s*7\b',
            r'\bsmall\s+(?:gap|error|omission)\b',
            r'\bvery\s+close\s+to\b',
            r'\btiny\s+(?:error|mistake)\b',
            r'\bminimal\s+(?:gap|error)\b',
        ],
        "partial": [
            r'\bsignificant\s+(?:progress|work|insight)\b',
            r'\bpartial\s+(?:credit|marks?|solution)\b',
            r'\bkey\s+lemma\b',
            r'\bmeaningful\s+progress\b',
            r'\b[1-4]\s*/\s*7\b',
            r'\bsubstantial\s+progress\b',
            r'\bsome\s+progress\b',
            r'\bincomplete\s+but\s+valid\b',
        ],
        "incorrect": [
            r'\bno\s+(?:valid|meaningful)\s+(?:progress|solution|work)\b',
            r'\bfundamentally\s+flawed\b',
            r'\bwrong\s+approach\b',
            r'\b0\s*/\s*7\b',
            r'\bno\s+credit\b',
            r'\bdoes\s+not\s+(?:solve|address)\b',
            r'\bcompletely\s+wrong\b',
            r'\bentirely\s+incorrect\b',
        ],
    }
    
    # Count matches for each grade to handle multiple indicators
    grade_scores = {grade: 0 for grade in VALID_LABELS}
    for label, patterns in indicator_patterns.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                grade_scores[label] += 1
    
    # Find the grade with the highest score
    max_score = max(grade_scores.values())
    if max_score > 0:
        # Return the first grade with max score (following hierarchy priority)
        for grade in GRADE_HIERARCHY:
            if grade_scores[grade] == max_score:
                return grade
    
    return None


def _extract_grade_from_reasoning(text: str) -> str | None:
    """Extract grade from reasoning text by analyzing the content.
    
    This function looks for linguistic patterns in the reasoning that indicate
    the grader's assessment, even when explicit grade labels aren't present.
    Uses weighted scoring to handle conflicting indicators.
    """
    if not text or not isinstance(text, str):
        return None
    
    text_lower = text.lower()
    
    # Weighted indicators for each grade - higher weight = stronger indicator
    grade_indicators = {
        "correct": [
            # Very strong indicators (weight 3)
            (r'\b7\s*/\s*7\b', 3),
            (r'\bfull\s+marks?\s*\(?7\b', 3),
            (r'\bfull\s+(?:marks?|credit|score|points?)\b', 3),
            # Strong indicators (weight 2)
            (r'\bcomplete\s+(?:proof|solution|answer)\b', 2),
            (r'\bcorrect\s+(?:proof|solution|answer|conclusion)\b', 2),
            (r'\bsolution\s+is\s+(?:correct|valid|complete)\b', 2),
            (r'\bproof\s+is\s+(?:correct|valid|complete)\b', 2),
            (r'\bfully\s+(?:correct|solved|proven)\b', 2),
            (r'\bperfect\s+solution\b', 2),
            # Medium indicators (weight 1)
            (r'\ball\s+necessary\s+steps\b', 1),
            (r'\bmeets\s+all\s+requirements\b', 1),
            (r'\bcompletely\s+correct\b', 1),
            (r'\bentirely\s+correct\b', 1),
            (r'\bwholly\s+correct\b', 1),
            (r'\bperfectly\s+valid\b', 1),
            (r'\bfully\s+justified\b', 1),
        ],
        "almost": [
            # Very strong indicators (weight 3)
            (r'\b5\s*/\s*7\b', 3),
            (r'\b6\s*/\s*7\b', 3),
            # Strong indicators (weight 2)
            (r'\bminor\s+(?:gap|error|mistake|issue|flaw)\b', 2),
            (r'\bnearly\s+complete\b', 2),
            (r'\balmost\s+(?:correct|complete|there|perfect)\b', 2),
            (r'\bvery\s+close\s+to\s+(?:correct|complete)\b', 2),
            (r'\bmostly\s+correct\b', 2),
            # Medium indicators (weight 1)
            (r'\bsmall\s+(?:gap|error|omission|issue)\b', 1),
            (r'\btiny\s+(?:error|mistake|gap)\b', 1),
            (r'\bjust\s+(?:missing|needs)\b', 1),
            (r'\bminor\s+correction\b', 1),
            (r'\bslight\s+(?:error|issue|problem)\b', 1),
            (r'\bminimal\s+(?:gap|error|issue)\b', 1),
            (r'\bnegligible\s+(?:error|issue)\b', 1),
            (r'\bnearly\s+there\b', 1),
        ],
        "partial": [
            # Very strong indicators (weight 3)
            (r'\b[1-4]\s*/\s*7\b', 3),
            # Strong indicators (weight 2)
            (r'\bsignificant\s+(?:progress|work|insight|contribution)\b', 2),
            (r'\bpartial\s+(?:credit|marks?|solution|progress)\b', 2),
            (r'\bkey\s+lemma\b', 2),
            (r'\bmeaningful\s+progress\b', 2),
            (r'\bsubstantial\s+progress\b', 2),
            (r'\bnon-trivial\s+progress\b', 2),
            (r'\bvaluable\s+insight\b', 2),
            # Medium indicators (weight 1)
            (r'\bgood\s+insight\b', 1),
            (r'\bcorrect\s+approach\s+but\b', 1),
            (r'\bsolved\s+(?:a|some|part)\b', 1),
            (r'\bincomplete\s+but\s+(?:valid|correct|good)\b', 1),
            (r'\bpartially\s+(?:correct|complete|solved)\b', 1),
            (r'\bsome\s+progress\b', 1),
            (r'\bimportant\s+step\b', 1),
        ],
        "incorrect": [
            # Very strong indicators (weight 3)
            (r'\b0\s*/\s*7\b', 3),
            (r'\bno\s+credit\b', 3),
            (r'\bfundamentally\s+flawed\b', 3),
            # Strong indicators (weight 2)
            (r'\bno\s+(?:valid|meaningful)\s+(?:progress|solution|work)\b', 2),
            (r'\bwrong\s+approach\b', 2),
            (r'\bcompletely\s+wrong\b', 2),
            (r'\bentirely\s+incorrect\b', 2),
            (r'\bwholly\s+incorrect\b', 2),
            (r'\bno\s+real\s+progress\b', 2),
            (r'\bfails\s+to\s+(?:solve|address|prove)\b', 2),
            # Medium indicators (weight 1)
            (r'\bdoes\s+not\s+(?:solve|address|answer)\b', 1),
            (r'\bincorrect\s+(?:approach|method|reasoning)\b', 1),
            (r'\bonly\s+restates\s+(?:the\s+)?problem\b', 1),
            (r'\bno\s+valid\s+mathematical\s+content\b', 1),
            (r'\btrivial\s+observations?\b', 1),
        ],
    }
    
    # Calculate weighted scores for each grade
    grade_scores = {grade: 0 for grade in VALID_LABELS}
    
    for grade, patterns in grade_indicators.items():
        for pattern, weight in patterns:
            if re.search(pattern, text_lower):
                grade_scores[grade] += weight
    
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
    
    # Handle case where grade is in various delimiters
    delimiter_patterns = [
        (r'\((correct|almost|partial|incorrect)\)', 'parentheses'),
        (r'\[(correct|almost|partial|incorrect)\]', 'brackets'),
        (r'\{(correct|almost|partial|incorrect)\}', 'braces'),
        (r'<(correct|almost|partial|incorrect)>', 'angle brackets'),
        (r'"(correct|almost|partial|incorrect)"', 'double quotes'),
        (r"'(correct|almost|partial|incorrect)'", 'single quotes'),
        (r'`(correct|almost|partial|incorrect)`', 'backticks'),
    ]
    
    for pattern, _ in delimiter_patterns:
        match = re.search(pattern, text_lower)
        if match:
            return match.group(1)
    
    # Handle case where grade is at the start or end with punctuation
    for label in VALID_LABELS:
        # At the start followed by punctuation or whitespace
        if re.search(rf'^{label}\b[.!?,;:\s]', text_lower):
            return label
        # At the end preceded by punctuation or whitespace
        if re.search(rf'[.!?,;:\s]\b{label}$', text_lower):
            return label
        # Just the label at start or end
        if text_lower.startswith(label + ' ') or text_lower.endswith(' ' + label):
            return label
    
    # Handle case where grade is in a sentence with various patterns
    sentence_patterns = [
        # Direct assignment patterns
        rf'\bthe\s+grade\s+is\s+(?:"|\'|`)?(correct|almost|partial|incorrect)(?:"|\'|`)?',
        rf'\bthe\s+answer\s+is\s+(?:"|\'|`)?(correct|almost|partial|incorrect)(?:"|\'|`)?',
        rf'\bthe\s+evaluation\s+is\s+(?:"|\'|`)?(correct|almost|partial|incorrect)(?:"|\'|`)?',
        rf'\bthe\s+result\s+is\s+(?:"|\'|`)?(correct|almost|partial|incorrect)(?:"|\'|`)?',
        rf'\bthe\s+verdict\s+is\s+(?:"|\'|`)?(correct|almost|partial|incorrect)(?:"|\'|`)?',
        rf'\bthe\s+assessment\s+is\s+(?:"|\'|`)?(correct|almost|partial|incorrect)(?:"|\'|`)?',
        # Student receives patterns
        rf'\bthe\s+student\s+(?:gets|receives|earns)\s+(?:a\s+)?(?:"|\'|`)?(correct|almost|partial|incorrect)(?:"|\'|`)?',
        rf'\bstudent\s+(?:gets|receives|earns)\s+(?:a\s+)?(?:"|\'|`)?(correct|almost|partial|incorrect)(?:"|\'|`)?',
        # Grader action patterns
        rf'\bi\s+(?:would\s+)?(?:assign|give|award)\s+(?:a\s+)?(?:"|\'|`)?(correct|almost|partial|incorrect)(?:"|\'|`)?',
        rf'\b(?:assign|give|award)\s+(?:a\s+)?(?:"|\'|`)?(correct|almost|partial|incorrect)(?:"|\'|`)?(?:\s+grade)?',
        # Classification patterns
        rf'\b(?:this|the\s+solution)\s+is\s+(?:"|\'|`)?(correct|almost|partial|incorrect)(?:"|\'|`)?',
        rf'\bthis\s+(?:should\s+)?(?:be|is)\s+(?:"|\'|`)?(correct|almost|partial|incorrect)(?:"|\'|`)?',
        # Grade as object patterns
        rf'\bgrade\s*[=:]\s*(?:"|\'|`)?(correct|almost|partial|incorrect)(?:"|\'|`)?',
        rf'\blabel\s*[=:]\s*(?:"|\'|`)?(correct|almost|partial|incorrect)(?:"|\'|`)?',
        rf'\bevaluation\s*[=:]\s*(?:"|\'|`)?(correct|almost|partial|incorrect)(?:"|\'|`)?',
        rf'\bresult\s*[=:]\s*(?:"|\'|`)?(correct|almost|partial|incorrect)(?:"|\'|`)?',
        # Conclusion patterns
        rf'\btherefore\s*,?\s*(?:the\s+)?(?:grade|evaluation|answer|result)\s+is\s+(?:"|\'|`)?(correct|almost|partial|incorrect)(?:"|\'|`)?',
        rf'\bthus\s*,?\s*(?:the\s+)?(?:grade|evaluation|answer|result)\s+is\s+(?:"|\'|`)?(correct|almost|partial|incorrect)(?:"|\'|`)?',
        rf'\bhence\s*,?\s*(?:the\s+)?(?:grade|evaluation|answer|result)\s+is\s+(?:"|\'|`)?(correct|almost|partial|incorrect)(?:"|\'|`)?',
        # Final/Overall patterns
        rf'\bfinal\s+(?:grade|evaluation|assessment)\s*[=:]?\s*(?:"|\'|`)?(correct|almost|partial|incorrect)(?:"|\'|`)?',
        rf'\boverall\s+(?:grade|evaluation|assessment)\s*[=:]?\s*(?:"|\'|`)?(correct|almost|partial|incorrect)(?:"|\'|`)?',
    ]
    
    for pattern in sentence_patterns:
        match = re.search(pattern, text_lower)
        if match:
            return match.group(1)
    
    # Handle JSON-like fragments that might be malformed
    json_fragments = [
        rf'"grade"\s*:\s*"(correct|almost|partial|incorrect)"',
        rf"'grade'\s*:\s*'(correct|almost|partial|incorrect)'",
        rf'"label"\s*:\s*"(correct|almost|partial|incorrect)"',
        rf"'label'\s*:\s*'(correct|almost|partial|incorrect)'",
        rf'"evaluation"\s*:\s*"(correct|almost|partial|incorrect)"',
        rf"'evaluation'\s*:\s*'(correct|almost|partial|incorrect)'",
        rf'"result"\s*:\s*"(correct|almost|partial|incorrect)"',
        rf"'result'\s*:\s*'(correct|almost|partial|incorrect)'",
    ]
    
    for pattern in json_fragments:
        match = re.search(pattern, text_lower)
        if match:
            return match.group(1)
    
    # Handle case where grade appears after "grade:" or similar
    colon_patterns = [
        rf'grade\s*:\s*(correct|almost|partial|incorrect)\b',
        rf'label\s*:\s*(correct|almost|partial|incorrect)\b',
        rf'evaluation\s*:\s*(correct|almost|partial|incorrect)\b',
        rf'result\s*:\s*(correct|almost|partial|incorrect)\b',
        rf'verdict\s*:\s*(correct|almost|partial|incorrect)\b',
        rf'assessment\s*:\s*(correct|almost|partial|incorrect)\b',
    ]
    
    for pattern in colon_patterns:
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
    
    # Store original for pattern matching
    original_value = value
    
    value = value.lower().strip()
    
    # Remove common prefixes/suffixes that might appear
    value = re.sub(r'^(is\s+|graded\s+as\s+|marked\s+as\s+|assigned\s+|final\s+|the\s+|a\s+|an\s+)', '', value)
    value = re.sub(r'\s*(grade|score|mark|evaluation|result|classification|verdict|assessment)$', '', value)
    
    # Remove surrounding quotes and punctuation
    value = value.strip('"\'`,;:.!()[]{}<> ')
    
    # Handle double-quoted strings (common LLM error)
    value = re.sub(r'^"+', '', value)
    value = re.sub(r'"+$', '', value)
    
    # Handle single-quoted strings
    value = re.sub(r"^'+", '', value)
    value = re.sub(r"'+$", '', value)
    
    # Handle backtick-quoted strings
    value = re.sub(r'^`+', '', value)
    value = re.sub(r'`+$', '', value)
    
    # Handle curly quotes
    value = value.replace('"', '').replace('"', '').replace(''', '').replace(''', '')
    
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
            'full credit', '100%', 'seven', 'full score', 'correctanswer', 'rightanswer',
            'validsolution', 'completeproof', 'fullsolution'
        ],
        "almost": [
            'nearly', 'close', 'mostly', 'almost there', 'minor gap', 'small error', 
            '5/7', '6/7', 'almost correct', 'nearly complete', 'minor issue', 'tiny error',
            'slight mistake', 'minimal gap', 'five', 'six', 'near complete', 'almostcorrect',
            'nearlycomplete', 'mostlycorrect', 'minorerror', 'smallgap'
        ],
        "partial": [
            'partially', 'part', 'some', 'incomplete', 'half', 'partial credit', 'progress', 
            'attempt', '1/7', '2/7', '3/7', '4/7', 'started', 'beginning', 'in progress',
            'one', 'two', 'three', 'four', 'partial solution', 'significant progress',
            'meaningful work', 'substantial progress', 'partialcredit', 'partiallycorrect',
            'incompletebutvalid', 'someprogress', 'keylemma'
        ],
        "incorrect": [
            'wrong', 'false', 'invalid', 'error', '0', '0/7', 'no credit', 'none', 'fail', 
            'failed', 'unsolved', 'no solution', 'zero', 'incorrect answer', 'wrong answer',
            'invalid solution', 'no marks', 'zero marks', 'failed attempt', 'unsuccessful',
            'wronganswer', 'incorrectanswer', 'invalidsolution', 'nosolution', 'fundamentallyflawed'
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
    
    # Handle single digit scores (0-7)
    single_digit_match = re.search(r'\b([0-7])\b', value)
    if single_digit_match:
        score = int(single_digit_match.group(1))
        if score == 7:
            return "correct"
        elif score >= 5:
            return "almost"
        elif score >= 1:
            return "partial"
        else:
            return "incorrect"
    
    # Try to extract from original value if it contains JSON-like patterns
    # This handles cases where the value might be a malformed JSON string
    json_grade_match = re.search(r'["\']?(grade|label|evaluation|result|response)["\']?\s*:\s*["\']?(correct|almost|partial|incorrect)["\']?', original_value, re.IGNORECASE)
    if json_grade_match:
        return json_grade_match.group(2).lower()
    
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
        r'^\s*don\'t\s+know\s*$',
        r'^\s*do\s+not\s+know\s*$',
        r'^\s*no\s+idea\s*$',
        r'^\s*not\s+sure\s*$',
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
        r'^\s*0\s*$',    # Just zero
        r'^\s*-\s*\d+\s*$',  # Just negative numbers
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
    
    # Check for placeholder text patterns (expanded)
    placeholder_patterns = [
        r'\[insert.*?\]',
        r'\[your answer.*?\]',
        r'\[solution.*?\]',
        r'\[work.*?\]',
        r'\[proof.*?\]',
        r'\[answer.*?\]',
        r'<insert.*?>',
        r'<your answer.*?>',
        r'<solution.*?>',
        r'<answer.*?>',
        r'\(insert.*?\)',
        r'\(your answer.*?\)',
        r'\(solution.*?\)',
        r'\(answer.*?\)',
        r'answer\s+goes\s+here',
        r'solution\s+goes\s+here',
        r'type\s+your\s+answer',
        r'enter\s+your\s+answer',
        r'write\s+your\s+solution',
        r'write\s+your\s+answer',
        r'put\s+your\s+answer',
        r'your\s+answer\s+here',
        r'your\s+solution\s+here',
        r'\$\\?boxed\{.*?\}',  # LaTeX boxed placeholder
        r'\\?boxed\{.*?\}',   # Boxed placeholder
        r'\$\$.*?\$\$',        # Empty math block
        r'\$\s*\$',            # Empty inline math
    ]
    
    for pattern in placeholder_patterns:
        if re.search(pattern, stripped, re.IGNORECASE):
            return False, "placeholder_text"
    
    # Check for common non-answer content
    non_answer_patterns = [
        r'^\s*see\s+attached\s*$',
        r'^\s*see\s+image\s*$',
        r'^\s*see\s+figure\s*$',
        r'^\s*see\s+above\s*$',
        r'^\s*see\s+below\s*$',
        r'^\s*refer\s+to\s*$',
        r'^\s*check\s+attachment\s*$',
        r'^\s*image\s+only\s*$',
        r'^\s*diagram\s+only\s*$',
    ]
    
    for pattern in non_answer_patterns:
        if re.search(pattern, stripped, re.IGNORECASE):
            return False, "non_answer_reference"
    
    # Check for mathematical content indicators
    # If there's LaTeX math, it's likely a real attempt
    math_indicators = [
        r'\$[^$]+\$',           # Inline math
        r'\$\$[^$]+\$\$',       # Display math
        r'\\\[.*?\\\]',        # LaTeX display math
        r'\\\(.*?\\\)',        # LaTeX inline math
        r'\\[a-zA-Z]+\{',      # LaTeX commands
        r'\\frac\{',           # Fractions
        r'\\sum',              # Sums
        r'\\int',              # Integrals
        r'\\sqrt',             # Square roots
        r'\\[a-zA-Z]+\^\{',    # Exponents
        r'\\[a-zA-Z]+_',       # Subscripts
    ]
    
    has_math = any(re.search(pattern, stripped) for pattern in math_indicators)
    
    # If it has mathematical content, it's likely valid even if short
    if has_math and len(content_only) >= 5:
        return True, "valid_math_content"
    
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

## KEY DISTINCTIONS (CRITICAL):

### "almost" vs "partial" - MOST COMMONLY CONFUSED:
- **"almost"**: The student is VERY CLOSE to the full solution. The fix would be SIMPLE (e.g., adding one small case, fixing a minor computation, filling a tiny logical gap). The main result is essentially proven.
- **"partial"**: Significant work REMAINS. The student made good progress but there's still a SUBSTANTIAL gap to a complete solution. Multiple steps or major insights are still missing.

**Decision rule**: If the fix would take less than 5 minutes to add → "almost". If more work is needed → "partial".

### "partial" vs "incorrect":
- **"partial"**: The student made GENUINE mathematical progress - found a key lemma, proved a useful property, solved a special case, or made substantial progress. There is real mathematical content beyond trivial observations.
- **"incorrect"**: Little to no valid progress. The approach is fundamentally wrong, or the answer only restates the problem, or contains only trivial observations (e.g., "let's try small cases" without actually solving any).

**Decision rule**: If there's any non-trivial correct mathematical insight → "partial". If everything is wrong or trivial → "incorrect".

## COMMON PITFALLS TO AVOID:

1. **Don't over-grade "almost"**: A solution with significant gaps should not be "almost" - use "partial" instead. "Almost" means the student is 90%+ done.
2. **Don't under-grade "partial"**: If the student made genuine mathematical progress (found a key lemma, solved a special case, proved a useful property), use "partial" not "incorrect".
3. **Check for completeness**: "correct" requires ALL necessary steps, not just the right idea.
4. **Watch for restatements**: Answers that only restate the problem without new mathematical content are "incorrect".
5. **Consider partial credit**: Even incomplete solutions with good insights deserve "partial".
6. **Be decisive**: Choose ONE grade. Do not hedge or give ambiguous assessments.
7. **Avoid grade inflation**: When in doubt between two grades, choose the LOWER grade. It's better to be slightly conservative.

## RESPONSE FORMAT (STRICT JSON REQUIRED - READ CAREFULLY):

You MUST respond with ONLY a JSON object wrapped in <json> tags. Do NOT include any other text, markdown formatting, or explanations outside the JSON tags.

<json>
{{
    "reasoning": "Detailed explanation of your evaluation including specific strengths and weaknesses found. Cite specific parts of the student's answer. Explain why you chose this specific grade over the alternatives.",
    "grade": "correct"
}}
</json>

CRITICAL INSTRUCTIONS FOR JSON FORMAT:
1. The "grade" field MUST be exactly one of: "correct", "almost", "partial", or "incorrect" (all lowercase)
2. The "grade" value must be a plain string WITHOUT extra quotes - write: "grade": "correct" NOT "grade": ""correct"" or "grade": "'correct'"
3. Do NOT include any text before or after the <json> tags
4. Do NOT use markdown formatting like ```json inside the <json> tags
5. The JSON must be valid - ensure proper escaping of quotes and newlines within the reasoning string
6. Double-check that your grade is one of the four valid options before responding
7. Be conservative with "correct" - only use it for truly complete solutions
8. The reasoning should be detailed and reference specific parts of the student's work
9. When deciding between grades, remember: "almost" requires 90%+ completion, "partial" requires genuine insight, "incorrect" is for fundamentally wrong or trivial answers

## DETAILED GRADING EXAMPLES:

**Example of "correct"**: Student proves all cases, handles all edge cases, provides rigorous justification for each step, and arrives at the correct answer with a complete logical chain. The solution is publication-ready.

**Example of "almost"**: Student has the right approach, proves the main case, but misses one small sub-case or has a minor computational error that doesn't affect the overall approach. The fix would be simple and quick to add.

**Example of "partial"**: Student identifies the key lemma or makes significant progress on the problem (e.g., solves a special case, finds a useful invariant, proves a necessary condition) but doesn't complete the full solution. There is genuine mathematical insight but substantial work remains.

**Example of "incorrect"**: Student's approach is completely wrong, or they only rewrite the problem statement without adding mathematical content, or they make trivial observations that don't advance toward the solution. No valid mathematical progress was made.

Example valid responses (COPY THIS EXACT FORMAT):
<json>{{"reasoning": "The student provided a complete proof with all necessary steps, handling all edge cases and providing rigorous justification for each claim. The solution would receive full marks (7/7) in an IMO competition.", "grade": "correct"}}</json>
<json>{{"reasoning": "The solution is nearly complete with only a minor case missing. The student has the right approach and proves the main result, but overlooked one small sub-case that could be easily fixed.", "grade": "almost"}}</json>
<json>{{"reasoning": "The student found a key lemma (the invariance property) and proved it correctly, but didn't complete the full solution. This represents genuine mathematical progress beyond trivial observations.", "grade": "partial"}}</json>
<json>{{"reasoning": "The approach is fundamentally flawed. The student incorrectly assumes the function is linear without justification, leading to an invalid conclusion. No meaningful mathematical progress was made.", "grade": "incorrect"}}</json>"""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction using multiple strategies with confidence scoring
        prediction = "None"
        extraction_log = []
        confidence_scores = {}  # Track confidence for each potential grade
        
        try:
            raw_text = msg_history[-1]["text"] if msg_history else ""
            
            if not raw_text:
                self.log_fn("Empty response from LLM")
                return "incorrect", msg_history
            
            # Pre-process: clean up common LLM output artifacts
            cleaned_text = raw_text.strip()
            # Remove common prefixes that LLMs add
            cleaned_text = re.sub(r'^(?:Here is|Here\'s|The|My|This is|I\'ll provide|Let me provide)\s+(?:the\s+)?(?:json\s+)?(?:response|answer|grade|evaluation|result)[:;]?\s*', '', cleaned_text, flags=re.IGNORECASE)
            
            # Strategy 1: Try to extract from JSON tags (highest confidence)
            extracted = _extract_jsons(cleaned_text)
            if extracted:
                extraction_log.append(f"JSON extraction found {len(extracted)} objects")
                # Sort by likelihood of being the main grade object (prefer objects with 'grade' key)
                sorted_extracted = sorted(extracted, key=lambda x: 0 if 'grade' in x else 1)
                for json_obj in sorted_extracted:
                    # Try multiple possible keys in order of preference
                    for key in ["grade", "label", "evaluation", "result", "response", 
                               "score", "verdict", "assessment", "classification", "decision", "answer"]:
                        if key in json_obj:
                            val = json_obj[key]
                            if val is not None:
                                # Handle case where value might be a nested structure
                                if isinstance(val, str):
                                    val_str = val
                                else:
                                    val_str = str(val)
                                normalized = _normalize_grade(val_str)
                                if normalized:
                                    # JSON extraction has highest confidence
                                    confidence_scores[normalized] = confidence_scores.get(normalized, 0) + 10
                                    if prediction == "None":
                                        prediction = normalized
                                        extraction_log.append(f"Found grade '{prediction}' from key '{key}' (JSON)")
                                    break
                    if prediction != "None":
                        break
            else:
                extraction_log.append("JSON extraction found no objects")
            
            # Strategy 2: Try text extraction with explicit grade patterns
            text_pred = _extract_label_from_text(cleaned_text)
            if text_pred:
                confidence_scores[text_pred] = confidence_scores.get(text_pred, 0) + 8
                if prediction == "None":
                    prediction = text_pred
                    extraction_log.append(f"Found grade '{prediction}' from text extraction")
                elif prediction != text_pred:
                    extraction_log.append(f"Text extraction suggests '{text_pred}' (conflict with '{prediction}')")
            
            # Strategy 3: Look for any valid label in the text (lower confidence)
            text_lower = cleaned_text.lower()
            for label in GRADE_HIERARCHY:
                if re.search(rf'\b{label}\b', text_lower):
                    confidence_scores[label] = confidence_scores.get(label, 0) + 5
                    if prediction == "None":
                        prediction = label
                        extraction_log.append(f"Found grade '{prediction}' from fallback search")
                    break
            
            # Strategy 4: Try to extract from reasoning text patterns
            reasoning_pred = _extract_grade_from_reasoning(cleaned_text)
            if reasoning_pred:
                confidence_scores[reasoning_pred] = confidence_scores.get(reasoning_pred, 0) + 6
                if prediction == "None":
                    prediction = reasoning_pred
                    extraction_log.append(f"Found grade '{prediction}' from reasoning analysis")
                elif prediction != reasoning_pred:
                    extraction_log.append(f"Reasoning analysis suggests '{reasoning_pred}'")
            
            # Strategy 5: Try edge case extraction for malformed responses
            edge_pred = _extract_grade_from_edge_cases(cleaned_text)
            if edge_pred:
                confidence_scores[edge_pred] = confidence_scores.get(edge_pred, 0) + 4
                if prediction == "None":
                    prediction = edge_pred
                    extraction_log.append(f"Found grade '{prediction}' from edge case analysis")
            
            # Strategy 6: Apply keyword-based disambiguation for commonly confused grades
            prediction = _apply_keyword_disambiguation(raw_text, confidence_scores, prediction, extraction_log)
            
            # Strategy 7: Check for grade in reasoning field if present in JSON
            if extracted:
                for json_obj in extracted:
                    if "reasoning" in json_obj and isinstance(json_obj["reasoning"], str):
                        reasoning_text = json_obj["reasoning"].lower()
                        # Look for explicit grade mentions in reasoning
                        for label in GRADE_HIERARCHY:
                            if re.search(rf'\b{label}\b', reasoning_text):
                                confidence_scores[label] = confidence_scores.get(label, 0) + 3
                                extraction_log.append(f"Found '{label}' mentioned in reasoning field")
                                break
            
            # Strategy 8: Direct pattern matching for grade in various formats
            # Look for patterns like "grade": "correct" or grade: correct
            direct_patterns = [
                r'"grade"\s*:\s*"(correct|almost|partial|incorrect)"',
                r"'grade'\s*:\s*'(correct|almost|partial|incorrect)'",
                r'grade\s*:\s*(correct|almost|partial|incorrect)\b',
                r'"label"\s*:\s*"(correct|almost|partial|incorrect)"',
                r"'label'\s*:\s*'(correct|almost|partial|incorrect)'",
                r'label\s*:\s*(correct|almost|partial|incorrect)\b',
            ]
            for pattern in direct_patterns:
                match = re.search(pattern, text_lower)
                if match:
                    direct_grade = match.group(1)
                    confidence_scores[direct_grade] = confidence_scores.get(direct_grade, 0) + 9
                    if prediction == "None":
                        prediction = direct_grade
                        extraction_log.append(f"Found grade '{prediction}' from direct pattern match")
                    break
            
            # If we have conflicting predictions, use confidence scores to resolve
            if len(confidence_scores) > 1:
                # Find the grade with highest confidence
                best_grade = max(confidence_scores.keys(), key=lambda k: confidence_scores[k])
                if best_grade != prediction:
                    extraction_log.append(f"Confidence resolution: '{best_grade}' (score: {confidence_scores[best_grade]}) beats '{prediction}' (score: {confidence_scores.get(prediction, 0)})")
                    prediction = best_grade
            
            # Log extraction details for debugging
            if extraction_log:
                self.log_fn(f"Grade extraction: {'; '.join(extraction_log)}")
            if confidence_scores:
                self.log_fn(f"Confidence scores: {confidence_scores}")
                    
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            import traceback
            self.log_fn(f"Traceback: {traceback.format_exc()}")

        # Default to incorrect if we couldn't extract anything
        if prediction == "None" or prediction is None:
            prediction = "incorrect"
            self.log_fn("No grade found, defaulting to 'incorrect'")
        elif prediction not in VALID_LABELS:
            # If we got an invalid grade, try to normalize it one more time
            normalized = _normalize_grade(str(prediction))
            if normalized:
                prediction = normalized
            else:
                self.log_fn(f"Invalid grade '{prediction}' found, defaulting to 'incorrect'")
                prediction = "incorrect"
            
        return str(prediction), msg_history


def _apply_keyword_disambiguation(text: str, confidence_scores: dict, current_prediction: str, extraction_log: list) -> str:
    """Apply keyword-based disambiguation to improve grade classification.
    
    This function boosts confidence scores based on strong keyword indicators
    to help resolve ambiguities between commonly confused grades.
    """
    text_lower = text.lower()
    
    # Boost confidence based on strong keyword matches
    for grade, keywords in GRADE_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text_lower:
                confidence_scores[grade] = confidence_scores.get(grade, 0) + 2
    
    # Special handling for commonly confused pairs
    # If both "almost" and "partial" have similar scores, use keyword analysis
    if "almost" in confidence_scores and "partial" in confidence_scores:
        almost_score = confidence_scores["almost"]
        partial_score = confidence_scores["partial"]
        
        # If scores are close (within 3 points), check for disambiguating keywords
        if abs(almost_score - partial_score) <= 3:
            # Check for "almost" indicators
            almost_indicators = ["minor gap", "small error", "tiny mistake", "minor omission", "nearly complete",
                                 "almost complete", "very close", "just needs", "minor correction", "small fix",
                                 "nearly there", "almost there", "minimal gap", "slight error", "tiny gap"]
            almost_count = sum(1 for ind in almost_indicators if ind in text_lower)
            
            # Check for "partial" indicators  
            partial_indicators = ["significant progress", "key lemma", "meaningful work", "substantial progress",
                                  "genuine insight", "important step", "correct approach", "solved special case",
                                  "proved lemma", "found invariant", "meaningful contribution", "non-trivial progress"]
            partial_count = sum(1 for ind in partial_indicators if ind in text_lower)
            
            if almost_count > partial_count:
                confidence_scores["almost"] += 3
                extraction_log.append("Keyword analysis favors 'almost' (minor issues)")
            elif partial_count > almost_count:
                confidence_scores["partial"] += 3
                extraction_log.append("Keyword analysis favors 'partial' (significant progress)")
    
    # Similar analysis for "partial" vs "incorrect"
    if "partial" in confidence_scores and "incorrect" in confidence_scores:
        partial_score = confidence_scores["partial"]
        incorrect_score = confidence_scores["incorrect"]
        
        if abs(partial_score - incorrect_score) <= 3:
            # Check for genuine progress indicators
            progress_indicators = ["key lemma", "useful insight", "correct approach", "solved special case",
                                   "proved lemma", "found invariant", "meaningful progress", "genuine insight",
                                   "important step", "correct observation", "valid technique", "correct method"]
            progress_count = sum(1 for ind in progress_indicators if ind in text_lower)
            
            # Check for fundamental flaw indicators
            flaw_indicators = ["fundamentally flawed", "completely wrong", "no valid progress", 
                              "trivial observation", "only restates", "wrong approach", "invalid reasoning",
                              "no real progress", "does not solve", "fails to prove", "incorrect assumption"]
            flaw_count = sum(1 for ind in flaw_indicators if ind in text_lower)
            
            if progress_count > flaw_count:
                confidence_scores["partial"] += 3
                extraction_log.append("Keyword analysis favors 'partial' (genuine progress found)")
            elif flaw_count > progress_count:
                confidence_scores["incorrect"] += 3
                extraction_log.append("Keyword analysis favors 'incorrect' (fundamental flaws found)")
    
    # Analysis for "correct" vs "almost"
    if "correct" in confidence_scores and "almost" in confidence_scores:
        correct_score = confidence_scores["correct"]
        almost_score = confidence_scores["almost"]
        
        if abs(correct_score - almost_score) <= 3:
            # Check for completeness indicators
            complete_indicators = ["complete proof", "full solution", "all cases", "fully justified",
                                   "rigorous proof", "handles all", "complete solution", "full marks"]
            complete_count = sum(1 for ind in complete_indicators if ind in text_lower)
            
            # Check for minor gap indicators
            gap_indicators = ["minor gap", "small error", "tiny mistake", "minor omission", "nearly complete",
                              "almost complete", "very close", "just needs", "minor correction", "small fix"]
            gap_count = sum(1 for ind in gap_indicators if ind in text_lower)
            
            if complete_count > gap_count:
                confidence_scores["correct"] += 3
                extraction_log.append("Keyword analysis favors 'correct' (complete solution)")
            elif gap_count > complete_count:
                confidence_scores["almost"] += 3
                extraction_log.append("Keyword analysis favors 'almost' (minor gaps found)")
    
    return current_prediction
