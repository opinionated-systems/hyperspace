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

# Valid grading labels
VALID_LABELS = ["correct", "incorrect", "partial", "almost"]

# Label priority for conflict resolution (higher = more preferred when uncertain)
# For IMO grading, we want to be conservative - prefer incorrect over false positives
# But we also want to catch partial credit cases - so partial gets high priority
# "almost" gets high priority to ensure it's not confused with "partial" or "correct"
LABEL_PRIORITY = {"incorrect": 4, "almost": 3, "partial": 2, "correct": 1}

# Additional priority for specific extraction scenarios
# When we need to distinguish between similar labels
ALMOST_VS_PARTIAL_PRIORITY = {"almost": 5, "partial": 1}  # Strongly prefer "almost" when in doubt

# Additional aliases for better matching
LABEL_ALIASES_EXTENDED = {
    "correct": ["full credit", "complete solution", "fully solved", "all correct", "perfect solution"],
    "incorrect": ["no credit", "zero marks", "completely wrong", "fundamentally flawed", "unsalvageable"],
    "partial": ["some credit", "half marks", "partially correct", "partial solution", "incomplete but valid"],
    "almost": ["near complete", "essentially correct", "minor fix needed", "small omission", "nearly solved"],
}

# Confidence thresholds for different labels
CONFIDENCE_THRESHOLDS = {
    "correct": 0.85,   # High bar for correct - must be very confident
    "incorrect": 0.6,  # Lower bar for incorrect - catch pattern matching cases
    "partial": 0.5,    # Lower bar for partial - catch valid attempts
    "almost": 0.65,    # Moderate bar for almost - needs to be clearly distinguished
}

# Thresholds for distinguishing almost vs partial
ALMOST_CONFIDENCE_BOOST = 0.3  # Boost "almost" score when strong indicators present
PARTIAL_CONFIDENCE_PENALTY = 0.2  # Penalize "partial" when "almost" indicators are strong

# Pattern matching indicators - these strongly suggest "incorrect"
PATTERN_MATCHING_INDICATORS = [
    "n=1", "n=2", "n=3", "n=4", "n=5",
    "for n=", "when n=", "if n=",
    "1, 2, 3", "1,2,3", "1+2+3", "1×2×3",
    "pattern holds", "pattern continues", "the pattern",
    "checking examples", "tested values", "verified for",
    "small values", "first few", "several cases",
]

# Valid proof technique indicators - these suggest "partial" or better
VALID_TECHNIQUE_INDICATORS = [
    "proof by contradiction", "contradiction",
    "induction", "base case", "inductive step", "inductive hypothesis",
    "pigeonhole principle", "pigeonhole",
    "by contradiction", "suppose not", "assume for contradiction",
    "without loss of generality", "wlog",
    "consider the cases", "case 1", "case 2", "case analysis",
    "let x be", "suppose that", "assume that",
    "theorem", "lemma", "corollary",
    "therefore", "thus", "hence", "it follows that",
]

# Mapping from labels to their canonical form
# Note: We keep "almost" as a separate label for evaluation, but it maps to "partial" for output
LABEL_CANONICAL = {
    "correct": "correct",
    "incorrect": "incorrect", 
    "partial": "partial",
    "almost": "almost",  # Keep as separate for evaluation purposes
}

# Common misspellings and variations mapping
LABEL_ALIASES = {
    "correct": ["correct", "right", "true", "valid", "accurate", "complete", "fully correct", "entirely correct", "perfect", "solved", "full marks"],
    "incorrect": ["incorrect", "wrong", "false", "error", "invalid", "inaccurate", "flawed", "mistaken", "fundamentally wrong", "no credit", "zero"],
    "partial": ["partial", "incomplete", "partially", "some", "part", "partial credit", "half"],
    "almost": ["almost", "nearly", "mostly correct", "almost correct", "nearly complete", "almost there", "minor gap", "tiny gap", "small gap", "nearly right"],
}


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks or raw JSON.
    
    Uses multiple strategies with priority ordering to handle various output formats.
    Returns list of dicts with "response" field, or None if no valid JSON found.
    
    Enhanced to handle:
    - Multi-line JSON blocks
    - Nested JSON structures
    - Various quote styles (single, double, no quotes)
    - Common LLM output errors
    - JSON with comments or extra text
    - Malformed JSON with extra braces
    - JSON with escaped characters
    """
    if not text or not isinstance(text, str):
        return None
        
    results = []
    text_lower = text.lower()
    
    # Strategy 0: Look for JSON block at the very end of the text (most reliable for our prompt)
    # This handles cases where the JSON is the last line but may have extra whitespace
    lines = text.split('\n')
    for i in range(len(lines) - 1, -1, -1):
        line = lines[i].strip()
        if not line:
            continue
        # Check if this line contains a complete JSON block
        if '<json>' in line and '</json>' in line:
            start = line.lower().find('<json>')
            end = line.lower().find('</json>') + 7
            if start != -1 and end > start:
                inner = line[start + 6:end - 7].strip()
                parsed = _try_parse_json_with_fallbacks(inner)
                if parsed and "response" in parsed:
                    results.append(parsed)
                    return results  # Return immediately on success
    
    # Strategy 0.5: Look for JSON block at the very end of the entire text (multi-line)
    # This handles cases where JSON spans multiple lines
    full_text = '\n'.join(lines)
    # Find the last occurrence of </json>
    last_json_end = full_text.lower().rfind('</json>')
    if last_json_end != -1:
        # Find the corresponding <json> before it
        last_json_start = full_text.lower().rfind('<json>', 0, last_json_end)
        if last_json_start != -1:
            inner = full_text[last_json_start + 6:last_json_end].strip()
            parsed = _try_parse_json_with_fallbacks(inner)
            if parsed and "response" in parsed:
                results.append(parsed)
                return results  # Return immediately on success
    
    # Strategy 0.6: Look for JSON block with nested braces (multi-line with newlines inside)
    # Use regex to find <json>...content with possible nested braces...</json>
    json_block_pattern = r'<json>\s*(\{[\s\S]*?\})\s*</json>'
    json_matches = list(re.finditer(json_block_pattern, text, re.IGNORECASE | re.DOTALL))
    if json_matches:
        # Use the last match (most likely to be the final verdict)
        for match in reversed(json_matches):
            inner = match.group(1).strip()
            parsed = _try_parse_json_with_fallbacks(inner)
            if parsed and "response" in parsed:
                results.append(parsed)
                return results  # Return immediately on success
    
    # Strategy 0.7: Look for JSON with extra braces or malformed structure
    # Sometimes LLMs output: <json>{{"response": "correct"}}</json>
    malformed_json_pattern = r'<json>\s*\{+\s*"response"\s*:\s*"([^"]+)"\s*\}+\s*</json>'
    for match in re.finditer(malformed_json_pattern, text, re.IGNORECASE):
        value = match.group(1).strip().lower()
        if value in VALID_LABELS:
            results.append({"response": value})
            return results
    
    # Strategy 0.8: Look for JSON with single quotes inside <json> tags
    single_quote_json_pattern = r"<json>\s*\{\s*'response'\s*:\s*'([^']+)'\s*\}\s*</json>"
    for match in re.finditer(single_quote_json_pattern, text, re.IGNORECASE):
        value = match.group(1).strip().lower()
        if value in VALID_LABELS:
            results.append({"response": value})
            return results
    
    # Strategy 1: Try to find <json>...</json> blocks (case insensitive) - HIGHEST PRIORITY
    # Search from the END to get the most recent/final JSON block first
    search_positions = []
    search_from = 0
    while True:
        start = text_lower.find("<json>", search_from)
        if start == -1:
            break
        end = text_lower.find("</json>", start)
        if end == -1:
            break
        search_positions.append((start, end))
        search_from = end + 7
    
    # Process from last to first (most recent JSON is likely the final verdict)
    for start, end in reversed(search_positions):
        inner = text[start + 6:end].strip()
        parsed = _try_parse_json_with_fallbacks(inner)
        if parsed and "response" in parsed:
            results.append(parsed)
    
    # Strategy 2: Try to find JSON in markdown code blocks
    code_block_patterns = [
        r'```(?:json)?\s*\n?(.*?)\n?```',
        r'`\s*(\{[\s\S]*?"response"[\s\S]*?\})\s*`',
    ]
    for pattern in code_block_patterns:
        for match in re.finditer(pattern, text, re.DOTALL):
            json_str = match.group(1).strip()
            if json_str:
                parsed = _try_parse_json_with_fallbacks(json_str)
                if parsed and "response" in parsed:
                    results.append(parsed)
    
    # Strategy 3: Try to find raw JSON objects with "response" field
    json_patterns = [
        r'\{\s*"response"\s*:\s*"([^"]+)"\s*\}',
        r"\{\s*'response'\s*:\s*'([^']+)'\s*\}",
        r'\{\s*["\']?response["\']?\s*:\s*["\']?([^"\'\}\n,]+)["\']?\s*\}',
        r'\{\s*"response"\s*:\s*"(correct|incorrect|partial|almost)"\s*\}',
    ]
    for pattern in json_patterns:
        for match in re.finditer(pattern, text, re.DOTALL | re.IGNORECASE):
            parsed = _try_parse_json_with_fallbacks(match.group(0))
            if parsed and "response" in parsed:
                results.append(parsed)
    
    # Strategy 4: Look for explicit verdict statements with strong indicators
    verdict_patterns = [
        (r'(?:the\s+)?(?:final\s+)?verdict\s*(?:is|:)\s*["\']?(correct|incorrect|partial|almost)["\']?', 1),
        (r'(?:the\s+)?(?:final\s+)?(?:grade|classification|label|assessment)\s*(?:is|:)\s*["\']?(correct|incorrect|partial|almost)["\']?', 1),
        (r'(?:i\s+(?:would|will)\s+)?(?:classify|grade|label|mark)\s*(?:this\s+)?(?:as\s+)?["\']?(correct|incorrect|partial|almost)["\']?', 1),
        (r'(?:this\s+(?:is|should\s+be))\s*["\']?(correct|incorrect|partial|almost)["\']?', 1),
        (r'["\']?response["\']?\s*[:=]\s*["\']?(correct|incorrect|partial|almost)["\']?', 1),
        (r'(?:^|\n)\s*["\']?(correct|incorrect|partial|almost)["\']?\s*$', 1),  # Standalone at end
    ]
    for pattern, group_idx in verdict_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE):
            try:
                value = match.group(group_idx).strip().lower()
                if value in VALID_LABELS:
                    results.append({"response": value})
            except Exception:
                continue
    
    # Strategy 5: Look for labels in backticks, bold, or emphasized
    formatting_patterns = [
        r'`(correct|incorrect|partial|almost)`',
        r'\*\*(correct|incorrect|partial|almost)\*\*',
        r'\*(correct|incorrect|partial|almost)\*',
        r'"(correct|incorrect|partial|almost)"',
        r"'(correct|incorrect|partial|almost)'",
        r'\b(correct|incorrect|partial|almost)\b',
    ]
    for pattern in formatting_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            try:
                value = match.group(1).strip().lower()
                if value in VALID_LABELS:
                    results.append({"response": value})
            except Exception:
                continue
    
    # Strategy 6: Look for standalone labels at the end (last 5 lines)
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    if lines:
        for line in reversed(lines[-5:]):
            line_clean = line.lower().strip('"\'.,!?:;`*[]{}()')
            for label in VALID_LABELS:
                if line_clean == label:
                    results.append({"response": label})
                    break
    
    # Strategy 7: Handle malformed JSON with common LLM errors
    # Look for patterns like: {"response": "correct"} (with extra braces or missing quotes)
    malformed_patterns = [
        r'\{\s*["\']?response["\']?\s*:\s*["\']?(correct|incorrect|partial|almost)["\']?\s*\}',
        r'response\s*:\s*["\']?(correct|incorrect|partial|almost)["\']?',
        r'["\']?response["\']?\s*[=:]\s*["\']?(correct|incorrect|partial|almost)["\']?',
    ]
    for pattern in malformed_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            try:
                # Try to extract the label from the match
                for group in match.groups():
                    if group and group.lower() in VALID_LABELS:
                        results.append({"response": group.lower()})
                        break
            except Exception:
                continue
    
    # Strategy 8: Look for JSON-like structures with single quotes
    single_quote_pattern = r"\{\s*'response'\s*:\s*'(correct|incorrect|partial|almost)'\s*\}"
    for match in re.finditer(single_quote_pattern, text, re.IGNORECASE):
        try:
            value = match.group(1).strip().lower()
            if value in VALID_LABELS:
                results.append({"response": value})
        except Exception:
            continue
    
    # Strategy 9: Look for response field in various formats
    response_field_patterns = [
        r'"response"\s*:\s*"([^"]+)"',
        r"'response'\s*:\s*'([^']+)'",
        r'response\s*:\s*(\w+)',
    ]
    for pattern in response_field_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            try:
                value = match.group(1).strip().lower()
                if value in VALID_LABELS:
                    results.append({"response": value})
            except Exception:
                continue
    
    # Remove duplicates while preserving order (last occurrence wins)
    seen = set()
    unique_results = []
    for r in reversed(results):
        resp_val = r.get("response", "")
        if resp_val and resp_val not in seen:
            seen.add(resp_val)
            unique_results.insert(0, r)
    
    return unique_results if unique_results else None


def _extract_final_label(text: str) -> str | None:
    """Extract the final grading label from LLM response text.
    
    This is a robust extraction function that tries multiple strategies
    in order of reliability to find the grading label.
    
    Enhanced with:
    - Better "almost" detection
    - Improved tie-breaking for ambiguous cases
    - Context-aware extraction
    - Better handling of malformed JSON
    - Enhanced pattern matching detection
    
    Returns one of: "correct", "incorrect", "partial", "almost", or None
    """
    if not text or not isinstance(text, str):
        return None
    
    text_lower = text.lower().strip()
    
    # Strategy 0: Look for JSON block with double braces (common LLM error)
    # Pattern: <json>{{"response": "correct"}}</json>
    double_brace_pattern = r'<json>\s*\{\s*\{\s*"response"\s*:\s*"([^"]+)"\s*\}\s*\}\s*</json>'
    match = re.search(double_brace_pattern, text, re.IGNORECASE)
    if match:
        value = match.group(1).strip().lower()
        if value in VALID_LABELS:
            return LABEL_CANONICAL.get(value, value)
    
    # Strategy 0.5: Look for JSON with single quotes
    single_quote_pattern = r"<json>\s*\{\s*'response'\s*:\s*'([^']+)'\s*\}\s*</json>"
    match = re.search(single_quote_pattern, text, re.IGNORECASE)
    if match:
        value = match.group(1).strip().lower()
        if value in VALID_LABELS:
            return LABEL_CANONICAL.get(value, value)
    
    # Strategy 1: Look for <json> block at the very end (most reliable)
    # Find the last occurrence of </json>
    last_json_end = text_lower.rfind('</json>')
    if last_json_end != -1:
        # Find the corresponding <json> before it
        last_json_start = text_lower.rfind('<json>', 0, last_json_end)
        if last_json_start != -1:
            inner = text[last_json_start + 6:last_json_end].strip()
            parsed = _try_parse_json_with_fallbacks(inner)
            if parsed and "response" in parsed:
                value = str(parsed["response"]).strip().lower()
                if value in VALID_LABELS:
                    return LABEL_CANONICAL.get(value, value)
    
    # Strategy 2: Look for JSON block with regex (handles multi-line)
    json_block_pattern = r'<json>\s*(\{[\s\S]*?\})\s*</json>'
    json_matches = list(re.finditer(json_block_pattern, text, re.IGNORECASE | re.DOTALL))
    if json_matches:
        # Use the last match
        for match in reversed(json_matches):
            inner = match.group(1).strip()
            parsed = _try_parse_json_with_fallbacks(inner)
            if parsed and "response" in parsed:
                value = str(parsed["response"]).strip().lower()
                if value in VALID_LABELS:
                    return LABEL_CANONICAL.get(value, value)
    
    # Strategy 3: Look for explicit "response" field patterns
    response_patterns = [
        r'"response"\s*:\s*"(correct|incorrect|partial|almost)"',
        r"'response'\s*:\s*'(correct|incorrect|partial|almost)'",
        r'response\s*:\s*(correct|incorrect|partial|almost)',
        r'\{\s*["\']?response["\']?\s*:\s*["\']?(correct|incorrect|partial|almost)["\']?\s*\}',
    ]
    for pattern in response_patterns:
        matches = list(re.finditer(pattern, text_lower))
        if matches:
            # Use the last match
            last_match = matches[-1]
            for group in last_match.groups():
                if group and group in VALID_LABELS:
                    return LABEL_CANONICAL.get(group, group)
    
    # Strategy 4: Look for standalone labels at the very end
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    if lines:
        # Check last 3 lines
        for line in reversed(lines[-3:]):
            line_clean = line.lower().strip('"\'.,!?:;`*[]{}()')
            if line_clean in VALID_LABELS:
                return LABEL_CANONICAL.get(line_clean, line_clean)
    
    # Strategy 5: Look for verdict/grade declarations
    verdict_patterns = [
        r'(?:final\s+)?(?:verdict|grade|assessment|classification)\s*[:=]\s*["\']?(correct|incorrect|partial|almost)',
        r'(?:the\s+answer\s+is|this\s+is)\s*["\']?(correct|incorrect|partial|almost)',
    ]
    for pattern in verdict_patterns:
        matches = list(re.finditer(pattern, text_lower))
        if matches:
            last_match = matches[-1]
            for group in last_match.groups():
                if group and group in VALID_LABELS:
                    return LABEL_CANONICAL.get(group, group)
    
    # Strategy 6: Count occurrences and use priority for tie-breaking
    label_counts = {}
    for label in VALID_LABELS:
        count = len(re.findall(rf'\b{label}\b', text_lower))
        label_counts[label] = count
    
    # Check for "almost" indicators in text to boost its score
    almost_indicators = [
        'nearly complete', 'almost complete', 'essentially correct', 'essentially complete',
        'minor gap', 'tiny gap', 'small gap', 'minor error', 'tiny error',
        'minor omission', 'tiny omission', 'small omission',
        'nearly there', 'almost there', 'almost correct', 'nearly correct',
        'mostly correct', 'just missing', 'only missing', 'forgot to state',
        'main proof is complete', 'core argument is correct',
        'proof structure is complete', 'just needs', 'only needs', 'tiny fix',
    ]
    almost_boost = 0
    for indicator in almost_indicators:
        if indicator in text_lower:
            almost_boost += 1
    # If we have strong "almost" indicators, boost its count
    if almost_boost >= 2:
        label_counts['almost'] = label_counts.get('almost', 0) + almost_boost * 0.5
    
    # Check for pattern matching indicators to boost "incorrect"
    pattern_indicators = [
        'only checked examples', 'only verified examples', 'pattern matching',
        'no general proof', 'no valid proof', 'just examples', 'only examples',
    ]
    pattern_boost = 0
    for indicator in pattern_indicators:
        if indicator in text_lower:
            pattern_boost += 1
    if pattern_boost >= 2:
        label_counts['incorrect'] = label_counts.get('incorrect', 0) + pattern_boost * 0.5
    
    if any(label_counts.values()):
        valid_counts = {k: v for k, v in label_counts.items() if v > 0}
        if valid_counts:
            max_count = max(valid_counts.values())
            candidates = [k for k, v in valid_counts.items() if v == max_count]
            if len(candidates) == 1:
                return candidates[0]
            else:
                # Use priority to break ties
                # Special handling for almost vs partial
                if 'almost' in candidates and 'partial' in candidates:
                    # Check if we have strong "almost" indicators
                    if almost_boost >= 2:
                        return 'almost'
                best = max(candidates, key=lambda x: LABEL_PRIORITY.get(x, 0))
                return best
    
    # Strategy 7: Try the old extraction methods as fallback
    extracted = _extract_jsons(text)
    if extracted:
        for item in reversed(extracted):
            if isinstance(item, dict) and "response" in item:
                value = str(item["response"]).strip().lower()
                if value in VALID_LABELS:
                    return LABEL_CANONICAL.get(value, value)
    
    # Strategy 8: Try text extraction as last resort
    text_pred = _extract_label_from_text(text)
    if text_pred:
        return LABEL_CANONICAL.get(text_pred, text_pred)
    
    # Strategy 9: Use confidence-based extraction as final fallback
    confidence_result = _confidence_based_extraction(text)
    if confidence_result:
        return LABEL_CANONICAL.get(confidence_result, confidence_result)
    
    return None


def _try_parse_json_with_fallbacks(json_str: str) -> dict | None:
    """Try to parse JSON string with multiple fallback strategies.
    
    Handles common LLM output issues like:
    - Single vs double quotes
    - Trailing commas
    - Missing quotes around keys/values
    - Extra whitespace and newlines
    - Unicode characters
    - Nested braces
    - Comments
    - Escaped characters
    - Extra braces ({{...}})
    - Missing braces
    """
    if not json_str or not isinstance(json_str, str):
        return None
    
    json_str = json_str.strip()
    
    # Try 1: Direct parsing
    try:
        parsed = json.loads(json_str)
        if isinstance(parsed, dict) and "response" in parsed:
            return parsed
    except json.JSONDecodeError:
        pass
    
    # Try 1.5: Handle double braces like {{"response": "correct"}}
    # This is common when LLMs output JSON inside template literals
    try:
        if json_str.startswith('{{') and json_str.endswith('}}'):
            fixed = json_str[1:-1]  # Remove one layer of braces
            parsed = json.loads(fixed)
            if isinstance(parsed, dict) and "response" in parsed:
                return parsed
    except (json.JSONDecodeError, IndexError):
        pass
    
    # Try 1.6: Handle extra opening/closing braces
    try:
        # Count braces and try to balance them
        open_count = json_str.count('{')
        close_count = json_str.count('}')
        if open_count > close_count:
            fixed = json_str + ('}' * (open_count - close_count))
            parsed = json.loads(fixed)
            if isinstance(parsed, dict) and "response" in parsed:
                return parsed
        elif close_count > open_count:
            fixed = ('{' * (close_count - open_count)) + json_str
            parsed = json.loads(fixed)
            if isinstance(parsed, dict) and "response" in parsed:
                return parsed
    except json.JSONDecodeError:
        pass
    
    # Try 2: Replace single quotes with double quotes (carefully)
    try:
        # Only replace single quotes that are likely JSON delimiters
        fixed = json_str.replace("'", '"')
        parsed = json.loads(fixed)
        if isinstance(parsed, dict) and "response" in parsed:
            return parsed
    except json.JSONDecodeError:
        pass
    
    # Try 3: Remove trailing commas
    try:
        fixed = re.sub(r',\s*}', '}', json_str)
        fixed = re.sub(r',\s*]', ']', fixed)
        parsed = json.loads(fixed)
        if isinstance(parsed, dict) and "response" in parsed:
            return parsed
    except json.JSONDecodeError:
        pass
    
    # Try 4: Fix common JSON syntax errors
    try:
        # Remove comments
        fixed = re.sub(r'//.*?\n', '\n', json_str)
        fixed = re.sub(r'/\*.*?\*/', '', fixed, flags=re.DOTALL)
        # Fix unquoted keys
        fixed = re.sub(r'(\w+):', r'"\1":', fixed)
        parsed = json.loads(fixed)
        if isinstance(parsed, dict) and "response" in parsed:
            return parsed
    except json.JSONDecodeError:
        pass
    
    # Try 5: Extract response field with regex (more flexible)
    try:
        # Match response field with various quote styles
        match = re.search(r'["\']?response["\']?\s*[:=]\s*["\']?([^"\'\}\n,]+)', json_str, re.IGNORECASE)
        if match:
            value = match.group(1).strip().lower()
            # Clean up trailing punctuation
            value = re.sub(r'[.,;:!?\'"`]+$', '', value)
            if value in VALID_LABELS:
                return {"response": value}
    except Exception:
        pass
    
    # Try 6: Look for any valid label in the string
    try:
        text_lower = json_str.lower()
        for label in VALID_LABELS:
            if re.search(rf'\b{label}\b', text_lower):
                return {"response": label}
    except Exception:
        pass
    
    # Try 7: Handle escaped characters
    try:
        fixed = json_str.encode('utf-8').decode('unicode_escape')
        parsed = json.loads(fixed)
        if isinstance(parsed, dict) and "response" in parsed:
            return parsed
    except (json.JSONDecodeError, UnicodeDecodeError):
        pass
    
    # Try 8: Extract from malformed JSON with nested structures
    try:
        # Look for {"response": "..."} pattern even with extra content
        match = re.search(r'\{\s*["\']?response["\']?\s*:\s*["\']?(correct|incorrect|partial|almost)["\']?\s*\}', json_str, re.IGNORECASE)
        if match:
            value = match.group(1).strip().lower()
            if value in VALID_LABELS:
                return {"response": value}
    except Exception:
        pass
    
    # Try 9: Handle JSON with newlines and extra whitespace
    try:
        fixed = re.sub(r'\s+', ' ', json_str)  # Normalize whitespace
        fixed = fixed.strip()
        parsed = json.loads(fixed)
        if isinstance(parsed, dict) and "response" in parsed:
            return parsed
    except json.JSONDecodeError:
        pass
    
    # Try 10: Extract just the response value from any format
    try:
        # Look for response followed by a valid label
        for label in VALID_LABELS:
            pattern = rf'["\']?response["\']?\s*[:=]\s*["\']?{label}["\']?'
            if re.search(pattern, json_str, re.IGNORECASE):
                return {"response": label}
    except Exception:
        pass
    
    # Try 11: Handle JSON with unescaped newlines in values
    try:
        # Remove newlines that might be breaking the JSON
        fixed = re.sub(r'\n+', ' ', json_str)
        fixed = re.sub(r'\r+', ' ', fixed)
        parsed = json.loads(fixed)
        if isinstance(parsed, dict) and "response" in parsed:
            return parsed
    except json.JSONDecodeError:
        pass
    
    # Try 12: Extract from template-style JSON like {{response: "correct"}}
    try:
        match = re.search(r'\{\s*\{?\s*["\']?response["\']?\s*:\s*["\']?(correct|incorrect|partial|almost)["\']?\s*\}?\s*\}', json_str, re.IGNORECASE)
        if match:
            value = match.group(1).strip().lower()
            if value in VALID_LABELS:
                return {"response": value}
    except Exception:
        pass
    
    return None


def _normalize_label(value: str) -> str | None:
    """Normalize a label value to one of the valid labels.
    
    Handles various input formats including:
    - Direct label matches
    - Common misspellings and variations
    - Punctuation and whitespace
    - Partial matches and fuzzy matching
    - Multi-word phrases
    """
    if not value or not isinstance(value, str):
        return None
    
    clean = value.strip().lower()
    
    # Remove common punctuation and whitespace variations
    clean_no_punct = re.sub(r'[^a-z\s]', '', clean).strip()
    clean_single = re.sub(r'[^a-z]', '', clean)
    
    # Direct match (exact)
    if clean_no_punct in VALID_LABELS:
        return LABEL_CANONICAL.get(clean_no_punct, clean_no_punct)
    
    # Check aliases (exact match)
    for label, aliases in LABEL_ALIASES.items():
        for alias in aliases:
            if clean_no_punct == alias.lower():
                return label
    
    # Check for partial matches in aliases
    for label, aliases in LABEL_ALIASES.items():
        for alias in aliases:
            alias_clean = re.sub(r'[^a-z]', '', alias.lower())
            if clean_single == alias_clean or clean_no_punct == alias.lower():
                return label
    
    # Fuzzy matching with priority order
    # Check for "incorrect" first (highest priority for safety)
    if clean_single == 'incorrect' or clean_single.startswith('incorr') or clean_single.startswith('wrong'):
        return 'incorrect'
    if 'notcorrect' in clean_single or 'notright' in clean_single:
        return 'incorrect'
    if 'notvalid' in clean_single:
        return 'incorrect'
    if 'fundamentallywrong' in clean_single or 'completelywrong' in clean_single:
        return 'incorrect'
    
    # Check for "almost" (before "partial" since "almost" is more specific)
    # "Almost" indicates a solution that is nearly complete with only tiny gaps
    if clean_single == 'almost' or clean_single.startswith('almost'):
        return 'almost'
    if clean_single == 'nearly' or clean_single.startswith('nearly'):
        return 'almost'
    if 'nearlycomplete' in clean_single or 'almostcomplete' in clean_single:
        return 'almost'
    if 'nearlycorrect' in clean_single or 'almostcorrect' in clean_single:
        return 'almost'
    if 'mostlycorrect' in clean_single:
        return 'almost'
    if 'minorgap' in clean_single or 'tinygap' in clean_single or 'smallgap' in clean_single:
        return 'almost'
    if 'nearlythere' in clean_single or 'almostthere' in clean_single:
        return 'almost'
    if 'essentiallycorrect' in clean_single:
        return 'almost'
    if 'minorfix' in clean_single or 'smallomission' in clean_single:
        return 'almost'
    if 'nearlysolved' in clean_single:
        return 'almost'
    
    # Check for "partial"
    if clean_single == 'partial' or clean_single.startswith('partial'):
        return 'partial'
    if clean_single == 'incomplete' or clean_single.startswith('incomplete'):
        return 'partial'
    if 'some' in clean_single and 'somehow' not in clean_single:
        return 'partial'
    if clean_single == 'half' or clean_single.startswith('half'):
        return 'partial'
    if 'partly' in clean_single:
        return 'partial'
    if 'partiallycorrect' in clean_single:
        return 'partial'
    if 'incompletebutvalid' in clean_single:
        return 'partial'
    if 'somecredit' in clean_single or 'partialcredit' in clean_single:
        return 'partial'
    if 'validapproach' in clean_single:
        return 'partial'
    
    # Check for "correct" (but not "incorrect")
    if clean_single == 'correct' or (clean_single.startswith('correct') and 'incorrect' not in clean_single):
        return 'correct'
    if clean_single == 'right' and 'wrong' not in clean_single:
        return 'correct'
    if clean_single == 'valid' and 'invalid' not in clean_single:
        return 'correct'
    if 'fullycorrect' in clean_single or 'entirelycorrect' in clean_single:
        return 'correct'
    if 'perfectsolution' in clean_single or 'completesolution' in clean_single:
        return 'correct'
    if 'fullmarks' in clean_single or 'fullcredit' in clean_single:
        return 'correct'
    if 'allsolved' in clean_single or 'fullysolved' in clean_single:
        return 'correct'
    
    # Check for negations that might indicate incorrect
    if 'no' in clean_single and ('credit' in clean_single or 'marks' in clean_single):
        return 'incorrect'
    if clean_single == 'zero' or clean_single == 'zeromarks' or clean_single == 'nocredit':
        return 'incorrect'
    if 'zerocredit' in clean_single:
        return 'incorrect'
    
    return None


def _confidence_based_extraction(text: str) -> str | None:
    """Extract label using confidence-based voting from multiple signals.
    
    This is a fallback method that combines multiple weak signals to make
    a best-effort prediction when other methods fail.
    
    Enhanced with:
    - Better pattern matching detection
    - Stronger signals for "almost" vs "partial" distinction
    - Context-aware scoring
    - Special handling for "almost" detection
    """
    if not text or not isinstance(text, str):
        return None
    
    text_lower = text.lower()
    scores = {label: 0.0 for label in VALID_LABELS}
    
    # Signal 1: Presence of label words (weighted by position - later is stronger)
    for label in VALID_LABELS:
        matches = list(re.finditer(rf'\b{label}\b', text_lower))
        for i, match in enumerate(matches):
            # Later occurrences get higher weight
            position_weight = (i + 1) / len(matches) if matches else 0
            scores[label] += 0.5 + 0.5 * position_weight
    
    # Signal 2: Extended aliases with context
    for label, aliases in LABEL_ALIASES_EXTENDED.items():
        for alias in aliases:
            if alias.lower() in text_lower:
                scores[label] += 0.3
    
    # Signal 3: Positive/negative sentiment indicators
    positive_indicators = ['valid', 'correct', 'right', 'true', 'complete', 'perfect', 'solved', 'rigorous', 'proof']
    negative_indicators = ['wrong', 'error', 'mistake', 'flawed', 'invalid', 'false', 'incorrect', 'fundamentally', 'fatal']
    
    for indicator in positive_indicators:
        if indicator in text_lower:
            scores['correct'] += 0.2
            scores['almost'] += 0.15  # Increased boost for almost
    
    for indicator in negative_indicators:
        if indicator in text_lower:
            scores['incorrect'] += 0.3
    
    # Signal 4: Partial indicators (but be careful not to over-trigger)
    partial_indicators = ['incomplete', 'missing', 'partial', 'some', 'part', 'half', 'unfinished']
    for indicator in partial_indicators:
        if indicator in text_lower:
            # Only boost partial if not also seeing "almost" indicators
            scores['partial'] += 0.2
    
    # Signal 5: Strong "almost" indicators (these are very specific)
    strong_almost_indicators = [
        'nearly complete', 'almost complete', 'essentially correct', 'essentially complete',
        'minor gap', 'tiny gap', 'small gap', 'minor error', 'tiny error', 'small error',
        'minor omission', 'tiny omission', 'small omission', 'minor fix', 'tiny fix',
        'nearly there', 'almost there', 'almost correct', 'nearly correct',
        'mostly correct', 'just missing', 'only missing', 'forgot to state',
        'did not explicitly', 'missing explicit', 'tiny detail', 'small detail',
        'minor detail', 'minor calculation', 'small calculation error',
    ]
    for indicator in strong_almost_indicators:
        if indicator in text_lower:
            scores['almost'] += 0.5  # Strong boost for almost-specific language
            scores['partial'] = max(0, scores['partial'] - 0.1)  # Slight penalty to partial
    
    # Signal 6: "Almost" vs "Partial" distinction signals
    # If we see "gap" or "missing" but also "tiny/minor/small", it's "almost"
    if ('gap' in text_lower or 'missing' in text_lower) and \
       ('tiny' in text_lower or 'minor' in text_lower or 'small' in text_lower or 'just' in text_lower):
        scores['almost'] += 0.4
        scores['partial'] = max(0, scores['partial'] - 0.15)
    
    # If we see "significant" or "major" gaps, it's "partial"
    if 'significant' in text_lower or 'major' in text_lower or 'substantial' in text_lower:
        scores['partial'] += 0.3
        scores['almost'] = max(0, scores['almost'] - 0.1)
    
    # Signal 7: Pattern matching indicators (strong signal for INCORRECT)
    pattern_count = 0
    for indicator in PATTERN_MATCHING_INDICATORS:
        if indicator.lower() in text_lower:
            pattern_count += 1
    # If we see multiple pattern matching indicators, strongly boost "incorrect"
    if pattern_count >= 2:
        scores['incorrect'] += 0.5 * pattern_count
    # Strong pattern matching signal
    if 'pattern holds' in text_lower or 'pattern continues' in text_lower:
        scores['incorrect'] += 0.6
    if 'only checked' in text_lower or 'only verified' in text_lower:
        scores['incorrect'] += 0.6
    if 'no general proof' in text_lower or 'without proof' in text_lower:
        scores['incorrect'] += 0.7
    
    # Signal 8: Valid proof technique indicators (signal for PARTIAL or better)
    technique_count = 0
    for indicator in VALID_TECHNIQUE_INDICATORS:
        if indicator.lower() in text_lower:
            technique_count += 1
    # If we see proof techniques, reduce "incorrect" score and boost "partial" or "almost"
    if technique_count >= 2:
        scores['partial'] += 0.2 * technique_count
        scores['almost'] += 0.15 * technique_count
        scores['incorrect'] = max(0, scores['incorrect'] - 0.2 * technique_count)
    
    # Signal 9: Completeness indicators
    if '100%' in text_lower or 'fully solved' in text_lower or 'complete proof' in text_lower:
        scores['correct'] += 0.3
    if '85%' in text_lower or '90%' in text_lower or '95%' in text_lower:
        scores['almost'] += 0.35
    if '30%' in text_lower or '50%' in text_lower or '70%' in text_lower:
        scores['partial'] += 0.3
    
    # Signal 10: Verdict/assessment section indicators (strongest signal)
    if 'verdict: incorrect' in text_lower or 'assessment: incorrect' in text_lower:
        scores['incorrect'] += 0.6
    if 'verdict: correct' in text_lower or 'assessment: correct' in text_lower:
        scores['correct'] += 0.6
    if 'verdict: partial' in text_lower or 'assessment: partial' in text_lower:
        scores['partial'] += 0.6
    if 'verdict: almost' in text_lower or 'assessment: almost' in text_lower:
        scores['almost'] += 0.6
    
    # Signal 11: Analysis section indicators for "almost"
    almost_analysis_patterns = [
        'main proof is complete', 'essentially complete', 'core argument is correct',
        'proof structure is complete', 'just needs', 'only needs', 'tiny fix',
        'small fix needed', 'minor adjustment', 'essentially correct',
        'proof is solid', 'argument is complete', 'nearly perfect',
    ]
    for pattern in almost_analysis_patterns:
        if pattern in text_lower:
            scores['almost'] += 0.4
    
    # Apply confidence thresholds
    for label, threshold in CONFIDENCE_THRESHOLDS.items():
        if scores[label] < threshold:
            scores[label] = 0
    
    # Return the highest scoring label if it meets threshold
    if scores:
        best_label = max(scores, key=scores.get)
        if scores[best_label] > 0:
            return best_label
    
    return None


def _detect_pattern_matching_in_analysis(text: str) -> bool:
    """Detect if the LLM's analysis mentions pattern matching as a concern.
    
    This helps identify when the LLM is analyzing a solution that only checks examples.
    Enhanced with more comprehensive pattern matching detection.
    """
    if not text or not isinstance(text, str):
        return False
    
    text_lower = text.lower()
    
    # Strong indicators that the solution is pattern matching only
    strong_indicators = [
        "only checked examples", "only verified examples", "only tested",
        "pattern matching without proof", "no general proof",
        "no general argument", "no valid proof technique",
        "just checking values", "just verifying cases",
        "computational only", "no demonstration of understanding",
        "only checks examples", "only verifies examples", "only tests",
        "pattern matching only", "no proof provided", "without proof",
        "no valid approach", "no mathematical insight",
        "just pattern matching", "purely computational",
        "no generalization", "no inductive step", "no proof structure",
    ]
    
    for indicator in strong_indicators:
        if indicator in text_lower:
            return True
    
    # Count pattern matching indicators
    pattern_count = sum(1 for ind in PATTERN_MATCHING_INDICATORS if ind.lower() in text_lower)
    technique_count = sum(1 for ind in VALID_TECHNIQUE_INDICATORS if ind.lower() in text_lower)
    
    # If we see many pattern indicators but few proof techniques, likely pattern matching
    if pattern_count >= 3 and technique_count < 2:
        return True
    
    # Additional pattern matching detection
    if 'pattern holds' in text_lower and 'proof' not in text_lower:
        return True
    if 'pattern continues' in text_lower and 'proof' not in text_lower:
        return True
    if 'verified for' in text_lower and 'general' not in text_lower:
        return True
    if 'checked' in text_lower and 'n=1' in text_lower and 'n=2' in text_lower:
        return True
    
    return False


def _extract_label_from_text(text: str) -> str | None:
    """Extract the grading label from raw text using pattern matching.
    
    Uses multiple priority strategies to find the most likely label:
    1. JSON blocks at the end (most reliable)
    2. Explicit declarations with colons/equals
    3. Formatted labels (backticks, bold)
    4. Labels at end of text
    5. Count-based with priority tie-breaking
    6. Last line matching
    7. Context-aware extraction using pattern matching detection
    8. Multi-line JSON block detection
    9. Verdict section analysis
    """
    if not text or not isinstance(text, str):
        return None
        
    text_lower = text.lower().strip()
    lines = [line.strip() for line in text_lower.split('\n') if line.strip()]
    
    # Priority 0: Look for JSON block at the very end (most reliable)
    for i in range(len(lines) - 1, -1, -1):
        line = lines[i].strip()
        if not line:
            continue
        # Check for JSON block format
        if '<json>' in line and '</json>' in line:
            start = line.find('<json>')
            end = line.find('</json>') + 7
            if start != -1 and end > start:
                inner = line[start + 6:end - 7].strip()
                parsed = _try_parse_json_with_fallbacks(inner)
                if parsed and "response" in parsed:
                    value = str(parsed["response"]).strip().lower()
                    if value in VALID_LABELS:
                        return LABEL_CANONICAL.get(value, value)
    
    # Priority 0.5: Look for JSON block across multiple lines (common LLM output format)
    full_text = '\n'.join(lines)
    json_block_pattern = r'<json>\s*(\{[^}]*\})\s*</json>'
    json_matches = list(re.finditer(json_block_pattern, full_text, re.DOTALL))
    if json_matches:
        # Use the last JSON block
        last_json = json_matches[-1].group(1)
        parsed = _try_parse_json_with_fallbacks(last_json)
        if parsed and "response" in parsed:
            value = str(parsed["response"]).strip().lower()
            if value in VALID_LABELS:
                return LABEL_CANONICAL.get(value, value)
    
    # Priority 0.6: Look for multi-line JSON block with nested content
    json_multiline_pattern = r'<json>\s*(\{[\s\S]*?\})\s*</json>'
    json_multiline_matches = list(re.finditer(json_multiline_pattern, full_text, re.DOTALL | re.IGNORECASE))
    if json_multiline_matches:
        # Use the last match
        for match in reversed(json_multiline_matches):
            inner = match.group(1).strip()
            parsed = _try_parse_json_with_fallbacks(inner)
            if parsed and "response" in parsed:
                value = str(parsed["response"]).strip().lower()
                if value in VALID_LABELS:
                    return LABEL_CANONICAL.get(value, value)
    
    # Priority 1: Look for explicit label declarations with colons
    label_alternatives = "correct|incorrect|partial|almost"
    declaration_patterns = [
        rf'(?:verdict|grade|label|classification|assessment|evaluation)\s*[:=]\s*["\']?({label_alternatives})["\']?\b',
        rf'(?:the\s+answer\s+is)\s*["\']?({label_alternatives})["\']?\b',
        rf'(?:this\s+is)\s*["\']?({label_alternatives})["\']?\b',
        rf'["\']?response["\']?\s*[:=]\s*["\']?({label_alternatives})["\']?\b',
        rf'(?:final\s+)?(?:answer|verdict|grade)\s*[:=]\s*["\']?({label_alternatives})["\']?\b',
        rf'(?:classification|grade)\s*[:=]\s*["\']?({label_alternatives})["\']?\b',
        rf'(?:i\s+(?:would|will)\s+)?(?:classify|grade|label)\s+(?:this\s+)?(?:as\s+)?["\']?({label_alternatives})["\']?\b',
    ]
    
    for pattern in declaration_patterns:
        matches = list(re.finditer(pattern, text_lower))
        if matches:
            # Use the LAST match (most likely to be the final verdict)
            last_match = matches[-1]
            for group in last_match.groups():
                if group in VALID_LABELS:
                    return LABEL_CANONICAL.get(group, group)
    
    # Priority 2: Look for labels in code blocks or backticks
    for label in VALID_LABELS:
        if re.search(rf'`{label}`', text_lower):
            return LABEL_CANONICAL.get(label, label)
        if re.search(rf'\*\*{label}\*\*|\*{label}\*', text_lower):
            return LABEL_CANONICAL.get(label, label)
        if re.search(rf'"{label}"|\'{label}\'', text_lower):
            return LABEL_CANONICAL.get(label, label)
    
    # Priority 3: Look for labels at the end of the text
    for label in VALID_LABELS:
        if re.search(rf'\b{label}\b[.!?]*\s*$', text_lower):
            return LABEL_CANONICAL.get(label, label)
    
    # Priority 4: Count occurrences and use priority to break ties
    label_counts = {}
    for label in VALID_LABELS:
        count = len(re.findall(rf'\b{label}\b', text_lower))
        label_counts[label] = count
    
    if any(label_counts.values()):
        valid_counts = {k: v for k, v in label_counts.items() if v > 0}
        if valid_counts:
            max_count = max(valid_counts.values())
            candidates = [k for k, v in valid_counts.items() if v == max_count]
            if len(candidates) == 1:
                return LABEL_CANONICAL.get(candidates[0], candidates[0])
            else:
                # Use priority to break ties
                best = max(candidates, key=lambda x: LABEL_PRIORITY.get(x, 0))
                return LABEL_CANONICAL.get(best, best)
    
    # Priority 5: Look at the last line for any valid label
    if lines:
        last_line = lines[-1]
        for label in VALID_LABELS:
            if label in last_line:
                return LABEL_CANONICAL.get(label, label)
    
    # Priority 6: Check last 3 lines for standalone labels with more flexible matching
    if lines:
        for line in reversed(lines[-3:]):
            line_clean = line.strip('"\'`*[]{}()')
            for label in VALID_LABELS:
                if line_clean == label:
                    return LABEL_CANONICAL.get(label, label)
    
    # Priority 7: Look for label mentions in analysis/verdict sections
    verdict_section_patterns = [
        r'(?:verdict|conclusion|assessment|grade)\s*[:\-]?\s*\n?\s*(correct|incorrect|partial|almost)',
        r'(?:therefore|thus|hence|so)\s*[,:]?\s*(correct|incorrect|partial|almost)',
        r'(?:final\s+)?(?:verdict|assessment|grade)\s*[:=]\s*(correct|incorrect|partial|almost)',
        r'(?:the\s+)?(?:solution|answer|work)\s+(?:is|should\s+be)\s+(correct|incorrect|partial|almost)',
    ]
    for pattern in verdict_section_patterns:
        matches = list(re.finditer(pattern, text_lower, re.IGNORECASE))
        if matches:
            last_match = matches[-1]
            for group in last_match.groups():
                if group in VALID_LABELS:
                    return LABEL_CANONICAL.get(group, group)
    
    # Priority 8: Context-aware extraction - check if analysis mentions pattern matching
    if _detect_pattern_matching_in_analysis(text_lower):
        # If pattern matching is detected in analysis, prefer "incorrect"
        # unless there's strong evidence of another label
        if label_counts.get('incorrect', 0) > 0 or label_counts.get('partial', 0) == 0:
            return 'incorrect'
    
    # Priority 9: Look for response field in various formats
    response_patterns = [
        r'"response"\s*:\s*"(correct|incorrect|partial|almost)"',
        r"'response'\s*:\s*'(correct|incorrect|partial|almost)'",
        r'response\s*:\s*(correct|incorrect|partial|almost)',
    ]
    for pattern in response_patterns:
        matches = list(re.finditer(pattern, text_lower, re.IGNORECASE))
        if matches:
            last_match = matches[-1]
            for group in last_match.groups():
                if group in VALID_LABELS:
                    return LABEL_CANONICAL.get(group, group)
    
    # Priority 10: Enhanced "almost" detection from analysis text
    # Look for strong "almost" indicators in the analysis
    almost_indicators = [
        'nearly complete', 'almost complete', 'essentially correct', 'essentially complete',
        'minor gap', 'tiny gap', 'small gap', 'minor error', 'tiny error',
        'minor omission', 'tiny omission', 'small omission',
        'nearly there', 'almost there', 'almost correct', 'nearly correct',
        'mostly correct', 'just missing', 'only missing', 'forgot to state',
        'main proof is complete', 'core argument is correct',
        'proof structure is complete', 'just needs', 'only needs', 'tiny fix',
    ]
    for indicator in almost_indicators:
        if indicator in text_lower:
            # Check if "almost" is mentioned at all
            if 'almost' in text_lower or label_counts.get('almost', 0) > 0:
                return 'almost'
    
    # Priority 11: Use confidence-based extraction as final fallback
    confidence_result = _confidence_based_extraction(text)
    if confidence_result:
        return LABEL_CANONICAL.get(confidence_result, confidence_result)
    
    return None


class TaskAgent:
    """Task agent that solves IMO grading problems."""

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
        # Extract fields from inputs
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        # Enhanced instruction with chain-of-thought reasoning and detailed examples
        instruction = f"""You are an expert IMO mathematics grader. Your task is to evaluate the student's solution and classify it into EXACTLY ONE category.

## CATEGORIES (choose exactly one)

1. **"correct"**: Complete, rigorous, correct solution with all steps valid. Full marks (100%).
   - Must have: Valid proof technique, all claims justified, no logical gaps
   - Can have: Minor notational issues that don't affect correctness
   - The solution would receive full credit in an IMO exam

2. **"incorrect"**: Fundamental flaw, only pattern matching, or no valid mathematical approach. Zero credit (0%).
   - Pattern matching: ONLY checks examples (n=1,2,3...) without GENERAL PROOF
   - No valid insight: No demonstration of understanding the problem structure
   - Fatal error: Logical flaw that invalidates the entire approach
   - No valid proof technique used

3. **"partial"**: Valid approach with genuine mathematical understanding, but incomplete or has significant errors. Partial credit (30-70%).
   - Has: Valid insight, correct approach started, genuine understanding shown
   - Missing: Key steps, complete proof, or has significant gaps/errors
   - Examples: Good start but incomplete, correct lemma but wrong conclusion, valid method but calculation errors
   - The student understands the problem but hasn't fully solved it

4. **"almost"**: Nearly complete solution (85-95%), only minor gap or tiny error. High partial credit (70-90%).
   - Has: Essentially complete proof, correct main argument, all major steps present
   - Missing: Tiny detail, edge case, explicit statement of obvious fact, minor justification
   - Examples: Forgot to state base case in induction (but it's obvious), missing "QED", tiny algebraic gap, minor calculation error in final step
   - The solution is ALMOST correct - just needs a tiny fix to be perfect
   - DISTINCTION from "partial": In "almost", the main proof structure is complete and correct; in "partial", significant work is still missing

## CRITICAL DECISION RULES

**Rule 1: Pattern Matching Detection (MOST IMPORTANT)**
- If student ONLY checks specific values (n=1,2,3,4,5) and claims "pattern holds" → "incorrect"
- If student says "verified for small values" without general argument → "incorrect"
- If student uses examples to DISCOVER pattern but then PROVES generally → "partial" or better

**Rule 2: Valid Insight Protection**
- If student shows ANY correct mathematical thinking → NOT "incorrect"
- Even with errors, if they demonstrate understanding → "partial" or "almost"
- Examples of valid insight: correct factorization, valid lemma, correct approach started

**Rule 3: Completeness Assessment**
- 100% complete, all steps valid → "correct"
- 85-95% complete, tiny gap only → "almost"
- 30-70% complete, significant work remaining → "partial"
- 0-20% valid, no real approach → "incorrect"

**Rule 4: ALMOST vs PARTIAL Distinction (CRITICAL)**
- "almost": Main proof is complete, just missing tiny details. The core argument is solid.
  - Example: Correct induction with all steps, just forgot to explicitly state the base case
  - Example: Correct proof of inequality, just didn't mention when equality holds
  - Example: Correct proof structure, one minor calculation error at the end
- "partial": Significant portions missing or major errors in the approach.
  - Example: Good factorization but didn't complete the proof
  - Example: Correct approach started but key step is wrong or missing
  - Example: Valid insight but the proof is incomplete

## DETAILED EXAMPLES

### Example 1: INCORRECT (Pattern Matching)
Problem: Prove that for all positive integers n, the sum of the first n odd numbers equals n².
Student Solution: 
For n=1: 1 = 1² ✓
For n=2: 1+3 = 4 = 2² ✓  
For n=3: 1+3+5 = 9 = 3² ✓
For n=4: 1+3+5+7 = 16 = 4² ✓
The pattern clearly holds for all n.
Analysis: Student only verified examples, no general proof. No induction, no algebraic argument.
<json>{{"response": "incorrect"}}</json>

### Example 2: PARTIAL (Valid Insight, Incomplete)
Problem: Prove that for all positive integers n, n³-n is divisible by 6.
Student Solution:
n³-n = n(n-1)(n+1). This is the product of three consecutive integers.
One of them must be even, so the product is divisible by 2.
Analysis: Excellent factorization and insight about divisibility by 2. But student didn't prove divisibility by 3 (among 3 consecutive integers, one is divisible by 3). Valid approach started but incomplete - significant work remaining.
<json>{{"response": "partial"}}</json>

### Example 3: ALMOST (Nearly Complete, Tiny Gap)
Problem: Prove AM-GM inequality: For positive reals a,b, (a+b)/2 ≥ √ab.
Student Solution:
(a+b)/2 ≥ √ab is equivalent to a+b ≥ 2√ab, which is equivalent to (√a-√b)² ≥ 0.
Since squares are always non-negative, this is always true.
Analysis: Perfect proof of the inequality. Only missing: explicit statement that equality holds when a=b (which is obvious from the square being zero). The main proof is complete and correct - just a tiny omission.
<json>{{"response": "almost"}}</json>

### Example 4: CORRECT (Complete Proof)
Problem: Prove there are infinitely many primes.
Student Solution:
Assume there are finitely many primes p₁, p₂, ..., pₙ. Consider N = p₁p₂...pₙ + 1.
N is not divisible by any pᵢ (remainder 1 when divided by each).
So N is either prime or has a prime factor not in our list.
Contradiction! Therefore infinitely many primes.
Analysis: Complete, rigorous proof by contradiction. All steps valid.
<json>{{"response": "correct"}}</json>

### Example 5: PARTIAL (Good Start, Wrong Direction)
Problem: Prove every even number > 2 is sum of two primes (Goldbach - simplified).
Student Solution:
Let's check: 4=2+2, 6=3+3, 8=3+5, 10=3+7=5+5, 12=5+7...
I notice all even numbers seem to work. I'll try to prove by induction.
Base case: 4=2+2 ✓
Inductive step: Assume 2k = p+q. Then 2(k+1) = 2k+2 = p+q+2. If q+2 is prime, we're done.
Analysis: Student correctly identified pattern and attempted induction. Valid insight! But the inductive step doesn't work (q+2 may not be prime). Good approach attempted but flawed execution - significant gap in the proof.
<json>{{"response": "partial"}}</json>

### Example 6: INCORRECT (No Valid Approach)
Problem: Prove √2 is irrational.
Student Solution:
√2 ≈ 1.41421356... The decimal goes on forever without repeating, so it must be irrational.
Analysis: No valid proof technique. Just an observation about decimals, not a proof.
<json>{{"response": "incorrect"}}</json>

### Example 7: ALMOST (Complete Proof, Tiny Error)
Problem: Prove by induction that 1 + 2 + ... + n = n(n+1)/2.
Student Solution:
Base case: n=1, LHS=1, RHS=1(2)/2=1 ✓
Inductive hypothesis: Assume true for n=k, so 1+2+...+k = k(k+1)/2.
Inductive step: For n=k+1:
1+2+...+k+(k+1) = k(k+1)/2 + (k+1) = (k(k+1) + 2(k+1))/2 = (k+1)(k+2)/2
This is the formula with n=k+1. ✓
Analysis: Perfect induction proof! All steps correct. Only tiny issue: didn't explicitly write "Therefore by induction, the formula holds for all n." at the end. The proof is essentially complete.
<json>{{"response": "almost"}}</json>

### Example 8: PARTIAL vs ALMOST Contrast
Problem: Prove n² + n is always even for integer n.

PARTIAL case:
Student: n² + n = n(n+1). One of n or n+1 is even.
Analysis: Correct factorization and insight! But student didn't finish the proof - didn't explicitly conclude that product of even and odd is even. Significant gap remains.
<json>{{"response": "partial"}}</json>

ALMOST case:
Student: n² + n = n(n+1). Since n and n+1 are consecutive, one is even. Therefore n(n+1) is even.
Analysis: Complete proof! Just missing explicit statement "so n²+n is divisible by 2" or "QED" at the very end. The argument is fully formed.
<json>{{"response": "almost"}}</json>

### Example 9: INCORRECT (Misleading Pattern)
Problem: Find all positive integers n such that n² + n + 41 is prime.
Student Solution:
For n=1: 1+1+41=43 (prime) ✓
For n=2: 4+2+41=47 (prime) ✓
For n=3: 9+3+41=53 (prime) ✓
...
For n=10: 100+10+41=151 (prime) ✓
The pattern shows this is always prime for all positive integers n.
Analysis: Student checked many examples but didn't prove for all n. In fact, n=40 gives 40²+40+41=1681=41² which is not prime. Pattern matching without proof is invalid.
<json>{{"response": "incorrect"}}</json>

### Example 10: ALMOST (Complete with Notation Gap)
Problem: Prove that for any real numbers a, b, c, we have (a+b+c)² ≤ 3(a²+b²+c²).
Student Solution:
Expanding: (a+b+c)² = a²+b²+c² + 2ab + 2bc + 2ca
We need to show: a²+b²+c² + 2ab + 2bc + 2ca ≤ 3a²+3b²+3c²
This is equivalent to: 2ab + 2bc + 2ca ≤ 2a²+2b²+2c²
Which is: ab + bc + ca ≤ a²+b²+c²
This follows from: 2a²+2b²+2c² - 2ab - 2bc - 2ca = (a-b)²+(b-c)²+(c-a)² ≥ 0
Analysis: Perfect proof with all steps correct. Only missing: explicit statement that equality holds when a=b=c. The main argument is complete and rigorous.
<json>{{"response": "almost"}}</json>

### Example 11: PARTIAL (Wrong Direction but Valid Start)
Problem: Prove that if a² + b² + c² = ab + bc + ca for real numbers a, b, c, then a = b = c.
Student Solution:
From a² + b² + c² = ab + bc + ca, we get:
a² - ab + b² - bc + c² - ca = 0
a(a-b) + b(b-c) + c(c-a) = 0
This means either a=b or b=c or c=a.
Analysis: Student made algebraic manipulation correctly but drew wrong conclusion. The factorization doesn't imply "either a=b or b=c or c=a" - this is a logical error. However, the approach was on the right track (rearranging to get sum of squares = 0). Valid insight but wrong execution.
<json>{{"response": "partial"}}</json>

### Example 12: CORRECT (Elegant Proof)
Problem: Prove that for any positive integer n, the number n(n+1)(n+2) is divisible by 6.
Student Solution:
Among any three consecutive integers, one is divisible by 3. Also, at least one is even (divisible by 2). Since 2 and 3 are coprime, the product is divisible by 6.
Analysis: Concise, complete, and correct proof. All logical steps valid.
<json>{{"response": "correct"}}</json>

## CURRENT PROBLEM

PROBLEM:
```
{problem}
```

OFFICIAL SOLUTION:
```
{solution}
```

GRADING GUIDELINES:
```
{grading_guidelines}
```

STUDENT'S ANSWER:
```
{student_answer}
```

## YOUR TASK - STEP BY STEP

**Step 1: Analyze the Approach**
- Did the student use a valid proof technique? (induction, contradiction, direct proof, etc.)
- Did they only check examples without general argument? (pattern matching)
- Did they show genuine mathematical understanding of the problem?

**Step 2: Assess Completeness**
- What percentage of the solution is complete? (100%, 90%, 50%, 10%?)
- Are there major gaps or just tiny omissions?
- Are the completed parts correct?

**Step 3: Distinguish ALMOST vs PARTIAL**
- Is the main proof structure complete and correct, with only tiny details missing? → "almost"
- Is there significant work still missing or major errors? → "partial"
- Would a small fix make it perfect? → "almost"
- Would substantial additional work be needed? → "partial"

**Step 4: Make Classification**
- If pattern matching only → "incorrect"
- If valid insight but significant gaps/errors → "partial"  
- If nearly complete, tiny gap only → "almost"
- If complete and correct → "correct"

**Step 5: Output Result**
Output EXACTLY ONE JSON block as the FINAL line of your response:
<json>{{"response": "correct"}}</json>
OR
<json>{{"response": "incorrect"}}</json>
OR
<json>{{"response": "partial"}}</json>
OR
<json>{{"response": "almost"}}</json>

CRITICAL: The JSON block MUST be the VERY LAST line with no text after it.
Remember: Choose "almost" when the proof is essentially complete with only tiny gaps, and "partial" when significant work remains.

**IMPORTANT REMINDERS:**
1. Pattern matching (only checking examples) is ALWAYS "incorrect" - no exceptions
2. A valid proof technique with errors is "partial" or "almost", never "incorrect"
3. "Almost" means the main proof structure is complete - just needs tiny fixes
4. "Partial" means significant work is still missing or major errors exist
5. When in doubt between "almost" and "partial", choose "partial" unless the proof is clearly nearly complete

        # Try up to 3 times to get a valid prediction
        max_retries = 3
        prediction = None
        msg_history = []
        all_responses = []  # Track all responses for consensus
        all_predictions = []  # Track all valid predictions for voting
        last_error = None
        
        for attempt in range(max_retries):
            try:
                # On retry, add a stronger reminder about output format
                msg_to_send = instruction
                if attempt > 0:
                    msg_to_send += f'''\n\nCRITICAL: Your previous response did not contain a valid JSON block. You MUST output exactly one of these as the FINAL line:
<json>{{"response": "correct"}}</json>
<json>{{"response": "incorrect"}}</json>
<json>{{"response": "partial"}}</json>
<json>{{"response": "almost"}}</json>

Remember: The JSON block MUST be the very last line of your response with no text after it.'''
                
                response, msg_history, info = get_response_from_llm(
                    msg=msg_to_send,
                    model=self.model,
                    msg_history=[],
                )
            except Exception as e:
                self.log_fn(f"Attempt {attempt + 1}: LLM call failed: {e}")
                last_error = str(e)
                continue

            text_to_parse = (response or "").strip()
            if not text_to_parse:
                self.log_fn(f"Attempt {attempt + 1}: Empty response from LLM")
                continue
                
            all_responses.append(text_to_parse)
            
            # Debug logging - log end of response where JSON should be
            preview_len = min(500, len(text_to_parse))
            self.log_fn(f"Attempt {attempt + 1}: Raw LLM response (last {preview_len} chars): ...{text_to_parse[-preview_len:]}")
            
            attempt_prediction = None
            
            try:
                # Use the new robust extraction function
                attempt_prediction = _extract_final_label(text_to_parse)
                
                if attempt_prediction:
                    self.log_fn(f"Attempt {attempt + 1}: Extracted prediction: '{attempt_prediction}'")
                else:
                    self.log_fn(f"Attempt {attempt + 1}: No valid prediction extracted from response")
                    # Try to extract any label as fallback
                    attempt_prediction = _extract_label_from_text(text_to_parse)
                    if attempt_prediction:
                        self.log_fn(f"Attempt {attempt + 1}: Fallback text extraction found: '{attempt_prediction}'")
                    
                # If we got a valid prediction, add to predictions list
                if attempt_prediction in VALID_LABELS:
                    all_predictions.append(attempt_prediction)
                    # On first successful attempt, use it immediately
                    if prediction is None:
                        prediction = attempt_prediction
                        # Don't break - continue to collect more predictions for voting
                        if attempt == 0:
                            break
                    
            except Exception as e:
                self.log_fn(f"Attempt {attempt + 1}: Error extracting prediction: {e}")
                last_error = str(e)
                attempt_prediction = None
        
        # If we have multiple predictions, use voting with priority tie-breaking
        if len(all_predictions) > 1:
            self.log_fn(f"Multiple predictions collected: {all_predictions}")
            prediction = self._vote_predictions(all_predictions, all_responses)
            self.log_fn(f"Voting result: {prediction}")
        
        # If all retries failed or no valid prediction, try consensus from all responses
        if prediction not in VALID_LABELS:
            if len(all_responses) > 0:
                self.log_fn("Attempting consensus extraction from all responses...")
                prediction = self._consensus_extraction(all_responses)
                if prediction:
                    self.log_fn(f"Consensus extraction result: {prediction}")
        
        # If still no valid prediction, try one more time with confidence-based extraction
        if prediction not in VALID_LABELS and all_responses:
            self.log_fn("Attempting confidence-based extraction...")
            for response in reversed(all_responses):  # Try most recent first
                prediction = _confidence_based_extraction(response)
                if prediction in VALID_LABELS:
                    self.log_fn(f"Confidence-based extraction result: {prediction}")
                    break
        
        # Final validation and canonicalization
        if prediction not in VALID_LABELS:
            self.log_fn(f"Invalid prediction '{prediction}', defaulting to 'incorrect' (last error: {last_error})")
            prediction = "incorrect"
        else:
            # Map "almost" to "partial" for final output
            if prediction == "almost":
                prediction = "partial"
            self.log_fn(f"Final prediction: {prediction}")

        return str(prediction), msg_history
    
    def _vote_predictions(self, predictions: list[str], responses: list[str] | None = None) -> str:
        """Vote among multiple predictions to select the best one.
        
        Uses a weighted voting scheme that considers:
        1. Frequency of each prediction
        2. Label priority for tie-breaking
        3. Special handling for pattern matching detection
        4. Context-aware tie-breaking for "almost" vs "partial"
        
        Args:
            predictions: List of prediction labels
            responses: Optional list of raw responses for context analysis
        """
        if not predictions:
            return "incorrect"
        
        if len(predictions) == 1:
            return predictions[0]
        
        # Count votes
        vote_counts = {}
        for pred in predictions:
            vote_counts[pred] = vote_counts.get(pred, 0) + 1
        
        self.log_fn(f"Vote counts: {vote_counts}")
        
        # Find the maximum vote count
        max_votes = max(vote_counts.values())
        
        # Get all candidates with max votes
        candidates = [k for k, v in vote_counts.items() if v == max_votes]
        
        if len(candidates) == 1:
            return candidates[0]
        
        # Special handling for "almost" vs "partial" tie
        if 'almost' in candidates and 'partial' in candidates:
            # Check if we have responses to analyze
            if responses:
                # Analyze responses for "almost" indicators
                combined_text = ' '.join(responses).lower()
                almost_indicators = [
                    'nearly complete', 'almost complete', 'essentially correct',
                    'minor gap', 'tiny gap', 'small gap', 'minor error',
                    'main proof is complete', 'core argument is correct',
                    'proof structure is complete', 'just needs', 'only needs',
                ]
                almost_score = sum(1 for ind in almost_indicators if ind in combined_text)
                if almost_score >= 2:
                    self.log_fn(f"Almost vs Partial tie: almost_score={almost_score}, choosing 'almost'")
                    return 'almost'
        
        # Tie-breaking using label priority
        # For IMO grading, we want to be conservative
        best = max(candidates, key=lambda x: LABEL_PRIORITY.get(x, 0))
        return best
    
    def _consensus_extraction(self, responses: list[str]) -> str | None:
        """Extract label by consensus from multiple responses.
        
        When multiple LLM calls are made, use voting to determine the most likely label.
        Enhanced with context-aware tie-breaking for "almost" vs "partial".
        """
        if not responses:
            return None
        
        votes = {label: 0 for label in VALID_LABELS}
        
        for response in responses:
            if not response:
                continue
            
            # Use the robust extraction function
            label = _extract_final_label(response)
            if label:
                votes[label] += 1
        
        # Find the label with most votes
        if any(votes.values()):
            max_votes = max(votes.values())
            candidates = [k for k, v in votes.items() if v == max_votes and v > 0]
            
            if len(candidates) == 1:
                return candidates[0]
            
            # Special handling for "almost" vs "partial" tie
            if 'almost' in candidates and 'partial' in candidates:
                # Analyze responses for "almost" indicators
                combined_text = ' '.join(responses).lower()
                almost_indicators = [
                    'nearly complete', 'almost complete', 'essentially correct',
                    'minor gap', 'tiny gap', 'small gap', 'minor error',
                    'main proof is complete', 'core argument is correct',
                    'proof structure is complete', 'just needs', 'only needs',
                ]
                almost_score = sum(1 for ind in almost_indicators if ind in combined_text)
                if almost_score >= 2:
                    self.log_fn(f"Consensus: Almost vs Partial tie, almost_score={almost_score}, choosing 'almost'")
                    return 'almost'
            
            # Sort by votes, then by priority for ties
            best_label = max(votes.keys(), key=lambda x: (votes[x], LABEL_PRIORITY.get(x, 0)))
            if votes[best_label] > 0:
                return best_label
        
        return None
