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
import time
from typing import Callable, Any

from agent.llm_client import get_response_from_llm, EVAL_MODEL

logger = logging.getLogger(__name__)


def retry_with_backoff(
    func: Callable[..., Any],
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 10.0,
    exceptions: tuple = (Exception,),
) -> Any:
    """Retry a function with exponential backoff."""
    last_exception = None
    for attempt in range(max_retries):
        try:
            return func()
        except exceptions as e:
            last_exception = e
            if attempt < max_retries - 1:
                delay = min(base_delay * (2 ** attempt), max_delay)
                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.1f}s...")
                time.sleep(delay)
    raise last_exception


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks."""
    results = []
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
        try:
            results.append(json.loads(inner))
        except json.JSONDecodeError as e:
            logger.debug(f"JSON decode error in <json> block: {e}")
            continue
    return results or None


def _extract_json_flexible(text: str) -> dict | None:
    """Extract JSON using multiple strategies with enhanced robustness.
    
    This function implements a comprehensive multi-strategy approach to extract
    JSON from LLM responses, handling various formatting issues and edge cases.
    Enhanced with additional parsing strategies and better error recovery.
    """
    if not text or not isinstance(text, str):
        return None
    
    text = text.strip()
    if not text:
        return None
    
    # Strategy 1: <json> tags (most reliable)
    json_blocks = _extract_jsons(text)
    if json_blocks:
        return json_blocks[-1]
    
    # Strategy 2: ```json code blocks with enhanced handling
    json_code_pattern = r'```json\s*(.*?)```'
    matches = re.findall(json_code_pattern, text, re.DOTALL | re.IGNORECASE)
    for match in matches:
        match = match.strip()
        if not match:
            continue
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            fixed = _fix_json_string(match)
            try:
                return json.loads(fixed)
            except json.JSONDecodeError:
                # Try aggressive fixing
                aggressive_fixed = _aggressive_json_fix(fixed)
                try:
                    return json.loads(aggressive_fixed)
                except json.JSONDecodeError:
                    # Try ultra-aggressive fixing as last resort
                    ultra_fixed = _ultra_aggressive_json_fix(aggressive_fixed)
                    try:
                        return json.loads(ultra_fixed)
                    except json.JSONDecodeError:
                        continue
    
    # Strategy 3: ``` code blocks (any language)
    code_pattern = r'```(?:\w+)?\s*(.*?)```'
    matches = re.findall(code_pattern, text, re.DOTALL | re.IGNORECASE)
    for match in matches:
        match = match.strip()
        if not match or match.startswith('{'):
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                fixed = _fix_json_string(match)
                try:
                    return json.loads(fixed)
                except json.JSONDecodeError:
                    aggressive_fixed = _aggressive_json_fix(fixed)
                    try:
                        return json.loads(aggressive_fixed)
                    except json.JSONDecodeError:
                        ultra_fixed = _ultra_aggressive_json_fix(aggressive_fixed)
                        try:
                            return json.loads(ultra_fixed)
                        except json.JSONDecodeError:
                            continue
    
    # Strategy 4: Find JSON objects directly (smart brace matching with nesting)
    best_json = None
    best_score = 0
    
    brace_count = 0
    start_idx = -1
    candidates = []
    
    for i, char in enumerate(text):
        if char == '{':
            if brace_count == 0:
                start_idx = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start_idx != -1:
                candidate = text[start_idx:i+1]
                candidates.append(candidate)
                start_idx = -1
    
    # Score candidates by likelihood of being valid response JSON
    for candidate in candidates:
        score = 0
        parsed = None
        
        # Try direct parsing
        try:
            parsed = json.loads(candidate)
            score += 100  # Base score for valid JSON
        except json.JSONDecodeError:
            fixed = _fix_json_string(candidate)
            try:
                parsed = json.loads(fixed)
                score += 50  # Lower score for fixed JSON
                candidate = fixed
            except json.JSONDecodeError:
                aggressive_fixed = _aggressive_json_fix(fixed)
                try:
                    parsed = json.loads(aggressive_fixed)
                    score += 25  # Even lower for aggressively fixed
                    candidate = aggressive_fixed
                except json.JSONDecodeError:
                    ultra_fixed = _ultra_aggressive_json_fix(aggressive_fixed)
                    try:
                        parsed = json.loads(ultra_fixed)
                        score += 10  # Lowest for ultra-aggressive
                        candidate = ultra_fixed
                    except json.JSONDecodeError:
                        continue
        
        if parsed is None:
            continue
            
        # Bonus points for having expected keys
        if isinstance(parsed, dict):
            if "response" in parsed:
                score += 50
            if any(k in parsed for k in ["correct", "almost", "partial", "incorrect"]):
                score += 30
            # Check for valid prediction values
            for v in parsed.values():
                norm_val = _normalize_prediction(v)
                if norm_val in ["correct", "almost", "partial", "incorrect"]:
                    score += 40
                    break
        
        # Prefer larger JSON objects (more complete)
        score += min(len(candidate) / 10, 20)
        
        if score > best_score:
            best_score = score
            best_json = parsed
    
    if best_json is not None:
        return best_json
    
    # Strategy 5: Look for key-value patterns with enhanced detection
    result = _extract_key_value_patterns(text)
    if result:
        return result
    
    # Strategy 6: Try to extract from malformed JSON with line-by-line parsing
    # Handle cases where LLM outputs multiple lines or mixed content
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        if line and (line.startswith('{') or line.startswith('"')):
            try:
                # Try to parse just this line
                parsed = json.loads(line)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                # Try fixing the line
                fixed = _fix_json_string(line)
                try:
                    parsed = json.loads(fixed)
                    if isinstance(parsed, dict):
                        return parsed
                except json.JSONDecodeError:
                    ultra_fixed = _ultra_aggressive_json_fix(fixed)
                    try:
                        parsed = json.loads(ultra_fixed)
                        if isinstance(parsed, dict):
                            return parsed
                    except json.JSONDecodeError:
                        continue
    
    # Strategy 7: Look for JSON-like structures with regex extraction
    # Handle cases with extra text before/after JSON
    json_like_patterns = [
        r'\{[^{}]*"[^"]+"\s*:\s*"[^"]+"[^{}]*\}',
        r'\{[^{}]*\'[^\']+\'\s*:\s*\'[^\']+\'[^{}]*\}',
        r'\{[^{}]*"[^"]+"\s*:\s*[^,}]+[^{}]*\}',
    ]
    for pattern in json_like_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            try:
                parsed = json.loads(match)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                fixed = _fix_json_string(match)
                try:
                    parsed = json.loads(fixed)
                    if isinstance(parsed, dict):
                        return parsed
                except json.JSONDecodeError:
                    continue
    
    # Strategy 8: Extract from HTML-like tags
    html_patterns = [
        r'<json>(.*?)</json>',
        r'<response>(.*?)</response>',
        r'<grade>(.*?)</grade>',
        r'<result>(.*?)</result>',
        r'<output>(.*?)</output>',
        r'<answer>(.*?)</answer>',
    ]
    for pattern in html_patterns:
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
        for match in matches:
            match = match.strip()
            if match:
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    fixed = _fix_json_string(match)
                    try:
                        return json.loads(fixed)
                    except json.JSONDecodeError:
                        # Try as plain text value
                        norm_val = _normalize_prediction(match)
                        if norm_val in ["correct", "almost", "partial", "incorrect"]:
                            return {"response": norm_val}
    
    # Strategy 9: Look for single-line JSON with common patterns
    # Handle cases like: "response": "correct" or response: correct
    simple_patterns = [
        r'["\']?response["\']?\s*[:=]\s*["\']?(correct|almost|partial|incorrect)["\']?',
        r'["\']?classification["\']?\s*[:=]\s*["\']?(correct|almost|partial|incorrect)["\']?',
        r'["\']?grade["\']?\s*[:=]\s*["\']?(correct|almost|partial|incorrect)["\']?',
        r'["\']?result["\']?\s*[:=]\s*["\']?(correct|almost|partial|incorrect)["\']?',
    ]
    for pattern in simple_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = match.group(1).lower()
            if _is_valid_prediction(value):
                return {"response": value}
    
    # Strategy 10: Try to find and repair truncated JSON
    # Handle cases where JSON is cut off or malformed at the end
    if '{' in text and '}' not in text:
        # JSON might be truncated, try adding closing brace
        truncated = text[text.find('{'):] + '}'
        try:
            return json.loads(truncated)
        except json.JSONDecodeError:
            fixed = _fix_json_string(truncated)
            try:
                return json.loads(fixed)
            except json.JSONDecodeError:
                pass
    
    # Strategy 11: Look for JSON with nested braces that might be split across lines
    # Handle multi-line JSON objects
    if '{' in text:
        # Try to find the outermost JSON object
        start = text.find('{')
        # Find matching closing brace by counting
        count = 0
        end = start
        for i in range(start, len(text)):
            if text[i] == '{':
                count += 1
            elif text[i] == '}':
                count -= 1
                if count == 0:
                    end = i + 1
                    break
        if end > start:
            candidate = text[start:end]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                fixed = _fix_json_string(candidate)
                try:
                    return json.loads(fixed)
                except json.JSONDecodeError:
                    aggressive_fixed = _aggressive_json_fix(fixed)
                    try:
                        return json.loads(aggressive_fixed)
                    except json.JSONDecodeError:
                        pass
    
    # Strategy 12: Extract from LaTeX-style or markdown formatting
    latex_patterns = [
        r'\\text\{response\}\s*[:=]\s*["\']?(correct|almost|partial|incorrect)["\']?',
        r'\\textbf\{(correct|almost|partial|incorrect)\}',
        r'\\textit\{(correct|almost|partial|incorrect)\}',
        r'\\emph\{(correct|almost|partial|incorrect)\}',
    ]
    for pattern in latex_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = match.group(1).lower()
            if _is_valid_prediction(value):
                return {"response": value}
    
    # Strategy 13: Look for response in parentheses or brackets at end of text
    end_patterns = [
        r'[\(\[](correct|almost|partial|incorrect)[\)\]]\s*$',
        r'["\'](correct|almost|partial|incorrect)["\']\s*$',
    ]
    for pattern in end_patterns:
        match = re.search(pattern, text.lower(), re.MULTILINE)
        if match:
            value = match.group(1)
            if _is_valid_prediction(value):
                return {"response": value}
    
    return None


def _aggressive_json_fix(text: str) -> str:
    """Aggressively fix common JSON formatting issues."""
    text = text.strip()
    
    # Remove any text before the first {
    start_idx = text.find('{')
    if start_idx > 0:
        text = text[start_idx:]
    
    # Remove any text after the last }
    end_idx = text.rfind('}')
    if end_idx != -1 and end_idx < len(text) - 1:
        text = text[:end_idx+1]
    
    # Replace all single quotes with double quotes (simpler approach)
    result = []
    in_string = False
    i = 0
    while i < len(text):
        char = text[i]
        if char == '"':
            # Check if escaped
            backslash_count = 0
            j = i - 1
            while j >= 0 and text[j] == '\\':
                backslash_count += 1
                j -= 1
            if backslash_count % 2 == 0:
                in_string = not in_string
            result.append(char)
        elif char == "'" and not in_string:
            result.append('"')
        elif char == '\n' and not in_string:
            result.append(' ')
        elif char == '\t' and not in_string:
            result.append(' ')
        else:
            result.append(char)
        i += 1
    
    fixed = ''.join(result)
    
    # Remove trailing commas
    fixed = re.sub(r',\s*}', '}', fixed)
    fixed = re.sub(r',\s*]', ']', fixed)
    
    # Fix missing quotes around keys
    fixed = re.sub(r'(\{|,\s*)(\w+)(\s*:)', r'\1"\2"\3', fixed)
    
    return fixed


def _ultra_aggressive_json_fix(text: str) -> str:
    """Ultra-aggressive JSON fixing for severely malformed responses.
    
    This is a last-resort function that attempts to extract any valid
    JSON-like structure from text, even if heavily corrupted.
    """
    text = text.strip()
    
    # Remove any text before the first { and after the last }
    start_idx = text.find('{')
    end_idx = text.rfind('}')
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        text = text[start_idx:end_idx+1]
    
    # Replace all single quotes with double quotes globally
    text = text.replace("'", '"')
    
    # Replace all whitespace control characters with spaces
    text = text.replace('\n', ' ').replace('\t', ' ').replace('\r', ' ')
    
    # Remove trailing commas before closing braces/brackets
    text = re.sub(r',\s*}', '}', text)
    text = re.sub(r',\s*]', ']', text)
    
    # Fix missing quotes around keys (more aggressive)
    text = re.sub(r'(\{|,\s*)([a-zA-Z_][a-zA-Z0-9_]*)(\s*:)', r'\1"\2"\3', text)
    
    # Fix unquoted values that look like they should be strings
    # Pattern: : word (not starting with {, [, ", digit, true, false, null)
    text = re.sub(r':\s*([a-zA-Z][a-zA-Z0-9_]*)\s*([,}])', r': "\1" \2', text)
    
    # Remove any non-printable characters
    text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t\r')
    
    # Ensure the text starts with { and ends with }
    if not text.startswith('{'):
        text = '{' + text
    if not text.endswith('}'):
        text = text + '}'
    
    return text


def _extract_key_value_patterns(text: str) -> dict | None:
    """Extract key-value patterns that look like JSON or grading responses.
    
    Enhanced with more comprehensive pattern matching and better value extraction.
    """
    result = {}
    text_lower = text.lower()
    
    # Pattern 1: "response": "value" or 'response': 'value' with various formats
    response_patterns = [
        # Standard JSON-like patterns
        r'["\']?response["\']?\s*[:=]\s*["\']?(correct|almost|partial|incorrect)["\']?',
        r'["\']?classification["\']?\s*[:=]\s*["\']?(correct|almost|partial|incorrect)["\']?',
        r'["\']?grade["\']?\s*[:=]\s*["\']?(correct|almost|partial|incorrect)["\']?',
        r'["\']?result["\']?\s*[:=]\s*["\']?(correct|almost|partial|incorrect)["\']?',
        r'["\']?evaluation["\']?\s*[:=]\s*["\']?(correct|almost|partial|incorrect)["\']?',
        r'["\']?answer["\']?\s*[:=]\s*["\']?(correct|almost|partial|incorrect)["\']?',
        r'["\']?verdict["\']?\s*[:=]\s*["\']?(correct|almost|partial|incorrect)["\']?',
        r'["\']?decision["\']?\s*[:=]\s*["\']?(correct|almost|partial|incorrect)["\']?',
        r'["\']?prediction["\']?\s*[:=]\s*["\']?(correct|almost|partial|incorrect)["\']?',
        r'["\']?label["\']?\s*[:=]\s*["\']?(correct|almost|partial|incorrect)["\']?',
        r'["\']?category["\']?\s*[:=]\s*["\']?(correct|almost|partial|incorrect)["\']?',
        r'["\']?output["\']?\s*[:=]\s*["\']?(correct|almost|partial|incorrect)["\']?',
        # Arrow notation
        r'["\']?response["\']?\s*=>?\s*["\']?(correct|almost|partial|incorrect)["\']?',
        r'["\']?classification["\']?\s*=>?\s*["\']?(correct|almost|partial|incorrect)["\']?',
        # Whitespace variations
        r'\bresponse\s+(correct|almost|partial|incorrect)\b',
        r'\bclassification\s+(correct|almost|partial|incorrect)\b',
        r'\bgrade\s+(correct|almost|partial|incorrect)\b',
        r'\bresult\s+(correct|almost|partial|incorrect)\b',
        r'\bprediction\s+(correct|almost|partial|incorrect)\b',
        r'\blabel\s+(correct|almost|partial|incorrect)\b',
        r'\bcategory\s+(correct|almost|partial|incorrect)\b',
    ]
    
    for pattern in response_patterns:
        match = re.search(pattern, text_lower)
        if match:
            value = match.group(1).lower()
            if _is_valid_prediction(value):
                result["response"] = value
                return result
    
    # Pattern 2: Look for any key with a valid prediction value
    kv_pattern = r'["\']?(\w+)["\']?\s*[:=]\s*["\']?([^"\'\n,}]+)["\']?'
    matches = re.findall(kv_pattern, text)
    for key, value in matches:
        value = value.strip().strip('"\'').lower()
        norm_value = _normalize_prediction(value)
        if norm_value in ["correct", "almost", "partial", "incorrect"]:
            result[key] = norm_value
    
    # Pattern 3: Look for standalone classification words in structured contexts
    # This handles cases where the LLM outputs just the word without proper JSON
    structured_patterns = [
        r'^\s*(correct|almost|partial|incorrect)\s*$',  # Just the word on its own line
        r'\bthe\s+(?:answer|classification|grade|result)\s+is\s+(correct|almost|partial|incorrect)',
        r'\bthis\s+(?:answer|solution|work)\s+is\s+(correct|almost|partial|incorrect)',
        r'\bi\s+(?:would\s+)?classify\s+(?:this|it)\s+as\s+(correct|almost|partial|incorrect)',
        r'\bthis\s+(?:should|would)\s+be\s+(correct|almost|partial|incorrect)',
        r'\bthe\s+(?:student\s+)?(?:answer|solution|work)\s+is\s+(correct|almost|partial|incorrect)',
        r'\bgiven\s+(?:the\s+)?(?:classification|grade)\s+(?:of\s+)?(correct|almost|partial|incorrect)',
        r'\bassigned\s+(?:the\s+)?(?:classification|grade)\s+(?:of\s+)?(correct|almost|partial|incorrect)',
    ]
    
    for pattern in structured_patterns:
        match = re.search(pattern, text_lower)
        if match:
            value = match.group(1).lower()
            if _is_valid_prediction(value):
                result["response"] = value
                return result
    
    # Pattern 4: Look for rubric markers in the text
    rubric_patterns = [
        (r'\(\s*correct\s*\)', "correct"),
        (r'\(\s*almost\s*\)', "almost"),
        (r'\(\s*partial\s*\)', "partial"),
        (r'\(\s*incorrect\s*\)', "incorrect"),
        (r'\[\s*correct\s*\]', "correct"),
        (r'\[\s*almost\s*\]', "almost"),
        (r'\[\s*partial\s*\]', "partial"),
        (r'\[\s*incorrect\s*\]', "incorrect"),
        (r'\{\s*correct\s*\}', "correct"),
        (r'\{\s*almost\s*\}', "almost"),
        (r'\{\s*partial\s*\}', "partial"),
        (r'\{\s*incorrect\s*\}', "incorrect"),
    ]
    
    for pattern, label in rubric_patterns:
        if re.search(pattern, text_lower):
            result["response"] = label
            return result
    
    return result if result else None


def _fix_json_string(text: str) -> str:
    """Fix common JSON formatting issues in LLM responses."""
    text = text.strip()
    
    # Replace single quotes with double quotes (carefully)
    result = []
    in_string = False
    string_char = None
    i = 0
    while i < len(text):
        char = text[i]
        if not in_string:
            if char in '"\'':
                in_string = True
                string_char = char
                result.append('"')
            elif char == '{':
                result.append(char)
            elif char == '}':
                # Remove trailing comma before closing brace
                while result and result[-1] in ' \t,':
                    result.pop()
                result.append(char)
            else:
                result.append(char)
        else:
            if char == string_char:
                # Check if it's escaped
                backslash_count = 0
                j = i - 1
                while j >= 0 and text[j] == '\\':
                    backslash_count += 1
                    j -= 1
                if backslash_count % 2 == 0:
                    # Not escaped, end of string
                    in_string = False
                    string_char = None
                    result.append('"')
                else:
                    result.append(char)
            elif char == '"':
                # Escape double quotes inside string
                result.append('\\"')
            elif char == '\n':
                # Replace newlines with escaped newlines
                result.append('\\n')
            else:
                result.append(char)
        i += 1
    
    fixed = ''.join(result)
    
    # Remove trailing commas before closing braces/brackets
    fixed = re.sub(r',(\s*[}\]])', r'\1', fixed)
    
    return fixed


def _normalize_prediction(raw_value) -> str:
    """Normalize a raw prediction value to one of the valid labels.
    
    Valid labels: "correct", "incorrect", "partial", "almost"
    
    This function handles various formats and variations that LLMs might produce,
    including quoted strings, boolean-like values, and descriptive phrases.
    Enhanced with more comprehensive pattern matching and better edge case handling.
    """
    if raw_value is None:
        return "unknown"
    
    # Handle non-string types
    if isinstance(raw_value, bool):
        return "correct" if raw_value else "incorrect"
    
    if isinstance(raw_value, (int, float)):
        if raw_value >= 0.9:
            return "correct"
        elif raw_value >= 0.7:
            return "almost"
        elif raw_value >= 0.4:
            return "partial"
        else:
            return "incorrect"
    
    raw_str = str(raw_value).lower().strip()
    
    # Direct matches (exact)
    if raw_str in ["correct", "incorrect", "partial", "almost"]:
        return raw_str
    
    # Check for exact matches with quotes removed
    clean_str = raw_str.strip('"\'').strip()
    if clean_str in ["correct", "incorrect", "partial", "almost"]:
        return clean_str
    
    # Handle common variations for "correct"
    correct_variations = [
        "true", "yes", "right", "valid", "1", "full", "complete", 
        "perfect", "flawless", "accurate", "proper", "sound",
        "fully correct", "entirely correct", "totally correct",
        "100%", "full marks", "full credit", "all correct",
        "excellent", "outstanding", "satisfactory", "acceptable",
        "pass", "passed", "success", "successful", "good",
        "well done", "correct solution", "correct answer",
        "right answer", "valid solution", "proper solution",
        "complete solution", "complete proof", "rigorous",
        "mathematically correct", "logically valid", "all steps correct",
        "no errors", "100% correct", "fully rigorous", "precise",
        "exact", "well reasoned", "good work", "right",
        "full score", "max score", "maximum score", "max credit",
        "maximum credit", "max marks", "maximum marks",
    ]
    if raw_str in correct_variations or clean_str in correct_variations:
        return "correct"
    
    # Handle common variations for "incorrect"
    incorrect_variations = [
        "false", "no", "wrong", "invalid", "0", "none", "fail", 
        "error", "bad", "unsatisfactory", "rejected", "failed",
        "not correct", "not right", "not valid", "zero",
        "no credit", "full wrong", "entirely wrong",
        "completely wrong", "totally wrong", "incorrect solution",
        "wrong answer", "invalid answer", "fail", "failing",
        "unsatisfactory", "poor", "inadequate", "deficient",
        "not correct", "not right", "not valid", "not accurate",
        "0% correct", "zero credit", "no marks", "zero marks",
        "fundamentally wrong", "totally incorrect", "completely incorrect",
        "entirely incorrect", "fatally flawed", "wrong solution",
        "no meaningful work", "no valid work", "irrelevant work",
        "off topic", "serious error", "fatal error", "logical error",
        "conceptual error", "no solution", "blank", "no progress",
        "zero score", "no score", "0 score", "failed",
        "not applicable", "n/a", "na", "none",
    ]
    if raw_str in incorrect_variations or clean_str in incorrect_variations:
        return "incorrect"
    
    # Handle common variations for "partial"
    partial_variations = [
        "part", "partially", "incomplete", "half", "some", 
        "mostly wrong", "mostly incorrect", "partial credit",
        "half credit", "some credit", "in progress",
        "partially correct", "half correct", "semi correct",
        "50%", "40%", "60%", "mixed", "incomplete solution",
        "partial solution", "some progress", "meaningful progress",
        "significant progress", "on the right track", "good start",
        "partial answer", "incomplete answer", "some correct",
        "partially right", "half right", "semi right",
        "partial success", "incomplete work", "unfinished",
        "partial proof", "incomplete proof", "some progress",
        "started correctly", "correct approach", "correct start",
        "correct idea", "correct method", "correct strategy",
        "50% correct", "60% correct", "40% correct",
        "halfway there", "partial success", "some right",
        "partial score", "half score", "some score",
        "minor credit", "partial marks", "half marks",
    ]
    if raw_str in partial_variations or clean_str in partial_variations:
        return "partial"
    
    # Handle common variations for "almost"
    almost_variations = [
        "almost correct", "close", "minor errors", "nearly", 
        "mostly correct", "trivial errors", "almost there",
        "near correct", "very close", "minor mistake",
        "small error", "slight error", "nearly right",
        "80%", "90%", "minor issue", "trivial issue",
        "almost right", "nearly correct", "close to correct",
        "minor flaw", "small flaw", "trivial flaw",
        "cosmetic error", "formatting error", "rounding error",
        "negligible error", "insignificant error", "tiny error",
        "almost perfect", "nearly perfect", "very good",
        "minor correction", "small correction", "slight correction",
        "tiny error", "minor mistake", "small mistake", "slight mistake",
        "minor omission", "small omission", "trivial mistake",
        "notational error", "sign error", "arithmetic error",
        "computation error", "95% correct", "90% correct",
        "most credit", "most marks", "most score",
        "significant credit", "high credit", "substantial credit",
    ]
    if raw_str in almost_variations or clean_str in almost_variations:
        return "almost"
    
    # Check for substring matches (more specific first to avoid misclassification)
    # Check for "almost" patterns - be careful to check before "partial" to avoid overlap
    almost_patterns = [
        "almost", "nearly", "minor", "trivial", "slight", "small error",
        "cosmetic", "formatting", "rounding", "negligible", "insignificant",
        "tiny error", "minor mistake", "small mistake", "slight mistake",
        "close to", "very close", "nearly right", "almost right",
        "80%", "90%", "minor flaw", "small flaw", "trivial flaw",
        "minor omission", "small omission", "notational error",
        "sign error", "arithmetic error", "computation error",
        "95% correct", "90% correct", "almost perfect", "nearly perfect",
        "most credit", "most marks", "most score",
    ]
    if any(term in raw_str for term in almost_patterns):
        return "almost"
    
    # Check for "partial" patterns
    partial_patterns = [
        "partial", "incomplete", "half", "some progress", "meaningful progress",
        "significant progress", "on the right track", "good start",
        "partially", "semi", "mixed", "50%", "40%", "60%",
        "incomplete solution", "partial solution", "partial answer",
        "unfinished", "in progress", "some correct", "half correct",
        "partial proof", "incomplete proof", "some progress",
        "started correctly", "correct approach", "correct start",
        "correct idea", "correct method", "correct strategy",
        "50% correct", "60% correct", "40% correct",
        "halfway there", "partial success", "some right",
        "partial score", "half score", "some score",
    ]
    if any(term in raw_str for term in partial_patterns):
        return "partial"
    
    # Check for "incorrect" patterns (before "correct" to avoid substring issues)
    incorrect_patterns = [
        "incorrect", "wrong", "invalid", "error", "not correct", "not right",
        "not valid", "fail", "failed", "rejected", "unsatisfactory",
        "completely wrong", "totally wrong", "entirely wrong",
        "fundamentally wrong", "no progress", "irrelevant", "off track",
        "major error", "critical error", "serious error", "fatal error",
        "0% correct", "zero credit", "no marks", "zero marks",
        "fundamentally wrong", "totally incorrect", "completely incorrect",
        "entirely incorrect", "fatally flawed", "wrong solution",
        "no meaningful work", "no valid work", "irrelevant work",
        "off topic", "serious error", "fatal error", "logical error",
        "conceptual error", "no solution", "blank", "no progress",
        "zero score", "no score", "0 score",
    ]
    if any(term in raw_str for term in incorrect_patterns):
        return "incorrect"
    
    # Check for "correct" patterns - be careful about negation
    correct_patterns = [
        "correct", "right", "valid", "accurate", "proper", "sound",
        "perfect", "flawless", "excellent", "outstanding", "complete",
        "full marks", "full credit", "100%", "true", "yes",
        "satisfactory", "acceptable", "pass", "success", "good",
        "complete solution", "complete proof", "rigorous",
        "mathematically correct", "logically valid", "all steps correct",
        "no errors", "100% correct", "fully rigorous", "precise",
        "exact", "well reasoned", "good work",
        "full score", "max score", "maximum score",
    ]
    # Only match if not negated
    for term in correct_patterns:
        if term in raw_str:
            # Check for negation
            idx = raw_str.find(term)
            before = raw_str[max(0, idx-20):idx]
            if not any(neg in before for neg in ['not ', 'in', "isn't", 'isnt', 'not,', 'not.', 'not:']):
                return "correct"
    
    return "unknown"


def _is_valid_prediction(prediction: str) -> bool:
    """Check if a prediction is one of the valid labels."""
    return prediction in ["correct", "incorrect", "partial", "almost"]


def _parse_grading_guidelines(guidelines: str) -> dict:
    """Parse grading guidelines to extract rubric indicators with enhanced accuracy.
    
    This function performs comprehensive analysis of grading guidelines to identify
    explicit markers, score patterns, and contextual indicators that help determine
    the appropriate classification for a student answer.
    
    Enhanced with more comprehensive marker detection and contextual analysis.
    """
    result = {
        "has_partial": False,
        "has_almost": False,
        "has_correct": False,
        "has_incorrect": False,
        "partial_context": [],
        "almost_context": [],
        "correct_context": [],
        "incorrect_context": [],
        "score_hints": {},
        "primary_category": None,
        "confidence": 0.0,
        "all_markers": [],  # Track all found markers for debugging
        "marker_count": {"partial": 0, "almost": 0, "correct": 0, "incorrect": 0},
    }
    
    if not guidelines:
        return result
    
    guidelines_lower = guidelines.lower()
    
    # Look for explicit markers with comprehensive patterns
    # Priority order: exact markers > credit markers > score patterns > alternative markers
    markers = [
        # === EXACT RUBRIC MARKERS (highest priority - 1.0) ===
        # Parentheses format - most common
        (r'\(\s*Partial\s*\)', "partial", 1.0),
        (r'\(\s*Almost\s*\)', "almost", 1.0),
        (r'\(\s*Correct\s*\)', "correct", 1.0),
        (r'\(\s*Incorrect\s*\)', "incorrect", 1.0),
        # Case variations
        (r'\(\s*PARTIAL\s*\)', "partial", 1.0),
        (r'\(\s*ALMOST\s*\)', "almost", 1.0),
        (r'\(\s*CORRECT\s*\)', "correct", 1.0),
        (r'\(\s*INCORRECT\s*\)', "incorrect", 1.0),
        
        # === CREDIT-BASED MARKERS (0.90-0.98) ===
        (r'\(\s*Full\s*Credit\s*\)', "correct", 0.98),
        (r'\(\s*No\s*Credit\s*\)', "incorrect", 0.98),
        (r'\(\s*Half\s*Credit\s*\)', "partial", 0.95),
        (r'\(\s*Most\s*Credit\s*\)', "almost", 0.95),
        (r'\(\s*Some\s*Credit\s*\)', "partial", 0.90),
        (r'\(\s*Partial\s*Credit\s*\)', "partial", 0.95),
        (r'\(\s*Full\s*Marks?\s*\)', "correct", 0.98),
        (r'\(\s*No\s*Marks?\s*\)', "incorrect", 0.98),
        (r'\(\s*Half\s*Marks?\s*\)', "partial", 0.95),
        (r'\(\s*Zero\s*(?:Credit|Marks?)?\s*\)', "incorrect", 0.98),
        (r'\(\s*Maximum\s*(?:Credit|Marks?|Points?)\s*\)', "correct", 0.98),
        
        # === ADDITIONAL CREDIT MARKERS (0.85-0.95) ===
        (r'\(\s*Full\s*Score\s*\)', "correct", 0.95),
        (r'\(\s*Zero\s*Score\s*\)', "incorrect", 0.95),
        (r'\(\s*Partial\s*Score\s*\)', "partial", 0.90),
        (r'\(\s*Minor\s*Credit\s*\)', "partial", 0.85),
        (r'\(\s*Significant\s*Credit\s*\)', "almost", 0.90),
        
        # === SCORE-BASED PATTERNS (0.80-0.90) ===
        (r'\(\s*\d+\s*/\s*\d+\s*\)', "score_based", 0.85),
        (r'\(\s*\d+\s*points?\s*\)', "score_based", 0.85),
        (r'\(\s*\d+\s*pts?\s*\)', "score_based", 0.85),
        (r'\[\s*\d+\s*/\s*\d+\s*\]', "score_based", 0.85),
        (r'\[\s*\d+\s*points?\s*\]', "score_based", 0.85),
        
        # === BRACKET FORMATS (0.85-0.90) ===
        # Square brackets
        (r'\[\s*Partial\s*\]', "partial", 0.90),
        (r'\[\s*Almost\s*\]', "almost", 0.90),
        (r'\[\s*Correct\s*\]', "correct", 0.90),
        (r'\[\s*Incorrect\s*\]', "incorrect", 0.90),
        # Curly braces
        (r'\{\s*Partial\s*\}', "partial", 0.90),
        (r'\{\s*Almost\s*\}', "almost", 0.90),
        (r'\{\s*Correct\s*\}', "correct", 0.90),
        (r'\{\s*Incorrect\s*\}', "incorrect", 0.90),
        # Angle brackets
        (r'<\s*Partial\s*>', "partial", 0.85),
        (r'<\s*Almost\s*>', "almost", 0.85),
        (r'<\s*Correct\s*>', "correct", 0.85),
        (r'<\s*Incorrect\s*>', "incorrect", 0.85),
        
        # === TEXT-BASED MARKERS (0.75-0.85) ===
        (r'\bPartial\s*Credit\b', "partial", 0.85),
        (r'\bAlmost\s*Correct\b', "almost", 0.85),
        (r'\bFull\s*Credit\b', "correct", 0.85),
        (r'\bNo\s*Credit\b', "incorrect", 0.85),
        (r'\bHalf\s*Credit\b', "partial", 0.85),
        (r'\bSome\s*Credit\b', "partial", 0.80),
        (r'\bMost\s*Credit\b', "almost", 0.85),
        
        # === AWARD/GIVE PATTERNS (0.75-0.85) ===
        (r'award\s+\d+\s*points?', "score_based", 0.80),
        (r'give\s+\d+\s*points?', "score_based", 0.80),
        (r'grant\s+\d+\s*points?', "score_based", 0.80),
        (r'assign\s+\d+\s*points?', "score_based", 0.80),
        (r'\d+\s*points?\s*awarded', "score_based", 0.80),
        (r'\d+\s*points?\s*given', "score_based", 0.80),
        
        # === FORMATTING MARKERS (0.80-0.85) ===
        (r'\*\s*Partial\s*\*', "partial", 0.85),
        (r'\*\s*Almost\s*\*', "almost", 0.85),
        (r'\*\s*Correct\s*\*', "correct", 0.85),
        (r'\*\s*Incorrect\s*\*', "incorrect", 0.85),
        (r'__Partial__', "partial", 0.85),
        (r'__Almost__', "almost", 0.85),
        (r'__Correct__', "correct", 0.85),
        (r'__Incorrect__', "incorrect", 0.85),
        (r'\*\*Partial\*\*', "partial", 0.85),
        (r'\*\*Almost\*\*', "almost", 0.85),
        (r'\*\*Correct\*\*', "correct", 0.85),
        (r'\*\*Incorrect\*\*', "incorrect", 0.85),
        
        # === COLON AND HEADER MARKERS (0.75-0.80) ===
        (r'Partial\s*:', "partial", 0.80),
        (r'Almost\s*:', "almost", 0.80),
        (r'Correct\s*:', "correct", 0.80),
        (r'Incorrect\s*:', "incorrect", 0.80),
        (r'^\s*Partial\s*$', "partial", 0.80),
        (r'^\s*Almost\s*$', "almost", 0.80),
        (r'^\s*Correct\s*$', "correct", 0.80),
        (r'^\s*Incorrect\s*$', "incorrect", 0.80),
        
        # === VERB-BASED MARKERS (0.70-0.80) ===
        (r'\b(?:is|are|was|were)\s+partial\b', "partial", 0.75),
        (r'\b(?:is|are|was|were)\s+almost\s+correct\b', "almost", 0.80),
        (r'\b(?:is|are|was|were)\s+correct\b', "correct", 0.75),
        (r'\b(?:is|are|was|were)\s+incorrect\b', "incorrect", 0.75),
        (r'\bclassified\s+as\s+partial\b', "partial", 0.80),
        (r'\bclassified\s+as\s+almost\b', "almost", 0.80),
        (r'\bclassified\s+as\s+correct\b', "correct", 0.80),
        (r'\bclassified\s+as\s+incorrect\b', "incorrect", 0.80),
        
        # === NUMERIC SCORE PATTERNS (0.75-0.85) ===
        (r'\bscore\s*[:=]\s*\d+', "score_based", 0.80),
        (r'\bgrade\s*[:=]\s*\d+', "score_based", 0.80),
        (r'\bpoints?\s*[:=]\s*\d+', "score_based", 0.80),
        (r'\bmarks?\s*[:=]\s*\d+', "score_based", 0.80),
    ]
    
    category_scores = {"partial": 0, "almost": 0, "correct": 0, "incorrect": 0}
    
    for pattern, label, confidence in markers:
        matches = list(re.finditer(pattern, guidelines, re.IGNORECASE))
        if matches:
            if label != "score_based":
                result[f"has_{label}"] = True
                result["marker_count"][label] += len(matches)
                # Boost confidence if multiple markers found
                multiplier = min(1.0 + (len(matches) - 1) * 0.1, 1.2)  # Up to 20% boost
                category_scores[label] = max(category_scores[label], confidence * multiplier)
            
            for match in matches:
                # Extract context around the marker (expanded range for more context)
                start = max(0, match.start() - 200)
                end = min(len(guidelines), match.end() + 250)
                context = guidelines[start:end].replace('\n', ' ').strip()
                if label != "score_based":
                    result[f"{label}_context"].append(context)
                    result["all_markers"].append({
                        "type": label,
                        "confidence": confidence,
                        "context": context[:100],
                        "position": match.start()
                    })
    
    # Extract score/point information with enhanced detail
    score_patterns = [
        (r'(\d+)\s*points?', "points"),
        (r'(\d+)\s*pts?', "points_short"),
        (r'score[:\s]+(\d+)', "score"),
        (r'\((\d+)\s*/\s*(\d+)\)', "fraction"),
        (r'\[(\d+)\s*/\s*(\d+)\]', "fraction_bracket"),
        (r'(\d+)\s*/\s*(\d+)\s*points?', "fraction_points"),
        (r'award[:\s]+(\d+)', "award"),
        (r'give[:\s]+(\d+)', "give"),
        (r'grant[:\s]+(\d+)', "grant"),
        (r'assign[:\s]+(\d+)', "assign"),
        (r'worth[:\s]+(\d+)', "worth"),
        (r'value[:\s]+(\d+)', "value"),
        (r'(\d+)\s*marks?', "marks"),
        (r'max(?:imum)?[:\s]+(\d+)', "max"),
        (r'total[:\s]+(\d+)', "total"),
        (r'full[:\s]+(\d+)', "full"),
        (r'out\s+of\s+(\d+)', "out_of"),
    ]
    
    extracted_scores = []
    max_points = None
    score_ratios = []
    
    for pattern, score_type in score_patterns:
        if score_type in ["fraction", "fraction_points", "fraction_bracket"]:
            matches = re.findall(pattern, guidelines, re.IGNORECASE)
            for match in matches:
                if len(match) == 2:
                    try:
                        score, total = int(match[0]), int(match[1])
                        if total > 0:
                            extracted_scores.append(score)
                            score_ratios.append(score / total)
                            if max_points is None or total > max_points:
                                max_points = total
                            # Infer category from score ratio with confidence weighting
                            ratio = score / total
                            if ratio >= 0.95:
                                category_scores["correct"] = max(category_scores["correct"], 0.95)
                            elif ratio >= 0.85:
                                category_scores["correct"] = max(category_scores["correct"], 0.85)
                            elif ratio >= 0.70:
                                category_scores["almost"] = max(category_scores["almost"], 0.90)
                            elif ratio >= 0.50:
                                category_scores["partial"] = max(category_scores["partial"], 0.85)
                            elif ratio >= 0.30:
                                category_scores["partial"] = max(category_scores["partial"], 0.75)
                            elif ratio >= 0.10:
                                category_scores["partial"] = max(category_scores["partial"], 0.60)
                            else:
                                category_scores["incorrect"] = max(category_scores["incorrect"], 0.90)
                    except (ValueError, IndexError):
                        pass
        elif score_type in ["max", "total", "full", "out_of"]:
            # These patterns indicate the maximum possible points
            matches = re.findall(pattern, guidelines, re.IGNORECASE)
            for match in matches:
                try:
                    if isinstance(match, tuple):
                        match = match[0]
                    total = int(match)
                    if max_points is None or total > max_points:
                        max_points = total
                except (ValueError, IndexError):
                    pass
        else:
            matches = re.findall(pattern, guidelines, re.IGNORECASE)
            for match in matches:
                try:
                    if isinstance(match, tuple):
                        match = match[0]
                    score = int(match)
                    extracted_scores.append(score)
                except (ValueError, IndexError):
                    pass
    
    if extracted_scores:
        result["score_hints"]["extracted_scores"] = extracted_scores
        result["score_hints"]["max_points"] = max_points
        result["score_hints"]["score_ratios"] = score_ratios
        if score_ratios:
            result["score_hints"]["avg_ratio"] = sum(score_ratios) / len(score_ratios)
    
    # Look for keywords that indicate quality with weighted scoring
    # Enhanced with more comprehensive keyword lists
    quality_keywords = {
        "correct": [
            ("complete", 0.7), ("correct", 0.8), ("valid", 0.6), 
            ("proper", 0.6), ("full", 0.7), ("perfect", 0.9),
            ("flawless", 0.9), ("excellent", 0.8), ("right answer", 0.8),
            ("fully correct", 0.9), ("entirely correct", 0.9),
            ("correct solution", 0.85), ("correct answer", 0.85),
            ("satisfactory", 0.7), ("acceptable", 0.7),
            ("complete solution", 0.85), ("complete proof", 0.85),
            ("rigorous", 0.8), ("sound", 0.75), ("accurate", 0.8),
            ("precise", 0.75), ("exact", 0.8), ("right", 0.7),
            ("well done", 0.75), ("good work", 0.7), ("well reasoned", 0.8),
            ("logically valid", 0.85), ("mathematically correct", 0.9),
            ("all steps correct", 0.9), ("no errors", 0.85),
            ("100% correct", 0.95), ("fully rigorous", 0.9),
        ],
        "almost": [
            ("minor", 0.7), ("small", 0.6), ("slight", 0.6), 
            ("typo", 0.8), ("nearly", 0.8), ("close", 0.7),
            ("trivial", 0.7), ("insignificant", 0.6), ("almost correct", 0.9),
            ("mostly correct", 0.8), ("small error", 0.75),
            ("minor error", 0.75), ("slight mistake", 0.75),
            ("cosmetic", 0.6), ("formatting", 0.5),
            ("rounding", 0.6), ("negligible", 0.7),
            ("tiny error", 0.75), ("minor mistake", 0.75),
            ("small mistake", 0.75), ("slight error", 0.75),
            ("almost there", 0.8), ("very close", 0.75),
            ("minor omission", 0.7), ("small omission", 0.7),
            ("trivial mistake", 0.75), ("cosmetic error", 0.6),
            ("notational error", 0.65), ("sign error", 0.7),
            ("arithmetic error", 0.7), ("computation error", 0.7),
            ("minor flaw", 0.7), ("small flaw", 0.7),
            ("95% correct", 0.9), ("90% correct", 0.85),
        ],
        "partial": [
            ("partial", 0.8), ("incomplete", 0.7), ("missing", 0.6), 
            ("some", 0.5), ("half", 0.7), ("progress", 0.6),
            ("attempt", 0.5), ("partial credit", 0.9), ("partially correct", 0.8),
            ("on the right track", 0.7), ("meaningful progress", 0.75),
            ("significant progress", 0.7), ("good start", 0.6),
            ("some correct", 0.7), ("partial solution", 0.8),
            ("incomplete solution", 0.75), ("missing steps", 0.65),
            ("partial proof", 0.75), ("incomplete proof", 0.7),
            ("some progress", 0.65), ("partial answer", 0.7),
            ("incomplete answer", 0.65), ("half correct", 0.75),
            ("partially right", 0.7), ("some right", 0.65),
            ("started correctly", 0.6), ("correct approach", 0.7),
            ("correct start", 0.65), ("correct idea", 0.65),
            ("correct method", 0.7), ("correct strategy", 0.7),
            ("50% correct", 0.8), ("60% correct", 0.75), ("40% correct", 0.7),
            ("halfway there", 0.7), ("partial success", 0.65),
        ],
        "incorrect": [
            ("wrong", 0.8), ("invalid", 0.7), ("error", 0.6), 
            ("incorrect", 0.8), ("fail", 0.7), ("no credit", 0.9),
            ("fundamentally wrong", 0.9), ("does not work", 0.7),
            ("completely wrong", 0.9), ("no progress", 0.7),
            ("irrelevant", 0.8), ("off track", 0.75),
            ("major error", 0.8), ("critical error", 0.85),
            ("incorrect approach", 0.8), ("wrong method", 0.8),
            ("no solution", 0.75), ("blank", 0.7),
            ("totally wrong", 0.9), ("entirely wrong", 0.9),
            ("completely incorrect", 0.9), ("totally incorrect", 0.9),
            ("fundamentally incorrect", 0.9), ("fatally flawed", 0.9),
            ("wrong answer", 0.85), ("incorrect answer", 0.85),
            ("wrong solution", 0.85), ("incorrect solution", 0.85),
            ("no meaningful work", 0.8), ("no valid work", 0.8),
            ("irrelevant work", 0.8), ("off topic", 0.75),
            ("0% correct", 0.95), ("zero credit", 0.95),
            ("serious error", 0.85), ("fatal error", 0.9),
            ("logical error", 0.8), ("conceptual error", 0.85),
        ],
    }
    
    for category, keywords in quality_keywords.items():
        for keyword, weight in keywords:
            if keyword in guidelines_lower:
                result[f"has_{category}"] = True
                category_scores[category] = max(category_scores[category], weight)
    
    # Determine primary category based on scores with tie-breaking logic
    if any(v > 0 for v in category_scores.values()):
        # Sort by score, then by priority order for ties
        priority_order = {"correct": 4, "almost": 3, "partial": 2, "incorrect": 1}
        sorted_categories = sorted(
            category_scores.items(),
            key=lambda x: (x[1], priority_order.get(x[0], 0)),
            reverse=True
        )
        best_category = sorted_categories[0][0]
        best_score = sorted_categories[0][1]
        if best_score > 0:
            result["primary_category"] = best_category
            result["confidence"] = best_score
            result["all_scores"] = dict(sorted_categories)
    
    return result


class TaskAgent:
    """Task agent that solves IMO grading problems."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem."""
        # Extract key information from inputs for better prompting
        problem = inputs.get('problem', '')
        solution = inputs.get('solution', '')
        grading_guidelines = inputs.get('grading_guidelines', '')
        student_answer = inputs.get('student_answer', '')
        points = inputs.get('points', '')
        reward = inputs.get('reward', '')
        
        # Parse grading guidelines to extract rubric indicators
        rubric = _parse_grading_guidelines(grading_guidelines)
        
        # Build rubric context for the prompt with enhanced signals
        rubric_context = ""
        
        # Primary category signal (highest priority)
        if rubric.get("primary_category") and rubric.get("confidence", 0) > 0.7:
            primary = rubric["primary_category"]
            confidence = rubric["confidence"]
            rubric_context += f"\n\n[STRONG RUBRIC SIGNAL: The grading guidelines strongly indicate this solution should be classified as '{primary}' (confidence: {confidence:.0%}). This is the PRIMARY classification hint.]"
        
        # Individual marker signals
        if rubric["has_partial"]:
            rubric_context += "\n\n[RUBRIC MARKER: The grading guidelines contain '(Partial)' or related markers, indicating partial credit. Consider 'partial' if the solution shows some progress but is incomplete.]"
            if rubric["partial_context"]:
                rubric_context += f"\nContext: {rubric['partial_context'][0][:250]}"
        if rubric["has_almost"]:
            rubric_context += "\n\n[RUBRIC MARKER: The grading guidelines contain '(Almost)' or related markers, indicating near-correctness with minor issues. Consider 'almost' if the solution is nearly correct with only trivial errors.]"
            if rubric["almost_context"]:
                rubric_context += f"\nContext: {rubric['almost_context'][0][:250]}"
        if rubric["has_correct"]:
            rubric_context += "\n\n[RUBRIC MARKER: The grading guidelines contain '(Correct)' or related markers, indicating full correctness. Consider 'correct' if the solution is essentially flawless.]"
            if rubric["correct_context"]:
                rubric_context += f"\nContext: {rubric['correct_context'][0][:250]}"
        if rubric["has_incorrect"]:
            rubric_context += "\n\n[RUBRIC MARKER: The grading guidelines contain '(Incorrect)' or related markers, indicating no credit. Consider 'incorrect' if the solution is fundamentally wrong.]"
            if rubric["incorrect_context"]:
                rubric_context += f"\nContext: {rubric['incorrect_context'][0][:250]}"
        
        # Score-based guidance with interpretation
        if rubric.get("score_hints", {}).get("extracted_scores"):
            scores = rubric["score_hints"]["extracted_scores"]
            max_pts = rubric["score_hints"].get("max_points")
            if max_pts and max_pts > 0:
                # Calculate ratio for interpretation
                avg_score = sum(scores) / len(scores)
                ratio = avg_score / max_pts
                ratio_guidance = ""
                if ratio >= 0.9:
                    ratio_guidance = "This high score ratio suggests 'correct' classification."
                elif ratio >= 0.7:
                    ratio_guidance = "This score ratio suggests 'almost' classification."
                elif ratio >= 0.4:
                    ratio_guidance = "This score ratio suggests 'partial' classification."
                else:
                    ratio_guidance = "This low score ratio suggests 'incorrect' classification."
                rubric_context += f"\n\n[SCORE INFORMATION: Extracted scores: {scores} out of {max_pts}. {ratio_guidance}]"
        
        instruction = f"""You are an expert mathematical grader evaluating student solutions to competition mathematics problems (like IMO, Putnam, etc.).

Your task is to evaluate the student's answer and classify it into EXACTLY ONE of these four categories:

1. "correct" - The student's answer is fully correct, complete, and rigorous. It matches or exceeds the official solution in completeness and correctness. Only use this if the solution is essentially flawless with no significant errors or omissions.

2. "almost" - The student's answer is nearly correct with only minor errors, typos, or small omissions that don't significantly affect the overall correctness. The core logic is sound but there are small imperfections. The student demonstrates strong understanding with only trivial mistakes.

3. "partial" - The student's answer has some correct elements, shows meaningful progress, or demonstrates understanding of key concepts, but is incomplete, has significant gaps, or contains major errors that prevent it from being fully correct. The student is on the right track but hasn't reached a complete solution.

4. "incorrect" - The student's answer is fundamentally wrong, contains major errors, shows no meaningful progress toward the solution, or is irrelevant/off-track. The approach or answer is completely wrong.

=== CLASSIFICATION DECISION TREE (FOLLOW THIS ORDER EXACTLY) ===

Step 1: RUBRIC MARKERS (HIGHEST PRIORITY - FOLLOW THESE EXPLICITLY)
The Grading Guidelines below contain EXPLICIT markers that indicate the intended classification. These markers are placed by human graders and are the STRONGEST signal of how to classify:

EXACT MARKERS (use these exact classifications):
- (Correct), [Correct], {{Correct}}, **Correct**, __Correct__, *Correct* → Use "correct"
- (Almost), [Almost], {{Almost}}, **Almost**, __Almost__, *Almost* → Use "almost"  
- (Partial), [Partial], {{Partial}}, **Partial**, __Partial__, *Partial* → Use "partial"
- (Incorrect), [Incorrect], {{Incorrect}}, **Incorrect**, __Incorrect__, *Incorrect* → Use "incorrect"

CREDIT MARKERS:
- (Full Credit), (Full Marks), (Maximum Credit), (Full Score) → Use "correct"
- (Most Credit), (Most Marks), (Significant Credit) → Use "almost"
- (Half Credit), (Some Credit), (Partial Credit), (Minor Credit) → Use "partial"
- (No Credit), (Zero Credit), (No Marks), (Zero Score) → Use "incorrect"

SCORE PATTERNS (X/Y points):
- Score ratio ≥ 95% → "correct"
- Score ratio 70-94% → "almost"
- Score ratio 30-69% → "partial"
- Score ratio < 30% → "incorrect"

CRITICAL: Unless the student's answer is completely off-topic, blank, or uses a fundamentally wrong approach, TRUST THE RUBRIC MARKERS. They represent the ground truth classification intended by the problem setters.

Step 2: CONTEXTUAL KEYWORDS (HIGH PRIORITY)
Look for these keywords in the grading guidelines:

For "correct": complete, correct, valid, proper, full, perfect, flawless, excellent, fully correct, entirely correct, correct solution, correct answer, satisfactory, acceptable, rigorous, sound, accurate, precise, exact, all steps correct, no errors, 100% correct, full score, max score, maximum score

For "almost": minor, small, slight, typo, nearly, close, trivial, insignificant, almost correct, mostly correct, small error, minor error, slight mistake, cosmetic, formatting, rounding, negligible, tiny error, minor mistake, almost there, very close, minor omission, notational error, sign error, arithmetic error, computation error, 90% correct, 95% correct, most credit, most marks, most score

For "partial": partial, incomplete, missing, some, half, progress, attempt, partial credit, partially correct, on the right track, meaningful progress, significant progress, good start, some correct, partial solution, incomplete solution, missing steps, partial proof, incomplete proof, partial answer, incomplete answer, half correct, started correctly, correct approach, correct start, correct idea, correct method, correct strategy, 50% correct, 40% correct, 60% correct, partial score, half score, some score, minor credit, partial marks, half marks

For "incorrect": wrong, invalid, error, incorrect, fail, no credit, fundamentally wrong, does not work, completely wrong, no progress, irrelevant, off track, major error, critical error, incorrect approach, wrong method, no solution, blank, totally wrong, entirely wrong, completely incorrect, totally incorrect, fundamentally incorrect, fatally flawed, wrong answer, incorrect answer, wrong solution, incorrect solution, no meaningful work, no valid work, irrelevant work, off topic, serious error, fatal error, logical error, conceptual error, 0% correct, zero credit, zero score, no score, 0 score

Step 3: SOLUTION ANALYSIS (when rubric is ambiguous)
Compare the student's solution against the official solution:
- Same final answer with valid reasoning? → "correct"
- Approach correct but minor computational/notational errors? → "almost"
- Correct initial steps but incomplete or with significant errors? → "partial"
- Approach fundamentally wrong or completely missing? → "incorrect"

Step 4: PROBLEM DIFFICULTY CONTEXT
- For difficult problems, partial credit is given for meaningful progress
- For easier problems, the standard is higher for "correct" and "almost"

=== PROBLEM STATEMENT ===
{problem}

=== OFFICIAL SOLUTION ===
{solution}

=== GRADING GUIDELINES (RUBRIC) ===
{grading_guidelines}
{rubric_context}

=== STUDENT'S ANSWER ===
{student_answer}

=== ADDITIONAL SIGNALS ===
Points: {points}
Reward: {reward}

=== GRADING INSTRUCTIONS ===
Carefully analyze:
1. What the problem is asking for
2. What the official solution provides as the correct answer
3. The grading guidelines above - these contain specific rubric markers indicating what constitutes each grade level
4. The student's answer - compare it against the official solution and rubric

Pay special attention to:
- Does the student identify the correct answer/approach?
- Are the key steps of the proof/solution present?
- Are there errors in reasoning or calculations?
- How complete is the solution?
- What do the grading guidelines explicitly indicate about this solution?
- Are there partial credit markers that suggest the intended classification?
- What keywords in the rubric describe the quality of the solution?

=== REASONING PROCESS ===
Before giving your final answer, think through:
1. What is the core question/problem being asked?
2. What is the official solution's approach and answer?
3. What does the student's answer provide?
4. What EXPLICIT markers (Correct/Almost/Partial/Incorrect) appear in the rubric?
5. What credit markers (Full Credit/Half Credit/No Credit/etc.) appear in the rubric?
6. What score patterns (X/Y points) appear in the rubric?
7. What contextual keywords describe the solution quality?
8. Which category best matches the student's work based on ALL these signals?

=== RESPONSE FORMAT (STRICT - MUST FOLLOW) ===
You MUST respond ONLY in the following JSON format. Do not include any other text, explanations, or markdown outside the JSON block:

<json>
{{
    "response": "correct" | "almost" | "partial" | "incorrect"
}}
</json>

CRITICAL: The response field must contain EXACTLY one of these four lowercase words: correct, almost, partial, incorrect.
- Use "correct" only for flawless solutions
- Use "almost" for solutions with only minor/trivial errors
- Use "partial" for incomplete solutions with some correct progress
- Use "incorrect" for fundamentally wrong solutions

DO NOT include any text before or after the JSON block. Your entire response should be just the JSON.

=== EXAMPLES OF CORRECT RESPONSES ===
Example 1 (correct classification):
<json>
{{
    "response": "correct"
}}
</json>

Example 2 (almost classification):
<json>
{{
    "response": "almost"
}}
</json>

Example 3 (partial classification):
<json>
{{
    "response": "partial"
}}
</json>

Example 4 (incorrect classification):
<json>
{{
    "response": "incorrect"
}}
</json>

Remember: ONLY output the JSON block. No other text."""

        # Use retry with backoff for LLM call to handle transient failures
        def _call_llm():
            return get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
        
        try:
            response, msg_history, info = retry_with_backoff(
                _call_llm,
                max_retries=3,
                base_delay=1.0,
                exceptions=(Exception,),
            )
        except Exception as e:
            logger.error(f"LLM call failed after retries: {e}")
            # Fallback to rubric-based prediction if LLM fails
            if rubric.get("primary_category"):
                logger.info(f"LLM failed, using rubric primary category: {rubric['primary_category']}")
                return rubric["primary_category"], []
            return "unknown", []

        # Extract prediction from response with enhanced robustness
        prediction = "unknown"
        extraction_attempts = []
        
        try:
            last_message = msg_history[-1]["text"]
            self.log_fn(f"Raw response: {last_message[:500]}...")
            
            # Attempt 1: Try flexible JSON extraction first (most reliable)
            extracted = _extract_json_flexible(last_message)
            if extracted and isinstance(extracted, dict):
                extraction_attempts.append(("json_flexible", extracted))
                # Try to get the response value
                if "response" in extracted:
                    prediction = _normalize_prediction(extracted["response"])
                    extraction_attempts.append(("response_key", prediction))
                elif len(extracted) == 1:
                    # If only one key, use its value
                    prediction = _normalize_prediction(list(extracted.values())[0])
                    extraction_attempts.append(("single_key", prediction))
                else:
                    # Try to find a value that looks like a grade
                    for key, value in extracted.items():
                        normalized = _normalize_prediction(value)
                        if _is_valid_prediction(normalized):
                            prediction = normalized
                            extraction_attempts.append((f"key_{key}", prediction))
                            break
            
            # Attempt 2: If still unknown, try direct text extraction with priority ordering
            if not _is_valid_prediction(prediction):
                text_prediction = _extract_prediction_from_text(last_message)
                extraction_attempts.append(("text_extraction", text_prediction))
                if _is_valid_prediction(text_prediction):
                    prediction = text_prediction
            
            # Attempt 3: Try to extract from reasoning section if present
            if not _is_valid_prediction(prediction):
                # Look for "Therefore" or "Conclusion" sections
                conclusion_patterns = [
                    r'(?:therefore|thus|hence|conclusion|verdict|decision)[,:]?\s*["\']?(correct|almost|partial|incorrect)["\']?',
                    r'(?:final\s+)?(?:answer|classification|grade|result)[\s:]+["\']?(correct|almost|partial|incorrect)["\']?',
                ]
                for pattern in conclusion_patterns:
                    match = re.search(pattern, last_message.lower())
                    if match:
                        conclusion_pred = match.group(1)
                        extraction_attempts.append(("conclusion", conclusion_pred))
                        if _is_valid_prediction(conclusion_pred):
                            prediction = conclusion_pred
                            break
            
            # Attempt 4: Try to extract from the last line of the response
            if not _is_valid_prediction(prediction):
                lines = last_message.strip().split('\n')
                for line in reversed(lines):
                    line = line.strip()
                    if line:
                        # Try to extract from the last non-empty line
                        last_line_pred = _extract_prediction_from_text(line)
                        if _is_valid_prediction(last_line_pred):
                            prediction = last_line_pred
                            extraction_attempts.append(("last_line", prediction))
                            break
            
            # Fallback 1: Use rubric primary category if LLM response is unclear but rubric is confident
            if not _is_valid_prediction(prediction):
                if rubric.get("primary_category") and rubric.get("confidence", 0) > 0.6:
                    prediction = rubric["primary_category"]
                    self.log_fn(f"Using rubric primary category as fallback: {prediction}")
                    extraction_attempts.append(("rubric_fallback", prediction))
            
            # Fallback 2: Use score-based inference if available
            if not _is_valid_prediction(prediction) and rubric.get("score_hints", {}).get("avg_ratio"):
                ratio = rubric["score_hints"]["avg_ratio"]
                if ratio >= 0.9:
                    prediction = "correct"
                elif ratio >= 0.7:
                    prediction = "almost"
                elif ratio >= 0.4:
                    prediction = "partial"
                else:
                    prediction = "incorrect"
                self.log_fn(f"Using score-based inference as fallback: {prediction}")
                extraction_attempts.append(("score_fallback", prediction))
            
            # Fallback 3: Use marker count to determine most likely category
            if not _is_valid_prediction(prediction):
                marker_counts = rubric.get("marker_count", {})
                if marker_counts:
                    # Find the category with the most markers
                    best_category = max(marker_counts.items(), key=lambda x: x[1])
                    if best_category[1] > 0:
                        prediction = best_category[0]
                        self.log_fn(f"Using marker count fallback: {prediction}")
                        extraction_attempts.append(("marker_count_fallback", prediction))
            
            # Log all extraction attempts for debugging
            self.log_fn(f"Extraction attempts: {extraction_attempts}")
            self.log_fn(f"Final prediction: {prediction}")
            
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            # Emergency fallback to rubric if available
            if rubric.get("primary_category"):
                prediction = rubric["primary_category"]
                self.log_fn(f"Emergency fallback to rubric: {prediction}")
            else:
                prediction = "unknown"

        # Final validation - ensure we return a valid prediction or unknown
        if not _is_valid_prediction(prediction):
            prediction = "unknown"

        return str(prediction), msg_history


def _extract_prediction_from_text(text: str) -> str:
    """Extract prediction from raw text using multiple strategies.
    
    This function implements a comprehensive approach to extract the classification
    from LLM responses when JSON parsing fails. Enhanced with more robust pattern
    matching and better handling of edge cases.
    """
    if not text:
        return "unknown"
    
    text_lower = text.lower().strip()
    
    # Priority 0: Check for explicit rubric markers in the text
    # These are the strongest signals
    rubric_markers = [
        (r'\(\s*correct\s*\)', "correct"),
        (r'\(\s*almost\s*\)', "almost"),
        (r'\(\s*partial\s*\)', "partial"),
        (r'\(\s*incorrect\s*\)', "incorrect"),
        (r'\[\s*correct\s*\]', "correct"),
        (r'\[\s*almost\s*\]', "almost"),
        (r'\[\s*partial\s*\]', "partial"),
        (r'\[\s*incorrect\s*\]', "incorrect"),
        (r'\{\s*correct\s*\}', "correct"),
        (r'\{\s*almost\s*\}', "almost"),
        (r'\{\s*partial\s*\}', "partial"),
        (r'\{\s*incorrect\s*\}', "incorrect"),
        (r'\*\s*correct\s*\*', "correct"),
        (r'\*\s*almost\s*\*', "almost"),
        (r'\*\s*partial\s*\*', "partial"),
        (r'\*\s*incorrect\s*\*', "incorrect"),
        (r'\*\*correct\*\*', "correct"),
        (r'\*\*almost\*\*', "almost"),
        (r'\*\*partial\*\*', "partial"),
        (r'\*\*incorrect\*\*', "incorrect"),
        (r'__correct__', "correct"),
        (r'__almost__', "almost"),
        (r'__partial__', "partial"),
        (r'__incorrect__', "incorrect"),
        # Credit markers
        (r'\(\s*full\s*credit\s*\)', "correct"),
        (r'\(\s*no\s*credit\s*\)', "incorrect"),
        (r'\(\s*half\s*credit\s*\)', "partial"),
        (r'\(\s*most\s*credit\s*\)', "almost"),
        (r'\(\s*some\s*credit\s*\)', "partial"),
        (r'\(\s*partial\s*credit\s*\)', "partial"),
        (r'\[\s*full\s*credit\s*\]', "correct"),
        (r'\[\s*no\s*credit\s*\]', "incorrect"),
        (r'\[\s*half\s*credit\s*\]', "partial"),
        (r'\[\s*most\s*credit\s*\]', "almost"),
    ]
    for pattern, label in rubric_markers:
        if re.search(pattern, text_lower):
            return label
    
    # Priority 1: Look for exact quoted labels in JSON-like context
    json_patterns = [
        (r'"response"\s*:\s*"(correct|almost|partial|incorrect)"', 1),
        (r"'response'\s*:\s*'(correct|almost|partial|incorrect)'", 1),
        (r'"classification"\s*:\s*"(correct|almost|partial|incorrect)"', 1),
        (r'"grade"\s*:\s*"(correct|almost|partial|incorrect)"', 1),
        (r'"result"\s*:\s*"(correct|almost|partial|incorrect)"', 1),
        (r'"evaluation"\s*:\s*"(correct|almost|partial|incorrect)"', 1),
        (r'"answer"\s*:\s*"(correct|almost|partial|incorrect)"', 1),
        (r'"verdict"\s*:\s*"(correct|almost|partial|incorrect)"', 1),
        (r'"decision"\s*:\s*"(correct|almost|partial|incorrect)"', 1),
        (r'"prediction"\s*:\s*"(correct|almost|partial|incorrect)"', 1),
        (r'"label"\s*:\s*"(correct|almost|partial|incorrect)"', 1),
        (r'"category"\s*:\s*"(correct|almost|partial|incorrect)"', 1),
        (r'"(correct|almost|partial|incorrect)"', 1),
        (r"'(correct|almost|partial|incorrect)'", 1),
    ]
    for pattern, group in json_patterns:
        match = re.search(pattern, text_lower)
        if match:
            candidate = match.group(group).lower()
            if _is_valid_prediction(candidate):
                return candidate
    
    # Priority 2: Look for labels after colons or equals signs
    colon_patterns = [
        (r'response\s*[=:]\s*(correct|almost|partial|incorrect)\b', 1),
        (r'classification\s*[=:]\s*(correct|almost|partial|incorrect)\b', 1),
        (r'grade\s*[=:]\s*(correct|almost|partial|incorrect)\b', 1),
        (r'result\s*[=:]\s*(correct|almost|partial|incorrect)\b', 1),
        (r'evaluation\s*[=:]\s*(correct|almost|partial|incorrect)\b', 1),
        (r'answer\s*[=:]\s*(correct|almost|partial|incorrect)\b', 1),
        (r'category\s*[=:]\s*(correct|almost|partial|incorrect)\b', 1),
        (r'label\s*[=:]\s*(correct|almost|partial|incorrect)\b', 1),
        (r'verdict\s*[=:]\s*(correct|almost|partial|incorrect)\b', 1),
        (r'decision\s*[=:]\s*(correct|almost|partial|incorrect)\b', 1),
        (r'prediction\s*[=:]\s*(correct|almost|partial|incorrect)\b', 1),
        (r'output\s*[=:]\s*(correct|almost|partial|incorrect)\b', 1),
    ]
    for pattern, group in colon_patterns:
        match = re.search(pattern, text_lower)
        if match:
            candidate = match.group(group).lower()
            if _is_valid_prediction(candidate):
                return candidate
    
    # Priority 3: Look for labels in specific contexts (after "is", "would be", etc.)
    context_patterns = [
        (r'\bis\s+(correct|almost|partial|incorrect)\b', 1),
        (r'\bwould\s+be\s+(correct|almost|partial|incorrect)\b', 1),
        (r'\bshould\s+be\s+(correct|almost|partial|incorrect)\b', 1),
        (r'\bclassified\s+as\s+(correct|almost|partial|incorrect)\b', 1),
        (r'\bcategory\s*[:=]?\s*(correct|almost|partial|incorrect)\b', 1),
        (r'\bthe\s+(correct|almost|partial|incorrect)\s+(?:answer|classification|grade|result)\b', 1),
        (r'\bfalls?\s+(?:under|into)\s+(correct|almost|partial|incorrect)\b', 1),
        (r'\bthis\s+is\s+(correct|almost|partial|incorrect)\b', 1),
        (r'\btherefore\s+(?:it\s+is\s+)?(correct|almost|partial|incorrect)\b', 1),
        (r'\bthus\s+(?:it\s+is\s+)?(correct|almost|partial|incorrect)\b', 1),
        (r'\bhence\s+(?:it\s+is\s+)?(correct|almost|partial|incorrect)\b', 1),
        (r'\bconclusion\s*[:=]?\s*(correct|almost|partial|incorrect)\b', 1),
        (r'\bverdict\s*[:=]?\s*(correct|almost|partial|incorrect)\b', 1),
        (r'\bdecision\s*[:=]?\s*(correct|almost|partial|incorrect)\b', 1),
        (r'\bi\s+(?:would\s+)?(?:classify|grade|rate)\s+(?:this|it)\s+as\s+(correct|almost|partial|incorrect)', 1),
        (r'\bthis\s+(?:should|would)\s+be\s+(?:classified\s+as\s+)?(correct|almost|partial|incorrect)', 1),
        (r'\bthe\s+(?:student\s+)?(?:answer|solution|work)\s+is\s+(correct|almost|partial|incorrect)', 1),
        (r'\bgiven\s+(?:the\s+)?(?:classification|grade)\s+(?:of\s+)?(correct|almost|partial|incorrect)', 1),
        (r'\bassigned\s+(?:the\s+)?(?:classification|grade)\s+(?:of\s+)?(correct|almost|partial|incorrect)', 1),
    ]
    for pattern, group in context_patterns:
        match = re.search(pattern, text_lower)
        if match:
            candidate = match.group(group).lower()
            if _is_valid_prediction(candidate):
                return candidate
    
    # Priority 4: Look for labels at the end of sentences or lines (often the final answer)
    final_patterns = [
        (r'(?:final\s+)?(?:answer|classification|grade|result)[\s:]+["\']?(correct|almost|partial|incorrect)["\']?[.!?]?\s*$', 1),
        (r'(?:therefore|thus|hence)[,:]?\s+(?:the\s+)?(?:answer|classification|grade|result)\s+(?:is\s+)?["\']?(correct|almost|partial|incorrect)["\']?[.!?]?\s*$', 1),
        (r'(?:therefore|thus|hence)[,:]?\s+(?:it\s+is\s+)?["\']?(correct|almost|partial|incorrect)["\']?[.!?]?\s*$', 1),
        (r'\b["\']?(correct|almost|partial|incorrect)["\']?\s*[.!?]?\s*$', 1),  # Last word
        (r'(?:^|\n)\s*["\']?(correct|almost|partial|incorrect)["\']?\s*(?:$|\n)', 1),  # On its own line
    ]
    for pattern, group in final_patterns:
        match = re.search(pattern, text_lower, re.MULTILINE)
        if match:
            candidate = match.group(group).lower()
            if _is_valid_prediction(candidate):
                return candidate
    
    # Priority 5: Check for standalone words at word boundaries
    # Check in order of specificity (more specific first to avoid substring issues)
    if re.search(r'\balmost\b', text_lower):
        return "almost"
    if re.search(r'\bpartial\b', text_lower):
        return "partial"
    if re.search(r'\bincorrect\b', text_lower):
        return "incorrect"
    if re.search(r'\bwrong\b', text_lower):
        return "incorrect"
    
    # For "correct", need to be careful about "incorrect"
    for match in re.finditer(r'\bcorrect\b', text_lower):
        start = max(0, match.start() - 20)
        context = text_lower[start:match.start()]
        # Check if preceded by "not", "in", or other negating words
        negating_patterns = ['not ', 'in', "isn't", 'isnt', 'not,', 'not.', 'not:', 
                            'not;', 'not)', 'not]', 'not}', 'not\t', 'not\n', 'not']
        if not any(neg in context for neg in negating_patterns):
            return "correct"
    
    # Priority 6: Find all occurrences and use smart disambiguation with enhanced scoring
    positions = []
    for label in ["correct", "almost", "partial", "incorrect"]:
        for match in re.finditer(r'\b' + label + r'\b', text_lower):
            # Calculate a score based on context
            score = 0
            start = max(0, match.start() - 50)
            end = min(len(text_lower), match.end() + 50)
            context = text_lower[start:end]
            
            # Boost score for labels near keywords indicating final answer
            final_keywords = ['final', 'answer', 'classification', 'grade', 'result', 
                            'conclusion', 'verdict', 'decision', 'therefore', 'thus', 'hence',
                            'response', 'evaluation', 'prediction', 'output']
            if any(kw in context for kw in final_keywords):
                score += 20
            
            # Boost score for labels near the end of the text (often the final answer)
            if match.end() > len(text_lower) * 0.8:
                score += 15
            
            # Boost score for labels in JSON-like context
            if '"' in context or "'" in context or ':' in context or '{' in context:
                score += 10
            
            # Penalize "correct" if near negating words
            if label == "correct":
                before = text_lower[max(0, match.start() - 25):match.start()]
                negating_words = ['not ', 'in', "isn't", 'isnt', 'not,', 'not.', 'not:', 
                                'not;', 'not)', 'not]', 'not}', 'not\t', 'not\n', 'not']
                if any(neg in before for neg in negating_words):
                    score -= 40
            
            # Boost score for "almost" and "partial" (more specific than "correct"/"incorrect")
            if label in ["almost", "partial"]:
                score += 8
            
            positions.append((match.start(), label, score))
    
    if positions:
        # Sort by score (descending), then by position (prefer later occurrences)
        positions.sort(key=lambda x: (-x[2], -x[0]))
        
        # Filter out negated "correct" instances
        filtered = []
        for pos, label, score in positions:
            if label == "correct":
                before = text_lower[max(0, pos - 25):pos]
                negating_words = ['not ', 'in', "isn't", 'isnt', 'not,', 'not.', 'not:',
                                'not;', 'not)', 'not]', 'not}', 'not\t', 'not\n', 'not']
                if any(neg in before for neg in negating_words):
                    continue
            filtered.append((pos, label, score))
        
        if filtered:
            # Return the highest scoring label
            best = filtered[0]
            return best[1]
        else:
            # Fallback to first non-negated
            return positions[0][1]
    
    return "unknown"
