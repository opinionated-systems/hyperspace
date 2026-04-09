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


def _fix_json_string(json_str: str) -> str:
    """Fix common JSON formatting issues."""
    # Remove trailing commas before closing braces/brackets
    json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
    
    # Fix single quotes to double quotes (carefully)
    # This is a simplified approach - handle common cases
    json_str = re.sub(r"(?<!\\)'", '"', json_str)
    
    # Fix unquoted keys (simple cases)
    json_str = re.sub(r'(\{|,\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', json_str)
    
    # Remove control characters
    json_str = re.sub(r'[\x00-\x1F\x7F]', '', json_str)
    
    # Fix escaped characters that might cause issues
    json_str = json_str.replace('\\"', '"').replace('\\n', '\n').replace('\\t', '\t')
    
    # Remove any BOM or zero-width characters
    json_str = json_str.replace('\ufeff', '').replace('\u200b', '').replace('\u200c', '').replace('\u200d', '')
    
    return json_str


def _extract_key_value_patterns(text: str) -> dict | None:
    """Extract key-value pairs from text using regex patterns with enhanced detection."""
    result = {}
    text_lower = text.lower()
    
    # Pattern 1: "key": "value" or 'key': 'value' - standard JSON
    pattern1 = r'["\'](\w+)["\']\s*:\s*["\']([^"\']+)["\']'
    matches = re.findall(pattern1, text)
    for key, value in matches:
        result[key.lower()] = value.lower()
    
    # Pattern 2: "key": value (unquoted) - including numbers and booleans
    pattern2 = r'["\'](\w+)["\']\s*:\s*([^,\}\]\n]+)'
    matches = re.findall(pattern2, text)
    for key, value in matches:
        key_lower = key.lower()
        if key_lower not in result:
            result[key_lower] = value.strip().strip('"\'').lower()
    
    # Pattern 3: key: "value" (unquoted key)
    pattern3 = r'(\w+)\s*:\s*["\']([^"\']+)["\']'
    matches = re.findall(pattern3, text)
    for key, value in matches:
        key_lower = key.lower()
        if key_lower not in result:
            result[key_lower] = value.lower()
    
    # Pattern 4: key: value (both unquoted, single word values)
    pattern4 = r'(?:^|\n)\s*(\w+)\s*:\s*(\w+)\s*(?:$|\n)'
    matches = re.findall(pattern4, text)
    for key, value in matches:
        key_lower = key.lower()
        if key_lower not in result and value in ["correct", "almost", "partial", "incorrect"]:
            result[key_lower] = value
    
    # Pattern 5: Look for response/classification/grade at start of lines
    pattern5 = r'(?:^|\n)\s*(response|classification|grade|result|evaluation|prediction|label|category|answer|output)\s*[=:]\s*["\']?(correct|almost|partial|incorrect)["\']?\b'
    matches = re.findall(pattern5, text, re.IGNORECASE)
    for key, value in matches:
        key_lower = key.lower()
        if key_lower not in result:
            result[key_lower] = value.lower()
    
    # Pattern 6: Look for key=value patterns (common in LLM outputs)
    pattern6 = r'\b(response|classification|grade|result|evaluation|prediction|label|category|answer|output)\s*=\s*["\']?(correct|almost|partial|incorrect)["\']?\b'
    matches = re.findall(pattern6, text, re.IGNORECASE)
    for key, value in matches:
        key_lower = key.lower()
        if key_lower not in result:
            result[key_lower] = value.lower()
    
    # Pattern 7: Look for standalone classification words at line start
    pattern7 = r'(?:^|\n)\s*(correct|almost|partial|incorrect)\s*(?:$|\n)'
    matches = re.findall(pattern7, text_lower)
    if matches and 'response' not in result:
        result['response'] = matches[0]
    
    # Pattern 8: Look for "The answer is X" or "This is X" patterns
    pattern8 = r'(?:the\s+(?:answer|result|grade|classification)|this)\s+is\s+["\']?(correct|almost|partial|incorrect)["\']?\b'
    matches = re.findall(pattern8, text_lower)
    if matches and 'response' not in result:
        result['response'] = matches[0]
    
    # Pattern 9: Look for "classified as X" or "graded as X" patterns
    pattern9 = r'(?:classif(?:ied|y)|grade[ds]?)\s+(?:as\s+)?["\']?(correct|almost|partial|incorrect)["\']?\b'
    matches = re.findall(pattern9, text_lower)
    if matches and 'response' not in result:
        result['response'] = matches[0]
    
    return result if result else None


def _extract_json_flexible(text: str) -> dict | None:
    """Extract JSON from text using multiple robust strategies with enhanced parsing.
    
    This function tries multiple approaches to extract JSON data:
    1. Standard <json>...</json> blocks
    2. ```json code blocks
    3. ``` code blocks (any language)
    4. Curly brace matching for inline JSON
    5. Key-value pattern extraction as fallback
    6. YAML-like key: value patterns
    7. Inline JSON with escaped characters
    """
    if not text:
        return None
    
    text = text.strip()
    
    # Strategy 1: Try standard <json>...</json> blocks
    try:
        json_blocks = _extract_jsons(text)
        if json_blocks:
            for block in json_blocks:
                if isinstance(block, dict):
                    return block
    except Exception:
        pass
    
    # Strategy 2: Try ```json code blocks with enhanced parsing
    try:
        json_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
        matches = re.findall(json_pattern, text, re.DOTALL | re.IGNORECASE)
        for match in matches:
            if match.strip():
                # Try multiple parsing attempts with increasing fixes
                attempts = [
                    match.strip(),  # Original
                    match.strip().replace('\n', ' ').replace('\t', ' '),  # Normalize whitespace
                    re.sub(r'\s+', ' ', match.strip()),  # Collapse all whitespace
                ]
                for attempt in attempts:
                    try:
                        data = json.loads(attempt)
                        if isinstance(data, dict):
                            return data
                    except json.JSONDecodeError:
                        # Try fixing common issues
                        try:
                            fixed = _fix_json_string(attempt)
                            data = json.loads(fixed)
                            if isinstance(data, dict):
                                return data
                        except json.JSONDecodeError:
                            continue
    except Exception:
        pass
    
    # Strategy 3: Try to find JSON objects by matching curly braces with improved tracking
    try:
        # Find all potential JSON objects by tracking brace depth
        potential_jsons = []
        start_indices = []
        in_string = False
        escape_next = False
        
        for i, char in enumerate(text):
            if escape_next:
                escape_next = False
                continue
            if char == '\\':
                escape_next = True
                continue
            if char == '"' and (i == 0 or text[i-1] != '\\'):
                in_string = not in_string
                continue
            if not in_string:
                if char == '{':
                    start_indices.append(i)
                elif char == '}' and start_indices:
                    start = start_indices.pop()
                    if not start_indices:  # Only consider outermost complete objects
                        potential_jsons.append(text[start:i+1])
        
        # Also get nested objects
        for i in range(len(start_indices)):
            start = start_indices[i]
            # Find matching end
            depth = 0
            for j in range(start, len(text)):
                if text[j] == '{':
                    depth += 1
                elif text[j] == '}':
                    depth -= 1
                    if depth == 0:
                        potential_jsons.append(text[start:j+1])
                        break
        
        # Try parsing each potential JSON from longest to shortest
        for json_str in sorted(potential_jsons, key=len, reverse=True):
            if len(json_str) < 10:  # Skip very short strings
                continue
            # Multiple parsing attempts
            attempts = [
                json_str,
                json_str.replace('\n', ' ').replace('\t', ' '),
                re.sub(r'\s+', ' ', json_str),
            ]
            for attempt in attempts:
                try:
                    data = json.loads(attempt)
                    if isinstance(data, dict):
                        return data
                except json.JSONDecodeError:
                    # Try fixing common issues
                    try:
                        fixed = _fix_json_string(attempt)
                        data = json.loads(fixed)
                        if isinstance(data, dict):
                            return data
                    except json.JSONDecodeError:
                        continue
    except Exception:
        pass
    
    # Strategy 4: Try to find JSON-like structures with specific keys (expanded patterns)
    try:
        # Look for patterns like {"response": "..."} or {"classification": "..."}
        key_patterns = [
            r'\{\s*"response"\s*:\s*"[^"]*"\s*\}',
            r'\{\s*"classification"\s*:\s*"[^"]*"\s*\}',
            r'\{\s*"grade"\s*:\s*"[^"]*"\s*\}',
            r'\{\s*"result"\s*:\s*"[^"]*"\s*\}',
            r'\{\s*"evaluation"\s*:\s*"[^"]*"\s*\}',
            r'\{\s*"prediction"\s*:\s*"[^"]*"\s*\}',
            r'\{\s*"label"\s*:\s*"[^"]*"\s*\}',
            r'\{\s*"category"\s*:\s*"[^"]*"\s*\}',
            r'\{\s*"answer"\s*:\s*"[^"]*"\s*\}',
            r"\{\s*'response'\s*:\s*'[^']*'\s*\}",
            r"\{\s*'classification'\s*:\s*'[^']*'\s*\}",
            r"\{\s*'grade'\s*:\s*'[^']*'\s*\}",
            r"\{\s*'result'\s*:\s*'[^']*'\s*\}",
            # Unquoted value patterns
            r'\{\s*"response"\s*:\s*(correct|almost|partial|incorrect)\s*\}',
            r'\{\s*"classification"\s*:\s*(correct|almost|partial|incorrect)\s*\}',
            r'\{\s*"grade"\s*:\s*(correct|almost|partial|incorrect)\s*\}',
            r'\{\s*"result"\s*:\s*(correct|almost|partial|incorrect)\s*\}',
        ]
        
        for pattern in key_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    data = json.loads(match)
                    if isinstance(data, dict):
                        return data
                except json.JSONDecodeError:
                    try:
                        fixed = _fix_json_string(match)
                        data = json.loads(fixed)
                        if isinstance(data, dict):
                            return data
                    except json.JSONDecodeError:
                        continue
    except Exception:
        pass
    
    # Strategy 5: Extract key-value patterns as fallback
    try:
        kv_result = _extract_key_value_patterns(text)
        if kv_result:
            return kv_result
    except Exception:
        pass
    
    # Strategy 6: Look for simple "key": "value" patterns anywhere in text (expanded)
    try:
        simple_patterns = [
            r'"(\w+)"\s*:\s*"(correct|almost|partial|incorrect)"',
            r"'(\w+)'\s*:\s*'(correct|almost|partial|incorrect)'",
            r'"(\w+)"\s*:\s*(correct|almost|partial|incorrect)\b',
            r"'(\w+)'\s*:\s*(correct|almost|partial|incorrect)\b",
        ]
        for pattern in simple_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                return {matches[0][0].lower(): matches[0][1].lower()}
    except Exception:
        pass
    
    # Strategy 7: Look for Python dict-like patterns
    try:
        python_dict_pattern = r'\{\s*[\'"](\w+)[\'"]\s*:\s*[\'"]([^\'"]+)[\'"]\s*\}'
        matches = re.findall(python_dict_pattern, text)
        for key, value in matches:
            if value.lower() in ["correct", "almost", "partial", "incorrect"]:
                return {key.lower(): value.lower()}
    except Exception:
        pass
    
    # Strategy 8: Look for assignment patterns like response = "correct"
    try:
        assignment_patterns = [
            r'\b(response|classification|grade|result|evaluation|prediction|label|category|answer)\s*=\s*["\'](correct|almost|partial|incorrect)["\']',
            r'\b(response|classification|grade|result|evaluation|prediction|label|category|answer)\s*:\s*["\'](correct|almost|partial|incorrect)["\']',
        ]
        for pattern in assignment_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return {match.group(1).lower(): match.group(2).lower()}
    except Exception:
        pass
    
    return None


def _normalize_prediction(raw_value: Any) -> str:
    """Normalize various prediction formats to standard labels with enhanced edge case handling."""
    if raw_value is None:
        return "unknown"
    
    # Handle numeric values (0-1 scale)
    if isinstance(raw_value, (int, float)):
        if raw_value >= 0.95:
            return "correct"
        elif raw_value >= 0.70:
            return "almost"
        elif raw_value >= 0.35:
            return "partial"
        else:
            return "incorrect"
    
    raw_str = str(raw_value).lower().strip().strip('"\'')
    
    # Direct matches (exact) - check first for exact matches
    if raw_str in ["correct", "incorrect", "partial", "almost"]:
        return raw_str
    
    # Check for compound phrases first (before simple substring matching)
    # These are the most specific patterns and should take priority
    almost_compounds = [
        "almost correct", "nearly correct", "mostly correct", "essentially correct",
        "practically correct", "virtually correct", "very close to correct",
        "almost complete", "nearly complete", "almost there", "nearly there",
        "almost solved", "nearly solved", "almost answered", "nearly answered",
        "almost right", "nearly right", "mostly right", "almost valid", "nearly valid",
    ]
    for term in almost_compounds:
        if term in raw_str:
            return "almost"
    
    partial_compounds = [
        "partially correct", "partial credit", "partial solution", "partially solved",
        "partially answered", "partially valid", "partial work", "partial answer",
        "partial success", "partially successful", "partial understanding",
        "half correct", "half credit", "half marks", "half points",
        "some credit", "some marks", "some points", "some correct",
        "some progress", "some valid", "some work", "some success",
        "incomplete solution", "incomplete proof", "incomplete work",
        "incomplete answer", "incomplete reasoning", "incomplete argument",
    ]
    for term in partial_compounds:
        if term in raw_str:
            return "partial"
    
    incorrect_compounds = [
        "not correct", "not right", "not valid", "not true",
        "isn't correct", "isnt correct", "isn't right", "isnt right",
        "isn't valid", "isnt valid", "doesn't work", "doesnt work",
        "didn't work", "didnt work", "not working", "no good",
        "fundamentally wrong", "completely wrong", "totally wrong",
        "entirely wrong", "wholly wrong", "absolutely wrong",
        "fundamentally incorrect", "completely incorrect", "totally incorrect",
        "entirely incorrect", "wholly incorrect", "absolutely incorrect",
        "incorrect approach", "incorrect solution", "incorrect answer",
        "incorrect reasoning", "incorrect logic", "incorrect method",
        "wrong approach", "wrong solution", "wrong answer",
        "wrong reasoning", "wrong logic", "wrong method",
        "invalid solution", "invalid answer", "invalid approach",
        "invalid reasoning", "invalid logic", "invalid method",
        "no solution", "no answer", "no progress", "no credit",
        "no marks", "no points", "zero credit", "zero marks", "zero points",
    ]
    for term in incorrect_compounds:
        if term in raw_str:
            return "incorrect"
    
    correct_compounds = [
        "fully correct", "completely correct", "entirely correct", "totally correct",
        "wholly correct", "absolutely correct", "definitely correct",
        "clearly correct", "obviously correct", "undoubtedly correct",
        "correct solution", "correct answer", "correct proof",
        "correct reasoning", "correct logic", "correct method",
        "correct approach", "correct work", "correct throughout",
        "valid solution", "valid answer", "valid proof",
        "valid reasoning", "valid logic", "valid method",
        "valid approach", "valid work", "valid throughout",
        "sound solution", "sound answer", "sound proof",
        "sound reasoning", "sound logic", "sound method",
        "complete solution", "complete answer", "complete proof",
        "perfect solution", "perfect answer", "flawless solution",
        "all correct", "all valid", "all sound",
        "no errors", "no mistakes", "no issues", "no problems",
        "100% correct", "100 percent correct", "hundred percent correct",
    ]
    for term in correct_compounds:
        if term in raw_str:
            return "correct"
    
    # Handle common variations for "correct" - comprehensive list
    correct_indicators = [
        "true", "yes", "right", "full credit", "full marks",
        "complete solution", "correct solution", "perfect", "flawless",
        "100% correct", "fully correct", "entirely correct", "totally correct",
        "no errors", "all steps correct", "mathematically correct",
        "full score", "max score", "maximum points", "award full",
        "excellent", "outstanding", "complete and correct", "100%",
        "perfect solution", "flawless solution", "all correct",
        "completely correct", "totally correct", "entirely correct",
        "valid solution", "valid answer", "correct answer", "correct final answer",
        "fully valid", "entirely valid", "completely valid", "totally valid",
        "all valid", "no issues", "no problems", "no mistakes",
        "correct throughout", "correctly solved", "correctly answered",
        "correct work", "correct reasoning", "correct logic",
        "sound solution", "sound proof", "sound argument",
        "rigorous solution", "rigorous proof", "complete proof",
        "valid proof", "valid reasoning", "valid logic",
        "correct and complete", "complete and valid", "sound and complete",
        "successful solution", "successful proof", "successful attempt",
        "complete proof structure", "complete argument",
        # Additional edge cases
        "100 percent", "hundred percent", "full points", "max points",
        "all correct", "totally right", "fully right", "entirely right",
        "completely right", "absolutely correct", "definitely correct",
        "undoubtedly correct", "clearly correct", "obviously correct",
        "full marks", "maximum marks", "all marks",
        "success", "successful", "passed", "pass",
    ]
    
    for term in correct_indicators:
        if term in raw_str:
            return "correct"
    
    # Check for "valid" but not "invalid" (with more context)
    if "valid" in raw_str and "invalid" not in raw_str:
        # Additional check to ensure it's not negated
        negation_patterns = ["not valid", "isn't valid", "isnt valid", "never valid", 
                            "not entirely valid", "not fully valid", "not valid at all"]
        if not any(neg in raw_str for neg in negation_patterns):
            return "correct"
    
    # Handle common variations for "incorrect" - comprehensive list
    incorrect_indicators = [
        "false", "no credit", "zero credit",
        "no marks", "zero marks", "fundamentally wrong", "completely wrong",
        "totally wrong", "entirely wrong", "no solution", "no progress",
        "no meaningful work", "blank", "fail", "failed", "rejected",
        "award no", "give no", "not correct", "not right", "not valid",
        "incorrect approach", "wrong approach", "major error", "critical error",
        "no valid", "invalid solution", "wrong solution", "incorrect solution",
        "isn't correct", "isnt correct", "never correct", "not correct at all",
        "invalid answer", "wrong answer", "incorrect answer",
        "fundamentally incorrect", "completely incorrect", "totally incorrect",
        "entirely incorrect", "wholly incorrect", "absolutely wrong",
        "completely invalid", "totally invalid", "entirely invalid",
        "not a solution", "not valid work", "not correct work",
        "failed solution", "failed attempt", "unsuccessful",
        "does not solve", "doesn't solve", "did not solve",
        "cannot solve", "can't solve", "unable to solve",
        "no attempt", "no work shown", "no working",
        "irrelevant", "off topic", "unrelated",
        "nonsense", "gibberish", "garbage",
        "zero", "0", "0/", "/0", "0 points", "0 marks",
        "unsuccessful attempt", "failed solution", "failed try",
        "does not understand", "did not understand", "doesn't understand",
        "did not grasp", "doesn't grasp", "failed to demonstrate",
        # Additional edge cases
        "zero percent", "0 percent", "0%", "no points", "zero points",
        "absolutely wrong", "definitely wrong", "clearly wrong", "obviously wrong",
        "not even close", "way off", "completely off", "totally off",
        "missed the point", "missed completely", "completely missed",
        "fundamental misunderstanding", "basic error", "critical mistake",
        "fatal error", "fatal flaw", "serious error", "serious mistake",
        "failure", "failing", "rejected", "denied",
        "none", "nil", "null", "undefined", "empty",
    ]
    
    for term in incorrect_indicators:
        if term in raw_str:
            return "incorrect"
    
    # Handle "wrong" and "invalid" separately to avoid matching "correct"
    if raw_str in ["wrong", "invalid", "no", "not", "none", "nil", "null", "false", "0", "zero"]:
        return "incorrect"
    
    # Handle common variations for "partial" - comprehensive list
    partial_indicators = [
        "partial credit", "half credit", "some credit", "incomplete",
        "partial solution", "partially correct", "half correct",
        "some progress", "meaningful progress", "on the right track",
        "good start", "correct approach", "correct idea", "started correctly",
        "50% correct", "60% correct", "40% correct", "award half",
        "award some", "give half", "give some", "incomplete solution",
        "missing steps", "significant gaps", "partially right",
        "partially valid", "incomplete proof", "missing conclusion",
        "incomplete work", "partial answer", "some correct", "part correct",
        "partial marks", "half marks", "some marks",
        "partially solved", "partially answered", "incomplete answer",
        "started well", "good beginning", "correct start",
        "correct method", "correct strategy", "valid approach",
        "valid start", "valid beginning", "on track",
        "heading in right direction", "making progress",
        "some valid steps", "some correct steps", "some valid work",
        "partial work", "partial working", "partial reasoning",
        "incomplete reasoning", "incomplete argument",
        "missing parts", "missing components", "missing elements",
        "not complete", "not fully complete", "not entirely complete",
        "not totally complete", "not wholly complete",
        "unfinished", "not finished", "incomplete work",
        "needs more", "needs completion", "needs finishing",
        "lacks conclusion", "lacks final step", "lacks final answer",
        "missing final step", "missing final answer", "missing conclusion",
        "partial success", "partially successful", "some success",
        "halfway", "half way", "half-way", "half done", "half complete",
        "50%", "40%", "60%", "30%", "70%", "35-69%",
        "partial understanding", "some understanding",
        "made progress", "showing progress", "demonstrated some",
        "in progress", "ongoing", "started", "begun",
        "partial result", "intermediate result", "intermediate step",
        "not done", "not finished", "not complete", "incomplete",
    ]
    
    for term in partial_indicators:
        if term in raw_str:
            return "partial"
    
    # Handle common variations for "almost" - comprehensive list
    almost_indicators = [
        "almost correct", "nearly correct", "mostly correct", "minor error",
        "small error", "trivial error", "typo", "minor mistake", "almost there",
        "very close", "95% correct", "90% correct", "most credit",
        "most marks", "award most", "give most", "minor omission",
        "notational error", "sign error", "arithmetic error", "slip",
        "careless error", "cosmetic error", "slight error", "nearly there",
        "almost complete", "nearly complete", "very close to correct",
        "just a minor", "only a minor", "minor issue", "small mistake",
        "trivial mistake", "calculation error", "computation error",
        "nearly", "almost", "mostly right", "mostly valid",
        "nearly solved", "nearly answered", "almost solved", "almost answered",
        "essentially correct", "practically correct", "virtually correct",
        "correct except", "correct apart from", "correct aside from",
        "minor slip", "minor oversight", "minor gap",
        "small slip", "small oversight", "small gap",
        "trivial slip", "trivial oversight", "trivial gap",
        "cosmetic issue", "cosmetic problem", "cosmetic mistake",
        "notational issue", "notational problem", "notation error",
        "sign mistake", "sign issue", "arithmetic mistake",
        "calculation mistake", "computation mistake", "computational error",
        "rounding error", "rounding mistake", "rounding issue",
        "transcription error", "transcription mistake",
        "copying error", "copying mistake",
        "90%", "95%", "96%", "97%", "98%", "99%", "94%",
        "70-94%", "70%", "75%", "80%", "85%", "89%",
        "small correction needed", "minor fix needed", "minor adjustment needed",
        "would be correct if", "just needs a minor", "only needs a minor",
        "nearly perfect", "almost perfect", "mostly perfect",
        "close", "very close", "so close", "getting close",
        "minor blemish", "small blemish", "tiny error", "little error",
    ]
    
    for term in almost_indicators:
        if term in raw_str:
            return "almost"
    
    # Check for "incorrect" before "correct" to avoid substring issues
    if "incorrect" in raw_str:
        return "incorrect"
    
    if "wrong" in raw_str or "invalid" in raw_str:
        return "incorrect"
    
    # Check for "correct" (but not negated) - with more careful negation detection
    if "correct" in raw_str:
        # Check for negation patterns
        idx = raw_str.find("correct")
        before = raw_str[max(0, idx-30):idx]
        negation_patterns = ['not ', 'in', "isn't", 'isnt', 'never', 'hardly', 'barely', 
                            'not entirely', 'not fully', 'not completely', 
                            'not totally', 'not wholly', 'not absolutely',
                            'not quite', 'not really', 'not actually']
        if not any(neg in before for neg in negation_patterns):
            return "correct"
    
    if "partial" in raw_str:
        return "partial"
    
    if "almost" in raw_str or "nearly" in raw_str:
        return "almost"
    
    return "unknown"


def _is_valid_prediction(prediction: str) -> bool:
    """Check if a prediction is one of the valid labels."""
    return prediction in ["correct", "incorrect", "partial", "almost"]


def _extract_prediction_from_text(text: str) -> str:
    """Extract prediction from raw text using multiple strategies with enhanced pattern matching."""
    if not text:
        return "unknown"
    
    text_lower = text.lower().strip()
    
    # Pre-process: Remove common prefixes and markers
    text_lower = re.sub(r'^(?:here is|here\'s|the|this is|my|final|output|result|answer|classification|grade|response|evaluation)[:\s]*', '', text_lower)
    text_lower = re.sub(r'^(?:json|the json)[:\s]*', '', text_lower)
    text_lower = re.sub(r'^\s*```\s*\n?', '', text_lower)
    text_lower = re.sub(r'\n?```\s*$', '', text_lower)
    
    # Priority 0: Check for explicit rubric markers in the text (highest priority)
    # These patterns are ordered by specificity - most specific first
    rubric_markers = [
        # Exact markers with parentheses - highest confidence
        (r'\(\s*correct\s*\)', "correct"),
        (r'\(\s*almost\s*\)', "almost"),
        (r'\(\s*partial\s*\)', "partial"),
        (r'\(\s*incorrect\s*\)', "incorrect"),
        # Bracket markers
        (r'\[\s*correct\s*\]', "correct"),
        (r'\[\s*almost\s*\]', "almost"),
        (r'\[\s*partial\s*\]', "partial"),
        (r'\[\s*incorrect\s*\]', "incorrect"),
        # Curly brace markers
        (r'\{\s*correct\s*\}', "correct"),
        (r'\{\s*almost\s*\}', "almost"),
        (r'\{\s*partial\s*\}', "partial"),
        (r'\{\s*incorrect\s*\}', "incorrect"),
        # Angle bracket markers
        (r'<\s*correct\s*>', "correct"),
        (r'<\s*almost\s*>', "almost"),
        (r'<\s*partial\s*>', "partial"),
        (r'<\s*incorrect\s*>', "incorrect"),
        # Double parentheses/brackets
        (r'\(\(\s*correct\s*\)\)', "correct"),
        (r'\(\(\s*almost\s*\)\)', "almost"),
        (r'\(\(\s*partial\s*\)\)', "partial"),
        (r'\(\(\s*incorrect\s*\)\)', "incorrect"),
        (r'\[\[\s*correct\s*\]\]', "correct"),
        (r'\[\[\s*almost\s*\]\]', "almost"),
        (r'\[\[\s*partial\s*\]\]', "partial"),
        (r'\[\[\s*incorrect\s*\]\]', "incorrect"),
        # Credit markers with parentheses
        (r'\(\s*full\s*credit\s*\)', "correct"),
        (r'\[\s*full\s*credit\s*\]', "correct"),
        (r'\{\s*full\s*credit\s*\}', "correct"),
        (r'<\s*full\s*credit\s*>', "correct"),
        (r'\(\s*no\s*credit\s*\)', "incorrect"),
        (r'\[\s*no\s*credit\s*\]', "incorrect"),
        (r'\{\s*no\s*credit\s*\}', "incorrect"),
        (r'<\s*no\s*credit\s*>', "incorrect"),
        (r'\(\s*half\s*credit\s*\)', "partial"),
        (r'\[\s*half\s*credit\s*\]', "partial"),
        (r'\{\s*half\s*credit\s*\}', "partial"),
        (r'<\s*half\s*credit\s*>', "partial"),
        (r'\(\s*most\s*credit\s*\)', "almost"),
        (r'\[\s*most\s*credit\s*\]', "almost"),
        (r'\{\s*most\s*credit\s*\}', "almost"),
        (r'<\s*most\s*credit\s*>', "almost"),
        (r'\(\s*some\s*credit\s*\)', "partial"),
        (r'\[\s*some\s*credit\s*\]', "partial"),
        (r'\{\s*some\s*credit\s*\}', "partial"),
        (r'<\s*some\s*credit\s*>', "partial"),
        (r'\(\s*partial\s*credit\s*\)', "partial"),
        (r'\[\s*partial\s*credit\s*\]', "partial"),
        (r'\{\s*partial\s*credit\s*\}', "partial"),
        (r'<\s*partial\s*credit\s*>', "partial"),
        # Award patterns with word boundaries
        (r'\baward\s+full\s+(?:credit|marks?|points?)\b', "correct"),
        (r'\baward\s+no\s+(?:credit|marks?|points?)\b', "incorrect"),
        (r'\baward\s+half\s+(?:credit|marks?|points?)\b', "partial"),
        (r'\baward\s+most\s+(?:credit|marks?|points?)\b', "almost"),
        (r'\baward\s+some\s+(?:credit|marks?|points?)\b', "partial"),
        # Give patterns with word boundaries
        (r'\bgive\s+full\s+(?:credit|marks?|points?)\b', "correct"),
        (r'\bgive\s+no\s+(?:credit|marks?|points?)\b', "incorrect"),
        (r'\bgive\s+half\s+(?:credit|marks?|points?)\b', "partial"),
        (r'\bgive\s+most\s+(?:credit|marks?|points?)\b', "almost"),
        (r'\bgive\s+some\s+(?:credit|marks?|points?)\b', "partial"),
        # Full/No/Half/Most/Some credit with word boundaries
        (r'\bfull\s+(?:credit|marks?|points?)\b', "correct"),
        (r'\bno\s+(?:credit|marks?|points?)\b', "incorrect"),
        (r'\bhalf\s+(?:credit|marks?|points?)\b', "partial"),
        (r'\bmost\s+(?:credit|marks?|points?)\b', "almost"),
        (r'\bsome\s+(?:credit|marks?|points?)\b', "partial"),
        # Additional patterns for edge cases
        (r'\b100%\s*(?:correct|right|valid)?\b', "correct"),
        (r'\b(?:zero|0)\s+(?:credit|marks?|points?)\b', "incorrect"),
        (r'\b(?:zero|0)\s*/\s*\d+\b', "incorrect"),
        # Colon/arrow patterns
        (r':\s*correct\s*$', "correct"),
        (r':\s*almost\s*$', "almost"),
        (r':\s*partial\s*$', "partial"),
        (r':\s*incorrect\s*$', "incorrect"),
        (r'->\s*correct\b', "correct"),
        (r'->\s*almost\b', "almost"),
        (r'->\s*partial\b', "partial"),
        (r'->\s*incorrect\b', "incorrect"),
        # Category/Grade/Classification patterns
        (r'\bcategory\s*[:=]\s*correct\b', "correct"),
        (r'\bcategory\s*[:=]\s*almost\b', "almost"),
        (r'\bcategory\s*[:=]\s*partial\b', "partial"),
        (r'\bcategory\s*[:=]\s*incorrect\b', "incorrect"),
        (r'\bgrade\s*[:=]\s*correct\b', "correct"),
        (r'\bgrade\s*[:=]\s*almost\b', "almost"),
        (r'\bgrade\s*[:=]\s*partial\b', "partial"),
        (r'\bgrade\s*[:=]\s*incorrect\b', "incorrect"),
        (r'\bclassification\s*[:=]\s*correct\b', "correct"),
        (r'\bclassification\s*[:=]\s*almost\b', "almost"),
        (r'\bclassification\s*[:=]\s*partial\b', "partial"),
        (r'\bclassification\s*[:=]\s*incorrect\b', "incorrect"),
    ]
    
    for pattern, label in rubric_markers:
        if label and re.search(pattern, text_lower):
            return label
    
    # Handle score patterns like (3/4 points) or (7/10 points) with enhanced patterns
    score_patterns = [
        # Parentheses
        r'\(\s*(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)\s*(?:points?|pts?|marks?)?\s*\)',
        # Brackets
        r'\[\s*(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)\s*(?:points?|pts?|marks?)?\s*\]',
        # Curly braces
        r'\{\s*(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)\s*(?:points?|pts?|marks?)?\s*\}',
        # Angle brackets
        r'<\s*(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)\s*(?:points?|pts?|marks?)?\s*>',
        # Word boundaries with points/marks
        r'\b(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)\s*(?:points?|pts?|marks?)\b',
        # Award patterns
        r'\baward\s+(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)\b',
        r'\baward\s+(\d+(?:\.\d+)?)\s+out\s+of\s+(\d+(?:\.\d+)?)\b',
        r'\baward\s+(\d+(?:\.\d+)?)\s+points?\b',
        # Out of patterns
        r'\b(\d+(?:\.\d+)?)\s+out\s+of\s*(\d+(?:\.\d+)?)\s*(?:points?|pts?|marks?)\b',
        r'\b(\d+(?:\.\d+)?)\s+out\s+of\s*(\d+(?:\.\d+)?)\b',
        # Score patterns
        r'\bscore[d]?\s+(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)\b',
        r'\bscore[d]?\s+(\d+(?:\.\d+)?)\s+out\s+of\s+(\d+(?:\.\d+)?)\b',
        # End of line patterns
        r'\b(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)\s*$',
        r'\b(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)\s*(?:points?|pts?|marks?)?\s*(?:\n|$)',
        # Percentage patterns
        r'\b(\d+(?:\.\d+)?)%\s*(?:score|grade|correct)?\b',
        # Percentage with parentheses
        r'\(\s*(\d+(?:\.\d+)?)%\s*\)',
        r'\[\s*(\d+(?:\.\d+)?)%\s*\]',
    ]
    
    for score_pattern in score_patterns:
        score_match = re.search(score_pattern, text_lower)
        if score_match:
            try:
                if len(score_match.groups()) == 1:
                    # Percentage pattern
                    ratio = float(score_match.group(1)) / 100.0
                else:
                    earned = float(score_match.group(1))
                    total = float(score_match.group(2))
                    if total > 0:
                        ratio = earned / total
                    else:
                        continue
                
                if ratio >= 0.95:
                    return "correct"
                elif ratio >= 0.70:
                    return "almost"
                elif ratio >= 0.35:
                    return "partial"
                else:
                    return "incorrect"
            except (ValueError, ZeroDivisionError):
                continue
    
    # Priority 1: Look for exact quoted labels in JSON-like context
    json_patterns = [
        r'"response"\s*:\s*"(correct|almost|partial|incorrect)"',
        r'"classification"\s*:\s*"(correct|almost|partial|incorrect)"',
        r'"grade"\s*:\s*"(correct|almost|partial|incorrect)"',
        r'"result"\s*:\s*"(correct|almost|partial|incorrect)"',
        r'"evaluation"\s*:\s*"(correct|almost|partial|incorrect)"',
        r'"answer"\s*:\s*"(correct|almost|partial|incorrect)"',
        r'"prediction"\s*:\s*"(correct|almost|partial|incorrect)"',
        r'"label"\s*:\s*"(correct|almost|partial|incorrect)"',
        r'"category"\s*:\s*"(correct|almost|partial|incorrect)"',
        r'"output"\s*:\s*"(correct|almost|partial|incorrect)"',
        r'"(correct|almost|partial|incorrect)"',
        # Single quote variants
        r"'response'\s*:\s*'(correct|almost|partial|incorrect)'",
        r"'classification'\s*:\s*'(correct|almost|partial|incorrect)'",
        r"'grade'\s*:\s*'(correct|almost|partial|incorrect)'",
        r"'result'\s*:\s*'(correct|almost|partial|incorrect)'",
        r"'evaluation'\s*:\s*'(correct|almost|partial|incorrect)'",
        r"'answer'\s*:\s*'(correct|almost|partial|incorrect)'",
        r"'prediction'\s*:\s*'(correct|almost|partial|incorrect)'",
        r"'label'\s*:\s*'(correct|almost|partial|incorrect)'",
        r"'category'\s*:\s*'(correct|almost|partial|incorrect)'",
        r"'output'\s*:\s*'(correct|almost|partial|incorrect)'",
        r"'(correct|almost|partial|incorrect)'",
        # Unquoted values
        r'"response"\s*:\s*(correct|almost|partial|incorrect)\b',
        r'"classification"\s*:\s*(correct|almost|partial|incorrect)\b',
        r'"grade"\s*:\s*(correct|almost|partial|incorrect)\b',
        r'"result"\s*:\s*(correct|almost|partial|incorrect)\b',
    ]
    for pattern in json_patterns:
        match = re.search(pattern, text_lower)
        if match:
            candidate = match.group(1).lower()
            if _is_valid_prediction(candidate):
                return candidate
    
    # Priority 2: Look for labels after colons or equals signs
    colon_patterns = [
        r'\bresponse\s*[=:]\s*["\']?(correct|almost|partial|incorrect)["\']?\b',
        r'\bclassification\s*[=:]\s*["\']?(correct|almost|partial|incorrect)["\']?\b',
        r'\bgrade\s*[=:]\s*["\']?(correct|almost|partial|incorrect)["\']?\b',
        r'\bresult\s*[=:]\s*["\']?(correct|almost|partial|incorrect)["\']?\b',
        r'\bevaluation\s*[=:]\s*["\']?(correct|almost|partial|incorrect)["\']?\b',
        r'\banswer\s*[=:]\s*["\']?(correct|almost|partial|incorrect)["\']?\b',
        r'\bprediction\s*[=:]\s*["\']?(correct|almost|partial|incorrect)["\']?\b',
        r'\blabel\s*[=:]\s*["\']?(correct|almost|partial|incorrect)["\']?\b',
        r'\bcategory\s*[=:]\s*["\']?(correct|almost|partial|incorrect)["\']?\b',
        r'\boutput\s*[=:]\s*["\']?(correct|almost|partial|incorrect)["\']?\b',
    ]
    for pattern in colon_patterns:
        match = re.search(pattern, text_lower)
        if match:
            candidate = match.group(1).lower()
            if _is_valid_prediction(candidate):
                return candidate
    
    # Priority 3: Look for labels in specific contexts
    context_patterns = [
        r'\bis\s+(?:a\s+)?(correct|almost|partial|incorrect)\b',
        r'\bwould\s+be\s+(?:a\s+)?(correct|almost|partial|incorrect)\b',
        r'\bshould\s+be\s+(?:a\s+)?(correct|almost|partial|incorrect)\b',
        r'\bclassified\s+as\s+(?:a\s+)?(correct|almost|partial|incorrect)\b',
        r'\bcategor(?:y|ized)\s+(?:as\s+)?(correct|almost|partial|incorrect)\b',
        r'\bthis\s+is\s+(?:a\s+)?(correct|almost|partial|incorrect)\b',
        r'\btherefore\s+(?:it\s+is\s+)?(correct|almost|partial|incorrect)\b',
        r'\bthe\s+answer\s+is\s+(correct|almost|partial|incorrect)\b',
        r'\bthe\s+solution\s+is\s+(correct|almost|partial|incorrect)\b',
        r'\bthe\s+grade\s+is\s+(correct|almost|partial|incorrect)\b',
        r'\bthe\s+classification\s+is\s+(correct|almost|partial|incorrect)\b',
        r'\bthe\s+result\s+is\s+(correct|almost|partial|incorrect)\b',
        r'\bthe\s+evaluation\s+is\s+(correct|almost|partial|incorrect)\b',
        r'\bgrade[d]?\s+(?:as\s+)?(correct|almost|partial|incorrect)\b',
        r'\bevaluate[d]?\s+(?:as\s+)?(correct|almost|partial|incorrect)\b',
        r'\bmark(?:ed)?\s+(?:as\s+)?(correct|almost|partial|incorrect)\b',
        r'\brated\s+(?:as\s+)?(correct|almost|partial|incorrect)\b',
        r'\bscored\s+(?:as\s+)?(correct|almost|partial|incorrect)\b',
        r'\bassigned\s+(?:as\s+)?(correct|almost|partial|incorrect)\b',
        r'\bdetermined\s+(?:as\s+)?(correct|almost|partial|incorrect)\b',
        r'\bassessed\s+(?:as\s+)?(correct|almost|partial|incorrect)\b',
    ]
    for pattern in context_patterns:
        match = re.search(pattern, text_lower)
        if match:
            candidate = match.group(1).lower()
            if _is_valid_prediction(candidate):
                return candidate
    
    # Priority 4: Look for labels at the end of sentences or lines
    final_patterns = [
        r'(?:final\s+)?(?:answer|classification|grade|result|evaluation|prediction|output)[\s:]+["\']?(correct|almost|partial|incorrect)["\']?[.!?]?\s*$',
        r'(?:final\s+)?(?:answer|classification|grade|result|evaluation|prediction|output)\s+is\s+["\']?(correct|almost|partial|incorrect)["\']?[.!?]?\s*$',
        r'\b["\']?(correct|almost|partial|incorrect)["\']?\s*[.!?]?\s*$',
        r'\bgrade\s*:\s*["\']?(correct|almost|partial|incorrect)["\']?\s*$',
        r'\bclassification\s*:\s*["\']?(correct|almost|partial|incorrect)["\']?\s*$',
        r'\bresult\s*:\s*["\']?(correct|almost|partial|incorrect)["\']?\s*$',
        r'\bevaluation\s*:\s*["\']?(correct|almost|partial|incorrect)["\']?\s*$',
    ]
    for pattern in final_patterns:
        match = re.search(pattern, text_lower, re.MULTILINE)
        if match:
            candidate = match.group(1).lower()
            if _is_valid_prediction(candidate):
                return candidate
    
    # Priority 5: Look for standalone labels at line starts
    line_start_patterns = [
        r'(?:^|\n)\s*(correct|almost|partial|incorrect)\s*(?:$|\n)',
        r'(?:^|\n)\s*[-*]\s*(correct|almost|partial|incorrect)\b',
        r'(?:^|\n)\s*\d+[.):]\s*(correct|almost|partial|incorrect)\b',
    ]
    for pattern in line_start_patterns:
        match = re.search(pattern, text_lower)
        if match:
            candidate = match.group(1).lower()
            if _is_valid_prediction(candidate):
                return candidate
    
    # Priority 6: Check for standalone words at word boundaries
    # Check in order of specificity (most specific first)
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
        start = max(0, match.start() - 30)
        context = text_lower[start:match.start()]
        # Check if preceded by negating words
        negating_patterns = ['not ', 'in', "isn't", 'isnt', 'not', 'never', 'hardly', 'barely', 
                          'not entirely', 'not fully', 'not completely', 'not totally', 'not wholly']
        if not any(neg in context for neg in negating_patterns):
            return "correct"
    
    return "unknown"


def _parse_grading_guidelines(guidelines: str) -> dict:
    """Parse grading guidelines to extract rubric indicators with enhanced pattern detection and comprehensive marker support."""
    result = {
        "has_partial": False,
        "has_almost": False,
        "has_correct": False,
        "has_incorrect": False,
        "primary_category": None,
        "confidence": 0.0,
        "score_ratio": None,
    }
    
    if not guidelines:
        return result
    
    text_lower = guidelines.lower()
    
    # Priority 0: Check for explicit markers with various delimiters (highest confidence)
    # These patterns look for markers like (Correct), [Correct], {Correct}, <Correct>, etc.
    marker_patterns = [
        # Parentheses variants - exact matches
        (r'\(\s*correct\s*\)', "correct", 0.99),
        (r'\(\s*almost\s*\)', "almost", 0.99),
        (r'\(\s*partial\s*\)', "partial", 0.99),
        (r'\(\s*incorrect\s*\)', "incorrect", 0.99),
        # Bracket variants - exact matches
        (r'\[\s*correct\s*\]', "correct", 0.99),
        (r'\[\s*almost\s*\]', "almost", 0.99),
        (r'\[\s*partial\s*\]', "partial", 0.99),
        (r'\[\s*incorrect\s*\]', "incorrect", 0.99),
        # Curly brace variants - exact matches
        (r'\{\s*correct\s*\}', "correct", 0.99),
        (r'\{\s*almost\s*\}', "almost", 0.99),
        (r'\{\s*partial\s*\}', "partial", 0.99),
        (r'\{\s*incorrect\s*\}', "incorrect", 0.99),
        # Angle bracket variants - exact matches
        (r'<\s*correct\s*>', "correct", 0.99),
        (r'<\s*almost\s*>', "almost", 0.99),
        (r'<\s*partial\s*>', "partial", 0.99),
        (r'<\s*incorrect\s*>', "incorrect", 0.99),
        # Double parentheses/brackets
        (r'\(\(\s*correct\s*\)\)', "correct", 0.99),
        (r'\(\(\s*almost\s*\)\)', "almost", 0.99),
        (r'\(\(\s*partial\s*\)\)', "partial", 0.99),
        (r'\(\(\s*incorrect\s*\)\)', "incorrect", 0.99),
        (r'\[\[\s*correct\s*\]\]', "correct", 0.99),
        (r'\[\[\s*almost\s*\]\]', "almost", 0.99),
        (r'\[\[\s*partial\s*\]\]', "partial", 0.99),
        (r'\[\[\s*incorrect\s*\]\]', "incorrect", 0.99),
        # Colon variants at end of line or before marker
        (r':\s*correct\s*$', "correct", 0.95),
        (r':\s*almost\s*$', "almost", 0.95),
        (r':\s*partial\s*$', "partial", 0.95),
        (r':\s*incorrect\s*$', "incorrect", 0.95),
        (r':\s*correct\s*\n', "correct", 0.95),
        (r':\s*almost\s*\n', "almost", 0.95),
        (r':\s*partial\s*\n', "partial", 0.95),
        (r':\s*incorrect\s*\n', "incorrect", 0.95),
        # Arrow variants
        (r'->\s*correct\b', "correct", 0.95),
        (r'->\s*almost\b', "almost", 0.95),
        (r'->\s*partial\b', "partial", 0.95),
        (r'->\s*incorrect\b', "incorrect", 0.95),
        (r'→\s*correct\b', "correct", 0.95),
        (r'→\s*almost\b', "almost", 0.95),
        (r'→\s*partial\b', "partial", 0.95),
        (r'→\s*incorrect\b', "incorrect", 0.95),
        # Equals variants
        (r'=\s*correct\b', "correct", 0.95),
        (r'=\s*almost\b', "almost", 0.95),
        (r'=\s*partial\b', "partial", 0.95),
        (r'=\s*incorrect\b', "incorrect", 0.95),
        # Dash variants
        (r'--?\s*correct\b', "correct", 0.95),
        (r'--?\s*almost\b', "almost", 0.95),
        (r'--?\s*partial\b', "partial", 0.95),
        (r'--?\s*incorrect\b', "incorrect", 0.95),
        # Word boundary markers (e.g., "Category: Correct" or "Grade: Almost")
        (r'\bcategory\s*[:=]\s*correct\b', "correct", 0.95),
        (r'\bcategory\s*[:=]\s*almost\b', "almost", 0.95),
        (r'\bcategory\s*[:=]\s*partial\b', "partial", 0.95),
        (r'\bcategory\s*[:=]\s*incorrect\b', "incorrect", 0.95),
        (r'\bgrade\s*[:=]\s*correct\b', "correct", 0.95),
        (r'\bgrade\s*[:=]\s*almost\b', "almost", 0.95),
        (r'\bgrade\s*[:=]\s*partial\b', "partial", 0.95),
        (r'\bgrade\s*[:=]\s*incorrect\b', "incorrect", 0.95),
        (r'\bclassification\s*[:=]\s*correct\b', "correct", 0.95),
        (r'\bclassification\s*[:=]\s*almost\b', "almost", 0.95),
        (r'\bclassification\s*[:=]\s*partial\b', "partial", 0.95),
        (r'\bclassification\s*[:=]\s*incorrect\b', "incorrect", 0.95),
    ]
    
    for pattern, category, confidence in marker_patterns:
        if re.search(pattern, text_lower, re.MULTILINE | re.IGNORECASE):
            result[f"has_{category}"] = True
            result["primary_category"] = category
            result["confidence"] = confidence
            return result  # Return immediately on exact marker match
    
    # Priority 1: Check for credit markers (very high confidence)
    credit_patterns = [
        # Full credit markers with various delimiters
        (r'\(\s*full\s*credit\s*\)', "correct", 0.98),
        (r'\[\s*full\s*credit\s*\]', "correct", 0.98),
        (r'\{\s*full\s*credit\s*\}', "correct", 0.98),
        (r'<\s*full\s*credit\s*>', "correct", 0.98),
        (r'\(\s*full\s*marks?\s*\)', "correct", 0.98),
        (r'\[\s*full\s*marks?\s*\]', "correct", 0.98),
        (r'\{\s*full\s*marks?\s*\}', "correct", 0.98),
        (r'<\s*full\s*marks?\s*>', "correct", 0.98),
        (r'\(\s*full\s*points?\s*\)', "correct", 0.98),
        (r'\[\s*full\s*points?\s*\]', "correct", 0.98),
        (r'\{\s*full\s*points?\s*\}', "correct", 0.98),
        (r'<\s*full\s*points?\s*>', "correct", 0.98),
        (r'\(\s*100%\s*(?:credit|marks?|points?)?\s*\)', "correct", 0.98),
        (r'\[\s*100%\s*(?:credit|marks?|points?)?\s*\]', "correct", 0.98),
        # No credit markers
        (r'\(\s*no\s*credit\s*\)', "incorrect", 0.98),
        (r'\[\s*no\s*credit\s*\]', "incorrect", 0.98),
        (r'\{\s*no\s*credit\s*\}', "incorrect", 0.98),
        (r'<\s*no\s*credit\s*>', "incorrect", 0.98),
        (r'\(\s*no\s*marks?\s*\)', "incorrect", 0.98),
        (r'\[\s*no\s*marks?\s*\]', "incorrect", 0.98),
        (r'\{\s*no\s*marks?\s*\}', "incorrect", 0.98),
        (r'<\s*no\s*marks?\s*>', "incorrect", 0.98),
        (r'\(\s*zero\s*(?:credit|marks?|points?)?\s*\)', "incorrect", 0.98),
        (r'\[\s*zero\s*(?:credit|marks?|points?)?\s*\]', "incorrect", 0.98),
        (r'\{\s*zero\s*(?:credit|marks?|points?)?\s*\}', "incorrect", 0.98),
        (r'\(\s*0\s*(?:credit|marks?|points?)?\s*\)', "incorrect", 0.98),
        (r'\[\s*0\s*(?:credit|marks?|points?)?\s*\]', "incorrect", 0.98),
        # Half credit markers
        (r'\(\s*half\s*credit\s*\)', "partial", 0.95),
        (r'\[\s*half\s*credit\s*\]', "partial", 0.95),
        (r'\{\s*half\s*credit\s*\}', "partial", 0.95),
        (r'<\s*half\s*credit\s*>', "partial", 0.95),
        (r'\(\s*half\s*marks?\s*\)', "partial", 0.95),
        (r'\[\s*half\s*marks?\s*\]', "partial", 0.95),
        (r'\{\s*half\s*marks?\s*\}', "partial", 0.95),
        (r'<\s*half\s*marks?\s*>', "partial", 0.95),
        (r'\(\s*50%\s*(?:credit|marks?|points?)?\s*\)', "partial", 0.95),
        (r'\[\s*50%\s*(?:credit|marks?|points?)?\s*\]', "partial", 0.95),
        # Most credit markers
        (r'\(\s*most\s*credit\s*\)', "almost", 0.95),
        (r'\[\s*most\s*credit\s*\]', "almost", 0.95),
        (r'\{\s*most\s*credit\s*\}', "almost", 0.95),
        (r'<\s*most\s*credit\s*>', "almost", 0.95),
        (r'\(\s*most\s*marks?\s*\)', "almost", 0.95),
        (r'\[\s*most\s*marks?\s*\]', "almost", 0.95),
        (r'\{\s*most\s*marks?\s*\}', "almost", 0.95),
        (r'<\s*most\s*marks?\s*>', "almost", 0.95),
        (r'\(\s*75%\s*(?:credit|marks?|points?)?\s*\)', "almost", 0.95),
        (r'\[\s*75%\s*(?:credit|marks?|points?)?\s*\]', "almost", 0.95),
        # Some credit markers
        (r'\(\s*some\s*credit\s*\)', "partial", 0.90),
        (r'\[\s*some\s*credit\s*\]', "partial", 0.90),
        (r'\{\s*some\s*credit\s*\}', "partial", 0.90),
        (r'<\s*some\s*credit\s*>', "partial", 0.90),
        (r'\(\s*some\s*marks?\s*\)', "partial", 0.90),
        (r'\[\s*some\s*marks?\s*\]', "partial", 0.90),
        (r'\{\s*some\s*marks?\s*\}', "partial", 0.90),
        (r'<\s*some\s*marks?\s*>', "partial", 0.90),
        (r'\(\s*25%\s*(?:credit|marks?|points?)?\s*\)', "partial", 0.90),
        (r'\[\s*25%\s*(?:credit|marks?|points?)?\s*\]', "partial", 0.90),
        # Partial credit markers
        (r'\(\s*partial\s*credit\s*\)', "partial", 0.95),
        (r'\[\s*partial\s*credit\s*\]', "partial", 0.95),
        (r'\{\s*partial\s*credit\s*\}', "partial", 0.95),
        (r'<\s*partial\s*credit\s*>', "partial", 0.95),
        # Award patterns - more flexible matching
        (r'\baward\s+full\s+(?:credit|marks?|points?)\b', "correct", 0.90),
        (r'\baward\s+no\s+(?:credit|marks?|points?)\b', "incorrect", 0.90),
        (r'\baward\s+half\s+(?:credit|marks?|points?)\b', "partial", 0.85),
        (r'\baward\s+most\s+(?:credit|marks?|points?)\b', "almost", 0.85),
        (r'\baward\s+some\s+(?:credit|marks?|points?)\b', "partial", 0.80),
        (r'\baward\s+(?:\d+)\s*/\s*(\d+)\b', "score", 0.85),  # Will be handled by score patterns
        # Give patterns - more flexible matching
        (r'\bgive\s+full\s+(?:credit|marks?|points?)\b', "correct", 0.90),
        (r'\bgive\s+no\s+(?:credit|marks?|points?)\b', "incorrect", 0.90),
        (r'\bgive\s+half\s+(?:credit|marks?|points?)\b', "partial", 0.85),
        (r'\bgive\s+most\s+(?:credit|marks?|points?)\b', "almost", 0.85),
        (r'\bgive\s+some\s+(?:credit|marks?|points?)\b', "partial", 0.80),
        # Full/No/Half/Most/Some credit without award/give
        (r'\bfull\s+(?:credit|marks?|points?)\b', "correct", 0.85),
        (r'\bno\s+(?:credit|marks?|points?)\b', "incorrect", 0.85),
        (r'\bhalf\s+(?:credit|marks?|points?)\b', "partial", 0.80),
        (r'\bmost\s+(?:credit|marks?|points?)\b', "almost", 0.80),
        (r'\bsome\s+(?:credit|marks?|points?)\b', "partial", 0.75),
    ]
    
    for pattern, category, confidence in credit_patterns:
        if category == "score":
            continue  # Skip score placeholder, handled below
        if re.search(pattern, text_lower, re.IGNORECASE):
            result[f"has_{category}"] = True
            if result["confidence"] < confidence:
                result["primary_category"] = category
                result["confidence"] = confidence
    
    # Priority 2: Check for score patterns like (3/4 points) or (7/10)
    score_patterns = [
        # Parentheses
        r'\(\s*(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)\s*(?:points?|pts?|marks?)?\s*\)',
        # Brackets
        r'\[\s*(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)\s*(?:points?|pts?|marks?)?\s*\]',
        # Curly braces
        r'\{\s*(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)\s*(?:points?|pts?|marks?)?\s*\}',
        # Angle brackets
        r'<\s*(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)\s*(?:points?|pts?|marks?)?\s*>',
        # Word boundaries with points/marks
        r'\b(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)\s*(?:points?|pts?|marks?)\b',
        # Award patterns
        r'\baward\s+(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)\b',
        r'\baward\s+(\d+(?:\.\d+)?)\s+out\s+of\s+(\d+(?:\.\d+)?)\b',
        r'\baward\s+(\d+(?:\.\d+)?)\s+points?\b',
        # Out of patterns
        r'\b(\d+(?:\.\d+)?)\s+out\s+of\s*(\d+(?:\.\d+)?)\s*(?:points?|pts?|marks?)\b',
        r'\b(\d+(?:\.\d+)?)\s+out\s+of\s*(\d+(?:\.\d+)?)\b',
        # Score patterns
        r'\bscore[d]?\s+(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)\b',
        r'\bscore[d]?\s+(\d+(?:\.\d+)?)\s+out\s+of\s+(\d+(?:\.\d+)?)\b',
        # End of line patterns
        r'\b(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)\s*$',
        r'\b(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)\s*(?:points?|pts?|marks?)?\s*(?:\n|$)',
        # Percentage patterns
        r'\b(\d+(?:\.\d+)?)%\s*(?:score|grade|correct)?\b',
    ]
    
    for pattern in score_patterns:
        match = re.search(pattern, text_lower)
        if match:
            try:
                if len(match.groups()) == 1:
                    # Percentage pattern
                    ratio = float(match.group(1)) / 100.0
                else:
                    earned = float(match.group(1))
                    total = float(match.group(2))
                    if total > 0:
                        ratio = earned / total
                    else:
                        continue
                
                result["score_ratio"] = ratio
                
                # Determine category based on score ratio
                if ratio >= 0.95:
                    category = "correct"
                    conf = 0.90
                elif ratio >= 0.70:
                    category = "almost"
                    conf = 0.90
                elif ratio >= 0.35:
                    category = "partial"
                    conf = 0.85
                else:
                    category = "incorrect"
                    conf = 0.90
                
                result[f"has_{category}"] = True
                if result["confidence"] < conf:
                    result["primary_category"] = category
                    result["confidence"] = conf
                break  # Use first score pattern found
            except (ValueError, ZeroDivisionError):
                continue
    
    # Priority 3: Check for grade/result keywords
    grade_patterns = [
        # Grade statements
        (r'\bgrade[d]?\s+(?:as\s+)?correct\b', "correct", 0.85),
        (r'\bgrade[d]?\s+(?:as\s+)?almost\b', "almost", 0.85),
        (r'\bgrade[d]?\s+(?:as\s+)?partial\b', "partial", 0.85),
        (r'\bgrade[d]?\s+(?:as\s+)?incorrect\b', "incorrect", 0.85),
        # Result statements
        (r'\bresult\s+(?:is\s+)?correct\b', "correct", 0.80),
        (r'\bresult\s+(?:is\s+)?almost\b', "almost", 0.80),
        (r'\bresult\s+(?:is\s+)?partial\b', "partial", 0.80),
        (r'\bresult\s+(?:is\s+)?incorrect\b', "incorrect", 0.80),
        # Classification statements
        (r'\bclassif(?:y|ied)\s+(?:as\s+)?correct\b', "correct", 0.80),
        (r'\bclassif(?:y|ied)\s+(?:as\s+)?almost\b', "almost", 0.80),
        (r'\bclassif(?:y|ied)\s+(?:as\s+)?partial\b', "partial", 0.80),
        (r'\bclassif(?:y|ied)\s+(?:as\s+)?incorrect\b', "incorrect", 0.80),
        # Category statements
        (r'\bcategor(?:y|ized)\s+(?:as\s+)?correct\b', "correct", 0.80),
        (r'\bcategor(?:y|ized)\s+(?:as\s+)?almost\b', "almost", 0.80),
        (r'\bcategor(?:y|ized)\s+(?:as\s+)?partial\b', "partial", 0.80),
        (r'\bcategor(?:y|ized)\s+(?:as\s+)?incorrect\b', "incorrect", 0.80),
        # Label statements
        (r'\blabel(?:ed|led)?\s+(?:as\s+)?correct\b', "correct", 0.80),
        (r'\blabel(?:ed|led)?\s+(?:as\s+)?almost\b', "almost", 0.80),
        (r'\blabel(?:ed|led)?\s+(?:as\s+)?partial\b', "partial", 0.80),
        (r'\blabel(?:ed|led)?\s+(?:as\s+)?incorrect\b', "incorrect", 0.80),
    ]
    
    for pattern, category, confidence in grade_patterns:
        if re.search(pattern, text_lower, re.IGNORECASE):
            result[f"has_{category}"] = True
            if result["confidence"] < confidence:
                result["primary_category"] = category
                result["confidence"] = confidence
    
    # Priority 4: Check for contextual keywords with comprehensive patterns
    keyword_patterns = [
        # Correct indicators - comprehensive
        (r'\b(?:full|complete|perfect|flawless)\s+(?:credit|marks?|score|solution|proof|answer)\b', "correct", 0.80),
        (r'\ball\s+(?:steps|work)\s+(?:correct|right|valid|sound)\b', "correct", 0.80),
        (r'\bcorrect\s+(?:final\s+)?(?:answer|solution|proof)\b', "correct", 0.75),
        (r'\b(?:entirely|totally|fully|completely|wholly|absolutely)\s+correct\b', "correct", 0.80),
        (r'\b100%\s+(?:correct|right|valid|accurate)\b', "correct", 0.85),
        (r'\bno\s+(?:errors|mistakes|issues|problems|flaws|gaps)\b', "correct", 0.75),
        (r'\b(?:valid|sound|rigorous|correct)\s+(?:proof|solution|argument|reasoning)\b', "correct", 0.75),
        (r'\bsolution\s+is\s+(?:correct|valid|sound|rigorous|complete)\b', "correct", 0.75),
        (r'\banswer\s+is\s+(?:correct|valid|right|accurate)\b', "correct", 0.75),
        (r'\bproof\s+is\s+(?:correct|valid|sound|rigorous|complete)\b', "correct", 0.75),
        (r'\bcorrectly\s+(?:solved|proved|answered|derived|shown)\b', "correct", 0.80),
        (r'\bsuccessful\s+(?:solution|proof|attempt)\b', "correct", 0.75),
        (r'\bcomplete\s+and\s+correct\b', "correct", 0.85),
        (r'\bcorrect\s+and\s+complete\b', "correct", 0.85),
        # Almost indicators - comprehensive
        (r'\b(?:minor|small|trivial|slight|tiny|little)\s+(?:error|mistake|issue|problem|flaw|gap|slip)\b', "almost", 0.75),
        (r'\balmost\s+(?:correct|there|right|complete|solved|perfect)\b', "almost", 0.80),
        (r'\bnearly\s+(?:correct|there|right|complete|solved|perfect)\b', "almost", 0.80),
        (r'\bvery\s+close\s+(?:to\s+)?(?:correct|right|solution|answer)\b', "almost", 0.75),
        (r'\b(?:just|only)\s+(?:a|one|some)\s+(?:minor|small|trivial|slight)\b', "almost", 0.70),
        (r'\bcosmetic\s+(?:error|issue|problem|mistake|flaw)\b', "almost", 0.70),
        (r'\b(?:slip|sloppy|careless)\s+(?:error|mistake)\b', "almost", 0.70),
        (r'\b(?:mostly|nearly|practically|virtually|essentially)\s+(?:correct|right|valid|accurate)\b', "almost", 0.75),
        (r'\b(?:small|minor|trivial)\s+(?:typo|arithmetic|calculation|computation|notation|sign)\s+(?:error|mistake|issue)\b', "almost", 0.75),
        (r'\b(?:95%|90%|96%|97%|98%|99%|94%)\s+(?:correct|right|accurate)\b', "almost", 0.80),
        (r'\bcorrect\s+(?:except|apart\s+from|aside\s+from|but)\b', "almost", 0.75),
        (r'\b(?:small|minor)\s+(?:correction|fix|adjustment)\s+(?:needed|required)\b', "almost", 0.80),
        (r'\bwould\s+be\s+correct\s+if\b', "almost", 0.80),
        (r'\b(?:just|only)\s+needs?\s+(?:a|one)\s+(?:minor|small)\b', "almost", 0.80),
        # Partial indicators - comprehensive
        (r'\bincomplete\s+(?:solution|answer|work|proof|argument|reasoning)\b', "partial", 0.75),
        (r'\bpartial\s+(?:credit|solution|progress|work|answer|proof|success)\b', "partial", 0.80),
        (r'\bsome\s+(?:progress|correct|valid|work|steps|success|credit)\b', "partial", 0.70),
        (r'\bon\s+the\s+right\s+track\b', "partial", 0.70),
        (r'\bgood\s+(?:start|beginning|attempt|effort)\b', "partial", 0.70),
        (r'\bcorrect\s+(?:approach|idea|method|strategy|start|beginning)\b', "partial", 0.70),
        (r'\bvalid\s+(?:approach|method|strategy|start|idea)\b', "partial", 0.70),
        (r'\bmissing\s+(?:steps|work|justification|proof|conclusion|parts|components)\b', "partial", 0.70),
        (r'\bincomplete\s+(?:justification|proof|argument|reasoning|work)\b', "partial", 0.70),
        (r'\bpartially\s+(?:correct|right|valid|accurate|solved|complete)\b', "partial", 0.75),
        (r'\b(?:half|50%|40%|60%|30%|70%)\s+(?:correct|right|valid|accurate|complete)\b', "partial", 0.75),
        (r'\b(?:some|part)\s+of\s+(?:the\s+)?(?:solution|answer|proof|work)\b', "partial", 0.70),
        (r'\bstarted\s+(?:correctly|well|right|properly)\b', "partial", 0.70),
        (r'\b(?:significant|major|substantial)\s+gaps?\b', "partial", 0.70),
        (r'\bnot\s+(?:fully|entirely|totally|completely)\s+(?:complete|correct|solved|finished)\b', "partial", 0.70),
        (r'\bmissing\s+(?:the\s+)?(?:final|last|concluding)\s+(?:step|part|section|answer)\b', "partial", 0.75),
        (r'\bneeds\s+(?:more|completion|finishing|work)\b', "partial", 0.70),
        (r'\bunfinished\b', "partial", 0.70),
        (r'\bpartially\s+(?:correct|right|valid|successful)\b', "partial", 0.80),
        (r'\b(?:made|showing)\s+progress\b', "partial", 0.70),
        (r'\b(?:some|partial)\s+understanding\b', "partial", 0.70),
        # Incorrect indicators - comprehensive
        (r'\bno\s+(?:credit|marks?|score|progress|meaningful|substantive|valid)\b', "incorrect", 0.80),
        (r'\bfundamentally\s+(?:wrong|incorrect|flawed|erroneous)\b', "incorrect", 0.85),
        (r'\b(?:completely|totally|entirely|wholly|absolutely)\s+(?:wrong|incorrect|flawed|erroneous)\b', "incorrect", 0.85),
        (r'\bmajor\s+(?:error|mistake|flaw|problem|issue)\b', "incorrect", 0.75),
        (r'\bcritical\s+(?:error|mistake|flaw|problem|issue)\b', "incorrect", 0.75),
        (r'\b(?:wrong|incorrect)\s+(?:approach|method|answer|solution|proof|reasoning)\b', "incorrect", 0.80),
        (r'\bno\s+(?:solution|answer|proof|progress|work|attempt)\b', "incorrect", 0.80),
        (r'\bno\s+valid\s+(?:solution|work|progress|attempt|answer)\b', "incorrect", 0.80),
        (r'\b(?:failed|fails)\s+to\s+(?:solve|prove|show|demonstrate|answer)\b', "incorrect", 0.75),
        (r'\b(?:invalid|flawed)\s+(?:approach|method|argument|reasoning|solution)\b', "incorrect", 0.75),
        (r"\b(?:does not|doesn't|dont|don't)\s+(?:solve|prove|show|demonstrate|answer|work)\b", "incorrect", 0.75),
        (r'\b(?:no|zero)\s+(?:substantive|meaningful|valid|real)\s+(?:work|progress|content|attempt)\b', "incorrect", 0.80),
        (r'\b(?:irrelevant|unrelated|off-topic|nonsense|gibberish)\b', "incorrect", 0.80),
        (r'\b(?:blank|empty|no\s+submission)\b', "incorrect", 0.85),
        (r'\b(?:unsuccessful|failed)\s+(?:attempt|solution|try)\b', "incorrect", 0.80),
        (r'\b(?:does|did)\s+not\s+(?:understand|grasp|comprehend)\b', "incorrect", 0.75),
    ]
    
    for pattern, category, confidence in keyword_patterns:
        if re.search(pattern, text_lower, re.IGNORECASE):
            result[f"has_{category}"] = True
            if result["confidence"] < confidence:
                result["primary_category"] = category
                result["confidence"] = confidence
    
    return result


class TaskAgent:
    """Task agent that evaluates student solutions using LLM."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_file = log_file
        self.log_fn = logger.info

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Evaluate a student solution and return classification."""
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        student_answer = inputs.get("student_answer", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        
        # Parse grading guidelines with enhanced rubric detection
        rubric = _parse_grading_guidelines(grading_guidelines)
        
        # Build rubric context with more detailed guidance
        rubric_context = ""
        if rubric["has_partial"]:
            rubric_context += "\n\n[RUBRIC: The grading guidelines contain '(Partial)' indicating partial credit for incomplete solutions with some correct elements.]"
        if rubric["has_almost"]:
            rubric_context += "\n\n[RUBRIC: The grading guidelines contain '(Almost)' indicating near-correctness with only minor errors or typos.]"
        if rubric["has_correct"]:
            rubric_context += "\n\n[RUBRIC: The grading guidelines contain '(Correct)' indicating full correctness and complete solutions.]"
        if rubric["has_incorrect"]:
            rubric_context += "\n\n[RUBRIC: The grading guidelines contain '(Incorrect)' indicating no credit for fundamentally wrong answers.]"
        
        # Add score ratio context if available
        if rubric.get("score_ratio") is not None:
            ratio = rubric["score_ratio"]
            rubric_context += f"\n\n[SCORE RATIO: The grading guidelines indicate a score ratio of {ratio:.0%}.]"
        
        instruction = f"""You are an expert mathematical grader evaluating student solutions to competition mathematics problems.

Your task is to evaluate the student's answer and classify it into EXACTLY ONE of these four categories:

## Grade Definitions (CRITICAL - READ CAREFULLY)

- **correct**: Complete and fully correct solution. All steps present and logically sound. No gaps, no errors, fully rigorous proof. EVERY step must be correct and justified. The student has demonstrated complete mastery of the problem.

- **incorrect**: No meaningful progress or fundamental errors. Wrong approach or no valid mathematical progress. Student failed to demonstrate understanding of key concepts. Minimal or no substantive work toward the solution. Just restating the problem or random calculations without valid reasoning.

- **partial**: Significant progress but INCOMPLETE. Found useful invariant/lemma or established valid approach but didn't finish. Student demonstrated understanding of key ideas but solution has MAJOR gaps or STOPS before completion. The student has NOT addressed all major components. Missing conclusion or final step is a KEY indicator. The student made MEANINGFUL progress but the solution is NOT complete.

- **almost**: Nearly complete solution with ONLY minor issues. Main proof structure is CORRECT and COMPLETE. Only minor computational errors, typos, or notation issues. Student has addressed ALL major components. The proof is COMPLETE from start to finish INCLUDING the conclusion. If you fix trivial errors (arithmetic, typos), it becomes correct. The student understood the complete approach and executed it with only small blemishes. The solution is ESSENTIALLY CORRECT with cosmetic issues only.

## THE MOST IMPORTANT DISTINCTION: "almost" vs "partial"

This is where most grading errors occur. Use these tests:

### TEST 1: The "COMPLETE STRUCTURE" Test (MOST IMPORTANT)
Ask: "Does the student have a COMPLETE proof structure?"
- YES (complete structure, minor blemishes) → "almost"
- NO (missing components, incomplete) → "partial"

### TEST 2: The "CONCLUSION" Test (CRITICAL)
Ask: "Did the student REACH the final conclusion/answer?"
- YES (even with minor errors) → Strong indicator for "almost"
- NO (stopped before the end) → Strong indicator for "partial"

### TEST 3: The "FIXABILITY" Test (Apply rigorously!)
Ask: "Can I fix this by correcting ONLY trivial errors?"
- YES (arithmetic, typos, notation only) → "almost"
- NO (need new proof steps, lemmas, cases) → "partial"

### TEST 4: The "WHAT'S MISSING" Test
Ask: "What would need to be added to make this correct?"
- Nothing major (just fix tiny errors) → "almost"
- Something major (conclusion, cases, lemmas, proof steps) → "partial"

### TEST 5: The "PROGRESS" Test (for partial vs incorrect)
Ask: "Did the student make MEANINGFUL progress?"
- YES (found lemma, proved intermediate result, established approach) → "partial"
- NO (just restated problem, random calculations) → "incorrect"

## Common Mistakes to AVOID

1. **DON'T** grade "partial" when the student has a complete proof with only minor errors → This should be "almost"
2. **DON'T** grade "almost" when the solution has major logical gaps or missing proof steps → This should be "partial"
3. **DON'T** be too stingy with "almost" - if the student has a complete proof structure with minor issues, it's "almost" not "partial"
4. **DON'T** be too generous with "partial" - "partial" requires meaningful progress, not just "attempted"
5. **DON'T** grade "partial" when the student only made minimal progress → This should be "incorrect"
6. **DON'T** grade "almost" when the student is missing the conclusion → This should be "partial"
7. **DON'T** grade "partial" when the student reached the conclusion with only minor errors → This should be "almost"
8. **DON'T** be fooled by length - a long incomplete proof is still "partial", a short complete proof with minor errors is "almost"
9. **DON'T** grade "partial" when the student has a complete proof but wrong final number → This should be "almost"
10. **DON'T** grade "almost" when the student has only proven a lemma but not connected it → This should be "partial"
11. **DON'T** grade "almost" when the student stopped mid-way through → This should be "partial"
12. **DON'T** grade "partial" when the student has a complete solution with just a sign error → This should be "almost"

## RUBRIC MARKERS (FOLLOW THESE EXPLICITLY - HIGHEST PRIORITY!)
The Grading Guidelines below contain EXPLICIT markers that indicate the intended classification. These markers OVERRIDE all other considerations and are the STRONGEST signal for your decision:

### Exact Markers (use these exactly as written):
- (Correct) or [Correct] or {{Correct}} or <Correct> → Use "correct"
- (Almost) or [Almost] or {{Almost}} or <Almost> → Use "almost"
- (Partial) or [Partial] or {{Partial}} or <Partial> → Use "partial"
- (Incorrect) or [Incorrect] or {{Incorrect}} or <Incorrect> → Use "incorrect"

### Credit Markers:
- (Full Credit) or [Full Credit] or <Full Credit> → Use "correct"
- (Most Credit) or [Most Credit] or <Most Credit> → Use "almost"
- (Half Credit) or [Half Credit] or <Half Credit> → Use "partial"
- (Some Credit) or [Some Credit] or <Some Credit> → Use "partial"
- (No Credit) or [No Credit] or <No Credit> → Use "incorrect"

### Score Patterns (X/Y points):
- 95-100% of points → "correct"
- 70-94% of points → "almost"
- 35-69% of points → "partial"
- 0-34% of points → "incorrect"

### Award/Give Patterns:
- "award full credit" or "give full marks" → "correct"
- "award most credit" or "give most marks" → "almost"
- "award half credit" or "give some marks" → "partial"
- "award no credit" or "give no marks" → "incorrect"

{rubric_context}

=== PROBLEM ===
{problem}

=== OFFICIAL SOLUTION ===
{solution}

=== STUDENT'S ANSWER ===
{student_answer}

=== GRADING GUIDELINES ===
{grading_guidelines}

## Final Verification Checklist (COMPLETE ALL CHECKS!)

Before submitting your grade, answer these questions:

1. **COMPLETE STRUCTURE TEST**: Does the student have a complete proof structure from start to finish? (YES/NO)
2. **CONCLUSION TEST**: Did the student REACH the final conclusion/answer (even if with errors)? (YES/NO)
3. **FIXABILITY TEST**: Can I fix this by correcting ONLY trivial errors (arithmetic, typos, notation)? (YES/NO)
4. **COMPONENTS TEST**: Are ALL major components present and addressed? (YES/NO)
5. **ERROR SEVERITY TEST**: Are the errors truly MINOR and not MAJOR? (YES/NO)
6. **PROGRESS TEST**: Did the student make MEANINGFUL progress? (YES/NO)
7. **RUBRIC MARKER TEST**: Are there explicit rubric markers (Correct/Almost/Partial/Incorrect) in the grading guidelines? (YES/NO - if YES, FOLLOW THEM!)

### DECISION RULES:
- **For "correct"**: Q1-Q5 must ALL be YES, Q7 can be YES with "correct" marker
- **For "almost"**: Q1-Q5 must ALL be YES (errors are truly minor), Q7 can be YES with "almost" marker
- **For "partial"**: Q6 is YES, but at least one of Q1-Q4 is NO, Q7 can be YES with "partial" marker
- **For "incorrect"**: Q6 is NO (no meaningful progress), Q7 can be YES with "incorrect" marker

## RESPONSE FORMAT (STRICT - MUST FOLLOW EXACTLY)

You MUST respond ONLY in the following JSON format. Do not include any text before or after the JSON block:

<json>
{{
    "response": "correct" | "almost" | "partial" | "incorrect"
}}
</json>

CRITICAL INSTRUCTIONS:
1. The response field must contain EXACTLY one of: correct, almost, partial, incorrect (lowercase)
2. **FOLLOW THE RUBRIC MARKERS** in the grading guidelines - they are the STRONGEST signal and OVERRIDE all other considerations
3. DO NOT include any text, explanation, or markdown before or after the JSON block
4. Use ONLY the <json> tags shown above, not ```json code blocks
5. Your entire response should be just the <json>...</json> block with nothing else
6. If you see (Correct), (Almost), (Partial), or (Incorrect) markers, USE THAT CLASSIFICATION
7. If you see score patterns like (3/4 points), use the score ratio guidelines above
8. **IMPORTANT**: When in doubt, trust the explicit rubric markers over your own analysis"""

        # Use retry with backoff for LLM call
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
                return rubric["primary_category"], []
            return "unknown", []

        # Extract prediction from response with enhanced extraction
        prediction = "unknown"
        
        try:
            last_message = msg_history[-1]["text"]
            self.log_fn(f"Raw response: {last_message[:500]}...")
            
            # Attempt 1: Try flexible JSON extraction
            extracted = _extract_json_flexible(last_message)
            if extracted and isinstance(extracted, dict):
                # Try "response" key first (most common)
                if "response" in extracted:
                    prediction = _normalize_prediction(extracted["response"])
                    self.log_fn(f"Extracted from 'response' key: {prediction}")
                # Try other common keys
                elif "classification" in extracted:
                    prediction = _normalize_prediction(extracted["classification"])
                    self.log_fn(f"Extracted from 'classification' key: {prediction}")
                elif "grade" in extracted:
                    prediction = _normalize_prediction(extracted["grade"])
                    self.log_fn(f"Extracted from 'grade' key: {prediction}")
                elif "result" in extracted:
                    prediction = _normalize_prediction(extracted["result"])
                    self.log_fn(f"Extracted from 'result' key: {prediction}")
                elif "evaluation" in extracted:
                    prediction = _normalize_prediction(extracted["evaluation"])
                    self.log_fn(f"Extracted from 'evaluation' key: {prediction}")
                elif "answer" in extracted:
                    prediction = _normalize_prediction(extracted["answer"])
                    self.log_fn(f"Extracted from 'answer' key: {prediction}")
                elif len(extracted) == 1:
                    # Single key-value pair, use the value
                    prediction = _normalize_prediction(list(extracted.values())[0])
                    self.log_fn(f"Extracted from single key-value pair: {prediction}")
                else:
                    # Try all values to find a valid prediction
                    for key, value in extracted.items():
                        normalized = _normalize_prediction(value)
                        if _is_valid_prediction(normalized):
                            prediction = normalized
                            self.log_fn(f"Extracted from key '{key}': {prediction}")
                            break
            
            # Attempt 2: If still unknown, try direct text extraction
            if not _is_valid_prediction(prediction):
                text_prediction = _extract_prediction_from_text(last_message)
                if _is_valid_prediction(text_prediction):
                    prediction = text_prediction
                    self.log_fn(f"Extracted from text patterns: {prediction}")
            
            # Attempt 3: Check for rubric markers directly in the response
            if not _is_valid_prediction(prediction):
                response_rubric = _parse_grading_guidelines(last_message)
                if response_rubric.get("primary_category") and response_rubric.get("confidence", 0) > 0.7:
                    prediction = response_rubric["primary_category"]
                    self.log_fn(f"Using response rubric markers: {prediction}")
            
            # Fallback 1: Use grading guidelines rubric if LLM response is unclear
            if not _is_valid_prediction(prediction):
                if rubric.get("primary_category") and rubric.get("confidence", 0) > 0.5:
                    prediction = rubric["primary_category"]
                    self.log_fn(f"Using grading guidelines rubric as fallback: {prediction}")
            
            # Fallback 2: Try to extract from any valid label in the response
            if not _is_valid_prediction(prediction):
                # Last resort: look for any valid label in the text
                for label in ["correct", "almost", "partial", "incorrect"]:
                    # Use word boundary to avoid matching "incorrect" when looking for "correct"
                    if label == "correct":
                        # Special handling to avoid matching "incorrect"
                        if re.search(r'\bcorrect\b', last_message.lower()):
                            # Check it's not part of "incorrect"
                            for match in re.finditer(r'\bcorrect\b', last_message.lower()):
                                start = max(0, match.start() - 10)
                                if "in" not in last_message.lower()[start:match.start()]:
                                    prediction = label
                                    self.log_fn(f"Fallback extraction found: {prediction}")
                                    break
                    elif re.search(rf'\b{label}\b', last_message.lower()):
                        prediction = label
                        self.log_fn(f"Fallback extraction found: {prediction}")
                        break
                    if _is_valid_prediction(prediction):
                        break
            
            # Fallback 3: Check for any classification-like words in the response
            if not _is_valid_prediction(prediction):
                # Try to find any of the four categories with more flexible matching
                text_lower = last_message.lower()
                # Check for "almost" first (most specific)
                if re.search(r'\balmost\b', text_lower):
                    prediction = "almost"
                    self.log_fn(f"Fallback: found 'almost' in text")
                elif re.search(r'\bpartial\b', text_lower):
                    prediction = "partial"
                    self.log_fn(f"Fallback: found 'partial' in text")
                elif re.search(r'\bincorrect\b', text_lower):
                    prediction = "incorrect"
                    self.log_fn(f"Fallback: found 'incorrect' in text")
                elif re.search(r'\bcorrect\b', text_lower):
                    # Make sure it's not part of "incorrect"
                    for match in re.finditer(r'\bcorrect\b', text_lower):
                        start = max(0, match.start() - 10)
                        if "in" not in text_lower[start:match.start()]:
                            prediction = "correct"
                            self.log_fn(f"Fallback: found 'correct' in text")
                            break
            
            self.log_fn(f"Final prediction: {prediction}")
            
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            logger.error(f"Error extracting prediction: {e}", exc_info=True)
            # Emergency fallback to rubric if available
            if rubric.get("primary_category"):
                prediction = rubric["primary_category"]
                self.log_fn(f"Emergency fallback to rubric: {prediction}")
            else:
                prediction = "unknown"

        # Final validation
        if not _is_valid_prediction(prediction):
            prediction = "unknown"

        return str(prediction), msg_history
