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
    """Extract JSON from LLM response using multiple robust strategies."""
    if not text or not isinstance(text, str):
        return None
    
    text = text.strip()
    if not text:
        return None
    
    # Pre-process: Remove common prefixes
    text = re.sub(r'^(?:Here is|Here\'s|The|This is|My|Final|Output|Result|Answer|Classification|Grade|Response)[:\s]*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^(?:json|JSON)[:\s]*', '', text, flags=re.IGNORECASE)
    
    # Strategy 1: <json> tags (most reliable and preferred format)
    json_blocks = _extract_jsons(text)
    if json_blocks:
        return json_blocks[-1]
    
    # Strategy 2: ```json code blocks
    json_code_pattern = r'```(?:json|JSON)?\s*(.*?)```'
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
                continue
    
    # Strategy 3: Find JSON objects with smart brace matching
    candidates = []
    brace_count = 0
    start_idx = -1
    
    for i, char in enumerate(text):
        if char == '{':
            if brace_count == 0:
                start_idx = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start_idx != -1:
                candidates.append(text[start_idx:i+1])
                start_idx = -1
    
    # Try candidates in reverse order (prefer later/larger JSON objects)
    for candidate in reversed(candidates):
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                # Check if it has a valid prediction value
                for v in parsed.values():
                    if _normalize_prediction(str(v)) in ["correct", "almost", "partial", "incorrect"]:
                        return parsed
                # Or if it has expected keys
                if any(k in parsed for k in ["response", "classification", "grade", "result"]):
                    return parsed
        except json.JSONDecodeError:
            fixed = _fix_json_string(candidate)
            try:
                parsed = json.loads(fixed)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                continue
    
    # Strategy 4: Look for simple key-value patterns
    result = _extract_key_value_patterns(text)
    if result:
        return result
    
    # Strategy 5: Extract from last non-empty line
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    if lines:
        last_line = lines[-1]
        # Check if it's just a valid label
        norm_val = _normalize_prediction(last_line)
        if _is_valid_prediction(norm_val):
            return {"response": norm_val}
    
    return None


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


def _extract_key_value_patterns(text: str) -> dict | None:
    """Extract key-value patterns that look like JSON or grading responses."""
    text_lower = text.lower()
    
    # Pattern 1: Look for explicit key-value pairs with valid labels
    kv_patterns = [
        r'["\']?response["\']?\s*[:=]\s*["\']?(correct|almost|partial|incorrect)["\']?',
        r'["\']?classification["\']?\s*[:=]\s*["\']?(correct|almost|partial|incorrect)["\']?',
        r'["\']?grade["\']?\s*[:=]\s*["\']?(correct|almost|partial|incorrect)["\']?',
        r'["\']?result["\']?\s*[:=]\s*["\']?(correct|almost|partial|incorrect)["\']?',
        r'["\']?answer["\']?\s*[:=]\s*["\']?(correct|almost|partial|incorrect)["\']?',
        r'["\']?prediction["\']?\s*[:=]\s*["\']?(correct|almost|partial|incorrect)["\']?',
        r'["\']?label["\']?\s*[:=]\s*["\']?(correct|almost|partial|incorrect)["\']?',
        r'["\']?category["\']?\s*[:=]\s*["\']?(correct|almost|partial|incorrect)["\']?',
    ]
    
    for pattern in kv_patterns:
        match = re.search(pattern, text_lower)
        if match:
            value = match.group(1).lower()
            if _is_valid_prediction(value):
                return {"response": value}
    
    # Pattern 2: Look for rubric markers in parentheses/brackets
    rubric_patterns = [
        (r'\(\s*correct\s*\)', "correct"),
        (r'\(\s*almost\s*\)', "almost"),
        (r'\(\s*partial\s*\)', "partial"),
        (r'\(\s*incorrect\s*\)', "incorrect"),
        (r'\[\s*correct\s*\]', "correct"),
        (r'\[\s*almost\s*\]', "almost"),
        (r'\[\s*partial\s*\]', "partial"),
        (r'\[\s*incorrect\s*\]', "incorrect"),
        # Credit markers
        (r'\(\s*full\s*credit\s*\)', "correct"),
        (r'\(\s*no\s*credit\s*\)', "incorrect"),
        (r'\(\s*half\s*credit\s*\)', "partial"),
        (r'\(\s*most\s*credit\s*\)', "almost"),
        (r'\(\s*some\s*credit\s*\)', "partial"),
        (r'\(\s*partial\s*credit\s*\)', "partial"),
    ]
    
    for pattern, label in rubric_patterns:
        if re.search(pattern, text_lower):
            return {"response": label}
    
    # Pattern 3: Look for standalone classification words
    standalone_patterns = [
        r'^\s*(correct|almost|partial|incorrect)\s*$',
        r'\bthe\s+(?:answer|classification|grade|result)\s+is\s+(correct|almost|partial|incorrect)',
        r'\bthis\s+is\s+(correct|almost|partial|incorrect)',
        r'\bclassified\s+as\s+(correct|almost|partial|incorrect)',
    ]
    
    for pattern in standalone_patterns:
        match = re.search(pattern, text_lower)
        if match:
            value = match.group(1).lower()
            if _is_valid_prediction(value):
                return {"response": value}
    
    return None


def _normalize_prediction(raw_value) -> str:
    """Normalize a raw prediction value to one of the valid labels."""
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
    
    raw_str = str(raw_value).lower().strip().strip('"\'')
    
    # Direct matches (exact)
    if raw_str in ["correct", "incorrect", "partial", "almost"]:
        return raw_str
    
    # Handle common variations for "correct"
    if any(term in raw_str for term in [
        "true", "yes", "right", "valid", "full credit", "full marks",
        "complete solution", "correct solution", "perfect", "flawless",
        "100% correct", "fully correct", "entirely correct", "totally correct",
        "no errors", "all steps correct", "mathematically correct",
        "full score", "max score", "maximum points", "award full"
    ]):
        return "correct"
    
    # Handle common variations for "incorrect"
    if any(term in raw_str for term in [
        "false", "no", "wrong", "invalid", "no credit", "zero credit",
        "no marks", "zero marks", "fundamentally wrong", "completely wrong",
        "totally wrong", "entirely wrong", "no solution", "no progress",
        "no meaningful work", "blank", "fail", "failed", "rejected",
        "award no", "give no", "not correct", "not right", "not valid"
    ]):
        return "incorrect"
    
    # Handle common variations for "partial"
    if any(term in raw_str for term in [
        "partial credit", "half credit", "some credit", "incomplete",
        "partial solution", "partially correct", "half correct",
        "some progress", "meaningful progress", "on the right track",
        "good start", "correct approach", "correct idea", "started correctly",
        "50% correct", "60% correct", "40% correct", "award half",
        "award some", "give half", "give some"
    ]):
        return "partial"
    
    # Handle common variations for "almost"
    if any(term in raw_str for term in [
        "almost correct", "nearly correct", "mostly correct", "minor error",
        "small error", "trivial error", "typo", "minor mistake", "almost there",
        "very close", "95% correct", "90% correct", "most credit",
        "most marks", "award most", "give most", "minor omission",
        "notational error", "sign error", "arithmetic error"
    ]):
        return "almost"
    
    # Check for "incorrect" before "correct" to avoid substring issues
    if "incorrect" in raw_str or "wrong" in raw_str or "invalid" in raw_str:
        return "incorrect"
    
    # Check for "correct" (but not negated)
    if "correct" in raw_str:
        # Check for negation
        idx = raw_str.find("correct")
        before = raw_str[max(0, idx-15):idx]
        if not any(neg in before for neg in ['not ', 'in', "isn't", 'isnt']):
            return "correct"
    
    if "partial" in raw_str:
        return "partial"
    
    if "almost" in raw_str:
        return "almost"
    
    return "unknown"


def _is_valid_prediction(prediction: str) -> bool:
    """Check if a prediction is one of the valid labels."""
    return prediction in ["correct", "incorrect", "partial", "almost"]


def _extract_prediction_from_text(text: str) -> str:
    """Extract prediction from raw text using multiple strategies."""
    if not text:
        return "unknown"
    
    text_lower = text.lower().strip()
    
    # Pre-process: Remove common prefixes
    text_lower = re.sub(r'^(?:here is|here\'s|the|this is|my|final|output|result|answer|classification|grade|response)[:\s]*', '', text_lower)
    text_lower = re.sub(r'^(?:json|the json)[:\s]*', '', text_lower)
    
    # Priority 0: Check for explicit rubric markers in the text
    rubric_markers = [
        (r'\(\s*correct\s*\)', "correct"),
        (r'\(\s*almost\s*\)', "almost"),
        (r'\(\s*partial\s*\)', "partial"),
        (r'\(\s*incorrect\s*\)', "incorrect"),
        (r'\[\s*correct\s*\]', "correct"),
        (r'\[\s*almost\s*\]', "almost"),
        (r'\[\s*partial\s*\]', "partial"),
        (r'\[\s*incorrect\s*\]', "incorrect"),
        # Credit markers
        (r'\(\s*full\s*credit\s*\)', "correct"),
        (r'\(\s*no\s*credit\s*\)', "incorrect"),
        (r'\(\s*half\s*credit\s*\)', "partial"),
        (r'\(\s*most\s*credit\s*\)', "almost"),
        (r'\(\s*some\s*credit\s*\)', "partial"),
    ]
    
    for pattern, label in rubric_markers:
        if re.search(pattern, text_lower):
            return label
    
    # Priority 1: Look for exact quoted labels in JSON-like context
    json_patterns = [
        r'"response"\s*:\s*"(correct|almost|partial|incorrect)"',
        r'"classification"\s*:\s*"(correct|almost|partial|incorrect)"',
        r'"grade"\s*:\s*"(correct|almost|partial|incorrect)"',
        r'"result"\s*:\s*"(correct|almost|partial|incorrect)"',
        r'"(correct|almost|partial|incorrect)"',
    ]
    for pattern in json_patterns:
        match = re.search(pattern, text_lower)
        if match:
            candidate = match.group(1).lower()
            if _is_valid_prediction(candidate):
                return candidate
    
    # Priority 2: Look for labels after colons or equals signs
    colon_patterns = [
        r'response\s*[=:]\s*(correct|almost|partial|incorrect)\b',
        r'classification\s*[=:]\s*(correct|almost|partial|incorrect)\b',
        r'grade\s*[=:]\s*(correct|almost|partial|incorrect)\b',
        r'result\s*[=:]\s*(correct|almost|partial|incorrect)\b',
    ]
    for pattern in colon_patterns:
        match = re.search(pattern, text_lower)
        if match:
            candidate = match.group(1).lower()
            if _is_valid_prediction(candidate):
                return candidate
    
    # Priority 3: Look for labels in specific contexts
    context_patterns = [
        r'\bis\s+(correct|almost|partial|incorrect)\b',
        r'\bwould\s+be\s+(correct|almost|partial|incorrect)\b',
        r'\bshould\s+be\s+(correct|almost|partial|incorrect)\b',
        r'\bclassified\s+as\s+(correct|almost|partial|incorrect)\b',
        r'\bthis\s+is\s+(correct|almost|partial|incorrect)\b',
        r'\btherefore\s+(?:it\s+is\s+)?(correct|almost|partial|incorrect)\b',
    ]
    for pattern in context_patterns:
        match = re.search(pattern, text_lower)
        if match:
            candidate = match.group(1).lower()
            if _is_valid_prediction(candidate):
                return candidate
    
    # Priority 4: Look for labels at the end of sentences or lines
    final_patterns = [
        r'(?:final\s+)?(?:answer|classification|grade|result)[\s:]+["\']?(correct|almost|partial|incorrect)["\']?[.!?]?\s*$',
        r'\b["\']?(correct|almost|partial|incorrect)["\']?\s*[.!?]?\s*$',
    ]
    for pattern in final_patterns:
        match = re.search(pattern, text_lower, re.MULTILINE)
        if match:
            candidate = match.group(1).lower()
            if _is_valid_prediction(candidate):
                return candidate
    
    # Priority 5: Check for standalone words at word boundaries
    # Check in order of specificity
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
        start = max(0, match.start() - 15)
        context = text_lower[start:match.start()]
        # Check if preceded by "not", "in", or other negating words
        negating_patterns = ['not ', 'in', "isn't", 'isnt']
        if not any(neg in context for neg in negating_patterns):
            return "correct"
    
    return "unknown"


def _parse_grading_guidelines(guidelines: str) -> dict:
    """Parse grading guidelines to extract rubric indicators."""
    result = {
        "has_partial": False,
        "has_almost": False,
        "has_correct": False,
        "has_incorrect": False,
        "primary_category": None,
        "confidence": 0.0,
    }
    
    if not guidelines:
        return result
    
    text_lower = guidelines.lower()
    
    # Check for explicit markers
    if re.search(r'\(\s*partial\s*\)', text_lower):
        result["has_partial"] = True
        result["primary_category"] = "partial"
        result["confidence"] = 0.9
    
    if re.search(r'\(\s*almost\s*\)', text_lower):
        result["has_almost"] = True
        result["primary_category"] = "almost"
        result["confidence"] = 0.9
    
    if re.search(r'\(\s*correct\s*\)', text_lower):
        result["has_correct"] = True
        result["primary_category"] = "correct"
        result["confidence"] = 0.9
    
    if re.search(r'\(\s*incorrect\s*\)', text_lower):
        result["has_incorrect"] = True
        result["primary_category"] = "incorrect"
        result["confidence"] = 0.9
    
    # Check for credit markers
    if re.search(r'\(\s*full\s*credit\s*\)', text_lower):
        result["has_correct"] = True
        result["primary_category"] = "correct"
        result["confidence"] = 0.95
    
    if re.search(r'\(\s*no\s*credit\s*\)', text_lower):
        result["has_incorrect"] = True
        result["primary_category"] = "incorrect"
        result["confidence"] = 0.95
    
    if re.search(r'\(\s*half\s*credit\s*\)', text_lower):
        result["has_partial"] = True
        result["primary_category"] = "partial"
        result["confidence"] = 0.9
    
    if re.search(r'\(\s*most\s*credit\s*\)', text_lower):
        result["has_almost"] = True
        result["primary_category"] = "almost"
        result["confidence"] = 0.9
    
    if re.search(r'\(\s*some\s*credit\s*\)', text_lower):
        result["has_partial"] = True
        result["primary_category"] = "partial"
        result["confidence"] = 0.85
    
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
        
        # Parse grading guidelines
        rubric = _parse_grading_guidelines(grading_guidelines)
        
        # Build rubric context
        rubric_context = ""
        if rubric["has_partial"]:
            rubric_context += "\n\n[RUBRIC: The grading guidelines contain '(Partial)' indicating partial credit.]"
        if rubric["has_almost"]:
            rubric_context += "\n\n[RUBRIC: The grading guidelines contain '(Almost)' indicating near-correctness with minor issues.]"
        if rubric["has_correct"]:
            rubric_context += "\n\n[RUBRIC: The grading guidelines contain '(Correct)' indicating full correctness.]"
        if rubric["has_incorrect"]:
            rubric_context += "\n\n[RUBRIC: The grading guidelines contain '(Incorrect)' indicating no credit.]"
        
        instruction = f"""You are an expert mathematical grader evaluating student solutions to competition mathematics problems.

Your task is to evaluate the student's answer and classify it into EXACTLY ONE of these four categories:

1. "correct" - The student's answer is fully correct, complete, and rigorous. Only use this if the solution is essentially flawless.

2. "almost" - The student's answer is nearly correct with only minor errors, typos, or small omissions that don't significantly affect the overall correctness.

3. "partial" - The student's answer has some correct elements, shows meaningful progress, but is incomplete or has significant gaps.

4. "incorrect" - The student's answer is fundamentally wrong, contains major errors, or shows no meaningful progress.

=== RUBRIC MARKERS (FOLLOW THESE EXPLICITLY) ===
The Grading Guidelines below contain EXPLICIT markers that indicate the intended classification:

- (Correct) or (Full Credit) → Use "correct"
- (Almost) or (Most Credit) → Use "almost"
- (Partial) or (Half Credit) or (Some Credit) → Use "partial"
- (Incorrect) or (No Credit) → Use "incorrect"

{rubric_context}

=== PROBLEM ===
{problem}

=== OFFICIAL SOLUTION ===
{solution}

=== STUDENT'S ANSWER ===
{student_answer}

=== GRADING GUIDELINES ===
{grading_guidelines}

=== RESPONSE FORMAT (STRICT) ===
You MUST respond ONLY in the following JSON format:

<json>
{{
    "response": "correct" | "almost" | "partial" | "incorrect"
}}
</json>

CRITICAL: 
1. The response field must contain EXACTLY one of: correct, almost, partial, incorrect
2. Follow the rubric markers in the grading guidelines - they are the strongest signal
3. DO NOT include any text before or after the JSON block
4. Use ONLY the <json> tags shown above"""

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

        # Extract prediction from response
        prediction = "unknown"
        
        try:
            last_message = msg_history[-1]["text"]
            self.log_fn(f"Raw response: {last_message[:500]}...")
            
            # Attempt 1: Try flexible JSON extraction
            extracted = _extract_json_flexible(last_message)
            if extracted and isinstance(extracted, dict):
                if "response" in extracted:
                    prediction = _normalize_prediction(extracted["response"])
                elif len(extracted) == 1:
                    prediction = _normalize_prediction(list(extracted.values())[0])
                else:
                    for key, value in extracted.items():
                        normalized = _normalize_prediction(value)
                        if _is_valid_prediction(normalized):
                            prediction = normalized
                            break
            
            # Attempt 2: If still unknown, try direct text extraction
            if not _is_valid_prediction(prediction):
                text_prediction = _extract_prediction_from_text(last_message)
                if _is_valid_prediction(text_prediction):
                    prediction = text_prediction
            
            # Fallback: Use rubric primary category if LLM response is unclear
            if not _is_valid_prediction(prediction):
                if rubric.get("primary_category") and rubric.get("confidence", 0) > 0.6:
                    prediction = rubric["primary_category"]
                    self.log_fn(f"Using rubric primary category as fallback: {prediction}")
            
            self.log_fn(f"Final prediction: {prediction}")
            
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
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
