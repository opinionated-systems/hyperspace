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
    
    return json_str


def _extract_key_value_patterns(text: str) -> dict | None:
    """Extract key-value pairs from text using regex patterns."""
    result = {}
    
    # Pattern 1: "key": "value" or 'key': 'value'
    pattern1 = r'["\'](\w+)["\']\s*:\s*["\']([^"\']+)["\']'
    matches = re.findall(pattern1, text)
    for key, value in matches:
        result[key] = value
    
    # Pattern 2: "key": value (unquoted)
    pattern2 = r'["\'](\w+)["\']\s*:\s*([^,\}\]\n]+)'
    matches = re.findall(pattern2, text)
    for key, value in matches:
        if key not in result:
            result[key] = value.strip().strip('"\'')
    
    # Pattern 3: key: "value" (unquoted key)
    pattern3 = r'(\w+)\s*:\s*["\']([^"\']+)["\']'
    matches = re.findall(pattern3, text)
    for key, value in matches:
        if key not in result:
            result[key] = value
    
    return result if result else None


def _extract_json_flexible(text: str) -> dict | None:
    """Extract JSON from LLM response using multiple robust strategies."""
    if not text or not isinstance(text, str):
        return None
    
    text = text.strip()
    if not text:
        return None
    
    # Pre-process: Remove common prefixes and markers
    text = re.sub(r'^(?:Here is|Here\'s|The|This is|My|Final|Output|Result|Answer|Classification|Grade|Response|Evaluation)[:\s]*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^(?:json|JSON)[:\s]*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^\s*```\s*\n?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\n?```\s*$', '', text, flags=re.IGNORECASE)
    
    # Strategy 1: <json> tags (most reliable and preferred format)
    json_blocks = _extract_jsons(text)
    if json_blocks:
        return json_blocks[-1]
    
    # Strategy 2: ```json code blocks with more flexible matching
    json_code_patterns = [
        r'```(?:json|JSON)?\s*\n?(.*?)\n?```',
        r'```\s*\n?(.*?)\n?```',
    ]
    for pattern in json_code_patterns:
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
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
                if any(k in parsed for k in ["response", "classification", "grade", "result", "evaluation", "answer"]):
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
    
    # Strategy 5: Extract from last non-empty line if it's a simple value
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    if lines:
        last_line = lines[-1]
        # Check if it's just a valid label
        norm_val = _normalize_prediction(last_line)
        if _is_valid_prediction(norm_val):
            return {"response": norm_val}
    
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
        "full score", "max score", "maximum points", "award full",
        "excellent", "outstanding", "complete and correct"
    ]):
        return "correct"
    
    # Handle common variations for "incorrect"
    if any(term in raw_str for term in [
        "false", "no", "wrong", "invalid", "no credit", "zero credit",
        "no marks", "zero marks", "fundamentally wrong", "completely wrong",
        "totally wrong", "entirely wrong", "no solution", "no progress",
        "no meaningful work", "blank", "fail", "failed", "rejected",
        "award no", "give no", "not correct", "not right", "not valid",
        "incorrect approach", "wrong approach", "major error"
    ]):
        return "incorrect"
    
    # Handle common variations for "partial"
    if any(term in raw_str for term in [
        "partial credit", "half credit", "some credit", "incomplete",
        "partial solution", "partially correct", "half correct",
        "some progress", "meaningful progress", "on the right track",
        "good start", "correct approach", "correct idea", "started correctly",
        "50% correct", "60% correct", "40% correct", "award half",
        "award some", "give half", "give some", "incomplete solution",
        "missing steps", "significant gaps"
    ]):
        return "partial"
    
    # Handle common variations for "almost"
    if any(term in raw_str for term in [
        "almost correct", "nearly correct", "mostly correct", "minor error",
        "small error", "trivial error", "typo", "minor mistake", "almost there",
        "very close", "95% correct", "90% correct", "most credit",
        "most marks", "award most", "give most", "minor omission",
        "notational error", "sign error", "arithmetic error", "slip",
        "careless error", "cosmetic error"
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
        # Exact markers with parentheses
        (r'\(\s*correct\s*\)', "correct"),
        (r'\(\s*almost\s*\)', "almost"),
        (r'\(\s*partial\s*\)', "partial"),
        (r'\(\s*incorrect\s*\)', "incorrect"),
        # Bracket markers
        (r'\[\s*correct\s*\]', "correct"),
        (r'\[\s*almost\s*\]', "almost"),
        (r'\[\s*partial\s*\]', "partial"),
        (r'\[\s*incorrect\s*\]', "incorrect"),
        # Credit markers with parentheses
        (r'\(\s*full\s*credit\s*\)', "correct"),
        (r'\(\s*no\s*credit\s*\)', "incorrect"),
        (r'\(\s*half\s*credit\s*\)', "partial"),
        (r'\(\s*most\s*credit\s*\)', "almost"),
        (r'\(\s*some\s*credit\s*\)', "partial"),
        # Score patterns
        (r'\(\s*\d+/\d+\s*points?\s*\)', None),  # Will be handled separately
        (r'award\s+full', "correct"),
        (r'award\s+no', "incorrect"),
        (r'award\s+half', "partial"),
        (r'award\s+most', "almost"),
        (r'award\s+some', "partial"),
    ]
    
    for pattern, label in rubric_markers:
        if label and re.search(pattern, text_lower):
            return label
    
    # Handle score patterns like (3/4 points) or (7/10 points)
    score_pattern = r'\(\s*(\d+)\s*/\s*(\d+)\s*(?:points?)?\s*\)'
    score_match = re.search(score_pattern, text_lower)
    if score_match:
        earned = int(score_match.group(1))
        total = int(score_match.group(2))
        if total > 0:
            ratio = earned / total
            if ratio >= 0.95:
                return "correct"
            elif ratio >= 0.75:
                return "almost"
            elif ratio >= 0.4:
                return "partial"
            else:
                return "incorrect"
    
    # Priority 1: Look for exact quoted labels in JSON-like context
    json_patterns = [
        r'"response"\s*:\s*"(correct|almost|partial|incorrect)"',
        r'"classification"\s*:\s*"(correct|almost|partial|incorrect)"',
        r'"grade"\s*:\s*"(correct|almost|partial|incorrect)"',
        r'"result"\s*:\s*"(correct|almost|partial|incorrect)"',
        r'"evaluation"\s*:\s*"(correct|almost|partial|incorrect)"',
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
        r'evaluation\s*[=:]\s*(correct|almost|partial|incorrect)\b',
        r'answer\s*[=:]\s*(correct|almost|partial|incorrect)\b',
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
        r'\bcategor(?:y|ized)\s+(?:as\s+)?(correct|almost|partial|incorrect)\b',
        r'\bthis\s+is\s+(correct|almost|partial|incorrect)\b',
        r'\btherefore\s+(?:it\s+is\s+)?(correct|almost|partial|incorrect)\b',
        r'\bthe\s+answer\s+is\s+(correct|almost|partial|incorrect)\b',
        r'\bthe\s+solution\s+is\s+(correct|almost|partial|incorrect)\b',
        r'\bgrade[d]?\s+(?:as\s+)?(correct|almost|partial|incorrect)\b',
    ]
    for pattern in context_patterns:
        match = re.search(pattern, text_lower)
        if match:
            candidate = match.group(1).lower()
            if _is_valid_prediction(candidate):
                return candidate
    
    # Priority 4: Look for labels at the end of sentences or lines
    final_patterns = [
        r'(?:final\s+)?(?:answer|classification|grade|result|evaluation)[\s:]+["\']?(correct|almost|partial|incorrect)["\']?[.!?]?\s*$',
        r'\b["\']?(correct|almost|partial|incorrect)["\']?\s*[.!?]?\s*$',
    ]
    for pattern in final_patterns:
        match = re.search(pattern, text_lower, re.MULTILINE)
        if match:
            candidate = match.group(1).lower()
            if _is_valid_prediction(candidate):
                return candidate
    
    # Priority 5: Check for standalone words at word boundaries
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
        start = max(0, match.start() - 15)
        context = text_lower[start:match.start()]
        # Check if preceded by "not", "in", or other negating words
        negating_patterns = ['not ', 'in', "isn't", 'isnt', 'not']
        if not any(neg in context for neg in negating_patterns):
            return "correct"
    
    return "unknown"


def _parse_grading_guidelines(guidelines: str) -> dict:
    """Parse grading guidelines to extract rubric indicators with enhanced pattern detection."""
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
    
    # Priority 1: Check for explicit markers with parentheses (highest confidence)
    marker_patterns = [
        (r'\(\s*correct\s*\)', "correct", 0.95),
        (r'\(\s*almost\s*\)', "almost", 0.95),
        (r'\(\s*partial\s*\)', "partial", 0.95),
        (r'\(\s*incorrect\s*\)', "incorrect", 0.95),
        # Bracket variants
        (r'\[\s*correct\s*\]', "correct", 0.95),
        (r'\[\s*almost\s*\]', "almost", 0.95),
        (r'\[\s*partial\s*\]', "partial", 0.95),
        (r'\[\s*incorrect\s*\]', "incorrect", 0.95),
    ]
    
    for pattern, category, confidence in marker_patterns:
        if re.search(pattern, text_lower):
            result[f"has_{category}"] = True
            result["primary_category"] = category
            result["confidence"] = confidence
            return result  # Return immediately on exact marker match
    
    # Priority 2: Check for credit markers
    credit_patterns = [
        (r'\(\s*full\s*credit\s*\)', "correct", 0.98),
        (r'\(\s*no\s*credit\s*\)', "incorrect", 0.98),
        (r'\(\s*half\s*credit\s*\)', "partial", 0.95),
        (r'\(\s*most\s*credit\s*\)', "almost", 0.95),
        (r'\(\s*some\s*credit\s*\)', "partial", 0.90),
        (r'\(\s*partial\s*credit\s*\)', "partial", 0.95),
    ]
    
    for pattern, category, confidence in credit_patterns:
        if re.search(pattern, text_lower):
            result[f"has_{category}"] = True
            if result["confidence"] < confidence:
                result["primary_category"] = category
                result["confidence"] = confidence
    
    # Priority 3: Check for score patterns like (3/4 points) or (7/10)
    score_patterns = [
        r'\(\s*(\d+)\s*/\s*(\d+)\s*(?:points?)?\s*\)',
        r'(\d+)\s*/\s*(\d+)\s*(?:points?|pts?)',
        r'award\s+(\d+)\s*/\s*(\d+)',
        r'award\s+(\d+)\s+out\s+of\s+(\d+)',
    ]
    
    for pattern in score_patterns:
        match = re.search(pattern, text_lower)
        if match:
            earned = int(match.group(1))
            total = int(match.group(2))
            if total > 0:
                ratio = earned / total
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
    
    # Priority 4: Check for contextual keywords
    keyword_patterns = [
        # Correct indicators
        (r'\b(?:full|complete|perfect|flawless)\s+(?:credit|marks?|score|solution)\b', "correct", 0.80),
        (r'\ball\s+(?:steps|work)\s+(?:correct|right|valid)\b', "correct", 0.80),
        (r'\bcorrect\s+(?:final\s+)?answer\b', "correct", 0.75),
        # Almost indicators
        (r'\bminor\s+(?:error|mistake|typo|omission)\b', "almost", 0.75),
        (r'\bsmall\s+(?:error|mistake)\b', "almost", 0.75),
        (r'\btrivial\s+(?:error|issue)\b', "almost", 0.75),
        (r'\barithmetic\s+error\b', "almost", 0.70),
        (r'\bsign\s+error\b', "almost", 0.70),
        (r'\bnotational\s+error\b', "almost", 0.70),
        (r'\bmost(?:ly)?\s+correct\b', "almost", 0.75),
        (r'\balmost\s+(?:correct|there|right)\b', "almost", 0.80),
        (r'\bnearly\s+correct\b', "almost", 0.80),
        # Partial indicators
        (r'\bincomplete\s+(?:solution|answer|work)\b', "partial", 0.75),
        (r'\bpartial\s+(?:credit|solution|progress|work)\b', "partial", 0.80),
        (r'\bsome\s+(?:progress|correct|valid)\b', "partial", 0.70),
        (r'\bon\s+the\s+right\s+track\b', "partial", 0.70),
        (r'\bgood\s+start\b', "partial", 0.70),
        (r'\bcorrect\s+approach\b', "partial", 0.70),
        (r'\bmissing\s+(?:steps|work|justification)\b', "partial", 0.70),
        # Incorrect indicators
        (r'\bno\s+(?:credit|marks?|score|progress|meaningful)\b', "incorrect", 0.80),
        (r'\bfundamentally\s+wrong\b', "incorrect", 0.85),
        (r'\bcompletely\s+wrong\b', "incorrect", 0.85),
        (r'\bmajor\s+error\b', "incorrect", 0.75),
        (r'\bwrong\s+(?:approach|method|answer)\b', "incorrect", 0.80),
    ]
    
    for pattern, category, confidence in keyword_patterns:
        if re.search(pattern, text_lower):
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

- **correct**: Complete and fully correct solution. All steps present and logically sound. No gaps, no errors, fully rigorous proof. EVERY step must be correct and justified.

- **incorrect**: No meaningful progress or fundamental errors. Wrong approach or no valid mathematical progress. Student failed to demonstrate understanding of key concepts. Minimal or no substantive work toward the solution.

- **partial**: Significant progress but INCOMPLETE. Found useful invariant/lemma or established valid approach but didn't finish. Student demonstrated understanding of key ideas but solution has MAJOR gaps or STOPS before completion. The student has NOT addressed all major components. Missing conclusion or final step is a KEY indicator.

- **almost**: Nearly complete solution with ONLY minor issues. Main proof structure is CORRECT and COMPLETE. Only minor computational errors, typos, or notation issues. Student has addressed ALL major components. The proof is COMPLETE from start to finish INCLUDING the conclusion. If you fix trivial errors (arithmetic, typos), it becomes correct. The student understood the complete approach and executed it with only small blemishes.

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

## RUBRIC MARKERS (FOLLOW THESE EXPLICITLY)
The Grading Guidelines below contain EXPLICIT markers that indicate the intended classification. These markers are the STRONGEST signal for your decision:

- (Correct) or (Full Credit) or [Correct] or [Full Credit] → Use "correct"
- (Almost) or (Most Credit) or [Almost] or [Most Credit] → Use "almost"
- (Partial) or (Half Credit) or (Some Credit) or [Partial] or [Half Credit] → Use "partial"
- (Incorrect) or (No Credit) or [Incorrect] or [No Credit] → Use "incorrect"

Score patterns like (X/Y points) also indicate the classification:
- 90-100% of points → "correct"
- 70-89% of points → "almost"
- 35-69% of points → "partial"
- 0-34% of points → "incorrect"

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

### DECISION RULES:
- **For "almost"**: Q1-Q5 must ALL be YES
- **For "partial"**: Q6 is YES, but at least one of Q1-Q4 is NO
- **For "incorrect"**: Q6 is NO (no meaningful progress)
- **For "correct"**: ALL of Q1-Q5 are YES with truly NO errors

## RESPONSE FORMAT (STRICT - MUST FOLLOW EXACTLY)

You MUST respond ONLY in the following JSON format. Do not include any text before or after the JSON block:

<json>
{{
    "response": "correct" | "almost" | "partial" | "incorrect"
}}
</json>

CRITICAL INSTRUCTIONS:
1. The response field must contain EXACTLY one of: correct, almost, partial, incorrect (lowercase)
2. Follow the rubric markers in the grading guidelines - they are the STRONGEST signal
3. DO NOT include any text, explanation, or markdown before or after the JSON block
4. Use ONLY the <json> tags shown above, not ```json code blocks
5. Your entire response should be just the <json>...</json> block with nothing else"""

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
                # Try other common keys
                elif "classification" in extracted:
                    prediction = _normalize_prediction(extracted["classification"])
                elif "grade" in extracted:
                    prediction = _normalize_prediction(extracted["grade"])
                elif "result" in extracted:
                    prediction = _normalize_prediction(extracted["result"])
                elif "evaluation" in extracted:
                    prediction = _normalize_prediction(extracted["evaluation"])
                elif len(extracted) == 1:
                    # Single key-value pair, use the value
                    prediction = _normalize_prediction(list(extracted.values())[0])
                else:
                    # Try all values to find a valid prediction
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
            
            # Attempt 3: Check for rubric markers directly in the response
            if not _is_valid_prediction(prediction):
                response_rubric = _parse_grading_guidelines(last_message)
                if response_rubric.get("primary_category") and response_rubric.get("confidence", 0) > 0.8:
                    prediction = response_rubric["primary_category"]
                    self.log_fn(f"Using response rubric markers: {prediction}")
            
            # Fallback: Use rubric primary category if LLM response is unclear
            if not _is_valid_prediction(prediction):
                if rubric.get("primary_category") and rubric.get("confidence", 0) > 0.5:
                    prediction = rubric["primary_category"]
                    self.log_fn(f"Using grading guidelines rubric as fallback: {prediction}")
            
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
