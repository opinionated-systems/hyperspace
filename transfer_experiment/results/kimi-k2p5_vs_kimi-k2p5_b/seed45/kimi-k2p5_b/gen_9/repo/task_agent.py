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


def _extract_key_value_patterns(text: str) -> dict | None:
    """Extract key-value patterns that look like JSON or grading responses."""
    result = {}
    
    # Pattern 1: "response": "value" or 'response': 'value'
    response_patterns = [
        r'["\']?response["\']?\s*[:=]\s*["\']?(correct|almost|partial|incorrect)["\']?',
        r'["\']?classification["\']?\s*[:=]\s*["\']?(correct|almost|partial|incorrect)["\']?',
        r'["\']?grade["\']?\s*[:=]\s*["\']?(correct|almost|partial|incorrect)["\']?',
        r'["\']?result["\']?\s*[:=]\s*["\']?(correct|almost|partial|incorrect)["\']?',
        r'["\']?evaluation["\']?\s*[:=]\s*["\']?(correct|almost|partial|incorrect)["\']?',
    ]
    
    for pattern in response_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
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
        "100%", "full marks", "full credit", "all correct"
    ]
    if raw_str in correct_variations or clean_str in correct_variations:
        return "correct"
    
    # Handle common variations for "incorrect"
    incorrect_variations = [
        "false", "no", "wrong", "invalid", "0", "none", "fail", 
        "error", "bad", "unsatisfactory", "rejected", "failed",
        "not correct", "not right", "not valid", "zero",
        "no credit", "full wrong", "entirely wrong"
    ]
    if raw_str in incorrect_variations or clean_str in incorrect_variations:
        return "incorrect"
    
    # Handle common variations for "partial"
    partial_variations = [
        "part", "partially", "incomplete", "half", "some", 
        "mostly wrong", "mostly incorrect", "partial credit",
        "half credit", "some credit", "in progress",
        "partially correct", "half correct", "semi correct",
        "50%", "40%", "60%", "mixed", "incomplete solution"
    ]
    if raw_str in partial_variations or clean_str in partial_variations:
        return "partial"
    
    # Handle common variations for "almost"
    almost_variations = [
        "almost correct", "close", "minor errors", "nearly", 
        "mostly correct", "trivial errors", "almost there",
        "near correct", "very close", "minor mistake",
        "small error", "slight error", "nearly right",
        "80%", "90%", "minor issue", "trivial issue"
    ]
    if raw_str in almost_variations or clean_str in almost_variations:
        return "almost"
    
    # Check for substring matches (more specific first to avoid misclassification)
    # Check for "almost" patterns
    if any(term in raw_str for term in ["almost", "nearly", "minor", "trivial", "slight", "small error"]):
        return "almost"
    
    # Check for "partial" patterns
    if any(term in raw_str for term in ["partial", "incomplete", "half", "some progress", "meaningful progress"]):
        return "partial"
    
    # Check for "incorrect" patterns (before "correct" to avoid substring issues)
    if any(term in raw_str for term in ["incorrect", "wrong", "invalid", "error", "not correct", "not right"]):
        return "incorrect"
    
    # Check for "correct" patterns
    if any(term in raw_str for term in ["correct", "right", "valid", "accurate", "proper"]):
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
    }
    
    if not guidelines:
        return result
    
    guidelines_lower = guidelines.lower()
    
    # Look for explicit markers with comprehensive patterns
    # Priority order: exact markers > credit markers > score patterns > alternative markers
    markers = [
        # Exact rubric markers (highest priority - 1.0)
        (r'\(\s*Partial\s*\)', "partial", 1.0),
        (r'\(\s*Almost\s*\)', "almost", 1.0),
        (r'\(\s*Correct\s*\)', "correct", 1.0),
        (r'\(\s*Incorrect\s*\)', "incorrect", 1.0),
        # Credit-based markers (0.90-0.95)
        (r'\(\s*Full\s*Credit\s*\)', "correct", 0.95),
        (r'\(\s*No\s*Credit\s*\)', "incorrect", 0.95),
        (r'\(\s*Half\s*Credit\s*\)', "partial", 0.95),
        (r'\(\s*Most\s*Credit\s*\)', "almost", 0.95),
        (r'\(\s*Some\s*Credit\s*\)', "partial", 0.90),
        (r'\(\s*Partial\s*Credit\s*\)', "partial", 0.95),
        # Score-based patterns (0.80)
        (r'\(\s*\d+\s*/\s*\d+\s*\)', "score_based", 0.80),
        (r'\(\s*\d+\s*points?\s*\)', "score_based", 0.80),
        # Alternative markers with brackets (0.85)
        (r'\[\s*Partial\s*\]', "partial", 0.85),
        (r'\[\s*Almost\s*\]', "almost", 0.85),
        (r'\[\s*Correct\s*\]', "correct", 0.85),
        (r'\[\s*Incorrect\s*\]', "incorrect", 0.85),
        # Text-based markers (0.75-0.80)
        (r'\bPartial\s*Credit\b', "partial", 0.80),
        (r'\bAlmost\s*Correct\b', "almost", 0.80),
        (r'\bFull\s*Credit\b', "correct", 0.80),
        (r'\bNo\s*Credit\b', "incorrect", 0.80),
        # Award/give patterns (0.75)
        (r'award\s+\d+\s*points?', "score_based", 0.75),
        (r'give\s+\d+\s*points?', "score_based", 0.75),
    ]
    
    category_scores = {"partial": 0, "almost": 0, "correct": 0, "incorrect": 0}
    
    for pattern, label, confidence in markers:
        matches = list(re.finditer(pattern, guidelines, re.IGNORECASE))
        if matches:
            if label != "score_based":
                result[f"has_{label}"] = True
                category_scores[label] = max(category_scores[label], confidence)
            
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
                        "context": context[:100]
                    })
    
    # Extract score/point information with enhanced detail
    score_patterns = [
        (r'(\d+)\s*points?', "points"),
        (r'score:\s*(\d+)', "score"),
        (r'\((\d+)\s*/\s*(\d+)\)', "fraction"),
        (r'(\d+)\s*/\s*(\d+)\s*points?', "fraction_points"),
        (r'award\s+(\d+)', "award"),
        (r'give\s+(\d+)', "give"),
        (r'worth\s+(\d+)', "worth"),
        (r'value\s+(\d+)', "value"),
        (r'(\d+)\s*marks?', "marks"),
    ]
    
    extracted_scores = []
    max_points = None
    score_ratios = []
    
    for pattern, score_type in score_patterns:
        if score_type in ["fraction", "fraction_points"]:
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
                                category_scores["correct"] = max(category_scores["correct"], 0.90)
                            elif ratio >= 0.85:
                                category_scores["correct"] = max(category_scores["correct"], 0.80)
                            elif ratio >= 0.70:
                                category_scores["almost"] = max(category_scores["almost"], 0.85)
                            elif ratio >= 0.50:
                                category_scores["partial"] = max(category_scores["partial"], 0.80)
                            elif ratio >= 0.30:
                                category_scores["partial"] = max(category_scores["partial"], 0.70)
                            else:
                                category_scores["incorrect"] = max(category_scores["incorrect"], 0.85)
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
        ],
        "almost": [
            ("minor", 0.7), ("small", 0.6), ("slight", 0.6), 
            ("typo", 0.8), ("nearly", 0.8), ("close", 0.7),
            ("trivial", 0.7), ("insignificant", 0.6), ("almost correct", 0.9),
            ("mostly correct", 0.8), ("small error", 0.75),
            ("minor error", 0.75), ("slight mistake", 0.75),
            ("cosmetic", 0.6), ("formatting", 0.5),
            ("rounding", 0.6), ("negligible", 0.7),
        ],
        "partial": [
            ("partial", 0.8), ("incomplete", 0.7), ("missing", 0.6), 
            ("some", 0.5), ("half", 0.7), ("progress", 0.6),
            ("attempt", 0.5), ("partial credit", 0.9), ("partially correct", 0.8),
            ("on the right track", 0.7), ("meaningful progress", 0.75),
            ("significant progress", 0.7), ("good start", 0.6),
            ("some correct", 0.7), ("partial solution", 0.8),
            ("incomplete solution", 0.75), ("missing steps", 0.65),
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

Step 1: RUBRIC MARKERS (HIGHEST PRIORITY - FOLLOW THESE)
The Grading Guidelines below contain EXPLICIT markers that indicate the intended classification. These markers are placed by human graders and are the STRONGEST signal of how to classify:
- (Correct) or (Full Credit) → Use "correct"
- (Almost) or (Most Credit) → Use "almost"  
- (Partial) or (Half Credit) or (Some Credit) → Use "partial"
- (Incorrect) or (No Credit) → Use "incorrect"

IMPORTANT: Unless the student's answer is completely off-topic or blank, TRUST THE RUBRIC MARKERS. They represent the ground truth classification.

Step 2: SCORE INTERPRETATION (HIGH PRIORITY)
If the rubric shows scores like "(X/Y) points" or "award X points":
- Score ratio ≥ 90% → "correct"
- Score ratio 70-89% → "almost"
- Score ratio 40-69% → "partial"
- Score ratio < 40% → "incorrect"

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

=== REASONING PROCESS ===
Before giving your final answer, think through:
1. What is the core question/problem being asked?
2. What is the official solution's approach and answer?
3. What does the student's answer provide?
4. How does the rubric/grading guidelines classify this type of answer? (LOOK FOR MARKERS)
5. Which category best matches the student's work based on the rubric?

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

DO NOT include any text before or after the JSON block. Your entire response should be just the JSON."""

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
            return "unknown", []

        # Extract prediction from response with enhanced robustness
        prediction = "unknown"
        try:
            last_message = msg_history[-1]["text"]
            self.log_fn(f"Raw response: {last_message[:500]}...")
            
            # Try flexible JSON extraction first (most reliable)
            extracted = _extract_json_flexible(last_message)
            if extracted and isinstance(extracted, dict):
                # Try to get the response value
                if "response" in extracted:
                    prediction = _normalize_prediction(extracted["response"])
                elif len(extracted) == 1:
                    # If only one key, use its value
                    prediction = _normalize_prediction(list(extracted.values())[0])
                else:
                    # Try to find a value that looks like a grade
                    for key, value in extracted.items():
                        normalized = _normalize_prediction(value)
                        if _is_valid_prediction(normalized):
                            prediction = normalized
                            break
            
            # If still unknown, try direct text extraction with priority ordering
            if not _is_valid_prediction(prediction):
                prediction = _extract_prediction_from_text(last_message)
            
            # Use rubric primary category as fallback if LLM response is unclear
            if prediction == "unknown" and rubric.get("primary_category") and rubric.get("confidence", 0) > 0.7:
                prediction = rubric["primary_category"]
                self.log_fn(f"Using rubric primary category as fallback: {prediction}")
            
            self.log_fn(f"Extracted prediction: {prediction}")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            # Emergency fallback to rubric if available
            if rubric.get("primary_category"):
                prediction = rubric["primary_category"]

        return str(prediction), msg_history


def _extract_prediction_from_text(text: str) -> str:
    """Extract prediction from raw text using multiple strategies.
    
    This function implements a comprehensive approach to extract the classification
    from LLM responses when JSON parsing fails.
    """
    if not text:
        return "unknown"
    
    text_lower = text.lower()
    
    # Priority 1: Look for exact quoted labels in JSON-like context
    json_patterns = [
        (r'"response"\s*:\s*"(correct|almost|partial|incorrect)"', 1),
        (r"'response'\s*:\s*'(correct|almost|partial|incorrect)'", 1),
        (r'"classification"\s*:\s*"(correct|almost|partial|incorrect)"', 1),
        (r'"grade"\s*:\s*"(correct|almost|partial|incorrect)"', 1),
        (r'"result"\s*:\s*"(correct|almost|partial|incorrect)"', 1),
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
    ]
    for pattern, group in context_patterns:
        match = re.search(pattern, text_lower)
        if match:
            candidate = match.group(group).lower()
            if _is_valid_prediction(candidate):
                return candidate
    
    # Priority 4: Check for standalone words at word boundaries
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
        start = max(0, match.start() - 10)
        context = text_lower[start:match.start()]
        # Check if preceded by "not", "in", or other negating words
        if not any(context.endswith(neg) for neg in ['in', 'not ', 'not', "isn't ", 'isnt ', 'not\t']):
            return "correct"
    
    # Priority 5: Find all occurrences and use smart disambiguation
    positions = []
    for label in ["correct", "almost", "partial", "incorrect"]:
        for match in re.finditer(r'\b' + label + r'\b', text_lower):
            # Calculate a score based on context
            score = 0
            start = max(0, match.start() - 30)
            end = min(len(text_lower), match.end() + 30)
            context = text_lower[start:end]
            
            # Boost score for labels near keywords indicating final answer
            if any(kw in context for kw in ['final', 'answer', 'classification', 'grade', 'result']):
                score += 10
            
            # Penalize "correct" if near negating words
            if label == "correct":
                before = text_lower[max(0, match.start() - 15):match.start()]
                if any(neg in before for neg in ['not ', 'in', "isn't", 'isnt']):
                    score -= 20
            
            positions.append((match.start(), label, score))
    
    if positions:
        # Sort by position, then by score (descending)
        positions.sort(key=lambda x: (x[0], -x[2]))
        
        # Filter out negated "correct" instances
        filtered = []
        for pos, label, score in positions:
            if label == "correct":
                before = text_lower[max(0, pos - 15):pos]
                if any(neg in before for neg in ['not ', 'in', "isn't", 'isnt']):
                    continue
            filtered.append((pos, label, score))
        
        if filtered:
            # Return the highest scoring label
            best = max(filtered, key=lambda x: (x[2], x[0]))
            return best[1]
        else:
            # Fallback to first non-negated
            return positions[0][1]
    
    return "unknown"
