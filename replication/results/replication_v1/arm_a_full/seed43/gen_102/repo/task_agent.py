"""
Task agent: solves a given task with chain-of-thought reasoning.

Reimplemented from facebookresearch/HyperAgents task_agent.py.
Same interface, same JSON output format, same extraction logic.

This is the INITIAL task agent. The meta agent modifies this file
during self-improvement. The evaluation harness loads whatever
task_agent.py exists at the agent's repo path.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time

from agent.llm_client import get_response_from_llm, EVAL_MODEL

logger = logging.getLogger(__name__)

# Cache statistics for monitoring (since lru_cache doesn't expose these directly)
_cache_hits = 0
_cache_misses = 0


def _get_cache_key(msg: str, model: str, temperature: float) -> str:
    """Generate a cache key from message parameters."""
    key_data = f"{model}:{temperature}:{msg}"
    return hashlib.sha256(key_data.encode()).hexdigest()[:32]


def get_cache_stats() -> dict:
    """Get cache statistics for monitoring."""
    total = _cache_hits + _cache_misses
    hit_rate = _cache_hits / total if total > 0 else 0
    return {
        "hits": _cache_hits,
        "misses": _cache_misses,
        "hit_rate": hit_rate,
        "size": len(_llm_response_cache),
        "max_size": _MAX_CACHE_SIZE,
    }


def clear_cache() -> None:
    """Clear the LLM response cache."""
    global _cache_hits, _cache_misses, _llm_response_cache
    _cache_hits = 0
    _cache_misses = 0
    _llm_response_cache.clear()
    logger.info("LLM response cache cleared")


# Simple LRU cache using OrderedDict-like behavior with plain dict
# Python 3.7+ dicts maintain insertion order, so we can use pop/insert for LRU
_llm_response_cache: dict[str, tuple[str, list, dict]] = {}
_MAX_CACHE_SIZE = 1000


def _store_in_cache(cache_key: str, value: tuple[str, list, dict]) -> None:
    """Store a value in the cache with LRU eviction."""
    # Simple LRU: if at capacity, remove oldest item (first in dict)
    if len(_llm_response_cache) >= _MAX_CACHE_SIZE:
        oldest_key = next(iter(_llm_response_cache))
        del _llm_response_cache[oldest_key]
        logger.debug(f"Evicted cache entry: {oldest_key[:16]}...")
    _llm_response_cache[cache_key] = value


def _get_cached_response(cache_key: str) -> tuple[str, list, dict] | None:
    """Get a value from the cache, updating LRU order."""
    if cache_key in _llm_response_cache:
        # Move to end (most recently used) by re-inserting
        value = _llm_response_cache.pop(cache_key)
        _llm_response_cache[cache_key] = value
        return value
    return None


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    Also handles nested JSON objects and markdown code blocks within the content.
    Includes improved handling for malformed tags, nested structures, and edge cases.
    """
    if not text or not isinstance(text, str):
        logger.warning("Invalid input to _extract_jsons: empty or non-string text")
        return None
        
    results = []
    search_from = 0
    max_iterations = 100  # Safety limit to prevent infinite loops
    iterations = 0
    extraction_attempts = 0
    successful_extractions = 0
    
    logger.debug(f"Starting JSON extraction from text of length {len(text)}")
    
    while iterations < max_iterations:
        iterations += 1
        start = text.find("<json>", search_from)
        if start == -1:
            break
        end = text.find("</json>", start)
        if end == -1:
            # Malformed: opening tag without closing tag - try to extract anyway
            inner = text[start + 6:].strip()
            # Only process if there's reasonable content
            if len(inner) > 10 and ('{' in inner or '[' in inner):
                extraction_attempts += 1
                parsed = _parse_json_with_fallback(inner)
                if parsed is not None:
                    results.append(parsed)
                    successful_extractions += 1
                    logger.debug(f"Successfully parsed malformed <json> block at position {start}")
                else:
                    logger.debug(f"Failed to parse malformed <json> block at position {start}")
            search_from = start + 6  # Move past this tag to continue searching
            continue
        
        inner = text[start + 6:end].strip()
        search_from = end + 7
        
        # Skip empty blocks
        if not inner:
            logger.debug(f"Skipping empty <json> block at position {start}")
            continue
        
        # Try to parse the inner content as JSON
        extraction_attempts += 1
        parsed = _parse_json_with_fallback(inner)
        if parsed is not None:
            results.append(parsed)
            successful_extractions += 1
            logger.debug(f"Successfully parsed <json> block at position {start}")
        else:
            logger.debug(f"Failed to parse <json> block at position {start}")
    
    if iterations >= max_iterations:
        logger.warning(f"JSON extraction hit max iterations ({max_iterations}), possible malformed input")
    
    # Also try to extract JSON from markdown code blocks if no <json> tags found
    # This handles LLMs that output JSON in ```json blocks instead of <json> tags
    if not results:
        logger.debug("No <json> tags found or all failed, trying markdown code blocks")
        markdown_json = _extract_json_from_markdown_blocks(text)
        if markdown_json:
            results.extend(markdown_json)
            successful_extractions += len(markdown_json)
            logger.debug(f"Extracted {len(markdown_json)} JSON objects from markdown blocks")
    
    logger.debug(f"JSON extraction complete: {successful_extractions}/{extraction_attempts} successful, "
                f"found {len(results)} total objects in {iterations} iterations")
    return results or None


def _extract_json_from_markdown_blocks(text: str) -> list[dict] | None:
    """Extract JSON objects from markdown code blocks (```json ... ```).
    
    Fallback for when LLMs output JSON in markdown format instead of <json> tags.
    """
    import re
    results = []
    
    # Find all markdown JSON code blocks
    pattern = r'```json\s*(.*?)\s*```'
    matches = re.finditer(pattern, text, re.DOTALL)
    
    for match in matches:
        inner = match.group(1).strip()
        if inner:
            parsed = _parse_json_with_fallback(inner)
            if parsed is not None:
                results.append(parsed)
    
    return results or None


def _parse_json_with_fallback(text: str) -> dict | None:
    """Try to parse JSON with multiple fallback strategies.
    
    1. Direct JSON parsing
    2. Extract from markdown code blocks
    3. Extract from raw JSON objects with brace matching
    4. Handle common LLM output issues (trailing commas, comments, etc.)
    """
    # Strategy 1: Direct parsing
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    
    # Strategy 2: Extract from markdown code blocks
    code_block_patterns = [
        r'```json\s*(.*?)\s*```',
        r'```\s*(\{.*?\})\s*```',
        r'`(\{.*?)`',
    ]
    for pattern in code_block_patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                continue
    
    # Strategy 3: Find JSON objects by brace matching
    result = _extract_json_by_brace_matching(text)
    if result is not None:
        return result
    
    # Strategy 4: Try to fix common LLM JSON issues
    return _fix_and_parse_json(text)


def _fix_and_parse_json(text: str) -> dict | None:
    """Try to fix common LLM JSON output issues and parse.
    
    Handles:
    - Trailing commas before closing braces/brackets
    - Single quotes instead of double quotes
    - Unquoted keys
    - Comments (// and /* */)
    - Control characters
    - Unicode escape sequences
    - Newlines in strings
    """
    import re
    
    if not text or not isinstance(text, str):
        return None
    
    # Clean up the text
    cleaned = text.strip()
    
    # Remove control characters except newlines and tabs
    cleaned = ''.join(ch for ch in cleaned if ch == '\n' or ch == '\t' or ord(ch) >= 32)
    
    # Remove comments (both // and /* */)
    cleaned = re.sub(r'//[^\n]*', '', cleaned)
    cleaned = re.sub(r'/\*.*?\*/', '', cleaned, flags=re.DOTALL)
    
    # Try multiple fix strategies
    fixes = [
        # Fix 1: Remove trailing commas before } or ]
        lambda t: re.sub(r',(\s*[}\]])', r'\1', t),
        # Fix 2: Convert single quotes to double quotes (carefully)
        lambda t: re.sub(r"(?<!\\)'([^']*?)'(?=\s*:)", r'"\1"', t),  # Keys
        lambda t: re.sub(r":\s*'([^']*?)'(?=\s*[,}\]])", lambda m: f': "{m.group(1)}"', t),  # String values
        # Fix 3: Add quotes to unquoted keys
        lambda t: re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', t),
        # Fix 4: Escape newlines in strings
        lambda t: re.sub(r'("[^"]*\n[^"]*")', lambda m: m.group(1).replace('\n', '\\n'), t),
        # Fix 5: Remove BOM if present
        lambda t: t.lstrip('\ufeff'),
    ]
    
    for fix in fixes:
        try:
            fixed = fix(cleaned)
            result = json.loads(fixed)
            if isinstance(result, dict):
                return result
        except (json.JSONDecodeError, ValueError):
            continue
    
    # Try all fixes combined
    try:
        fixed = cleaned
        for fix in fixes:
            fixed = fix(fixed)
        result = json.loads(fixed)
        if isinstance(result, dict):
            return result
    except (json.JSONDecodeError, ValueError):
        pass
    
    # Last resort: try to extract just the first valid JSON object
    try:
        # Find the first { and last }
        start = cleaned.find('{')
        end = cleaned.rfind('}')
        if start != -1 and end != -1 and end > start:
            candidate = cleaned[start:end+1]
            result = json.loads(candidate)
            if isinstance(result, dict):
                return result
    except (json.JSONDecodeError, ValueError):
        pass
    
    return None


def _extract_json_by_brace_matching(text: str) -> dict | None:
    """Extract JSON object by matching braces with proper nesting.
    
    Handles strings, escape sequences, and nested objects correctly.
    Returns the first valid JSON object with expected keys, or the first
    valid JSON if no expected keys are found.
    """
    json_candidates = []
    
    # Find all potential JSON object starts
    for match in re.finditer(r'\{\s*"', text):
        start = match.start()
        # Try to find the matching end brace
        brace_count = 0
        in_string = False
        escape_next = False
        
        for i in range(start, len(text)):
            char = text[i]
            
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
                        # Found a complete JSON object
                        json_candidates.append(text[start:i+1])
                        break
    
    # Try to parse each candidate, preferring ones with expected keys
    expected_keys = ["response", "grade", "score", "answer", "reasoning", "result"]
    for candidate in json_candidates:
        # Try direct parsing first
        try:
            parsed = json.loads(candidate)
            # Prioritize candidates with expected keys
            if any(key in parsed for key in expected_keys):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass
        
        # Try with fixes
        parsed = _fix_and_parse_json(candidate)
        if isinstance(parsed, dict) and any(key in parsed for key in expected_keys):
            return parsed
    
    # If no prioritized candidate found, return the first valid one
    for candidate in json_candidates:
        # Try direct parsing first
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            pass
        
        # Try with fixes
        parsed = _fix_and_parse_json(candidate)
        if isinstance(parsed, dict):
            return parsed
    
    return None


def _extract_json_fallback(text: str) -> dict | None:
    """Fallback JSON extraction for non-tagged JSON or markdown code blocks.

    Tries to find JSON in markdown code blocks or raw JSON objects.
    Uses a robust approach to handle nested braces and common LLM JSON issues.
    """
    # First try markdown code blocks
    patterns = [
        r'```json\s*(.*?)\s*```',  # Markdown JSON blocks
        r'```\s*(\{.*?\})\s*```',  # Generic code blocks with JSON
        r'`(\{.*?)`',  # Inline code with JSON
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            # Try direct parsing first
            try:
                return json.loads(match.strip())
            except json.JSONDecodeError:
                pass
            # Try with fixes
            parsed = _fix_and_parse_json(match.strip())
            if parsed is not None:
                return parsed
    
    # Try to find JSON objects by matching braces with proper nesting
    result = _extract_json_by_brace_matching(text)
    if result is not None:
        return result
    
    # Last resort: try to fix and parse the entire text
    return _fix_and_parse_json(text)


def _validate_and_normalize_prediction(prediction: str, grading_guidelines: str) -> str:
    """Validate and normalize the prediction based on grading guidelines.
    
    Ensures the prediction matches expected format from grading guidelines.
    Handles various edge cases and normalizes common variations.
    """
    if not prediction or prediction.strip() == "":
        return "None"
    
    prediction = prediction.strip()
    
    # Remove common prefixes that LLMs sometimes add
    prefixes_to_remove = [
        "the answer is", "answer:", "score:", "grade:", 
        "final answer:", "prediction:", "result:", "output:"
    ]
    pred_lower = prediction.lower()
    for prefix in prefixes_to_remove:
        if pred_lower.startswith(prefix):
            prediction = prediction[len(prefix):].strip()
            pred_lower = prediction.lower()
            break
    
    # Extract expected score patterns from grading guidelines
    # Common IMO patterns: "7" (full score), "0" (no score), "1-6" (partial)
    if re.search(r'\b[0-7]\b', grading_guidelines):
        # IMO-style 0-7 scoring - look for single digit 0-7
        match = re.search(r'\b([0-7])\b', prediction)
        if match:
            return match.group(1)
        # Also check for spelled-out numbers
        number_words = {"zero": "0", "one": "1", "two": "2", "three": "3", 
                       "four": "4", "five": "5", "six": "6", "seven": "7"}
        for word, digit in number_words.items():
            if re.search(rf'\b{word}\b', pred_lower):
                return digit
    
    # Check for "Correct"/"Incorrect" format
    if "correct" in grading_guidelines.lower() or "incorrect" in grading_guidelines.lower():
        pred_lower = prediction.lower()
        # Check for explicit correct/incorrect mentions
        if "incorrect" in pred_lower or "wrong" in pred_lower or "false" in pred_lower:
            return "Incorrect"
        elif "correct" in pred_lower or "right" in pred_lower or "true" in pred_lower:
            return "Correct"
    
    # Check for Yes/No format
    if re.search(r'\b(yes|no)\b', grading_guidelines, re.IGNORECASE):
        if re.search(r'\byes\b', pred_lower):
            return "Yes"
        elif re.search(r'\bno\b', pred_lower):
            return "No"
    
    # Check for Pass/Fail format
    if re.search(r'\b(pass|fail)\b', grading_guidelines, re.IGNORECASE):
        if re.search(r'\bpass\b', pred_lower):
            return "Pass"
        elif re.search(r'\bfail\b', pred_lower):
            return "Fail"
    
    return prediction


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

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
        # Extract fields for structured prompting
        domain = inputs.get("domain", "")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = f"""You are an expert grader for the International Mathematical Olympiad (IMO). Your task is to evaluate a student's answer to a mathematical problem with precision and consistency.

## Domain
{domain}

## Problem Statement
{problem}

## Official Solution
{solution}

## Grading Guidelines
{grading_guidelines}

## Student's Answer
{student_answer}

## Your Task

Please evaluate the student's answer following this structured approach:

### Step 1: Problem Analysis
- Summarize what the problem is asking
- Identify key mathematical concepts and required techniques
- Note any critical assumptions or constraints

### Step 2: Solution Mapping
- Break down the official solution into key milestones
- Identify which milestones are essential vs. optional
- Note common alternative valid approaches

### Step 3: Student Answer Evaluation
- Map the student's work to solution milestones
- Identify correct steps with clear justification
- Flag any errors, gaps, or logical flaws
- Check for partial credit opportunities per guidelines

### Step 4: Grade Determination
- Apply grading guidelines systematically
- Consider: correctness, completeness, and clarity
- Assign the most appropriate grade/score

Respond ONLY in JSON format wrapped in <json> tags with the following exact schema:
<json>
{{
    "reasoning": "Your detailed chain-of-thought analysis following the 4 steps above...",
    "response": "The final grade/score as specified in the grading guidelines"
}}
</json>

CRITICAL INSTRUCTIONS:
1. The "response" field must contain ONLY the grade/score value (e.g., "7", "2", "0", "Correct", "Incorrect", etc.) exactly as specified in the grading guidelines.
2. Do NOT add explanations, reasoning, or extra text in the "response" field.
3. Do NOT use markdown formatting, code blocks, or any other formatting inside the JSON.
4. The JSON must be valid and properly escaped.
5. Wrap your entire JSON response in <json>...</json> tags."""

        # Check cache first to avoid redundant LLM calls
        cache_key = _get_cache_key(instruction, self.model, 0.0)
        cached_result = _get_cached_response(cache_key)
        if cached_result is not None:
            global _cache_hits
            _cache_hits += 1
            response, msg_history, info = cached_result
            self.log_fn(f"Cache hit! Using cached response. Cache stats: {get_cache_stats()}")
        else:
            # Get LLM response with retry logic for transient failures
            max_retries = 3
            response = None
            msg_history = []
            info = {}
            
            for attempt in range(max_retries):
                try:
                    response, msg_history, info = get_response_from_llm(
                        msg=instruction,
                        model=self.model,
                        msg_history=[],
                    )
                    # Store in cache for future use
                    global _cache_misses
                    _cache_misses += 1
                    # Store in the lru_cache by calling with the result
                    _store_in_cache(cache_key, (response, msg_history, info))
                    break
                except Exception as e:
                    self.log_fn(f"LLM call failed (attempt {attempt + 1}/{max_retries}): {type(e).__name__}: {e}")
                    if attempt == max_retries - 1:
                        self.log_fn("All LLM call attempts failed, returning error prediction")
                        return "Error: LLM call failed", msg_history
                    # Brief pause before retry with exponential backoff
                    time.sleep(0.5 * (2 ** attempt))

        # Extract prediction from JSON
        prediction = "None"
        last_text = msg_history[-1]["text"] if msg_history else response or ""
        extraction_method = "none"
        
        try:
            # Validate we have content to extract from
            if not last_text or not last_text.strip():
                self.log_fn("Warning: Empty response from LLM")
                prediction = "None"
            else:
                # First try: extract from <json> tags
                extracted = _extract_jsons(last_text)
                if extracted:
                    extraction_method = "json_tags"
                    # Try to get response field, fall back to other common fields
                    last_json = extracted[-1]
                    prediction = _extract_prediction_from_json(last_json)
                    self.log_fn(f"Extracted prediction from JSON tags: {prediction}")
                else:
                    # Second try: fallback extraction for non-tagged JSON
                    fallback = _extract_json_fallback(last_text)
                    if fallback:
                        extraction_method = "fallback"
                        prediction = _extract_prediction_from_json(fallback)
                        self.log_fn(f"Used fallback JSON extraction: {prediction}")
                    else:
                        # Third try: direct text extraction for simple responses
                        extraction_method = "direct"
                        prediction = _extract_prediction_from_text(last_text, grading_guidelines)
                        if prediction != "None":
                            self.log_fn(f"Used direct text extraction: {prediction}")
                
                # Validate and normalize the prediction
                original_prediction = prediction
                prediction = _validate_and_normalize_prediction(prediction, grading_guidelines)
                
                if original_prediction != prediction:
                    self.log_fn(f"Normalized prediction from '{original_prediction}' to '{prediction}'")
            
            self.log_fn(f"Extraction method: {extraction_method}, final prediction: {prediction}")
            
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {type(e).__name__}: {e}")
            # Last resort: try to find any numeric or keyword answer in the text
            try:
                prediction = _last_resort_extraction(last_text, grading_guidelines)
                extraction_method = "last_resort"
                self.log_fn(f"Last resort extraction: {prediction}")
            except Exception as e2:
                self.log_fn(f"Last resort extraction also failed: {e2}")
                prediction = "None"

        return str(prediction), msg_history


def _extract_prediction_from_json(json_obj: dict) -> str:
    """Extract prediction value from a JSON object.
    
    Tries multiple common field names in order of priority.
    """
    priority_fields = ["response", "grade", "score", "answer", "result", "value", "output"]
    
    for field in priority_fields:
        if field in json_obj:
            value = json_obj[field]
            # Handle nested structures
            if isinstance(value, (dict, list)):
                return json.dumps(value)
            return str(value).strip()
    
    # If no recognized field, return the whole JSON as string
    return json.dumps(json_obj)


def _extract_prediction_from_text(text: str, grading_guidelines: str) -> str:
    """Extract prediction directly from text when JSON parsing fails.
    
    Looks for simple answers in the last non-empty lines.
    """
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    # Check lines from the end (most likely to contain the answer)
    for line in reversed(lines[-5:] if len(lines) > 5 else lines):
        # Skip lines that look like JSON or XML tags
        if line.startswith('{') or line.startswith('[') or line.startswith('<'):
            continue
        
        # Skip lines that are too long (likely not a simple answer)
        if len(line) > 100:
            continue
        
        # Check for IMO scores (0-7)
        if re.search(r'\b[0-7]\b', grading_guidelines):
            match = re.search(r'\b([0-7])\b', line)
            if match:
                return match.group(1)
        
        # Check for Yes/No
        if re.search(r'\b(yes|no)\b', grading_guidelines, re.IGNORECASE):
            if re.search(r'\byes\b', line, re.IGNORECASE):
                return "Yes"
            if re.search(r'\bno\b', line, re.IGNORECASE):
                return "No"
        
        # Check for Correct/Incorrect
        if re.search(r'\b(correct|incorrect)\b', grading_guidelines, re.IGNORECASE):
            if re.search(r'\bcorrect\b', line, re.IGNORECASE):
                return "Correct"
            if re.search(r'\bincorrect\b', line, re.IGNORECASE):
                return "Incorrect"
        
        # Check for Pass/Fail
        if re.search(r'\b(pass|fail)\b', grading_guidelines, re.IGNORECASE):
            if re.search(r'\bpass\b', line, re.IGNORECASE):
                return "Pass"
            if re.search(r'\bfail\b', line, re.IGNORECASE):
                return "Fail"
        
        # If line looks like a simple answer (short, no special chars)
        if len(line) < 50 and not any(c in line for c in ['{', '}', '[', ']', '<', '>', '"', "'"]):
            return line
    
    return "None"


def _last_resort_extraction(text: str, grading_guidelines: str) -> str:
    """Last resort extraction when all other methods fail.
    
    Uses regex patterns to find likely answer values.
    """
    text_lower = text.lower()
    
    # Look for IMO scores (0-7)
    if re.search(r'\b[0-7]\b', grading_guidelines):
        score_match = re.search(r'\b([0-7])\b', text)
        if score_match:
            return score_match.group(1)
    
    # Look for common answer patterns
    patterns = [
        (r'\bfinal\s+(?:grade|score|answer|result)[\s:]+\s*(\S+)', 1),
        (r'\b(?:grade|score|answer|result)[\s:]+\s*(\S+)', 1),
        (r'\bthe\s+answer\s+is[\s:]+\s*(\S+)', 1),
        (r'\btherefore[,:]?\s+(\S+)', 1),
    ]
    
    for pattern, group in patterns:
        match = re.search(pattern, text_lower)
        if match:
            return match.group(group).strip()
    
    return "None"
