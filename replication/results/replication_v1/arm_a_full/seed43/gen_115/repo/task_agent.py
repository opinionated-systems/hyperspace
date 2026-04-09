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
from functools import lru_cache

from agent.llm_client import get_response_from_llm, EVAL_MODEL

logger = logging.getLogger(__name__)

# Simple in-memory cache for LLM responses to improve performance
# and reduce API costs for repeated similar queries
_response_cache: dict[str, tuple[str, list, dict]] = {}
_MAX_CACHE_SIZE = 1000


def _get_cache_key(msg: str, model: str, temperature: float) -> str:
    """Generate a cache key from message parameters."""
    key_data = f"{msg}:{model}:{temperature}"
    return hashlib.sha256(key_data.encode()).hexdigest()[:32]


def _cache_response(key: str, response: tuple[str, list, dict]) -> None:
    """Cache an LLM response with LRU eviction."""
    global _response_cache
    if len(_response_cache) >= _MAX_CACHE_SIZE:
        # Evict oldest entry (simple FIFO)
        _response_cache.pop(next(iter(_response_cache)))
    _response_cache[key] = response


def _get_cached_response(key: str) -> tuple[str, list, dict] | None:
    """Retrieve a cached LLM response if available."""
    return _response_cache.get(key)


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    Also handles nested JSON objects and markdown code blocks within the content.
    Includes improved handling for malformed tags and nested structures.
    """
    results = []
    search_from = 0
    max_iterations = 100  # Safety limit to prevent infinite loops
    iterations = 0
    
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
                parsed = _parse_json_with_fallback(inner)
                if parsed is not None:
                    results.append(parsed)
            break
        
        inner = text[start + 6:end].strip()
        search_from = end + 7
        
        # Skip empty blocks
        if not inner:
            continue
        
        # Try to parse the inner content as JSON
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
    """
    import re
    
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
    """Task agent that solves IMO grading problems with chain-of-thought reasoning.
    
    This agent uses an LLM to evaluate student answers to mathematical problems
    based on official solutions and grading guidelines. It implements structured
    prompting with 4-step analysis and robust JSON response extraction.
    
    Attributes:
        model: The LLM model identifier to use for evaluation
        log_fn: Logging function for agent activity
    """

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict containing:
                - domain: Problem domain/category
                - problem: The problem statement
                - solution: Official solution
                - grading_guidelines: How to grade the answer
                - student_answer: The student's submitted answer

        Returns:
            tuple of (prediction, msg_history) where:
                - prediction: The extracted grade/score as a string
                - msg_history: Full conversation history with the LLM
        
        Raises:
            ValueError: If required fields are missing from inputs
        """
        # Validate required inputs
        required_fields = ["problem", "solution", "grading_guidelines", "student_answer"]
        missing = [f for f in required_fields if not inputs.get(f)]
        if missing:
            raise ValueError(f"Missing required input fields: {missing}")
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
        cached = _get_cached_response(cache_key)
        
        if cached:
            response, msg_history, info = cached
            self.log_fn("Cache hit: Using cached LLM response")
        else:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
            # Cache the response for future use
            _cache_response(cache_key, (response, msg_history, info))
            self.log_fn("Cache miss: LLM response cached for future use")

        # Extract prediction from JSON
        prediction = "None"
        last_text = msg_history[-1]["text"] if msg_history else ""
        extraction_method = "none"
        
        try:
            # First try: extract from <json> tags
            extracted = _extract_jsons(last_text)
            if extracted:
                extraction_method = "json_tags"
                # Try to get response field, fall back to other common fields
                last_json = extracted[-1]
                prediction = _extract_prediction_from_json(last_json)
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
            self.log_fn(f"Error extracting prediction: {e}")
            # Last resort: try to find any numeric or keyword answer in the text
            prediction = _last_resort_extraction(last_text, grading_guidelines)

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


# Cache management utilities
def clear_response_cache() -> None:
    """Clear the LLM response cache."""
    global _response_cache
    _response_cache.clear()
    logger.info("Response cache cleared")


def get_cache_stats() -> dict:
    """Get statistics about the response cache.
    
    Returns:
        dict with cache size, max size, and hit/miss info
    """
    return {
        "size": len(_response_cache),
        "max_size": _MAX_CACHE_SIZE,
        "utilization": len(_response_cache) / _MAX_CACHE_SIZE,
    }


def set_max_cache_size(size: int) -> None:
    """Set the maximum cache size.
    
    Args:
        size: New maximum cache size (must be positive)
    """
    global _MAX_CACHE_SIZE
    if size < 1:
        raise ValueError("Cache size must be at least 1")
    _MAX_CACHE_SIZE = size
    # Trim cache if needed
    while len(_response_cache) > _MAX_CACHE_SIZE:
        _response_cache.pop(next(iter(_response_cache)))
    logger.info(f"Max cache size set to {size}")
