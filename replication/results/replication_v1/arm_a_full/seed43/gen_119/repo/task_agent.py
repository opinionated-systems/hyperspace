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

# Simple in-memory cache for LLM responses to avoid redundant calls
_response_cache: dict[str, tuple[str, list[dict], dict]] = {}


def _get_cache_key(msg: str, model: str, temperature: float) -> str:
    """Generate a cache key from message and model parameters."""
    key_data = f"{model}:{temperature}:{msg}"
    return hashlib.sha256(key_data.encode()).hexdigest()[:32]


def _cached_get_response_from_llm(
    msg: str,
    model: str = EVAL_MODEL,
    temperature: float = 0.0,
    msg_history: list[dict] | None = None,
) -> tuple[str, list[dict], dict]:
    """Cached version of get_response_from_llm to avoid redundant calls.
    
    Only caches when msg_history is empty (fresh conversation).
    """
    global _response_cache
    
    # Only use cache for fresh conversations
    if msg_history is None or len(msg_history) == 0:
        cache_key = _get_cache_key(msg, model, temperature)
        if cache_key in _response_cache:
            logger.info(f"Cache hit for LLM call (key: {cache_key[:8]}...)")
            return _response_cache[cache_key]
        
        result = get_response_from_llm(msg, model, temperature, msg_history)
        _response_cache[cache_key] = result
        return result
    
    # For non-fresh conversations, don't cache
    return get_response_from_llm(msg, model, temperature, msg_history)


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    Also handles nested JSON objects within the content.
    Includes improved handling for malformed JSON and nested structures.
    """
    results = []
    search_from = 0
    extraction_errors = []
    
    while True:
        start = text.find("<json>", search_from)
        if start == -1:
            break
        end = text.find("</json>", start)
        if end == -1:
            extraction_errors.append(f"Unclosed <json> tag at position {start}")
            break
        inner = text[start + 6:end].strip()
        search_from = end + 7
        
        # Try to parse the inner content as JSON
        try:
            results.append(json.loads(inner))
        except json.JSONDecodeError as e:
            extraction_errors.append(f"JSON parse error at position {start}: {e}")
            # Try to extract JSON from markdown code blocks within the content
            code_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', inner, re.DOTALL)
            if code_block_match:
                try:
                    results.append(json.loads(code_block_match.group(1)))
                    extraction_errors.pop()  # Remove the error since we recovered
                except json.JSONDecodeError as e2:
                    extraction_errors.append(f"Code block JSON parse error: {e2}")
            else:
                # Try to find JSON object directly in the content
                # Handle cases where JSON might be wrapped in other text
                json_match = re.search(r'\{[\s\S]*?"response"[\s\S]*?\}', inner)
                if json_match:
                    try:
                        results.append(json.loads(json_match.group(0)))
                        extraction_errors.pop()  # Remove the error since we recovered
                    except json.JSONDecodeError as e3:
                        extraction_errors.append(f"Direct JSON match parse error: {e3}")
            continue
    
    # Log extraction errors for debugging
    if extraction_errors:
        logger.debug(f"JSON extraction had {len(extraction_errors)} errors: {extraction_errors[:3]}")
    
    return results or None


def _extract_json_fallback(text: str) -> dict | None:
    """Fallback JSON extraction for non-tagged JSON or markdown code blocks.

    Tries to find JSON in markdown code blocks or raw JSON objects.
    Uses a robust approach to handle nested braces.
    """
    # First try markdown code blocks
    patterns = [
        r'```json\s*(.*?)\s*```',  # Markdown JSON blocks
        r'```\s*(\{.*?\})\s*```',  # Generic code blocks with JSON
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
    
    # Try to find JSON objects by matching braces with proper nesting
    # Look for patterns that look like JSON objects with expected keys
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
    for candidate in json_candidates:
        try:
            parsed = json.loads(candidate)
            # Prioritize candidates with expected keys
            if any(key in parsed for key in ["response", "grade", "score", "answer", "reasoning"]):
                return parsed
        except json.JSONDecodeError:
            continue
    
    # If no prioritized candidate found, return the first valid one
    for candidate in json_candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    
    return None


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


def clear_response_cache() -> None:
    """Clear the LLM response cache. Useful for testing or when memory is constrained."""
    global _response_cache
    cache_size = len(_response_cache)
    _response_cache.clear()
    logger.info(f"Cleared response cache (was {cache_size} entries)")


def get_cache_stats() -> dict:
    """Get statistics about the response cache."""
    global _response_cache
    return {
        "size": len(_response_cache),
        "estimated_memory_kb": sum(
            len(str(v[0])) + len(str(v[1])) + len(str(v[2])) 
            for v in _response_cache.values()
        ) // 1024,
    }


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning.
    
    Features:
    - Caching of LLM responses for improved performance
    - Enhanced error handling and logging
    - Multiple JSON extraction strategies with fallback
    """

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "", use_cache: bool = True) -> None:
        self.model = model
        self.log_fn = logger.info
        self.use_cache = use_cache
        
        if not use_cache:
            logger.info("TaskAgent initialized with caching disabled")
    
    def get_stats(self) -> dict:
        """Get agent statistics including cache info."""
        return {
            "model": self.model,
            "cache_enabled": self.use_cache,
            "cache_stats": get_cache_stats() if self.use_cache else None,
        }

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

        # Use cached LLM call for better performance (if enabled)
        try:
            if self.use_cache:
                response, msg_history, info = _cached_get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=[],
                )
            else:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=[],
                )
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return "None", [{"role": "error", "text": f"LLM call failed: {e}"}]

        # Extract prediction from JSON
        prediction = "None"
        last_text = msg_history[-1]["text"] if msg_history else ""
        extraction_method = "none"
        extraction_errors = []
        
        try:
            # First try: extract from <json> tags
            extracted = _extract_jsons(last_text)
            if extracted:
                extraction_method = "json_tags"
                # Try to get response field, fall back to other common fields
                last_json = extracted[-1]
                if "response" in last_json:
                    prediction = str(last_json["response"])
                elif "grade" in last_json:
                    prediction = str(last_json["grade"])
                elif "score" in last_json:
                    prediction = str(last_json["score"])
                elif "answer" in last_json:
                    prediction = str(last_json["answer"])
                else:
                    # If no recognized field, use the whole JSON as string
                    prediction = json.dumps(last_json)
                    extraction_errors.append("No recognized field in JSON")
            else:
                # Second try: fallback extraction for non-tagged JSON
                fallback = _extract_json_fallback(last_text)
                if fallback:
                    extraction_method = "fallback"
                    if "response" in fallback:
                        prediction = str(fallback["response"])
                    elif "grade" in fallback:
                        prediction = str(fallback["grade"])
                    elif "score" in fallback:
                        prediction = str(fallback["score"])
                    elif "answer" in fallback:
                        prediction = str(fallback["answer"])
                    else:
                        prediction = json.dumps(fallback)
                        extraction_errors.append("No recognized field in fallback JSON")
                    self.log_fn(f"Used fallback JSON extraction: {prediction}")
                else:
                    # Third try: direct text extraction for simple responses
                    extraction_method = "direct"
                    # Look for the last line that might be the answer
                    lines = [line.strip() for line in last_text.split('\n') if line.strip()]
                    if lines:
                        # Check if the last non-empty line looks like a simple answer
                        last_line = lines[-1]
                        if len(last_line) < 50 and not last_line.startswith('{') and not last_line.startswith('<'):
                            prediction = last_line
                            self.log_fn(f"Used direct text extraction: {prediction}")
                        else:
                            extraction_errors.append("Last line not suitable for direct extraction")
                    else:
                        extraction_errors.append("No non-empty lines found")
            
            # Validate and normalize the prediction
            original_prediction = prediction
            prediction = _validate_and_normalize_prediction(prediction, grading_guidelines)
            
            if original_prediction != prediction:
                self.log_fn(f"Normalized prediction from '{original_prediction}' to '{prediction}'")
            
            # Log extraction details for debugging
            if extraction_errors:
                self.log_fn(f"Extraction method: {extraction_method}, errors: {extraction_errors[:2]}")
            else:
                self.log_fn(f"Extraction method: {extraction_method}, final prediction: {prediction}")
            
        except Exception as e:
            extraction_errors.append(f"Error extracting prediction: {e}")
            self.log_fn(f"Error extracting prediction: {e}")
            # Last resort: try to find any numeric or keyword answer in the text
            try:
                # Look for IMO scores (0-7)
                score_match = re.search(r'\b([0-7])\b', last_text)
                if score_match:
                    prediction = score_match.group(1)
                    extraction_method = "regex_fallback"
                    self.log_fn(f"Used regex extraction for score: {prediction}")
            except Exception as e2:
                extraction_errors.append(f"Regex fallback also failed: {e2}")

        return str(prediction), msg_history
