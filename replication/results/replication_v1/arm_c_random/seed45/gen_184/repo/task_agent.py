"""
Task agent: solves a given task with chain-of-thought reasoning and self-reflection.

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
from typing import Any

from agent.llm_client import get_response_from_llm, EVAL_MODEL

logger = logging.getLogger(__name__)

# Simple LRU cache for LLM responses to avoid redundant grading calls
_llm_cache: dict[str, tuple[str, list[dict], dict]] = {}
_cache_max_size = 100
_cache_hits = 0
_cache_misses = 0


def _get_cache_key(msg: str, model: str, temperature: float) -> str:
    """Generate a cache key from message parameters.
    
    Uses a robust hashing approach that includes model, temperature, and message content.
    The key is deterministic and collision-resistant for practical purposes.
    """
    # Normalize inputs to ensure consistent keys
    normalized_msg = msg.strip()
    key_data = f"{model}|{temperature:.6f}|{normalized_msg}"
    return hashlib.sha256(key_data.encode()).hexdigest()[:32]


def _get_cached_response(msg: str, model: str, temperature: float) -> tuple[str, list[dict], dict] | None:
    """Get cached LLM response if available."""
    global _cache_hits
    key = _get_cache_key(msg, model, temperature)
    if key in _llm_cache:
        _cache_hits += 1
        logger.debug(f"Cache hit (total hits: {_cache_hits}, misses: {_cache_misses})")
        return _llm_cache[key]
    return None


def _cache_response(msg: str, model: str, temperature: float, response: tuple[str, list[dict], dict]) -> None:
    """Cache an LLM response with LRU eviction."""
    global _cache_misses
    key = _get_cache_key(msg, model, temperature)
    
    # LRU eviction: remove oldest entries if cache is full
    if len(_llm_cache) >= _cache_max_size:
        oldest_key = next(iter(_llm_cache))
        del _llm_cache[oldest_key]
    
    _llm_cache[key] = response
    _cache_misses += 1


def get_cache_stats() -> dict[str, int]:
    """Get cache statistics for monitoring."""
    total = _cache_hits + _cache_misses
    hit_rate = (_cache_hits / total * 100) if total > 0 else 0
    return {
        "hits": _cache_hits,
        "misses": _cache_misses,
        "size": len(_llm_cache),
        "hit_rate_percent": round(hit_rate, 2),
    }


def clear_cache() -> None:
    """Clear the LLM response cache."""
    global _cache_hits, _cache_misses
    _llm_cache.clear()
    _cache_hits = 0
    _cache_misses = 0


def save_cache_to_disk(filepath: str) -> bool:
    """Save the current cache to disk for persistence.
    
    Args:
        filepath: Path to save the cache file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        import pickle
        cache_data = {
            'cache': _llm_cache,
            'hits': _cache_hits,
            'misses': _cache_misses,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(cache_data, f)
        logger.info(f"Cache saved to {filepath} ({len(_llm_cache)} entries)")
        return True
    except Exception as e:
        logger.error(f"Failed to save cache: {e}")
        return False


def load_cache_from_disk(filepath: str) -> bool:
    """Load cache from disk.
    
    Args:
        filepath: Path to the cache file
        
    Returns:
        True if successful, False otherwise
    """
    global _llm_cache, _cache_hits, _cache_misses
    try:
        import pickle
        import os
        if not os.path.exists(filepath):
            logger.info(f"Cache file not found: {filepath}")
            return False
        with open(filepath, 'rb') as f:
            cache_data = pickle.load(f)
        _llm_cache = cache_data.get('cache', {})
        _cache_hits = cache_data.get('hits', 0)
        _cache_misses = cache_data.get('misses', 0)
        logger.info(f"Cache loaded from {filepath} ({len(_llm_cache)} entries)")
        return True
    except Exception as e:
        logger.error(f"Failed to load cache: {e}")
        return False


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    Also handles markdown code blocks and inline JSON as fallbacks.
    Includes robust error recovery for common JSON formatting issues.
    """
    results = []
    search_from = 0
    
    # First try to find <json>...</json> blocks
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
            # Try to fix common JSON issues before giving up
            fixed_json = _attempt_json_repair(inner)
            if fixed_json:
                results.append(fixed_json)
            else:
                logger.debug(f"JSON decode error in <json> block: {e}")
            continue
    
    # Fallback 1: try markdown code blocks if no <json> blocks found
    if not results:
        markdown_pattern = r'```(?:json)?\s*\n(.*?)\n```'
        for match in re.finditer(markdown_pattern, text, re.DOTALL):
            try:
                results.append(json.loads(match.group(1).strip()))
            except json.JSONDecodeError:
                # Try repair on markdown blocks too
                fixed_json = _attempt_json_repair(match.group(1).strip())
                if fixed_json:
                    results.append(fixed_json)
                continue
    
    # Fallback 2: look for inline JSON objects (e.g., {"key": "value"})
    if not results:
        # Find JSON-like structures with balanced braces using stack-based matching
        results = _extract_balanced_json(text, results)
    
    return results or None


def _extract_balanced_json(text: str, results: list[dict]) -> list[dict]:
    """Extract JSON objects using brace balance counting for nested structures."""
    i = 0
    while i < len(text):
        if text[i] == '{':
            start = i
            brace_count = 1
            i += 1
            in_string = False
            escape_next = False
            
            while i < len(text) and brace_count > 0:
                char = text[i]
                if escape_next:
                    escape_next = False
                elif char == '\\' and in_string:
                    escape_next = True
                elif char == '"' and not in_string:
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
                candidate = text[start:i].strip()
                # Only accept if it looks like a grading result
                if '"score"' in candidate or '"response"' in candidate:
                    try:
                        results.append(json.loads(candidate))
                    except json.JSONDecodeError:
                        fixed_json = _attempt_json_repair(candidate)
                        if fixed_json and ('score' in fixed_json or 'response' in fixed_json):
                            results.append(fixed_json)
        else:
            i += 1
    
    return results


def _attempt_json_repair(json_str: str) -> dict | None:
    """Attempt to repair common JSON formatting issues.
    
    Fixes:
    - Trailing commas before closing braces/brackets
    - Single quotes instead of double quotes
    - Unescaped newlines in strings
    - Missing quotes around keys
    - Comments in JSON (// and /* */)
    - Control characters in strings
    - Malformed unicode escapes
    - Extra whitespace and BOM markers
    """
    import re
    
    original = json_str.strip()
    
    # Remove BOM if present
    if original.startswith('\ufeff'):
        original = original[1:]
    
    repaired = original
    
    # Fix 1: Remove single-line comments (// ...)
    repaired = re.sub(r'//[^\n]*', '', repaired)
    
    # Fix 2: Remove multi-line comments (/* ... */)
    repaired = re.sub(r'/\*.*?\*/', '', repaired, flags=re.DOTALL)
    
    # Fix 3: Remove trailing commas before } or ]
    repaired = re.sub(r',\s*\}', '}', repaired)
    repaired = re.sub(r',\s*\]', ']', repaired)
    
    # Fix 4: Replace single quotes with double quotes (carefully)
    # Only replace if not inside a string - use a more careful approach
    def replace_quotes(match):
        content = match.group(1)
        # Replace single quotes that are not escaped
        return '"' + content.replace("'", '"') + '"'
    
    # Replace single-quoted strings with double-quoted ones
    repaired = re.sub(r"'([^']*?)'", replace_quotes, repaired)
    
    # Fix 5: Escape unescaped newlines and control characters in string values
    # Use a more robust approach with proper string parsing
    repaired = _escape_control_chars(repaired)
    
    # Fix 6: Try to handle missing quotes around keys
    # Match word: followed by space or value (but not already quoted)
    repaired = re.sub(r'([{,]\s*)(\w+)(\s*:)', r'\1"\2"\3', repaired)
    
    # Fix 7: Handle malformed unicode escapes
    repaired = re.sub(r'\\u([0-9a-fA-F]{0,3})(?![0-9a-fA-F])', r'\\u000$1', repaired)
    
    # Fix 8: Remove control characters (except valid whitespace)
    repaired = ''.join(char for char in repaired if ord(char) >= 32 or char in '\n\r\t')
    
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        # Last resort: try to extract just the object structure
        return _extract_minimal_json(repaired)


def _escape_control_chars(text: str) -> str:
    """Escape control characters in JSON string values."""
    result = []
    in_string = False
    escape_next = False
    
    for char in text:
        if escape_next:
            result.append(char)
            escape_next = False
            continue
            
        if char == '\\':
            result.append(char)
            escape_next = True
            continue
            
        if char == '"' and not escape_next:
            in_string = not in_string
            result.append(char)
            continue
            
        if in_string and ord(char) < 32:
            # Escape control characters
            if char == '\n':
                result.append('\\n')
            elif char == '\r':
                result.append('\\r')
            elif char == '\t':
                result.append('\\t')
            else:
                result.append(f'\\u{ord(char):04x}')
        else:
            result.append(char)
    
    return ''.join(result)


def _extract_minimal_json(text: str) -> dict | None:
    """Last resort: try to extract key-value pairs from malformed JSON."""
    import re
    
    result = {}
    
    # Try to find score and max_score patterns
    score_match = re.search(r'["\']?score["\']?\s*[:=]\s*(\d+(?:\.\d+)?)', text, re.IGNORECASE)
    max_score_match = re.search(r'["\']?max_score["\']?\s*[:=]\s*(\d+(?:\.\d+)?)', text, re.IGNORECASE)
    response_match = re.search(r'["\']?response["\']?\s*[:=]\s*["\']([^"\']+)["\']', text, re.IGNORECASE)
    
    if score_match:
        try:
            result['score'] = float(score_match.group(1))
        except ValueError:
            pass
    
    if max_score_match:
        try:
            result['max_score'] = float(max_score_match.group(1))
        except ValueError:
            pass
    
    if response_match:
        result['response'] = response_match.group(1)
    
    # Also try to find score/max_score in format like "3/5"
    if not result.get('score') or not result.get('max_score'):
        fraction_match = re.search(r'(\d+)\s*/\s*(\d+)', text)
        if fraction_match:
            try:
                if not result.get('score'):
                    result['score'] = float(fraction_match.group(1))
                if not result.get('max_score'):
                    result['max_score'] = float(fraction_match.group(2))
            except ValueError:
                pass
    
    return result if result else None


def _validate_grading_result(result: dict) -> tuple[bool, str]:
    """Validate that a grading result has the required fields and valid values.
    
    Returns:
        (is_valid, error_message)
    """
    if not isinstance(result, dict):
        return False, f"Result must be a dict, got {type(result).__name__}"
    
    required_fields = ["score", "max_score"]
    
    for field in required_fields:
        if field not in result:
            return False, f"Missing required field: {field}"
    
    try:
        score = float(result["score"])
        max_score = float(result["max_score"])
    except (ValueError, TypeError):
        return False, "Score and max_score must be numeric"
    
    if score < 0:
        return False, f"Score cannot be negative: {score}"
    
    if max_score <= 0:
        return False, f"Max score must be positive: {max_score}"
    
    if score > max_score:
        return False, f"Score {score} exceeds max_score {max_score}"
    
    # Check for NaN or infinity
    if score != score or max_score != max_score:  # NaN check
        return False, "Score or max_score is NaN"
    
    if score == float('inf') or score == float('-inf'):
        return False, "Score is infinite"
    
    if max_score == float('inf') or max_score == float('-inf'):
        return False, "Max score is infinite"
    
    return True, ""


def _normalize_score(score: float, max_score: float, target_max: float = 10.0) -> float:
    """Normalize a score to a standard scale for comparison.
    
    Args:
        score: The raw score
        max_score: The maximum possible score
        target_max: The target maximum scale (default 10.0)
    
    Returns:
        Normalized score on the target scale
    """
    if max_score <= 0:
        return 0.0
    return (score / max_score) * target_max


# Few-shot examples for IMO grading
FEW_SHOT_EXAMPLES = """
Example 1:
Problem: Find all positive integers n such that n^2 + 3n + 2 is divisible by 4.
Solution: n^2 + 3n + 2 = (n+1)(n+2). For divisibility by 4, either n+1 or n+2 must be even, and one of them must be divisible by 4. This happens when n ≡ 0 or 3 (mod 4).
Grading Guidelines: Award 1 point for factoring, 1 point for analyzing cases, 1 point for correct answer.
Student Answer: "I factored it as (n+1)(n+2). Since these are consecutive integers, one is even. For divisibility by 4, we need one factor divisible by 4. This happens when n=3,7,11,... or n=0,4,8,... So n ≡ 0 or 3 (mod 4)."
Grade: {"score": 3, "max_score": 3, "rationale": "Complete solution with correct factoring, case analysis, and answer."}

Example 2:
Problem: Prove that the sum of two odd numbers is even.
Solution: Let the odd numbers be 2k+1 and 2m+1. Their sum is 2k+1+2m+1 = 2(k+m+1), which is even.
Grading Guidelines: Award 1 point for setting up odd number representation, 1 point for algebraic manipulation, 1 point for conclusion.
Student Answer: "Odd numbers end in 1,3,5,7,9. Adding two odd numbers gives an even number. For example, 3+5=8 which is even."
Grade: {"score": 1, "max_score": 3, "rationale": "Student only provided examples without general proof. Missing algebraic representation and general reasoning."}
"""


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self._call_count = 0
        self._total_latency = 0.0

    def _log_metrics(self, latency: float, success: bool) -> None:
        """Log performance metrics for monitoring."""
        self._call_count += 1
        self._total_latency += latency
        avg_latency = self._total_latency / self._call_count
        self.log_fn(f"[Metrics] Call #{self._call_count}, latency={latency:.2f}s, avg={avg_latency:.2f}s, success={success}")

    def get_cache_stats(self) -> dict[str, int]:
        """Get LLM response cache statistics."""
        return get_cache_stats()

    def clear_cache(self) -> None:
        """Clear the LLM response cache."""
        clear_cache()
        self.log_fn("LLM response cache cleared")

    def save_cache(self, filepath: str) -> bool:
        """Save the LLM response cache to disk.
        
        Args:
            filepath: Path to save the cache file
            
        Returns:
            True if successful, False otherwise
        """
        result = save_cache_to_disk(filepath)
        if result:
            self.log_fn(f"Cache saved to {filepath}")
        return result

    def load_cache(self, filepath: str) -> bool:
        """Load the LLM response cache from disk.
        
        Args:
            filepath: Path to the cache file
            
        Returns:
            True if successful, False otherwise
        """
        result = load_cache_from_disk(filepath)
        if result:
            self.log_fn(f"Cache loaded from {filepath}")
        return result

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with reasoning and reflection.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        start_time = time.time()
        
        # Step 1: Initial grading with chain-of-thought
        instruction = f"""You are an expert mathematical grader for the International Mathematical Olympiad (IMO).

Your task is to grade a student's answer to a mathematical problem.

{FEW_SHOT_EXAMPLES}

Now grade the following problem:

Domain: {inputs.get('domain', 'Mathematics')}

Problem:
{inputs.get('problem', '')}

Official Solution:
{inputs.get('solution', '')}

Grading Guidelines:
{inputs.get('grading_guidelines', '')}

Student Answer:
{inputs.get('student_answer', '')}

Think step by step:
1. Analyze what the student did correctly according to the official solution
2. Identify any errors, gaps, or missing steps
3. Compare against the grading guidelines
4. Determine the score and provide detailed rationale

Respond in JSON format with the following schema:
<json>
{{
    "thinking": "Your detailed step-by-step analysis here",
    "score": <numerical score>,
    "max_score": <maximum possible score>,
    "rationale": "Detailed explanation of why this score was awarded",
    "response": "<score>/<max_score> - <brief summary>"
}}
</json>"""

        try:
            # Check cache first for identical grading requests
            cached = _get_cached_response(instruction, self.model, 0.0)
            if cached:
                response, msg_history, info = cached
                self.log_fn("Using cached LLM response for initial grading")
            else:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=[],
                )
                # Cache the response for future use
                _cache_response(instruction, self.model, 0.0, (response, msg_history, info))
        except Exception as e:
            self.log_fn(f"Error in initial LLM call: {e}")
            latency = time.time() - start_time
            self._log_metrics(latency, False)
            return "None", [{"role": "error", "text": str(e)}]

        # Extract prediction from JSON with detailed logging and validation
        prediction = "None"
        extraction_success = False
        try:
            if msg_history and len(msg_history) > 0:
                last_msg = msg_history[-1].get("text", "")
                if last_msg:
                    extracted = _extract_jsons(last_msg)
                    if extracted:
                        result = extracted[-1]
                        self.log_fn(f"Extracted JSON result: {result}")
                        
                        # Validate the grading result
                        is_valid, error_msg = _validate_grading_result(result)
                        if not is_valid:
                            self.log_fn(f"Validation failed: {error_msg}")
                            # Try to use response field if available despite validation failure
                            if "response" in result:
                                prediction = str(result["response"])
                                self.log_fn(f"Using 'response' field despite validation failure: {prediction}")
                                extraction_success = True
                        elif "response" in result:
                            prediction = str(result["response"])
                            self.log_fn(f"Using 'response' field: {prediction}")
                            extraction_success = True
                        elif "score" in result and "max_score" in result:
                            prediction = f"{result['score']}/{result['max_score']}"
                            self.log_fn(f"Using score/max_score fields: {prediction}")
                            extraction_success = True
                        else:
                            self.log_fn(f"Warning: JSON missing expected fields. Keys: {list(result.keys())}")
                    else:
                        self.log_fn("Warning: No JSON blocks found in response")
                        # Try to extract any numeric score pattern as fallback
                        score_match = re.search(r'(\d+)\s*/\s*(\d+)', last_msg)
                        if score_match:
                            prediction = f"{score_match.group(1)}/{score_match.group(2)}"
                            self.log_fn(f"Fallback extraction: {prediction}")
                            extraction_success = True
                else:
                    self.log_fn("Warning: Last message has no text content")
            else:
                self.log_fn("Warning: Empty message history")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            import traceback
            self.log_fn(f"Traceback: {traceback.format_exc()}")

        # Step 2: Self-reflection to verify the grade
        if prediction != "None" and len(msg_history) >= 2:
            reflection_msg = f"""Review your grading above. Check for:
1. Did you award points the student didn't earn?
2. Did you miss any errors in the student's work?
3. Is your score consistent with the grading guidelines?
4. Would another grader agree with your assessment?

If you need to revise your grade, provide the corrected JSON. If your grade is correct, confirm it.

Respond in JSON format:
<json>
{{
    "reflection": "Your self-review here",
    "revised_score": <score>,
    "revised_max_score": <max_score>,
    "final_response": "<score>/<max_score> - <brief summary>"
}}
</json>"""
            
            try:
                # Check cache for reflection (includes msg_history context)
                reflection_cached = _get_cached_response(reflection_msg + str(msg_history), self.model, 0.0)
                if reflection_cached:
                    reflection_response, msg_history, _ = reflection_cached
                    self.log_fn("Using cached LLM response for reflection")
                else:
                    reflection_response, msg_history, _ = get_response_from_llm(
                        msg=reflection_msg,
                        model=self.model,
                        msg_history=msg_history,
                    )
                    # Cache the reflection response
                    _cache_response(reflection_msg + str(msg_history), self.model, 0.0, (reflection_response, msg_history, {}))
                
                # Try to extract revised prediction with detailed logging and validation
                try:
                    if msg_history and len(msg_history) > 0:
                        last_msg = msg_history[-1].get("text", "")
                        if last_msg:
                            extracted = _extract_jsons(last_msg)
                            if extracted:
                                result = extracted[-1]
                                self.log_fn(f"Reflection extracted JSON: {result}")
                                
                                # Validate reflection result before using it
                                # Map reflection fields to standard fields for validation
                                validation_result = result.copy()
                                if "revised_score" in result:
                                    validation_result["score"] = result["revised_score"]
                                if "revised_max_score" in result:
                                    validation_result["max_score"] = result["revised_max_score"]
                                
                                is_valid, error_msg = _validate_grading_result(validation_result)
                                if not is_valid:
                                    self.log_fn(f"Reflection validation failed: {error_msg}, keeping original prediction")
                                elif "final_response" in result:
                                    prediction = str(result["final_response"])
                                    self.log_fn(f"Using 'final_response' field: {prediction}")
                                    extraction_success = True
                                elif "revised_score" in result and "revised_max_score" in result:
                                    prediction = f"{result['revised_score']}/{result['revised_max_score']}"
                                    self.log_fn(f"Using revised_score/revised_max_score: {prediction}")
                                    extraction_success = True
                                elif "score" in result and "max_score" in result:
                                    # Fallback to standard fields if revision fields missing
                                    prediction = f"{result['score']}/{result['max_score']}"
                                    self.log_fn(f"Using score/max_score fields: {prediction}")
                                    extraction_success = True
                                else:
                                    self.log_fn(f"Warning: Reflection JSON missing expected fields. Keys: {list(result.keys())}")
                            else:
                                self.log_fn("Warning: No JSON found in reflection response, keeping original prediction")
                except Exception as e:
                    self.log_fn(f"Error extracting revised prediction: {e}")
                    # Keep original prediction on error
                    pass
            except Exception as e:
                self.log_fn(f"Error during reflection LLM call: {e}")
                # Keep original prediction if reflection fails
                pass

        # Log final metrics
        latency = time.time() - start_time
        self._log_metrics(latency, extraction_success)
        
        # Log cache statistics periodically
        if self._call_count % 10 == 0:
            cache_stats = get_cache_stats()
            self.log_fn(f"[Cache Stats] hits={cache_stats['hits']}, misses={cache_stats['misses']}, "
                       f"size={cache_stats['size']}, hit_rate={cache_stats['hit_rate_percent']}%")
        
        return str(prediction), msg_history
