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
from functools import lru_cache
from typing import Any

from agent.llm_client import get_response_from_llm, EVAL_MODEL

logger = logging.getLogger(__name__)

# Simple in-memory cache for LLM responses to improve efficiency
_llm_cache: dict[str, tuple[str, list[dict], dict]] = {}
_cache_hits = 0
_cache_misses = 0


def _get_cache_key(msg: str, model: str, temperature: float) -> str:
    """Generate a cache key for an LLM request."""
    key_data = f"{model}:{temperature}:{msg}"
    return hashlib.sha256(key_data.encode()).hexdigest()[:32]


def get_cached_response(
    msg: str,
    model: str = EVAL_MODEL,
    temperature: float = 0.0,
    msg_history: list[dict] | None = None,
) -> tuple[str, list[dict], dict] | None:
    """Get cached LLM response if available.
    
    Only caches when temperature is 0 (deterministic) and no history.
    """
    global _cache_hits, _cache_misses
    
    if temperature != 0.0 or msg_history:
        _cache_misses += 1
        return None
    
    cache_key = _get_cache_key(msg, model, temperature)
    if cache_key in _llm_cache:
        _cache_hits += 1
        logger.debug(f"Cache hit for key {cache_key[:8]}... (hits: {_cache_hits}, misses: {_cache_misses})")
        return _llm_cache[cache_key]
    
    _cache_misses += 1
    return None


def set_cached_response(
    msg: str,
    model: str,
    temperature: float,
    result: tuple[str, list[dict], dict],
) -> None:
    """Cache an LLM response."""
    if temperature != 0.0:
        return
    
    cache_key = _get_cache_key(msg, model, temperature)
    _llm_cache[cache_key] = result
    
    # Limit cache size to prevent memory issues
    if len(_llm_cache) > 1000:
        # Remove oldest entries (simple FIFO)
        oldest_key = next(iter(_llm_cache))
        del _llm_cache[oldest_key]


def get_cache_stats() -> dict:
    """Get cache statistics."""
    total = _cache_hits + _cache_misses
    hit_rate = _cache_hits / total if total > 0 else 0
    return {
        "hits": _cache_hits,
        "misses": _cache_misses,
        "hit_rate": round(hit_rate, 4),
        "size": len(_llm_cache),
    }


def clear_cache() -> None:
    """Clear the LLM response cache."""
    global _llm_cache, _cache_hits, _cache_misses
    _llm_cache.clear()
    _cache_hits = 0
    _cache_misses = 0


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    Also handles markdown code blocks and inline JSON as fallbacks.
    Includes robust error recovery for common LLM formatting issues.
    """
    if not text or not isinstance(text, str):
        logger.debug(f"Invalid input to _extract_jsons: {type(text)}")
        return None
        
    results = []
    search_from = 0
    
    # First try to find <json>...</json> blocks
    while True:
        start = text.find("<json>", search_from)
        if start == -1:
            break
        end = text.find("</json>", start)
        if end == -1:
            logger.debug("Found <json> but no closing </json>")
            break
        inner = text[start + 6:end].strip()
        search_from = end + 7
        
        # Skip empty blocks
        if not inner:
            continue
            
        try:
            results.append(json.loads(inner))
        except json.JSONDecodeError as e:
            logger.debug(f"JSON decode error in <json> block: {e}")
            # Try to fix common LLM formatting issues
            fixed = _attempt_json_repair(inner)
            if fixed:
                results.append(fixed)
                logger.debug("Successfully repaired JSON from <json> block")
            continue
    
    # Fallback 1: try markdown code blocks if no <json> blocks found
    if not results:
        # Look for ```json ... ``` blocks
        markdown_pattern = r'```(?:json)?\s*\n(.*?)\n```'
        for match in re.finditer(markdown_pattern, text, re.DOTALL):
            content = match.group(1).strip()
            if not content:
                continue
            try:
                results.append(json.loads(content))
            except json.JSONDecodeError:
                # Try repair on markdown blocks too
                fixed = _attempt_json_repair(content)
                if fixed:
                    results.append(fixed)
                    logger.debug("Successfully repaired JSON from markdown block")
                continue
    
    # Fallback 2: try to find JSON objects directly in the text
    if not results:
        # Look for patterns that look like JSON objects
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        for match in re.finditer(json_pattern, text, re.DOTALL):
            try:
                candidate = match.group(0).strip()
                if len(candidate) < 20:  # Too short to be a valid grading result
                    continue
                # Only accept if it has expected keys
                parsed = json.loads(candidate)
                if any(key in parsed for key in ['score', 'response', 'thinking', 'rationale']):
                    results.append(parsed)
                    logger.debug("Successfully extracted JSON from inline text")
            except (json.JSONDecodeError, ValueError):
                continue
    
    if results:
        logger.debug(f"Extracted {len(results)} JSON object(s)")
    else:
        logger.debug("No JSON objects found in text")
        
    return results or None


def _attempt_json_repair(text: str) -> dict | None:
    """Attempt to repair common JSON formatting errors from LLM outputs.
    
    Fixes:
    - Trailing commas before closing braces/brackets
    - Single quotes instead of double quotes
    - Unquoted keys
    - Missing quotes around string values
    - Newlines in string values
    - Comments in JSON (// and /* */)
    """
    import re
    
    original = text.strip()
    
    # Fix 1: Remove C-style comments (// and /* */)
    repaired = re.sub(r'//[^\n]*', '', original)
    repaired = re.sub(r'/\*.*?\*/', '', repaired, flags=re.DOTALL)
    
    # Fix 2: Remove trailing commas before } or ]
    repaired = re.sub(r',\s*([}\]])', r'\1', repaired)
    
    # Fix 3: Replace single quotes with double quotes (carefully)
    # Only replace single quotes that appear to be delimiters
    repaired = re.sub(r"(?<!\\)'([^']*?)'(?=\s*[:}\],])", r'"\1"', repaired)
    
    # Fix 4: Add quotes to unquoted keys
    repaired = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', repaired)
    
    # Fix 5: Fix common escape sequence issues
    repaired = repaired.replace('\\"', '"').replace('\\n', '\n').replace('\\t', '\t')
    
    # Fix 6: Handle newlines in string values by escaping them
    # This is a more aggressive fix for multiline strings
    def escape_newlines_in_strings(match):
        content = match.group(1)
        # Escape unescaped newlines
        content = re.sub(r'(?<!\\)\n', r'\\n', content)
        content = re.sub(r'(?<!\\)\t', r'\\t', content)
        return '"' + content + '"'
    
    repaired = re.sub(r'"([^"]*(?:\n[^"]*)*)"', escape_newlines_in_strings, repaired)
    
    try:
        return json.loads(repaired)
    except json.JSONDecodeError as e:
        logger.debug(f"JSON repair failed: {e}")
        return None


def validate_grading_result(result: dict) -> tuple[bool, str]:
    """Validate that a grading result has all required fields.
    
    Returns:
        (is_valid, error_message)
    """
    if not isinstance(result, dict):
        return False, f"Result must be a dictionary, got {type(result).__name__}"
    
    required_fields = ['score', 'max_score', 'rationale']
    missing = [f for f in required_fields if f not in result]
    
    if missing:
        return False, f"Missing required fields: {missing}"
    
    # Check for extra fields that might indicate confusion
    unexpected_fields = set(result.keys()) - set(required_fields + ['thinking', 'response', 'reflection', 'revised_score', 'revised_max_score', 'final_response'])
    if unexpected_fields:
        logger.debug(f"Unexpected fields in grading result: {unexpected_fields}")
    
    # Validate score is numeric
    try:
        score = float(result['score'])
        max_score = float(result['max_score'])
        if score < 0 or max_score <= 0:
            return False, f"Invalid score values (score={score} must be >= 0, max_score={max_score} must be > 0)"
        if score > max_score:
            return False, f"Score {score} exceeds max_score {max_score}"
        # Check for reasonable score precision (avoid floating point issues)
        if score != int(score) and abs(score - round(score, 2)) > 0.001:
            logger.debug(f"Score has unusual precision: {score}")
    except (ValueError, TypeError) as e:
        return False, f"Score and max_score must be numeric: {e}"
    
    # Validate rationale is non-empty string
    if not isinstance(result['rationale'], str):
        return False, f"Rationale must be a string, got {type(result['rationale']).__name__}"
    if not result['rationale'].strip():
        return False, "Rationale must be a non-empty string"
    if len(result['rationale']) < 10:
        return False, "Rationale is too short (minimum 10 characters)"
    
    return True, ""


def compute_grading_accuracy(predicted_score: float, max_score: float, reference_score: float | None = None) -> dict:
    """Compute accuracy metrics for a grading result.
    
    Args:
        predicted_score: The score assigned by the grader
        max_score: Maximum possible score
        reference_score: Optional reference score to compare against
        
    Returns:
        Dictionary with accuracy metrics
    """
    # Validate inputs
    try:
        predicted_score = float(predicted_score)
        max_score = float(max_score)
    except (ValueError, TypeError):
        logger.warning(f"Invalid score types: predicted={type(predicted_score)}, max={type(max_score)}")
        return {"error": "Invalid score types"}
    
    if max_score <= 0:
        logger.warning(f"Invalid max_score: {max_score}")
        return {"error": "max_score must be positive"}
    
    metrics = {
        "normalized_score": round(predicted_score / max_score, 4),
        "is_perfect": predicted_score == max_score,
        "is_zero": predicted_score == 0,
        "score_percentage": round(100 * predicted_score / max_score, 2),
    }
    
    if reference_score is not None:
        try:
            reference_score = float(reference_score)
            metrics["reference_score"] = reference_score
            metrics["absolute_error"] = round(abs(predicted_score - reference_score), 4)
            metrics["relative_error"] = round(abs(predicted_score - reference_score) / max_score, 4)
            metrics["matches_reference"] = abs(predicted_score - reference_score) < 0.001  # Allow small floating point differences
            metrics["error_direction"] = "over" if predicted_score > reference_score else "under" if predicted_score < reference_score else "exact"
        except (ValueError, TypeError):
            logger.warning(f"Invalid reference_score type: {type(reference_score)}")
            metrics["reference_error"] = "Invalid reference score"
    
    return metrics


class GradingError(Exception):
    """Custom exception for grading-related errors."""
    pass


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
Grading Guidelines: Award 1 point for setting up odd number representation, 1 point for correct algebra, 1 point for conclusion.
Student Answer: "Let two odd numbers be 2a+1 and 2b+1. Adding: 2a+1+2b+1 = 2a+2b+2 = 2(a+b+1). This is clearly divisible by 2, so it's even."
Grade: {"score": 3, "max_score": 3, "rationale": "Correct representation, algebra, and conclusion."}
"""


def validate_grading_inputs(inputs: dict) -> tuple[bool, str]:
    """Validate grading inputs before processing.
    
    Returns:
        (is_valid, error_message)
    """
    if not isinstance(inputs, dict):
        return False, f"Inputs must be a dictionary, got {type(inputs).__name__}"
    
    required_fields = ['problem', 'solution', 'grading_guidelines', 'student_answer']
    missing = [f for f in required_fields if f not in inputs or not inputs[f]]
    
    if missing:
        return False, f"Missing or empty required fields: {missing}"
    
    # Validate field types and content
    for field in required_fields:
        value = inputs.get(field)
        if not isinstance(value, str):
            return False, f"Field '{field}' must be a string, got {type(value).__name__}"
        if len(value.strip()) < 5:
            return False, f"Field '{field}' is too short (minimum 5 characters)"
    
    return True, ""


def compute_grading_confidence(
    grading_result: dict,
    extraction_method: str,
    has_reflection: bool,
    reflection_consistent: bool | None,
) -> dict:
    """Compute confidence metrics for a grading result.
    
    Args:
        grading_result: The parsed grading result
        extraction_method: How the result was extracted ('json', 'fallback', 'repair')
        has_reflection: Whether reflection step was performed
        reflection_consistent: Whether reflection confirmed the grade (None if no reflection)
        
    Returns:
        Dictionary with confidence metrics (0.0 to 1.0)
    """
    confidence = 1.0
    factors = {}
    
    # Extraction method affects confidence
    if extraction_method == 'json':
        factors['extraction'] = 1.0
    elif extraction_method == 'repair':
        factors['extraction'] = 0.8
        confidence *= 0.8
    else:  # fallback
        factors['extraction'] = 0.6
        confidence *= 0.6
    
    # Reflection consistency affects confidence
    if has_reflection:
        if reflection_consistent is True:
            factors['reflection'] = 1.0
        elif reflection_consistent is False:
            factors['reflection'] = 0.7
            confidence *= 0.7
        else:
            factors['reflection'] = 0.8
            confidence *= 0.8
    else:
        factors['reflection'] = 0.5
        confidence *= 0.5
    
    # Validate result structure
    required_fields = ['score', 'max_score', 'rationale']
    has_all_fields = all(f in grading_result for f in required_fields)
    if has_all_fields:
        factors['completeness'] = 1.0
    else:
        missing = [f for f in required_fields if f not in grading_result]
        factors['completeness'] = 0.5
        confidence *= 0.5
        factors['missing_fields'] = missing
    
    # Rationale quality
    rationale = grading_result.get('rationale', '')
    if len(rationale) > 100:
        factors['rationale_length'] = 1.0
    elif len(rationale) > 50:
        factors['rationale_length'] = 0.8
        confidence *= 0.95
    else:
        factors['rationale_length'] = 0.5
        confidence *= 0.8
    
    return {
        'overall_confidence': round(confidence, 4),
        'factors': factors,
        'confidence_level': 'high' if confidence >= 0.9 else 'medium' if confidence >= 0.7 else 'low',
    }


class TaskAgent:
    """Task agent for grading mathematical problems with chain-of-thought reasoning.
    
    This agent uses a two-step process:
    1. Initial grading with detailed chain-of-thought analysis
    2. Self-reflection to verify and potentially revise the grade
    
    The agent supports both simple grading (forward) and enhanced grading
    with validation and accuracy metrics (grade_with_validation).
    
    Improvements:
    - Input validation for better error handling
    - Response caching for efficiency
    - Confidence scoring for quality assessment
    - Retry mechanism for failed JSON extraction
    """

    def __init__(self, model: str = EVAL_MODEL, temperature: float = 0.0, use_cache: bool = True) -> None:
        self.model = model
        self.temperature = temperature
        self.log_fn = logger.info
        self.use_cache = use_cache
        self._grading_stats = {
            'total': 0,
            'successful': 0,
            'failed': 0,
            'cache_hits': 0,
        }

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Grade a student's answer to a mathematical problem.

        Args:
            inputs: Dictionary containing 'problem', 'solution', 'grading_guidelines',
                   'student_answer', and optionally 'domain'

        Returns:
            Tuple of (prediction string, message history)
        """
        self._grading_stats['total'] += 1
        
        # Validate inputs first
        is_valid, error_msg = validate_grading_inputs(inputs)
        if not is_valid:
            self.log_fn(f"Input validation failed: {error_msg}")
            self._grading_stats['failed'] += 1
            return f"Error: {error_msg}", [{"role": "system", "text": f"Input validation failed: {error_msg}"}]
        
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

        # Try cache first if enabled
        cached_result = None
        if self.use_cache and self.temperature == 0.0:
            cached_result = get_cached_response(instruction, self.model, self.temperature)
            if cached_result:
                self._grading_stats['cache_hits'] += 1
                self.log_fn(f"Cache hit! Using cached response.")
                response, msg_history, info = cached_result
            else:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    temperature=self.temperature,
                    msg_history=[],
                )
                set_cached_response(instruction, self.model, self.temperature, (response, msg_history, info))
        else:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                temperature=self.temperature,
                msg_history=[],
            )

        # Extract prediction from JSON with detailed logging
        prediction = "None"
        extraction_method = "none"
        initial_result = None
        
        try:
            last_msg = msg_history[-1]["text"]
            extracted = _extract_jsons(last_msg)
            if extracted:
                result = extracted[-1]
                initial_result = result
                extraction_method = "json"
                self.log_fn(f"Extracted JSON result: {result}")
                if "response" in result:
                    prediction = result["response"]
                    self.log_fn(f"Using 'response' field: {prediction}")
                elif "score" in result and "max_score" in result:
                    prediction = f"{result['score']}/{result['max_score']}"
                    self.log_fn(f"Using score/max_score fields: {prediction}")
                else:
                    self.log_fn(f"Warning: JSON missing expected fields. Keys: {list(result.keys())}")
            else:
                self.log_fn("Warning: No JSON blocks found in response")
                # Try to extract any numeric score pattern as fallback
                score_match = re.search(r'(\d+)\s*/\s*(\d+)', last_msg)
                if score_match:
                    prediction = f"{score_match.group(1)}/{score_match.group(2)}"
                    extraction_method = "fallback"
                    self.log_fn(f"Fallback extraction: {prediction}")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            import traceback
            self.log_fn(f"Traceback: {traceback.format_exc()}")

        # Step 2: Self-reflection to verify the grade
        reflection_consistent = None
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
            
            reflection_response, msg_history, _ = get_response_from_llm(
                msg=reflection_msg,
                model=self.model,
                temperature=self.temperature,
                msg_history=msg_history,
            )
            
            # Try to extract revised prediction with detailed logging
            try:
                last_msg = msg_history[-1]["text"]
                extracted = _extract_jsons(last_msg)
                if extracted:
                    result = extracted[-1]
                    self.log_fn(f"Reflection extracted JSON: {result}")
                    if "final_response" in result:
                        new_prediction = result["final_response"]
                        reflection_consistent = (new_prediction == prediction)
                        prediction = new_prediction
                        self.log_fn(f"Using 'final_response' field: {prediction}")
                    elif "revised_score" in result and "revised_max_score" in result:
                        new_prediction = f"{result['revised_score']}/{result['revised_max_score']}"
                        reflection_consistent = (new_prediction == prediction)
                        prediction = new_prediction
                        self.log_fn(f"Using revised_score/revised_max_score: {prediction}")
                    else:
                        self.log_fn(f"Warning: Reflection JSON missing expected fields. Keys: {list(result.keys())}")
                        reflection_consistent = True  # Assume consistent if no revision
                else:
                    self.log_fn("Warning: No JSON found in reflection response")
                    reflection_consistent = None
            except Exception as e:
                self.log_fn(f"Error extracting revised prediction: {e}")
                import traceback
                self.log_fn(f"Traceback: {traceback.format_exc()}")
                reflection_consistent = None

        if prediction != "None":
            self._grading_stats['successful'] += 1
        else:
            self._grading_stats['failed'] += 1

        return str(prediction), msg_history

    def grade_with_validation(self, inputs: dict, reference_score: float | None = None) -> dict:
        """Grade a student's answer with enhanced validation and metrics.
        
        Args:
            inputs: Dictionary containing grading inputs
            reference_score: Optional reference score for accuracy calculation
            
        Returns:
            Dictionary with prediction, validation results, accuracy metrics, and confidence
        """
        start_time = time.time()
        prediction, msg_history = self.forward(inputs)
        processing_time = time.time() - start_time
        
        result = {
            "prediction": prediction,
            "msg_history": msg_history,
            "is_valid": False,
            "validation_error": None,
            "accuracy_metrics": None,
            "confidence": None,
            "processing_time": round(processing_time, 4),
            "cache_stats": get_cache_stats() if self.use_cache else None,
        }
        
        # Try to extract and validate the grading result
        extraction_method = "none"
        initial_result = None
        reflection_consistent = None
        
        try:
            # Find the last valid JSON in the history
            for msg in reversed(msg_history):
                if msg.get("role") == "assistant":
                    text = msg.get("text", "")
                    extracted = _extract_jsons(text)
                    if extracted:
                        grading_result = extracted[-1]
                        initial_result = grading_result
                        extraction_method = "json"
                        
                        is_valid, error_msg = validate_grading_result(grading_result)
                        result["is_valid"] = is_valid
                        result["validation_error"] = error_msg if not is_valid else None
                        
                        if is_valid and "score" in grading_result and "max_score" in grading_result:
                            result["accuracy_metrics"] = compute_grading_accuracy(
                                float(grading_result["score"]),
                                float(grading_result["max_score"]),
                                reference_score
                            )
                        
                        # Check for reflection consistency
                        if len(msg_history) >= 4:  # Initial + reflection
                            reflection_consistent = True  # Default if we got here
                        
                        break
            
            # Compute confidence if we have a result
            if initial_result:
                result["confidence"] = compute_grading_confidence(
                    initial_result,
                    extraction_method,
                    has_reflection=len(msg_history) >= 4,
                    reflection_consistent=reflection_consistent,
                )
                
        except Exception as e:
            result["validation_error"] = f"Error during validation: {e}"
            import traceback
            logger.debug(f"Validation error traceback: {traceback.format_exc()}")
        
        return result
    
    def get_stats(self) -> dict:
        """Get grading statistics for this agent instance."""
        stats = dict(self._grading_stats)
        if stats['total'] > 0:
            stats['success_rate'] = round(stats['successful'] / stats['total'], 4)
        else:
            stats['success_rate'] = 0.0
        return stats
    
    def reset_stats(self) -> None:
        """Reset grading statistics."""
        self._grading_stats = {
            'total': 0,
            'successful': 0,
            'failed': 0,
            'cache_hits': 0,
        }


def batch_grade(
    agent: TaskAgent,
    problems: list[dict],
    reference_scores: list[float] | None = None,
    continue_on_error: bool = True,
    progress_callback: callable | None = None,
) -> list[dict]:
    """Grade multiple problems in batch with enhanced error handling.
    
    Args:
        agent: TaskAgent instance to use for grading
        problems: List of input dictionaries for each problem
        reference_scores: Optional list of reference scores for accuracy calculation
        continue_on_error: If True, continue grading remaining problems after errors
        progress_callback: Optional callback function(current, total, result) for progress updates
        
    Returns:
        List of grading results with validation and metrics
    """
    results = []
    errors = []
    
    for i, problem in enumerate(problems):
        try:
            ref_score = reference_scores[i] if reference_scores and i < len(reference_scores) else None
            result = agent.grade_with_validation(problem, ref_score)
            results.append(result)
            
            if progress_callback:
                try:
                    progress_callback(i + 1, len(problems), result)
                except Exception as e:
                    logger.warning(f"Progress callback error: {e}")
                    
        except Exception as e:
            error_msg = f"Error grading problem {i}: {e}"
            logger.error(error_msg)
            errors.append({"index": i, "error": str(e)})
            
            if continue_on_error:
                # Add error result placeholder
                results.append({
                    "prediction": "Error",
                    "msg_history": [],
                    "is_valid": False,
                    "validation_error": error_msg,
                    "accuracy_metrics": None,
                    "confidence": None,
                })
            else:
                raise
    
    # Log summary
    total = len(problems)
    successful = sum(1 for r in results if r.get("is_valid", False))
    failed = total - successful
    
    logger.info(f"Batch grading complete: {successful}/{total} successful, {failed} failed, {len(errors)} errors")
    
    return results


def compare_grades(
    agent1: TaskAgent,
    agent2: TaskAgent,
    problems: list[dict],
    reference_scores: list[float] | None = None,
) -> dict:
    """Compare grading results between two agents.
    
    Args:
        agent1: First TaskAgent instance
        agent2: Second TaskAgent instance
        problems: List of input dictionaries for each problem
        reference_scores: Optional list of reference scores
        
    Returns:
        Dictionary with comparison metrics
    """
    results1 = batch_grade(agent1, problems, reference_scores)
    results2 = batch_grade(agent2, problems, reference_scores)
    
    agreements = 0
    disagreements = 0
    agent1_better = 0
    agent2_better = 0
    
    for r1, r2 in zip(results1, results2):
        pred1 = r1.get("prediction", "")
        pred2 = r2.get("prediction", "")
        
        # Extract scores from predictions
        score1_match = re.search(r'(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)', str(pred1))
        score2_match = re.search(r'(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)', str(pred2))
        
        if score1_match and score2_match:
            s1 = float(score1_match.group(1))
            s2 = float(score2_match.group(1))
            
            if abs(s1 - s2) < 0.001:
                agreements += 1
            else:
                disagreements += 1
                
            # Compare against reference if available
            ref = r1.get("accuracy_metrics", {}).get("reference_score")
            if ref is not None:
                err1 = abs(s1 - ref)
                err2 = abs(s2 - ref)
                if err1 < err2:
                    agent1_better += 1
                elif err2 < err1:
                    agent2_better += 1
    
    total = len(problems)
    
    return {
        "total_problems": total,
        "agreements": agreements,
        "disagreements": disagreements,
        "agreement_rate": round(agreements / total, 4) if total > 0 else 0,
        "agent1_better": agent1_better,
        "agent2_better": agent2_better,
        "agent1_stats": agent1.get_stats(),
        "agent2_stats": agent2.get_stats(),
    }
