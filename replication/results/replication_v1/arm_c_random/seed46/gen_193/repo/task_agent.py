"""
Task agent: solves a given task with a single LLM call.

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
import os
import re
from pathlib import Path

from agent.llm_client import get_response_from_llm, EVAL_MODEL

logger = logging.getLogger(__name__)

# Cache configuration
_CACHE_DIR = os.environ.get("TASK_AGENT_CACHE_DIR", "")
_cache_hits = 0
_cache_misses = 0


def _get_cache_path(inputs: dict, model: str) -> Path | None:
    """Generate cache file path for given inputs.
    
    Uses a stable hash of the inputs and model name for the cache key.
    Includes a version prefix to invalidate cache if extraction logic changes.
    """
    if not _CACHE_DIR:
        return None
    # Include extraction version in cache key for invalidation on logic changes
    cache_inputs = {
        **inputs,
        "_cache_version": "v2",  # Bump when extraction logic changes
    }
    cache_key = hashlib.sha256(
        json.dumps(cache_inputs, sort_keys=True, default=str).encode()
    ).hexdigest()
    cache_dir = Path(_CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{model.replace('/', '_')}_{cache_key}.json"


def _load_from_cache(cache_path: Path) -> tuple[str, list[dict]] | None:
    """Load cached response if available."""
    global _cache_hits
    if cache_path.exists():
        try:
            with open(cache_path) as f:
                data = json.load(f)
            _cache_hits += 1
            logger.info(f"Cache hit ({_cache_hits} total hits, {_cache_misses} misses)")
            return data["prediction"], data["msg_history"]
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
    return None


def _save_to_cache(cache_path: Path, prediction: str, msg_history: list[dict]) -> None:
    """Save response to cache."""
    global _cache_misses
    _cache_misses += 1
    try:
        with open(cache_path, "w") as f:
            json.dump({
                "prediction": prediction,
                "msg_history": msg_history,
            }, f, default=str)
        logger.info(f"Cache miss - saved to cache ({_cache_hits} hits, {_cache_misses} misses)")
    except Exception as e:
        logger.warning(f"Failed to save cache: {e}")


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    Improved to handle nested JSON structures and whitespace better.
    """
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
        
        # Try to parse the JSON content
        try:
            results.append(json.loads(inner))
        except json.JSONDecodeError as e:
            # Try to clean up common issues before giving up
            # Remove trailing commas before closing braces/brackets
            cleaned = re.sub(r',(\s*[}\]])', r'\1', inner)
            # Fix single quotes to double quotes (common LLM mistake)
            cleaned = re.sub(r"(?<!\\)'", '"', cleaned)
            try:
                results.append(json.loads(cleaned))
                logger.debug(f"JSON extraction succeeded after cleanup: {e}")
            except json.JSONDecodeError as e2:
                logger.debug(f"JSON extraction failed even after cleanup: {e2}")
                continue
    return results or None


def _log_extraction_failure(text: str, error: Exception, method: str) -> None:
    """Log detailed information about JSON extraction failures for debugging.
    
    This helps identify patterns in LLM output that cause extraction issues.
    """
    preview_len = min(200, len(text))
    text_preview = text[:preview_len] + "..." if len(text) > preview_len else text
    logger.warning(
        f"JSON extraction failed using {method} method. "
        f"Error: {error}. "
        f"Text preview: {repr(text_preview)}"
    )


def _extract_json_fallback(text: str) -> list[dict] | None:
    """Fallback JSON extraction using regex for unwrapped JSON objects.

    This handles cases where the model outputs valid JSON without <json> tags.
    Searches for JSON objects with a "response" key.
    Improved to handle nested braces more robustly.
    """
    results = []
    
    # First, try to find JSON objects by parsing brace depth
    # This handles nested structures better than regex
    def find_json_objects(s: str) -> list[str]:
        """Find potential JSON objects by tracking brace depth."""
        objects = []
        i = 0
        while i < len(s):
            if s[i] == '{':
                start = i
                depth = 1
                i += 1
                while i < len(s) and depth > 0:
                    if s[i] == '{':
                        depth += 1
                    elif s[i] == '}':
                        depth -= 1
                    i += 1
                if depth == 0:
                    objects.append(s[start:i])
            else:
                i += 1
        return objects
    
    # Try to parse each potential JSON object
    for obj_str in find_json_objects(text):
        try:
            obj = json.loads(obj_str)
            if isinstance(obj, dict) and "response" in obj:
                results.append(obj)
        except json.JSONDecodeError as e:
            logger.debug(f"Fallback brace-depth parsing failed for object: {e}")
            continue
    
    # If no results, try regex pattern for simpler cases
    if not results:
        # Pattern to match JSON objects with response key (simpler cases)
        pattern = r'\{[^{}]*"response"[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.finditer(pattern, text, re.DOTALL)
        for match in matches:
            try:
                obj = json.loads(match.group())
                if "response" in obj:
                    results.append(obj)
            except json.JSONDecodeError as e:
                logger.debug(f"Fallback regex parsing failed: {e}")
                continue

    # If still no results, try to parse the entire text as JSON
    if not results:
        try:
            obj = json.loads(text.strip())
            if isinstance(obj, dict) and "response" in obj:
                results.append(obj)
        except json.JSONDecodeError as e:
            logger.debug(f"Full text JSON parsing failed: {e}")
    
    # Final fallback: try to extract any dict-like structure with response key
    if not results:
        # Look for patterns like "response": "..." or 'response': '...'
        response_pattern = r'["\']response["\']\s*:\s*["\']([^"\']+)["\']'
        match = re.search(response_pattern, text)
        if match:
            results.append({"response": match.group(1)})
            logger.debug("Final fallback pattern match succeeded")
        else:
            _log_extraction_failure(text, Exception("All fallback methods exhausted"), "fallback")

    return results or None


def get_cache_stats() -> dict:
    """Get cache hit/miss statistics."""
    return {
        "hits": _cache_hits,
        "misses": _cache_misses,
        "hit_rate": _cache_hits / (_cache_hits + _cache_misses) if (_cache_hits + _cache_misses) > 0 else 0.0,
    }


class TaskAgent:
    """Task agent that solves IMO grading problems with robust JSON extraction and caching."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self._extraction_stats = {"primary": 0, "fallback": 0, "failed": 0}

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Check cache first
        cache_path = _get_cache_path(inputs, self.model)
        if cache_path:
            cached = _load_from_cache(cache_path)
            if cached:
                return cached

        instruction = f"""You are an agent.

Task input:
```
{inputs}
```

Respond in JSON format with the following schema:
<json>
{{
    "response": ...
}}
</json>"""

        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
        except Exception as e:
            self.log_fn(f"Error calling LLM: {e}")
            return "Error: LLM call failed", [{"role": "system", "text": f"Error: {e}"}]

        # Check if we got a valid response
        if not msg_history or len(msg_history) < 2:
            self.log_fn("Warning: Empty or incomplete message history from LLM")
            return "Error: No response from LLM", msg_history if msg_history else [{"role": "system", "text": "No response"}]

        # Extract prediction from JSON using primary method
        prediction = "None"
        extraction_method = "primary"
        raw_response = msg_history[-1].get("text", "")
        
        if not raw_response or not raw_response.strip():
            self.log_fn("Warning: Empty response text from LLM")
            self._extraction_stats["failed"] += 1
            return "Error: Empty response from LLM", msg_history
        
        try:
            extracted = _extract_jsons(raw_response)
            if extracted and len(extracted) > 0 and "response" in extracted[-1]:
                prediction = extracted[-1]["response"]
                self._extraction_stats["primary"] += 1
                self.log_fn(f"Primary extraction succeeded, response type: {type(prediction).__name__}")
            else:
                # Try fallback extraction
                self.log_fn("Primary extraction failed, trying fallback...")
                extracted = _extract_json_fallback(raw_response)
                if extracted and len(extracted) > 0 and "response" in extracted[-1]:
                    prediction = extracted[-1]["response"]
                    extraction_method = "fallback"
                    self._extraction_stats["fallback"] += 1
                    self.log_fn(f"Fallback extraction succeeded, response type: {type(prediction).__name__}")
                else:
                    self._extraction_stats["failed"] += 1
                    self.log_fn("Warning: No valid JSON with 'response' key found in LLM output")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            # Try fallback on exception
            try:
                extracted = _extract_json_fallback(raw_response)
                if extracted and len(extracted) > 0 and "response" in extracted[-1]:
                    prediction = extracted[-1]["response"]
                    extraction_method = "fallback"
                    self._extraction_stats["fallback"] += 1
                    self.log_fn(f"Fallback extraction succeeded after exception, response type: {type(prediction).__name__}")
                else:
                    self._extraction_stats["failed"] += 1
            except Exception as fallback_e:
                self._extraction_stats["failed"] += 1
                self.log_fn(f"Fallback extraction also failed: {fallback_e}")

        self.log_fn(f"Extraction method used: {extraction_method} (stats: {self._extraction_stats})")
        
        # Convert prediction to string, handling various types
        if prediction is None:
            prediction = "None"
        elif isinstance(prediction, (list, dict)):
            prediction = json.dumps(prediction)
        else:
            prediction = str(prediction)
        
        # Save to cache if enabled
        if cache_path:
            _save_to_cache(cache_path, prediction, msg_history)
        
        return prediction, msg_history

    def get_extraction_stats(self) -> dict:
        """Get extraction method statistics for this agent instance."""
        total = sum(self._extraction_stats.values())
        if total == 0:
            return {**self._extraction_stats, "success_rate": 0.0}
        success_rate = (self._extraction_stats["primary"] + self._extraction_stats["fallback"]) / total
        return {**self._extraction_stats, "success_rate": success_rate}

    def reset_extraction_stats(self) -> None:
        """Reset extraction method statistics for this agent instance."""
        self._extraction_stats = {"primary": 0, "fallback": 0, "failed": 0}
        self.log_fn("Extraction statistics reset")

    def clear_cache(self) -> None:
        """Clear the cache directory if caching is enabled."""
        global _cache_hits, _cache_misses
        if _CACHE_DIR:
            cache_path = Path(_CACHE_DIR)
            if cache_path.exists():
                import shutil
                try:
                    shutil.rmtree(cache_path)
                    cache_path.mkdir(parents=True, exist_ok=True)
                    self.log_fn(f"Cache cleared at {cache_path}")
                except Exception as e:
                    self.log_fn(f"Error clearing cache: {e}")
        # Reset cache statistics
        _cache_hits = 0
        _cache_misses = 0
