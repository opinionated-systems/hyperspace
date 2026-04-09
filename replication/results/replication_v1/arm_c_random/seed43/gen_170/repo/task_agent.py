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
import re
import time
from dataclasses import dataclass
from typing import Any

from agent.llm_client import get_response_from_llm, EVAL_MODEL

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with TTL support."""
    result: tuple[str, list[dict]]
    timestamp: float
    access_count: int = 1


# Simple in-memory cache for LLM responses to avoid redundant calls
_response_cache: dict[str, CacheEntry] = {}
_CACHE_TTL_SECONDS = 3600  # 1 hour TTL
_MAX_CACHE_SIZE = 500  # Reduced from 1000 for better memory management


def _get_cache_key(inputs: dict, model: str) -> str:
    """Generate a cache key from inputs and model."""
    key_data = json.dumps(inputs, sort_keys=True) + model
    return hashlib.sha256(key_data.encode()).hexdigest()[:32]


def clear_response_cache() -> None:
    """Clear the response cache. Useful for testing or memory management."""
    global _response_cache
    _response_cache.clear()
    logger.info("Response cache cleared")


def get_cache_stats() -> dict[str, Any]:
    """Get cache statistics for monitoring and debugging."""
    global _response_cache
    if not _response_cache:
        return {"size": 0, "hit_rate": 0.0, "avg_access_count": 0.0}
    
    total_accesses = sum(entry.access_count for entry in _response_cache.values())
    avg_access = total_accesses / len(_response_cache) if _response_cache else 0
    
    return {
        "size": len(_response_cache),
        "max_size": _MAX_CACHE_SIZE,
        "ttl_seconds": _CACHE_TTL_SECONDS,
        "total_accesses": total_accesses,
        "avg_access_count": round(avg_access, 2),
    }


def _cleanup_expired_cache_entries() -> None:
    """Remove expired cache entries based on TTL."""
    global _response_cache
    current_time = time.time()
    expired_keys = [
        key for key, entry in _response_cache.items()
        if current_time - entry.timestamp > _CACHE_TTL_SECONDS
    ]
    for key in expired_keys:
        del _response_cache[key]
    if expired_keys:
        logger.debug(f"Removed {len(expired_keys)} expired cache entries")


def _evict_lru_cache_entry() -> None:
    """Evict least recently used cache entry when size exceeds limit."""
    global _response_cache
    if len(_response_cache) >= _MAX_CACHE_SIZE:
        # Find entry with lowest access count and oldest timestamp
        lru_key = min(
            _response_cache.keys(),
            key=lambda k: (_response_cache[k].access_count, _response_cache[k].timestamp)
        )
        del _response_cache[lru_key]
        logger.debug(f"Evicted LRU cache entry: {lru_key[:16]}...")


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    Also handles markdown code blocks and raw JSON objects.
    Includes enhanced error recovery for common JSON formatting issues.
    """
    results = []
    search_from = 0
    
    # First, try to find <json>...</json> blocks
    while True:
        start = text.find("<json>", search_from)
        if start == -1:
            break
        end = text.find("</json>", start)
        if end == -1:
            break
        inner = text[start + 6:end].strip()
        search_from = end + 7
        
        # Try multiple parsing strategies
        parsed = _try_parse_json(inner)
        if parsed:
            results.append(parsed)
    
    # If no <json> blocks found, try markdown code blocks
    if not results:
        md_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
        for match in re.finditer(md_pattern, text, re.DOTALL):
            parsed = _try_parse_json(match.group(1).strip())
            if parsed:
                results.append(parsed)
    
    # Last resort: try to find any JSON-like structure with "response" field
    if not results:
        # Look for objects with "response" key - more permissive pattern
        json_pattern = r'\{[^{}]*"response"[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        for match in re.finditer(json_pattern, text):
            parsed = _try_parse_json(match.group())
            if parsed:
                results.append(parsed)
    
    # Final fallback: try to find any valid JSON object
    if not results:
        # Look for balanced braces pattern
        brace_pattern = r'\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}'
        for match in re.finditer(brace_pattern, text):
            parsed = _try_parse_json(match.group())
            if parsed and "response" in parsed:
                results.append(parsed)
                break  # Only take the first valid one with response field
    
    return results or None


def _try_parse_json(text: str) -> dict | None:
    """Attempt to parse JSON with multiple recovery strategies.
    
    Returns the parsed dict if successful, None otherwise.
    """
    # Strategy 1: Direct parsing
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Strategy 2: Remove trailing commas before closing braces/brackets
    try:
        cleaned = re.sub(r',(\s*[}\]])', r'\1', text)
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    
    # Strategy 3: Fix single quotes to double quotes (common LLM error)
    try:
        # Replace single quotes around keys and string values
        fixed = re.sub(r"'([^']*?)'\s*:", r'"\1":', text)
        fixed = re.sub(r":\s*'([^']*?)'", r': "\1"', fixed)
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    
    # Strategy 4: Fix unescaped newlines in string values
    try:
        # Replace newlines within string values with escaped newlines
        fixed = re.sub(r'("[^"]*?)\n([^"]*?")', r'\1\\n\2', text)
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    
    # Strategy 5: Extract just the response field if present
    try:
        response_match = re.search(r'"response"\s*:\s*"([^"]*?)"', text)
        if response_match:
            return {"response": response_match.group(1)}
    except Exception:
        pass
    
    # Strategy 6: Extract response with single quotes
    try:
        response_match = re.search(r"'response'\s*:\s*'([^']*?)'", text)
        if response_match:
            return {"response": response_match.group(1)}
    except Exception:
        pass
    
    return None


class TaskAgent:
    """Task agent that solves IMO grading problems with enhanced reasoning and caching."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "", use_cache: bool = True) -> None:
        self.model = model
        self.log_fn = logger.info
        self.use_cache = use_cache

    def _build_prompt(self, inputs: dict) -> str:
        """Build a structured prompt for IMO grading with chain-of-thought reasoning."""
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        # Truncate very long inputs to prevent token overflow
        max_len = 8000
        problem = problem[:max_len] if len(problem) > max_len else problem
        solution = solution[:max_len] if len(solution) > max_len else solution
        student_answer = student_answer[:max_len] if len(student_answer) > max_len else student_answer
        grading_guidelines = grading_guidelines[:max_len] if len(grading_guidelines) > max_len else grading_guidelines

        return f"""You are an expert IMO (International Mathematical Olympiad) grader.

Your task is to evaluate a student's solution to a mathematical problem.

## Problem Domain
{domain}

## Problem Statement
{problem}

## Official Solution
{solution}

## Grading Guidelines
{grading_guidelines}

## Student's Answer
{student_answer}

## Instructions

1. First, analyze the student's answer step by step. Compare it against the official solution.
2. Identify any errors, missing steps, or creative alternative approaches.
3. Consider the grading guidelines carefully.
4. Provide your reasoning before giving the final grade.

Respond ONLY in JSON format with the following schema (no other text before or after):
<json>
{{
    "reasoning": "Your detailed analysis of the student's answer...",
    "response": "Your final grade/evaluation (e.g., '7', '6', 'Partial credit: 3', 'Incorrect', etc.)"
}}
</json>

The "response" field should contain only the final grade/evaluation, while "reasoning" contains your full analysis."""

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with enhanced reasoning and caching.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        global _response_cache
        
        # Cleanup expired entries periodically
        if self.use_cache and len(_response_cache) % 100 == 0:
            _cleanup_expired_cache_entries()
        
        # Check cache first if enabled
        if self.use_cache:
            cache_key = _get_cache_key(inputs, self.model)
            if cache_key in _response_cache:
                entry = _response_cache[cache_key]
                # Check if entry is still valid
                if time.time() - entry.timestamp <= _CACHE_TTL_SECONDS:
                    entry.access_count += 1
                    self.log_fn(f"Cache hit for problem: {inputs.get('problem', '')[:50]}...")
                    return entry.result
                else:
                    # Entry expired, remove it
                    del _response_cache[cache_key]
        
        instruction = self._build_prompt(inputs)

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON with better error handling
        prediction = "None"
        reasoning = ""
        try:
            # Try to extract from the last assistant message
            last_msg = msg_history[-1]["text"] if msg_history else ""
            extracted = _extract_jsons(last_msg)
            
            if extracted:
                last_json = extracted[-1]
                if "response" in last_json:
                    prediction = last_json["response"]
                if "reasoning" in last_json:
                    reasoning = last_json["reasoning"]
                
                # Log the reasoning for debugging
                if reasoning:
                    self.log_fn(f"Reasoning: {reasoning[:200]}...")
            else:
                # Fallback: try to find any JSON-like structure
                json_match = re.search(r'\{[^}]*"response"[^}]*\}', last_msg)
                if json_match:
                    try:
                        fallback = json.loads(json_match.group())
                        prediction = fallback.get("response", "None")
                    except json.JSONDecodeError:
                        pass
                        
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        result = (str(prediction), msg_history)
        
        # Store in cache if enabled
        if self.use_cache:
            # Evict LRU entry if cache is full
            _evict_lru_cache_entry()
            
            cache_key = _get_cache_key(inputs, self.model)
            _response_cache[cache_key] = CacheEntry(
                result=result,
                timestamp=time.time(),
                access_count=1
            )
        
        return result
