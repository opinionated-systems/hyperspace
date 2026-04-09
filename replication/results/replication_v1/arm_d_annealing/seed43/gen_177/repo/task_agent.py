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
from functools import lru_cache

from agent.llm_client import get_response_from_llm, EVAL_MODEL

logger = logging.getLogger(__name__)


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    Also handles nested JSON objects and common formatting issues.
    Includes enhanced recovery for malformed JSON responses.
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
        
        # Try to parse the JSON
        try:
            results.append(json.loads(inner))
        except json.JSONDecodeError:
            # Try common fixes: remove trailing commas, fix quotes
            fixed = _fix_json_string(inner)
            try:
                results.append(json.loads(fixed))
            except json.JSONDecodeError:
                # Try extracting just the outermost JSON object
                try:
                    obj = _extract_outermost_json(inner)
                    if obj:
                        results.append(obj)
                except Exception:
                    # Final fallback: try to extract key-value pairs manually
                    try:
                        obj = _extract_key_value_pairs(inner)
                        if obj:
                            results.append(obj)
                    except Exception:
                        continue
    return results or None


def _extract_key_value_pairs(text: str) -> dict | None:
    """Extract key-value pairs from malformed JSON as a last resort.
    
    Looks for patterns like "key": "value" or "key": "multi
    line value" and constructs a valid dict.
    """
    result = {}
    
    # Pattern to match "key": "value" pairs, handling multiline values
    # This is a best-effort approach for recovery
    import re
    
    # Try to find response field
    response_patterns = [
        r'"response"\s*:\s*"([^"]*)"',
        r'"response"\s*:\s*"((?:[^"\\]|\\.)*)"',
        r"'response'\s*:\s*'([^']*)'",
        r'response["\']?\s*:\s*["\']?([^"\'\n,}]+)',
    ]
    
    for pattern in response_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            result["response"] = match.group(1).strip()
            break
    
    # Try to find reasoning field - capture everything until we hit another key
    reasoning_patterns = [
        r'"reasoning"\s*:\s*"(.*?)"\s*,?\s*"',
        r'"reasoning"\s*:\s*"((?:[^"\\]|\\.)*?)"',
        r'"reasoning"\s*:\s*"([^"]*)"',
    ]
    
    for pattern in reasoning_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            reasoning = match.group(1).strip()
            # Clean up escaped characters
            reasoning = reasoning.replace('\\n', '\n').replace('\\t', '\t').replace('\\"', '"')
            result["reasoning"] = reasoning
            break
    
    # If we found at least a response, return the result
    if "response" in result:
        return result
    
    return None


def _normalize_grade(prediction: str) -> str:
    """Normalize various grade formats to standard values.
    
    Handles numeric scores, case variations, and common synonyms.
    """
    if not prediction:
        return "None"
    
    pred_lower = str(prediction).lower().strip()
    
    # Handle numeric scores (0-100 scale)
    try:
        score = float(prediction)
        if score >= 90:
            return "Correct"
        elif score >= 60:
            return "Partial"
        else:
            return "Incorrect"
    except (ValueError, TypeError):
        pass
    
    # Handle letter grades
    grade_map = {
        "a": "Correct",
        "b": "Partial", 
        "c": "Partial",
        "d": "Incorrect",
        "f": "Incorrect",
    }
    if pred_lower in grade_map:
        return grade_map[pred_lower]
    
    # Handle common variations
    if any(x in pred_lower for x in ["correct", "right", "accurate", "valid", "true", "yes", "pass", "full", "complete"]):
        return "Correct"
    elif any(x in pred_lower for x in ["partial", "some", "incomplete", "partly", "half", "partially correct", "partial credit"]):
        return "Partial"
    elif any(x in pred_lower for x in ["incorrect", "wrong", "error", "invalid", "false", "no", "fail", "none", "missing"]):
        return "Incorrect"
    
    # Return original if no normalization applied
    return prediction


def _fix_json_string(text: str) -> str:
    """Apply common JSON fixes."""
    # Remove trailing commas before closing braces/brackets
    import re
    text = re.sub(r',(\s*[}\]])', r'\1', text)
    # Fix single quotes to double quotes (basic)
    text = text.replace("'", '"')
    return text


def _extract_outermost_json(text: str) -> dict | None:
    """Extract the outermost JSON object from text, handling nesting."""
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
                try:
                    return json.loads(text[start_idx:i+1])
                except json.JSONDecodeError:
                    continue
    return None


def _extract_json_with_regex(text: str) -> list[dict] | None:
    """Fallback JSON extraction using regex for malformed responses."""
    results = []
    
    # Try to find JSON objects in code blocks
    json_pattern = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if json_pattern:
        try:
            results.append(json.loads(json_pattern.group(1)))
        except json.JSONDecodeError:
            # Try with fixes
            try:
                fixed = _fix_json_string(json_pattern.group(1))
                results.append(json.loads(fixed))
            except json.JSONDecodeError:
                pass
    
    # Try to find any JSON-like structure with "response" key
    response_pattern = re.search(r'"response"\s*:\s*"([^"]*)"', text)
    if response_pattern and not results:
        results.append({"response": response_pattern.group(1)})
    
    # Try to find JSON without code blocks (look for { ... } patterns)
    if not results:
        # Find all potential JSON objects
        potential_jsons = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
        for pj in potential_jsons:
            try:
                obj = json.loads(pj)
                if "response" in obj or "reasoning" in obj:
                    results.append(obj)
            except json.JSONDecodeError:
                try:
                    fixed = _fix_json_string(pj)
                    obj = json.loads(fixed)
                    if "response" in obj or "reasoning" in obj:
                        results.append(obj)
                except json.JSONDecodeError:
                    continue
    
    return results or None


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.max_retries = 3
        self._extraction_stats = {"success": 0, "fallback": 0, "failed": 0}
        self._response_cache: dict[str, tuple[str, str]] = {}
        self._cache_hits = 0
        self._cache_misses = 0

    def _get_cache_key(self, inputs: dict) -> str:
        """Generate a cache key from inputs."""
        # Hash the key inputs that determine the grading result
        key_data = {
            "problem": inputs.get("problem", ""),
            "solution": inputs.get("solution", ""),
            "student_answer": inputs.get("student_answer", ""),
            "guidelines": inputs.get("grading_guidelines", ""),
        }
        return hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()

    def _get_cached_response(self, inputs: dict) -> tuple[str, str] | None:
        """Check for cached response."""
        cache_key = self._get_cache_key(inputs)
        if cache_key in self._response_cache:
            self._cache_hits += 1
            self.log_fn(f"Cache hit! (hits: {self._cache_hits}, misses: {self._cache_misses})")
            return self._response_cache[cache_key]
        self._cache_misses += 1
        return None

    def _cache_response(self, inputs: dict, prediction: str, reasoning: str) -> None:
        """Cache a response for future use."""
        cache_key = self._get_cache_key(inputs)
        # Limit cache size to prevent memory issues
        if len(self._response_cache) >= 1000:
            # Clear oldest entries (simple approach: clear half)
            keys = list(self._response_cache.keys())[:500]
            for k in keys:
                del self._response_cache[k]
        self._response_cache[cache_key] = (prediction, reasoning)

    def _build_grading_prompt(self, inputs: dict) -> str:
        """Build a structured prompt for the grading task with chain-of-thought."""
        domain = inputs.get("domain", "unknown")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        return f"""You are an expert grader for {domain} problems. Your task is to evaluate a student's answer against the correct solution and grading guidelines.

## Problem Statement
{problem}

## Correct Solution
{solution}

## Grading Guidelines
{guidelines}

## Student's Answer
{student_answer}

## Instructions
1. First, analyze the student's answer step by step. Compare it against the correct solution.
2. Identify any errors, omissions, or alternative valid approaches.
3. Consider the grading guidelines carefully.
4. Provide your reasoning for the grade you will assign.
5. Finally, provide your grade/assessment in the JSON format below.

## Grading Rubric
When assigning grades, consider:
- **Correct**: The answer matches the solution or uses an equivalent valid approach with correct reasoning and final result.
- **Partial**: The answer shows some correct reasoning but has minor errors, incomplete steps, or partially correct results.
- **Incorrect**: The answer contains fundamental errors, wrong approach, or completely wrong results.

## Response Format (REQUIRED)
You MUST respond with valid JSON wrapped in <json>...</json> tags. Do not include any text outside the JSON tags.

<json>
{{
    "reasoning": "Your detailed step-by-step analysis and reasoning...",
    "response": "Your final grade/assessment (e.g., 'Correct', 'Partial', 'Incorrect', or a numeric score)"
}}
</json>

IMPORTANT: Ensure your JSON is valid and properly formatted. The 'response' field should contain only the grade, not the reasoning."""

    def _extract_prediction(self, text: str) -> tuple[str, str]:
        """Extract prediction and reasoning from response text.
        
        Returns:
            (prediction, reasoning) tuple
        """
        prediction = "None"
        reasoning = ""
        
        # Try primary extraction method
        extracted = _extract_jsons(text)
        if extracted is not None:
            self._extraction_stats["success"] += 1
            self.log_fn("JSON extraction: primary method succeeded")
        else:
            # Fallback to regex extraction
            extracted = _extract_json_with_regex(text)
            if extracted is not None:
                self._extraction_stats["fallback"] += 1
                self.log_fn("JSON extraction: fallback method succeeded")
            else:
                self._extraction_stats["failed"] += 1
                self.log_fn("JSON extraction: all methods failed")
        
        if extracted:
            last_json = extracted[-1]
            if "response" in last_json:
                prediction = str(last_json["response"]).strip()
            if "reasoning" in last_json:
                reasoning = str(last_json["reasoning"]).strip()
        
        return prediction, reasoning

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        start_time = time.time()
        
        # Check cache first
        cached = self._get_cached_response(inputs)
        if cached is not None:
            prediction, reasoning = cached
            self.log_fn(f"Using cached response: {prediction}")
            return prediction, [{"role": "assistant", "text": reasoning}]
        
        instruction = self._build_grading_prompt(inputs)
        
        msg_history = []
        prediction = "None"
        reasoning = ""
        
        # Retry loop for robust extraction
        for attempt in range(self.max_retries):
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=msg_history if attempt > 0 else [],
                )
                
                last_text = msg_history[-1]["text"] if msg_history else ""
                prediction, reasoning = self._extract_prediction(last_text)
                
                # Normalize the grade
                prediction = _normalize_grade(prediction)
                
                if prediction != "None":
                    self.log_fn(f"Successfully extracted prediction: {prediction}")
                    if reasoning:
                        self.log_fn(f"Reasoning: {reasoning[:200]}...")
                    break
                else:
                    self.log_fn(f"Attempt {attempt + 1}: Failed to extract prediction, retrying...")
                    # Add feedback for retry with more specific guidance
                    if attempt < self.max_retries - 1:
                        instruction = f"""ERROR: Your previous response did not contain valid JSON with a 'response' field.

Your response was:
---
{last_text[:500]}
---

You MUST respond with ONLY valid JSON wrapped in <json>...</json> tags. Do not include any text before or after the JSON tags.

Correct format:
<json>
{{
    "reasoning": "Your detailed analysis here...",
    "response": "Correct"
}}
</json>

IMPORTANT: 
- The JSON must be valid (no trailing commas, proper quotes)
- Both 'reasoning' and 'response' fields are required
- The 'response' field should contain only the grade

Now try again with the original task:

{self._build_grading_prompt(inputs)}"""
                    
            except Exception as e:
                self.log_fn(f"Error in attempt {attempt + 1}: {e}")
                if attempt == self.max_retries - 1:
                    prediction = f"Error: {e}"
        
        # Cache the response for future use
        if prediction != "None" and not prediction.startswith("Error:"):
            self._cache_response(inputs, prediction, reasoning)
        
        # Log extraction statistics periodically
        total = sum(self._extraction_stats.values())
        if total > 0 and total % 10 == 0:
            self.log_fn(f"Extraction stats after {total} attempts: {self._extraction_stats}")
        
        elapsed = time.time() - start_time
        self.log_fn(f"Task completed in {elapsed:.2f}s")
        
        return str(prediction), msg_history
