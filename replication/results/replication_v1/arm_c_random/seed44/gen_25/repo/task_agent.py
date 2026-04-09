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
from functools import lru_cache

from agent.llm_client import get_response_from_llm, EVAL_MODEL

logger = logging.getLogger(__name__)


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
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
        try:
            results.append(json.loads(inner))
        except json.JSONDecodeError:
            continue
    return results or None


def _extract_json_with_retry(text: str, max_retries: int = 3) -> list[dict] | None:
    """Extract JSON with multiple fallback strategies.
    
    Tries multiple extraction methods:
    1. Standard <json>...</json> blocks
    2. JSON code blocks ```json...```
    3. Raw JSON objects in text
    """
    # Try standard extraction first
    results = _extract_jsons(text)
    if results:
        return results
    
    # Try JSON code blocks
    results = []
    json_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    matches = re.findall(json_block_pattern, text, re.DOTALL)
    for match in matches:
        try:
            results.append(json.loads(match.strip()))
        except json.JSONDecodeError:
            continue
    
    if results:
        return results
    
    # Try to find raw JSON objects (objects with curly braces)
    # Use a more robust pattern that handles nested braces
    results = []
    # Find all potential JSON objects by tracking brace depth
    potential_objects = []
    start_indices = []
    brace_depth = 0
    in_string = False
    escape_next = False
    
    for i, char in enumerate(text):
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
                if brace_depth == 0:
                    start_indices.append(i)
                brace_depth += 1
            elif char == '}':
                brace_depth -= 1
                if brace_depth == 0 and start_indices:
                    start = start_indices.pop()
                    potential_objects.append(text[start:i+1])
    
    for obj_str in potential_objects:
        try:
            results.append(json.loads(obj_str.strip()))
        except json.JSONDecodeError:
            continue
    
    return results or None


class TaskAgent:
    """Task agent that solves IMO grading problems with enhanced reasoning and caching."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.max_retries = 3
        self._cache: dict[str, tuple[str, list[dict]]] = {}
        self._cache_hits = 0
        self._cache_misses = 0

    def _get_cache_key(self, inputs: dict) -> str:
        """Generate a cache key from inputs."""
        # Create deterministic key from relevant fields
        key_data = {
            "domain": inputs.get("domain", ""),
            "problem": inputs.get("problem", ""),
            "solution": inputs.get("solution", ""),
            "grading_guidelines": inputs.get("grading_guidelines", ""),
            "student_answer": inputs.get("student_answer", ""),
            "model": self.model,
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()[:32]

    def get_cache_stats(self) -> dict:
        """Return cache statistics."""
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "hit_rate": hit_rate,
            "size": len(self._cache),
        }

    def _validate_inputs(self, inputs: dict) -> tuple[bool, str]:
        """Validate that all required inputs are present and non-empty.
        
        Returns:
            (is_valid, error_message)
        """
        required_fields = ["domain", "problem", "solution", "grading_guidelines", "student_answer"]
        
        for field in required_fields:
            if field not in inputs:
                return False, f"Missing required field: {field}"
            if not inputs[field] or not str(inputs[field]).strip():
                return False, f"Empty required field: {field}"
        
        return True, ""

    def _build_grading_prompt(self, inputs: dict) -> str:
        """Build a comprehensive prompt for the grading task."""
        domain = inputs.get("domain", "")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        # Truncate very long inputs to prevent token overflow
        max_len = 8000
        problem = problem[:max_len] + "..." if len(problem) > max_len else problem
        solution = solution[:max_len] + "..." if len(solution) > max_len else solution
        student_answer = student_answer[:max_len] + "..." if len(student_answer) > max_len else student_answer
        
        return f"""You are an expert grader for {domain} problems. Your task is to evaluate a student's answer against the correct solution and grading guidelines.

## Problem
{problem}

## Correct Solution
{solution}

## Grading Guidelines
{grading_guidelines}

## Student's Answer
{student_answer}

## Instructions
1. Carefully analyze the student's answer step by step
2. Compare it against the correct solution
3. Apply the grading guidelines strictly
4. Provide your reasoning before giving the final grade
5. Respond in JSON format with the following schema:

<json>
{{
    "reasoning": "Your detailed step-by-step analysis and reasoning",
    "response": "The final grade/score (a number or specific grade value)"
}}
</json>

Important: 
- The "response" field must contain only the final grade/score
- Use the "reasoning" field to show your work
- Be precise and follow the grading guidelines exactly
- If the student answer is empty or nonsensical, assign the minimum grade"""

    def _try_extract_prediction(self, text: str) -> tuple[str, str | None]:
        """Try to extract prediction from response text.
        
        Returns:
            (prediction, reasoning)
        """
        if not text or not text.strip():
            return "None", None
            
        try:
            extracted = _extract_json_with_retry(text)
            if extracted:
                # Try to find the best JSON object with both response and reasoning
                best_match = None
                for obj in extracted:
                    if isinstance(obj, dict) and "response" in obj:
                        best_match = obj
                        break
                
                # If no object with "response" found, use the last one
                if best_match is None:
                    best_match = extracted[-1]
                
                if isinstance(best_match, dict):
                    prediction = best_match.get("response", "None")
                    reasoning = best_match.get("reasoning")
                    
                    # Handle numeric predictions
                    if prediction is None:
                        prediction = "None"
                    elif isinstance(prediction, (int, float)):
                        prediction = str(prediction)
                    elif not isinstance(prediction, str):
                        prediction = str(prediction)
                    
                    return prediction, reasoning
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
        
        return "None", None

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with retry logic and caching.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Validate inputs first
        is_valid, error_msg = self._validate_inputs(inputs)
        if not is_valid:
            self.log_fn(f"Input validation failed: {error_msg}")
            return f"Error: {error_msg}", []
        
        # Check cache first
        cache_key = self._get_cache_key(inputs)
        if cache_key in self._cache:
            self._cache_hits += 1
            self.log_fn(f"Cache hit for key {cache_key[:8]}...")
            return self._cache[cache_key]
        
        self._cache_misses += 1
        
        instruction = self._build_grading_prompt(inputs)
        msg_history = []
        prediction = "None"
        
        # Try with retries
        for attempt in range(self.max_retries):
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=msg_history if attempt > 0 else [],
                )
                
                # Extract prediction from the last assistant message
                text = ""
                for msg in reversed(msg_history):
                    if msg.get("role") == "assistant":
                        text = msg.get("text", "")
                        break
                
                pred, reasoning = self._try_extract_prediction(text)
                
                if pred != "None":
                    prediction = pred
                    if reasoning:
                        self.log_fn(f"Grading reasoning: {reasoning[:200]}...")
                    break
                
                # If extraction failed, add a follow-up message asking for proper format
                if attempt < self.max_retries - 1:
                    instruction = "Please respond in the required JSON format with 'response' and 'reasoning' fields. Your previous response could not be parsed."
                    
            except Exception as e:
                self.log_fn(f"Attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    break
        
        result = (str(prediction), msg_history)
        
        # Store in cache
        self._cache[cache_key] = result
        
        return result
