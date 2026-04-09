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
import hashlib
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
            # Try to fix common JSON issues before giving up
            try:
                # Remove trailing commas before closing braces/brackets
                fixed = re.sub(r',(\s*[}\]])', r'\1', inner)
                # Fix single quotes to double quotes (common LLM mistake)
                fixed = re.sub(r"(?<!\\)'", '"', fixed)
                # Fix unescaped newlines in strings
                fixed = re.sub(r'(?<!\\)\n', '\\n', fixed)
                results.append(json.loads(fixed))
            except json.JSONDecodeError:
                continue
    return results or None


def _extract_json_code_blocks(text: str) -> list[dict]:
    """Extract JSON from markdown code blocks."""
    results = []
    json_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    matches = re.findall(json_block_pattern, text, re.DOTALL)
    for match in matches:
        try:
            results.append(json.loads(match.strip()))
        except json.JSONDecodeError:
            # Try common fixes
            try:
                fixed = re.sub(r',(\s*[}\]])', r'\1', match.strip())
                fixed = re.sub(r"(?<!\\)'", '"', fixed)
                results.append(json.loads(fixed))
            except json.JSONDecodeError:
                continue
    return results


def _extract_json_with_retry(text: str, max_retries: int = 3) -> list[dict] | None:
    """Extract JSON with multiple fallback strategies.
    
    Tries multiple extraction methods in order of reliability:
    1. Standard <json>...</json> blocks
    2. JSON code blocks ```json...```
    3. Raw JSON objects in text (improved pattern)
    """
    # Try standard extraction first
    results = _extract_jsons(text)
    if results:
        return results
    
    # Try JSON code blocks
    results = _extract_json_code_blocks(text)
    if results:
        return results
    
    # Try to find raw JSON objects with improved pattern
    # Match balanced braces with nested content
    object_pattern = r'\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}'
    matches = re.findall(object_pattern, text, re.DOTALL)
    for match in matches:
        try:
            parsed = json.loads(match.strip())
            # Only include if it has expected fields
            if isinstance(parsed, dict) and ("response" in parsed or "reasoning" in parsed):
                results.append(parsed)
        except json.JSONDecodeError:
            continue
    
    return results or None


class TaskAgent:
    """Task agent that solves IMO grading problems with enhanced reasoning and caching."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.max_retries = 3
        # Cache for validated inputs to avoid re-validation
        self._validation_cache: dict[str, tuple[bool, str]] = {}

    def _get_cache_key(self, inputs: dict) -> str:
        """Generate a cache key for inputs."""
        # Use a simple hash of the student answer and problem as cache key
        key_data = f"{inputs.get('problem', '')[:100]}:{inputs.get('student_answer', '')[:100]}"
        return hashlib.md5(key_data.encode()).hexdigest()[:16]

    def _validate_inputs(self, inputs: dict) -> tuple[bool, str]:
        """Validate that all required inputs are present and non-empty.
        Uses caching to avoid repeated validation of similar inputs.
        
        Returns:
            (is_valid, error_message)
        """
        cache_key = self._get_cache_key(inputs)
        if cache_key in self._validation_cache:
            return self._validation_cache[cache_key]
        
        required_fields = ["domain", "problem", "solution", "grading_guidelines", "student_answer"]
        
        for field in required_fields:
            if field not in inputs:
                result = (False, f"Missing required field: {field}")
                self._validation_cache[cache_key] = result
                return result
            if not inputs[field] or not str(inputs[field]).strip():
                result = (False, f"Empty required field: {field}")
                self._validation_cache[cache_key] = result
                return result
        
        result = (True, "")
        self._validation_cache[cache_key] = result
        return result

    def _build_grading_prompt(self, inputs: dict) -> str:
        """Build a comprehensive prompt for the grading task with smart truncation."""
        domain = inputs.get("domain", "")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        # Smart truncation: preserve structure by truncating at paragraph boundaries
        def smart_truncate(text: str, max_len: int) -> str:
            if len(text) <= max_len:
                return text
            # Try to find a good break point (paragraph end)
            truncated = text[:max_len]
            # Look for last paragraph break
            last_para = truncated.rfind('\n\n')
            if last_para > max_len * 0.7:  # Only use if we keep at least 70%
                return truncated[:last_para] + "\n\n[...content truncated...]"
            # Look for last sentence
            last_sentence = truncated.rfind('. ')
            if last_sentence > max_len * 0.8:
                return truncated[:last_sentence + 1] + " [...]"
            return truncated + "..."
        
        # Adaptive max length based on total content
        total_len = len(problem) + len(solution) + len(student_answer)
        if total_len > 20000:
            max_len = 5000
        elif total_len > 10000:
            max_len = 6000
        else:
            max_len = 8000
        
        problem = smart_truncate(problem, max_len)
        solution = smart_truncate(solution, max_len)
        student_answer = smart_truncate(student_answer, max_len)
        
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
- If the student answer is empty or nonsensical, assign the minimum grade
- Ensure your JSON is valid and properly escaped"""

    def _try_extract_prediction(self, text: str) -> tuple[str, str | None]:
        """Try to extract prediction from response text with enhanced error handling.
        
        Returns:
            (prediction, reasoning)
        """
        if not text or not text.strip():
            return "None", None
            
        try:
            extracted = _extract_json_with_retry(text)
            if extracted:
                last = extracted[-1]
                prediction = last.get("response", "None")
                reasoning = last.get("reasoning")
                # Validate prediction is not empty
                if prediction and str(prediction).strip():
                    return str(prediction), reasoning
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
        
        return "None", None

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with retry logic and progressive refinement.

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
        
        instruction = self._build_grading_prompt(inputs)
        msg_history = []
        prediction = "None"
        
        # Try with retries and progressive refinement
        for attempt in range(self.max_retries):
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=msg_history if attempt > 0 else [],
                )
                
                # Extract prediction
                text = msg_history[-1]["text"] if msg_history else ""
                pred, reasoning = self._try_extract_prediction(text)
                
                if pred != "None":
                    prediction = pred
                    if reasoning:
                        self.log_fn(f"Grading reasoning (attempt {attempt + 1}): {reasoning[:200]}...")
                    break
                
                # Progressive refinement: more specific instructions on each retry
                if attempt < self.max_retries - 1:
                    if attempt == 0:
                        instruction = "Your previous response did not contain valid JSON. Please respond ONLY with a JSON object in the format: <json>{\"reasoning\": \"...\", \"response\": \"grade\"}</json>"
                    else:
                        instruction = f"Still no valid JSON found. Last response was: {text[:200]}... Please provide ONLY the JSON response with 'reasoning' and 'response' fields."
                    
            except Exception as e:
                self.log_fn(f"Attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    break
        
        # Clear validation cache periodically to prevent memory growth
        if len(self._validation_cache) > 1000:
            self._validation_cache.clear()
        
        return str(prediction), msg_history
