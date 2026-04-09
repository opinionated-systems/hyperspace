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
import time
from typing import Any

from agent.llm_client import get_response_from_llm, EVAL_MODEL

logger = logging.getLogger(__name__)


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    Also handles markdown code blocks with json language specifier.
    """
    results = []
    search_from = 0
    
    # First, try to find explicit <json> tags
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
            # Try to clean up common JSON formatting issues
            cleaned = _clean_json_string(inner)
            try:
                results.append(json.loads(cleaned))
            except json.JSONDecodeError:
                continue
    
    # If no <json> tags found, look for markdown code blocks
    if not results:
        md_pattern = r'```(?:json)?\s*([\s\S]*?)```'
        matches = re.findall(md_pattern, text)
        for match in matches:
            try:
                results.append(json.loads(match.strip()))
            except json.JSONDecodeError:
                cleaned = _clean_json_string(match.strip())
                try:
                    results.append(json.loads(cleaned))
                except json.JSONDecodeError:
                    continue
    
    return results or None


def _clean_json_string(text: str) -> str:
    """Clean up common JSON formatting issues."""
    # Remove trailing commas before closing braces/brackets
    text = re.sub(r',(\s*[}\]])', r'\1', text)
    # Fix single quotes to double quotes (basic handling)
    text = re.sub(r"'([^']*?)'", r'"\1"', text)
    # Remove comments
    text = re.sub(r'//.*?\n', '\n', text)
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    # Handle escaped newlines that might break JSON parsing
    text = text.replace('\\n', '\n')
    # Remove control characters
    text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\r\t')
    return text.strip()


def _extract_any_json(text: str) -> list[dict] | None:
    """Fallback JSON extraction that looks for any JSON objects in text.
    
    Uses a robust brace-matching algorithm that handles nested structures
    and string literals correctly.
    """
    results = []
    brace_count = 0
    start_idx = -1
    in_string = False
    escape_next = False
    
    for i, char in enumerate(text):
        if escape_next:
            escape_next = False
            continue
        if char == '\\' and in_string:
            escape_next = True
            continue
        if char == '"' and not in_string:
            in_string = True
            continue
        if char == '"' and in_string:
            in_string = False
            continue
        
        if not in_string:
            if char == '{':
                if brace_count == 0:
                    start_idx = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start_idx != -1:
                    try:
                        obj = json.loads(text[start_idx:i+1])
                        results.append(obj)
                    except json.JSONDecodeError:
                        # Try cleaning the string
                        cleaned = _clean_json_string(text[start_idx:i+1])
                        try:
                            obj = json.loads(cleaned)
                            results.append(obj)
                        except json.JSONDecodeError:
                            pass
                    start_idx = -1
    
    return results or None


def _extract_json_with_regex(text: str) -> list[dict] | None:
    """Additional fallback using regex to find JSON-like structures."""
    results = []
    # Look for JSON blocks that might be wrapped in markdown code blocks
    json_pattern = r'```(?:json)?\s*([\s\S]*?)```'
    matches = re.findall(json_pattern, text)
    for match in matches:
        try:
            obj = json.loads(match.strip())
            if isinstance(obj, dict):
                results.append(obj)
        except json.JSONDecodeError:
            # Try to find nested JSON objects within the match
            brace_count = 0
            start_idx = -1
            for i, char in enumerate(match):
                if char == '{':
                    if brace_count == 0:
                        start_idx = i
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0 and start_idx != -1:
                        try:
                            obj = json.loads(match[start_idx:i+1])
                            results.append(obj)
                        except json.JSONDecodeError:
                            pass
                        start_idx = -1
    return results or None


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.max_retries = 3
        self.base_delay = 1.0
        self._cache: dict[str, tuple[str, list[dict]]] = {}
        self._cache_enabled = True

    def _get_cache_key(self, inputs: dict) -> str:
        """Generate a cache key from inputs."""
        import hashlib
        key_data = json.dumps(inputs, sort_keys=True)
        return hashlib.md5(key_data.encode()).hexdigest()

    def clear_cache(self) -> None:
        """Clear the response cache."""
        self._cache.clear()

    def set_cache_enabled(self, enabled: bool) -> None:
        """Enable or disable caching."""
        self._cache_enabled = enabled

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Check cache first
        if self._cache_enabled:
            cache_key = self._get_cache_key(inputs)
            if cache_key in self._cache:
                self.log_fn("Cache hit: returning cached result")
                return self._cache[cache_key]

        # Extract task components for better prompting
        domain = inputs.get("domain", "Unknown")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = f"""You are an expert mathematical grader for International Mathematical Olympiad (IMO) problems.

Your task is to evaluate a student's answer to a mathematics problem and provide a grade.

DOMAIN: {domain}

PROBLEM:
{problem}

OFFICIAL SOLUTION:
{solution}

GRADING GUIDELINES:
{grading_guidelines}

STUDENT'S ANSWER:
{student_answer}

Think step by step:
1. Analyze what the problem is asking and identify key concepts
2. Review the official solution approach and identify critical steps
3. Compare the student's answer to the official solution - check for:
   - Correctness of the final answer
   - Validity of the reasoning process
   - Completeness of the solution
   - Mathematical rigor and clarity
4. Check if the student followed the grading guidelines precisely
5. Determine the appropriate grade based on the guidelines

IMPORTANT: Your response must be valid JSON wrapped in <json> tags. Do not include any text outside the JSON tags.

Respond in this exact format:
<json>
{{
    "reasoning": "Your detailed chain-of-thought analysis here. Explain your evaluation process clearly.",
    "response": "The final grade/prediction (e.g., 'Correct', 'Incorrect', 'Partial', or a numeric score)",
    "confidence": "High|Medium|Low - indicate how confident you are in this grade"
}}
</json>"""

        # Retry loop with exponential backoff for improved reliability
        last_exception = None
        for attempt in range(self.max_retries):
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=[],
                )
                break
            except Exception as e:
                last_exception = e
                self.log_fn(f"LLM call failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    delay = self.base_delay * (2 ** attempt)
                    self.log_fn(f"Retrying in {delay:.1f}s...")
                    time.sleep(delay)
                else:
                    self.log_fn("Max retries reached, returning error prediction")
                    return "Error: LLM call failed", []

        # Extract prediction from JSON with multiple fallback mechanisms
        prediction = "None"
        try:
            last_message = msg_history[-1]["text"]
            
            # Try primary extraction method (explicit <json> tags)
            extracted = _extract_jsons(last_message)
            
            # Fallback 1: generic JSON extraction from braces
            if extracted is None:
                extracted = _extract_any_json(last_message)
            
            # Fallback 2: regex-based extraction for markdown code blocks
            if extracted is None:
                extracted = _extract_json_with_regex(last_message)
            
            if extracted:
                # Prefer response field, but accept other common field names
                last_json = extracted[-1]
                if "response" in last_json:
                    prediction = last_json["response"]
                elif "grade" in last_json:
                    prediction = last_json["grade"]
                elif "answer" in last_json:
                    prediction = last_json["answer"]
                elif "result" in last_json:
                    prediction = last_json["result"]
                elif "evaluation" in last_json:
                    prediction = last_json["evaluation"]
                elif "prediction" in last_json:
                    prediction = last_json["prediction"]
                elif "score" in last_json:
                    prediction = last_json["score"]
                else:
                    # If no known field, use the first string value found
                    for key, value in last_json.items():
                        if isinstance(value, str):
                            prediction = value
                            break
                        elif isinstance(value, (int, float)):
                            prediction = str(value)
                            break
                
                # Log confidence if available (for debugging/monitoring)
                if "confidence" in last_json:
                    self.log_fn(f"Grading confidence: {last_json['confidence']}")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        result = (str(prediction), msg_history)
        
        # Cache the result
        if self._cache_enabled:
            cache_key = self._get_cache_key(inputs)
            self._cache[cache_key] = result
        
        return result

    def forward_with_confidence(self, inputs: dict) -> tuple[str, str, list[dict]]:
        """Run the task agent and return prediction with confidence.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, confidence, msg_history)
        """
        prediction, msg_history = self.forward(inputs)
        
        # Extract confidence from the last message if available
        confidence = "Unknown"
        try:
            last_message = msg_history[-1]["text"]
            extracted = _extract_jsons(last_message)
            if extracted and "confidence" in extracted[-1]:
                confidence = str(extracted[-1]["confidence"])
        except Exception:
            pass
        
        return prediction, confidence, msg_history

    def forward_batch(self, inputs_list: list[dict]) -> list[tuple[str, list[dict]]]:
        """Process multiple grading tasks efficiently.
        
        Args:
            inputs_list: List of input dicts, each containing domain, problem, 
                        solution, grading_guidelines, student_answer
                        
        Returns:
            List of (prediction, msg_history) tuples
        """
        results = []
        for inputs in inputs_list:
            result = self.forward(inputs)
            results.append(result)
        return results
