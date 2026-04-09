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
    """Task agent that solves IMO grading problems with chain-of-thought reasoning.
    
    Enhanced with robust validation, confidence scoring, and structured output parsing.
    """

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.max_retries = 3
        self.base_delay = 1.0
        self._cache: dict[str, tuple[str, list[dict]]] = {}
        self._cache_enabled = True
        # Valid grade categories for IMO-style problems
        self._valid_grades = {"Correct", "Incorrect", "Partial", "0", "1", "2", "3", "4", "5", "6", "7"}

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

    def _validate_and_normalize_grade(self, grade: str) -> str:
        """Validate and normalize the extracted grade.
        
        Args:
            grade: Raw grade string from LLM
            
        Returns:
            Normalized grade string
        """
        if not grade or not isinstance(grade, str):
            return "None"
        
        grade = grade.strip()
        
        # Direct match with valid grades
        if grade in self._valid_grades:
            return grade
        
        # Normalize case variations
        grade_lower = grade.lower()
        if grade_lower in {"correct", "right", "true", "yes", "valid"}:
            return "Correct"
        if grade_lower in {"incorrect", "wrong", "false", "no", "invalid"}:
            return "Incorrect"
        if grade_lower in {"partial", "partially correct", "incomplete"}:
            return "Partial"
        
        # Try to extract numeric score (0-7 for IMO problems)
        import re
        numeric_match = re.search(r'\b([0-7])\b', grade)
        if numeric_match:
            return numeric_match.group(1)
        
        # If no valid grade found, return the original (may need manual review)
        return grade

    def _build_grading_prompt(self, inputs: dict) -> str:
        """Build a comprehensive grading prompt with clear instructions.
        
        Args:
            inputs: Dictionary containing problem components
            
        Returns:
            Formatted prompt string
        """
        domain = inputs.get("domain", "Unknown")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        return f"""You are an expert mathematical grader for International Mathematical Olympiad (IMO) problems.

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

EVALUATION INSTRUCTIONS:
1. Carefully analyze the problem requirements and identify all key concepts and theorems needed
2. Review the official solution to understand the expected approach and critical proof steps
3. Compare the student's answer against the official solution, evaluating:
   - Mathematical correctness of the final answer
   - Logical validity and rigor of the reasoning process
   - Completeness (are all required steps present?)
   - Clarity and proper mathematical notation
   - Adherence to the specific grading guidelines provided
4. Assign a grade based on IMO standards (0-7 points, or Correct/Incorrect/Partial for binary problems)

IMPORTANT: Your response must be valid JSON wrapped in <json> tags. Do not include any text outside the JSON tags.

Respond in this exact format:
<json>
{{
    "reasoning": "Your detailed chain-of-thought analysis here. Explain your evaluation process clearly, citing specific aspects of the student's answer.",
    "response": "The final grade (e.g., 'Correct', 'Incorrect', 'Partial', or numeric score 0-7)",
    "confidence": "High|Medium|Low - your confidence in this grading decision"
}}
</json>"""

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with enhanced validation.

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

        # Build comprehensive grading prompt
        instruction = self._build_grading_prompt(inputs)

        # Retry loop with exponential backoff for improved reliability
        last_exception = None
        msg_history = []
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
        confidence = "Unknown"
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
                last_json = extracted[-1]
                
                # Extract grade with priority ordering
                prediction = self._extract_grade_from_json(last_json)
                
                # Extract confidence if available
                confidence = last_json.get("confidence", "Unknown")
                
                # Validate and normalize the grade
                prediction = self._validate_and_normalize_grade(prediction)
                
                self.log_fn(f"Extracted grade: {prediction}, Confidence: {confidence}")
            else:
                self.log_fn("No JSON found in response, attempting text extraction")
                # Last resort: try to find grade-like text in the response
                prediction = self._extract_grade_from_text(last_message)
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        result = (str(prediction), msg_history)
        
        # Cache the result
        if self._cache_enabled:
            cache_key = self._get_cache_key(inputs)
            self._cache[cache_key] = result
        
        return result

    def _extract_grade_from_json(self, json_obj: dict) -> str:
        """Extract grade from JSON object with priority ordering.
        
        Args:
            json_obj: Parsed JSON object
            
        Returns:
            Extracted grade string or "None"
        """
        # Priority order of field names
        priority_fields = ["response", "grade", "answer", "result", 
                          "evaluation", "prediction", "score"]
        
        for field in priority_fields:
            if field in json_obj:
                value = json_obj[field]
                if isinstance(value, str):
                    return value
                elif isinstance(value, (int, float)):
                    return str(value)
        
        # If no known field, use the first string or numeric value found
        for key, value in json_obj.items():
            if isinstance(value, str):
                return value
            elif isinstance(value, (int, float)):
                return str(value)
        
        return "None"

    def _extract_grade_from_text(self, text: str) -> str:
        """Extract grade from plain text as last resort.
        
        Args:
            text: Raw text response
            
        Returns:
            Extracted grade or "None"
        """
        import re
        
        # Look for explicit grade statements
        patterns = [
            r'[Gg]rade:\s*(\S+)',
            r'[Ss]core:\s*(\S+)',
            r'[Ff]inal\s+(?:grade|score):\s*(\S+)',
            r'[Ee]valuation:\s*(\S+)',
            r'[Rr]esponse:\s*(\S+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return self._validate_and_normalize_grade(match.group(1))
        
        # Look for standalone grade keywords
        text_lower = text.lower()
        if any(word in text_lower for word in ["correct", "right", "valid"]):
            if "incorrect" not in text_lower and "not correct" not in text_lower:
                return "Correct"
        if any(word in text_lower for word in ["incorrect", "wrong", "invalid"]):
            return "Incorrect"
        if "partial" in text_lower:
            return "Partial"
        
        return "None"
