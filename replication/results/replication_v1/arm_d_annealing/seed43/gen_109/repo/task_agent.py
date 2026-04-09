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
import random
import re
import time
from typing import Any

from agent.llm_client import get_response_from_llm, EVAL_MODEL

logger = logging.getLogger(__name__)

# Valid grading responses for validation
VALID_GRADES = {"Correct", "Partial", "Incorrect"}

# Grade synonyms for text-based extraction
GRADE_SYNONYMS = {
    "Correct": ["correct", "right", "accurate", "valid", "true", "yes"],
    "Partial": ["partial", "partially correct", "some credit", "incomplete"],
    "Incorrect": ["incorrect", "wrong", "false", "invalid", "no", "error"]
}


def _clean_json_string(text: str) -> str:
    """Clean common JSON formatting issues with comprehensive fixes."""
    if not text:
        return text
    
    cleaned = text.strip()
    
    # Remove markdown code block markers
    cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r'\s*```$', '', cleaned, flags=re.MULTILINE)
    
    # Remove trailing commas before closing braces/brackets
    cleaned = re.sub(r',(\s*[}\]])', r'\1', cleaned)
    
    # Fix single quotes to double quotes for keys and string values
    # Handle nested quotes carefully
    cleaned = re.sub(r"(?<=[{,\s])'([^']+)'(?=\s*:)", r'"\1"', cleaned)
    cleaned = re.sub(r":\s*'([^']*?)'(?=\s*[,}])", r': "\1"', cleaned)
    
    # Remove control characters except newlines and tabs
    cleaned = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', cleaned)
    
    # Fix escaped newlines that might break parsing
    cleaned = cleaned.replace('\\n', '\n')
    cleaned = cleaned.replace('\\t', '\t')
    
    # Fix double-escaped quotes
    cleaned = cleaned.replace('\\"', '"')
    
    # Remove BOM if present
    cleaned = cleaned.lstrip('\ufeff')
    
    return cleaned.strip()


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks with robust parsing.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    Also handles nested JSON objects and common formatting issues.
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
        
        if not inner:
            continue
        
        # Try multiple parsing strategies
        parsed = _try_parse_json(inner)
        if parsed:
            results.append(parsed)
            continue
        
        # Try extracting fields directly if JSON parsing fails
        extracted = _extract_fields_directly(inner)
        if extracted:
            results.append(extracted)
    
    return results if results else None


def _try_parse_json(text: str) -> dict | None:
    """Try to parse JSON with multiple cleaning strategies."""
    strategies = [
        lambda x: x,  # Try raw first
        _clean_json_string,  # Try cleaned
        lambda x: re.sub(r'\n\s*', ' ', x),  # Try with newlines removed
        lambda x: re.sub(r'\s+', ' ', x),  # Try with all whitespace normalized
    ]
    
    for strategy in strategies:
        try:
            cleaned = strategy(text)
            if cleaned:
                return json.loads(cleaned)
        except (json.JSONDecodeError, ValueError):
            continue
    
    return None


def _extract_fields_directly(text: str) -> dict | None:
    """Extract response and reasoning fields directly from malformed JSON."""
    result = {}
    
    # Extract response field with multiple patterns
    response_patterns = [
        r'"response"\s*:\s*"([^"]+)"',
        r'"response"\s*:\s*\'([^\']+)\'',
        r'response["\']?\s*:\s*["\']?([^"\',}\n]+)',
    ]
    
    for pattern in response_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            result["response"] = match.group(1).strip()
            break
    
    # Extract reasoning field with multiple patterns
    reasoning_patterns = [
        r'"reasoning"\s*:\s*"([^"]*)"',
        r'"reasoning"\s*:\s*\'([^\']*)\'',
        r'reasoning["\']?\s*:\s*["\']?([^"\']*?)(?:["\']?\s*[,}]|$)',
    ]
    
    for pattern in reasoning_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            result["reasoning"] = match.group(1).strip()
            break
    
    return result if "response" in result else None


def _extract_json_with_regex(text: str) -> list[dict] | None:
    """Fallback JSON extraction using multiple strategies for malformed responses."""
    results = []
    
    # Strategy 1: Try to find JSON objects in code blocks
    code_block_patterns = [
        r'```(?:json)?\s*(\{[\s\S]*?\})\s*```',
        r'```\s*(\{[\s\S]*?\})\s*```',
        r'`(\{[\s\S]*?\})`',
    ]
    
    for pattern in code_block_patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            parsed = _try_parse_json(match)
            if parsed:
                results.append(parsed)
    
    # Strategy 2: Try to find any JSON-like structure with braces
    if not results:
        # Find content between outermost braces
        brace_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(brace_pattern, text, re.DOTALL)
        for match in matches:
            parsed = _try_parse_json(match)
            if parsed:
                results.append(parsed)
    
    # Strategy 3: Direct field extraction
    if not results:
        extracted = _extract_fields_directly(text)
        if extracted:
            results.append(extracted)
    
    # Strategy 4: Look for grade keywords in text
    if not results:
        text_lower = text.lower()
        for grade, synonyms in GRADE_SYNONYMS.items():
            for synonym in synonyms:
                # Look for the grade as a standalone word
                if re.search(rf'\b{re.escape(synonym)}\b', text_lower):
                    results.append({
                        "response": grade,
                        "reasoning": f"Extracted from text analysis: found keyword '{synonym}'"
                    })
                    break
            if results:
                break
    
    return results if results else None


def _validate_grading_output(data: dict) -> tuple[bool, str]:
    """Validate that the grading output has the correct format with detailed feedback.
    
    Returns:
        (is_valid, error_message) tuple
    """
    if not isinstance(data, dict):
        return False, f"Expected dict, got {type(data).__name__}"
    
    if "response" not in data:
        return False, "Missing 'response' field"
    
    response = data.get("response")
    if not isinstance(response, str):
        return False, f"'response' must be a string, got {type(response).__name__}"
    
    # Normalize the response
    response_normalized = response.strip()
    
    if response_normalized not in VALID_GRADES:
        # Check for case-insensitive match
        for grade in VALID_GRADES:
            if response_normalized.lower() == grade.lower():
                data["response"] = grade  # Fix the case
                return True, ""
        return False, f"Invalid response value: '{response}'. Must be one of {VALID_GRADES}"
    else:
        # Ensure correct case
        data["response"] = response_normalized
    
    # Validate reasoning field if present
    if "reasoning" in data and not isinstance(data["reasoning"], str):
        data["reasoning"] = str(data["reasoning"])
    
    return True, ""


def _normalize_grade(grade: str) -> str:
    """Normalize a grade string to one of the valid grades."""
    if not grade:
        return "None"
    
    grade_lower = grade.lower().strip()
    
    for valid_grade in VALID_GRADES:
        if grade_lower == valid_grade.lower():
            return valid_grade
    
    # Check synonyms
    for valid_grade, synonyms in GRADE_SYNONYMS.items():
        for synonym in synonyms:
            if synonym in grade_lower:
                return valid_grade
    
    return "None"


class TaskAgent:
    """Task agent for grading student answers with robust error handling."""

    def __init__(
        self,
        model: str = EVAL_MODEL,
        temperature: float = 0.0,
        max_retries: int = 3,
        log_fn: Any = logger.info,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.log_fn = log_fn

    def _build_grading_prompt(self, inputs: dict, is_retry: bool = False, previous_error: str = "") -> str:
        """Build an optimized grading prompt with clear instructions and examples."""
        domain = inputs.get("domain", "")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        base_prompt = f"""You are an expert grader evaluating student answers. Your task is to carefully assess whether the student's answer matches the correct solution according to the grading guidelines.

=== DOMAIN ===
{domain}

=== PROBLEM ===
{problem}

=== CORRECT SOLUTION ===
{solution}

=== GRADING GUIDELINES ===
{grading_guidelines}

=== STUDENT'S ANSWER ===
{student_answer}

=== EVALUATION CRITERIA ===
1. Correctness: Is the answer mathematically/technically correct?
2. Completeness: Does it address all parts of the problem?
3. Methodology: Is the approach sound and well-reasoned?
4. Guidelines: Does it follow the specific grading guidelines provided?

=== GRADING DEFINITIONS ===
- "Correct": The answer is fully correct, complete, and follows all guidelines
- "Partial": The answer has some correct elements but is incomplete or has minor errors
- "Incorrect": The answer is wrong, fundamentally flawed, or doesn't address the problem

=== RESPONSE FORMAT ===
You MUST respond with ONLY a JSON object wrapped in <json>...</json> tags. No other text before or after.

The JSON must have exactly these two fields:
- "reasoning": A detailed explanation of your evaluation (string, 1-3 sentences)
- "response": Exactly one of "Correct", "Partial", or "Incorrect" (string, case-sensitive)

=== EXAMPLE RESPONSES ===
Example 1 (Correct):
<json>
{{
    "reasoning": "The student correctly applied the quadratic formula, showed all work, and arrived at the exact answer x=2 and x=3 as specified in the solution.",
    "response": "Correct"
}}
</json>

Example 2 (Partial):
<json>
{{
    "reasoning": "The student found one correct solution (x=2) but missed the second solution (x=3). The method was correct but incomplete.",
    "response": "Partial"
}}
</json>

Example 3 (Incorrect):
<json>
{{
    "reasoning": "The student used an incorrect formula and arrived at wrong answers. The calculation errors show a fundamental misunderstanding.",
    "response": "Incorrect"
}}
</json>

Now provide your evaluation:"""
        
        if is_retry and previous_error:
            return f"""ERROR: Your previous response was invalid: {previous_error}

You MUST fix this and respond with ONLY valid JSON wrapped in <json>...</json> tags.

CRITICAL REQUIREMENTS:
1. Use EXACTLY <json> and </json> tags (not <json_format> or anything else)
2. The "response" field must be exactly "Correct", "Partial", or "Incorrect" (case-sensitive)
3. No text before or after the JSON tags
4. Valid JSON syntax with double quotes for keys and string values

Correct format:
<json>
{{
    "reasoning": "Your detailed explanation here...",
    "response": "Correct"
}}
</json>

Now try again with the original task:

{base_prompt}"""
        
        return base_prompt

    def _extract_prediction(self, text: str) -> tuple[str, str, str]:
        """Extract prediction and reasoning from response text with comprehensive validation.
        
        Returns:
            (prediction, reasoning, error_message) tuple
        """
        prediction = "None"
        reasoning = ""
        error_msg = ""
        
        if not text or not text.strip():
            return "None", "", "Empty response"
        
        # Try primary extraction method
        extracted = _extract_jsons(text)
        if extracted is None:
            # Fallback to regex extraction
            extracted = _extract_json_with_regex(text)
        
        if not extracted:
            return "None", "", "No JSON found in response"
        
        # Validate each extracted JSON
        for data in extracted:
            is_valid, error_msg = _validate_grading_output(data)
            if is_valid:
                prediction = str(data.get("response", "None"))
                reasoning = str(data.get("reasoning", ""))
                # Normalize the grade
                prediction = _normalize_grade(prediction)
                if prediction in VALID_GRADES:
                    return prediction, reasoning, ""
        
        # If none valid, try to extract something useful from the last attempt
        if extracted:
            last_json = extracted[-1]
            raw_prediction = str(last_json.get("response", "None"))
            prediction = _normalize_grade(raw_prediction)
            reasoning = str(last_json.get("reasoning", ""))
            
            if prediction in VALID_GRADES:
                return prediction, reasoning, ""
        
        return prediction, reasoning, error_msg or "Validation failed"

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with robust retry logic.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        instruction = self._build_grading_prompt(inputs)
        
        msg_history = []
        prediction = "None"
        reasoning = ""
        last_error = ""
        
        # Retry loop with exponential backoff
        for attempt in range(self.max_retries):
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=msg_history if attempt > 0 else [],
                )
                
                last_text = msg_history[-1]["text"] if msg_history else ""
                prediction, reasoning, error_msg = self._extract_prediction(last_text)
                
                # Validate the prediction
                if prediction in VALID_GRADES:
                    self.log_fn(f"Successfully extracted prediction: {prediction}")
                    if reasoning:
                        self.log_fn(f"Reasoning: {reasoning[:200]}...")
                    break
                else:
                    last_error = error_msg if error_msg else f"Invalid prediction: {prediction}"
                    self.log_fn(f"Attempt {attempt + 1}/{self.max_retries}: {last_error}")
                    
                    # Add feedback for retry
                    if attempt < self.max_retries - 1:
                        instruction = self._build_grading_prompt(inputs, is_retry=True, previous_error=last_error)
                    
            except Exception as e:
                last_error = str(e)
                self.log_fn(f"Error in attempt {attempt + 1}/{self.max_retries}: {e}")
                if attempt < self.max_retries - 1:
                    # Exponential backoff with jitter
                    base_wait = min(60, 2 ** attempt)
                    jitter = random.uniform(0, 1)
                    wait_time = base_wait + jitter
                    time.sleep(wait_time)
                else:
                    prediction = f"Error: {e}"
        
        # Final validation - ensure we return a valid grade or error
        if prediction not in VALID_GRADES:
            if prediction.startswith("Error:"):
                self.log_fn(f"LLM error occurred: {prediction}")
            else:
                self.log_fn(f"Warning: Final prediction '{prediction}' not in valid grades, defaulting to 'Incorrect'")
                prediction = "Incorrect"
        
        return str(prediction), msg_history