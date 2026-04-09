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
from typing import Any

from agent.llm_client import get_response_from_llm, EVAL_MODEL

logger = logging.getLogger(__name__)


# Valid grading domains supported by this agent
VALID_DOMAINS = {"math", "physics", "chemistry", "biology", "computer_science", "imo"}


def _validate_inputs(inputs: dict) -> tuple[bool, str]:
    """Validate that all required inputs are present and valid.
    
    Returns:
        (is_valid, error_message)
    """
    required_fields = ["problem", "solution", "grading_guidelines", "student_answer"]
    
    for field in required_fields:
        if field not in inputs:
            return False, f"Missing required field: {field}"
        if not inputs[field] or not str(inputs[field]).strip():
            return False, f"Empty required field: {field}"
    
    # Validate domain if provided
    domain = inputs.get("domain", "").lower()
    if domain and domain not in VALID_DOMAINS:
        logger.warning(f"Unknown domain '{domain}', defaulting to generic grading")
    
    return True, ""


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
        
        # Try to parse the JSON
        parsed = _try_parse_json(inner)
        if parsed is not None:
            results.append(parsed)
    
    return results or None


def _try_parse_json(text: str) -> dict | None:
    """Try to parse JSON with multiple fallback strategies.
    
    Returns the parsed dict or None if parsing fails.
    """
    # Strategy 1: Direct parsing
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Strategy 2: Fix common JSON issues
    try:
        fixed = text
        # Remove trailing commas before closing braces/brackets
        fixed = re.sub(r',\s*}', '}', fixed)
        fixed = re.sub(r',\s*]', ']', fixed)
        # Fix single quotes to double quotes (but not within strings)
        fixed = _fix_single_quotes(fixed)
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    
    # Strategy 3: Extract just the first valid JSON object
    try:
        # Find the first { and matching }
        start_idx = text.find('{')
        if start_idx != -1:
            # Count braces to find matching closing brace
            brace_count = 0
            for i, char in enumerate(text[start_idx:]):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        candidate = text[start_idx:start_idx + i + 1]
                        return json.loads(candidate)
    except (json.JSONDecodeError, ValueError):
        pass
    
    return None


def _fix_single_quotes(text: str) -> str:
    """Fix single quotes to double quotes, being careful about apostrophes."""
    result = []
    in_string = False
    string_char = None
    
    for char in text:
        if not in_string:
            if char == '"':
                in_string = True
                string_char = '"'
                result.append(char)
            elif char == "'":
                # Convert single quote to double quote
                result.append('"')
            else:
                result.append(char)
        else:
            if char == string_char:
                in_string = False
                string_char = None
            result.append(char)
    
    return ''.join(result)


def _extract_json_with_retry(text: str, max_retries: int = 3) -> list[dict] | None:
    """Extract JSON with multiple fallback strategies.
    
    Tries multiple extraction methods:
    1. Standard <json>...</json> blocks
    2. JSON code blocks ```json...```
    3. Raw JSON objects in text
    """
    results = []
    
    # Method 1: Standard <json>...</json> blocks
    json_results = _extract_jsons(text)
    if json_results:
        results.extend(json_results)
    
    # Method 2: JSON code blocks ```json...``` or ```...```
    json_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    matches = re.findall(json_block_pattern, text, re.DOTALL)
    for match in matches:
        parsed = _try_parse_json(match.strip())
        if parsed is not None:
            results.append(parsed)
    
    # Method 3: Look for JSON-like structures with "response" or "reasoning" keys
    # This handles cases where the model outputs JSON without code blocks
    if not results:
        # Try to find objects that look like grading responses
        response_pattern = r'"response"\s*:\s*"([^"]+)"'
        reasoning_pattern = r'"reasoning"\s*:\s*"([^"]*)"'
        
        response_match = re.search(response_pattern, text, re.DOTALL)
        reasoning_match = re.search(reasoning_pattern, text, re.DOTALL)
        
        if response_match:
            # Try to extract the full object
            parsed = _try_parse_json(text)
            if parsed and ("response" in parsed or "reasoning" in parsed):
                results.append(parsed)
    
    # Method 4: Last resort - look for any JSON-like structure
    if not results:
        # Find content between first { and last }
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            candidate = text[start:end+1]
            parsed = _try_parse_json(candidate)
            if parsed is not None:
                results.append(parsed)
    
    return results or None


class TaskAgent:
    """Task agent that solves IMO grading problems with enhanced reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.max_retries = 3
        self.validation_enabled = True
    
    def set_validation(self, enabled: bool) -> None:
        """Enable or disable input validation.
        
        Args:
            enabled: True to enable validation, False to disable
        """
        self.validation_enabled = enabled
        self.log_fn(f"Input validation {'enabled' if enabled else 'disabled'}")

    def _build_grading_prompt(self, inputs: dict) -> str:
        """Build a comprehensive prompt for the grading task."""
        domain = inputs.get("domain", "")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        # Extract max score from grading guidelines if available
        max_score = self._extract_max_score(grading_guidelines)
        
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
4. Identify any errors, omissions, or incorrect reasoning
5. Award partial credit where appropriate based on the guidelines
6. Provide your reasoning before giving the final grade
7. Respond ONLY in the following JSON format:

<json>
{{
    "reasoning": "Your detailed step-by-step analysis and reasoning. Include: (1) What the student did correctly, (2) Any errors or omissions, (3) How partial credit was determined",
    "response": "The final grade/score (a number or specific grade value){f', max {max_score}' if max_score else ''}"
}}
</json>

Critical Requirements:
- The "response" field must contain ONLY the final grade/score (e.g., "7", "0", "1", "Correct", "Incorrect")
- For numeric scores, provide only the number without any explanation
- The "reasoning" field should contain your detailed analysis
- Do NOT include any text outside the <json>...</json> tags
- Ensure the JSON is valid with proper quotes and no trailing commas
- Be precise and follow the grading guidelines exactly
- If the guidelines specify a maximum score, do not exceed it"""

    def _extract_max_score(self, grading_guidelines: str) -> int | None:
        """Extract maximum score from grading guidelines if specified."""
        import re
        # Look for patterns like "maximum score: 7", "out of 7", "total: 7 points", etc.
        patterns = [
            r'(?:maximum|max|total)(?:\s+score)?[:\s]+(\d+)',
            r'out\s+of\s+(\d+)',
            r'(\d+)\s+points?',
            r'score\s+range[:\s]+\d+[-\s]+(\d+)',
        ]
        for pattern in patterns:
            match = re.search(pattern, grading_guidelines, re.IGNORECASE)
            if match:
                return int(match.group(1))
        return None

    def _try_extract_prediction(self, text: str) -> tuple[str, str | None]:
        """Try to extract prediction from response text.
        
        Returns:
            (prediction, reasoning)
        """
        try:
            extracted = _extract_json_with_retry(text)
            if extracted:
                # Use the last valid JSON object
                last = extracted[-1]
                
                # Extract prediction from "response" field
                prediction = last.get("response")
                if prediction is None:
                    # Try alternative field names
                    prediction = last.get("grade", last.get("score", last.get("answer", "None")))
                
                # Extract reasoning
                reasoning = last.get("reasoning")
                
                # Clean up the prediction
                if prediction is not None and prediction != "None":
                    prediction = str(prediction).strip()
                    # Remove any extra text after the grade
                    # e.g., "7 points" -> "7", "7/7" -> "7"
                    import re
                    numeric_match = re.match(r'^([\d.]+)', prediction)
                    if numeric_match:
                        prediction = numeric_match.group(1)
                else:
                    prediction = "None"
                
                return prediction, reasoning
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
        
        return "None", None

    def _extract_grade_fallback(self, text: str) -> str | None:
        """Last-resort extraction for grades from non-JSON responses.
        
        Tries to find numeric grades or common grade patterns in text.
        """
        import re
        
        # Look for patterns like "Grade: 7", "Score: 7", "Final grade: 7"
        patterns = [
            r'(?:grade|score|final|result)[:\s]+([\d.]+)',
            r'(?:award|give|assign)(?:ed)?[:\s]+([\d.]+)',
            r'(?:^|\n)([\d.]+)\s*(?:points?|/\s*\d+)',
            r'(?:^|\n)([\d.]+)\s*(?:out\s+of)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        # Look for standalone numbers that could be grades (0-10 range)
        numbers = re.findall(r'\b([0-9]|10)\b', text)
        if numbers:
            # Return the last number found (often the final grade)
            return numbers[-1]
        
        return None

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with retry logic.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Validate inputs if enabled
        if self.validation_enabled:
            is_valid, error_msg = _validate_inputs(inputs)
            if not is_valid:
                self.log_fn(f"Input validation failed: {error_msg}")
                return f"Error: {error_msg}", []
        
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
                
                # Extract prediction
                text = msg_history[-1]["text"] if msg_history else ""
                pred, reasoning = self._try_extract_prediction(text)
                
                if pred != "None":
                    prediction = pred
                    if reasoning:
                        self.log_fn(f"Grading reasoning: {reasoning[:200]}...")
                    break
                
                # If extraction failed, try fallback extraction
                if pred == "None":
                    fallback = self._extract_grade_fallback(text)
                    if fallback:
                        self.log_fn(f"Used fallback extraction, found grade: {fallback}")
                        prediction = fallback
                        break
                
                # If extraction failed, add a follow-up message asking for proper format
                if attempt < self.max_retries - 1:
                    instruction = """Your previous response did not contain valid JSON in the required format.

Please respond ONLY with a valid JSON object wrapped in <json>...</json> tags like this:

<json>
{
    "reasoning": "Your detailed analysis here",
    "response": "The final grade/score here"
}
</json>

Do not include any other text outside the JSON tags."""
                    
            except Exception as e:
                self.log_fn(f"Attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    break
        
        return str(prediction), msg_history
