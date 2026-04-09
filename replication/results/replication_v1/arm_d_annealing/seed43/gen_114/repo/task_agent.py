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

# Grade aliases for fuzzy matching
GRADE_ALIASES = {
    "correct": "Correct",
    "right": "Correct",
    "true": "Correct",
    "yes": "Correct",
    "partial": "Partial",
    "partially correct": "Partial",
    "partially": "Partial",
    "somewhat correct": "Partial",
    "incorrect": "Incorrect",
    "wrong": "Incorrect",
    "false": "Incorrect",
    "no": "Incorrect",
    "none": "Incorrect",
}


def _clean_json_string(text: str) -> str:
    """Clean common JSON formatting issues with comprehensive fixes."""
    cleaned = text.strip()
    
    # Remove markdown code block markers
    cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
    cleaned = re.sub(r'\s*```$', '', cleaned)
    
    # Remove trailing commas before closing braces/brackets
    cleaned = re.sub(r',(\s*[}\]])', r'\1', cleaned)
    
    # Fix single quotes to double quotes (handle nested cases)
    cleaned = re.sub(r"'([^']*?)'(?=\s*:)", r'"\1"', cleaned)
    cleaned = re.sub(r":\s*'([^']*?)'(?=\s*[,}\]])", r': "\1"', cleaned)
    
    # Remove control characters except newlines and tabs
    cleaned = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', cleaned)
    
    # Fix escaped newlines that might break parsing
    cleaned = cleaned.replace('\\n', '\n')
    cleaned = cleaned.replace('\\t', '\t')
    
    # Fix double-escaped quotes
    cleaned = cleaned.replace('\\"', '"')
    
    # Remove BOM if present
    cleaned = cleaned.lstrip('\ufeff')
    
    # Fix missing quotes around keys (simple heuristic)
    cleaned = re.sub(r'(\{|,\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', cleaned)
    
    return cleaned


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks with enhanced robustness.

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
        
        # Try to parse the JSON with multiple fallback strategies
        parsed = _try_parse_json(inner)
        if parsed:
            results.append(parsed)
    
    return results or None


def _try_parse_json(text: str) -> dict | None:
    """Try multiple strategies to parse JSON text.
    
    Returns the parsed dict or None if all strategies fail.
    """
    # Strategy 1: Direct parsing
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Strategy 2: Clean and parse
    try:
        cleaned = _clean_json_string(text)
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    
    # Strategy 3: Extract fields using regex
    try:
        result = {}
        
        # Extract response field with various patterns
        response_patterns = [
            r'"response"\s*:\s*"([^"]+)"',
            r'"response"\s*:\s*\'([^\']+)\'',
            r'"response"\s*:\s*([a-zA-Z_][a-zA-Z0-9_]*)',
            r'response["\']?\s*[=:]\s*["\']?([^"\',}\]]+)',
        ]
        for pattern in response_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                result["response"] = match.group(1).strip().strip('"\'')
                break
        
        # Extract reasoning field with various patterns
        reasoning_patterns = [
            r'"reasoning"\s*:\s*"([^"]*)"',
            r'"reasoning"\s*:\s*\'([^\']*)\'',
            r'"reasoning"\s*:\s*"((?:[^"\\]|\\.)*)"',
            r'reasoning["\']?\s*[=:]\s*["\']?([^"\',}\]]*)',
        ]
        for pattern in reasoning_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                reasoning = match.group(1).strip().strip('"\'')
                # Unescape if needed
                reasoning = reasoning.replace('\\n', '\n').replace('\\t', '\t')
                result["reasoning"] = reasoning
                break
        
        if "response" in result:
            return result
    except Exception:
        pass
    
    # Strategy 4: Try to find any JSON-like structure
    try:
        # Look for content between first { and last }
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            candidate = text[start:end+1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                cleaned = _clean_json_string(candidate)
                return json.loads(cleaned)
    except Exception:
        pass
    
    return None


def _extract_json_with_regex(text: str) -> list[dict] | None:
    """Enhanced fallback JSON extraction using multiple strategies."""
    results = []
    
    # Strategy 1: Try to find JSON objects in code blocks
    code_block_patterns = [
        r'```(?:json)?\s*(\{[\s\S]*?\})\s*```',
        r'```\s*(\{[\s\S]*?\})\s*```',
        r'<json>\s*(\{[\s\S]*?\})\s*</json>',
    ]
    for pattern in code_block_patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            parsed = _try_parse_json(match)
            if parsed:
                results.append(parsed)
    
    # Strategy 2: Try to find any JSON-like structure with braces
    if not results:
        # Find all potential JSON objects
        brace_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(brace_pattern, text, re.DOTALL)
        for match in matches:
            parsed = _try_parse_json(match)
            if parsed:
                results.append(parsed)
    
    # Strategy 3: Extract fields directly from text
    if not results:
        parsed = _try_parse_json(text)
        if parsed:
            results.append(parsed)
    
    # Strategy 4: Look for grade keywords in the text
    if not results:
        text_lower = text.lower()
        for alias, canonical in GRADE_ALIASES.items():
            if alias in text_lower:
                # Check if it's a standalone word
                pattern = r'\b' + re.escape(alias) + r'\b'
                if re.search(pattern, text_lower):
                    results.append({
                        "response": canonical,
                        "reasoning": f"Extracted from text analysis: found '{alias}'"
                    })
                    break
    
    return results or None


def _validate_grading_output(data: dict) -> tuple[bool, str]:
    """Validate and normalize the extracted grading output.
    
    Returns:
        (is_valid, error_message) tuple
    """
    if not isinstance(data, dict):
        return False, "Extracted data is not a dictionary"
    
    if "response" not in data:
        return False, "Missing 'response' field"
    
    response = str(data.get("response", "")).strip()
    
    # Check if response is a valid grade
    if response not in VALID_GRADES:
        # Try to normalize using GRADE_ALIASES
        response_lower = response.lower()
        if response_lower in GRADE_ALIASES:
            data["response"] = GRADE_ALIASES[response_lower]
        else:
            # Try fuzzy matching for partial matches
            for alias, canonical in GRADE_ALIASES.items():
                if alias in response_lower:
                    data["response"] = canonical
                    break
            else:
                return False, f"Invalid response value: '{response}'. Expected one of: {VALID_GRADES}"
    
    # Validate reasoning field exists and is non-empty
    reasoning = data.get("reasoning", "")
    if not reasoning or not str(reasoning).strip():
        data["reasoning"] = "No reasoning provided"
    
    return True, ""


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.max_retries = 3

    def _build_grading_prompt(self, inputs: dict, is_retry: bool = False, previous_error: str = "") -> str:
        """Build an optimized prompt for the grading task with chain-of-thought.
        
        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer
            is_retry: Whether this is a retry attempt
            previous_error: Error message from previous attempt (if any)
        """
        domain = inputs.get("domain", "unknown")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        base_prompt = f"""You are an expert grader for {domain} problems. Evaluate the student's answer with precision and fairness.

## Problem Statement
{problem}

## Correct Solution
{solution}

## Grading Guidelines
{guidelines}

## Student's Answer
{student_answer}

## Evaluation Process
Follow this systematic approach:

1. **Understand the Problem**: Identify the key concepts, required steps, and expected outcomes.

2. **Analyze the Student's Approach**:
   - What method did they use?
   - Is it mathematically valid (even if different from the solution)?
   - Did they show their work clearly?

3. **Compare with Correct Solution**:
   - Check each step for correctness
   - Verify the final answer
   - Note any missing or extra steps

4. **Apply Grading Guidelines**:
   - Consider partial credit rules
   - Check for common error patterns mentioned in guidelines
   - Evaluate reasoning quality

5. **Determine Final Grade**:
   - **Correct**: Fully correct reasoning AND correct final answer (or equivalent valid approach)
   - **Partial**: Some correct reasoning OR partially correct answer (e.g., correct method but calculation error)
   - **Incorrect**: Fundamentally wrong approach OR completely wrong answer with no valid reasoning

## Response Format (CRITICAL - FOLLOW EXACTLY)
You MUST respond with ONLY valid JSON wrapped in <json>...</json> tags.

<json>
{{
    "reasoning": "Step-by-step analysis: [1] Problem understanding... [2] Student approach... [3] Comparison with solution... [4] Grade justification...",
    "response": "Correct"
}}
</json>

STRICT REQUIREMENTS:
1. ONLY output the JSON block - no other text
2. 'response' MUST be exactly: "Correct", "Partial", or "Incorrect" (case-sensitive, no extra spaces)
3. 'reasoning' must be detailed but concise (explain your decision)
4. Ensure valid JSON: proper quotes, no trailing commas, no unescaped newlines in strings"""

        if is_retry and previous_error:
            return f"""PREVIOUS ATTEMPT FAILED: {previous_error}

You MUST correct this error. Follow these instructions EXACTLY:

1. Output ONLY a valid JSON object wrapped in <json>...</json> tags
2. NO text before or after the JSON block
3. The 'response' field must be exactly one of: "Correct", "Partial", "Incorrect"
4. Check your JSON syntax carefully - ensure all quotes match and no trailing commas

Example of CORRECT format:
<json>
{{
    "reasoning": "The student used the correct formula and arrived at the right answer through valid steps.",
    "response": "Correct"
}}
</json>

Now retry the grading task:

{base_prompt}"""
        
        return base_prompt

    def _extract_prediction(self, text: str) -> tuple[str, str, str]:
        """Extract prediction and reasoning from response text with validation.
        
        Returns:
            (prediction, reasoning, error_message) tuple
        """
        prediction = "None"
        reasoning = ""
        error_msg = ""
        
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
                return prediction, reasoning, ""
        
        # If none valid, return the last one with error
        if extracted:
            last_json = extracted[-1]
            prediction = str(last_json.get("response", "None"))
            reasoning = str(last_json.get("reasoning", ""))
        
        return prediction, reasoning, error_msg or "Validation failed"

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with enhanced error handling.

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
        
        # Retry loop with exponential backoff and enhanced error handling
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
                if prediction != "None" and prediction in VALID_GRADES:
                    self.log_fn(f"Successfully extracted prediction: {prediction} (attempt {attempt + 1})")
                    if reasoning:
                        self.log_fn(f"Reasoning: {reasoning[:200]}...")
                    return str(prediction), msg_history
                else:
                    last_error = error_msg if error_msg else f"Invalid prediction: '{prediction}'"
                    self.log_fn(f"Attempt {attempt + 1} failed: {last_error}")
                    
                    # Add feedback for retry
                    if attempt < self.max_retries - 1:
                        instruction = self._build_grading_prompt(inputs, is_retry=True, previous_error=last_error)
                    
            except Exception as e:
                last_error = str(e)
                self.log_fn(f"Error in attempt {attempt + 1}: {e}")
                if attempt < self.max_retries - 1:
                    # Exponential backoff with jitter
                    base_wait = min(4, 2 ** attempt)
                    jitter = random.uniform(0, 1)
                    wait = base_wait + jitter
                    time.sleep(wait)
                else:
                    prediction = f"Error: {e}"
        
        # Final validation - ensure we return a valid grade or error
        if prediction not in VALID_GRADES and not prediction.startswith("Error:"):
            self.log_fn(f"Warning: Final prediction '{prediction}' not in valid grades, defaulting to 'Incorrect'")
            prediction = "Incorrect"
        
        return str(prediction), msg_history
