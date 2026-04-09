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
    cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\s*```$', '', cleaned)
    cleaned = re.sub(r'\s*```\s*$', '', cleaned)
    
    # Remove trailing commas before closing braces/brackets
    cleaned = re.sub(r',(\s*[}\]])', r'\1', cleaned)
    
    # Fix single quotes to double quotes (handle nested cases)
    # First handle keys (before colon)
    cleaned = re.sub(r"'([^']*?)'(?=\s*:)", r'"\1"', cleaned)
    # Then handle values (after colon)
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
    
    # Fix unescaped newlines in string values (common LLM error)
    # This is a more aggressive fix - replace newlines in the middle of strings
    def fix_newlines_in_strings(match):
        content = match.group(1)
        # Replace literal newlines with \n escape sequence
        content = content.replace('\n', '\\n').replace('\r', '\\r')
        return '"' + content + '"'
    
    # Find string values and fix newlines within them
    cleaned = re.sub(r'"([^"]*\n[^"]*)"', fix_newlines_in_strings, cleaned)
    
    # Fix common LLM JSON errors: missing quotes around string values
    # Pattern: : something (not in quotes, not a number, not true/false/null, not an object/array)
    def fix_unquoted_values(match):
        prefix = match.group(1)
        value = match.group(2)
        suffix = match.group(3)
        # Check if value is a simple string that should be quoted
        if value and value not in ('true', 'false', 'null') and not re.match(r'^-?\d', value):
            if not (value.startswith('"') or value.startswith('[') or value.startswith('{')):
                return f'{prefix}"{value}"{suffix}'
        return match.group(0)
    
    cleaned = re.sub(r'(:\s*)([^"\s\[\{][^,}\]]*?)(\s*[,}\]])', fix_unquoted_values, cleaned)
    
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
    
    # If no <json> tags found, try to find JSON objects directly
    if not results:
        # Try to find JSON objects in code blocks
        code_block_pattern = r'```(?:json)?\s*(\{[\s\S]*?\})\s*```'
        matches = re.findall(code_block_pattern, text, re.DOTALL | re.IGNORECASE)
        for match in matches:
            parsed = _try_parse_json(match)
            if parsed:
                results.append(parsed)
    
    # If still no results, try to find any JSON-like structure
    if not results:
        # Look for content between first { and last }
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            candidate = text[start:end+1]
            parsed = _try_parse_json(candidate)
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
    
    # Strategy 3: Extract fields using regex with improved patterns
    try:
        result = {}
        
        # Extract response field with various patterns (case-insensitive)
        response_patterns = [
            r'"response"\s*:\s*"([^"]+)"',
            r'"response"\s*:\s*\'([^\']+)\'',
            r'"response"\s*:\s*([a-zA-Z_][a-zA-Z0-9_]*)',
            r'response["\']?\s*[=:]\s*["\']?([^"\',}\]]+)',
            r'response\s*[=:]\s*([A-Za-z]+)',  # Simple word match
        ]
        for pattern in response_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                result["response"] = match.group(1).strip().strip('"\'')
                break
        
        # Extract reasoning field with various patterns
        reasoning_patterns = [
            r'"reasoning"\s*:\s*"((?:[^"\\]|\\.)*?)"',
            r'"reasoning"\s*:\s*\'((?:[^\'\\]|\\.)*?)\'',
            r'"reasoning"\s*:\s*"([^"]*)"',
            r'"reasoning"\s*:\s*\'([^\']*)\'',
            r'reasoning["\']?\s*[=:]\s*["\']?([^"\',}\]]*)',
        ]
        for pattern in reasoning_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                reasoning = match.group(1).strip().strip('"\'')
                # Unescape if needed
                reasoning = reasoning.replace('\\n', '\n').replace('\\t', '\t').replace('\\"', '"')
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
    
    # Strategy 5: Look for grade keywords directly in text
    try:
        text_lower = text.lower()
        for alias, canonical in GRADE_ALIASES.items():
            # Use word boundary matching for more precision
            pattern = r'\b' + re.escape(alias) + r'\b'
            if re.search(pattern, text_lower):
                return {
                    "response": canonical,
                    "reasoning": f"Extracted from text analysis: found '{alias}'"
                }
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
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
        for match in matches:
            parsed = _try_parse_json(match)
            if parsed:
                results.append(parsed)
    
    # Strategy 2: Try to find any JSON-like structure with braces (improved pattern)
    if not results:
        # Find all potential JSON objects - improved to handle nested structures better
        # Use a more robust approach: find all { and try to match with corresponding }
        start_indices = [m.start() for m in re.finditer(r'\{', text)]
        for start in start_indices:
            # Try to find a valid JSON object starting at this position
            for end in range(start + 1, len(text) + 1):
                candidate = text[start:end]
                if candidate.count('{') == candidate.count('}'):
                    parsed = _try_parse_json(candidate)
                    if parsed:
                        results.append(parsed)
                        break
    
    # Strategy 3: Extract fields directly from text
    if not results:
        parsed = _try_parse_json(text)
        if parsed:
            results.append(parsed)
    
    # Strategy 4: Look for grade keywords in the text (improved with priority)
    if not results:
        text_lower = text.lower()
        # Priority order: check for exact matches first
        priority_aliases = ["correct", "partial", "incorrect"]
        for alias in priority_aliases:
            pattern = r'\b' + re.escape(alias) + r'\b'
            if re.search(pattern, text_lower):
                results.append({
                    "response": GRADE_ALIASES[alias],
                    "reasoning": f"Extracted from text analysis: found '{alias}'"
                })
                break
        else:
            # Check other aliases if priority ones not found
            for alias, canonical in GRADE_ALIASES.items():
                if alias not in priority_aliases:
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
                # Try to extract just the first word and match
                first_word = re.match(r'([a-zA-Z]+)', response_lower)
                if first_word:
                    first_word = first_word.group(1)
                    if first_word in GRADE_ALIASES:
                        data["response"] = GRADE_ALIASES[first_word]
                    else:
                        return False, f"Invalid response value: '{response}'. Expected one of: {VALID_GRADES}"
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
        self._last_raw_response = ""  # Store last raw response for debugging

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
        
        base_prompt = f"""You are an expert grader for {domain} problems. Your task is to evaluate the student's answer with precision and fairness.

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
You MUST respond with ONLY valid JSON wrapped in <json>...</json> tags. Do not include any text before or after the JSON block.

<json>
{{
    "reasoning": "Your step-by-step analysis here. Explain: [1] Problem understanding... [2] Student approach... [3] Comparison with solution... [4] Grade justification...",
    "response": "Correct"
}}
</json>

STRICT REQUIREMENTS:
1. ONLY output the JSON block - no other text before or after
2. 'response' MUST be exactly one of: "Correct", "Partial", or "Incorrect" (case-sensitive, no extra spaces)
3. 'reasoning' must be detailed but concise (explain your decision in 1-3 sentences)
4. Ensure valid JSON: use double quotes only, no trailing commas, escape newlines with \\n"""

        if is_retry and previous_error:
            return f"""PREVIOUS ATTEMPT FAILED: {previous_error}

You MUST correct this error. Follow these instructions EXACTLY:

1. Output ONLY a valid JSON object wrapped in <json>...</json> tags
2. NO text before or after the JSON block
3. The 'response' field must be exactly one of: "Correct", "Partial", "Incorrect"
4. Check your JSON syntax carefully:
   - Use double quotes (") not single quotes (')
   - No trailing commas after the last property
   - Escape any newlines in strings with \\n
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
        
        # If none valid, try to extract from the last one anyway
        if extracted:
            last_json = extracted[-1]
            prediction = str(last_json.get("response", "None"))
            reasoning = str(last_json.get("reasoning", ""))
            
            # Try to normalize the prediction if it's not a valid grade
            if prediction not in VALID_GRADES:
                prediction_lower = prediction.lower()
                if prediction_lower in GRADE_ALIASES:
                    prediction = GRADE_ALIASES[prediction_lower]
                else:
                    # Try to find any valid grade in the text
                    for alias, canonical in GRADE_ALIASES.items():
                        if alias in prediction_lower:
                            prediction = canonical
                            break
        
        return prediction, reasoning, error_msg or "Validation failed"

    def _extract_with_fallback(self, text: str) -> tuple[str, str, str]:
        """Extract prediction with multiple fallback strategies.
        
        Returns:
            (prediction, reasoning, error_message) tuple
        """
        # Strategy 1: Try standard extraction
        prediction, reasoning, error_msg = self._extract_prediction(text)
        if prediction in VALID_GRADES:
            return prediction, reasoning, ""
        
        # Strategy 2: Try to find grade keywords directly in text
        text_lower = text.lower()
        for alias, canonical in GRADE_ALIASES.items():
            pattern = r'\b' + re.escape(alias) + r'\b'
            if re.search(pattern, text_lower):
                return canonical, f"Extracted from text: found '{alias}'", ""
        
        # Strategy 3: Look for patterns like "grade: Correct" or "result: Partial"
        grade_patterns = [
            r'grade\s*[:=]\s*([a-zA-Z]+)',
            r'result\s*[:=]\s*([a-zA-Z]+)',
            r'evaluation\s*[:=]\s*([a-zA-Z]+)',
            r'answer\s*is\s+([a-zA-Z]+)',
        ]
        for pattern in grade_patterns:
            match = re.search(pattern, text_lower)
            if match:
                word = match.group(1).lower()
                if word in GRADE_ALIASES:
                    return GRADE_ALIASES[word], f"Extracted from pattern: '{match.group(0)}'", ""
        
        return prediction, reasoning, error_msg or "All extraction strategies failed"

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
                self._last_raw_response = last_text  # Store for debugging
                
                # Use enhanced extraction with fallbacks
                prediction, reasoning, error_msg = self._extract_with_fallback(last_text)
                
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
            # Try one more time to extract from the last response
            if msg_history:
                last_text = msg_history[-1]["text"] if msg_history else ""
                self._last_raw_response = last_text  # Store for debugging
                # Try to find any valid grade in the text
                text_lower = last_text.lower()
                for alias, canonical in GRADE_ALIASES.items():
                    pattern = r'\b' + re.escape(alias) + r'\b'
                    if re.search(pattern, text_lower):
                        prediction = canonical
                        self.log_fn(f"Final extraction: Found grade '{canonical}' in text")
                        break
                else:
                    self.log_fn(f"Warning: Final prediction '{prediction}' not in valid grades, defaulting to 'Incorrect'")
                    prediction = "Incorrect"
            else:
                self.log_fn(f"Warning: Final prediction '{prediction}' not in valid grades, defaulting to 'Incorrect'")
                prediction = "Incorrect"
        
        return str(prediction), msg_history
    
    def get_last_raw_response(self) -> str:
        """Get the last raw LLM response for debugging purposes."""
        return self._last_raw_response
