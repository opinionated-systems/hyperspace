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


def _clean_and_parse_json(json_str: str) -> dict | None:
    """Try to parse JSON with various cleaning strategies.
    
    Pre-compiled regex patterns for efficiency.
    """
    json_str = json_str.strip()
    if not json_str:
        return None
    
    # Pre-compiled patterns for efficiency
    _TRAILING_COMMA_RE = re.compile(r',(\s*[}\]])')
    _SINGLE_QUOTE_RE = re.compile(r"(?<!\\)'")
    _LINE_COMMENT_RE = re.compile(r'//.*?$')
    _BLOCK_COMMENT_RE = re.compile(r'/\*.*?\*/')
    
    # Try direct parsing first (fast path)
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass
    
    # Cleaning strategies in order of likelihood of success
    strategies = [
        # Strategy 1: Remove trailing commas
        lambda s: _TRAILING_COMMA_RE.sub(r'\1', s),
        # Strategy 2: Fix single quotes
        lambda s: _SINGLE_QUOTE_RE.sub('"', s),
        # Strategy 3: Remove comments
        lambda s: _BLOCK_COMMENT_RE.sub('', _LINE_COMMENT_RE.sub('', s)),
        # Strategy 4: Combined cleaning
        lambda s: _TRAILING_COMMA_RE.sub(r'\1', _SINGLE_QUOTE_RE.sub('"', s)),
    ]
    
    for strategy in strategies:
        try:
            cleaned = strategy(json_str)
            return json.loads(cleaned)
        except json.JSONDecodeError:
            continue
    
    return None


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    Also handles markdown code blocks with json tag.
    Includes robust JSON cleaning for common LLM output errors.
    
    Optimized for speed with early termination and batch processing.
    """
    results = []
    
    # Fast path: check if text contains any JSON indicators
    if '<json>' not in text and '```' not in text and '{' not in text:
        return None
    
    # First, try to find <json>...</json> tags (most reliable)
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
        
        parsed = _clean_and_parse_json(inner)
        if parsed is not None:
            results.append(parsed)
    
    # If no <json> tags found, try markdown code blocks
    if not results:
        search_from = 0
        while True:
            start = text.find("```json", search_from)
            if start == -1:
                # Also try plain ``` without json specifier
                start = text.find("```", search_from)
                if start == -1:
                    break
                content_start = start + 3
            else:
                content_start = start + 7
            
            # Find the closing ```
            end = text.find("```", content_start)
            if end == -1:
                break
            
            inner = text[content_start:end].strip()
            search_from = end + 3
            
            parsed = _clean_and_parse_json(inner)
            if parsed is not None:
                results.append(parsed)
    
    return results or None


def _extract_json_with_regex(text: str) -> list[dict] | None:
    """Fallback JSON extraction using regex for malformed responses.
    
    Handles nested braces, code blocks, and various formatting edge cases.
    Optimized with pre-compiled patterns and efficient parsing.
    """
    # Pre-compiled patterns for efficiency
    _CODE_BLOCK_RE = re.compile(r'```(?:json)?\s*([\s\S]*?)\s*```')
    _JSON_TAG_RE = re.compile(r'<json>\s*([\s\S]*?)\s*</json>', re.IGNORECASE)
    _RESPONSE_RE = re.compile(r'["\']?response["\']?\s*[:=]\s*["\']?([^"\'\n,}]+)', re.IGNORECASE)
    _REASONING_RE = re.compile(r'["\']?reasoning["\']?\s*[:=]\s*["\']?([^"\']*(?:\.[^"\']*)*)["\']?', re.IGNORECASE | re.DOTALL)
    
    results = []
    
    def extract_balanced_json(content: str) -> list[dict]:
        """Extract all balanced JSON objects using stack-based parsing."""
        objects = []
        brace_count = 0
        start_idx = -1
        in_string = False
        escape_next = False
        
        for i, char in enumerate(content):
            if escape_next:
                escape_next = False
                continue
            if char == '\\':
                escape_next = True
                continue
            if char == '"' and not in_string:
                in_string = True
                continue
            if char == '"' and in_string:
                in_string = False
                continue
            if in_string:
                continue
            
            if char == '{':
                if brace_count == 0:
                    start_idx = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start_idx != -1:
                    snippet = content[start_idx:i+1]
                    parsed = _clean_and_parse_json(snippet)
                    if parsed:
                        objects.append(parsed)
                    start_idx = -1
        return objects
    
    # Try code blocks first
    for match in _CODE_BLOCK_RE.finditer(text):
        objects = extract_balanced_json(match.group(1))
        results.extend(objects)
    
    # Try <json> tags
    if not results:
        for match in _JSON_TAG_RE.finditer(text):
            objects = extract_balanced_json(match.group(1))
            results.extend(objects)
    
    # Try raw balanced JSON
    if not results:
        results = extract_balanced_json(text)
    
    # Final fallback: key-value patterns
    if not results:
        result: dict[str, Any] = {}
        
        match = _RESPONSE_RE.search(text)
        if match:
            result["response"] = match.group(1).strip()
        
        match = _REASONING_RE.search(text)
        if match:
            result["reasoning"] = match.group(1).strip()
        
        if result:
            results.append(result)
    
    return results or None


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning.
    
    Optimized for efficiency with pre-compiled patterns and caching.
    """
    
    # Pre-compiled patterns for prediction normalization
    _VALID_GRADES = frozenset(["correct", "partial", "incorrect"])
    _CORRECT_SYNONYMS = frozenset(["correct", "right", "true", "yes", "1", "full", "valid", "accurate"])
    _INCORRECT_SYNONYMS = frozenset(["incorrect", "wrong", "false", "no", "0", "none", "invalid", "inaccurate", "error"])
    _PARTIAL_SYNONYMS = frozenset(["partial", "partially correct", "partial credit", "half", "mostly correct", "incomplete"])
    
    # Template for retry instructions (cached)
    _RETRY_TEMPLATE = """ERROR: Your previous response did not contain valid JSON with a 'response' field.

You MUST respond with ONLY valid JSON wrapped in <json>...</json> tags. Do not include any text before or after the JSON tags.

Correct format:
<json>
{{
    "reasoning": "Your detailed analysis here...",
    "response": "Correct"
}}
</json>

Examples of valid responses:

Example 1 (Correct):
<json>
{{
    "reasoning": "The student's answer correctly identifies the key theorem and applies it properly.",
    "response": "Correct"
}}
</json>

Example 2 (Partial):
<json>
{{
    "reasoning": "The student started with the right approach but made an arithmetic error.",
    "response": "Partial"
}}
</json>

Example 3 (Incorrect):
<json>
{{
    "reasoning": "The student misunderstood the problem and applied an incorrect theorem.",
    "response": "Incorrect"
}}
</json>

Common mistakes to avoid:
- Do not use markdown code blocks (```json) - use <json> tags instead
- Ensure the JSON is valid: no trailing commas, proper quotes around strings
- The 'response' field must contain only the grade: "Correct", "Partial", or "Incorrect"
- The 'reasoning' field should contain your detailed analysis
- Do not include any text outside the <json>...</json> tags

Now try again with the original task:

{prompt}"""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.max_retries = 3

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
- **Correct**: The answer matches the solution or uses an equivalent valid approach with correct reasoning and final result. All key steps are present and correct.
- **Partial**: The answer shows some correct reasoning but has minor errors, incomplete steps, or partially correct results. The student understood the core concept but made execution mistakes.
- **Incorrect**: The answer contains fundamental errors, wrong approach, or completely wrong results. The student misunderstood the problem or made critical errors.

## Response Format (REQUIRED - READ CAREFULLY)
You MUST respond with ONLY valid JSON wrapped in <json>...</json> tags. Do not include any text before or after the JSON tags.

<json>
{{
    "reasoning": "Your detailed step-by-step analysis and reasoning...",
    "response": "Correct"
}}
</json>

CRITICAL REQUIREMENTS:
1. Use <json> tags, NOT markdown code blocks (```json)
2. The JSON must be valid: no trailing commas, all strings in double quotes
3. The 'response' field must contain ONLY one of: "Correct", "Partial", or "Incorrect"
4. The 'reasoning' field should contain your detailed analysis
5. Do not include any text outside the <json>...</json> tags
6. Ensure proper JSON syntax: {{"key": "value", "key2": "value2"}}

## Examples of Valid Responses

Example 1 (Correct):
<json>
{{
    "reasoning": "The student's answer correctly identifies the key theorem and applies it properly. All steps are logically sound and the final result matches the solution.",
    "response": "Correct"
}}
</json>

Example 2 (Partial):
<json>
{{
    "reasoning": "The student started with the right approach and identified the correct formula, but made an arithmetic error in the final calculation step.",
    "response": "Partial"
}}
</json>

Example 3 (Incorrect):
<json>
{{
    "reasoning": "The student misunderstood the problem statement and applied an incorrect theorem. The approach is fundamentally flawed.",
    "response": "Incorrect"
}}
</json>"""

    def _normalize_prediction(self, prediction: str) -> str:
        """Normalize prediction to standard grade values.
        
        Uses frozenset lookups for O(1) performance.
        """
        if prediction == "None":
            return "None"
        
        # Clean up prediction
        prediction = prediction.strip('"\'').rstrip('.!?,:;')
        prediction_lower = prediction.lower()
        
        # Fast lookup using frozensets
        if prediction_lower in self._CORRECT_SYNONYMS:
            return "Correct"
        elif prediction_lower in self._INCORRECT_SYNONYMS:
            return "Incorrect"
        elif prediction_lower in self._PARTIAL_SYNONYMS:
            return "Partial"
        elif prediction_lower in self._VALID_GRADES:
            return prediction.capitalize()
        else:
            self.log_fn(f"Warning: Unrecognized prediction value: '{prediction}'")
            return "None"

    def _extract_prediction(self, text: str) -> tuple[str, str]:
        """Extract prediction and reasoning from response text.
        
        Returns:
            (prediction, reasoning) tuple
        """
        prediction = "None"
        reasoning = ""
        
        # Try primary extraction method
        extracted = _extract_jsons(text)
        if extracted is None:
            # Fallback to regex extraction
            extracted = _extract_json_with_regex(text)
        
        if extracted:
            # Find the best JSON object (prefer one with both response and reasoning)
            best_json = None
            for json_obj in extracted:
                if "response" in json_obj:
                    if best_json is None or ("reasoning" in json_obj and "reasoning" not in best_json):
                        best_json = json_obj
            
            if best_json is None:
                best_json = extracted[-1]
            
            if "response" in best_json:
                prediction = str(best_json["response"]).strip()
            if "reasoning" in best_json:
                reasoning = str(best_json["reasoning"]).strip()
        
        # Normalize prediction
        prediction = self._normalize_prediction(prediction)
        
        return prediction, reasoning

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        base_prompt = self._build_grading_prompt(inputs)
        instruction = base_prompt
        
        msg_history: list[dict] = []
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
                
                last_text = msg_history[-1].get("text", "") if msg_history else ""
                prediction, reasoning = self._extract_prediction(last_text)
                
                if prediction != "None":
                    self.log_fn(f"Successfully extracted prediction: {prediction}")
                    if reasoning:
                        self.log_fn(f"Reasoning: {reasoning[:200]}...")
                    break
                else:
                    self.log_fn(f"Attempt {attempt + 1}: Failed to extract prediction, retrying...")
                    # Add feedback for retry using cached template
                    if attempt < self.max_retries - 1:
                        instruction = self._RETRY_TEMPLATE.format(prompt=base_prompt)
                    
            except Exception as e:
                self.log_fn(f"Error in attempt {attempt + 1}: {e}")
                if attempt == self.max_retries - 1:
                    prediction = f"Error: {e}"
        
        return str(prediction), msg_history
