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

from agent.llm_client import get_response_from_llm, EVAL_MODEL

logger = logging.getLogger(__name__)


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    Includes comprehensive JSON repair for common LLM output issues.
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
        
        # Try parsing with progressively more aggressive fixes
        parsed = _try_parse_json_with_repair(inner)
        if parsed is not None:
            results.append(parsed)
    
    return results or None


def _try_parse_json_with_repair(text: str) -> dict | None:
    """Attempt to parse JSON with multiple repair strategies.
    
    Tries raw parsing first, then applies progressively more aggressive fixes
    for common LLM JSON output issues.
    """
    # Strategy 0: Direct parsing
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Strategy 1: Basic cleanup
    cleaned = text.strip('\ufeff\u200b\u200c\u200d\x00\x01\x02')
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    
    # Strategy 2: Remove trailing commas
    fixed = re.sub(r',(\s*[}\]])', r'\1', cleaned)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    
    # Strategy 3: Fix single quotes used as delimiters
    # Replace single quotes around keys with double quotes
    fixed = re.sub(r"(?<=[{\s,])'([^']+)'(?=\s*:)", r'"\1"', fixed)
    # Replace single quotes around string values with double quotes
    fixed = re.sub(r":\s*'([^']*?)'(?=\s*[,}\]])", r': "\1"', fixed)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    
    # Strategy 4: Handle escaped characters
    fixed = fixed.replace('\\n', '\n').replace('\\t', '\t').replace('\\"', '"')
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    
    # Strategy 5: Extract just the object if there's extra text
    # Find the first { and last } that form a balanced pair
    try:
        start = fixed.find('{')
        if start != -1:
            brace_count = 0
            for i, char in enumerate(fixed[start:], start):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        candidate = fixed[start:i+1]
                        return json.loads(candidate)
    except (json.JSONDecodeError, ValueError):
        pass
    
    return None


def _extract_json_with_retry(text: str, max_retries: int = 3) -> list[dict] | None:
    """Extract JSON with multiple fallback strategies.
    
    Tries multiple extraction methods:
    1. Standard <json>...</json> blocks
    2. JSON code blocks ```json...```
    3. Raw JSON objects in text (with nested object support)
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
        parsed = _try_parse_json_with_repair(match.strip())
        if parsed is not None:
            results.append(parsed)
    
    # Method 3: Raw JSON objects with balanced brace matching
    for match in _find_json_objects(text):
        parsed = _try_parse_json_with_repair(match.strip())
        if parsed is not None:
            results.append(parsed)
    
    return results or None


def _find_json_objects(s: str) -> list[str]:
    """Find JSON objects using brace counting for proper nesting.
    
    This handles nested objects better than simple regex.
    """
    objects = []
    i = 0
    while i < len(s):
        if s[i] == '{':
            start = i
            brace_count = 1
            i += 1
            in_string = False
            escape_next = False
            
            while i < len(s) and brace_count > 0:
                char = s[i]
                if escape_next:
                    escape_next = False
                elif char == '\\':
                    escape_next = True
                elif char == '"' and not escape_next:
                    in_string = not in_string
                elif not in_string:
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                i += 1
            
            if brace_count == 0:
                candidate = s[start:i]
                # Only keep if it looks like a JSON object (has quoted keys)
                if re.search(r'"[^"]+"\s*:', candidate):
                    objects.append(candidate)
        else:
            i += 1
    return objects


class TaskAgent:
    """Task agent that solves IMO grading problems with enhanced reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.max_retries = 3
        self._call_count = 0

    def _build_grading_prompt(self, inputs: dict) -> str:
        """Build a comprehensive prompt for the grading task."""
        domain = inputs.get("domain", "")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        # Truncate very long inputs to prevent token overflow
        max_len = 8000
        problem = problem[:max_len] if len(problem) > max_len else problem
        solution = solution[:max_len] if len(solution) > max_len else solution
        student_answer = student_answer[:max_len] if len(student_answer) > max_len else student_answer
        
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
- Ensure your JSON is valid and properly formatted"""

    def _try_extract_prediction(self, text: str) -> tuple[str, str | None]:
        """Try to extract prediction from response text.
        
        Returns:
            (prediction, reasoning)
        """
        try:
            extracted = _extract_json_with_retry(text)
            if extracted:
                last = extracted[-1]
                prediction = last.get("response", "None")
                reasoning = last.get("reasoning")
                
                # Validate prediction is not empty or whitespace
                if prediction is not None:
                    pred_str = str(prediction).strip()
                    if pred_str and pred_str.lower() not in ("none", "null", ""):
                        return pred_str, reasoning
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
        
        return "None", None

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with retry logic.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        self._call_count += 1
        instruction = self._build_grading_prompt(inputs)
        msg_history = []
        prediction = "None"
        last_error = None
        
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
                        self.log_fn(f"Call {self._call_count}, attempt {attempt + 1}: Grading reasoning: {reasoning[:200]}...")
                    break
                
                # If extraction failed, add a follow-up message asking for proper format
                if attempt < self.max_retries - 1:
                    last_error = "Failed to extract valid JSON prediction"
                    instruction = (
                        "Your previous response did not contain valid JSON in the required format. "
                        "Please respond ONLY with a JSON object wrapped in <json>...</json> tags, "
                        "containing 'reasoning' and 'response' fields."
                    )
                    
            except Exception as e:
                last_error = str(e)
                self.log_fn(f"Call {self._call_count}, attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    break
        
        if prediction == "None" and last_error:
            self.log_fn(f"Call {self._call_count}: All retries exhausted. Last error: {last_error}")
        
        return str(prediction), msg_history
