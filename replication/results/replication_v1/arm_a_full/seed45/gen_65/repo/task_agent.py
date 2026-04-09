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


def _extract_json_flexible(text: str) -> list[dict] | None:
    """Extract JSON objects with multiple fallback strategies.
    
    Tries multiple patterns in order of reliability:
    1. <json>...</json> tags
    2. ```json code blocks
    3. Raw JSON objects using efficient brace matching
    4. Truncated or malformed JSON repair attempts
    """
    results = []
    
    # Strategy 1: <json> tags (original) - most reliable
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
            repaired = _repair_json(inner)
            if repaired:
                results.append(repaired)
    
    if results:
        return results
    
    # Strategy 2: ```json code blocks
    pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    matches = re.findall(pattern, text, re.DOTALL)
    for match in matches:
        try:
            results.append(json.loads(match.strip()))
        except json.JSONDecodeError:
            repaired = _repair_json(match.strip())
            if repaired:
                results.append(repaired)
    
    if results:
        return results
    
    # Strategy 3: Efficient brace matching - find all JSON objects
    # Use a stack-based approach for better performance
    i = 0
    while i < len(text):
        # Find next opening brace
        try:
            i = text.index('{', i)
        except ValueError:
            break
        
        # Try to find matching closing brace using stack
        start = i
        brace_depth = 0
        in_string = False
        escape_next = False
        
        for j in range(i, len(text)):
            char = text[j]
            
            if escape_next:
                escape_next = False
                continue
            
            if char == '\\':
                escape_next = True
                continue
            
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
            
            if not in_string:
                if char == '{':
                    brace_depth += 1
                elif char == '}':
                    brace_depth -= 1
                    if brace_depth == 0:
                        # Found complete JSON object
                        obj_str = text[start:j+1]
                        try:
                            results.append(json.loads(obj_str))
                        except json.JSONDecodeError:
                            repaired = _repair_json(obj_str)
                            if repaired:
                                results.append(repaired)
                        i = j + 1
                        break
        else:
            # No matching brace found, try to repair truncated JSON
            if brace_depth > 0:
                obj_str = text[start:]
                repaired = _repair_json(obj_str)
                if repaired:
                    results.append(repaired)
            i += 1
    
    return results or None


def _repair_json(text: str) -> dict | None:
    """Attempt to repair common JSON formatting issues.
    
    Handles:
    - Truncated JSON (missing closing braces)
    - Unescaped newlines/tabs in strings
    - Trailing commas
    - Single quotes instead of double quotes
    - Missing quotes around keys
    """
    original = text.strip()
    
    # Quick check: if it's already valid, return it
    try:
        return json.loads(original)
    except json.JSONDecodeError:
        pass
    
    repaired = original
    
    # 1. Add missing closing braces/brackets
    open_braces = repaired.count('{') - repaired.count('}')
    open_brackets = repaired.count('[') - repaired.count(']')
    if open_braces > 0:
        repaired += '}' * open_braces
    if open_brackets > 0:
        repaired += ']' * open_brackets
    
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass
    
    # 2. Remove trailing commas before closing braces/brackets
    repaired = re.sub(r',\s*}', '}', repaired)
    repaired = re.sub(r',\s*]', ']', repaired)
    
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass
    
    # 3. Fix single quotes to double quotes (simple cases only)
    # Use a safer approach: only replace quotes that are clearly delimiters
    repaired = re.sub(r"(?<=[{,\s:])'([^']*?)'(?=\s*[:,}])", r'"\1"', repaired)
    
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass
    
    # 4. Fix unescaped newlines and tabs in strings
    in_string = False
    escape_next = False
    result_chars = []
    for char in repaired:
        if escape_next:
            result_chars.append(char)
            escape_next = False
            continue
        if char == '\\':
            result_chars.append(char)
            escape_next = True
            continue
        if char == '"' and not escape_next:
            in_string = not in_string
            result_chars.append(char)
            continue
        if in_string:
            if char == '\n':
                result_chars.append('\\n')
            elif char == '\t':
                result_chars.append('\\t')
            elif char == '\r':
                result_chars.append('\\r')
            else:
                result_chars.append(char)
        else:
            result_chars.append(char)
    repaired = ''.join(result_chars)
    
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass
    
    # 5. Extract key-value pairs as last resort
    try:
        # Simple pattern for "key": value pairs
        pattern = r'"([^"]+)"\s*:\s*("(?:[^"\\]|\\.)*"|\[[^\]]*\]|\{[^}]*\}|[^,\s\}]+)'
        matches = re.findall(pattern, repaired)
        if matches:
            result = {}
            for key, val in matches:
                val = val.strip()
                if val.startswith('"') and val.endswith('"'):
                    # Unescape string value
                    try:
                        result[key] = json.loads(val)
                    except json.JSONDecodeError:
                        result[key] = val[1:-1]
                elif val == 'true':
                    result[key] = True
                elif val == 'false':
                    result[key] = False
                elif val == 'null':
                    result[key] = None
                elif val.startswith('['):
                    try:
                        result[key] = json.loads(val)
                    except json.JSONDecodeError:
                        result[key] = val
                elif val.startswith('{'):
                    try:
                        result[key] = json.loads(val)
                    except json.JSONDecodeError:
                        result[key] = val
                else:
                    # Try numeric
                    try:
                        if '.' in val:
                            result[key] = float(val)
                        else:
                            result[key] = int(val)
                    except ValueError:
                        result[key] = val
            if result:
                return result
    except Exception:
        pass
    
    return None


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def _build_prompt(self, inputs: dict) -> str:
        """Build a structured prompt for IMO grading with chain-of-thought reasoning."""
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        prompt = f"""You are an expert {domain} grader evaluating student solutions to competition problems.

Your task is to carefully analyze the student's answer and determine if it is correct according to the official solution and grading guidelines.

## Problem Statement
{problem}

## Official Solution
{solution}

## Grading Guidelines
{grading_guidelines}

## Student's Answer
{student_answer}

## Instructions

Think through this step-by-step:

1. **Understand the Problem**: What is being asked? What are the key concepts and constraints?

2. **Analyze the Official Solution**: What is the correct approach? What is the final answer? What are the critical steps?

3. **Review the Student's Answer**: What approach did the student take? What is their final answer? Did they show their work?

4. **Compare**: Does the student's answer match the official solution? Consider:
   - Is the final answer numerically/algebraically equivalent to the official answer?
   - Did the student show sufficient reasoning and justification?
   - Are there any errors, gaps, or incorrect assumptions in the student's work?
   - Does the student answer satisfy all problem constraints?

5. **Conclusion**: Based on your analysis, provide your evaluation.

Respond in JSON format with the following schema:
<json>
{{
    "reasoning": "Your detailed step-by-step analysis here. Explain your thought process clearly.",
    "response": "Your final answer/evaluation here. Be specific: state the correct answer or provide a clear correctness determination."
}}
</json>

The "response" field should contain your final determination. For IMO problems, this typically includes:
- The correct numerical answer (e.g., "42", "7/7", "0")
- Or a clear correctness evaluation (e.g., "Correct", "Incorrect", "Partial credit: 3/7")
- Or the official solution's answer if the student is wrong"""
        
        return prompt

    def _extract_prediction(self, msg_history: list[dict]) -> str:
        """Extract prediction from message history with multiple fallback strategies.
        
        Enhanced extraction that handles:
        - JSON with <json> tags
        - JSON code blocks
        - Direct JSON objects
        - Plain text responses
        - Nested JSON structures
        - IMO-specific answer formats (scores, correct/incorrect, etc.)
        """
        if not msg_history:
            self.log_fn("Warning: Empty message history")
            return "None"
        
        last_text = msg_history[-1].get("text", "")
        if not last_text or not last_text.strip():
            self.log_fn("Warning: Empty text in last message")
            return "None"
        
        # Try flexible extraction first
        extracted = _extract_json_flexible(last_text)
        if extracted:
            last_obj = extracted[-1]
            self.log_fn(f"Successfully extracted JSON with keys: {list(last_obj.keys())}")
            
            # Priority order for extraction
            priority_keys = [
                "response", "answer", "evaluation", "result", 
                "grade", "verdict", "decision", "score", 
                "correct", "is_correct", "mark", "points"
            ]
            
            for key in priority_keys:
                if key in last_obj:
                    value = last_obj[key]
                    result = self._format_value(value, key)
                    if result:
                        return result
            
            # If no known field, return the whole object as string
            self.log_fn(f"No priority keys found, returning full object: {str(last_obj)[:100]}...")
            return str(last_obj)
        
        # Fallback: Try regex patterns for common JSON-like structures
        result = self._extract_via_regex(last_text)
        if result:
            return result
        
        # Last resort: If the text looks like a direct answer (short, no JSON), use it
        stripped = last_text.strip()
        if len(stripped) < 200 and '"' not in stripped and '{' not in stripped:
            self.log_fn(f"Using direct text answer: {stripped[:100]}...")
            return stripped
        
        self.log_fn("Warning: Could not extract valid prediction from response")
        snippet = last_text[:500].replace('\n', ' ')
        self.log_fn(f"Raw response snippet: {snippet}...")
        return "None"
    
    def _format_value(self, value: Any, key: str) -> str | None:
        """Format a value from JSON extraction for return."""
        # Handle nested structures
        if isinstance(value, dict):
            for sub_key in ["value", "result", "answer", "text", "score"]:
                if sub_key in value:
                    result = str(value[sub_key])
                    self.log_fn(f"Extracted nested value from '{key}.{sub_key}': {result[:100]}...")
                    return result
            result = str(value)
            self.log_fn(f"Extracted dict from '{key}': {result[:100]}...")
            return result
        
        if isinstance(value, list) and value:
            result = str(value[0]) if len(value) == 1 else str(value)
            self.log_fn(f"Extracted list from '{key}': {result[:100]}...")
            return result
        
        if isinstance(value, bool):
            result = "Correct" if value else "Incorrect"
            self.log_fn(f"Extracted boolean from '{key}': {result}")
            return result
        
        if isinstance(value, (int, float)):
            result = str(value)
            self.log_fn(f"Extracted number from '{key}': {result}")
            return result
        
        result = str(value)
        self.log_fn(f"Extracted string from '{key}': {result[:100]}...")
        return result
    
    def _extract_via_regex(self, text: str) -> str | None:
        """Extract prediction using regex patterns as fallback."""
        # Pattern groups: (regex, description, group_index)
        patterns = [
            # JSON field patterns
            (r'"response"\s*:\s*"([^"]+)"', "response field", 1),
            (r'"answer"\s*:\s*"([^"]+)"', "answer field", 1),
            (r'"evaluation"\s*:\s*"([^"]+)"', "evaluation field", 1),
            (r'"result"\s*:\s*"([^"]+)"', "result field", 1),
            (r'"grade"\s*:\s*"([^"]+)"', "grade field", 1),
            (r'"score"\s*:\s*(\d+)', "score field", 1),
            (r'"correct"\s*:\s*(true|false)', "correct field", 1),
            # Conclusion markers
            (r'(?:final|conclusion|verdict|answer|result)[:\s]+(.+?)(?:\n|$)', "conclusion marker", 1),
            (r'(?:therefore|thus|so)[:\s]+(.+?)(?:\n|$)', "reasoning marker", 1),
            (r'(?:the student answer is|student answer)[:\s]+(.+?)(?:\n|$)', "student answer marker", 1),
            (r'(?:score|grade|mark)[:\s]+(\d+(?:/\d+)?)', "score marker", 1),
            (r'(?:evaluation|assessment)[:\s]+(.+?)(?:\n|$)', "evaluation marker", 1),
            # IMO-specific patterns
            (r'\b(\d+/\d+)\b', "fraction score", 1),
            (r'\b(full score|partial score|no score|zero|full marks|no marks)\b', "text score", 1),
            (r'\b(correct|incorrect|right|wrong|valid|invalid)\b', "correctness", 1),
        ]
        
        for pattern, desc, group in patterns:
            try:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    result = match.group(group).strip()
                    self.log_fn(f"Extracted via {desc}: {result[:100]}...")
                    return result
            except Exception as e:
                self.log_fn(f"Regex pattern '{desc}' failed: {e}")
                continue
        
        return None

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        instruction = self._build_prompt(inputs)

        self.log_fn(f"Processing problem in domain: {inputs.get('domain', 'Unknown')}")
        
        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON
        prediction = self._extract_prediction(msg_history)
        
        self.log_fn(f"Final prediction: {prediction[:100]}..." if len(str(prediction)) > 100 else f"Final prediction: {prediction}")

        return str(prediction), msg_history
