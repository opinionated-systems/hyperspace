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
    3. Raw JSON objects at start/end of text
    4. JSON-like structures with relaxed parsing
    5. Truncated or malformed JSON repair attempts
    """
    results = []
    
    # Strategy 1: <json> tags (original)
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
            # Try to repair common issues
            repaired = _repair_json(inner)
            if repaired:
                results.append(repaired)
            continue
    
    # Strategy 2: ```json code blocks
    if not results:
        pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                results.append(json.loads(match.strip()))
            except json.JSONDecodeError:
                # Try to repair common issues
                repaired = _repair_json(match.strip())
                if repaired:
                    results.append(repaired)
                continue
    
    # Strategy 3: Look for JSON objects directly using brace matching
    if not results:
        # Find all potential JSON starting points
        for start in range(len(text)):
            if text[start] == '{':
                try:
                    # Use brace counting to find the matching end
                    brace_count = 0
                    in_string = False
                    escape_next = False
                    end = start
                    
                    for i, char in enumerate(text[start:]):
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
                                brace_count += 1
                            elif char == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    end = start + i + 1
                                    break
                    
                    if end > start and brace_count == 0:
                        obj_str = text[start:end]
                        try:
                            results.append(json.loads(obj_str))
                        except json.JSONDecodeError:
                            # Try to repair
                            repaired = _repair_json(obj_str)
                            if repaired:
                                results.append(repaired)
                except Exception:
                    continue
    
    return results or None


def _repair_json(text: str) -> dict | None:
    """Attempt to repair common JSON formatting issues.
    
    Handles:
    - Truncated JSON (missing closing braces)
    - Unescaped newlines in strings
    - Trailing commas
    - Single quotes instead of double quotes
    - Missing quotes around keys
    - Escaped quotes issues
    - Unicode escape sequences
    """
    try:
        # First, try the original
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Try common repairs
    repaired = text.strip()
    
    # 1. Add missing closing braces
    open_braces = repaired.count('{') - repaired.count('}')
    if open_braces > 0:
        repaired += '}' * open_braces
    
    # 2. Remove trailing commas before closing braces/brackets
    repaired = re.sub(r',\s*}', '}', repaired)
    repaired = re.sub(r',\s*]', ']', repaired)
    
    # 3. Try to fix single quotes to double quotes (carefully)
    # Only replace single quotes that appear to be used as JSON string delimiters
    # This is a heuristic: replace 'key' with "key" and 'value' with "value"
    try:
        # Replace single-quoted strings with double-quoted ones
        # Pattern: '...' where ... doesn't contain "
        repaired = re.sub(r"'([^']*?)'", r'"\1"', repaired)
    except Exception:
        pass
    
    # 4. Try to fix unescaped newlines in strings
    # Replace literal newlines within string values with \n
    try:
        # This is a complex repair - look for newlines between quotes
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
            if in_string and char == '\n':
                result_chars.append('\\n')
            elif in_string and char == '\t':
                result_chars.append('\\t')
            else:
                result_chars.append(char)
        repaired = ''.join(result_chars)
    except Exception:
        pass
    
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass
    
    # 5. Try extracting just the object structure without content validation
    # Look for key-value pairs with more flexible patterns
    try:
        # Extract all "key": value or 'key': value patterns
        pattern = r'["\']([^"\']+)["\']\s*:\s*(["\'][^"\']*["\']|\[[^\]]*\]|\{[^}]*\}|[^,\s\}]+)'
        matches = re.findall(pattern, repaired)
        if matches:
            result = {}
            for key, val in matches:
                try:
                    # Try to parse the value
                    val = val.strip()
                    if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                        result[key] = val[1:-1]
                    elif val == 'true':
                        result[key] = True
                    elif val == 'false':
                        result[key] = False
                    elif val == 'null':
                        result[key] = None
                    elif val.startswith('[') or val.startswith('{'):
                        try:
                            result[key] = json.loads(val)
                        except json.JSONDecodeError:
                            result[key] = val
                    else:
                        try:
                            result[key] = int(val)
                        except ValueError:
                            try:
                                result[key] = float(val)
                            except ValueError:
                                result[key] = val
                except Exception:
                    result[key] = val
            if result:
                return result
    except Exception:
        pass
    
    # 6. Last resort: try to extract any valid JSON-like structure
    try:
        # Look for the first { and last }
        start = repaired.find('{')
        end = repaired.rfind('}')
        if start != -1 and end != -1 and end > start:
            candidate = repaired[start:end+1]
            return json.loads(candidate)
    except json.JSONDecodeError:
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

1. **Understand the Problem**: What is being asked? What are the key concepts?

2. **Analyze the Official Solution**: What is the correct approach? What is the final answer?

3. **Review the Student's Answer**: What approach did the student take? What is their final answer?

4. **Compare**: Does the student's answer match the official solution? Consider:
   - Is the final answer numerically/algebraically equivalent?
   - Did the student show sufficient reasoning?
   - Are there any errors in the student's work?

5. **Conclusion**: Based on your analysis, provide your evaluation.

Respond in JSON format with the following schema:
<json>
{{
    "reasoning": "Your detailed step-by-step analysis here",
    "response": "Your final answer/evaluation here"
}}
</json>

The "response" field should contain your final determination (e.g., the correct answer, a score, or an evaluation of correctness)."""
        
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
        try:
            extracted = _extract_json_flexible(last_text)
            if extracted:
                # Prefer response field, fallback to other common fields
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
                        # Handle nested structures
                        if isinstance(value, dict):
                            # Try to extract from nested dict
                            for sub_key in ["value", "result", "answer", "text", "score"]:
                                if sub_key in value:
                                    result = str(value[sub_key])
                                    self.log_fn(f"Extracted nested value from '{key}.{sub_key}': {result[:100]}...")
                                    return result
                            result = str(value)
                            self.log_fn(f"Extracted dict from '{key}': {result[:100]}...")
                            return result
                        elif isinstance(value, list) and value:
                            # If it's a list, return the first meaningful element or the whole list
                            result = str(value[0]) if len(value) == 1 else str(value)
                            self.log_fn(f"Extracted list from '{key}': {result[:100]}...")
                            return result
                        elif isinstance(value, bool):
                            # Convert boolean to meaningful string
                            result = "Correct" if value else "Incorrect"
                            self.log_fn(f"Extracted boolean from '{key}': {result}")
                            return result
                        elif isinstance(value, (int, float)):
                            result = str(value)
                            self.log_fn(f"Extracted number from '{key}': {result}")
                            return result
                        else:
                            result = str(value)
                            self.log_fn(f"Extracted string from '{key}': {result[:100]}...")
                            return result
                
                # If no known field, return the whole object as string
                self.log_fn(f"No priority keys found, returning full object: {str(last_obj)[:100]}...")
                return str(last_obj)
        except Exception as e:
            self.log_fn(f"Flexible extraction failed: {e}")
        
        # Fallback 1: Look for patterns like "response": "..." or 'response': '...'
        try:
            patterns = [
                (r'"response"\s*:\s*"([^"]+)"', "response"),
                (r"'response'\s*:\s*'([^']+)'", "response"),
                (r'"answer"\s*:\s*"([^"]+)"', "answer"),
                (r"'answer'\s*:\s*'([^']+)'", "answer"),
                (r'"evaluation"\s*:\s*"([^"]+)"', "evaluation"),
                (r'"result"\s*:\s*"([^"]+)"', "result"),
                (r'"grade"\s*:\s*"([^"]+)"', "grade"),
                (r'"score"\s*:\s*(\d+)', "score"),
                (r'"correct"\s*:\s*(true|false)', "correct"),
            ]
            for pattern, key in patterns:
                match = re.search(pattern, last_text, re.IGNORECASE)
                if match:
                    result = match.group(1)
                    self.log_fn(f"Extracted via regex pattern '{key}': {result[:100]}...")
                    return result
        except Exception as e:
            self.log_fn(f"Regex extraction failed: {e}")
        
        # Fallback 2: Try to extract any meaningful text after common markers
        try:
            markers = [
                (r'(?:final|conclusion|verdict|answer|result)[:\s]+(.+?)(?:\n|$)', "conclusion marker"),
                (r'(?:therefore|thus|so)[:\s]+(.+?)(?:\n|$)', "reasoning marker"),
                (r'(?:the student answer is|student answer)[:\s]+(.+?)(?:\n|$)', "student answer marker"),
                (r'(?:score|grade|mark)[:\s]+(\d+(?:/\d+)?)', "score marker"),
                (r'(?:evaluation|assessment)[:\s]+(.+?)(?:\n|$)', "evaluation marker"),
            ]
            for pattern, desc in markers:
                match = re.search(pattern, last_text, re.IGNORECASE)
                if match:
                    result = match.group(1).strip()
                    self.log_fn(f"Extracted via {desc}: {result[:100]}...")
                    return result
        except Exception as e:
            self.log_fn(f"Marker extraction failed: {e}")
        
        # Fallback 3: Look for IMO-specific patterns (scores like 7/7, 0/7, etc.)
        try:
            imo_patterns = [
                (r'\b(\d+/\d+)\b', "fraction score"),
                (r'\b(full score|partial score|no score|zero|full marks|no marks)\b', "text score"),
                (r'\b(correct|incorrect|right|wrong|valid|invalid)\b', "correctness"),
            ]
            for pattern, desc in imo_patterns:
                match = re.search(pattern, last_text, re.IGNORECASE)
                if match:
                    result = match.group(1)
                    self.log_fn(f"Extracted IMO {desc}: {result}")
                    return result
        except Exception as e:
            self.log_fn(f"IMO pattern extraction failed: {e}")
        
        # Fallback 4: If the text looks like a direct answer (short, no JSON), use it
        stripped = last_text.strip()
        if len(stripped) < 200 and '"' not in stripped and '{' not in stripped:
            self.log_fn(f"Using direct text answer: {stripped[:100]}...")
            return stripped
        
        self.log_fn("Warning: Could not extract valid prediction from response")
        # Log a snippet of the raw response for debugging
        snippet = last_text[:500].replace('\n', ' ')
        self.log_fn(f"Raw response snippet: {snippet}...")
        return "None"

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
