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
    
    # 3. Try to fix unescaped newlines in strings (simple heuristic)
    # This is risky but sometimes necessary for truncated outputs
    
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass
    
    # 4. Try extracting just the object structure without content validation
    # Look for key-value pairs
    try:
        # Extract all "key": value patterns
        pattern = r'"([^"]+)"\s*:\s*("[^"]*"|\[[^\]]*\]|\{[^}]*\}|[^,\s\}]+)'
        matches = re.findall(pattern, repaired)
        if matches:
            result = {}
            for key, val in matches:
                try:
                    # Try to parse the value
                    if val.startswith('"') and val.endswith('"'):
                        result[key] = val[1:-1]
                    elif val == 'true':
                        result[key] = True
                    elif val == 'false':
                        result[key] = False
                    elif val == 'null':
                        result[key] = None
                    elif val.startswith('[') or val.startswith('{'):
                        result[key] = json.loads(val)
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
        - Empty or malformed responses
        """
        if not msg_history:
            self.log_fn("Warning: Empty message history, cannot extract prediction")
            return "None"
        
        last_text = msg_history[-1].get("text", "")
        if not last_text or not last_text.strip():
            self.log_fn("Warning: Last message has no text content")
            return "None"
        
        # Log the raw response length for debugging
        self.log_fn(f"Extracting prediction from response of {len(last_text)} characters")
        
        # Try flexible extraction first
        try:
            extracted = _extract_json_flexible(last_text)
            if extracted:
                # Prefer response field, fallback to other common fields
                last_obj = extracted[-1]
                self.log_fn(f"Successfully extracted JSON with keys: {list(last_obj.keys())}")
                
                for key in ["response", "answer", "evaluation", "result", "grade", "verdict", "decision"]:
                    if key in last_obj:
                        value = last_obj[key]
                        self.log_fn(f"Found prediction in key '{key}'")
                        # Handle nested structures
                        if isinstance(value, dict):
                            # Try to extract from nested dict
                            for sub_key in ["value", "result", "answer", "text"]:
                                if sub_key in value:
                                    return str(value[sub_key])
                            return str(value)
                        elif isinstance(value, list) and value:
                            # If it's a list, return the first meaningful element or the whole list
                            return str(value[0]) if len(value) == 1 else str(value)
                        else:
                            return str(value)
                # If no known field, return the whole object as string
                self.log_fn(f"Warning: No recognized prediction field found in JSON. Keys: {list(last_obj.keys())}")
                return str(last_obj)
            else:
                self.log_fn("No JSON objects found in response, trying fallback extraction")
        except Exception as e:
            self.log_fn(f"Flexible extraction failed: {type(e).__name__}: {e}")
        
        # Fallback 1: Look for patterns like "response": "..." or 'response': '...'
        try:
            patterns = [
                r'"response"\s*:\s*"([^"]+)"',
                r"'response'\s*:\s*'([^']+)'",
                r'"answer"\s*:\s*"([^"]+)"',
                r"'answer'\s*:\s*'([^']+)'",
            ]
            for pattern in patterns:
                match = re.search(pattern, last_text)
                if match:
                    return match.group(1)
        except Exception:
            pass
        
        # Fallback 2: Try to extract any meaningful text after common markers
        try:
            markers = [
                r'(?:final|conclusion|verdict|answer|result)[:\s]+(.+?)(?:\n|$)',
                r'(?:therefore|thus|so)[:\s]+(.+?)(?:\n|$)',
            ]
            for pattern in markers:
                match = re.search(pattern, last_text, re.IGNORECASE)
                if match:
                    return match.group(1).strip()
        except Exception:
            pass
        
        # Fallback 3: If the text looks like a direct answer (short, no JSON), use it
        stripped = last_text.strip()
        if len(stripped) < 200 and '"' not in stripped and '{' not in stripped:
            return stripped
        
        return "None"

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Log problem domain for tracking
        domain = inputs.get("domain", "Unknown")
        problem_preview = inputs.get("problem", "")[:100]
        self.log_fn(f"Processing {domain} problem: {problem_preview}...")
        
        instruction = self._build_prompt(inputs)

        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
            
            # Log usage stats if available
            usage = info.get("usage", {})
            if usage:
                self.log_fn(f"LLM usage - Prompt tokens: {usage.get('prompt_tokens', 'N/A')}, "
                           f"Completion tokens: {usage.get('completion_tokens', 'N/A')}")
        except Exception as e:
            self.log_fn(f"Error calling LLM: {type(e).__name__}: {e}")
            return "Error: LLM call failed", []

        # Extract prediction from JSON
        prediction = self._extract_prediction(msg_history)
        
        if prediction == "None":
            self.log_fn(f"Warning: Could not extract valid prediction from response")
            # Log the raw response for debugging
            if msg_history:
                raw_text = msg_history[-1].get("text", "")
                self.log_fn(f"Raw response preview: {raw_text[:500]}...")
                self.log_fn(f"Raw response length: {len(raw_text)} characters")
        else:
            self.log_fn(f"Successfully extracted prediction: {prediction[:200]}...")

        return str(prediction), msg_history
