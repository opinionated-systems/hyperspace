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
    6. JSON with comments (// and /* */ style)
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
    
    # Strategy 4: Try to extract JSON with comments removed
    if not results:
        text_no_comments = _remove_json_comments(text)
        if text_no_comments != text:
            # Try all previous strategies on the cleaned text
            try:
                # Try direct JSON parsing first
                start = text_no_comments.find('{')
                end = text_no_comments.rfind('}')
                if start != -1 and end != -1 and end > start:
                    cleaned_obj = json.loads(text_no_comments[start:end+1])
                    results.append(cleaned_obj)
            except json.JSONDecodeError:
                # Try brace matching on cleaned text
                for start in range(len(text_no_comments)):
                    if text_no_comments[start] == '{':
                        try:
                            brace_count = 0
                            in_string = False
                            escape_next = False
                            end = start
                            
                            for i, char in enumerate(text_no_comments[start:]):
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
                                obj_str = text_no_comments[start:end]
                                try:
                                    results.append(json.loads(obj_str))
                                except json.JSONDecodeError:
                                    repaired = _repair_json(obj_str)
                                    if repaired:
                                        results.append(repaired)
                        except Exception:
                            continue
    
    return results or None


def _remove_json_comments(text: str) -> str:
    """Remove C-style comments from text (both // and /* */ styles).
    
    This is useful for handling JSON that LLMs sometimes output with comments.
    """
    result = []
    i = 0
    while i < len(text):
        # Check for // style comments
        if i < len(text) - 1 and text[i:i+2] == '//':
            # Skip until end of line
            while i < len(text) and text[i] != '\n':
                i += 1
            continue
        
        # Check for /* */ style comments
        if i < len(text) - 1 and text[i:i+2] == '/*':
            # Skip until */
            i += 2
            while i < len(text) - 1:
                if text[i:i+2] == '*/':
                    i += 2
                    break
                i += 1
            continue
        
        result.append(text[i])
        i += 1
    
    return ''.join(result)


def _repair_json(text: str) -> dict | None:
    """Attempt to repair common JSON formatting issues.
    
    Handles:
    - Truncated JSON (missing closing braces)
    - Unescaped newlines in strings
    - Trailing commas
    - Single quotes instead of double quotes
    - Missing quotes around keys
    - Unicode escape issues
    - Incomplete/truncated responses
    """
    try:
        # First, try the original
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Try common repairs
    repaired = text.strip()
    
    # 1. Add missing closing braces (handle nested structures)
    open_braces = repaired.count('{') - repaired.count('}')
    open_brackets = repaired.count('[') - repaired.count(']')
    if open_braces > 0:
        repaired += '}' * open_braces
    if open_brackets > 0:
        repaired += ']' * open_brackets
    
    # 2. Remove trailing commas before closing braces/brackets
    repaired = re.sub(r',\s*}', '}', repaired)
    repaired = re.sub(r',\s*]', ']', repaired)
    
    # 3. Replace single quotes with double quotes (carefully)
    # Only replace quotes that appear to be delimiters, not apostrophes in words
    repaired = re.sub(r"(?<=[\{\,\[])\s*'([^']+)'\s*:", r'"\1":', repaired)  # keys
    repaired = re.sub(r":\s*'([^']*)'(?=\s*[\,\}\]])", r':"\1"', repaired)  # values
    
    # 4. Try to fix unescaped newlines in strings (simple heuristic)
    # Replace newlines within string values with escaped newlines
    def escape_newlines_in_strings(match):
        content = match.group(1)
        # Escape unescaped newlines
        content = re.sub(r'(?<!\\)\n', r'\\n', content)
        content = re.sub(r'(?<!\\)\r', r'\\r', content)
        return '"' + content + '"'
    
    repaired = re.sub(r'"([^"]*(?:\.[^"]*)*)"', escape_newlines_in_strings, repaired)
    
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass
    
    # 5. Try extracting just the object structure without content validation
    # Look for key-value pairs with improved pattern
    try:
        # Extract all "key": value patterns, handling nested structures better
        pattern = r'"([^"]+)"\s*:\s*("(?:[^"\\]|\\.)*"|\[[^\]]*\]|\{[^}]*\}|[^,\s\}\]]+)'
        matches = re.findall(pattern, repaired)
        if matches:
            result = {}
            for key, val in matches:
                try:
                    # Try to parse the value
                    val = val.strip()
                    if val.startswith('"') and val.endswith('"'):
                        # String value - unescape it
                        result[key] = val[1:-1].replace('\\n', '\n').replace('\\t', '\t').replace('\\"', '"')
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
                            result[key] = val  # Keep as string if can't parse
                    else:
                        # Try numeric
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
        # Find the first { and last }
        start = repaired.find('{')
        end = repaired.rfind('}')
        if start != -1 and end != -1 and end > start:
            subset = repaired[start:end+1]
            return json.loads(subset)
    except json.JSONDecodeError:
        pass
    
    # 7. NEW: Handle incomplete/truncated JSON by extracting partial objects
    # This handles cases where the LLM output was cut off mid-response
    try:
        # Look for the start of a JSON object
        start = repaired.find('{')
        if start != -1:
            # Try to find a complete key-value pair
            partial = repaired[start:]
            # Extract any complete key-value pairs we can find
            kv_pattern = r'"([^"]+)"\s*:\s*"([^"]*)"'
            kv_matches = re.findall(kv_pattern, partial)
            if kv_matches:
                result = {k: v for k, v in kv_matches}
                if result:
                    return result
            # Try with single quotes
            kv_pattern2 = r"'([^']+)'\s*:\s*'([^']*)'"
            kv_matches2 = re.findall(kv_pattern2, partial)
            if kv_matches2:
                result = {k: v for k, v in kv_matches2}
                if result:
                    return result
    except Exception:
        pass
    
    # 8. NEW: Handle JSON with unclosed strings (common with truncated responses)
    try:
        # Find and close unclosed string values
        # Pattern: "key": "value... (missing closing quote)
        unclosed_pattern = r'"([^"]+)"\s*:\s*"([^"]*)$'
        match = re.search(unclosed_pattern, repaired)
        if match:
            key = match.group(1)
            value = match.group(2)
            # Close the string and the object
            fixed = repaired[:match.end()] + '"}'
            try:
                return json.loads(fixed)
            except json.JSONDecodeError:
                # Return what we have as a partial result
                return {key: value}
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
        - IMO-specific grading outputs (Correct/Incorrect/Partial)
        """
        if not msg_history:
            self.log_fn("No message history available for extraction")
            return "None"
        
        last_text = msg_history[-1].get("text", "")
        if not last_text or not last_text.strip():
            self.log_fn("Empty response in message history")
            return "None"
        
        self.log_fn(f"Extracting prediction from {len(last_text)} characters")
        
        # Try flexible extraction first
        try:
            extracted = _extract_json_flexible(last_text)
            if extracted:
                self.log_fn(f"Successfully extracted {len(extracted)} JSON object(s)")
                # Prefer response field, fallback to other common fields
                last_obj = extracted[-1]
                
                # IMO-specific: Check for grading-specific fields first
                imo_fields = ["grade", "score", "evaluation", "verdict", "correctness", "mark"]
                standard_fields = ["response", "answer", "result", "decision"]
                all_fields = imo_fields + standard_fields
                
                for key in all_fields:
                    if key in last_obj:
                        value = last_obj[key]
                        self.log_fn(f"Found field '{key}' with value type: {type(value).__name__}")
                        # Handle nested structures
                        if isinstance(value, dict):
                            # Try to extract from nested dict
                            for sub_key in ["value", "result", "answer", "text", "grade", "score"]:
                                if sub_key in value:
                                    result = str(value[sub_key])
                                    self.log_fn(f"Extracted nested value from '{key}.{sub_key}': {result[:100]}...")
                                    return result
                            result = str(value)
                            self.log_fn(f"Returning nested dict as string: {result[:100]}...")
                            return result
                        elif isinstance(value, list) and value:
                            # If it's a list, return the first meaningful element or the whole list
                            result = str(value[0]) if len(value) == 1 else str(value)
                            self.log_fn(f"Extracted from list field '{key}': {result[:100]}...")
                            return result
                        else:
                            result = str(value)
                            self.log_fn(f"Extracted from field '{key}': {result[:100]}...")
                            return result
                # If no known field, return the whole object as string
                self.log_fn("No recognized fields found, returning full object")
                return str(last_obj)
            else:
                self.log_fn("No JSON objects found in response")
        except Exception as e:
            self.log_fn(f"Flexible extraction failed: {type(e).__name__}: {e}")
        
        # Fallback 1: Look for patterns like "response": "..." or 'response': '...'
        try:
            patterns = [
                r'"response"\s*:\s*"([^"]+)"',
                r"'response'\s*:\s*'([^']+)'",
                r'"answer"\s*:\s*"([^"]+)"',
                r"'answer'\s*:\s*'([^']+)'",
                r'"grade"\s*:\s*"([^"]+)"',
                r'"score"\s*:\s*"?([^"},]+)"?',
                r'"evaluation"\s*:\s*"([^"]+)"',
            ]
            for pattern in patterns:
                match = re.search(pattern, last_text, re.IGNORECASE)
                if match:
                    result = match.group(1).strip()
                    self.log_fn(f"Regex extraction succeeded with pattern: {pattern[:30]}...")
                    return result
        except Exception as e:
            self.log_fn(f"Regex extraction failed: {e}")
        
        # Fallback 2: Try to extract any meaningful text after common markers
        try:
            markers = [
                r'(?:final|conclusion|verdict|answer|result|grade|score|evaluation)[:\s]+(.+?)(?:\n|$)',
                r'(?:therefore|thus|so|in conclusion)[:\s]+(.+?)(?:\n|$)',
                r'(?:the student\s*(?:answer|solution)\s*(?:is|should be|receives))[:\s]+(.+?)(?:\n|$)',
            ]
            for pattern in markers:
                match = re.search(pattern, last_text, re.IGNORECASE)
                if match:
                    result = match.group(1).strip()
                    self.log_fn(f"Marker extraction succeeded")
                    return result
        except Exception as e:
            self.log_fn(f"Marker extraction failed: {e}")
        
        # Fallback 3: Look for IMO-specific grading keywords in the text
        try:
            text_lower = last_text.lower()
            if any(word in text_lower for word in ["correct", "full marks", "full credit", "7/7", "7 points"]):
                self.log_fn("Detected 'correct' in response")
                return "Correct"
            elif any(word in text_lower for word in ["incorrect", "wrong", "0/7", "zero", "no credit"]):
                self.log_fn("Detected 'incorrect' in response")
                return "Incorrect"
            elif any(word in text_lower for word in ["partial", "partial credit", "partial marks"]):
                self.log_fn("Detected 'partial' in response")
                return "Partial"
        except Exception as e:
            self.log_fn(f"Keyword extraction failed: {e}")
        
        # Fallback 4: If the text looks like a direct answer (short, no JSON), use it
        stripped = last_text.strip()
        if len(stripped) < 200 and '"' not in stripped and '{' not in stripped:
            self.log_fn(f"Using direct text response: {stripped[:100]}...")
            return stripped
        
        self.log_fn("All extraction methods failed, returning 'None'")
        return "None"

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Validate inputs
        required_fields = ["problem", "solution", "student_answer"]
        missing = [f for f in required_fields if not inputs.get(f)]
        if missing:
            self.log_fn(f"Warning: Missing required fields: {missing}")
        
        # Log problem domain for context
        domain = inputs.get("domain", "Mathematics")
        self.log_fn(f"Processing {domain} grading task")
        
        instruction = self._build_prompt(inputs)
        
        # Log prompt size
        self.log_fn(f"Prompt size: {len(instruction)} characters")

        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
            
            # Log LLM call info
            usage = info.get("usage", {})
            if usage:
                self.log_fn(f"LLM usage - Input: {usage.get('prompt_tokens', 'N/A')}, Output: {usage.get('completion_tokens', 'N/A')}")
            
        except Exception as e:
            self.log_fn(f"Error calling LLM: {type(e).__name__}: {e}")
            return "Error: LLM call failed", [{"role": "assistant", "text": f"Error: {e}"}]

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
            self.log_fn(f"Extracted prediction: {prediction[:200]}...")

        return str(prediction), msg_history
