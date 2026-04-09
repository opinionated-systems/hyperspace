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
    5. Repair common JSON syntax errors
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
            # Try to repair common JSON errors
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
                repaired = _repair_json(match.strip())
                if repaired:
                    results.append(repaired)
                continue
    
    # Strategy 3: Look for JSON objects directly with improved brace matching
    if not results:
        # Find all potential JSON object starts
        start_indices = [m.start() for m in re.finditer(r'\{', text)]
        
        for start in start_indices:
            # Try to find the matching closing brace
            brace_count = 0
            end = -1
            for i, char in enumerate(text[start:]):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end = start + i + 1
                        break
            
            if end > start:
                obj_str = text[start:end]
                # Quick validation: must contain at least one quoted key
                if '"' not in obj_str and "'" not in obj_str:
                    continue
                    
                try:
                    results.append(json.loads(obj_str))
                except json.JSONDecodeError:
                    repaired = _repair_json(obj_str)
                    if repaired:
                        results.append(repaired)
    
    # Strategy 4: Look for JSON at the very start or end of text
    if not results:
        # Try to find JSON at the start (after any whitespace)
        text_stripped = text.strip()
        if text_stripped.startswith('{'):
            # Find the matching closing brace
            brace_count = 0
            for i, char in enumerate(text_stripped):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        try:
                            obj = json.loads(text_stripped[:i+1])
                            results.append(obj)
                            break
                        except json.JSONDecodeError:
                            repaired = _repair_json(text_stripped[:i+1])
                            if repaired:
                                results.append(repaired)
                            break
    
    return results or None


def _repair_json(text: str) -> dict | None:
    """Attempt to repair common JSON syntax errors.
    
    Fixes:
    - Trailing commas before closing braces/brackets
    - Single quotes instead of double quotes
    - Unescaped newlines, tabs, and quotes in strings
    - Missing closing braces/brackets
    - Unicode escape sequences
    - Control characters
    - Malformed escape sequences
    - BOM and other invisible characters
    """
    try:
        # First, try parsing as-is
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Remove BOM and other invisible characters at the start
    repaired = text.lstrip('\ufeff\u200b\u200c\u200d\u2060\ufeff')
    
    # Remove trailing commas before } or ]
    repaired = re.sub(r',\s*([}\]])', r'\1', repaired)
    
    # Try to fix single quotes (more comprehensive approach)
    # Replace single quotes around keys with double quotes
    repaired = re.sub(r"(?<=[{\s,])'([^']*?)'(?=\s*:)", r'"\1"', repaired)
    # Replace single quotes around string values with double quotes
    repaired = re.sub(r":\s*'([^']*?)'(?=\s*[,}\]])", r': "\1"', repaired)
    
    # Fix common escape sequence issues
    # Replace invalid escape sequences with valid ones
    repaired = re.sub(r'\\(?!"|\\|/|b|f|n|r|t|u[0-9a-fA-F]{4})', r'\\\\', repaired)
    
    # Escape unescaped newlines, tabs, and carriage returns in strings
    # Use a more careful approach: find string content and escape within
    def escape_string_content(match):
        content = match.group(1)
        # Escape backslashes first (to avoid double-escaping)
        content = content.replace('\\', '\\\\')
        # Escape newlines, tabs, carriage returns
        content = content.replace('\n', '\\n')
        content = content.replace('\t', '\\t')
        content = content.replace('\r', '\\r')
        # Escape unescaped quotes
        content = re.sub(r'(?<!\\)"', r'\\"', content)
        return f'"{content}"'
    
    # Match string values in JSON (content between unescaped quotes)
    # Use a simpler regex to avoid issues
    try:
        repaired = re.sub(r'"((?:[^"\\]|\\.)*?)"', escape_string_content, repaired)
    except re.error:
        # If regex fails, skip this step
        pass
    
    # Remove control characters (except valid escapes)
    repaired = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', repaired)
    
    # Try to balance braces and brackets
    open_braces = repaired.count('{') - repaired.count('}')
    if open_braces > 0:
        repaired += '}' * open_braces
    elif open_braces < 0:
        # Remove extra closing braces
        repaired = repaired.rstrip('}')
    
    open_brackets = repaired.count('[') - repaired.count(']')
    if open_brackets > 0:
        repaired += ']' * open_brackets
    elif open_brackets < 0:
        # Remove extra closing brackets
        repaired = repaired.rstrip(']')
    
    # Final attempt to parse
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass
    
    # Last resort: try to extract just the first valid JSON object
    # Find the first { and try to find its matching }
    start = repaired.find('{')
    if start != -1:
        brace_count = 0
        for i, char in enumerate(repaired[start:]):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    try:
                        return json.loads(repaired[start:start+i+1])
                    except json.JSONDecodeError:
                        break
    
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
        
        This method implements a robust extraction pipeline that handles various
        response formats and edge cases commonly encountered in LLM outputs.
        """
        if not msg_history:
            self.log_fn("Warning: Empty message history")
            return "None"
        
        last_text = msg_history[-1].get("text", "")
        if not last_text:
            self.log_fn("Warning: Last message has no text content")
            return "None"
        
        # Log the raw response for debugging (truncated)
        raw_preview = last_text[:300].replace('\n', ' ')
        self.log_fn(f"Raw response preview: {raw_preview}...")
        
        # Strategy 1: Try flexible JSON extraction
        try:
            extracted = _extract_json_flexible(last_text)
            if extracted:
                # Prefer response field, fallback to other common fields
                last_obj = extracted[-1]
                self.log_fn(f"Successfully extracted JSON with keys: {list(last_obj.keys())}")
                
                # Priority order for response fields
                priority_keys = ["response", "answer", "evaluation", "result", "grade", "conclusion", "reasoning"]
                for key in priority_keys:
                    if key in last_obj:
                        value = last_obj[key]
                        # Handle different value types
                        if isinstance(value, (str, int, float, bool)):
                            result = str(value)
                            self.log_fn(f"Extracted '{key}': {result[:100]}...")
                            return result
                        elif isinstance(value, (list, dict)):
                            return json.dumps(value)
                
                # If no known field, return the whole object as string
                self.log_fn(f"No priority keys found, returning full object")
                return str(last_obj)
        except Exception as e:
            self.log_fn(f"Flexible extraction failed: {e}")
        
        # Strategy 2: Try to find any JSON-like structure with regex
        try:
            # Look for patterns like "response": "..." or 'response': '...'
            patterns = [
                (r'"response"\s*:\s*"([^"]+)"', "response (double-quoted)"),
                (r'"response"\s*:\s*\'([^\']+)\'', "response (single-quoted)"),
                (r'"answer"\s*:\s*"([^"]+)"', "answer"),
                (r'"evaluation"\s*:\s*"([^"]+)"', "evaluation"),
                (r'"result"\s*:\s*"([^"]+)"', "result"),
                (r'"grade"\s*:\s*"([^"]+)"', "grade"),
            ]
            for pattern, name in patterns:
                match = re.search(pattern, last_text)
                if match:
                    result = match.group(1)
                    self.log_fn(f"Regex extracted '{name}': {result[:100]}...")
                    return result
        except Exception as e:
            self.log_fn(f"Regex extraction failed: {e}")
        
        # Strategy 3: Look for explicit correctness indicators
        text_lower = last_text.lower()
        if "correct" in text_lower and "incorrect" not in text_lower:
            self.log_fn("Detected 'correct' in response")
            return "correct"
        elif "incorrect" in text_lower:
            self.log_fn("Detected 'incorrect' in response")
            return "incorrect"
        
        # Strategy 4: Look for numerical answers (common in math problems)
        try:
            # Look for patterns like "The answer is 42" or "Answer: 42"
            num_patterns = [
                r'[Tt]he answer is[:\s]+([\d\.\-\/\s]+)',
                r'[Aa]nswer[:\s]+([\d\.\-\/\s]+)',
                r'[Ff]inal answer[:\s]+([\d\.\-\/\s]+)',
            ]
            for pattern in num_patterns:
                match = re.search(pattern, text_lower)
                if match:
                    result = match.group(1).strip()
                    self.log_fn(f"Extracted numerical answer: {result}")
                    return result
        except Exception as e:
            self.log_fn(f"Numerical extraction failed: {e}")
        
        # Strategy 5: Return first 200 chars of text as prediction
        if last_text.strip():
            preview = last_text.strip()[:200]
            self.log_fn(f"Using text preview as prediction: {preview}...")
            return preview
        
        self.log_fn("All extraction strategies failed, returning 'None'")
        return "None"

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Validate required inputs
        required_keys = ["problem", "solution", "student_answer"]
        missing_keys = [k for k in required_keys if k not in inputs or not inputs[k]]
        if missing_keys:
            error_msg = f"Error: Missing required inputs: {missing_keys}"
            self.log_fn(error_msg)
            return error_msg, [{"role": "assistant", "text": error_msg}]
        
        instruction = self._build_prompt(inputs)
        
        # Log the problem being processed (truncated for privacy)
        problem_preview = inputs.get("problem", "")[:100]
        self.log_fn(f"Processing problem: {problem_preview}...")

        # Retry mechanism with exponential backoff for transient errors
        max_retries = 3
        base_delay = 2.0
        
        for attempt in range(max_retries):
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=[],
                )
                # Success - break out of retry loop
                break
            except Exception as e:
                error_str = str(e).lower()
                # Check for transient errors that are worth retrying
                transient_indicators = [
                    "timeout", "rate limit", "connection", "temporarily",
                    "503", "502", "504", "429", "too many requests",
                    "server error", "service unavailable"
                ]
                is_transient = any(indicator in error_str for indicator in transient_indicators)
                
                if is_transient and attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)  # Exponential backoff: 2, 4, 8 seconds
                    self.log_fn(f"Transient error on attempt {attempt + 1}: {e}. Retrying in {delay}s...")
                    import time
                    time.sleep(delay)
                    continue
                else:
                    # Non-transient error or max retries reached
                    error_msg = f"Error: LLM call failed after {attempt + 1} attempt(s): {e}"
                    self.log_fn(error_msg)
                    return error_msg, [{"role": "assistant", "text": error_msg}]

        # Extract prediction from JSON
        prediction = self._extract_prediction(msg_history)
        
        if prediction == "None":
            self.log_fn(f"Warning: Could not extract valid prediction from response")
            # Log the raw response for debugging
            if msg_history:
                raw_text = msg_history[-1].get('text', '')
                self.log_fn(f"Raw response preview: {raw_text[:500]}...")
        else:
            # Log successful extraction
            preview = prediction[:100] if len(prediction) > 100 else prediction
            self.log_fn(f"Extracted prediction: {preview}")

        return str(prediction), msg_history
