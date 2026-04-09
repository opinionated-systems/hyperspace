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
    
    # Strategy 3: Look for JSON objects directly
    if not results:
        # Try to find JSON objects between curly braces
        pattern = r'\{[^{}]*"[^"]+"[^{}]*\}'
        for match in re.finditer(pattern, text, re.DOTALL):
            try:
                # Expand to capture nested structures
                start = match.start()
                brace_count = 0
                end = start
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
                    try:
                        results.append(json.loads(obj_str))
                    except json.JSONDecodeError:
                        repaired = _repair_json(obj_str)
                        if repaired:
                            results.append(repaired)
            except (json.JSONDecodeError, ValueError):
                continue
    
    # Strategy 4: Try to find JSON at the very end of the text
    # Sometimes models put JSON at the end after explanatory text
    if not results:
        # Look for the last occurrence of a pattern that looks like JSON
        last_brace = text.rfind('}')
        if last_brace != -1:
            # Try to find the matching opening brace
            brace_count = 0
            for i in range(last_brace, -1, -1):
                if text[i] == '}':
                    brace_count += 1
                elif text[i] == '{':
                    brace_count -= 1
                    if brace_count == 0:
                        obj_str = text[i:last_brace+1]
                        try:
                            results.append(json.loads(obj_str))
                            break
                        except json.JSONDecodeError:
                            repaired = _repair_json(obj_str)
                            if repaired:
                                results.append(repaired)
                                break
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
    """
    try:
        # First, try parsing as-is
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Remove trailing commas before } or ]
    repaired = re.sub(r',\s*([}\]])', r'\1', text)
    
    # Try to fix single quotes (more comprehensive approach)
    # Replace single quotes around keys with double quotes
    repaired = re.sub(r"(?<=[{\s,])'([^']*?)'(?=\s*:)", r'"\1"', repaired)
    # Replace single quotes around string values with double quotes
    repaired = re.sub(r":\s*'([^']*?)'(?=\s*[,}\]])", r': "\1"', repaired)
    
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
    repaired = re.sub(r'"((?:[^"\\]|\\.)*?)"', escape_string_content, repaired)
    
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

## Output Format (CRITICAL)

You MUST respond with a valid JSON object wrapped in <json> tags. The JSON must be properly formatted with double quotes for all strings and keys.

<json>
{{
    "reasoning": "Your detailed step-by-step analysis here. Be thorough and specific.",
    "response": "Your final answer/evaluation here. Be concise and clear."
}}
</json>

Important:
- Use ONLY double quotes in the JSON (no single quotes)
- Ensure all strings are properly escaped
- The "response" field should contain your final determination (e.g., the correct answer, a score, or an evaluation of correctness)
- Do not include any text outside the <json> tags"""
        
        return prompt

    def _extract_prediction(self, msg_history: list[dict]) -> str:
        """Extract prediction from message history with multiple fallback strategies."""
        if not msg_history:
            self.log_fn("Warning: Empty message history")
            return "None"
        
        last_text = msg_history[-1].get("text", "")
        if not last_text:
            self.log_fn("Warning: Last message has no text content")
            return "None"
        
        # Try flexible extraction first
        try:
            extracted = _extract_json_flexible(last_text)
            if extracted:
                # Prefer response field, fallback to other common fields
                last_obj = extracted[-1]
                for key in ["response", "answer", "evaluation", "result", "grade", "conclusion", "final_answer"]:
                    if key in last_obj:
                        value = last_obj[key]
                        # Handle different value types
                        if isinstance(value, (str, int, float, bool)):
                            return str(value)
                        elif isinstance(value, (list, dict)):
                            return json.dumps(value)
                # If no known field, return the whole object as string
                return str(last_obj)
        except Exception as e:
            self.log_fn(f"Flexible extraction failed: {e}")
        
        # Fallback 1: try to find any JSON-like structure with regex
        try:
            # Look for patterns like "response": "..." or 'response': '...'
            patterns = [
                r'"response"\s*:\s*"([^"]+)"',
                r'"response"\s*:\s*\'([^\']+)\'',
                r'"answer"\s*:\s*"([^"]+)"',
                r'"evaluation"\s*:\s*"([^"]+)"',
                r'"result"\s*:\s*"([^"]+)"',
            ]
            for pattern in patterns:
                match = re.search(pattern, last_text, re.IGNORECASE)
                if match:
                    return match.group(1)
        except Exception as e:
            self.log_fn(f"Regex extraction failed: {e}")
        
        # Fallback 2: if text contains "correct" or "incorrect", extract that
        text_lower = last_text.lower()
        if "correct" in text_lower and "incorrect" not in text_lower:
            return "correct"
        elif "incorrect" in text_lower:
            return "incorrect"
        
        # Fallback 3: return first 200 chars of text as prediction
        if last_text.strip():
            return last_text.strip()[:200]
        
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

        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
        except Exception as e:
            error_msg = f"Error: LLM call failed: {e}"
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
