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
    6. Extract from markdown-style responses
    """
    results = []
    extraction_log = []
    
    # Strategy 1: <json> tags (original)
    search_from = 0
    json_tag_count = 0
    while True:
        start = text.find("<json>", search_from)
        if start == -1:
            break
        end = text.find("</json>", start)
        if end == -1:
            extraction_log.append(f"Unclosed <json> tag at position {start}")
            break
        inner = text[start + 6:end].strip()
        search_from = end + 7
        json_tag_count += 1
        try:
            results.append(json.loads(inner))
            extraction_log.append(f"Successfully parsed <json> block #{json_tag_count}")
        except json.JSONDecodeError as e:
            extraction_log.append(f"JSON parse error in <json> block #{json_tag_count}: {e}")
            # Try to repair common JSON errors
            repaired = _repair_json(inner)
            if repaired:
                results.append(repaired)
                extraction_log.append(f"Successfully repaired <json> block #{json_tag_count}")
            continue
    
    # Strategy 2: ```json code blocks
    if not results:
        pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
        matches = re.findall(pattern, text, re.DOTALL)
        extraction_log.append(f"Found {len(matches)} code blocks")
        for i, match in enumerate(matches):
            try:
                results.append(json.loads(match.strip()))
                extraction_log.append(f"Successfully parsed code block #{i+1}")
            except json.JSONDecodeError as e:
                extraction_log.append(f"JSON parse error in code block #{i+1}: {e}")
                repaired = _repair_json(match.strip())
                if repaired:
                    results.append(repaired)
                    extraction_log.append(f"Successfully repaired code block #{i+1}")
                continue
    
    # Strategy 3: Look for JSON objects directly with improved brace matching
    if not results:
        # Try to find JSON objects between curly braces with proper nesting
        start_indices = [m.start() for m in re.finditer(r'\{', text)]
        extraction_log.append(f"Found {len(start_indices)} potential JSON starts")
        
        for start_idx in start_indices:
            try:
                # Use proper brace counting to find the matching end
                brace_count = 0
                end_idx = start_idx
                in_string = False
                escape_next = False
                
                for i, char in enumerate(text[start_idx:]):
                    if escape_next:
                        escape_next = False
                        continue
                    if char == '\\' and in_string:
                        escape_next = True
                        continue
                    if char == '"' and not in_string:
                        in_string = True
                    elif char == '"' and in_string:
                        in_string = False
                    elif not in_string:
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                end_idx = start_idx + i + 1
                                break
                
                if end_idx > start_idx and brace_count == 0:
                    obj_str = text[start_idx:end_idx]
                    # Validate it looks like JSON (has quotes)
                    if '"' in obj_str:
                        try:
                            results.append(json.loads(obj_str))
                            extraction_log.append(f"Successfully parsed JSON at position {start_idx}")
                            break  # Only take the first valid JSON
                        except json.JSONDecodeError:
                            repaired = _repair_json(obj_str)
                            if repaired:
                                results.append(repaired)
                                extraction_log.append(f"Successfully repaired JSON at position {start_idx}")
                                break
            except Exception as e:
                extraction_log.append(f"Error processing position {start_idx}: {e}")
                continue
    
    # Log extraction attempts for debugging
    if not results:
        logger.debug(f"JSON extraction failed. Log: {extraction_log}")
    
    return results or None


def _repair_json(text: str) -> dict | None:
    """Attempt to repair common JSON syntax errors.
    
    Fixes:
    - Trailing commas before closing braces/brackets
    - Single quotes instead of double quotes
    - Unescaped newlines in strings
    - Missing closing braces/brackets
    - Unescaped quotes in strings
    - Comments (// and /* */)
    - Control characters
    """
    try:
        # First, try parsing as-is
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    repaired = text
    
    # Remove comments (both // and /* */ styles)
    repaired = re.sub(r'//[^\n]*', '', repaired)
    repaired = re.sub(r'/\*.*?\*/', '', repaired, flags=re.DOTALL)
    
    # Remove trailing commas before } or ]
    repaired = re.sub(r',\s*([}\]])', r'\1', repaired)
    
    # Try to fix single quotes (simple cases only)
    # Replace single quotes around keys/values with double quotes
    # This is a best-effort approach
    repaired = re.sub(r"'([^']*?)':", r'"\1":', repaired)
    repaired = re.sub(r":\s*'([^']*?)'([,}\]])", r': "\1"\2', repaired)
    
    # Escape unescaped newlines in strings
    repaired = re.sub(r'(?<!\\)\n', r'\\n', repaired)
    repaired = re.sub(r'(?<!\\)\r', r'\\r', repaired)
    repaired = re.sub(r'(?<!\\)\t', r'\\t', repaired)
    
    # Remove control characters
    repaired = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', repaired)
    
    # Try to balance braces
    open_braces = repaired.count('{') - repaired.count('}')
    if open_braces > 0:
        repaired += '}' * open_braces
    elif open_braces < 0:
        # Too many closing braces - try to find valid subset
        repaired = repaired[:repaired.rfind('}')]
    
    open_brackets = repaired.count('[') - repaired.count(']')
    if open_brackets > 0:
        repaired += ']' * open_brackets
    elif open_brackets < 0:
        # Too many closing brackets
        repaired = repaired[:repaired.rfind(']')]
    
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        # Last resort: try to extract just the first valid object
        try:
            # Find the first { and last } and try that
            first_brace = repaired.find('{')
            last_brace = repaired.rfind('}')
            if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                subset = repaired[first_brace:last_brace+1]
                return json.loads(subset)
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
        
        This method implements a robust extraction pipeline that tries multiple
        strategies in order of reliability:
        1. Flexible JSON extraction with field preference
        2. Regex-based field extraction
        3. Keyword-based classification
        4. Raw text truncation
        
        Each failure is logged for debugging purposes.
        """
        if not msg_history:
            self.log_fn("Warning: Empty message history")
            return "None"
        
        last_text = msg_history[-1].get("text", "")
        if not last_text:
            self.log_fn("Warning: Last message has no text content")
            return "None"
        
        # Log the response length for debugging
        self.log_fn(f"Processing response of {len(last_text)} characters")
        
        # Try flexible extraction first
        try:
            extracted = _extract_json_flexible(last_text)
            if extracted:
                # Prefer response field, fallback to other common fields
                last_obj = extracted[-1]
                self.log_fn(f"Successfully extracted JSON with keys: {list(last_obj.keys())}")
                
                # Priority order for fields
                priority_fields = [
                    "response", "answer", "evaluation", "result", 
                    "grade", "conclusion", "verdict", "decision",
                    "output", "prediction", "assessment"
                ]
                
                for key in priority_fields:
                    if key in last_obj:
                        value = last_obj[key]
                        # Handle different value types
                        if isinstance(value, (str, int, float, bool)):
                            result = str(value)
                            self.log_fn(f"Extracted '{key}': {result[:100]}")
                            return result
                        elif isinstance(value, (list, dict)):
                            result = json.dumps(value)
                            self.log_fn(f"Extracted '{key}' (complex): {result[:100]}")
                            return result
                
                # If no known field, check if there's a single value field
                if len(last_obj) == 1:
                    key, value = list(last_obj.items())[0]
                    if isinstance(value, (str, int, float, bool)):
                        result = str(value)
                        self.log_fn(f"Extracted single field '{key}': {result[:100]}")
                        return result
                
                # If no known field, return the whole object as string
                self.log_fn(f"No priority field found, returning full object")
                return str(last_obj)
        except Exception as e:
            self.log_fn(f"Flexible extraction failed: {e}")
        
        # Fallback 1: try to find any JSON-like structure with regex
        try:
            # Look for patterns like "response": "..." or 'response': '...'
            patterns = [
                (r'"response"\s*:\s*"([^"]+)"', "response"),
                (r'"response"\s*:\s*\'([^\']+)\'', "response (single quote)"),
                (r'"answer"\s*:\s*"([^"]+)"', "answer"),
                (r'"evaluation"\s*:\s*"([^"]+)"', "evaluation"),
                (r'"result"\s*:\s*"([^"]+)"', "result"),
                (r'"grade"\s*:\s*"([^"]+)"', "grade"),
                (r'"verdict"\s*:\s*"([^"]+)"', "verdict"),
            ]
            for pattern, field_name in patterns:
                match = re.search(pattern, last_text)
                if match:
                    result = match.group(1)
                    self.log_fn(f"Regex extracted '{field_name}': {result[:100]}")
                    return result
        except Exception as e:
            self.log_fn(f"Regex extraction failed: {e}")
        
        # Fallback 2: if text contains "correct" or "incorrect", extract that
        text_lower = last_text.lower()
        if "correct" in text_lower and "incorrect" not in text_lower:
            self.log_fn("Keyword extraction: 'correct'")
            return "correct"
        elif "incorrect" in text_lower:
            self.log_fn("Keyword extraction: 'incorrect'")
            return "incorrect"
        
        # Fallback 3: return first 200 chars of text as prediction
        if last_text.strip():
            result = last_text.strip()[:200]
            self.log_fn(f"Returning truncated text: {result[:100]}")
            return result
        
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
        
        # Validate input types
        for key in required_keys:
            if not isinstance(inputs[key], str):
                error_msg = f"Error: Input '{key}' must be a string, got {type(inputs[key]).__name__}"
                self.log_fn(error_msg)
                return error_msg, [{"role": "assistant", "text": error_msg}]
        
        instruction = self._build_prompt(inputs)
        
        # Log the problem being processed (truncated for privacy)
        problem_preview = inputs.get("problem", "")[:100]
        student_preview = inputs.get("student_answer", "")[:50]
        self.log_fn(f"Processing problem: {problem_preview}...")
        self.log_fn(f"Student answer preview: {student_preview}...")

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
