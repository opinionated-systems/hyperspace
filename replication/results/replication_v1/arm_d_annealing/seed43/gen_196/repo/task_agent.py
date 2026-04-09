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
    Also handles nested JSON objects and common formatting issues.
    Falls back to extracting raw JSON objects if no tags are found.
    """
    results = []
    search_from = 0
    
    # First, try to find JSON in <json> tags
    while True:
        start = text.find("<json>", search_from)
        if start == -1:
            break
        end = text.find("</json>", start)
        if end == -1:
            break
        inner = text[start + 6:end].strip()
        search_from = end + 7
        
        # Try to parse the JSON
        try:
            results.append(json.loads(inner))
        except json.JSONDecodeError:
            # Try to fix common JSON issues
            try:
                # Remove trailing commas before closing braces/brackets
                fixed = re.sub(r',(\s*[}\]])', r'\1', inner)
                # Fix single quotes to double quotes (common LLM mistake)
                # Use DOTALL flag to handle multiline strings
                fixed = re.sub(r"'([^']*?)'\s*:", r'"\1":', fixed, flags=re.DOTALL)
                fixed = re.sub(r":\s*'([^']*?)'", r': "\1"', fixed, flags=re.DOTALL)
                # Fix unescaped newlines in strings (common LLM issue)
                fixed = _fix_unescaped_newlines(fixed)
                results.append(json.loads(fixed))
            except json.JSONDecodeError:
                # Try extracting just the first valid JSON object
                try:
                    # Find the first { and matching }
                    brace_start = inner.find('{')
                    if brace_start != -1:
                        brace_count = 0
                        brace_end = brace_start
                        for i, char in enumerate(inner[brace_start:], start=brace_start):
                            if char == '{':
                                brace_count += 1
                            elif char == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    brace_end = brace_start + i + 1
                                    break
                        if brace_count == 0:
                            results.append(json.loads(inner[brace_start:brace_end]))
                except (json.JSONDecodeError, ValueError):
                    continue
    
    # If no results from tags, try to find raw JSON objects
    if not results:
        # Look for JSON-like structures with balanced braces
        brace_pattern = r'\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}'
        for match in re.finditer(brace_pattern, text, re.DOTALL):
            try:
                json_str = match.group(0)
                # Clean up common issues
                json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
                results.append(json.loads(json_str))
            except json.JSONDecodeError:
                continue
    
    return results or None


def _fix_unescaped_newlines(json_str: str) -> str:
    """Fix unescaped newlines inside JSON string values.
    
    This is a common issue where LLMs output raw newlines inside
    JSON strings instead of \\n escape sequences.
    """
    result = []
    in_string = False
    i = 0
    while i < len(json_str):
        char = json_str[i]
        
        if char == '"' and (i == 0 or json_str[i-1] != '\\'):
            in_string = not in_string
            result.append(char)
        elif in_string and char == '\n':
            result.append('\\n')
        elif in_string and char == '\r':
            result.append('\\r')
        elif in_string and char == '\t':
            result.append('\\t')
        else:
            result.append(char)
        i += 1
    
    return ''.join(result)


def _extract_json_with_regex(text: str) -> list[dict] | None:
    """Fallback JSON extraction using regex for malformed responses."""
    results = []
    
    # Try to find JSON objects in code blocks (with or without json label)
    code_block_patterns = [
        r'```(?:json)?\s*(\{[\s\S]*?\})\s*```',  # Standard code blocks
        r'```\s*(\{[\s\S]*?\})\s*```',  # Any code block with braces
    ]
    
    for pattern in code_block_patterns:
        for match in re.finditer(pattern, text, re.DOTALL):
            try:
                json_str = match.group(1)
                # Clean up common issues
                json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)  # Trailing commas
                results.append(json.loads(json_str))
            except json.JSONDecodeError:
                continue
    
    # Try to find raw JSON objects (not in code blocks)
    if not results:
        # Look for JSON-like structures with balanced braces
        brace_pattern = r'\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}'
        for match in re.finditer(brace_pattern, text, re.DOTALL):
            try:
                results.append(json.loads(match.group(0)))
            except json.JSONDecodeError:
                continue
    
    # Last resort: extract response and reasoning fields directly
    if not results:
        response_match = re.search(r'"response"\s*:\s*"([^"]*)"', text)
        reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]*)"', text)
        if response_match or reasoning_match:
            obj = {}
            if response_match:
                obj["response"] = response_match.group(1)
            if reasoning_match:
                obj["reasoning"] = reasoning_match.group(1)
            results.append(obj)
    
    # Extract from plain text if no JSON found
    if not results:
        # Look for grade/assessment in plain text
        grade_patterns = [
            r'(?:grade|assessment|score|result)\s*[:=]\s*["\']?([^"\'\n]+)["\']?',
            r'(?:the answer is|conclusion)\s*[:=]\s*["\']?([^"\'\n]+)["\']?',
        ]
        for pattern in grade_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                results.append({"response": match.group(1).strip()})
                break
    
    return results or None


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.max_retries = 3
        self._call_count = 0

    def _validate_inputs(self, inputs: dict) -> tuple[bool, str]:
        """Validate that all required fields are present in inputs.
        
        Returns:
            (is_valid, error_message) tuple
        """
        required_fields = ["domain", "problem", "solution", "grading_guidelines", "student_answer"]
        missing = [f for f in required_fields if f not in inputs or not inputs[f]]
        if missing:
            return False, f"Missing required fields: {', '.join(missing)}"
        return True, ""

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
- **Correct**: The answer matches the solution or uses an equivalent valid approach with correct reasoning and final result.
- **Partial**: The answer shows some correct reasoning but has minor errors, incomplete steps, or partially correct results.
- **Incorrect**: The answer contains fundamental errors, wrong approach, or completely wrong results.

## Response Format (REQUIRED - STRICT)
You MUST respond with ONLY valid JSON wrapped in <json>...</json> tags. Do not include any text before or after the JSON tags.

<json>
{{
    "reasoning": "Your detailed step-by-step analysis and reasoning here. Be thorough and specific about what the student did right or wrong.",
    "response": "Your final grade/assessment (e.g., 'Correct', 'Partial', 'Incorrect', or a numeric score)"
}}
</json>

## CRITICAL RULES
1. The JSON MUST be wrapped in <json>...</json> tags
2. The JSON must be valid - no trailing commas, no unclosed quotes
3. The "response" field must contain ONLY the grade, not the reasoning
4. The "reasoning" field must contain your detailed analysis
5. Do not include any text outside the <json> tags
6. Use double quotes for all JSON strings, not single quotes

Example of a CORRECT response:
<json>
{{
    "reasoning": "The student correctly identified the approach and followed all steps. The final answer matches the solution exactly.",
    "response": "Correct"
}}
</json>"""

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
            last_json = extracted[-1]
            if "response" in last_json:
                prediction = str(last_json["response"])
            if "reasoning" in last_json:
                reasoning = str(last_json["reasoning"])
        
        return prediction, reasoning

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        self._call_count += 1
        
        # Validate inputs first
        is_valid, error_msg = self._validate_inputs(inputs)
        if not is_valid:
            self.log_fn(f"Input validation failed: {error_msg}")
            return f"Error: {error_msg}", []
        
        instruction = self._build_grading_prompt(inputs)
        
        msg_history = []
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
                
                last_text = msg_history[-1]["text"] if msg_history else ""
                prediction, reasoning = self._extract_prediction(last_text)
                
                if prediction != "None":
                    self.log_fn(f"[Call {self._call_count}] Successfully extracted prediction: {prediction}")
                    if reasoning:
                        self.log_fn(f"[Call {self._call_count}] Reasoning: {reasoning[:200]}...")
                    break
                else:
                    self.log_fn(f"[Call {self._call_count}] Attempt {attempt + 1}: Failed to extract prediction, retrying...")
                    # Add feedback for retry with stronger emphasis
                    if attempt < self.max_retries - 1:
                        instruction = f"""ERROR: Your previous response did not contain valid JSON with a 'response' field.

You MUST respond with ONLY valid JSON wrapped in <json>...</json> tags. Do not include any text before or after the JSON tags.

Correct format:
<json>
{{
    "reasoning": "Your detailed analysis here...",
    "response": "Correct"
}}
</json>

## COMMON MISTAKES TO AVOID:
1. Do NOT include markdown formatting outside the <json> tags
2. Do NOT use single quotes in JSON - only double quotes
3. Do NOT include trailing commas in JSON objects
4. Do NOT put the grade inside the reasoning field
5. Do NOT forget the closing </json> tag

Now try again with the original task:

{self._build_grading_prompt(inputs)}"""
                    
            except Exception as e:
                self.log_fn(f"[Call {self._call_count}] Error in attempt {attempt + 1}: {e}")
                if attempt == self.max_retries - 1:
                    prediction = f"Error: {e}"
        
        return str(prediction), msg_history
