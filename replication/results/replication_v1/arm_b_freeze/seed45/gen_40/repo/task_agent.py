"""
Task agent: solves a given task with chain-of-thought reasoning.

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
    Also handles markdown code blocks and bare JSON objects.
    Includes robust handling for nested braces and common JSON formatting issues.
    """
    results = []
    search_from = 0
    
    # First try to find <json>...</json> blocks
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
            # Try to extract JSON from within the content using brace matching
            try:
                json_str = _extract_json_with_brace_matching(inner)
                if json_str:
                    results.append(json.loads(json_str))
            except (json.JSONDecodeError, ValueError):
                continue
    
    # If no <json> blocks found, try markdown code blocks
    if not results:
        # Try ```json ... ``` blocks
        json_blocks = re.findall(r'```json\s*(.*?)```', text, re.DOTALL)
        for block in json_blocks:
            try:
                results.append(json.loads(block.strip()))
            except json.JSONDecodeError:
                # Try brace matching as fallback
                try:
                    json_str = _extract_json_with_brace_matching(block)
                    if json_str:
                        results.append(json.loads(json_str))
                except (json.JSONDecodeError, ValueError):
                    continue
        
        # Try bare JSON objects as fallback
        if not results:
            # Use brace matching to find JSON objects
            json_str = _extract_json_with_brace_matching(text)
            if json_str:
                try:
                    results.append(json.loads(json_str))
                except json.JSONDecodeError:
                    pass
    
    return results or None


def _extract_json_with_brace_matching(text: str) -> str | None:
    """Extract a JSON object from text using brace counting.
    
    This handles nested braces correctly by counting open/close braces.
    Returns the first valid JSON object found, or None if no valid JSON.
    """
    start = text.find("{")
    if start == -1:
        return None
    
    brace_count = 0
    in_string = False
    escape_next = False
    
    for i, char in enumerate(text[start:], start):
        if escape_next:
            escape_next = False
            continue
        
        if char == "\\" and in_string:
            escape_next = True
            continue
        
        if char == '"' and not in_string:
            in_string = True
        elif char == '"' and in_string:
            in_string = False
        elif not in_string:
            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0:
                    # Found a complete JSON object
                    return text[start:i + 1]
    
    return None


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.max_retries = 2  # Max retries for JSON extraction failures
        self._response_keys = ["response", "grade", "answer", "result", "assessment", "evaluation", "score"]
        self._reasoning_keys = ["reasoning", "analysis", "thought", "explanation", "rationale", "thinking"]

    def _extract_value(self, result: dict, keys: list[str], default: str) -> str:
        """Extract a value from result dict using multiple possible keys.
        
        Args:
            result: The JSON result dictionary
            keys: List of possible keys to try
            default: Default value if no key is found
            
        Returns:
            The extracted value or default
        """
        for key in keys:
            if key in result:
                value = result[key]
                # Handle both string and non-string values
                if isinstance(value, str):
                    return value.strip()
                else:
                    return str(value)
        return default

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with chain-of-thought reasoning.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Extract fields for better prompting
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = f"""You are an expert {domain} grader evaluating student solutions.

Your task is to grade a student's answer to a problem by comparing it against the official solution and following the grading guidelines.

## Problem Statement:
{problem}

## Official Solution:
{solution}

## Grading Guidelines:
{grading_guidelines}

## Student's Answer:
{student_answer}

## Instructions:
1. First, analyze the student's answer step by step. Identify what they did correctly and incorrectly.
2. Compare their approach to the official solution.
3. Consider the grading guidelines carefully.
4. Provide your reasoning before giving the final grade.
5. Respond ONLY in JSON format wrapped in <json>...</json> tags with the following exact schema:

<json>
{{
    "reasoning": "Your detailed analysis and reasoning about the student's answer...",
    "response": "The final grade/assessment (e.g., 'Correct', 'Partially Correct', 'Incorrect', or a numeric score)"
}}
</json>

IMPORTANT:
- Your entire response must be valid JSON inside <json>...</json> tags
- Do not include any text outside the JSON tags
- The "reasoning" field should contain your detailed analysis
- The "response" field should contain only the final grade/assessment
- Ensure the JSON is properly formatted with no syntax errors

Think carefully and provide a fair assessment based on the official solution and grading guidelines."""

        msg_history = []
        prediction = "None"
        reasoning = ""
        
        for attempt in range(self.max_retries + 1):
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=msg_history,
            )

            # Extract prediction from JSON
            extracted = None
            try:
                # Try the last assistant message
                last_msg = msg_history[-1]["text"] if msg_history else ""
                extracted = _extract_jsons(last_msg)
                
                # If not found, try searching all messages
                if not extracted:
                    for msg in reversed(msg_history):
                        text = msg.get("text", "")
                        extracted = _extract_jsons(text)
                        if extracted:
                            break
                
                if extracted:
                    result = extracted[-1]
                    
                    # Try multiple possible keys for the response
                    prediction = self._extract_value(result, self._response_keys, "None")
                    
                    # Extract reasoning if available
                    reasoning = self._extract_value(result, self._reasoning_keys, "")
                    
                    # Log reasoning if available
                    if reasoning:
                        self.log_fn(f"Reasoning: {reasoning[:200]}...")
                    
                    # Success - break out of retry loop
                    break
                elif attempt < self.max_retries:
                    # No JSON found - add feedback and retry
                    self.log_fn(f"No JSON found in attempt {attempt + 1}, retrying with feedback...")
                    feedback = (
                        "Your previous response did not contain valid JSON in the required format. "
                        "Please respond with a JSON object wrapped in <json>...</json> tags. "
                        "The JSON must have 'reasoning' and 'response' fields."
                    )
                    msg_history.append({"role": "user", "text": feedback})
                    instruction = feedback  # Update instruction for next iteration
                else:
                    self.log_fn(f"No JSON found after {self.max_retries + 1} attempts")
                    
            except Exception as e:
                self.log_fn(f"Error extracting prediction (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries:
                    feedback = (
                        f"Error parsing your response: {e}. "
                        "Please ensure your response contains valid JSON wrapped in <json>...</json> tags."
                    )
                    msg_history.append({"role": "user", "text": feedback})
                    instruction = feedback

        return str(prediction), msg_history
