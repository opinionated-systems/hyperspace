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


def _extract_jsons_with_retry(text: str, max_retries: int = 3) -> list[dict] | None:
    """Extract JSON objects with multiple fallback strategies.
    
    Tries multiple extraction methods in order of reliability:
    1. Standard <json>...</json> block extraction
    2. Direct JSON object parsing from the entire text
    3. Regex-based JSON extraction for malformed responses
    """
    # Strategy 1: Standard extraction
    results = _extract_jsons(text)
    if results:
        return results
    
    # Strategy 2: Look for JSON objects directly (without tags)
    try:
        # Try to find JSON objects in the text
        decoder = json.JSONDecoder()
        idx = 0
        while idx < len(text):
            try:
                # Skip whitespace
                while idx < len(text) and text[idx].isspace():
                    idx += 1
                if idx >= len(text):
                    break
                # Try to parse JSON starting at this position
                obj, end_idx = decoder.raw_decode(text, idx)
                if isinstance(obj, dict):
                    results.append(obj)
                idx += end_idx
            except json.JSONDecodeError:
                idx += 1
    except Exception:
        pass
    
    if results:
        return results
    
    # Strategy 3: Regex-based extraction for edge cases
    try:
        # Find anything that looks like a JSON object
        json_pattern = re.compile(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}')
        for match in json_pattern.finditer(text):
            try:
                obj = json.loads(match.group())
                if isinstance(obj, dict):
                    results.append(obj)
            except json.JSONDecodeError:
                continue
    except Exception:
        pass
    
    return results or None


class TaskAgent:
    """Task agent that solves IMO grading problems with enhanced reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def _validate_inputs(self, inputs: dict) -> tuple[bool, str]:
        """Validate that all required fields are present in inputs."""
        required_fields = ["domain", "problem", "solution", "grading_guidelines", "student_answer"]
        missing = [f for f in required_fields if f not in inputs]
        if missing:
            return False, f"Missing required fields: {missing}"
        return True, ""

    def _build_prompt(self, inputs: dict) -> str:
        """Build an enhanced prompt with chain-of-thought reasoning instructions."""
        domain = inputs.get("domain", "unknown")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        return f"""You are an expert grader for {domain} problems. Your task is to evaluate a student's answer against the correct solution and grading guidelines.

## Problem Statement
{problem}

## Correct Solution
{solution}

## Grading Guidelines
{grading_guidelines}

## Student's Answer
{student_answer}

## Instructions
Please evaluate the student's answer carefully. Follow these steps:

1. **Understand the Problem**: Make sure you understand what the problem is asking.
2. **Review the Solution**: Study the correct solution provided.
3. **Analyze the Guidelines**: Understand the grading criteria.
4. **Evaluate the Answer**: Compare the student's answer to the correct solution using the grading guidelines.
5. **Provide Your Assessment**: Give a clear, concise evaluation.

Think step by step before providing your final response.

Respond in JSON format with the following schema:
<json>
{{
    "reasoning": "Your step-by-step analysis and reasoning process",
    "response": "Your final evaluation/grade for the student's answer"
}}
</json>

Ensure your response is valid JSON wrapped in <json>...</json> tags."""

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with enhanced error handling.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Validate inputs
        is_valid, error_msg = self._validate_inputs(inputs)
        if not is_valid:
            self.log_fn(f"Input validation failed: {error_msg}")
            return f"Error: {error_msg}", []

        # Build enhanced prompt
        instruction = self._build_prompt(inputs)

        # Call LLM with retry logic
        max_attempts = 2
        prediction = "None"
        msg_history = []
        
        for attempt in range(max_attempts):
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=[] if attempt == 0 else msg_history,
                )

                # Extract prediction from JSON with retry logic
                extracted = _extract_jsons_with_retry(msg_history[-1]["text"])
                if extracted:
                    last_extract = extracted[-1]
                    # Prefer "response" field, fallback to other common fields
                    if "response" in last_extract:
                        prediction = last_extract["response"]
                    elif "answer" in last_extract:
                        prediction = last_extract["answer"]
                    elif "grade" in last_extract:
                        prediction = last_extract["grade"]
                    elif "evaluation" in last_extract:
                        prediction = last_extract["evaluation"]
                    else:
                        # If no recognized field, use the whole object as string
                        prediction = json.dumps(last_extract)
                    break  # Success, exit retry loop
                else:
                    if attempt < max_attempts - 1:
                        self.log_fn(f"No JSON extracted on attempt {attempt + 1}, retrying...")
                        # Modify instruction to be more explicit
                        instruction = instruction + "\n\nIMPORTANT: You MUST respond with valid JSON in <json>...</json> tags."
                    else:
                        self.log_fn("Failed to extract JSON after all attempts")
                        
            except Exception as e:
                self.log_fn(f"Error on attempt {attempt + 1}: {e}")
                if attempt == max_attempts - 1:
                    prediction = f"Error: {e}"

        return str(prediction), msg_history
