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
    Also attempts to extract JSON from markdown code blocks as a fallback.
    """
    results = []
    search_from = 0
    
    # Primary extraction: <json>...</json> blocks
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
        except json.JSONDecodeError as e:
            logger.debug(f"Failed to parse JSON from <json> block: {e}")
            continue
    
    # Fallback extraction: markdown code blocks with json
    if not results:
        pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                results.append(json.loads(match.strip()))
            except json.JSONDecodeError:
                continue
    
    # Final fallback: try to find any JSON-like structure
    if not results:
        try:
            # Look for content between first { and last }
            start = text.find('{')
            end = text.rfind('}')
            if start != -1 and end != -1 and end > start:
                results.append(json.loads(text[start:end+1]))
        except json.JSONDecodeError:
            pass
    
    return results or None


class TaskAgent:
    """Task agent that solves IMO grading problems."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.log_file = log_file

    def _validate_inputs(self, inputs: dict) -> tuple[bool, str]:
        """Validate that all required fields are present in the inputs.
        
        Returns:
            (is_valid, error_message)
        """
        required_fields = ["domain", "problem", "solution", "grading_guidelines", "student_answer"]
        missing_fields = [field for field in required_fields if field not in inputs]
        
        if missing_fields:
            return False, f"Missing required fields: {', '.join(missing_fields)}"
        
        # Validate that fields are not empty
        empty_fields = [field for field in required_fields if not inputs.get(field)]
        if empty_fields:
            return False, f"Empty required fields: {', '.join(empty_fields)}"
        
        return True, ""

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Validate inputs before processing
        is_valid, error_msg = self._validate_inputs(inputs)
        if not is_valid:
            self.log_fn(f"Input validation failed: {error_msg}")
            return f"Error: {error_msg}", []
        
        # Build a more structured prompt for better results
        instruction = f"""You are an expert grading agent. Your task is to evaluate a student's answer based on the provided problem, solution, and grading guidelines.

Task Input:
```json
{json.dumps(inputs, indent=2, default=str)}
```

Instructions:
1. Carefully read the problem, solution, and grading guidelines
2. Evaluate the student's answer against the guidelines
3. Provide your evaluation in the exact JSON format below

Respond in JSON format with the following schema:
<json>
{{
    "response": "Your detailed evaluation and grade here"
}}
</json>

Important: Your response must be valid JSON wrapped in <json>...</json> tags."""

        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
        except Exception as e:
            self.log_fn(f"LLM call failed: {e}")
            return f"Error: LLM call failed - {e}", []

        # Extract prediction from JSON
        prediction = "None"
        try:
            if msg_history and len(msg_history) > 0:
                extracted = _extract_jsons(msg_history[-1]["text"])
                if extracted and len(extracted) > 0:
                    last_extract = extracted[-1]
                    if "response" in last_extract:
                        prediction = last_extract["response"]
                        self.log_fn(f"Successfully extracted prediction: {prediction[:100]}...")
                    else:
                        self.log_fn(f"No 'response' key in extracted JSON: {last_extract}")
                        prediction = str(last_extract)
                else:
                    self.log_fn("No JSON extracted from response")
                    # Fallback: use raw response text
                    prediction = msg_history[-1]["text"][:500] if msg_history else "None"
            else:
                self.log_fn("Empty message history")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            prediction = f"Error: {e}"

        return str(prediction), msg_history
