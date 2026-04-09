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
    
    Also attempts fallback extraction from markdown code blocks and 
    raw JSON objects if <json> tags are not found.
    """
    results = []
    search_from = 0
    
    # Primary: Extract from <json>...</json> blocks
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
    
    # Fallback 1: Extract from markdown code blocks ```json ... ```
    if not results:
        markdown_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
        for match in re.finditer(markdown_pattern, text, re.DOTALL):
            try:
                parsed = json.loads(match.group(1).strip())
                results.append(parsed)
            except json.JSONDecodeError:
                continue
    
    # Fallback 2: Extract raw JSON objects from text
    if not results:
        # Match JSON objects: starts with { and ends with }
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        for match in re.finditer(json_pattern, text):
            try:
                parsed = json.loads(match.group(0))
                results.append(parsed)
            except json.JSONDecodeError:
                continue
    
    return results or None


class TaskAgent:
    """Task agent that solves IMO grading problems."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.call_count = 0

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        self.call_count += 1
        self.log_fn(f"[TaskAgent #{self.call_count}] Processing new task")
        
        # Validate inputs
        if not isinstance(inputs, dict):
            error_msg = f"Invalid input type: expected dict, got {type(inputs).__name__}"
            self.log_fn(f"[TaskAgent #{self.call_count}] {error_msg}")
            return f"Error: {error_msg}", [{"role": "error", "text": error_msg}]
        
        # Check for required fields and provide helpful warnings
        required_fields = ["problem", "solution", "grading_guidelines", "student_answer"]
        missing_fields = [f for f in required_fields if f not in inputs]
        if missing_fields:
            self.log_fn(f"[TaskAgent #{self.call_count}] Warning: Missing fields: {missing_fields}")
        
        # Build structured instruction with clearer formatting
        # Truncate very long inputs to prevent token overflow
        max_field_length = 10000
        truncated_inputs = {}
        for key, value in inputs.items():
            if isinstance(value, str) and len(value) > max_field_length:
                truncated_inputs[key] = value[:max_field_length] + f"\n... [truncated, total length: {len(value)} chars]"
                self.log_fn(f"[TaskAgent #{self.call_count}] Truncated field '{key}' from {len(value)} to {max_field_length} chars")
            else:
                truncated_inputs[key] = value
        
        instruction = f"""You are an expert grading agent. Your task is to evaluate a student's answer.

Task Input:
{json.dumps(truncated_inputs, indent=2, default=str)}

Instructions:
1. Carefully analyze the problem, solution, grading guidelines, and student answer
2. Provide your evaluation in the exact JSON format below
3. Your response must be wrapped in <json>...</json> tags

Response Format:
<json>
{{
    "response": "Your detailed evaluation here"
}}
</json>"""

        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
            
            self.log_fn(f"[TaskAgent #{self.call_count}] LLM call successful, usage: {info.get('usage', {})}")
        except Exception as e:
            self.log_fn(f"[TaskAgent #{self.call_count}] LLM call failed: {e}")
            return "Error: LLM call failed", [{"role": "error", "text": str(e)}]

        # Extract prediction from JSON with enhanced error handling
        prediction = "None"
        extraction_errors = []
        
        try:
            if not msg_history:
                extraction_errors.append("Empty message history")
            else:
                last_message = msg_history[-1].get("text", "")
                if not last_message:
                    extraction_errors.append("Last message has no text content")
                else:
                    extracted = _extract_jsons(last_message)
                    if not extracted:
                        extraction_errors.append("No valid JSON found in response")
                    elif "response" not in extracted[-1]:
                        extraction_errors.append(f"JSON missing 'response' key. Keys found: {list(extracted[-1].keys())}")
                    else:
                        prediction = extracted[-1]["response"]
                        self.log_fn(f"[TaskAgent #{self.call_count}] Successfully extracted prediction")
        except Exception as e:
            extraction_errors.append(f"Exception during extraction: {e}")
            self.log_fn(f"[TaskAgent #{self.call_count}] Error extracting prediction: {e}")
        
        if extraction_errors:
            self.log_fn(f"[TaskAgent #{self.call_count}] Extraction issues: {'; '.join(extraction_errors)}")

        return str(prediction), msg_history
