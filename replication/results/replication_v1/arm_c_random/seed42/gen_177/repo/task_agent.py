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
        # Improved regex to handle nested JSON structures more robustly
        # Uses a depth-based approach to match balanced braces
        def find_json_objects(text):
            """Find JSON objects by tracking brace depth."""
            objects = []
            i = 0
            while i < len(text):
                if text[i] == '{':
                    start = i
                    depth = 1
                    i += 1
                    in_string = False
                    escape_next = False
                    
                    while i < len(text) and depth > 0:
                        char = text[i]
                        if escape_next:
                            escape_next = False
                        elif char == '\\':
                            escape_next = True
                        elif char == '"' and not in_string:
                            in_string = True
                        elif char == '"' and in_string:
                            in_string = False
                        elif not in_string:
                            if char == '{':
                                depth += 1
                            elif char == '}':
                                depth -= 1
                        i += 1
                    
                    if depth == 0:
                        objects.append(text[start:i])
                else:
                    i += 1
            return objects
        
        for json_str in find_json_objects(text):
            try:
                parsed = json.loads(json_str)
                if isinstance(parsed, dict):
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
        
        # Build structured instruction with clearer formatting
        instruction = f"""You are an expert grading agent. Your task is to evaluate a student's answer.

Task Input:
{json.dumps(inputs, indent=2, default=str)}

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
