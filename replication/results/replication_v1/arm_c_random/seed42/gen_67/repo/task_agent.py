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
    
    Also handles markdown code blocks (```json) as a fallback.
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
            # Try to extract just the first JSON object if there's extra text
            try:
                # Find the first { and last }
                json_start = inner.find('{')
                json_end = inner.rfind('}')
                if json_start != -1 and json_end != -1 and json_end > json_start:
                    results.append(json.loads(inner[json_start:json_end+1]))
            except json.JSONDecodeError:
                continue
    
    # Fallback: try markdown code blocks if no <json> blocks found
    if not results:
        search_from = 0
        while True:
            start = text.find("```json", search_from)
            if start == -1:
                # Also try without language specifier
                start = text.find("```", search_from)
                if start == -1:
                    break
                end_marker = text.find("```", start + 3)
            else:
                end_marker = text.find("```", start + 7)
            
            if end_marker == -1:
                break
                
            if text.find("```json", search_from) == start:
                inner = text[start + 7:end_marker].strip()
            else:
                inner = text[start + 3:end_marker].strip()
                
            search_from = end_marker + 3
            try:
                results.append(json.loads(inner))
            except json.JSONDecodeError:
                continue
    
    return results or None


class TaskAgent:
    """Task agent that solves IMO grading problems."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        instruction = f"""You are an expert grading agent. Your task is to evaluate student answers based on the provided problem, solution, and grading guidelines.

Task input:
```
{inputs}
```

Instructions:
1. Carefully read the problem, solution, grading guidelines, and student answer
2. Determine if the student's answer is correct, partially correct, or incorrect
3. Provide your evaluation in the exact JSON format below

You MUST respond in JSON format with the following schema:
<json>
{{
    "response": "Your evaluation here - be specific about what was correct/incorrect"
}}
</json>

Important: 
- Always wrap your JSON in <json>...</json> tags
- The "response" field should contain your detailed evaluation
- Be thorough and reference specific parts of the grading guidelines"""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON with enhanced error handling
        prediction = "None"
        try:
            if not msg_history:
                self.log_fn("Warning: Empty message history, cannot extract prediction")
            else:
                last_message = msg_history[-1].get("text", "")
                if not last_message:
                    self.log_fn("Warning: Last message has no text content")
                else:
                    extracted = _extract_jsons(last_message)
                    if extracted:
                        # Try to find the response key in any of the extracted JSONs
                        for json_obj in reversed(extracted):
                            if "response" in json_obj:
                                prediction = json_obj["response"]
                                self.log_fn(f"Successfully extracted prediction: {prediction}")
                                break
                        else:
                            # If no response key found, use the last JSON's string representation
                            prediction = str(extracted[-1])
                            self.log_fn(f"Warning: No 'response' key found, using full JSON: {prediction}")
                    else:
                        # Fallback: use the raw response text if no JSON found
                        prediction = last_message.strip()
                        self.log_fn(f"Warning: No JSON blocks found, using raw response")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {type(e).__name__}: {e}")
            # Last resort fallback
            if msg_history and msg_history[-1].get("text"):
                prediction = msg_history[-1]["text"].strip()

        return str(prediction), msg_history
