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
from typing import Any

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


def _extract_json_from_markdown(text: str) -> list[dict] | None:
    """Extract JSON objects from markdown code blocks (```json ... ```).

    Fallback for models that use markdown formatting instead of <json> tags.
    """
    results = []
    pattern = r'```(?:json)?\s*\n(.*?)\n```'
    matches = re.findall(pattern, text, re.DOTALL)
    for match in matches:
        try:
            results.append(json.loads(match.strip()))
        except json.JSONDecodeError:
            continue
    return results or None


def _extract_any_json(text: str) -> list[dict] | None:
    """Extract JSON objects from any format (<json> tags or markdown blocks).

    Tries <json> tags first, then falls back to markdown code blocks.
    """
    # Try <json> tags first (preferred format)
    results = _extract_jsons(text)
    if results:
        return results
    
    # Fallback to markdown code blocks
    results = _extract_json_from_markdown(text)
    if results:
        return results
    
    # Last resort: try to find any JSON-like structure
    try:
        # Look for content between curly braces
        brace_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(brace_pattern, text, re.DOTALL)
        for match in matches:
            try:
                results.append(json.loads(match))
            except json.JSONDecodeError:
                continue
    except Exception:
        pass
    
    return results or None


class TaskAgent:
    """Task agent that solves IMO grading problems."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.log_file = log_file

    def _log_interaction(self, inputs: dict, prediction: str, msg_history: list[dict]) -> None:
        """Log the interaction for debugging and analysis."""
        if self.log_file:
            try:
                log_entry = {
                    "inputs": inputs,
                    "prediction": prediction,
                    "history_length": len(msg_history),
                }
                with open(self.log_file, "a") as f:
                    f.write(json.dumps(log_entry) + "\n")
            except Exception as e:
                self.log_fn(f"Failed to write to log file: {e}")

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        instruction = f"""You are an agent.

Task input:
```
{inputs}
```

Respond in JSON format with the following schema:
<json>
{{
    "response": ...
}}
</json>"""

        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
        except Exception as e:
            self.log_fn(f"Error calling LLM: {e}")
            return "Error: LLM call failed", []

        # Extract prediction from JSON using multiple strategies
        prediction = "None"
        extraction_errors = []
        
        try:
            if not msg_history:
                extraction_errors.append("Empty message history")
            else:
                last_message = msg_history[-1]
                text_content = last_message.get("text", "")
                
                if not text_content:
                    extraction_errors.append("No text content in last message")
                else:
                    extracted = _extract_any_json(text_content)
                    
                    if extracted:
                        # Try to find response in any of the extracted JSONs
                        for json_obj in extracted:
                            if isinstance(json_obj, dict) and "response" in json_obj:
                                prediction = json_obj["response"]
                                break
                        else:
                            # If no "response" key, use the last extracted JSON as string
                            prediction = str(extracted[-1])
                    else:
                        extraction_errors.append("No valid JSON found in response")
                        # Fallback: use raw text if it's short enough
                        if len(text_content) < 500:
                            prediction = text_content.strip()
                        else:
                            prediction = text_content[:500].strip() + "..."
                            
        except Exception as e:
            extraction_errors.append(f"Exception during extraction: {e}")
            self.log_fn(f"Error extracting prediction: {e}")

        if extraction_errors:
            self.log_fn(f"Extraction issues: {extraction_errors}")

        # Log the interaction
        self._log_interaction(inputs, str(prediction), msg_history)

        return str(prediction), msg_history
