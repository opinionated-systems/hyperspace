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
        except json.JSONDecodeError as e:
            logger.debug(f"JSON decode error: {e}")
            continue
    return results or None


def _extract_json_fuzzy(text: str) -> list[dict] | None:
    """Fallback JSON extraction for malformed responses.
    
    Attempts to find JSON-like structures even without proper tags.
    """
    results = []
    # Try to find JSON objects between curly braces
    pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.finditer(pattern, text, re.DOTALL)
    for match in matches:
        try:
            obj = json.loads(match.group())
            if isinstance(obj, dict):
                results.append(obj)
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
        logger.info(f"TaskAgent call #{self.call_count}")
        
        instruction = f"""You are an expert grading agent. Analyze the student answer carefully.

Task input:
```
{inputs}
```

Respond in JSON format with the following schema:
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
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return "Error: LLM call failed", []

        # Extract prediction from JSON
        prediction = "None"
        extraction_method = "none"
        
        try:
            if msg_history and len(msg_history) > 0:
                last_msg = msg_history[-1].get("text", "")
                
                # Try primary extraction method
                extracted = _extract_jsons(last_msg)
                if extracted and "response" in extracted[-1]:
                    prediction = extracted[-1]["response"]
                    extraction_method = "primary"
                else:
                    # Try fallback extraction
                    extracted = _extract_json_fuzzy(last_msg)
                    if extracted and "response" in extracted[-1]:
                        prediction = extracted[-1]["response"]
                        extraction_method = "fallback"
                    else:
                        # Last resort: try to extract any text
                        prediction = last_msg[:500]  # Limit length
                        extraction_method = "raw"
                        
                logger.info(f"Extraction method: {extraction_method}")
        except Exception as e:
            logger.error(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
