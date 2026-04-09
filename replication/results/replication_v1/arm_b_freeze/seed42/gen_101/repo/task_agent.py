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
            logger.debug(f"Failed to parse JSON block: {e}")
            continue
    return results or None


def _extract_json_fallback(text: str) -> dict | None:
    """Fallback JSON extraction for malformed or missing <json> tags.
    
    Attempts to find JSON objects directly in the text, handling common
    formatting issues like markdown code blocks.
    """
    # Try to extract from markdown code blocks
    code_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    matches = re.findall(code_block_pattern, text, re.DOTALL)
    for match in matches:
        try:
            return json.loads(match.strip())
        except json.JSONDecodeError:
            continue
    
    # Try to find JSON-like structures with braces
    brace_pattern = r'\{[^{}]*"response"[^{}]*\}'
    match = re.search(brace_pattern, text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    
    return None


class TaskAgent:
    """Task agent that solves IMO grading problems."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.log_file = log_file

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Build a more structured prompt with clear instructions
        instruction = f"""You are an expert grading agent. Your task is to evaluate a student's answer to a problem.

Task input:
```
{json.dumps(inputs, indent=2)}
```

Carefully analyze the problem, the correct solution, the grading guidelines, and the student's answer.

Respond in JSON format with the following schema:
<json>
{{
    "response": "Your evaluation result here"
}}
</json>

The response field should contain your complete evaluation of the student's answer."""

        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return "Error: LLM call failed", [{"role": "error", "text": str(e)}]

        # Extract prediction from JSON with fallback mechanisms
        prediction = "None"
        extraction_method = "none"
        
        try:
            if not msg_history:
                logger.warning("Empty message history from LLM")
                return str(prediction), msg_history
                
            last_message = msg_history[-1].get("text", "")
            if not last_message:
                logger.warning("Empty text in last message")
                return str(prediction), msg_history
            
            # Primary extraction: <json> tags
            extracted = _extract_jsons(last_message)
            if extracted and "response" in extracted[-1]:
                prediction = extracted[-1]["response"]
                extraction_method = "json_tags"
            else:
                # Fallback extraction
                fallback = _extract_json_fallback(last_message)
                if fallback and "response" in fallback:
                    prediction = fallback["response"]
                    extraction_method = "fallback"
                else:
                    # Last resort: use raw text if no JSON found
                    prediction = last_message.strip()
                    extraction_method = "raw_text"
                    logger.warning(f"No JSON found, using raw text: {prediction[:100]}...")
            
            self.log_fn(f"Extraction method: {extraction_method}, prediction: {str(prediction)[:100]}...")
            
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            logger.exception("Prediction extraction failed")

        return str(prediction), msg_history
