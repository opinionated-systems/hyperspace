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


def _extract_json_fallback(text: str) -> list[dict] | None:
    """Fallback JSON extraction using regex for unwrapped JSON objects.

    This handles cases where the model outputs valid JSON without <json> tags.
    Searches for JSON objects with a "response" key.
    """
    results = []
    # Improved pattern to match JSON objects with nested structures
    # Uses a balanced brace approach to handle nested objects
    pattern = r'\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*"response"(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}'
    matches = re.finditer(pattern, text, re.DOTALL)

    for match in matches:
        try:
            obj = json.loads(match.group())
            if "response" in obj:
                results.append(obj)
        except json.JSONDecodeError:
            continue

    # If regex fails, try to find any JSON-like structure
    if not results:
        try:
            # Try to parse the entire text as JSON
            obj = json.loads(text.strip())
            if isinstance(obj, dict) and "response" in obj:
                results.append(obj)
        except json.JSONDecodeError:
            pass

    # Final fallback: try to extract any valid JSON object
    if not results:
        # Look for JSON objects between curly braces
        brace_count = 0
        start_idx = -1
        for i, char in enumerate(text):
            if char == '{':
                if brace_count == 0:
                    start_idx = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start_idx != -1:
                    try:
                        obj = json.loads(text[start_idx:i+1])
                        if isinstance(obj, dict) and "response" in obj:
                            results.append(obj)
                    except json.JSONDecodeError:
                        pass
                    start_idx = -1

    return results or None


class TaskAgent:
    """Task agent that solves IMO grading problems with robust JSON extraction."""

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
        # Format inputs nicely for the LLM
        formatted_inputs = json.dumps(inputs, indent=2, default=str)
        
        instruction = f"""You are an expert grading agent. Your task is to evaluate a student's answer based on the provided problem, solution, and grading guidelines.

Task input:
```json
{formatted_inputs}
```

Analyze the student's answer carefully and provide your evaluation.

Respond in JSON format with the following schema:
<json>
{{
    "response": "Your detailed evaluation and grade here"
}}
</json>

Important: Ensure your response is valid JSON inside the <json> tags."""

        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
        except Exception as e:
            self.log_fn(f"Error calling LLM: {e}")
            return "Error: LLM call failed", [{"role": "system", "text": f"Error: {e}"}]

        # Extract prediction from JSON using primary method
        prediction = "None"
        extraction_method = "none"
        raw_response = msg_history[-1]["text"] if msg_history else ""
        
        try:
            # Try primary extraction first
            extracted = _extract_jsons(raw_response)
            if extracted and len(extracted) > 0:
                last_extracted = extracted[-1]
                if isinstance(last_extracted, dict) and "response" in last_extracted:
                    prediction = last_extracted["response"]
                    extraction_method = "primary"
                else:
                    self.log_fn(f"Primary extraction: 'response' key not found in {list(last_extracted.keys()) if isinstance(last_extracted, dict) else 'non-dict'}")
            else:
                self.log_fn("Primary extraction: no JSON objects found")
                
            # Try fallback if primary failed
            if extraction_method == "none":
                extracted = _extract_json_fallback(raw_response)
                if extracted and len(extracted) > 0:
                    last_extracted = extracted[-1]
                    if isinstance(last_extracted, dict) and "response" in last_extracted:
                        prediction = last_extracted["response"]
                        extraction_method = "fallback"
                        self.log_fn("Fallback extraction succeeded")
                    else:
                        self.log_fn(f"Fallback extraction: 'response' key not found")
                else:
                    self.log_fn("Fallback extraction: no JSON objects found")
        except Exception as e:
            self.log_fn(f"Error during extraction: {e}")
            # Try fallback on exception
            try:
                extracted = _extract_json_fallback(raw_response)
                if extracted and len(extracted) > 0:
                    last_extracted = extracted[-1]
                    if isinstance(last_extracted, dict) and "response" in last_extracted:
                        prediction = last_extracted["response"]
                        extraction_method = "fallback_exception"
            except Exception as fallback_e:
                self.log_fn(f"Fallback extraction also failed: {fallback_e}")

        self.log_fn(f"Extraction method used: {extraction_method}")
        
        # Log a preview of the prediction for debugging
        preview = str(prediction)[:100] + "..." if len(str(prediction)) > 100 else str(prediction)
        self.log_fn(f"Prediction preview: {preview}")
        
        return str(prediction), msg_history
