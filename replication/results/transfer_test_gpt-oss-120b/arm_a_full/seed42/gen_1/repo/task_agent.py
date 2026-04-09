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
    """Extract JSON objects from the LLM response.

    Supports multiple formats:
    1. Explicit <json>...</json> tags (original behavior).
    2. Markdown code blocks with ```json.
    3. Bare JSON objects possibly surrounded by other text.
    4. JSON arrays wrapped in square brackets.
    Returns a list of parsed JSON dictionaries or None if none found.
    """
    import re, json
    results = []
    
    # First, try tag-based extraction (original)
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
    
    # Second, try markdown code blocks
    if not results:
        md_pattern = r"```(?:json)?\s*\n?(.*?)\n?```"
        for match in re.finditer(md_pattern, text, re.DOTALL):
            try:
                results.append(json.loads(match.group(1).strip()))
            except json.JSONDecodeError:
                continue
    
    # Third: try to find JSON arrays
    if not results:
        try:
            # Look for array patterns
            array_match = re.search(r"\[\s*\{.*\}\s*\]", text, re.DOTALL)
            if array_match:
                parsed = json.loads(array_match.group(0))
                if isinstance(parsed, list):
                    results.extend([item for item in parsed if isinstance(item, dict)])
        except (json.JSONDecodeError, ValueError):
            pass
    
    # Fallback: find any JSON object using regex with nested brace support
    if not results:
        # Improved regex that handles nested braces
        json_candidates = re.findall(r"\{(?:[^{}]|\{[^{}]*\})*\}", text)
        for cand in json_candidates:
            try:
                results.append(json.loads(cand))
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
        # Validate inputs
        if not isinstance(inputs, dict):
            error_msg = f"Invalid inputs type: expected dict, got {type(inputs).__name__}"
            self.log_fn(f"[TaskAgent] {error_msg}")
            return error_msg, [{"role": "system", "text": error_msg}]
        
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
            error_msg = f"LLM call failed: {e}"
            self.log_fn(f"[TaskAgent] {error_msg}")
            return error_msg, [{"role": "system", "text": error_msg}]

        # Extract prediction from JSON
        prediction = "None"
        try:
            if not msg_history:
                self.log_fn("[TaskAgent] Empty message history from LLM")
                return "None", msg_history
                
            last_message = msg_history[-1]
            last_text = last_message.get("text", "") if isinstance(last_message, dict) else str(last_message)
            
            if not last_text:
                self.log_fn("[TaskAgent] Empty response text from LLM")
                return "None", msg_history
            
            extracted = _extract_jsons(last_text)
            if extracted:
                # Try to find response in any of the extracted JSONs
                for item in extracted:
                    if isinstance(item, dict) and "response" in item:
                        prediction = item["response"]
                        self.log_fn(f"[TaskAgent] Extracted response from JSON: {prediction[:100] if isinstance(prediction, str) else prediction}...")
                        break
                else:
                    # If no response key found, use the last extracted item as fallback
                    if isinstance(extracted[-1], dict):
                        prediction = extracted[-1].get("response", str(extracted[-1]))
                        self.log_fn(f"[TaskAgent] Using fallback extraction: {prediction[:100] if isinstance(prediction, str) else prediction}...")
                    else:
                        prediction = str(extracted[-1])
            else:
                self.log_fn(f"[TaskAgent] No JSON found in response: {last_text[:200]}...")
                # Fallback: try to use raw response if JSON extraction fails
                if len(last_text) < 1000:
                    prediction = last_text.strip()
        except Exception as e:
            self.log_fn(f"[TaskAgent] Error extracting prediction: {e}")
            # Fallback: try to use raw response if JSON extraction fails
            if msg_history:
                last_message = msg_history[-1]
                raw_response = last_message.get("text", "") if isinstance(last_message, dict) else str(last_message)
                if raw_response and len(raw_response) < 1000:
                    prediction = raw_response.strip()

        return str(prediction), msg_history
