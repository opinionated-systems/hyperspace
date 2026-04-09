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
    Also handles nested JSON objects and common formatting issues.
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

        # Try to parse the JSON
        try:
            results.append(json.loads(inner))
        except json.JSONDecodeError as e:
            # Try common fixes: remove trailing commas, fix quotes
            try:
                # Remove trailing commas before closing braces/brackets
                fixed = re.sub(r',(\s*[}\]])', r'\1', inner)
                results.append(json.loads(fixed))
            except json.JSONDecodeError:
                # Try extracting just the first valid JSON object
                try:
                    # Find matching braces
                    brace_count = 0
                    in_string = False
                    escape_next = False
                    json_start = -1

                    for i, char in enumerate(inner):
                        if escape_next:
                            escape_next = False
                            continue
                        if char == '\\':
                            escape_next = True
                            continue
                        if char == '"' and not escape_next:
                            in_string = not in_string
                            continue
                        if not in_string:
                            if char == '{':
                                if brace_count == 0:
                                    json_start = i
                                brace_count += 1
                            elif char == '}':
                                brace_count -= 1
                                if brace_count == 0 and json_start >= 0:
                                    try:
                                        results.append(json.loads(inner[json_start:i+1]))
                                        break
                                    except json.JSONDecodeError:
                                        pass
                except Exception:
                    pass
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
        # Format inputs nicely for the LLM
        formatted_inputs = json.dumps(inputs, indent=2, default=str)

        instruction = f"""You are an expert grading agent. Your task is to evaluate a student's answer to a problem.

Task input:
```json
{formatted_inputs}
```

Instructions:
1. Carefully read the problem, solution, and grading guidelines
2. Evaluate the student's answer against the correct solution
3. Provide your assessment in the exact JSON format below

Respond in JSON format with the following schema:
<json>
{{
    "response": "Your detailed evaluation here. Include: (1) what the student got right/wrong, (2) score if applicable, (3) specific feedback"
}}
</json>

Important: Your response MUST be valid JSON inside <json> tags."""

        self.log_fn("TaskAgent: Sending request to LLM...")

        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
            self.log_fn("TaskAgent: Received response from LLM")
        except Exception as e:
            self.log_fn(f"TaskAgent: LLM call failed: {e}")
            return f"Error: LLM call failed - {e}", []

        # Extract prediction from JSON
        prediction = "None"
        try:
            if msg_history and len(msg_history) > 0:
                last_msg = msg_history[-1]
                text = last_msg.get("text", "") if isinstance(last_msg, dict) else str(last_msg)
                extracted = _extract_jsons(text)
                if extracted:
                    last_json = extracted[-1]
                    if isinstance(last_json, dict) and "response" in last_json:
                        prediction = last_json["response"]
                        self.log_fn("TaskAgent: Successfully extracted prediction")
                    else:
                        self.log_fn(f"TaskAgent: JSON missing 'response' key: {last_json.keys() if isinstance(last_json, dict) else 'N/A'}")
                else:
                    self.log_fn("TaskAgent: No JSON found in response")
                    # Fallback: use the raw text if no JSON found
                    prediction = text[:1000] if len(text) > 1000 else text
            else:
                self.log_fn("TaskAgent: Empty message history")
        except Exception as e:
            self.log_fn(f"TaskAgent: Error extracting prediction: {e}")
            # Try to get any text from the last message as fallback
            try:
                if msg_history and len(msg_history) > 0:
                    last_msg = msg_history[-1]
                    text = last_msg.get("text", "") if isinstance(last_msg, dict) else str(last_msg)
                    prediction = text[:1000] if len(text) > 1000 else text
            except Exception:
                pass

        return str(prediction), msg_history
