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

try:
    from agent.llm_client import get_response_from_llm, EVAL_MODEL
except Exception:
    def get_response_from_llm(*args, **kwargs):
        return "", [], {}
    EVAL_MODEL = "gpt-3.5"


logger = logging.getLogger(__name__)


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks safely.
    Returns a list of dicts or None if none found.
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

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction using flexible extraction
        prediction = "None"
        try:
            response_text = msg_history[-1]["text"] if msg_history else ""
            def _extract_response_flexible(text: str) -> str | None:
                json_results = _extract_jsons(text)
                if json_results:
                    for result in json_results:
                        if isinstance(result, dict):
                            for key in ["response", "classification", "answer", "result", "grade", "evaluation", "verdict"]:
                                if key in result:
                                    val = result[key]
                                    if isinstance(val, str):
                                        normalized = _normalize_prediction(val.strip())
                                        if normalized:
                                            return normalized
                                    elif isinstance(val, bool):
                                        return "Correct" if val else "Incorrect"
                markdown_json_pattern = re.search(r'```(?:json)?\s*\n?(\{[^}]+\})\n?```', text, re.DOTALL)
                if markdown_json_pattern:
                    try:
                        json_obj = json.loads(markdown_json_pattern.group(1))
                        if isinstance(json_obj, dict):
                            for key in ["response", "classification", "answer", "result", "grade", "evaluation", "verdict"]:
                                if key in json_obj:
                                    val = json_obj[key]
                                    if isinstance(val, str):
                                        normalized = _normalize_prediction(val.strip())
                                        if normalized:
                                            return normalized
                                    elif isinstance(val, bool):
                                        return "Correct" if val else "Incorrect"
                    except json.JSONDecodeError:
                        pass
                json_pattern = re.search(r'\{\s*"(?:response|classification|answer|result|grade|evaluation|verdict)"\s*:\s*"([^"]+)"\s*\}', text, re.IGNORECASE)
                if json_pattern:
                    normalized = _normalize_prediction(json_pattern.group(1).strip())
                    if normalized:
                        return normalized
                for cat in ["Correct", "Incorrect", "Partial", "Almost"]:
                    if re.search(rf'\b{cat}\b', text, re.IGNORECASE):
                        return cat
                return None
            def _normalize_prediction(prediction: str) -> str | None:
                if not prediction:
                    return None
                pred_lower = prediction.lower().strip()
                allowed = ["correct", "incorrect", "partial", "almost"]
                if pred_lower in allowed:
                    return pred_lower.capitalize()
                correct_variations = ['correct', 'right', 'true', 'valid', 'complete', 'accurate', 'perfect']
                incorrect_variations = ['incorrect', 'wrong', 'false', 'invalid', 'error', 'mistake', 'flawed']
                partial_variations = ['partial', 'partly', 'some', 'half']
                almost_variations = ['almost', 'nearly', 'close', 'minor']
                for var in correct_variations:
                    if var in pred_lower:
                        return "Correct"
                for var in incorrect_variations:
                    if var in pred_lower:
                        return "Incorrect"
                for var in partial_variations:
                    if var in pred_lower:
                        return "Partial"
                for var in almost_variations:
                    if var in pred_lower:
                        return "Almost"
                return None
            extracted = _extract_response_flexible(response_text)
            if extracted:
                prediction = extracted
                self.log_fn(f"Extracted prediction: {prediction}")
            else:
                normalized = _normalize_prediction(response_text)
                if normalized:
                    prediction = normalized
                    self.log_fn(f"Normalized prediction: {prediction}")
                else:
                    self.log_fn(f"Could not extract prediction from response: {response_text[:200]}...")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        # Redundant extraction block removed

        # Ensure prediction is a string without surrounding whitespace
        if isinstance(prediction, str):
            prediction = prediction.strip()
        # Validate prediction against allowed categories
        allowed_categories = ["Correct", "Incorrect", "Partial", "Almost"]
        if prediction not in allowed_categories:
            self.log_fn(f"Invalid prediction '{prediction}', defaulting to None")
            prediction = "None"
        return str(prediction), msg_history
