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
    """Extract JSON objects from <json>...</json> blocks or raw JSON.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    Also tries to find raw JSON objects if tags are not present.
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
            continue
    
    # If no results, try to find raw JSON objects (looking for {...})
    if not results:
        # Find JSON objects by looking for balanced braces
        i = 0
        while i < len(text):
            if text[i] == '{':
                # Try to find the matching closing brace
                brace_count = 1
                j = i + 1
                while j < len(text) and brace_count > 0:
                    if text[j] == '{':
                        brace_count += 1
                    elif text[j] == '}':
                        brace_count -= 1
                    j += 1
                
                if brace_count == 0:
                    json_str = text[i:j].strip()
                    try:
                        results.append(json.loads(json_str))
                    except json.JSONDecodeError:
                        pass
                i = j
            else:
                i += 1
    
    return results or None


def _extract_response_from_text(text: str) -> str | None:
    """Extract response from various formats in the text."""
    # Try to find "response": "..." pattern
    patterns = [
        r'"response"\s*:\s*"([^"]*)"',
        r'"response"\s*:\s*\{([^}]*)\}',
        r'"response"\s*:\s*(\d+)',
        r'response[:\s]+([\w\s]+)',
        r'Answer[:\s]+([\w\s]+)',
        r'Prediction[:\s]+([\w\s]+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    return None


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
        # Extract grading guidelines to understand the expected output format
        grading_guidelines = inputs.get('grading_guidelines', '')
        
        instruction = f"""You are an expert grader evaluating student solutions to math problems.

Your task is to evaluate the student's answer based on the problem, solution, and grading guidelines provided.

Problem:
{inputs.get('problem', '')}

Official Solution:
{inputs.get('solution', '')}

Grading Guidelines:
{grading_guidelines}

Student Answer:
{inputs.get('student_answer', '')}

Based on the grading guidelines, classify the student's answer as one of:
- "correct" - if the answer is fully correct
- "incorrect" - if the answer is wrong or significantly flawed  
- "partial" - if the answer has some correct elements but is incomplete

Respond in JSON format with the following schema:
<json>
{{
    "response": "correct" | "incorrect" | "partial"
}}
</json>

Your evaluation:"""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON
        prediction = "None"
        try:
            last_message = msg_history[-1]["text"]
            extracted = _extract_jsons(last_message)
            if extracted:
                last_json = extracted[-1]
                if isinstance(last_json, dict) and "response" in last_json:
                    prediction = last_json["response"]
                else:
                    # Try to get any value from the JSON as prediction
                    prediction = str(list(last_json.values())[0]) if last_json else "None"
            else:
                # Fallback: try to extract response from text patterns
                extracted_text = _extract_response_from_text(last_message)
                if extracted_text:
                    prediction = extracted_text
                else:
                    # Last resort: look for the keywords directly in the text
                    text_lower = last_message.lower()
                    if '"correct"' in text_lower or 'correct' in text_lower:
                        prediction = "correct"
                    elif '"incorrect"' in text_lower or 'incorrect' in text_lower:
                        prediction = "incorrect"
                    elif '"partial"' in text_lower or 'partial' in text_lower:
                        prediction = "partial"
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        # Normalize prediction to expected values
        prediction_str = str(prediction).lower().strip()
        if "correct" in prediction_str and "incorrect" not in prediction_str and "partial" not in prediction_str:
            prediction = "correct"
        elif "partial" in prediction_str:
            prediction = "partial"
        elif "incorrect" in prediction_str:
            prediction = "incorrect"
        
        return str(prediction), msg_history
