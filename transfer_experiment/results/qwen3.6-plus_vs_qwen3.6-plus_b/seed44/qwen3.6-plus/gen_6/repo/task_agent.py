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

from agent.llm_client import get_response_from_llm, EVAL_MODEL

logger = logging.getLogger(__name__)

__version__ = "1.1.0"


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    Also falls back to scanning for bare JSON objects if no tags are found.
    Handles nested braces within JSON values by tracking brace depth.
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

    # Fallback: if no tagged JSON found, try to extract bare JSON objects
    if not results:
        try:
            # Try parsing the entire text as JSON first
            parsed = json.loads(text.strip())
            if isinstance(parsed, dict):
                results.append(parsed)
            elif isinstance(parsed, list):
                results.extend(item for item in parsed if isinstance(item, dict))
        except json.JSONDecodeError:
            # Try to find JSON-like objects using regex as last resort
            # Use a more robust pattern that handles nested braces
            pattern = r'\{(?:[^{}]|(?R))*\}'
            # Since Python re doesn't support (?R), use a manual approach
            results = _extract_json_objects_manual(text)

    return results or None


def _extract_json_objects_manual(text: str) -> list[dict]:
    """Manually extract JSON objects by tracking brace depth."""
    results = []
    i = 0
    while i < len(text):
        if text[i] == '{':
            # Find the matching closing brace
            depth = 0
            start = i
            in_string = False
            escape_next = False
            j = i
            while j < len(text):
                ch = text[j]
                if escape_next:
                    escape_next = False
                    j += 1
                    continue
                if ch == '\\' and in_string:
                    escape_next = True
                    j += 1
                    continue
                if ch == '"' and not escape_next:
                    in_string = not in_string
                elif not in_string:
                    if ch == '{':
                        depth += 1
                    elif ch == '}':
                        depth -= 1
                        if depth == 0:
                            candidate = text[start:j+1]
                            try:
                                obj = json.loads(candidate)
                                if isinstance(obj, dict):
                                    results.append(obj)
                            except json.JSONDecodeError:
                                pass
                            i = j
                            break
                j += 1
            else:
                # No matching brace found
                pass
        i += 1
    return results


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
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", inputs.get("response", ""))
        domain = inputs.get("domain", "")

        domain_context = f"\nDomain: {domain}" if domain else ""

        instruction = f"""You are an expert IMO (International Mathematical Olympiad) grader. Your task is to evaluate a student's answer to a math problem.{domain_context}

**Problem:**
{problem}

**Official Solution:**
{solution}

**Grading Guidelines:**
{grading_guidelines}

**Student's Answer:**
{student_answer}

Carefully analyze the student's answer step by step:
1. Compare the student's reasoning against the official solution
2. Check if the student's approach is mathematically valid
3. Identify any errors, gaps, or missing steps
4. Determine if the final answer is correct
5. Assign a grade based on the grading guidelines

Respond in JSON format with the following schema:
<json>
{{
    "reasoning": "Your step-by-step analysis of the student's answer",
    "response": "correct" or "incorrect" or "partial"
}}
</json>"""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON
        prediction = "None"
        try:
            extracted = _extract_jsons(msg_history[-1]["text"])
            if extracted and "response" in extracted[-1]:
                prediction = extracted[-1]["response"]
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
